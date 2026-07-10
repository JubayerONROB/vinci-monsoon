"""Container entrypoint for the Track 1 hybrid routing agent.

Contract with the grading harness:
  * read  /input/tasks.json   (list of {"task_id", "prompt"})
  * write /output/results.json (list of {"task_id", "answer"}) before exiting
  * exit 0 on success, non-zero on failure
  * total runtime < 10 min, per-request < 30 s, ready < 60 s

INPUT_PATH / OUTPUT_PATH env vars exist only for local development; the
defaults match the harness contract.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

# Make repo-root imports work no matter where the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.api_clients.fireworks import FireworksClient          # noqa: E402
from src.local_models.loader import get_local_model            # noqa: E402
from src.router.dispatch import Router                         # noqa: E402

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# Soft ceiling on total runtime, configurable via env. Once elapsed time
# crosses it we STOP escalating to Fireworks and answer every remaining task
# with the local model only: a possibly-weaker local answer always beats a
# TIMEOUT, which zeros the whole submission.
TIME_BUDGET_SECONDS = float(os.environ.get("TIME_BUDGET_SECONDS", "420"))

# Hard wall-clock deadline for the ENTIRE process, enforced by a watchdog
# thread independent of whatever the main loop is doing. llama.cpp releases
# the GIL during inference, so this thread keeps ticking even while a local
# model call is stuck/slow on a weak grading CPU. 540s leaves 60s of margin
# inside the 10-minute (600s) hard cap for process teardown/exit.
HARD_DEADLINE_SECONDS = float(os.environ.get("HARD_DEADLINE_SECONDS", "540"))

# Per-task ceiling so one slow local-model call (classify + generate, no
# built-in timeout of its own) can never eat the whole remaining budget or
# blow the 30s per-request cap. Comfortably under 30s.
PER_TASK_TIMEOUT_SECONDS = float(os.environ.get("PER_TASK_TIMEOUT_SECONDS", "27"))

_write_lock = threading.Lock()
_written = threading.Event()


def _fallback_answer(prompt: str, router: Router) -> str:
    """Best-effort local answer used when a task raises unexpectedly."""
    try:
        return router._local_answer(prompt)
    except Exception:
        return "Unable to produce an answer for this task."


def _quick_answer(prompt: str) -> str:
    """Near-instant answer for when even the local model is too slow to trust."""
    head = " ".join(prompt.split())[:160]
    return f"Best-effort response (time budget exceeded) to: {head}"


def _write_results(results: list, path: str) -> None:
    with _write_lock:
        if _written.is_set():
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=1)
        tmp.replace(out)
        _written.set()


def _start_watchdog(start: float, results: list, all_task_ids: list, output_path: str) -> None:
    """Force a valid results.json onto disk before the hard runtime cap,
    no matter what the main thread is doing. This decouples "did we exit
    on time" from the speed of any individual (possibly stuck) task."""

    def _run():
        while not _written.is_set():
            remaining = HARD_DEADLINE_SECONDS - (time.time() - start)
            if remaining <= 0:
                break
            time.sleep(min(remaining, 2.0))
        if _written.is_set():
            return
        print(
            f"WATCHDOG: hard deadline {HARD_DEADLINE_SECONDS:.0f}s reached — "
            "force-writing partial results and exiting",
            file=sys.stderr, flush=True,
        )
        answered = {r["task_id"] for r in results}
        padded = list(results)
        for task_id in all_task_ids:
            if task_id not in answered:
                padded.append({"task_id": task_id, "answer": "No answer available (time budget exceeded)."})
        _write_results(padded, output_path)
        os._exit(0)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def main() -> int:
    start = time.time()

    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as fh:
            tasks = json.load(fh)
        assert isinstance(tasks, list)
    except Exception as exc:
        print(f"FATAL: cannot read tasks from {INPUT_PATH}: {exc}", file=sys.stderr)
        return 1

    local_model = get_local_model()
    print(f"local model backend: {local_model.backend}", flush=True)
    router = Router(local_model, FireworksClient())
    # Resolved role -> model map (from runtime ALLOWED_MODELS, never hardcoded)
    print(f"resolved model map: {router.resolved_map()}", flush=True)

    results: list[dict] = []
    all_task_ids = [t.get("task_id", "") for t in tasks]
    _start_watchdog(start, results, all_task_ids, OUTPUT_PATH)

    budget_hit = False
    # A stuck task's thread is abandoned (not joined) on timeout so it never
    # blocks the next task; each task gets its own single-worker pool.
    for task in tasks:
        task_id = task.get("task_id", "")
        prompt = task.get("prompt", "")
        try:
            if time.time() - start > TIME_BUDGET_SECONDS:
                if not budget_hit:
                    budget_hit = True
                    print(
                        f"TIME BUDGET {TIME_BUDGET_SECONDS:.0f}s exceeded — "
                        "answering all remaining tasks locally (no more API calls)",
                        flush=True,
                    )
                target, args = _fallback_answer, (prompt, router)
                meta = {"route": "deadline_local"}
            else:
                target, args = router.route, (prompt,)
                meta = None

            remaining_hard = HARD_DEADLINE_SECONDS - (time.time() - start)
            per_task_timeout = max(1.0, min(PER_TASK_TIMEOUT_SECONDS, remaining_hard))
            # Not a context manager: pool.__exit__ would block on shutdown()
            # waiting for a stuck worker thread, defeating the timeout below.
            # shutdown(wait=False) lets a timed-out call keep running in the
            # background (abandoned, never joined) while we move on.
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(target, *args)
            try:
                if meta is None:
                    answer, meta = future.result(timeout=per_task_timeout)
                else:
                    answer = future.result(timeout=per_task_timeout)
            except FutureTimeoutError:
                answer = _quick_answer(prompt)
                meta = {"route": "task_timeout", "decision": {}}
            finally:
                pool.shutdown(wait=False)
        except Exception as exc:  # one bad task must never sink the run
            answer, meta = _fallback_answer(prompt, router), {"route": "error", "error": str(exc)}
        if not isinstance(answer, str) or not answer.strip():
            answer = "No answer available."
        results.append({"task_id": task_id, "answer": answer})
        print(f"[{task_id}] route={meta.get('route')} model={meta.get('model', '-')}", flush=True)
        # Per-task diagnostic (stderr, non-sensitive): lets a failed grading
        # run be diagnosed per category instead of tuning blind.
        cat = meta.get("decision", {}).get("intent", "?")
        print(
            f"DIAG {task_id} | {cat} | {meta.get('route')} | "
            f"{meta.get('model', '-')} | finish={meta.get('finish_reason', '-')} | "
            f"answer_len={len(answer)} | "
            f"truncated={'yes' if meta.get('truncated') else 'no'}",
            file=sys.stderr, flush=True,
        )

        if time.time() - start > HARD_DEADLINE_SECONDS:
            print("HARD DEADLINE reached mid-loop — stopping task processing", flush=True)
            break

    _write_results(results, OUTPUT_PATH)

    elapsed = time.time() - start
    print(
        f"done: {len(results)} tasks in {elapsed:.1f}s | "
        f"fireworks calls={router.fireworks.calls} tokens={router.fireworks.total_tokens}",
        flush=True,
    )
    # Any per-task timeout leaves an orphaned, non-daemon executor thread
    # still running in the background. concurrent.futures registers an
    # atexit hook that JOINS such threads before the interpreter can exit,
    # which would silently reintroduce the exact hang this file exists to
    # prevent. os._exit() bypasses atexit/thread-joining entirely — results
    # are already durably written to disk by this point.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    exit_code = main()
    if exit_code is not None:
        sys.exit(exit_code)
