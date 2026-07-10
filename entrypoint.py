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
import time
from pathlib import Path

# Make repo-root imports work no matter where the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.api_clients.fireworks import FireworksClient          # noqa: E402
from src.local_models.loader import get_local_model            # noqa: E402
from src.router.dispatch import Router                         # noqa: E402
from src.router.formatting import format_answer                # noqa: E402

INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# Dynamic global time guard. Before each task we compute a ceiling that
# leaves enough runway to answer ALL remaining tasks locally (no network):
#   ceiling = 540 - remaining_tasks * ~2s, floored at 300s.
# Crossing it switches every remaining task to LOCAL-ONLY, guaranteeing
# results.json is written and the container exits well under the 10-min cap.
HARD_CEILING_SECONDS = float(os.environ.get("HARD_CEILING_SECONDS", "540"))
LOCAL_EST_SECONDS = float(os.environ.get("LOCAL_EST_SECONDS", "2"))
MIN_CEILING_SECONDS = 300.0


def _write_outputs(results: list, diag_rows: list) -> None:
    """Write results.json (clean harness schema) AND diag.json beside it.

    Shared flush path: called on the normal exit AND from the crash handler,
    so diagnostics survive even a partially-failed run.
    """
    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)
    with open(out.parent / "diag.json", "w", encoding="utf-8") as fh:
        json.dump(diag_rows, fh, ensure_ascii=False, indent=1)


def _fallback_answer(prompt: str, router: Router) -> str:
    """Best-effort local answer used when a task raises unexpectedly."""
    try:
        return router._local_answer(prompt)
    except Exception:
        return "Unable to produce an answer for this task."


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
    if router.force_all_remote:
        print(
            "WARNING: local GGUF unavailable (heuristic backend) — forcing "
            "ALL categories remote for this run",
            flush=True,
        )

    results = []
    diag_rows = []
    try:
        _route_all(tasks, router, results, diag_rows, start)
    finally:
        # Same flush path for success and crash: results.json + diag.json
        # are emitted no matter what happened mid-run.
        _write_outputs(results, diag_rows)

    elapsed = time.time() - start
    print(
        f"done: {len(results)} tasks in {elapsed:.1f}s | "
        f"fireworks calls={router.fireworks.calls} tokens={router.fireworks.total_tokens}",
        flush=True,
    )
    return 0


def _route_all(tasks, router, results, diag_rows, start):
    budget_hit = False
    for idx, task in enumerate(tasks):
        task_id = task.get("task_id", "")
        prompt = task.get("prompt", "")
        remaining_tasks = len(tasks) - idx
        ceiling = max(
            HARD_CEILING_SECONDS - remaining_tasks * LOCAL_EST_SECONDS,
            MIN_CEILING_SECONDS,
        )
        try:
            if budget_hit or time.time() - start > ceiling:
                if not budget_hit:
                    budget_hit = True
                    print(
                        f"TIME CUTOVER at {time.time() - start:.0f}s "
                        f"(ceiling {ceiling:.0f}s): forcing the remaining "
                        f"{remaining_tasks} task(s) LOCAL-ONLY, no more API calls",
                        flush=True,
                    )
                answer, meta = _fallback_answer(prompt, router), {"route": "deadline_local"}
            else:
                answer, meta = router.route(prompt)
        except Exception as exc:  # one bad task must never sink the run
            answer, meta = _fallback_answer(prompt, router), {"route": "error", "error": str(exc)}
        cat = meta.get("decision", {}).get("intent", "?")
        if isinstance(answer, str) and answer.strip():
            answer = format_answer(cat, answer)
        if not isinstance(answer, str) or not answer.strip():
            answer = "No answer available."
        results.append({"task_id": task_id, "answer": answer})
        print(f"[{task_id}] route={meta.get('route')} model={meta.get('model', '-')}", flush=True)
        # Per-task diagnostic (stderr + /output/diag.json, non-sensitive):
        # lets a failed grading run be diagnosed instead of tuning blind.
        t = meta.get("timing", {})
        diag_rows.append({
            "task_id": task_id,
            "detected_category": cat,
            "route": meta.get("route"),
            "model_used": meta.get("model", "-"),
            "finish_reason": meta.get("finish_reason", "-"),
            "answer_len": len(answer),
            "truncated": bool(meta.get("truncated")),
            "primary_call_secs": t.get("primary_secs", 0),
            "alternate_fired": bool(t.get("alternate_fired")),
            "total_task_secs": t.get("total_secs", 0),
        })
        print(
            f"DIAG {task_id} | {cat} | {meta.get('route')} | "
            f"{meta.get('model', '-')} | finish={meta.get('finish_reason', '-')} | "
            f"answer_len={len(answer)} | "
            f"truncated={'yes' if meta.get('truncated') else 'no'} | "
            f"task_secs={t.get('total_secs', 0)} "
            f"(primary={t.get('primary_secs', 0)}s, "
            f"alt={'yes ' + str(t.get('alternate_secs', 0)) + 's' if t.get('alternate_fired') else 'no'})",
            file=sys.stderr, flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
