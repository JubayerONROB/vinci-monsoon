"""STEP-1 structural-bug hunt: defects that pass a lenient self-judge but
zero out the REAL judge (0/19 = every answer wrong).

A. task_id integrity — ids UNLIKE our mock ids must survive exactly (string
   type, value, order) and each answer must belong to ITS OWN task.
B. Output shape — exactly [{task_id, answer}], answer a plain string.
C. Single write — the good results can never be overwritten by a fallback
   set; a crash mid-run still flushes the answers collected so far, and a
   formatter explosion must not sink real answers.
D. Model resolution — a different/reordered ALLOWED_MODELS still lands on
   real allowed models and NEVER silently picks gemma.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ids deliberately unlike mock-01/faq-t01: judge-style, numeric-looking,
# uuid, and whitespace-bearing. Prompts carry unique markers so alignment
# is provable from the fallback answers (which echo the prompt head).
UNSEEN_TASKS = [
    {"task_id": "grade_007", "prompt": "MARKER-ALPHA What is the capital of France?"},
    {"task_id": "12", "prompt": "MARKER-BRAVO A warehouse starts with 2,400 units. How many remain?"},
    {"task_id": "3f2a9c1e-7b4d-4e0a-9c66-0d1e2f3a4b5c", "prompt": "MARKER-CHARLIE Classify the sentiment of this review: fine."},
    {"task_id": "task 4 with spaces", "prompt": "MARKER-DELTA Summarize this in exactly one sentence: things happened."},
    {"task_id": "T-05", "prompt": "MARKER-ECHO Extract all named entities from: Ada Lovelace in London."},
]


def _run_entrypoint(tmp_path, tasks, extra_env=None):
    """Run entrypoint.py as a real subprocess, no Fireworks key, no Ollama:
    every task takes the deterministic fallback path (which echoes the
    prompt head, making answer-to-id alignment checkable)."""
    inp = tmp_path / "tasks.json"
    outp = tmp_path / "results.json"
    inp.write_text(json.dumps(tasks), encoding="utf-8")
    env = {**os.environ,
           "INPUT_PATH": str(inp), "OUTPUT_PATH": str(outp),
           "ALLOWED_MODELS": "", "FIREWORKS_API_KEY": "",
           "FIREWORKS_BASE_URL": "", "LOCAL_CATEGORIES": ""}
    env.update(extra_env or {})
    proc = subprocess.run(
        [sys.executable, str(ROOT / "entrypoint.py")],
        capture_output=True, text=True, timeout=120, env=env, cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(outp.read_text(encoding="utf-8"))


def test_task_id_exact_preservation_and_alignment(tmp_path):
    results = _run_entrypoint(tmp_path, UNSEEN_TASKS)
    assert len(results) == len(UNSEEN_TASKS)
    for task, entry in zip(UNSEEN_TASKS, results):
        # B: exact shape
        assert set(entry.keys()) == {"task_id", "answer"}
        # A: id survives exactly — type, value, order
        assert isinstance(entry["task_id"], str)
        assert entry["task_id"] == task["task_id"]
        # answer is a plain non-empty string
        assert isinstance(entry["answer"], str) and entry["answer"].strip()
        # A: the answer belongs to THIS task (fallback echoes the prompt head)
        marker = task["prompt"].split()[0]
        assert marker in entry["answer"], (
            f"answer for {task['task_id']!r} does not contain its own "
            f"prompt marker {marker!r} — misalignment"
        )
    # no id lost or duplicated
    assert [e["task_id"] for e in results] == [t["task_id"] for t in UNSEEN_TASKS]


def test_results_file_is_bare_array_no_wrapper(tmp_path):
    results = _run_entrypoint(tmp_path, UNSEEN_TASKS[:2])
    assert isinstance(results, list)  # not {"results": [...]}
    for e in results:
        assert not isinstance(e["answer"], (dict, list))


def _import_entrypoint():
    for k in ("INPUT_PATH", "OUTPUT_PATH"):
        os.environ.pop(k, None)
    if "entrypoint" in sys.modules:
        return importlib.reload(sys.modules["entrypoint"])
    return importlib.import_module("entrypoint")


def test_crash_mid_run_still_flushes_collected_answers(tmp_path, monkeypatch):
    """C: an exception escaping the collection loop must not lose or blank
    the answers already collected (finally-flush), and the output must
    never be re-written with a fallback set afterwards."""
    ep = _import_entrypoint()
    inp = tmp_path / "tasks.json"
    outp = tmp_path / "results.json"
    inp.write_text(json.dumps(UNSEEN_TASKS), encoding="utf-8")
    monkeypatch.setattr(ep, "INPUT_PATH", str(inp))
    monkeypatch.setattr(ep, "OUTPUT_PATH", str(outp))
    monkeypatch.setenv("ALLOWED_MODELS", "")
    monkeypatch.setenv("LOCAL_CATEGORIES", "")

    calls = {"n": 0}
    real_route_all = ep._route_all

    def exploding_route_all(tasks, router, results, diag_rows, full_rows, runinfo):
        real_route_all(tasks[:3], router, results, diag_rows, full_rows, runinfo)
        raise RuntimeError("simulated crash after 3 tasks")

    monkeypatch.setattr(ep, "_route_all", exploding_route_all)

    write_counts = {"n": 0}
    real_write = ep._write_outputs

    def counting_write(*a, **kw):
        write_counts["n"] += 1
        return real_write(*a, **kw)

    monkeypatch.setattr(ep, "_write_outputs", counting_write)

    try:
        ep.main()
    except RuntimeError:
        pass  # the crash propagates — but the flush must already have run

    assert write_counts["n"] == 1, "output must be written exactly once"
    results = json.loads(outp.read_text(encoding="utf-8"))
    assert len(results) == 3
    for task, entry in zip(UNSEEN_TASKS[:3], results):
        assert entry["task_id"] == task["task_id"]
        assert entry["answer"].strip() and entry["answer"] != "No answer available."


def test_formatter_explosion_never_sinks_answers(tmp_path, monkeypatch):
    """C: format_answer raising must degrade to the raw answer for that
    task only — never crash the run, never blank the answer."""
    ep = _import_entrypoint()
    inp = tmp_path / "tasks.json"
    outp = tmp_path / "results.json"
    inp.write_text(json.dumps(UNSEEN_TASKS), encoding="utf-8")
    monkeypatch.setattr(ep, "INPUT_PATH", str(inp))
    monkeypatch.setattr(ep, "OUTPUT_PATH", str(outp))
    monkeypatch.setenv("ALLOWED_MODELS", "")
    monkeypatch.setenv("LOCAL_CATEGORIES", "")

    def bomb(*a, **kw):
        raise ValueError("formatter bug")

    monkeypatch.setattr(ep, "format_answer", bomb)
    assert ep.main() == 0
    results = json.loads(outp.read_text(encoding="utf-8"))
    assert len(results) == len(UNSEEN_TASKS)
    for task, entry in zip(UNSEEN_TASKS, results):
        assert entry["task_id"] == task["task_id"]
        assert entry["answer"].strip()
        assert task["prompt"].split()[0] in entry["answer"]
