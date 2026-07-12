"""Objective checker for the grading-conditions sandbox — NOT a self-judge.

Reads the RAW results.json the judge would read and fails hard on every
structural or catastrophic-content defect we know can produce a uniform 0:
  * id set/order/type drift, answer-to-id misalignment
  * wrapper contamination (dict/list answers, wrapper object)
  * blank answers
  * the all-fallback catastrophe ("Best-effort response (remote unavailable)")
  * deterministic correctness probes on known-answer tasks (warehouse=1672,
    RGB=red/green/blue, cat owner=Sam, NER must contain all five entities)
"""

import json
import sys

tasks = json.load(open(sys.argv[1], encoding="utf-8"))
results = json.load(open(sys.argv[2], encoding="utf-8"))

fails = []

if not isinstance(results, list):
    fails.append(f"results.json is {type(results).__name__}, not a bare array")
elif len(results) != len(tasks):
    fails.append(f"{len(results)} results for {len(tasks)} tasks")
else:
    by_id = {}
    for i, (task, entry) in enumerate(zip(tasks, results)):
        tid = task["task_id"]
        if set(entry.keys()) != {"task_id", "answer"}:
            fails.append(f"[{tid}] extra/missing keys: {sorted(entry.keys())}")
            continue
        if not isinstance(entry["task_id"], str):
            fails.append(f"[{tid}] task_id type {type(entry['task_id']).__name__}")
        if entry["task_id"] != tid:
            fails.append(f"row {i}: id {entry['task_id']!r} != input {tid!r} (order/value drift)")
        if not isinstance(entry["answer"], str):
            fails.append(f"[{tid}] answer is {type(entry['answer']).__name__}, not str")
            continue
        a = entry["answer"]
        if not a.strip():
            fails.append(f"[{tid}] blank answer")
        if a.startswith("Best-effort response"):
            fails.append(f"[{tid}] FALLBACK answer leaked: {a[:80]!r}")
        if a.strip() == "No answer available.":
            fails.append(f"[{tid}] placeholder answer leaked")
        by_id[tid] = a.lower()

    # Deterministic correctness probes (objective, no LLM judging)
    probes = {
        "302": lambda a: "1,672" in a or "1672" in a,
        "eval_101": lambda a: "red" in a and "green" in a and "blue" in a,
        "eval_107": lambda a: "sam" in a,
        "X-05": lambda a: all(e in a for e in
                              ("sundar pichai", "google", "zurich", "eth zurich", "march 15 2023")),
        "9d41f6a2-0c3b-4d5e-8f70-a1b2c3d4e5f6":
            lambda a: any(l in a for l in ("mixed", "positive", "neutral"))
            and "negative" not in a.split("\n")[0][:40],
    }
    for tid, probe in probes.items():
        if tid in by_id and not probe(by_id[tid]):
            fails.append(f"[{tid}] correctness probe FAILED: {by_id[tid][:120]!r}")

if fails:
    print("SANDBOX CHECK FAILED:")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print(f"SANDBOX CHECK OK: {len(results)} results, ids exact, answers aligned, "
      f"no fallback leakage, correctness probes pass")
