"""THE 0/19 CORRUPTION DETECTOR: prove the local lane's presence cannot
alter ANY remote-routed task's answer.

Lane-ON has zeroed grading twice while lane-OFF never has — if initializing
the lane perturbs the remote path (different model, prompt, max_tokens,
reasoning_effort, or answer text), that IS the bug. The mock Fireworks
client fingerprints EVERY request parameter into its answer string, so the
comparison catches parameter drift, not just text drift.

Three configurations over the full mock set:
  A. lane OFF (None)                       — the proven pure-remote baseline
  B. lane ON but sidecar absent (dead)     — every task escalates
  C. lane ON and SERVING its categories    — local answers for its lanes
Remote-routed answers must be byte-identical across A/B/C.
"""

import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_models.loader import get_local_model   # noqa: E402
from src.local_models.local import LocalLane           # noqa: E402
from src.router.dispatch import Router                 # noqa: E402

ALLOWED = ("accounts/fireworks/models/minimax-m3,"
           "accounts/fireworks/models/kimi-k2p7-code")


class FingerprintFireworks:
    """Deterministic, stateless-per-call: the answer encodes every parameter
    of the request, so ANY drift caused by lane presence changes the text."""

    def __init__(self):
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.calls = 0
        self.last_finish_reason = "stop"
        self.last_usage = (10, 10)
        self._lock = threading.Lock()

    configured = True

    def chat(self, model, system, user, max_tokens=512, timeout=25.0,
             reasoning_effort="none"):
        with self._lock:
            self.calls += 1
        return (f"FP|model={model}|sys={hash(system)}|user={hash(user)}"
                f"|max={max_tokens}|effort={reasoning_effort}")


def _run(monkeypatch, lane_mode: str) -> dict:
    """Route every mock task; return {task_id: (route, answer)}."""
    monkeypatch.setenv("ALLOWED_MODELS", ALLOWED)
    tasks = json.loads((ROOT / "tests" / "mock_tasks.json").read_text(encoding="utf-8"))

    lane = None
    if lane_mode != "off":
        monkeypatch.setenv("LOCAL_CATEGORIES", "sentiment,ner,summarization")
        lane = LocalLane()
        if lane_mode == "dead":
            lane._dead = True
        elif lane_mode == "serving":
            def fake_try(category, prompt, _lane=lane):
                if category not in _lane.categories:
                    return None
                return f"LOCAL_VERIFIED_ANSWER for {category}"
            lane.try_answer = fake_try

    router = Router(get_local_model(), FingerprintFireworks(), local_lane=lane)
    out = {}
    for t in tasks:
        answer, meta = router.route(t["prompt"])
        out[t["task_id"]] = (meta["route"], answer)
    return out


def test_lane_presence_never_alters_remote_answers(monkeypatch):
    baseline = _run(monkeypatch, "off")
    dead = _run(monkeypatch, "dead")
    serving = _run(monkeypatch, "serving")

    assert all(r == "remote" for r, _ in baseline.values())

    # B: dead lane => every task still remote, byte-identical to baseline.
    for tid, (route, answer) in dead.items():
        assert route == "remote", f"{tid}: dead lane changed route to {route}"
        assert answer == baseline[tid][1], (
            f"{tid}: DEAD LANE CHANGED A REMOTE ANSWER\n"
            f"  off: {baseline[tid][1]}\n  on:  {answer}"
        )

    # C: serving lane => its categories go local; EVERY other task's remote
    # answer must be byte-identical to the lane-off baseline.
    local_count = 0
    for tid, (route, answer) in serving.items():
        if route == "local_model":
            local_count += 1
            assert answer.startswith("LOCAL_VERIFIED_ANSWER")
            continue
        assert route == "remote"
        assert answer == baseline[tid][1], (
            f"{tid}: SERVING LANE CHANGED A REMOTE ANSWER — this is the "
            f"0/19 bug\n  off: {baseline[tid][1]}\n  on:  {answer}"
        )
    assert local_count == 10  # 4 sentiment + 3 ner + 3 summarization mocks
