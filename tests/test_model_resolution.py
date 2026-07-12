"""STEP-1D: lane resolution must survive any ALLOWED_MODELS the grading env
injects — different order, extra models, gemma variants present — and land
on real allowed models, NEVER silently on gemma or on garbage."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_models.loader import get_local_model  # noqa: E402
from src.router.dispatch import Router               # noqa: E402

FULL_LAUNCH_LIST = (
    "accounts/fireworks/models/gemma-4-31b-it,"
    "accounts/fireworks/models/gemma-4-26b-a4b-it,"
    "accounts/fireworks/models/minimax-m3,"
    "accounts/fireworks/models/kimi-k2p7-code,"
    "accounts/fireworks/models/gemma-4-31b-it-nvfp4"
)


class _NoFireworks:
    configured = False
    calls = 0
    total_tokens = 0


def _router(monkeypatch, allowed):
    monkeypatch.setenv("ALLOWED_MODELS", allowed)
    return Router(get_local_model(), _NoFireworks())


def test_full_launch_list_gemma_first(monkeypatch):
    r = _router(monkeypatch, FULL_LAUNCH_LIST)
    m = r.resolved_map()
    assert m["general"].endswith("minimax-m3")
    assert m["reasoning"].endswith("minimax-m3")
    assert m["code"].endswith("kimi-k2p7-code")
    assert all("gemma" not in v.lower() for v in m.values())


def test_reversed_and_two_model_lists(monkeypatch):
    for allowed in (
        ",".join(reversed(FULL_LAUNCH_LIST.split(","))),
        "accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code",
    ):
        m = _router(monkeypatch, allowed).resolved_map()
        assert all(v is not None and "gemma" not in v.lower() for v in m.values())
        assert m["code"].endswith("kimi-k2p7-code")


def test_hint_miss_prefers_non_gemma(monkeypatch):
    # Platform revs the ids so no hint matches: fallback must skip gemma.
    r = _router(monkeypatch,
                "accounts/fireworks/models/gemma-4-31b-it,"
                "accounts/fireworks/models/some-new-model-v9")
    m = r.resolved_map()
    assert all(v == "accounts/fireworks/models/some-new-model-v9" for v in m.values())


def test_local_kill_switch_no_lane_traffic(monkeypatch):
    """LOCAL_CATEGORIES='' => try_answer returns None for every category
    without a single HTTP request."""
    monkeypatch.setenv("LOCAL_CATEGORIES", "")
    import requests as _requests
    from src.local_models.local import LocalLane

    def forbidden(*a, **kw):
        raise AssertionError("local lane made an HTTP call while disabled")

    monkeypatch.setattr(_requests, "post", forbidden)
    lane = LocalLane()
    assert lane.categories == set()
    for cat in ("sentiment", "ner", "summarization", "factual_knowledge"):
        assert lane.try_answer(cat, "anything") is None
