"""Stage 2 — routing decision + category->role->model resolution.

Decision flow per task:
  1. classify locally (0 tokens)
  2. apply the per-category escalation policy (config/routing_map.yaml)
  3. apply the AGGRESSIVE override: prompts containing code syntax, math, or
     step-by-step cues escalate even when the classifier calls them easy
  4. answer locally (0 tokens) OR escalate to the role-resolved Fireworks
     model; if the primary remote model fails, try the OTHER allowed model
     before falling back to local — never emit an empty answer.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import yaml

from config.prompts import REMOTE_SYSTEM, remote_user_prompt
from src.api_clients.fireworks import FireworksClient, FireworksError
from src.local_models.loader import LocalModel
from src.router.classifier import classify_task

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "routing_map.yaml"

# The stage-1 classifier emits categorical confidence; map it to a score so
# the config threshold (local_confidence_threshold) can gate it numerically.
_CONFIDENCE_SCORE = {"high": 0.95, "low": 0.50}

# --- Aggressive escalation cues (checked BEFORE accepting a local answer) ---
# Code syntax / language names
_CODE_CUES = re.compile(
    r"(def |class |return\b|function\b|=>|;|\{|\}|import |print\(|"
    r"\bpython\b|\bjavascript\b|\bjava\b|\bc\+\+\b|\bsql\b|\brust\b|\bbug\b)",
    re.IGNORECASE,
)
# Math symbols / multi-step calculation words
_MATH_CUES = re.compile(
    r"(\d+\s*[+\-*/^=]\s*\d+|%|\bpercent|\bcalculate\b|\bhow many\b|\baverage\b|"
    r"\bsum\b|\btotal\b|\bprofit\b|\bratio\b|\brate\b|\bprojection\b)",
    re.IGNORECASE,
)
# Explicit reasoning cues
_REASONING_CUES = re.compile(
    r"(step[- ]by[- ]step|explain your reasoning|deduce|puzzle|constraint|"
    r"each own|who owns|what is the order)",
    re.IGNORECASE,
)


def load_config(path: Optional[Path] = None) -> dict:
    with open(path or _CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def allowed_models() -> list[str]:
    """Parse ALLOWED_MODELS from the environment (never hardcoded)."""
    raw = os.environ.get("ALLOWED_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


class Router:
    def __init__(self, local_model: LocalModel, fireworks: FireworksClient,
                 config: Optional[dict] = None):
        self.local = local_model
        self.fireworks = fireworks
        self.cfg = config or load_config()
        self.allowed = allowed_models()
        self.limits = self.cfg.get("limits", {})
        self.thresholds = self.cfg.get("thresholds", {})
        # Safety net: if the GGUF failed to load (heuristic backend), local
        # answers would be echo templates — force EVERYTHING remote instead.
        self.force_all_remote = getattr(local_model, "backend", "") == "heuristic"

    # ------------------------------------------------------------------ #
    # role -> concrete allowed model ID                                   #
    # ------------------------------------------------------------------ #
    def resolve_role(self, role: str) -> Optional[str]:
        """Resolve a role to a model ID present in ALLOWED_MODELS.

        Hint substrings are matched case-insensitively; graceful fallback to
        the first allowed model; None only if ALLOWED_MODELS is empty.
        """
        if not self.allowed:
            return None
        for hint in self.cfg["role_model_hints"].get(role, []):
            for model_id in self.allowed:
                if hint.lower() in model_id.lower():
                    return model_id
        return self.allowed[0]

    def resolve_model(self, category: str) -> Optional[str]:
        return self.resolve_role(self.cfg["category_roles"].get(category, "general"))

    def resolved_map(self) -> dict:
        """role -> model ID map, for the startup log."""
        roles = sorted(set(self.cfg["category_roles"].values()))
        return {role: self.resolve_role(role) for role in roles}

    # ------------------------------------------------------------------ #
    # escalation decision                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def aggressive_override(prompt: str) -> Optional[str]:
        """Return the cue type if the prompt should escalate regardless of
        the classifier's opinion, else None."""
        if _CODE_CUES.search(prompt):
            return "code_cue"
        if _MATH_CUES.search(prompt):
            return "math_cue"
        if _REASONING_CUES.search(prompt):
            return "reasoning_cue"
        return None

    def should_answer_locally(self, decision: dict, prompt: str) -> bool:
        if self.force_all_remote:
            return False
        policy_cfg = self.cfg.get("escalation_policy", {})
        policy = policy_cfg.get("overrides", {}).get(
            decision["intent"], policy_cfg.get("default", "strict")
        )
        if policy == "always":
            return False
        if decision["difficulty"] != "shallow":
            return False
        # Confidence gate: categorical confidence maps to a score which must
        # EXCEED the config threshold (default 0.90) to stay local.
        conf = _CONFIDENCE_SCORE.get(decision.get("confidence"), 0.0)
        if conf <= self.thresholds.get("local_confidence_threshold", 0.90):
            return False
        # Aggressive cue check runs LAST, before accepting a local answer —
        # but NOT for sentiment/summarization/ner: a stray digit or code
        # fragment inside a review/passage must not escalate those.
        if decision["intent"] not in ("sentiment", "summarization", "ner") and \
                self.thresholds.get("aggressive_escalation", True) and \
                self.aggressive_override(prompt):
            return False
        return True

    # ------------------------------------------------------------------ #
    # full pipeline for one task                                         #
    # ------------------------------------------------------------------ #
    def route(self, task_prompt: str) -> tuple[str, dict]:
        """Return (answer, meta). meta records how the task was routed,
        including per-call wall-clock timing.

        HARD per-task budget: all remote attempts combined must fit inside
        per_task_budget_seconds (~20s). Each request's timeout is clamped to
        the budget remainder, so a slow primary consumes the budget and the
        alternate is skipped — one slow task can never sink the run.
        """
        t_start = time.time()
        timing = {"primary_secs": 0.0, "alternate_fired": False,
                  "alternate_secs": 0.0, "total_secs": 0.0}
        decision = classify_task(
            task_prompt, self.local,
            max_chars=self.limits.get("classify_prompt_chars", 1500),
            max_tokens=self.limits.get("classify_max_tokens", 64),
        )
        category = decision["intent"]
        meta = {
            "decision": decision, "route": "local", "model": "local",
            "finish_reason": "-", "truncated": False,
            "escalation_cue": self.aggressive_override(task_prompt) or "-",
            "timing": timing,
        }

        def _finish_local(route: str = "local", err: Optional[str] = None) -> tuple[str, dict]:
            if err:
                meta.update(route=route, model="local", error=err)
            answer = self._local_answer(task_prompt)
            timing["total_secs"] = round(time.time() - t_start, 2)
            return answer, meta

        if self.should_answer_locally(decision, task_prompt):
            return _finish_local()

        primary = self.resolve_model(category)
        if primary is None:  # ALLOWED_MODELS empty: local is all we have
            return _finish_local()

        role = self.cfg["category_roles"].get(category, "general")
        max_tokens = self.limits.get("remote_max_tokens_by_role", {}).get(
            role, self.limits.get("remote_max_tokens", 512)
        )
        task_budget = self.limits.get("per_task_budget_seconds", 20)
        req_timeout = self.limits.get("remote_timeout_seconds", 18)
        # Primary model, then the first DIFFERENT allowed model, then local.
        alternate = next((m for m in self.allowed if m != primary), None)
        last_err: Optional[Exception] = None
        for idx, model_id in enumerate([m for m in (primary, alternate) if m]):
            remaining = task_budget - (time.time() - t_start)
            if remaining < 3:
                # Not enough runway for another remote attempt — the
                # alternate must never push a task past the budget.
                break
            call_t0 = time.time()
            # minimax on the reasoning role gets "low" effort (real reasoning
            # for math/logic); everything else stays "none" for speed and
            # guaranteed non-empty content.
            effort = "low" if (role == "reasoning" and "minimax" in model_id.lower()) else "none"
            try:
                answer = self.fireworks.chat(
                    model=model_id,
                    system=REMOTE_SYSTEM,
                    user=remote_user_prompt(category, task_prompt),
                    max_tokens=max_tokens,
                    timeout=min(req_timeout, remaining),
                    reasoning_effort=effort,
                )
                call_secs = round(time.time() - call_t0, 2)
                if idx == 0:
                    timing["primary_secs"] = call_secs
                else:
                    timing["alternate_fired"] = True
                    timing["alternate_secs"] = call_secs
                finish = getattr(self.fireworks, "last_finish_reason", None)
                meta.update(
                    route="remote", model=model_id,
                    finish_reason=finish or "?",
                    truncated=finish == "length",
                )
                timing["total_secs"] = round(time.time() - t_start, 2)
                return answer, meta
            except FireworksError as exc:
                call_secs = round(time.time() - call_t0, 2)
                if idx == 0:
                    timing["primary_secs"] = call_secs
                else:
                    timing["alternate_fired"] = True
                    timing["alternate_secs"] = call_secs
                last_err = exc

        # Remote attempts failed or budget exhausted — degrade to local.
        return _finish_local("local_fallback", str(last_err) if last_err else "task budget exhausted")

    def _local_answer(self, task_prompt: str) -> str:
        return self.local.generate(
            task_prompt, max_tokens=self.limits.get("local_max_tokens", 300)
        )
