"""Fireworks client — OpenAI-compatible chat completions over HTTP.

Hard rules honoured here:
  * FIREWORKS_API_KEY / FIREWORKS_BASE_URL are read from the ENVIRONMENT only.
  * Every call goes through FIREWORKS_BASE_URL (the judging proxy).
  * One retry with backoff on transient failures (network/429/5xx);
    4xx is permanent and never retried.
  * THREAD-SAFE: the router now dispatches tasks from a ThreadPoolExecutor,
    so token/call counters are lock-guarded and last_finish_reason is
    thread-local (each worker reads the finish_reason of ITS own call).
"""

from __future__ import annotations

import os
import threading
import time

import requests


class FireworksError(RuntimeError):
    """Raised when a Fireworks call fails after its retry."""


class EmptyCompletion(FireworksError):
    """A 200 response whose message content was empty (hidden-reasoning
    models can burn the whole max_tokens budget on invisible thinking).
    Distinct type so the dispatcher can try the OTHER allowed model once."""


class FireworksClient:
    RETRIES = 1          # one retry with backoff (transient failures only)
    BACKOFF_SECONDS = 2.0

    def __init__(self):
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = os.environ.get("FIREWORKS_BASE_URL", "").rstrip("/")
        self.total_tokens = 0   # prompt + completion tokens as reported by API
        self.calls = 0
        self._lock = threading.Lock()
        self._tls = threading.local()   # per-thread last_finish_reason
        # Models that rejected the reasoning_effort param (400/422): we drop
        # the param for them for the rest of the run instead of failing.
        self._no_reasoning_param: set[str] = set()

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    @property
    def last_finish_reason(self):
        """finish_reason of the calling thread's most recent success."""
        return getattr(self._tls, "finish", None)

    def chat(self, model: str, system: str, user: str,
             max_tokens: int = 512, timeout: float = 25.0,
             reasoning_effort: str = "none") -> str:
        if not self.configured:
            raise FireworksError("FIREWORKS_API_KEY / FIREWORKS_BASE_URL not set")

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        # reasoning_effort control: "none" keeps thinking models fast and
        # non-empty (general lane, code lane); the dispatcher passes "low"
        # only for minimax on the reasoning role (math/logic), where some
        # actual reasoning buys accuracy. Dropped per-model on 400/422.
        if reasoning_effort and model not in self._no_reasoning_param:
            payload["reasoning_effort"] = reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Exception | str | None = None
        for attempt in range(self.RETRIES + 1):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            except requests.RequestException as exc:
                # Network error / timeout: genuinely transient — retry once.
                last_err = exc
                if attempt < self.RETRIES:
                    time.sleep(self.BACKOFF_SECONDS)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                # Rate limit / server error: transient — retry once.
                last_err = f"HTTP {resp.status_code}"
                if attempt < self.RETRIES:
                    time.sleep(self.BACKOFF_SECONDS)
                continue
            if resp.status_code in (400, 422) and "reasoning_effort" in payload:
                # Model rejected the reasoning_effort param. Remember it,
                # drop the param, and re-issue ONCE (recursion is bounded:
                # the model is now in the reject set, so no param next time).
                with self._lock:
                    self._no_reasoning_param.add(model)
                return self.chat(model=model, system=system, user=user,
                                 max_tokens=max_tokens, timeout=timeout,
                                 reasoning_effort=reasoning_effort)
            if resp.status_code >= 400:
                # Auth / unknown model / bad request: permanent. Retrying
                # would waste another full timeout — fail fast, caller
                # writes the deterministic fallback answer.
                raise FireworksError(f"HTTP {resp.status_code} (permanent, not retried)")

            try:
                data = resp.json()
                choice = data["choices"][0]
                text = (choice["message"].get("content") or "").strip()
                finish = choice.get("finish_reason")
            except Exception as exc:
                raise FireworksError(f"malformed response (not retried): {exc}")

            # Tokens were consumed even if the answer is unusable — count them.
            with self._lock:
                self.total_tokens += data.get("usage", {}).get("total_tokens", 0)

            if not text:
                # Empty content: a same-model retry would just repeat it and
                # waste another timeout — fail immediately with the distinct
                # type so the dispatcher can try the other allowed model.
                raise EmptyCompletion(
                    f"empty completion (finish_reason={finish}) — not retried"
                )
            if finish == "length":
                # Truncated but NON-EMPTY: usable — accept it rather than
                # discard a real answer and burn fallback time.
                print("fireworks: truncated completion accepted "
                      "(finish_reason=length)", flush=True)

            with self._lock:
                self.calls += 1
            self._tls.finish = finish
            return text
        raise FireworksError(f"Fireworks call failed after retry: {last_err}")
