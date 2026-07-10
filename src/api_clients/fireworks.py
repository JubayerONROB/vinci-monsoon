"""Fireworks client — OpenAI-compatible chat completions over HTTP.

Hard rules honoured here:
  * FIREWORKS_API_KEY / FIREWORKS_BASE_URL are read from the ENVIRONMENT only.
  * Every call goes through FIREWORKS_BASE_URL (the judging proxy).
  * One retry with backoff, 25s default timeout — under the 30s per-request cap.
"""

from __future__ import annotations

import os
import time

import requests


class FireworksError(RuntimeError):
    """Raised when a Fireworks call fails after its retry."""


class FireworksClient:
    RETRIES = 1          # one retry with backoff
    BACKOFF_SECONDS = 2.0

    def __init__(self):
        self.api_key = os.environ.get("FIREWORKS_API_KEY", "")
        self.base_url = os.environ.get("FIREWORKS_BASE_URL", "").rstrip("/")
        self.total_tokens = 0   # prompt + completion tokens as reported by API
        self.calls = 0
        self.last_finish_reason = None  # finish_reason of the last success
        # Models that rejected the reasoning_effort param (400/422): we drop
        # the param for them for the rest of the run instead of failing.
        self._no_reasoning_param: set[str] = set()

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    def chat(self, model: str, system: str, user: str,
             max_tokens: int = 512, timeout: float = 25.0) -> str:
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
        # Disable hidden reasoning: thinking models (minimax-m3) otherwise
        # burn the token budget reasoning and return empty/truncated content
        # after ~25s. Dropped per-model if the API rejects the param.
        if model not in self._no_reasoning_param:
            payload["reasoning_effort"] = "none"
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
                self._no_reasoning_param.add(model)
                return self.chat(model=model, system=system, user=user,
                                 max_tokens=max_tokens, timeout=timeout)
            if resp.status_code >= 400:
                # Auth / unknown model / bad request: permanent. Retrying
                # would waste another full timeout — fail fast, caller
                # falls back to the local model.
                raise FireworksError(f"HTTP {resp.status_code} (permanent, not retried)")

            try:
                data = resp.json()
                choice = data["choices"][0]
                text = (choice["message"].get("content") or "").strip()
                finish = choice.get("finish_reason")
            except Exception as exc:
                raise FireworksError(f"malformed response (not retried): {exc}")

            # Tokens were consumed even if the answer is unusable — count them.
            self.total_tokens += data.get("usage", {}).get("total_tokens", 0)

            if not text:
                # Hidden-reasoning models can burn the whole max_tokens budget
                # and return EMPTY content with finish_reason=length. A retry
                # would just repeat it and waste another timeout — fail
                # IMMEDIATELY so the caller falls back locally.
                raise FireworksError(
                    f"empty completion (finish_reason={finish}) — not retried"
                )
            if finish == "length":
                # Truncated but NON-EMPTY: usable — accept it rather than
                # discard a real answer and burn local-fallback time.
                print("fireworks: truncated completion accepted "
                      "(finish_reason=length)", flush=True)

            self.calls += 1
            self.last_finish_reason = finish
            return text
        raise FireworksError(f"Fireworks call failed after retry: {last_err}")
