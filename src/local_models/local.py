"""Verifier-gated LOCAL MODEL lane (Ollama sidecar) — zero scored tokens.

Local answers cost ZERO toward the token ranking, so this lane tries the
baked-in Ollama model first for the categories in LOCAL_CATEGORIES (default
sentiment,ner,summarization) and escalates to the normal PAID remote path
whenever anything is off — FAIL-OPEN by design:

  * Ollama unreachable / call errors / timeout        -> escalate
  * done_reason == "length" (cut mid-thought)         -> escalate
  * deterministic format verifier rejects the answer  -> escalate
  * global deadline budget below the guard threshold  -> skip local entirely

A single LOCAL_LOCK serializes model calls (one 3B model on a 2-core box).
Local prompts optimize for JUDGE satisfaction, not brevity — local tokens
are free, so the answers can be as thorough as the rubric wants.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Optional

import requests

# --------------------------------------------------------------------------- #
# Deterministic format verifiers (pure functions — unit-tested offline)        #
# --------------------------------------------------------------------------- #

# Abbreviations whose trailing '.' must not count as a sentence boundary.
_ABBREV_RE = re.compile(
    r"\b(?:U\.S\.A|U\.S|U\.K|E\.U|Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr|No|vs|etc|"
    r"e\.g|i\.e|approx|Inc|Ltd|Co|Fig|al)\.",
    re.IGNORECASE,
)

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _to_num(tok: str) -> Optional[int]:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return _NUM_WORDS.get(tok)


def count_sentences(text: str) -> int:
    """Abbreviation- and decimal-safe sentence counter."""
    t = text.strip()
    # Mask abbreviation dots and decimal points so they don't split.
    t = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\x00"), t)
    t = re.sub(r"(\d)\.(\d)", "\\1\x00\\2", t)
    parts = re.split(r"[.!?]+(?:\s+|$)", t)
    return sum(1 for p in parts if p.strip())


def requested_sentences(prompt: str) -> Optional[int]:
    m = re.search(r"exactly\s+(\w+)\s+sentences?", prompt, re.IGNORECASE)
    if not m:
        m = re.search(r"\bin\s+(\w+)\s+sentences?", prompt, re.IGNORECASE)
    return _to_num(m.group(1)) if m else None


def requested_bullets(prompt: str) -> Optional[int]:
    m = re.search(r"exactly\s+(\w+)\s+bullet", prompt, re.IGNORECASE)
    if not m:
        m = re.search(r"\bin\s+(\w+)\s+bullet", prompt, re.IGNORECASE)
    return _to_num(m.group(1)) if m else None


def requested_word_limit(prompt: str) -> Optional[int]:
    m = re.search(r"no (?:longer|more) than\s+(\d+)\s+words", prompt, re.IGNORECASE)
    if not m:
        m = re.search(r"(?:under|at most|fewer than|each under)\s+(\d+)\s+words", prompt, re.IGNORECASE)
    return int(m.group(1)) if m else None


_BULLET_RE = re.compile(r"^[-*•]\s+")


def verify_summarization(prompt: str, answer: str) -> bool:
    """Sentence/bullet count (and per-bullet word limit) must match the
    prompt's explicit constraint — the T04/T04b rubrics fail on any drift."""
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    if not lines:
        return False
    bullets = [l for l in lines if _BULLET_RE.match(l)]
    nb = requested_bullets(prompt)
    if nb is not None:
        if len(bullets) != nb or len(bullets) != len(lines):
            return False
        limit = requested_word_limit(prompt)
        if limit is not None:
            for b in bullets:
                if len(_BULLET_RE.sub("", b).split()) > limit:
                    return False
        return True
    ns = requested_sentences(prompt)
    if ns is not None:
        if bullets:  # asked for sentences, got a list
            return False
        return count_sentences(answer) == ns
    return True  # no explicit constraint — any non-empty summary


_NER_LINE_RE = re.compile(r"^[-*•]?\s*[^:]{1,80}:\s*\S")

# Leading words of a capitalized run that are not part of an entity name,
# plus generic acronyms that capitalized-run extraction would false-flag.
_CAP_STOPWORDS = {
    "The", "A", "An", "In", "On", "At", "He", "She", "It", "They", "We", "I",
    "This", "That", "These", "Those", "However", "But", "And", "Or", "After",
    "Before", "Then", "From", "With", "During", "Extract", "Label", "Text",
    "CEO", "CTO", "CFO", "Dr", "Mr", "Mrs", "Ms", "Prof",
    "AI", "IT", "OK", "TV", "PC", "LLM", "API", "Q1", "Q2", "Q3", "Q4",
}

_DATE_RE = re.compile(
    r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b|\b[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}\b"
)
_CAP_RUN_RE = re.compile(r"\b[A-Z][\w&'-]*(?:\s+[A-Z][\w&'-]*)*")


def required_ner_mentions(task_prompt: str) -> list[str]:
    """Deterministic completeness set: capitalized runs + dates from the
    SOURCE text (after the instruction prefix). A local NER answer missing
    any of these escalates — a 3B model silently dropping an entity is the
    known failure mode, and the T05 rubric fails any missing entity."""
    text = task_prompt.split(":", 1)[1] if ":" in task_prompt else task_prompt
    required: list[str] = []
    for m in _DATE_RE.finditer(text):
        required.append(m.group(0))
    for m in _CAP_RUN_RE.finditer(text):
        words = [w.rstrip(".") for w in m.group(0).split()]
        while words and words[0].rstrip(".") in _CAP_STOPWORDS:
            words = words[1:]
        if words and not all(w in _CAP_STOPWORDS for w in words):
            required.append(" ".join(words))
    return required


def verify_ner(answer: str, task_prompt: str = "") -> bool:
    """List-shaped '<entity>: <TYPE>' lines AND no source entity missing."""
    lines = [l.strip() for l in answer.splitlines() if l.strip()]
    if not lines:
        return False
    good = sum(1 for l in lines if _NER_LINE_RE.match(l))
    # Allow one header-ish line, but the body must be entity lines.
    if not (good >= max(1, len(lines) - 1) and good >= 1):
        return False
    low = answer.lower()
    for mention in required_ner_mentions(task_prompt):
        if mention.lower() not in low:
            return False
    return True


_LABEL_RE = re.compile(r"\b(positive|negative|neutral|mixed)\b", re.IGNORECASE)
_CONTRAST_RE = re.compile(r"\b(but|however|although|though|yet)\b", re.IGNORECASE)
_REASON_ASK_RE = re.compile(r"\b(reason|justify|justification|explain|why)\b", re.IGNORECASE)


def verify_sentiment(prompt: str, answer: str) -> bool:
    m = _LABEL_RE.search(answer)
    if not m:
        return False
    # T03/T03b rubric: a contrastive (mixed) review must NOT be labelled
    # Negative — that fails regardless of the reason given.
    if _CONTRAST_RE.search(prompt) and m.group(1).lower() == "negative":
        return False
    # If the prompt asks for a reason, a bare label is not enough.
    if _REASON_ASK_RE.search(prompt) and len(answer.split()) < 6:
        return False
    return True


def verify(category: str, prompt: str, answer: str) -> bool:
    if not answer or not answer.strip():
        return False
    if category == "summarization":
        return verify_summarization(prompt, answer)
    if category == "ner":
        return verify_ner(answer, prompt)
    if category == "sentiment":
        return verify_sentiment(prompt, answer)
    return False  # lane only certifies the categories it knows how to check


# --------------------------------------------------------------------------- #
# Judge-optimized local prompts (local tokens are free)                        #
# --------------------------------------------------------------------------- #

def build_local_prompt(category: str, task_prompt: str) -> str:
    if category == "sentiment":
        wants_reason = bool(_REASON_ASK_RE.search(task_prompt))
        base = (
            f"{task_prompt}\n\n"
            "Give the sentiment label (Positive, Negative, Neutral, or Mixed)."
        )
        if wants_reason or _CONTRAST_RE.search(task_prompt):
            base += (
                " If the text contains both good and bad points, do NOT answer"
                " Negative — answer Mixed or Positive, and give exactly one"
                " sentence that acknowledges BOTH the negative points AND the"
                " positive points mentioned."
            )
        if wants_reason:
            base += " Format: <Label> — <one-sentence reason>."
        else:
            base += " Reply with the label only."
        return base
    if category == "ner":
        return (
            f"{task_prompt}\n\n"
            "List EVERY named entity, one per line, in the exact format"
            " '- <entity>: <TYPE>' using the types PERSON, ORGANIZATION,"
            " LOCATION, DATE. Include every person, organization, location"
            " and date mentioned — do not miss any, and output nothing but"
            " the list."
        )
    # summarization
    return (
        f"{task_prompt}\n\n"
        "Obey the requested format EXACTLY: if a number of sentences is"
        " requested, output exactly that many sentences; if bullet points are"
        " requested, output exactly that many bullets (one per line, starting"
        " with '- ') and respect any per-bullet word limit. Cover both the"
        " positives and the concerns in the source. Output only the summary."
    )


# --------------------------------------------------------------------------- #
# The lane                                                                     #
# --------------------------------------------------------------------------- #

class LocalLane:
    def __init__(self, deadline_ts: Optional[float] = None):
        self.url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
        raw = os.environ.get("LOCAL_CATEGORIES", "sentiment,ner,summarization")
        self.categories = {c.strip() for c in raw.split(",") if c.strip()}
        self.num_predict = int(os.environ.get("LOCAL_NUM_PREDICT", "450"))
        self.min_remaining = float(os.environ.get("LOCAL_MIN_REMAINING_SECONDS", "180"))
        self.deadline_ts = deadline_ts
        self.launch_ts = time.time()
        self.last_done_reason: Optional[str] = None
        self._lock = threading.Lock()          # ONE model, 2 cores: serialize
        self._first_done = False
        self._dead = False
        self._consec_errors = 0

    # ------------------------------------------------------------------ #
    def warm_async(self) -> None:
        """Load the model in the background (num_predict=1, keep_alive 30m)
        so the cold start doesn't land on the first real task."""
        def _warm():
            try:
                self._post("hi", num_predict=1, timeout=180)
                print("local lane: model warmed", flush=True)
            except Exception as exc:
                print(f"local lane: warm-up skipped ({type(exc).__name__})", flush=True)
        threading.Thread(target=_warm, daemon=True).start()

    def _post(self, prompt: str, num_predict: int, timeout: float) -> dict:
        """One /api/generate call. During startup, connection-refused is
        retried until the server binds (bounded by the call's own timeout)."""
        call_deadline = time.time() + timeout
        while True:
            try:
                r = requests.post(
                    f"{self.url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": "30m",
                        "options": {"num_predict": num_predict, "temperature": 0.2},
                    },
                    timeout=max(5.0, call_deadline - time.time()),
                )
                r.raise_for_status()
                return r.json()
            except requests.exceptions.ConnectionError:
                # Sidecar still binding? A locally-baked server binds within
                # seconds — wait only inside a short launch window, so a
                # genuinely dead sidecar costs seconds, not whole timeouts.
                if self._first_done or time.time() > call_deadline - 5 \
                        or time.time() - self.launch_ts > 25:
                    raise
                time.sleep(2)

    # ------------------------------------------------------------------ #
    def try_answer(self, category: str, task_prompt: str) -> Optional[str]:
        """Return a verified local answer, or None to escalate (fail-open)."""
        if self._dead or category not in self.categories:
            return None
        # Deadline guard: never let a slow local queue starve the run.
        if self.deadline_ts is not None and \
                time.time() > self.deadline_ts - self.min_remaining:
            return None
        with self._lock:
            # Re-check after waiting for the lock — the queue ahead of us
            # may have eaten the budget.
            if self._dead:
                return None
            if self.deadline_ts is not None and \
                    time.time() > self.deadline_ts - self.min_remaining:
                return None
            timeout = 60.0 if not self._first_done else 15.0
            if category == "summarization":
                # CPU generation is slow and summaries are the longest local
                # outputs; local time is nearly free (deadline-guarded), so
                # give them room instead of escalating to a paid call.
                timeout = max(timeout, 25.0) + min(45.0, len(task_prompt) / 150.0)
            try:
                data = self._post(build_local_prompt(category, task_prompt),
                                  num_predict=self.num_predict, timeout=timeout)
                self._first_done = True
                self._consec_errors = 0
            except Exception as exc:
                print(f"local lane: escalate {category} "
                      f"({type(exc).__name__})", flush=True)
                self._consec_errors += 1
                # Never came up at all, or persistently failing -> stop trying.
                if (not self._first_done and self._consec_errors >= 2) \
                        or self._consec_errors >= 3:
                    self._dead = True
                    print("local lane: disabled after repeated failures "
                          "(fail-open to remote)", flush=True)
                return None
            text = (data.get("response") or "").strip()
            done_reason = data.get("done_reason")
            self.last_done_reason = done_reason
            if not text:
                print(f"local lane: escalate {category} (empty)", flush=True)
                return None
            if done_reason == "length":
                print(f"local lane: escalate {category} (length)", flush=True)
                return None  # cut mid-thought — never submit it
            if not verify(category, task_prompt, text):
                print(f"local lane: escalate {category} (verifier)", flush=True)
                return None
            return text
