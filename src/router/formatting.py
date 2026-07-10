"""Per-category output-format discipline.

The LLM judge grades partial credit; preamble-wrapped or verbose answers bleed
points. format_answer() is CONSERVATIVE: it strips filler and tightens shape
per category, and it must NEVER blank out a non-empty answer — if a transform
would empty the text, the original is returned unchanged.
"""

from __future__ import annotations

import re

# Leading filler common to chatty models ("Sure, here's ...", "The answer is:")
_PREAMBLE = re.compile(
    r"^(sure[,!.]?\s+|certainly[,!.]?\s+|of course[,!.]?\s+|"
    r"here(?:'s| is)\b[^:\n]{0,60}:\s*|the answer is:?\s+)",
    re.IGNORECASE,
)

_SENTIMENT_LABELS = ("positive", "negative", "neutral", "mixed")

# Lines that look like structured entity output ("- Maria Sanchez: PERSON")
_NER_LINE = re.compile(r"^\s*([-*•]|\d+[.)])\s+\S|^\s*\{")


def _strip_preamble(text: str) -> str:
    # Filler stacks ("Sure, here is the answer: ..."), so strip repeatedly
    # (bounded) until no leading filler remains.
    stripped = text
    for _ in range(3):
        new = _PREAMBLE.sub("", stripped, count=1).strip()
        if new == stripped or not new:
            break
        stripped = new
    return stripped or text


def format_answer(category: str, text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    original = text
    text = _strip_preamble(text.strip())

    if category == "sentiment":
        # Reduce to the single bare label mentioned earliest in the answer.
        head = text.lower()[:120]
        found = [(head.find(lbl), lbl) for lbl in _SENTIMENT_LABELS if lbl in head]
        if found:
            return min(found)[1]
        return text  # no recognizable label — keep the model's words

    if category == "ner":
        # Keep only the structured entity lines, drop surrounding prose.
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        entity_lines = [ln for ln in lines if _NER_LINE.match(ln)]
        if entity_lines:
            return "\n".join(entity_lines)
        return text  # answer wasn't list-shaped — keep as-is

    if category == "math_reasoning":
        # Prompt already demands the result as the last "Answer:" line;
        # here we only ensure no filler precedes the working.
        return text

    if category == "factual_knowledge":
        return text  # preamble already stripped; keep the direct answer

    # summarization / code / everything else: substantive content stays,
    # only the leading filler (already stripped above) is removed.
    return text if text.strip() else original
