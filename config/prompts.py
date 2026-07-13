"""Prompt templates for both pipeline stages.

Token discipline: the stage-1 classifier runs locally (0 scored tokens) but a
compact prompt keeps CPU latency low. The stage-2 remote prompts DO count
toward the token score, so every word here is deliberate — keep system prompts
minimal and push the model toward short, judge-friendly answers.
"""

INTENTS = [
    "factual_knowledge",
    "math_reasoning",
    "sentiment",
    "summarization",
    "ner",
    "code_debugging",
    "logical_reasoning",
    "code_generation",
]

# --- Stage 1: local classification (grammar-constrained JSON) ----------------
CLASSIFY_SYSTEM = (
    "Classify the user task. Reply with JSON only:\n"
    '{"intent": one of ' + "|".join(INTENTS) + ",\n"
    ' "difficulty": "shallow" if a small 3B model can answer it reliably else "deep",\n'
    ' "confidence": "high" or "low"}'
)

# --- Stage 2a: local answering ------------------------------------------------
LOCAL_ANSWER_SYSTEM = (
    "You are a precise assistant. Answer in English, correctly and concisely. "
    "No preamble, no repetition of the question."
)

# --- Stage 2b: remote (Fireworks) answering ----------------------------------
# One shared minimal system prompt + a per-category output-format hint.
# These hints keep OUTPUT tokens low while still satisfying an LLM judge.
# "Answer in English" stays: kimi/minimax can drift languages and a
# non-English answer is a guaranteed judge fail — cheap insurance.
REMOTE_SYSTEM = "Answer in English, correct and concise."

CATEGORY_STYLE = {
    "factual_knowledge": "Answer in 1-3 sentences.",
    # Judge rubric wants the calculation shown or implied — keep brief working.
    # "only" is load-bearing: dropping it in run #59 coincided with +234
    # completion tokens across 5 math tasks. Byte-exact d6fcd4d string.
    "math_reasoning": "Brief working only, then end with 'Answer: <value>'.",
    # T03 rubric: the reason must acknowledge BOTH sides of a mixed review.
    "sentiment": "Sentiment label (positive/negative/neutral/mixed) + one-sentence justification.",
    "summarization": "Follow the requested length/format exactly. Output only the summary.",
    "ner": "List each entity as '- <entity>: <TYPE>' using types like PERSON, ORG, LOCATION, DATE.",
    # Code lanes: code-only is the cheap default; the task prompt overrides
    # when it explicitly wants explanation.
    "code_debugging": "Only the corrected code in one code block; prose only if asked.",
    # A/B verdict: conclusion-only broke hard-logic-01 (kimi's visible
    # reasoning IS its thinking) — step-by-step-brief is the proven floor.
    "logical_reasoning": "Reason step by step briefly, then end with 'Answer: <conclusion>'.",
    "code_generation": "Only the code in one code block; prose only if asked.",
}


def remote_user_prompt(category: str, task_prompt: str) -> str:
    """Compose the stage-2 remote prompt: task + tiny format instruction."""
    style = CATEGORY_STYLE.get(category, "")
    return f"{task_prompt}\n\n{style}".strip()
