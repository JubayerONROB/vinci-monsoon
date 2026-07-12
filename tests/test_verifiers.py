"""Unit tests for the local-lane deterministic verifiers (pure functions)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_models.local import (  # noqa: E402
    count_sentences,
    requested_bullets,
    requested_sentences,
    requested_word_limit,
    required_ner_mentions,
    verify,
    verify_ner,
    verify_sentiment,
    verify_summarization,
)

T04_PROMPT = "Summarize the following passage in exactly two sentences: ..."
T04B_PROMPT = ("Summarize the following passage in exactly three bullet points, "
               "each no longer than 15 words: ...")


def test_sentence_counter_abbreviation_safe():
    assert count_sentences("Dr. Smith visited the U.S. last year. He liked it.") == 2
    assert count_sentences("Costs rose 3.5 percent. Revenue fell.") == 2
    assert count_sentences("One sentence only, e.g. this one.") == 1


def test_requested_format_extraction():
    assert requested_sentences(T04_PROMPT) == 2
    assert requested_sentences("Summarize the following in exactly one sentence: x") == 1
    assert requested_bullets(T04B_PROMPT) == 3
    assert requested_word_limit(T04B_PROMPT) == 15


def test_summarization_verifier_sentence_count():
    ok = "ML helps healthcare in many tasks. However, concerns remain unresolved."
    assert verify_summarization(T04_PROMPT, ok)
    assert not verify_summarization(T04_PROMPT, ok + " A third sentence sneaks in.")
    assert not verify_summarization(T04_PROMPT, "- a bullet instead\n- of sentences")


def test_summarization_verifier_bullets_and_word_limit():
    good = "- Remote work boosts flexibility.\n- Challenges persist around culture.\n- Firms invest in digital tools."
    assert verify_summarization(T04B_PROMPT, good)
    assert not verify_summarization(T04B_PROMPT, good + "\n- a fourth bullet")
    long_bullet = ("- " + " ".join(["word"] * 16) + "\n- short one here.\n- another short one.")
    assert not verify_summarization(T04B_PROMPT, long_bullet)


def test_ner_verifier():
    assert verify_ner("- Maria Sanchez: PERSON\n- Berlin: LOCATION")
    assert verify_ner("Sundar Pichai: PERSON\nGoogle: ORGANIZATION")
    assert not verify_ner("The text mentions several people and places in passing.")
    assert not verify_ner("")


DENSE_NER_PROMPT = ("Extract all named entities and their types from: On 12 March 2025, "
                    "CEO Lena Vogel of Baltic Dynamics met European Commission officials "
                    "in Brussels, then flew to Warsaw with Dr. Tomasz Kowalski of the "
                    "University of Gdansk.")

# The exact run #46 local failure: 7 entities listed, Baltic Dynamics dropped.
DENSE_NER_MISSING_ONE = ("- Lena Vogel: PERSON\n- European Commission: ORGANIZATION\n"
                         "- Brussels: LOCATION\n- Dr. Tomasz Kowalski: PERSON\n"
                         "- University of Gdansk: ORGANIZATION\n- Warsaw: LOCATION\n"
                         "- 12 March 2025: DATE")


def test_ner_completeness_gate():
    assert not verify_ner(DENSE_NER_MISSING_ONE, DENSE_NER_PROMPT)
    assert verify_ner(DENSE_NER_MISSING_ONE + "\n- Baltic Dynamics: ORGANIZATION",
                      DENSE_NER_PROMPT)
    req = required_ner_mentions(DENSE_NER_PROMPT)
    assert any("Baltic Dynamics" in r for r in req)
    assert any("12 March 2025" in r for r in req)
    # T05 source: generic 'AI' must not be demanded as an entity.
    t05 = ("Extract all named entities from the following text and label each as "
           "PERSON, ORGANIZATION, LOCATION, or DATE:\n\n'On March 15 2023, Sundar "
           "Pichai announced that Google would open a new AI research lab in Zurich, "
           "partnering with ETH Zurich to focus on large language model safety.'")
    t05_answer = ("- March 15 2023: DATE\n- Sundar Pichai: PERSON\n- Google: ORGANIZATION\n"
                  "- Zurich: LOCATION\n- ETH Zurich: ORGANIZATION")
    assert verify_ner(t05_answer, t05)


def test_sentiment_verifier():
    ask = "Classify the sentiment ... and give a one-sentence reason: 'slow but great.'"
    assert verify_sentiment(ask, "Mixed — slow delivery but a great product overall.")
    # Bare label when a reason was requested -> escalate.
    assert not verify_sentiment(ask, "Mixed")
    # T03 rubric: Negative never passes a contrastive (both-sides) review.
    assert not verify_sentiment(ask, "Negative — the delivery was slow despite a great product.")
    # No label at all -> escalate.
    assert not verify_sentiment(ask, "The customer seems torn about the product.")
    # Bare label is fine when no reason was requested.
    assert verify_sentiment("Classify the sentiment of this review: meh but fine", "mixed")


def test_verify_dispatcher_unknown_category_rejects():
    assert not verify("math_reasoning", "2+2?", "4")
    assert not verify("sentiment", "prompt", "")
