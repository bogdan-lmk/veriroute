"""Format verifiers for locally-produced answers.

A local answer ships only when code can defend it; anything else escalates.
These are deliberately cheap, deterministic checks — the point is to catch
the small model's obvious failure modes (empty output, ignored format
constraints, off-list labels), not to grade semantics.
"""
from __future__ import annotations

import re

_ONE_SENTENCE_CUE = re.compile(r"\b(exactly\s+)?one sentence\b", re.I)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
# "Classify X as positive, negative, or neutral" — capture an enumerated label set.
_LABEL_SET_CUE = re.compile(
    r"\bas\b[:\s]+((?:[\w-]+,\s*)+(?:or\s+)?[\w-]+)", re.I
)
_COMMON_SENTIMENT_LABELS = ("positive", "negative", "neutral", "mixed")


def verify_sentiment(prompt: str, answer: str) -> bool:
    a = answer.strip()
    if not a or len(a) > 600:
        return False
    low = a.lower()
    m = _LABEL_SET_CUE.search(prompt)
    if m:
        labels = [w.strip().lower() for w in re.split(r",|\bor\b", m.group(1)) if w.strip()]
        return any(lbl in low for lbl in labels)
    return any(lbl in low for lbl in _COMMON_SENTIMENT_LABELS)


def verify_ner(prompt: str, answer: str) -> bool:  # noqa: ARG001 - symmetry
    a = answer.strip()
    if not a or len(a) > 1500:
        return False
    # An extraction should name at least one Capitalized entity and a type-ish
    # word; a refusal/apology has neither shape.
    has_capitalized = bool(re.search(r"\b[A-Z][a-z]+", a))
    has_typing = bool(
        re.search(r"\b(person|organi[sz]ation|location|date|company|place)\b", a, re.I)
        or re.search(r"[|:\-–]\s*\w+", a)
    )
    return has_capitalized and has_typing


def verify_summarization(prompt: str, answer: str) -> bool:
    a = answer.strip()
    if not a:
        return False
    if _ONE_SENTENCE_CUE.search(prompt):
        # Exactly one terminal punctuation mark = one sentence, allowing
        # trailing whitespace. Abbreviation dots are rare enough in summaries.
        return len(_SENTENCE_END.findall(a)) == 1 and a[-1] in ".!?"
    return len(a) < len(prompt)  # a summary must actually compress
