"""Deterministic task-category classification.

The classifier drives the whole policy table, so it must be conservative:
anything ambiguous (no match, multiple competing matches, non-English) falls
back to ESCALATE — a small token cost beats a silent accuracy loss on the
16/19 gate. The LLM never classifies its own work.
"""
from __future__ import annotations

import re

CATEGORIES = (
    "sentiment",
    "ner",
    "summarization",
    "math",
    "logic",
    "code_debug",
    "code_gen",
    "factual",
)
ESCALATE = "escalate"

_PATTERNS: dict[str, list[re.Pattern]] = {
    "sentiment": [
        re.compile(r"\bsentiment\b", re.I),
        re.compile(r"\bclassify\b.{0,40}\b(review|tone|opinion)\b", re.I),
    ],
    "ner": [
        re.compile(r"\bnamed entit(y|ies)\b", re.I),
        re.compile(r"\bextract\b.{0,60}\bentit(y|ies)\b", re.I),
    ],
    "summarization": [
        re.compile(r"\bsummari[sz]e\b", re.I),
        re.compile(r"\bsummary\b", re.I),
        re.compile(r"\bcondense\b|\btl;?dr\b", re.I),
    ],
    # Math needs BOTH a digit somewhere and an arithmetic cue (see classify);
    # these patterns are only the cue half.
    "math": [
        re.compile(r"\bhow (many|much)\b", re.I),
        re.compile(r"\d+\s*%|\bpercent", re.I),
        re.compile(r"\b(remain(s|ing)?|total|left over|altogether|in total)\b", re.I),
        re.compile(r"\bcalculate\b|\bsum\b|\bdifference\b", re.I),
    ],
    "logic": [
        re.compile(r"\bwho (owns?|has|likes|drinks|lives)\b", re.I),
        re.compile(r"\beach (own|have|like|drink)s?\b", re.I),
        re.compile(r"\b(logic|deduce|deduction|constraint)\b", re.I),
    ],
    "code_debug": [
        re.compile(r"\b(bug|buggy|broken)\b", re.I),
        re.compile(r"\bfix (it|this|the)\b", re.I),
        re.compile(r"\bdebug\b", re.I),
    ],
    "code_gen": [
        re.compile(r"\bwrite\b.{0,40}\b(function|method|class|program|script)\b", re.I),
        re.compile(r"\bimplement\b.{0,40}\b(function|method|algorithm)\b", re.I),
    ],
}

_CODE_SIGNAL = re.compile(r"```|\bdef \w+\(|\bfunction\s+\w+\(|\breturn\b")
_WORD = re.compile(r"[A-Za-z]")


def _non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ord(ch) > 127) / len(text)


def classify(prompt: str) -> str:
    """Return one category or ESCALATE when unsure."""
    if not prompt.strip() or not _WORD.search(prompt):
        return ESCALATE
    # Sub-2B local models degrade sharply off-English; play it safe.
    if _non_ascii_ratio(prompt) > 0.15:
        return ESCALATE

    hits = {
        cat: True
        for cat, patterns in _PATTERNS.items()
        if any(p.search(prompt) for p in patterns)
    }
    # The math cue only counts when there are actual numbers to compute with.
    if "math" in hits and not re.search(r"\d", prompt):
        del hits["math"]

    # Code snippets pull debug/gen apart: with code present, a "fix" cue wins
    # over generation cues; without code, "write a function" is generation.
    has_code = bool(_CODE_SIGNAL.search(prompt))
    if "code_debug" in hits and not has_code:
        del hits["code_debug"]
    if "code_debug" in hits and "code_gen" in hits:
        del hits["code_gen"]
    # Math word problems often trip the logic patterns ("each", "who has"):
    # numbers + arithmetic cues make it math.
    if "math" in hits and "logic" in hits:
        del hits["logic"]

    if len(hits) == 1:
        return next(iter(hits))
    if len(hits) > 1:
        return ESCALATE  # multi-part or ambiguous prompt — pay, don't guess
    # No pattern matched: short question-shaped prompts are factual recall.
    if len(prompt) < 400 and prompt.rstrip().endswith("?"):
        return "factual"
    return ESCALATE
