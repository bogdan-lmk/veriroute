"""ALLOWED_MODELS parsing and token-oriented ranking.

The leaderboard ranks by total tokens recorded at the judging proxy, not by
dollars, so the ranking optimizes: (1) low hidden-reasoning overhead —
thinking models bill their reasoning as output tokens; (2) expected accuracy
(a larger model costs the same capped tokens but passes the gate more often).
Parameter count is only used as an accuracy proxy inside each group.

Hard rule enforced by the caller: never emit a model id that is not literally
present in ALLOWED_MODELS — that invalidates the whole submission.
"""
from __future__ import annotations

import re

# Name-only heuristics; the real list arrives at runtime and may contain anything.
_THINKING_MARKERS = ("thinking", "reasoner", "-r1", "reason")
_EFFORT_CONTROLLABLE = ("gpt-oss",)  # reasoning_effort=low tames these
# Thinking families that accept reasoning_effort=none, which fully suppresses
# the hidden reasoning stream. Measured live 2026-07-10 on the real API:
# minimax-m3 29->23 and kimi-k2p7-code 78->9 completion tokens, reasoning_content
# gone. These are the only two thinking families in the Track 1 ALLOWED_MODELS,
# so turning reasoning off is what closes the ~9x token gap to the leaders.
# Kept to measured families only; unverified ones stay in _KNOWN_THINKING_FAMILIES.
_REASONING_OFF = ("minimax-m", "kimi-k2")
_KNOWN_THINKING_FAMILIES = ("deepseek-v4", "kimi-k2", "glm-5", "minimax-m")
_CHEAP_VARIANT_MARKERS = ("flash", "mini", "nano", "lite", "tiny", "air", "distill")

_SIZE_RE = re.compile(r"(?:^|[^a-z0-9])(\d+(?:p\d+)?)b(?:$|[^a-z0-9])")
_UNKNOWN_SIZE_B = 30.0  # unranked size sorts between small and frontier


def parse_allowed_models(raw: str | None) -> list[str]:
    """Tolerant CSV parse: strips whitespace, drops empties and duplicates."""
    if not raw:
        return []
    seen: set[str] = set()
    models: list[str] = []
    for part in raw.split(","):
        name = part.strip()
        if name and name not in seen:
            seen.add(name)
            models.append(name)
    return models


def base_name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1].lower()


def size_billions(model_id: str) -> float | None:
    m = _SIZE_RE.search(base_name(model_id))
    return float(m.group(1).replace("p", ".")) if m else None


def reasoning_effort_for(model_id: str) -> str | None:
    """The reasoning_effort value to send this model, or None to omit it.

    gpt-oss is tamed with 'low'; the known thinking families accept 'none',
    which suppresses the reasoning stream entirely (the single biggest token
    lever — their reasoning is billed as completion tokens by the proxy)."""
    name = base_name(model_id)
    if any(k in name for k in _EFFORT_CONTROLLABLE):
        return "low"
    if any(k in name for k in _REASONING_OFF):
        return "none"
    return None


def supports_reasoning_effort(model_id: str) -> bool:
    return reasoning_effort_for(model_id) is not None


def is_thinking_likely(model_id: str) -> bool:
    name = base_name(model_id)
    if supports_reasoning_effort(model_id):
        # Reasoning is controllable (low/none): it bills like a non-thinking
        # model, so it earns tight caps and a good escalation rank.
        return False
    return any(k in name for k in _THINKING_MARKERS) or any(
        f in name for f in _KNOWN_THINKING_FAMILIES
    )


def rank_models(models: list[str]) -> list[str]:
    """Best-first escalation order. Output is always a permutation of input."""

    def score(model_id: str) -> tuple:
        name = base_name(model_id)
        thinking = is_thinking_likely(model_id)
        size = size_billions(model_id)
        if not thinking:
            # Larger => better accuracy for the same capped output tokens.
            return (0, -(size if size is not None else _UNKNOWN_SIZE_B))
        # Forced into thinking models: prefer the cheap/fast variants (their
        # reasoning streams are shorter), then smaller sizes.
        cheap = any(k in name for k in _CHEAP_VARIANT_MARKERS)
        return (1, 0 if cheap else 1, size if size is not None else _UNKNOWN_SIZE_B)

    return sorted(models, key=score)
