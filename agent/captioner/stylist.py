"""The stylist: Gemma 3 4B writes all four captions (two few-shot calls).

Few-shot beat both LoRA arms in the blind bake-off (style 7.75 vs 7.38/7.50)
at zero weight cost — see gemmacap/training/bakeoff.py.
"""
from __future__ import annotations

import json
import logging
import re

from ..local_llm import LocalLLM

log = logging.getLogger("agent.captioner.stylist")

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")

_FEWSHOT_PAIR1 = (
    'Example scene: A river winds through a dense forest.\n'
    'Example output: {"formal": "A river winds through a dense forest, '
    'bordered by tall trees.", "sarcastic": "A river, boldly committing to '
    'the one direction it was always going to take."}\n\n'
)
_FEWSHOT_PAIR2 = (
    'Example scene: A small orange kitten walks through green foliage.\n'
    'Example output: {"humorous_tech": "The kitten patrols the undergrowth '
    'like a garbage collector hunting unreferenced mice.", '
    '"humorous_non_tech": "Somewhere between the second and third leaf, the '
    'kitten forgot what it was hunting."}\n\n'
)

PAIR1 = (
    _FEWSHOT_PAIR1 +
    "Video scene: {desc}\n\n"
    "Write two one-sentence captions for this video as JSON "
    '{{"formal": "...", "sarcastic": "..."}}.\n'
    "formal: professional, objective, factual tone.\n"
    "sarcastic: dry, ironic, lightly mocking — understatement and deadpan "
    "beat exclamation marks.\n"
    "Use only facts from the scene description. Reply with JSON only."
)
PAIR2 = (
    _FEWSHOT_PAIR2 +
    "Video scene: {desc}\n\n"
    "Write two funny one-sentence captions for this video as JSON "
    '{{"humorous_tech": "...", "humorous_non_tech": "..."}}.\n'
    "humorous_tech: witty, with a technology/programming reference that fits "
    "the scene naturally.\n"
    "humorous_non_tech: funny everyday humour, zero technical jargon.\n"
    "Use only facts from the scene description. Reply with JSON only."
)


def _extract(raw: str, keys: tuple[str, ...]) -> dict | None:
    best = None
    for m in re.finditer(r"\{[^{}]*\}", raw, re.S):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if all(isinstance(obj.get(k), str) and len(obj[k].strip()) >= 15
               and obj[k].strip() != "..." for k in keys):
            best = obj
    return best


def fallback_caption(desc: str, style: str) -> str:
    """Deterministic last resort: judged weak on style but factual."""
    base = desc.split(".")[0].strip() or "A short video clip"
    return {
        "formal": f"{base}.",
        "sarcastic": f"{base} — truly groundbreaking footage.",
        "humorous_tech": f"{base}, running smoothly with zero software updates required.",
        "humorous_non_tech": f"{base}, and honestly it seems to be having a better day than most of us.",
    }[style]


def style_captions(local: LocalLLM, desc: str, styles: list[str],
                   budget_s: float) -> dict[str, str]:
    """All requested styles via the two-pair calls, with retries + fallbacks."""
    out: dict[str, str] = {}
    per_call = max(6.0, budget_s / 2)
    for template, keys in ((PAIR1, ("formal", "sarcastic")),
                           (PAIR2, ("humorous_tech", "humorous_non_tech"))):
        wanted = tuple(k for k in keys if k in styles)
        if not wanted:
            continue
        parsed = None
        for attempt, temp in enumerate((0.7, 0.4)):
            try:
                raw = local.chat(template.format(desc=desc),
                                 max_tokens=220, timeout_s=per_call,
                                 temperature=temp)
            except Exception as exc:  # noqa: BLE001
                log.warning("stylist call failed (try %d): %s", attempt + 1, exc)
                continue
            parsed = _extract(raw, keys)
            if parsed:
                break
        for k in wanted:
            out[k] = parsed[k].strip() if parsed else fallback_caption(desc, k)
    return out
