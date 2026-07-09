"""The stylist: Gemma 3 4B writes all four captions in ONE call per clip.

Few-shot beat both LoRA arms in the blind bake-off (style 7.75 vs 7.38/7.50)
at zero weight cost — see gemmacap/training/bakeoff.py. One call per clip
because decode dominates on 2 grader vCPUs (~3-6 tok/s): halving the number
of calls/prefills is the difference between fitting the 10-minute cap or not.
"""
from __future__ import annotations

import json
import logging
import re

from ..local_llm import LocalLLM

log = logging.getLogger("agent.captioner.stylist")

STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")

# NOTE: substituted via str.replace, NOT str.format — the few-shot example
# contains literal JSON braces that .format() would treat as placeholders.
ALL_STYLES_PROMPT = (
    'Example scene: A small orange kitten walks through green foliage.\n'
    'Example output: {"formal": "A small orange kitten walks through dense '
    'green foliage.", "sarcastic": "A kitten, boldly going where thousands '
    'of kittens have gone before.", "humorous_tech": "The kitten patrols the '
    'undergrowth like a garbage collector hunting unreferenced mice.", '
    '"humorous_non_tech": "Somewhere between the second and third leaf, the '
    'kitten forgot what it was hunting."}\n\n'
    "Video scene: {desc}\n\n"
    "Write four one-sentence captions for this video as JSON with keys "
    "formal, sarcastic, humorous_tech, humorous_non_tech.\n"
    "formal: professional, objective. sarcastic: dry, deadpan, no "
    "exclamations. humorous_tech: witty with a natural technology reference. "
    "humorous_non_tech: relatable everyday humour, no tech jargon.\n"
    "Each caption under 20 words, facts only from the scene. "
    "Reply with the raw JSON object only, no markdown fences."
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


def prompt_prefix() -> str:
    """Constant prompt prefix to prewarm into the Gemma server's cache."""
    return ALL_STYLES_PROMPT.split("{desc}")[0]


def style_all(local: LocalLLM, desc: str, styles: list[str],
              budget_s: float) -> dict[str, str]:
    """All requested styles in one generation; per-style fallbacks on failure."""
    wanted = tuple(k for k in STYLES if k in styles)
    if not wanted:
        return {}
    parsed = None
    for attempt, temp in enumerate((0.7, 0.4)):
        try:
            raw = local.chat(ALL_STYLES_PROMPT.replace("{desc}", desc),
                             max_tokens=220, timeout_s=budget_s,
                             temperature=temp)
        except Exception as exc:  # noqa: BLE001
            log.warning("stylist failed (try %d): %s", attempt + 1, exc)
            continue
        parsed = _extract(raw, wanted)
        if parsed:
            break
        log.warning("stylist parse failed (try %d): %r", attempt + 1, raw[:160])
    return {k: parsed[k].strip() if parsed else fallback_caption(desc, k)
            for k in wanted}
