"""Program-of-Thought and self-tested codegen on the local model.

Small models are weak at mental arithmetic but decent at writing trivial
programs (PAL/EffGen: Qwen2.5-1.5B math jumps ~36% -> ~74% with execution).
The execution result is the verification: no run, no answer — escalate.
"""
from __future__ import annotations

import logging
import re

from .local_llm import LocalLLM
from .sandbox import extract_code, run_python

log = logging.getLogger("agent.pot")

# Few-shot with one worked example: a 1.5B model writes far more careful
# arithmetic when it sees named intermediate steps (PAL-style).
_MATH_PROMPT = (
    "Solve the problem by writing a short Python program. Use a named "
    "variable for every intermediate step, in the order the story tells it. "
    "Define solve() returning the final answer and print(solve()).\n\n"
    "Example problem: A shop has 80 cakes. It sells 25% in the morning and "
    "10 more in the afternoon. How many cakes remain?\n"
    "Example answer:\n"
    "```python\n"
    "def solve():\n"
    "    start = 80\n"
    "    sold_morning = start * 0.25      # sold -> leaves the shop\n"
    "    sold_afternoon = 10              # '10 more' also SOLD -> also leaves\n"
    "    remaining = start - sold_morning - sold_afternoon\n"
    "    return round(remaining)\n"
    "print(solve())\n"
    "```\n\n"
    "Problem: {prompt}\n\n"
    "Reply with a single ```python code block."
)

# One generation, not two: at the grader's 2-vCPU decode speed a second
# call blows the 28s per-task budget and the whole category degrades to
# paid escalation.
_CODEGEN_PROMPT = (
    "{prompt}\n\n"
    "Reply with a single ```python code block containing the function "
    "followed by exactly 3 assert statements that test it with literal inputs."
)


_COUNT_QUESTION = re.compile(r"\bhow (many|much)\b|\bremain", re.I)


def _plausible(prompt: str, answer: str) -> bool:
    """Executable is not correct: cheap sanity checks on the number itself."""
    try:
        value = float(answer.replace(",", "").rstrip("%"))
    except ValueError:
        return False
    # Counts of things can't be negative; that's the classic sign-error bug.
    if _COUNT_QUESTION.search(prompt) and value < 0:
        return False
    return True


def math_pot(local: LocalLLM, prompt: str, budget_s: float) -> str | None:
    """Numeric answer via generated-and-executed code, or None.

    One local retry on rejection — local tokens are free; only time matters.
    """
    per_try = max(4.0, budget_s / 2)
    for attempt in (1, 2):
        try:
            raw = local.chat(
                _MATH_PROMPT.format(prompt=prompt),
                max_tokens=250, timeout_s=per_try,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("math PoT generation failed (try %d): %s", attempt, exc)
            return None
        code = extract_code(raw)
        if "print" not in code:
            code += "\nprint(solve())"
        ok, out = run_python(code)
        if not ok or not out:
            log.info("math PoT execution rejected (try %d): %s", attempt, out[:120])
            continue
        answer = out.splitlines()[-1].strip()
        if not re.fullmatch(r"-?[\d,]+(\.\d+)?%?", answer.replace(" ", "")):
            log.info("math PoT output not numeric (try %d): %r", attempt, answer[:60])
            continue
        if not _plausible(prompt, answer):
            log.info("math PoT implausible (try %d): %r", attempt, answer[:60])
            continue
        # Integers read better than 144.0 to an LLM judge.
        if answer.endswith(".0"):
            answer = answer[:-2]
        return answer
    return None


def codegen_selftested(local: LocalLLM, prompt: str, budget_s: float) -> str | None:
    """Generated function that passes its own generated asserts, or None."""
    try:
        raw = local.chat(
            _CODEGEN_PROMPT.format(prompt=prompt),
            max_tokens=400, timeout_s=budget_s,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("codegen generation failed: %s", exc)
        return None
    block = extract_code(raw)
    if "def " not in block or "assert" not in block:
        return None
    ok, out = run_python(block)
    if not ok:
        log.info("codegen self-tests failed: %s", out[:120])
        return None
    # Ship the verified function without the test lines.
    solution = "\n".join(
        line for line in block.splitlines()
        if not line.lstrip().startswith("assert")
    ).strip()
    return f"```python\n{solution}\n```"
