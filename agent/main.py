"""Entry point: Phase 0 "escalate everything" baseline with full guardrails.

Layered defenses (each one maps to a real failure status on the leaderboard):
1. /output is created and probed, and a stub results.json keyed by the real
   task_ids is written before anything else can fail   -> no OUTPUT_MISSING
2. results.json is atomically rewritten after every answered task
                                                        -> kill-safe progress
3. Per-task try/except with stub fallback               -> no RUNTIME_ERROR
4. Global deadline with a parallel drain at T-flush     -> no TIMEOUT
5. ALLOWED_MODELS invariant inside FireworksClient      -> no MODEL_VIOLATION
6. Token budget hard stop                                -> never exceed leader
7. SIGTERM/SIGINT handlers + top-level except            -> always exit 0
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import signal
import sys

from . import io_utils
from .deadline import Deadline
from .fireworks_client import (
    DisallowedModelError,
    FireworksClient,
    ModelUnavailableError,
    TokenMeter,
    TransientAPIError,
)
from .model_ranking import (
    is_thinking_likely,
    parse_allowed_models,
    rank_models,
    supports_reasoning_effort,
)

log = logging.getLogger("agent.main")

STUB_ANSWER = ""

# Multi-part questions die to verbosity, not to model weakness: the model
# spends its budget on preamble and never reaches the second sub-question.
TERSE_SUFFIX = (
    "\n\nAnswer directly and completely. Address every part of the question. "
    "No preamble, no restating the question."
)


def _max_completion_tokens(model: str) -> int:
    """Thinking models bill hidden reasoning as completion tokens — a tight
    cap silently truncates their visible answer (observed live: 3/8 answers
    cut at 300). Give them headroom; keep non-thinking models tight."""
    if is_thinking_likely(model):
        # Measured live on minimax-m3 (8 practice tasks): cap 600 starved
        # codegen (empty content -> paid retry, 4411 total); cap 1000 kept
        # every answer intact at 3573 total. Reasoning models need headroom.
        return int(os.environ.get("AGENT_MAX_TOKENS_THINKING", "1000"))
    return int(os.environ.get("AGENT_MAX_COMPLETION_TOKENS", "300"))


class Router:
    """Escalate-all baseline: every task goes to the best allowed model."""

    def __init__(self, client: FireworksClient, models: list[str]):
        self.client = client
        self.models = list(models)  # ranked best-first, demoted in place
        self._demoted: set[str] = set()

    def answer(self, prompt: str, budget_s: float) -> str:
        """One task. Never raises; returns a stub on any failure."""
        if not prompt.strip():
            return STUB_ANSWER
        for model in self.models:
            if model in self._demoted:
                continue
            if self.client.meter.exhausted():
                log.warning("token budget exhausted, stubbing remaining work")
                return STUB_ANSWER
            max_tokens = _max_completion_tokens(model)
            effort = "low" if supports_reasoning_effort(model) else None
            try:
                result = self.client.chat(
                    model, prompt + TERSE_SUFFIX, max_tokens,
                    reasoning_effort=effort, timeout_s=budget_s,
                )
                if result.truncated and not self.client.meter.exhausted():
                    # Reasoning ate the whole budget: exactly one paid retry.
                    log.warning("%s truncated by reasoning, one retry x2 tokens", model)
                    result = self.client.chat(
                        model, prompt + TERSE_SUFFIX, max_tokens * 2,
                        reasoning_effort=effort, timeout_s=budget_s,
                    )
                answer = result.content.strip()
                if answer:
                    return answer
                log.warning("%s returned empty content, trying next model", model)
            except ModelUnavailableError as exc:
                log.warning("demoting %s: %s", model, exc)
                self._demoted.add(model)
            except TransientAPIError as exc:
                log.warning("%s failed after retries: %s; trying next model", model, exc)
            except DisallowedModelError:
                raise  # programming error: must never be swallowed
        return STUB_ANSWER


def run() -> int:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    deadline = Deadline()

    # --- Guardrail 1: output writable + stubs before anything can fail ---
    io_utils.ensure_output_dir()
    tasks = io_utils.load_tasks()
    answers: dict[str, str] = {t["task_id"]: STUB_ANSWER for t in tasks}
    io_utils.write_results_atomic(answers)
    if not tasks:
        log.error("no usable tasks; wrote empty results and exiting 0")
        return 0

    # --- Guardrail 7: clean exit on signals, best-known answers on disk ---
    def _terminate(signum: int, _frame) -> None:
        log.warning("signal %d received, flushing results and exiting", signum)
        try:
            io_utils.write_results_atomic(answers)
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)

    # --- Environment (injected by the harness; absent in dev is fine) ---
    meter = TokenMeter()
    allowed = parse_allowed_models(os.environ.get("ALLOWED_MODELS"))
    client = FireworksClient(
        base_url=os.environ.get("FIREWORKS_BASE_URL"),
        api_key=os.environ.get("FIREWORKS_API_KEY"),
        allowed_models=allowed,
        meter=meter,
    )
    if not client.usable:
        log.error(
            "Fireworks not usable (base_url=%r, %d allowed models); "
            "keeping stub answers",
            client.base_url, len(allowed),
        )
        return 0

    router = Router(client, rank_models(allowed))
    log.info("model escalation order: %s", router.models)

    # --- Sequential pass with per-task budget ---
    pending = list(tasks)
    while pending and not deadline.flush_due():
        task = pending.pop(0)
        budget = deadline.task_budget()
        if budget <= 1.0:
            pending.insert(0, task)
            break
        try:
            answers[task["task_id"]] = router.answer(task["prompt"], budget)
        except Exception:  # noqa: BLE001 - guardrail 3
            log.exception("task %s failed unexpectedly, stubbed", task["task_id"])
        io_utils.write_results_atomic(answers)

    # --- Guardrail 4: parallel drain of the remainder inside the margin ---
    if pending:
        log.warning("draining %d unfinished tasks in parallel", len(pending))
        # Conservative affordability: assume every drained task pays the
        # thinking-model cap, so the token budget can never be overshot.
        worst_case_tokens = max(
            _max_completion_tokens(m) for m in router.models
        ) if router.models else 1
        affordable = max(0, (meter.budget - meter.total) // worst_case_tokens)
        to_escalate, to_stub = pending[:affordable], pending[affordable:]
        if to_stub:
            log.warning("%d tasks stubbed to respect the token budget", len(to_stub))
        if to_escalate:
            drain_timeout = max(5.0, deadline.remaining() - 15.0)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(8, len(to_escalate))
            ) as pool:
                futures = {
                    pool.submit(router.answer, t["prompt"], 20.0): t["task_id"]
                    for t in to_escalate
                }
                done, not_done = concurrent.futures.wait(
                    futures, timeout=drain_timeout
                )
                for fut in done:
                    try:
                        answers[futures[fut]] = fut.result()
                    except Exception:  # noqa: BLE001
                        log.exception("drain task %s failed, stubbed", futures[fut])
                for fut in not_done:
                    fut.cancel()
        io_utils.write_results_atomic(answers)

    log.info(
        "done: %d tasks, %d fireworks calls, %d prompt + %d completion = %d tokens, %.1fs elapsed",
        len(tasks), meter.calls, meter.prompt_tokens,
        meter.completion_tokens, meter.total, deadline.elapsed(),
    )
    return 0


def main() -> None:
    try:
        code = run()
    except BaseException:  # noqa: BLE001 - guardrail 7: never exit non-zero
        log.exception("fatal error; results.json already holds best-known answers")
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    main()
