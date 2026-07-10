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
import json
import logging
import os
import signal
import sys

from . import io_utils, pot, verifiers
from .classifier import classify
from .deadline import Deadline
from .fireworks_client import (
    DisallowedModelError,
    FireworksClient,
    ModelUnavailableError,
    TokenMeter,
    TransientAPIError,
)
from .local_llm import LocalLLM
from .model_ranking import (
    is_thinking_likely,
    parse_allowed_models,
    rank_models,
    reasoning_effort_for,
)

log = logging.getLogger("agent.main")

STUB_ANSWER = ""

# Multi-part questions die to verbosity, not to model weakness: the model
# spends its budget on preamble and never reaches the second sub-question.
TERSE_SUFFIX = (
    "\n\nAnswer directly and completely. Address every part of the question. "
    "No preamble, no restating the question."
)


def _max_completion_tokens(model: str, category: str = "") -> int:
    """Thinking models bill hidden reasoning as completion tokens — a tight
    cap silently truncates their visible answer (observed live: 3/8 answers
    cut at 300, and escalated codegen shipped a cut reasoning draft at 1000).
    Code tasks get double headroom; non-thinking models stay tight."""
    if is_thinking_likely(model):
        base = int(os.environ.get("AGENT_MAX_TOKENS_THINKING", "1000"))
        if category in ("code_gen", "code_debug"):
            # 2x still shipped a cut reasoning draft on one task; code pays 3x.
            return base * 3
        return base
    if category in ("code_gen", "code_debug"):
        return int(os.environ.get("AGENT_MAX_COMPLETION_TOKENS", "300")) * 2
    return int(os.environ.get("AGENT_MAX_COMPLETION_TOKENS", "300"))


# Only categories whose answers code can defend run locally; everything
# else pays. (category -> local generation budget)
# Eval rounds 2026-07-10 (32 hidden-like tasks, judged): every purely-local
# text category eventually lost to escalation — NER 0/4, summarization 2/4,
# sentiment 3/4 twice (mixed/negative confusion), math-PoT 3/4 (compound
# interest class). Escalated categories score 4/4. What stays local is what
# code can PROVE: executed code tasks. Gate margin beats token rank.
LOCAL_CATEGORIES: dict[str, int] = {}
LOCAL_VERIFIERS: dict = {}
# Execution-verified categories: the sandbox run IS the verifier.
EXECUTED_CATEGORIES = ("code_gen",)


class Router:
    """Local-first cascade: verified local answers are free; only proven
    failures and unverifiable categories escalate to Fireworks."""

    def __init__(
        self,
        client: FireworksClient,
        models: list[str],
        local: LocalLLM | None = None,
    ):
        self.client = client
        self.models = list(models)  # ranked best-first, demoted in place
        self._demoted: set[str] = set()
        self.local = local
        self.local_enabled = local is not None
        self.stats = {"local": 0, "escalated": 0, "local_rejected": 0}

    def answer(self, prompt: str, budget_s: float) -> str:
        """One task. Never raises; returns a stub on any failure."""
        if not prompt.strip():
            return STUB_ANSWER
        category = classify(prompt)
        if self.local_enabled and category in LOCAL_CATEGORIES:
            local_answer = self._try_local(category, prompt, budget_s)
            if local_answer is not None:
                self.stats["local"] += 1
                return local_answer
            self.stats["local_rejected"] += 1
        elif self.local_enabled and category in EXECUTED_CATEGORIES:
            local_answer = self._try_executed(category, prompt, budget_s)
            if local_answer is not None:
                self.stats["local"] += 1
                return local_answer
            self.stats["local_rejected"] += 1
        self.stats["escalated"] += 1
        answer = self._escalate(prompt, budget_s, category)
        if answer:
            return answer
        # Never ship an empty answer: an unverified local guess scores a
        # judge roll; an empty string scores zero with certainty.
        return self._last_resort_local(prompt, category)

    def _last_resort_local(self, prompt: str, category: str = "") -> str:
        if self.local is None or not self.local_enabled:
            return STUB_ANSWER
        try:
            if not self.local.ensure_alive():
                return STUB_ANSWER
            if category in ("code_gen", "code_debug"):
                draft = pot.codegen_best_effort(self.local, prompt, 25.0)
                if draft:
                    log.warning("shipping unverified local code draft")
                    return draft
            guess = self.local.chat(prompt, max_tokens=180, timeout_s=20.0)
            if guess.strip():
                log.warning("escalation empty -> unverified local guess shipped")
                return guess.strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("last-resort local failed: %s", exc)
        return STUB_ANSWER

    def _try_executed(self, category: str, prompt: str, budget_s: float) -> str | None:
        local = self.local
        assert local is not None
        if not local.fits_context(prompt, 300) or not local.ensure_alive():
            if not local.alive():
                self.local_enabled = False
            return None
        # Prefixes are prewarmed at startup, so this window is generation
        # time only; 26s keeps the whole task under the 30s/request rule.
        gen_budget = max(5.0, min(budget_s - 3.0, 26.0))
        return pot.codegen_selftested(local, prompt, gen_budget)

    def _try_local(self, category: str, prompt: str, budget_s: float) -> str | None:
        """Verified local answer or None. Disables the local path for the
        rest of the run if the server dies twice."""
        gen_budget = LOCAL_CATEGORIES[category]
        local = self.local
        assert local is not None
        if not local.fits_context(prompt, gen_budget):
            log.info("local skip (%s): prompt exceeds local context", category)
            return None
        if not local.ensure_alive():
            log.error("llama-server dead after restart; disabling local path")
            self.local_enabled = False
            return None
        try:
            answer = local.chat(
                prompt, max_tokens=gen_budget,
                timeout_s=max(5.0, min(budget_s - 2.0, 20.0)),
            ).strip()
        except Exception as exc:  # noqa: BLE001 - local failure just escalates
            log.warning("local inference failed (%s): %s", category, exc)
            return None
        if answer and LOCAL_VERIFIERS[category](prompt, answer):
            log.info("local answer accepted (%s)", category)
            return answer
        log.info("local answer rejected by verifier (%s)", category)
        return None

    def _escalate(self, prompt: str, budget_s: float, category: str = "") -> str:
        for model in self.models:
            if model in self._demoted:
                continue
            if self.client.meter.exhausted():
                log.warning("token budget exhausted, stubbing remaining work")
                return STUB_ANSWER
            max_tokens = _max_completion_tokens(model, category)
            effort = reasoning_effort_for(model)
            try:
                result = self.client.chat(
                    model, prompt + TERSE_SUFFIX, max_tokens,
                    reasoning_effort=effort, timeout_s=budget_s,
                )
                needs_code = category in ("code_gen", "code_debug")
                cut_draft = (result.finish_reason == "length"
                             and needs_code and "def " not in result.content)
                if (result.truncated or cut_draft) and not self.client.meter.exhausted():
                    # Reasoning ate the budget (empty or a cut draft without
                    # code): exactly one paid retry with more headroom.
                    log.warning("%s truncated by reasoning, one retry x2 tokens", model)
                    result = self.client.chat(
                        model, prompt + TERSE_SUFFIX, max_tokens * 2,
                        reasoning_effort=effort, timeout_s=budget_s,
                    )
                if needs_code and "def " not in (result.content or ""):
                    log.warning("escalated code answer has no def; trying next path")
                    continue
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

    # Dual-mode dispatch: Track 2 tasks carry video_url instead of prompt.
    try:
        with open(io_utils.input_path(), "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list) and any(
            isinstance(t, dict) and "video_url" in t for t in raw
        ):
            log.info("video_url detected -> captioning mode (Track 2)")
            from .captioner import runner as captioner_runner
            video_tasks = [
                t for t in raw
                if isinstance(t, dict) and isinstance(t.get("task_id"), str)
            ]
            return captioner_runner.run(video_tasks)
    except (OSError, json.JSONDecodeError):
        pass  # fall through to the Track 1 defensive parser

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

    # --- Local model: free answers where verifiers can defend them ---
    local: LocalLLM | None = LocalLLM()
    if local.available and local.start():
        local.prewarm(prefixes=pot.prompt_prefixes())
    else:
        local = None

    if not client.usable and local is None:
        log.error(
            "neither Fireworks (base_url=%r, %d models) nor local model "
            "usable; keeping stub answers",
            client.base_url, len(allowed),
        )
        return 0

    router = Router(client, rank_models(allowed), local=local)
    log.info(
        "model escalation order: %s | local model: %s",
        router.models, "on" if local else "off",
    )

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

    if local is not None:
        local.stop()
    log.info(
        "done: %d tasks (%d local / %d escalated / %d local-rejected), "
        "%d fireworks calls, %d prompt + %d completion = %d tokens, %.1fs elapsed",
        len(tasks), router.stats["local"], router.stats["escalated"],
        router.stats["local_rejected"], meter.calls, meter.prompt_tokens,
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
