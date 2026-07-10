"""The single choke point for all Fireworks traffic.

Invariant: no request leaves this client unless its model id is literally in
ALLOWED_MODELS — calling anything else invalidates the whole submission, so
the check happens before any network I/O and is unit-tested.

Requests are non-streaming only: the usage block is then reliably present,
and our token meter must reconcile with the judging proxy (>10% divergence
is treated as a blocking bug per the plan).
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

log = logging.getLogger("agent.fireworks")


class DisallowedModelError(RuntimeError):
    """Refused before any network I/O: model id not in ALLOWED_MODELS."""


class ModelUnavailableError(RuntimeError):
    """400/404 for this model (serverless rotation): demote it, try the next."""


class TransientAPIError(RuntimeError):
    """Retryable failure (429/5xx/network); raised after retries are exhausted."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TokenMeter:
    """Global token accounting with a hard stop.

    The stop is a runaway guard, NOT an optimization target: an exhausted
    budget stubs remaining answers and forfeits the accuracy gate — which is
    exactly how an earlier submission scored 9/19. Rank means nothing below
    the gate, so the default leaves generous headroom.
    """

    def __init__(self, budget: int | None = None):
        self.budget = (
            budget
            if budget is not None
            else int(os.environ.get("AGENT_TOKEN_BUDGET", "15000"))
        )
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, usage: dict) -> None:
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def exhausted(self) -> bool:
        return self.total >= self.budget


class ChatResult:
    def __init__(self, content: str, finish_reason: str, usage: dict):
        self.content = content
        self.finish_reason = finish_reason
        self.usage = usage

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length" and not self.content.strip()


class FireworksClient:
    def __init__(
        self,
        base_url: str | None,
        api_key: str | None,
        allowed_models: list[str],
        meter: TokenMeter | None = None,
        timeout_s: float | None = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.allowed = frozenset(allowed_models)
        self.meter = meter or TokenMeter()
        self.timeout_s = (
            timeout_s
            if timeout_s is not None
            else float(os.environ.get("AGENT_HTTP_TIMEOUT_S", "25"))
        )

    @property
    def usable(self) -> bool:
        return bool(self.base_url and self.allowed)

    def chat(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        reasoning_effort: str | None = None,
        retries: int = 2,
        timeout_s: float | None = None,
    ) -> ChatResult:
        if model not in self.allowed:
            raise DisallowedModelError(f"model {model!r} is not in ALLOWED_MODELS")
        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort

        attempt = 0
        while True:
            attempt += 1
            try:
                return self._post_chat(body, timeout_s or self.timeout_s)
            except TransientAPIError as exc:
                if attempt > retries:
                    raise
                delay = exc.retry_after if exc.retry_after else 1.5 * attempt
                log.warning(
                    "transient error from %s (attempt %d/%d), retrying in %.1fs: %s",
                    model, attempt, retries + 1, delay, exc,
                )
                time.sleep(min(delay, 10.0))

    def _post_chat(self, body: dict, timeout_s: float) -> ChatResult:
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001 - diagnostics only
                pass
            if exc.code in (400, 404):
                raise ModelUnavailableError(
                    f"{body['model']}: HTTP {exc.code} {detail}"
                ) from exc
            retry_after = None
            ra = exc.headers.get("Retry-After") if exc.headers else None
            if ra:
                try:
                    retry_after = float(ra)
                except ValueError:
                    retry_after = None
            raise TransientAPIError(
                f"HTTP {exc.code} {detail}", retry_after=retry_after
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransientAPIError(f"network error: {exc}") from exc

        usage = payload.get("usage") or {}
        self.meter.add(usage)
        choices = payload.get("choices") or [{}]
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        finish_reason = choices[0].get("finish_reason") or ""
        log.info(
            "fireworks call model=%s prompt_tokens=%s completion_tokens=%s total_so_far=%d",
            body["model"],
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            self.meter.total,
        )
        return ChatResult(content=content, finish_reason=finish_reason, usage=usage)
