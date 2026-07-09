"""Local llama-server lifecycle and client. Zero tokens by the rules.

Hard separation from the Fireworks path: this client is constructed with a
hardcoded loopback base URL and shares no retry/fallback code with
FireworksClient, so a local model id can never reach the judging proxy.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request

log = logging.getLogger("agent.local")

# One shared system prefix across ALL local calls: llama-server's prompt
# cache then prefills it once and reuses it for every task (prefill is the
# CPU bottleneck at 2 vCPUs).
SYSTEM_PROMPT = (
    "You are a precise assistant. Answer directly and completely, "
    "addressing every part of the question. No preamble. Answer in English."
)


def _default_bin() -> str:
    return os.environ.get("AGENT_LLAMA_BIN", "/app/llama/llama-server")


def _default_model() -> str:
    return os.environ.get("AGENT_LLAMA_MODEL", "/app/model.gguf")


class LocalLLM:
    """Spawns llama-server as a subprocess and talks OpenAI-compatible HTTP."""

    def __init__(
        self,
        bin_path: str | None = None,
        model_path: str | None = None,
        port: int | None = None,
        ctx: int = 2048,
        threads: int = 2,
        mmproj_path: str | None = None,
        system_prompt: str | None = None,
    ):
        self.bin_path = bin_path or _default_bin()
        self.model_path = model_path or _default_model()
        self.port = port or int(os.environ.get("AGENT_LLAMA_PORT", "8080"))
        self.ctx = ctx
        self.threads = threads
        self.mmproj_path = mmproj_path  # set -> vision model
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        self._proc: subprocess.Popen | None = None
        self._restarts = 0

    # -- lifecycle -----------------------------------------------------------

    @property
    def available(self) -> bool:
        return os.path.isfile(self.bin_path) and os.path.isfile(self.model_path)

    def start(self, wait_s: float = 55.0) -> bool:
        """Launch and wait for /health. False on any failure (caller falls
        back to escalation) — never raises."""
        if not self.available:
            log.warning(
                "local model unavailable (bin=%s model=%s)",
                self.bin_path, self.model_path,
            )
            return False
        cmd = [
            self.bin_path,
            "-m", self.model_path,
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "-t", str(self.threads),
            "-c", str(self.ctx),
            "--parallel", "1",
            "--jinja",
            # Without --cache-reuse the server saves prompt KV but never
            # reuses partial prefixes; measured cached=1 token per call.
            "--cache-reuse", "256",
        ]
        if self.mmproj_path:
            cmd += ["--mmproj", self.mmproj_path]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as exc:
            log.error("cannot start llama-server: %s", exc)
            return False
        if self._wait_healthy(wait_s):
            log.info("llama-server ready on port %d", self.port)
            return True
        log.error("llama-server did not become healthy in %.0fs", wait_s)
        self.stop()
        return False

    def _wait_healthy(self, wait_s: float) -> bool:
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                return False  # process died
            if self.alive(timeout_s=2.0):
                return True
            time.sleep(0.5)
        return False

    def alive(self, timeout_s: float = 2.0) -> bool:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/health")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def ensure_alive(self) -> bool:
        """Health-gate before each local call: one restart, then give up
        (caller flips the category to escalation for the rest of the run)."""
        if self.alive():
            return True
        if self._restarts >= 1:
            return False
        self._restarts += 1
        log.warning("llama-server unhealthy, attempting restart %d", self._restarts)
        self.stop()
        return self.start(wait_s=30.0)

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    # -- inference -----------------------------------------------------------

    def prewarm(self, prefixes: list[str] | None = None) -> None:
        """Page the weights in and prime the prompt cache.

        Startup has its own 60s budget, so prefilling the constant PoT/codegen
        prefixes here is free — real calls then pay only for generation.
        """
        try:
            self.chat("Reply with OK.", max_tokens=4, timeout_s=30.0)
        except Exception as exc:  # noqa: BLE001 - prewarm is best-effort
            log.warning("prewarm failed: %s", exc)
            return
        for prefix in prefixes or []:
            try:
                self.chat(prefix, max_tokens=1, timeout_s=45.0)
            except Exception as exc:  # noqa: BLE001
                log.warning("prefix prewarm failed: %s", exc)

    def token_estimate(self, text: str) -> int:
        """Cheap upper-bound estimate (~3.5 chars/token for English)."""
        return int(len(text) / 3.5) + 1

    def fits_context(self, prompt: str, gen_budget: int) -> bool:
        overhead = self.token_estimate(SYSTEM_PROMPT) + 32
        return self.token_estimate(prompt) + overhead + gen_budget <= self.ctx

    def chat(self, prompt, max_tokens: int, timeout_s: float = 25.0,
             temperature: float = 0.0) -> str:
        """One local completion. Raises on failure — caller decides fallback.

        `prompt` is a string or an OpenAI-style content-part list (vision).
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": "local",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "cache_prompt": True,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        choices = payload.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""
