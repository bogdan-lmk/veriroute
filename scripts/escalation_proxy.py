"""Minimal escalation relay: keeps the Fireworks key OFF the public image.

Runs on our own box for the hackathon window. Hard limits: one endpoint,
one allowed model, capped max_tokens, small per-IP rate budget. Kill the
process (or the droplet) and the exposure window closes — the agent inside
the public image degrades to its local-only path automatically.

Usage: FIREWORKS_API_KEY=... python3 escalation_proxy.py [port]
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://api.fireworks.ai/inference/v1/chat/completions"
ALLOWED_MODEL = "accounts/fireworks/models/minimax-m3"
MAX_TOKENS_CAP = 700
RATE_LIMIT = 60          # requests per window per IP
RATE_WINDOW_S = 600

KEY = os.environ["FIREWORKS_API_KEY"]
_hits: dict[str, list[float]] = defaultdict(list)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: N802
        sys.stderr.write("%s %s\n" % (self.client_address[0], fmt % args))

    def _reject(self, code: int, msg: str) -> None:
        body = json.dumps({"error": {"message": msg}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._reject(404, "unknown path")
        ip = self.client_address[0]
        now = time.time()
        _hits[ip] = [t for t in _hits[ip] if now - t < RATE_WINDOW_S]
        if len(_hits[ip]) >= RATE_LIMIT:
            return self._reject(429, "rate limited")
        _hits[ip].append(now)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
        except Exception:  # noqa: BLE001
            return self._reject(400, "bad json")
        if body.get("model") != ALLOWED_MODEL:
            return self._reject(403, "model not allowed")
        body["max_tokens"] = min(int(body.get("max_tokens", 300)), MAX_TOKENS_CAP)
        body["stream"] = False
        req = urllib.request.Request(
            UPSTREAM, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {KEY}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = r.read()
        except urllib.error.HTTPError as exc:
            return self._reject(exc.code, "upstream error")
        except Exception:  # noqa: BLE001
            return self._reject(502, "upstream unreachable")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
    print(f"escalation proxy on :{port} -> {ALLOWED_MODEL}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
