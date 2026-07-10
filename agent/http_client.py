"""The one place that speaks HTTP JSON.

The review found three hand-rolled urllib variants (fireworks client, local
llama-server client, vision escalation). They differ only in headers and
error policy — both stay with the callers; the wire code lives here.
"""
from __future__ import annotations

import json
import urllib.request


def post_json(url: str, body: dict, headers: dict[str, str] | None = None,
              timeout_s: float = 25.0) -> dict:
    """POST body as JSON, return the parsed JSON response.

    Raises urllib.error.HTTPError / URLError / TimeoutError / OSError /
    json.JSONDecodeError untouched — callers own the error policy.
    """
    all_headers = {"Content-Type": "application/json"}
    if headers:
        all_headers.update(headers)
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers=all_headers, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_message(payload: dict) -> dict:
    """The message object of the first choice, tolerating absent fields."""
    choices = payload.get("choices") or [{}]
    return choices[0].get("message") or {}
