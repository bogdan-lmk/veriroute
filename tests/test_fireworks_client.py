"""FireworksClient: the ALLOWED_MODELS invariant and failure handling."""
import io
import json
import urllib.error

import pytest

from agent.fireworks_client import (
    DisallowedModelError,
    FireworksClient,
    ModelUnavailableError,
    TokenMeter,
    TransientAPIError,
)

MODEL = "accounts/fireworks/models/gpt-oss-20b"


def make_client(**kwargs) -> FireworksClient:
    defaults = dict(
        base_url="https://proxy.example/v1",
        api_key="test-key",
        allowed_models=[MODEL],
        timeout_s=5.0,
    )
    defaults.update(kwargs)
    return FireworksClient(**defaults)


def ok_response(content="four", prompt_tokens=10, completion_tokens=2,
                finish_reason="stop"):
    body = json.dumps({
        "choices": [{
            "message": {"content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }).encode()

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return _Resp(body)


class TestDisallowedModelInvariant:
    def test_raises_before_any_network_io(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("network I/O attempted for a disallowed model")

        monkeypatch.setattr("urllib.request.urlopen", explode)
        client = make_client()
        with pytest.raises(DisallowedModelError):
            client.chat("accounts/fireworks/models/not-allowed", "hi", 10)

    def test_local_model_id_never_reaches_proxy(self, monkeypatch):
        """The disqualification scenario: a local llama model id in escalation."""
        def explode(*args, **kwargs):
            raise AssertionError("local model id sent to the judging proxy")

        monkeypatch.setattr("urllib.request.urlopen", explode)
        client = make_client()
        with pytest.raises(DisallowedModelError):
            client.chat("qwen2.5-1.5b-instruct", "hi", 10)

    def test_unusable_without_base_url_or_models(self):
        assert not make_client(base_url="").usable
        assert not make_client(allowed_models=[]).usable
        assert make_client().usable


class TestChatHappyPath:
    def test_content_and_usage_metered(self, monkeypatch):
        meter = TokenMeter(budget=1000)
        client = make_client(meter=meter)
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout: ok_response()
        )
        result = client.chat(MODEL, "2+2?", 10)
        assert result.content == "four"
        assert meter.total == 12
        assert meter.calls == 1

    def test_truncated_detection(self, monkeypatch):
        client = make_client()
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout: ok_response(content="", finish_reason="length"),
        )
        assert client.chat(MODEL, "2+2?", 10).truncated


class TestFailureHandling:
    def _http_error(self, code, headers=None):
        return urllib.error.HTTPError(
            "https://proxy.example/v1/chat/completions", code, "err",
            headers or {}, io.BytesIO(b"detail"),
        )

    def test_404_is_model_unavailable(self, monkeypatch):
        def raise_404(req, timeout):
            raise self._http_error(404)

        monkeypatch.setattr("urllib.request.urlopen", raise_404)
        with pytest.raises(ModelUnavailableError):
            make_client().chat(MODEL, "hi", 10)

    def test_400_is_model_unavailable(self, monkeypatch):
        def raise_400(req, timeout):
            raise self._http_error(400)

        monkeypatch.setattr("urllib.request.urlopen", raise_400)
        with pytest.raises(ModelUnavailableError):
            make_client().chat(MODEL, "hi", 10)

    def test_429_retries_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        def flaky(req, timeout):
            calls["n"] += 1
            if calls["n"] < 3:
                raise self._http_error(429)
            return ok_response()

        sleeps = []
        monkeypatch.setattr("urllib.request.urlopen", flaky)
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        result = make_client().chat(MODEL, "hi", 10, retries=2)
        assert result.content == "four"
        assert calls["n"] == 3
        assert len(sleeps) == 2

    def test_persistent_5xx_exhausts_retries(self, monkeypatch):
        def always_500(req, timeout):
            raise self._http_error(500)

        monkeypatch.setattr("urllib.request.urlopen", always_500)
        monkeypatch.setattr("time.sleep", lambda s: None)
        with pytest.raises(TransientAPIError):
            make_client().chat(MODEL, "hi", 10, retries=2)

    def test_network_error_is_transient(self, monkeypatch):
        def no_route(req, timeout):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr("urllib.request.urlopen", no_route)
        monkeypatch.setattr("time.sleep", lambda s: None)
        with pytest.raises(TransientAPIError):
            make_client().chat(MODEL, "hi", 10, retries=1)


class TestTokenMeter:
    def test_budget_stop(self):
        meter = TokenMeter(budget=100)
        meter.add({"prompt_tokens": 60, "completion_tokens": 30})
        assert not meter.exhausted()
        meter.add({"prompt_tokens": 10, "completion_tokens": 5})
        assert meter.exhausted()

    def test_missing_usage_fields_tolerated(self):
        meter = TokenMeter(budget=100)
        meter.add({})
        meter.add({"prompt_tokens": None})
        assert meter.total == 0
