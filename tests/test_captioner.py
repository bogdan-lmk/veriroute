"""Captioner mode: dispatch, consistency check, stylist parsing, fallbacks."""
import json

from agent.captioner.eye import frames_agree
from agent.captioner.stylist import STYLES, _extract, fallback_caption


class TestFramesAgree:
    def test_consistent_descriptions(self):
        assert frames_agree([
            "A kitten walks through green foliage in a forest.",
            "The orange kitten moves between leaves in the forest.",
        ])

    def test_drifting_descriptions(self):
        assert not frames_agree([
            "A black motorcycle helmet with a visor sits on a table.",
            "Children play football on a sunny beach near water.",
        ])

    def test_single_description_trusted(self):
        assert frames_agree(["A dog runs."])


class TestStylistTemplates:
    def test_replace_substitution_survives_fewshot_braces(self):
        """Regression: .format() blew up on few-shot JSON braces (KeyError)."""
        from agent.captioner.stylist import ALL_STYLES_PROMPT
        rendered = ALL_STYLES_PROMPT.replace("{desc}", "A dog runs.")
        assert "A dog runs." in rendered
        assert "{desc}" not in rendered
        assert '"formal"' in rendered and '"humorous_tech"' in rendered


class TestStylistParsing:
    def test_extract_valid(self):
        raw = 'text {"formal": "A river flows through the forest.", ' \
              '"sarcastic": "A river doing exactly what rivers do."} tail'
        obj = _extract(raw, ("formal", "sarcastic"))
        assert obj and obj["formal"].startswith("A river")

    def test_placeholder_rejected(self):
        raw = '{"formal": "...", "sarcastic": "..."}'
        assert _extract(raw, ("formal", "sarcastic")) is None

    def test_fallbacks_cover_all_styles(self):
        desc = "A kitten walks through the forest. It is orange."
        for style in STYLES:
            cap = fallback_caption(desc, style)
            assert cap and "kitten" in cap.lower()


class TestModeDispatch:
    def test_video_tasks_route_to_captioner(self, tmp_path, monkeypatch):
        tasks = [{"task_id": "v1",
                  "video_url": "https://example.com/x.mp4",
                  "styles": ["formal"]}]
        (tmp_path / "tasks.json").write_text(json.dumps(tasks))
        monkeypatch.setenv("AGENT_INPUT_PATH", str(tmp_path / "tasks.json"))
        monkeypatch.setenv("AGENT_OUTPUT_PATH", str(tmp_path / "out" / "results.json"))
        called = {}

        from agent.captioner import runner
        def fake_run(ts):
            called["tasks"] = ts
            return 0

        monkeypatch.setattr(runner, "run", fake_run)
        from agent import main
        assert main.run() == 0
        assert called["tasks"][0]["task_id"] == "v1"

    def test_prompt_tasks_stay_in_router(self, tmp_path, monkeypatch):
        tasks = [{"task_id": "t1", "prompt": "2+2?"}]
        (tmp_path / "tasks.json").write_text(json.dumps(tasks))
        monkeypatch.setenv("AGENT_INPUT_PATH", str(tmp_path / "tasks.json"))
        monkeypatch.setenv("AGENT_OUTPUT_PATH", str(tmp_path / "out" / "results.json"))
        for var in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS",
                    "AGENT_LLAMA_BIN", "AGENT_LLAMA_MODEL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("AGENT_LLAMA_MODEL", str(tmp_path / "nope.gguf"))
        from agent import main
        assert main.run() == 0
        results = json.loads((tmp_path / "out" / "results.json").read_text())
        assert results == [{"task_id": "t1", "answer": ""}]
