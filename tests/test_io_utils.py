"""I/O contract: defensive parsing and crash-safe atomic writes."""
import json

import pytest

from agent import io_utils


def _write(tmp_path, content: str):
    p = tmp_path / "tasks.json"
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestLoadTasks:
    def test_valid_tasks(self, tmp_path):
        path = _write(tmp_path, json.dumps([
            {"task_id": "t1", "prompt": "What is 2+2?"},
            {"task_id": "t2", "prompt": "Summarise this."},
        ]))
        tasks = io_utils.load_tasks(path)
        assert [t["task_id"] for t in tasks] == ["t1", "t2"]

    def test_missing_file(self, tmp_path):
        assert io_utils.load_tasks(str(tmp_path / "nope.json")) == []

    def test_invalid_json(self, tmp_path):
        assert io_utils.load_tasks(_write(tmp_path, "{not json")) == []

    def test_root_not_a_list(self, tmp_path):
        assert io_utils.load_tasks(_write(tmp_path, '{"task_id": "t1"}')) == []

    def test_non_dict_entries_skipped(self, tmp_path):
        path = _write(tmp_path, json.dumps(
            ["junk", 42, None, {"task_id": "t1", "prompt": "p"}]
        ))
        assert [t["task_id"] for t in io_utils.load_tasks(path)] == ["t1"]

    def test_missing_or_bad_task_id_skipped(self, tmp_path):
        path = _write(tmp_path, json.dumps([
            {"prompt": "no id"},
            {"task_id": 7, "prompt": "int id"},
            {"task_id": "", "prompt": "empty id"},
            {"task_id": "ok", "prompt": "p"},
        ]))
        assert [t["task_id"] for t in io_utils.load_tasks(path)] == ["ok"]

    def test_duplicate_task_id_first_wins(self, tmp_path):
        path = _write(tmp_path, json.dumps([
            {"task_id": "t1", "prompt": "first"},
            {"task_id": "t1", "prompt": "second"},
        ]))
        tasks = io_utils.load_tasks(path)
        assert len(tasks) == 1
        assert tasks[0]["prompt"] == "first"

    def test_non_string_prompt_degrades_to_stub(self, tmp_path):
        path = _write(tmp_path, json.dumps([{"task_id": "t1", "prompt": 42}]))
        tasks = io_utils.load_tasks(path)
        assert tasks == [{"task_id": "t1", "prompt": ""}]


class TestWriteResultsAtomic:
    def test_roundtrip_schema(self, tmp_path):
        out = str(tmp_path / "results.json")
        io_utils.write_results_atomic({"t1": "a1", "t2": ""}, out)
        data = json.loads((tmp_path / "results.json").read_text())
        assert data == [
            {"task_id": "t1", "answer": "a1"},
            {"task_id": "t2", "answer": ""},
        ]

    def test_overwrites_previous(self, tmp_path):
        out = str(tmp_path / "results.json")
        io_utils.write_results_atomic({"t1": "old"}, out)
        io_utils.write_results_atomic({"t1": "new"}, out)
        data = json.loads((tmp_path / "results.json").read_text())
        assert data == [{"task_id": "t1", "answer": "new"}]

    def test_no_tmp_litter(self, tmp_path):
        out = str(tmp_path / "results.json")
        io_utils.write_results_atomic({"t1": "a"}, out)
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "results.json"]
        assert leftovers == []

    def test_unicode_preserved(self, tmp_path):
        out = str(tmp_path / "results.json")
        io_utils.write_results_atomic({"t1": "ответ 攻殻"}, out)
        data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
        assert data[0]["answer"] == "ответ 攻殻"


class TestEnsureOutputDir:
    def test_creates_missing_dir(self, tmp_path):
        target = tmp_path / "deep" / "output" / "results.json"
        io_utils.ensure_output_dir(str(target))
        assert target.parent.is_dir()

    def test_probe_cleaned_up(self, tmp_path):
        target = tmp_path / "results.json"
        io_utils.ensure_output_dir(str(target))
        assert list(tmp_path.iterdir()) == []

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_OUTPUT_PATH", str(tmp_path / "o" / "results.json"))
        io_utils.ensure_output_dir()
        assert (tmp_path / "o").is_dir()
