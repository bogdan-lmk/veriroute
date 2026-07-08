"""Integration: results.json must survive any way the process dies.

Runs `python -m agent.main` as a real subprocess with path overrides and
checks the guarantees the grader cares about: valid results.json keyed by
the real task_ids, exit code 0.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TASKS = [
    {"task_id": "t1", "prompt": "What is the capital of Australia?"},
    {"task_id": "t2", "prompt": "2+2?"},
    {"task_id": "t3", "prompt": "Summarise: hello world."},
]


def _env(tmp_path, **extra):
    env = os.environ.copy()
    env.update({
        "AGENT_INPUT_PATH": str(tmp_path / "tasks.json"),
        "AGENT_OUTPUT_PATH": str(tmp_path / "out" / "results.json"),
        "PYTHONPATH": str(REPO_ROOT),
    })
    # Never inherit real credentials into tests.
    for key in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"):
        env.pop(key, None)
    env.update(extra)
    return env


def _write_tasks(tmp_path, tasks=TASKS):
    (tmp_path / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")


def _read_results(tmp_path):
    return json.loads((tmp_path / "out" / "results.json").read_text())


def _run(tmp_path, env):
    return subprocess.run(
        [sys.executable, "-m", "agent.main"],
        env=env, cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=60,
    )


class TestCleanRuns:
    def test_no_fireworks_env_stubs_and_exit_zero(self, tmp_path):
        _write_tasks(tmp_path)
        proc = _run(tmp_path, _env(tmp_path))
        assert proc.returncode == 0, proc.stderr
        results = _read_results(tmp_path)
        assert {r["task_id"] for r in results} == {"t1", "t2", "t3"}
        assert all(isinstance(r["answer"], str) for r in results)

    def test_malformed_input_still_valid_output(self, tmp_path):
        (tmp_path / "tasks.json").write_text("{broken", encoding="utf-8")
        proc = _run(tmp_path, _env(tmp_path))
        assert proc.returncode == 0, proc.stderr
        assert _read_results(tmp_path) == []

    def test_missing_input_still_valid_output(self, tmp_path):
        proc = _run(tmp_path, _env(tmp_path))
        assert proc.returncode == 0, proc.stderr
        assert _read_results(tmp_path) == []

    def test_unroutable_proxy_finishes_with_stubs(self, tmp_path):
        """Black-hole proxy: transport fails fast, all tasks stubbed, exit 0."""
        _write_tasks(tmp_path)
        env = _env(
            tmp_path,
            FIREWORKS_BASE_URL="http://127.0.0.1:1",  # connection refused
            FIREWORKS_API_KEY="fake",
            ALLOWED_MODELS="accounts/fireworks/models/fake-model",
            AGENT_HTTP_TIMEOUT_S="1",
            AGENT_TOTAL_BUDGET_S="30",
        )
        proc = _run(tmp_path, env)
        assert proc.returncode == 0, proc.stderr
        results = _read_results(tmp_path)
        assert {r["task_id"] for r in results} == {"t1", "t2", "t3"}


class TestKillResilience:
    def _spawn(self, tmp_path, env):
        return subprocess.Popen(
            [sys.executable, "-m", "agent.main"],
            env=env, cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def _wait_for_results(self, tmp_path, timeout=10.0):
        target = tmp_path / "out" / "results.json"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if target.exists():
                return
            time.sleep(0.05)
        raise AssertionError("initial stub results.json never appeared")

    def _slow_env(self, tmp_path):
        # Unroutable IP + long timeout keeps the process busy mid-run.
        return _env(
            tmp_path,
            FIREWORKS_BASE_URL="http://10.255.255.1:9",
            FIREWORKS_API_KEY="fake",
            ALLOWED_MODELS="accounts/fireworks/models/fake-model",
            AGENT_HTTP_TIMEOUT_S="30",
            AGENT_TOTAL_BUDGET_S="300",
        )

    def test_sigkill_mid_run_leaves_valid_results(self, tmp_path):
        _write_tasks(tmp_path)
        proc = self._spawn(tmp_path, self._slow_env(tmp_path))
        try:
            self._wait_for_results(tmp_path)
            time.sleep(0.3)
            proc.send_signal(signal.SIGKILL)
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
        results = _read_results(tmp_path)
        assert {r["task_id"] for r in results} == {"t1", "t2", "t3"}

    def test_sigterm_flushes_and_exits_zero(self, tmp_path):
        _write_tasks(tmp_path)
        proc = self._spawn(tmp_path, self._slow_env(tmp_path))
        try:
            self._wait_for_results(tmp_path)
            time.sleep(0.3)
            proc.send_signal(signal.SIGTERM)
            code = proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert code == 0
        results = _read_results(tmp_path)
        assert {r["task_id"] for r in results} == {"t1", "t2", "t3"}
