"""Input validation and crash-safe output for the grading I/O contract.

The grader mounts /input/tasks.json and expects /output/results.json with
exit code 0. Every failure mode here maps to a scored failure status
(OUTPUT_MISSING, INVALID_RESULTS_SCHEMA), so this module is deliberately
paranoid: probe writability first, validate every task entry, and only
ever replace results.json atomically.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile

log = logging.getLogger("agent.io")

DEFAULT_INPUT_PATH = "/input/tasks.json"
DEFAULT_OUTPUT_PATH = "/output/results.json"


def input_path() -> str:
    return os.environ.get("AGENT_INPUT_PATH", DEFAULT_INPUT_PATH)


def output_path() -> str:
    return os.environ.get("AGENT_OUTPUT_PATH", DEFAULT_OUTPUT_PATH)


def ensure_output_dir(path: str | None = None) -> None:
    """Create the output dir and prove it is writable before anything else runs."""
    out_dir = os.path.dirname(path or output_path()) or "."
    os.makedirs(out_dir, exist_ok=True)
    probe = os.path.join(out_dir, ".write-probe")
    with open(probe, "w", encoding="utf-8") as f:
        f.write("ok")
    os.remove(probe)


def load_tasks(path: str | None = None) -> list[dict]:
    """Parse tasks.json defensively.

    Returns only well-formed tasks as {"task_id": str, "prompt": str}.
    First occurrence wins on duplicate ids; entries without a usable
    task_id are dropped (nothing to key an answer on); a non-string
    prompt degrades to "" so the task still gets a stub answer.
    """
    src = path or input_path()
    try:
        with open(src, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.error("cannot read %s: %s", src, exc)
        return []
    if not isinstance(raw, list):
        log.error("tasks.json root is %s, expected list", type(raw).__name__)
        return []

    tasks: list[dict] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            log.warning("entry %d is not an object, skipping", i)
            continue
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            log.warning("entry %d has no usable task_id, skipping", i)
            continue
        if task_id in seen:
            log.warning("duplicate task_id %r, keeping first", task_id)
            continue
        prompt = entry.get("prompt")
        if not isinstance(prompt, str):
            log.warning("task %r has non-string prompt, stubbing", task_id)
            prompt = ""
        seen.add(task_id)
        tasks.append({"task_id": task_id, "prompt": prompt})
    return tasks


def write_results_atomic(answers: dict[str, str], path: str | None = None) -> None:
    """Atomically replace results.json.

    Tempfile lives in the same directory so os.replace stays a true
    same-filesystem rename; fsync before rename so a container kill right
    after the rename still leaves complete bytes on disk.
    """
    dst = path or output_path()
    payload = [{"task_id": tid, "answer": ans} for tid, ans in answers.items()]
    out_dir = os.path.dirname(dst) or "."
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".results-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
