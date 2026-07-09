"""Sandboxed execution of model-generated Python.

The code comes from our own local model (not from task input), so the threat
model is accidents, not attacks: infinite loops, huge output, stray writes.
Isolation: `python -I` (no site, no user paths), empty environment, temp cwd,
hard wall-clock timeout, output size cap.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)
_MAX_OUTPUT = 4000


def extract_code(text: str) -> str:
    """Pull the first fenced code block, or take the raw text as code."""
    m = _CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def run_python(code: str, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Execute code, return (ok, stdout). ok=False on any failure."""
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/snippet.py"
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                capture_output=True, text=True,
                timeout=timeout_s, cwd=tmp, env={},
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except OSError as exc:
            return False, f"exec error: {exc}"
    if proc.returncode != 0:
        return False, (proc.stderr or "")[:_MAX_OUTPUT]
    return True, proc.stdout[:_MAX_OUTPUT].strip()
