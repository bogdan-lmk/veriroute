"""Generate a hidden-set-like eval: 8 categories x 4 tasks with expected answers.

Style mirrors the official practice tasks. The writer produces task+expected
pairs; we commit the result so the eval is deterministic and reviewable.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).parent
FW_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
WRITER = "accounts/fireworks/models/gpt-oss-120b"

SPECS = {
    "factual": "short factual-knowledge questions (capitals+geography, inventors, chemistry basics, history dates), possibly two-part",
    "math": "grade-school word problems mixing percentages and absolute amounts over 2-3 steps (stores selling stock, tanks draining, savings)",
    "sentiment": "classify the sentiment of a short product/service review with mixed or clear polarity; answer positive/negative/mixed with brief justification",
    "summarization": "summarize a given 3-5 sentence passage in exactly one sentence",
    "ner": "extract all named entities and their types from one sentence containing 2-4 entities (person, org, location, date)",
    "code_debug": "a short buggy Python function (off-by-one, wrong operator, wrong index) with 'find and fix it'",
    "logic": "constraint puzzles with 3 people/objects and 2-3 clues, asking who owns/does what",
    "code_gen": "write a small Python function with a subtle requirement (duplicates, empty input, ordering)",
}

PROMPT = """Write 4 evaluation tasks of this kind: {spec}.

Format: JSON array of 4 objects, each {{"prompt": "...", "expected": "..."}}.
- prompt: the task exactly as given to an AI agent (self-contained).
- expected: the correct answer / key facts a judge should check for.
Vary difficulty and surface wording. Reply with the JSON array only."""


def fw_key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if key:
        return key
    return subprocess.run(
        ["/Users/bogdan/.local/bin/fireconnect", "--home", "/Users/bogdan",
         "key", "export", "--stored-only"],
        capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    key = fw_key()
    tasks = []
    for cat, spec in SPECS.items():
        body = {"model": WRITER,
                "messages": [{"role": "user",
                              "content": PROMPT.format(spec=spec)}],
                "max_tokens": 1800, "temperature": 0.8,
                "reasoning_effort": "low"}
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    FW_URL, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    raw = json.loads(r.read())["choices"][0]["message"]["content"]
                m = re.search(r"\[.*\]", raw, re.S)
                items = json.loads(m.group(0))
                assert len(items) >= 3
                break
            except Exception as exc:  # noqa: BLE001
                print(f"{cat}: retry {attempt+1} ({exc})")
                time.sleep(2)
                items = []
        for i, item in enumerate(items[:4]):
            tasks.append({"task_id": f"{cat}-{i+1}", "category": cat,
                          "prompt": item["prompt"],
                          "expected": item["expected"]})
        print(f"{cat}: {len(items[:4])} tasks")

    with open(ROOT / "hidden_like.jsonl", "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    # grader-format tasks.json for running the real container
    with open(ROOT / "hidden_like_tasks.json", "w", encoding="utf-8") as f:
        json.dump([{"task_id": t["task_id"], "prompt": t["prompt"]}
                   for t in tasks], f, ensure_ascii=False, indent=1)
    print(f"TOTAL: {len(tasks)} tasks")


if __name__ == "__main__":
    main()
