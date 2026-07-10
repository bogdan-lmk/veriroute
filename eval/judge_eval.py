"""Judge container answers against expected intent, per category.

Usage: python3 eval/judge_eval.py <results.json path>
Prints per-category accuracy, the failing tasks with reasons, and the
projected gate verdict (>= 84.2%).
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gen_eval import FW_URL, fw_key  # noqa: E402

ROOT = pathlib.Path(__file__).parent
JUDGE = "accounts/fireworks/models/glm-5p2"

JUDGE_PROMPT = """Task given to an AI agent:
{prompt}

Expected correct answer / key facts:
{expected}

Agent's answer:
{answer}

Judge against expected intent: does the agent's answer correctly address the
task? Minor wording differences are fine; missing key facts, wrong values,
empty or truncated answers are failures.
Reply with JSON only: {{"correct": true/false, "reason": "<one short sentence>"}}"""


def judge_one(key: str, prompt: str, expected: str, answer: str) -> tuple[bool, str]:
    body = {"model": JUDGE,
            "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
                prompt=prompt, expected=expected, answer=answer or "(empty)")}],
            "max_tokens": 1600, "temperature": 0}
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                FW_URL, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                msg = json.loads(r.read())["choices"][0]["message"]
            raw = (msg.get("content") or msg.get("reasoning_content") or "")
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
            for m in re.finditer(r"\{[^{}]*\}", raw, re.S):
                try:
                    obj = json.loads(m.group(0))
                    if "correct" in obj:
                        return bool(obj["correct"]), str(obj.get("reason", ""))[:120]
                except json.JSONDecodeError:
                    continue
        except Exception:  # noqa: BLE001
            time.sleep(2)
    return False, "judge failed"


def main() -> None:
    results_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results.json")
    answers = {r["task_id"]: r.get("answer", "")
               for r in json.load(open(results_path))}
    tasks = [json.loads(l) for l in open(ROOT / "hidden_like.jsonl")]
    key = fw_key()

    per_cat: dict[str, list[bool]] = {}
    failures = []
    for t in tasks:
        ans = answers.get(t["task_id"], "")
        ok, reason = judge_one(key, t["prompt"], t["expected"], ans)
        per_cat.setdefault(t["category"], []).append(ok)
        mark = "+" if ok else "-"
        print(f"[{mark}] {t['task_id']}: {reason}", flush=True)
        if not ok:
            failures.append((t["task_id"], reason, (ans or "(empty)")[:100]))

    total = sum(sum(v) for v in per_cat.values())
    n = sum(len(v) for v in per_cat.values())
    print("\n=== PER CATEGORY ===")
    for cat, marks in sorted(per_cat.items()):
        print(f"{cat:15} {sum(marks)}/{len(marks)}")
    print(f"\nTOTAL: {total}/{n} = {total/n*100:.1f}%  "
          f"(gate 84.2% -> {'PASS' if total/n >= 0.842 else 'FAIL'})")
    if failures:
        print("\n=== FAILURES ===")
        for tid, reason, ans in failures:
            print(f"{tid}: {reason}\n   answer: {ans}")


if __name__ == "__main__":
    main()
