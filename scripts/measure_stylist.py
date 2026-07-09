"""Measure real stylist-call latency inside the duo image.

Run: docker run --rm --memory=4g --cpus=2 \
       -v $PWD/scripts/measure_stylist.py:/m.py --entrypoint python3 IMAGE /m.py
Prints per-call wall time and usage for: prewarm, first styled call, second
styled call (same prefix, different desc) — isolating prompt-cache behavior.
"""
import json
import time
import urllib.request

from agent.captioner.stylist import PAIRS
from agent.local_llm import LocalLLM

DESCS = [
    "A dog runs across a sunny park chasing a ball thrown by its owner.",
    "A chef plates creamy pasta in a steel pan inside a restaurant kitchen.",
]


def timed_call(local, prompt, max_tokens):
    t0 = time.time()
    body = {"model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.7,
            "cache_prompt": True}
    req = urllib.request.Request(
        f"http://127.0.0.1:{local.port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        payload = json.loads(r.read())
    dt = time.time() - t0
    usage = payload.get("usage", {})
    content = payload["choices"][0]["message"]["content"]
    return dt, usage, content


def main():
    local = LocalLLM(model_path="/app/gemma.gguf", port=8092,
                     ctx=2048, system_prompt="")
    assert local.start(wait_s=90), "gemma failed to start"
    template, _ = PAIRS["pair1"]
    prefix = template.split("{desc}")[0]

    t0 = time.time()
    local.prewarm(prefixes=[prefix])
    print(f"prewarm: {time.time()-t0:.1f}s")

    for i, desc in enumerate(DESCS, 1):
        dt, usage, content = timed_call(local, template.replace("{desc}", desc), 160)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        print(f"call{i}: {dt:.1f}s prompt={usage.get('prompt_tokens')} "
              f"cached={cached} completion={usage.get('completion_tokens')}")
        print(f"  out: {content[:110]!r}")
    local.stop()


if __name__ == "__main__":
    main()
