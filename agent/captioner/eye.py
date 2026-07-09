"""The eye: SmolVLM2-500M frame descriptions with a consistency check,
escalating inconsistent clips to a Fireworks VLM when credentials exist.

Prove-or-escalate, same philosophy as the Track 1 router: a local answer
ships only when independent frame descriptions agree on the subject.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request

from ..local_llm import LocalLLM

log = logging.getLogger("agent.captioner.eye")

DESCRIBE = ("<__media__>\nDescribe this video frame in 2-3 factual sentences: "
            "scene, main subjects, actions. Reply with the description only.")

MERGE_PROMPT = (
    "These are descriptions of frames from ONE video, in order:\n{descs}\n\n"
    "Write 2-3 sentences describing the video, keeping ONLY details that the "
    "frame descriptions agree on. Reply with the description only."
)

_STOPWORDS = frozenset(
    "the a an of in on at with and or is are was were this that video frame "
    "shows depicts captures image scene main subject".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) > 3 and w not in _STOPWORDS}


def frames_agree(descs: list[str]) -> bool:
    """Cheap consistency signal: adjacent descriptions must share vocabulary.

    A hallucinating small VLM drifts between frames ('helmet' -> 'robot' ->
    'toy'); real scenes keep a stable content-word core.
    """
    if len(descs) < 2:
        return True
    sets = [_content_words(d) for d in descs]
    for a, b in zip(sets, sets[1:]):
        if not a or not b:
            return False
        overlap = len(a & b) / min(len(a), len(b))
        if overlap < 0.2:
            return False
    return True


class Eye:
    def __init__(self, local: LocalLLM, fw_key: str | None, fw_url: str | None):
        self.local = local
        self.fw_key = fw_key
        self.fw_url = (fw_url or "").rstrip("/")

    def describe(self, frames_b64: list[str], budget_s: float) -> tuple[str | None, str]:
        """Returns (description, source). Local first, escalate on distrust."""
        descs = []
        per_frame = max(4.0, budget_s / (len(frames_b64) + 1))
        for b64 in frames_b64:
            try:
                d = self.local.chat(
                    [{"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                     {"type": "text", "text": DESCRIBE}],
                    max_tokens=90, timeout_s=per_frame,
                )
                if d.strip():
                    descs.append(d.strip())
            except Exception as exc:  # noqa: BLE001
                log.warning("local describe failed: %s", exc)
        if descs and frames_agree(descs):
            # Frames agree -> the most detailed description carries the most
            # usable facts; a merge call would cost ~8s of 2-vCPU time per clip.
            return max(descs, key=len), "local"
        log.info("frame descriptions disagree -> escalate eye")
        escalated = self._escalate(frames_b64[:2])
        if escalated:
            return escalated, "escalated"
        if descs:
            return descs[0], "local-unverified"
        return None, "failed"

    def _escalate(self, frames_b64: list[str]) -> str | None:
        if not (self.fw_key and self.fw_url):
            return None
        content = [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
            for b in frames_b64
        ] + [{"type": "text", "text":
              "Describe this video in 2-3 factual sentences: scene, main "
              "subjects, actions. Reply with the description only."}]
        body = {"model": "accounts/fireworks/models/minimax-m3",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 600, "temperature": 0}
        try:
            req = urllib.request.Request(
                f"{self.fw_url}/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.fw_key}"})
            with urllib.request.urlopen(req, timeout=45) as r:
                msg = json.loads(r.read())["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            # Thinking models sometimes lead with meta-analysis; keep the tail.
            if text and len(text.split("\n\n")[-1]) > 40:
                text = text.split("\n\n")[-1]
            return text or None
        except Exception as exc:  # noqa: BLE001
            log.warning("eye escalation failed: %s", exc)
            return None
