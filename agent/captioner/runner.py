"""Track 2 orchestration: two-pass pipeline under 4 GB RAM.

Pass 1: SmolVLM (0.6 GB) describes every clip (downloads prefetched in
background threads). Pass 2: SmolVLM stops, Gemma 3 4B (2.4 GB) styles every
description. Same guardrails as Track 1: stub-first output, atomic rewrites,
deadlines, exit 0 always.
"""
from __future__ import annotations


import concurrent.futures
import json
import logging
import os
import tempfile

from .. import io_utils
from ..deadline import Deadline
from ..local_llm import LocalLLM
from . import frames as frames_mod
from .eye import Eye
from .stylist import STYLES, fallback_caption

log = logging.getLogger("agent.captioner")

STUB = "A short video clip."


def _fireworks_creds() -> tuple[str | None, str]:
    """No secrets ship in the public image. Dev runs use env creds; graded
    runs use our escalation relay (URL only — the key lives on our box and
    dies with it). Relay unreachable -> the eye's local fallback covers it."""
    key = os.environ.get("FIREWORKS_API_KEY")
    if key:
        url = os.environ.get("FIREWORKS_BASE_URL",
                             "https://api.fireworks.ai/inference/v1")
        return key, url
    relay = os.environ.get("ESCALATION_RELAY_URL")
    if relay:
        return "relay", relay  # non-secret placeholder token
    return None, ""


def _write(answers: dict[str, dict[str, str]]) -> None:
    payload = [{"task_id": tid, "captions": caps} for tid, caps in answers.items()]
    dst = io_utils.output_path()
    out_dir = os.path.dirname(dst) or "."
    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=".cap-", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def run(tasks: list[dict]) -> int:
    # T2 has no external flush drain to pay for — use more of the 600s cap.
    deadline = Deadline(total_s=560.0, flush_margin_s=30.0, per_task_s=60.0)
    answers = {
        t["task_id"]: {s: STUB for s in t.get("styles") or list(STYLES)}
        for t in tasks
    }
    _write(answers)
    if not tasks:
        return 0

    # Prefetch clips in parallel — network overlaps model loading.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    clip_futures = {
        t["task_id"]: pool.submit(frames_mod.fetch_clip, t["task_id"],
                                  t.get("video_url", ""))
        for t in tasks
    }

    fw_key, fw_url = _fireworks_creds()

    # --- Pass 1: describe everything with the small eye ---
    smol = LocalLLM(
        model_path=os.environ.get("AGENT_SMOL_MODEL", "/app/smolvlm.gguf"),
        mmproj_path=os.environ.get("AGENT_SMOL_MMPROJ", "/app/smolvlm-mmproj.gguf"),
        port=8091, ctx=4096, system_prompt="",
    )
    eye_ok = smol.available and smol.start()
    eye = Eye(smol, fw_key, fw_url)
    descriptions: dict[str, str] = {}
    for task in tasks:
        tid = task["task_id"]
        if deadline.flush_due():
            log.warning("deadline flush during describe pass")
            break
        try:
            clip = clip_futures[tid].result(timeout=max(5.0, deadline.task_budget()))
            if clip is None:
                continue
            frame_list = frames_mod.extract_frames(clip, count=2)
            if not frame_list:
                continue
            if eye_ok:
                desc, source = eye.describe(frame_list, deadline.task_budget())
            else:
                desc, source = eye._escalate(frame_list[:2]), "escalated"  # noqa: SLF001
            if desc:
                descriptions[tid] = desc
                log.info("described %s via %s", tid, source)
        except Exception as exc:  # noqa: BLE001
            log.warning("describe failed for %s: %s", tid, exc)
    smol.stop()
    pool.shutdown(wait=False, cancel_futures=True)

    # --- Pass 2: style everything with Gemma ---
    gemma = LocalLLM(
        model_path=os.environ.get("AGENT_GEMMA_MODEL", "/app/gemma.gguf"),
        port=8092, ctx=2048, system_prompt="",
    )
    gemma_ok = gemma.available and gemma.start()
    from .stylist import prompt_prefix, style_all
    if gemma_ok:
        gemma.prewarm(prefixes=[prompt_prefix()])
    for task in tasks:
        tid = task["task_id"]
        desc = descriptions.get(tid)
        styles = task.get("styles") or list(STYLES)
        if not desc:
            continue  # stubs already in place
        if gemma_ok and not deadline.flush_due():
            caps = style_all(gemma, desc, styles,
                             min(50.0, deadline.task_budget()))
        else:
            caps = {s: fallback_caption(desc, s) for s in styles}
        answers[tid].update(caps)
        _write(answers)
    gemma.stop()
    log.info("captioner done: %d/%d described, %.1fs elapsed",
             len(descriptions), len(tasks), deadline.elapsed())
    return 0
