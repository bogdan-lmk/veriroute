"""Clip download and frame extraction for the captioning pipeline."""
from __future__ import annotations

import base64
import logging
import pathlib
import subprocess
import tempfile

log = logging.getLogger("agent.captioner.frames")

WORKDIR = pathlib.Path(tempfile.gettempdir()) / "clips"


def fetch_clip(task_id: str, url: str, timeout_s: float = 60.0) -> pathlib.Path | None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    dst = WORKDIR / f"{task_id}.mp4"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    r = subprocess.run(
        ["curl", "-fsSL", "-A", "Mozilla/5.0", "--max-time", str(int(timeout_s)),
         "-o", str(dst), url],
        capture_output=True,
    )
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        log.error("download failed for %s: %s", task_id, r.stderr[:120])
        return None
    return dst


def extract_frames(clip: pathlib.Path, count: int = 3) -> list[str]:
    """Uniformly sampled frames as base64 JPEG strings (896px wide)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = 30.0
    frames: list[str] = []
    for i in range(count):
        ts = duration * (i + 1) / (count + 1)
        out = clip.with_suffix(f".f{i}.jpg")
        r = subprocess.run(
            ["ffmpeg", "-y", "-v", "quiet", "-ss", f"{ts:.2f}", "-i", str(clip),
             "-vf", "scale=896:-1", "-frames:v", "1", str(out)],
            capture_output=True,
        )
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            frames.append(base64.b64encode(out.read_bytes()).decode())
            out.unlink(missing_ok=True)
    return frames
