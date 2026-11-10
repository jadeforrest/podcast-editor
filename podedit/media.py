"""ffprobe / ffmpeg helpers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Probe:
    path: Path
    duration: float
    has_audio: bool
    has_video: bool
    width: int = 0
    height: int = 0
    fps: Fraction | None = None


def parse_rate(rate: str | None) -> Fraction | None:
    """ffprobe frame rates look like '30000/1001'; '0/0' means unknown."""
    if not rate:
        return None
    try:
        r = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return None
    return r if 0 < r <= 240 else None


def parse_probe(path: Path, data: dict) -> Probe:
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    if video is None:
        return Probe(path, duration, audio is not None, False)
    fps = parse_rate(video.get("avg_frame_rate")) or parse_rate(video.get("r_frame_rate"))
    return Probe(path, duration, audio is not None, True,
                 int(video.get("width", 0)), int(video.get("height", 0)), fps)


def probe(path: Path) -> Probe:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return parse_probe(path, json.loads(out))


def decode_audio(path: Path, rate: int = 16000) -> np.ndarray:
    """Decode a file's audio to mono float32 at `rate`."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(rate), "-f", "f32le", "pipe:1"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()
