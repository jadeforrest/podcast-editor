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
    audio_offset: float = 0.0  # seconds the audio stream starts after the video stream (negative: before)


AUDIO_GRID = Fraction(1000)  # audio-only episodes cut to the millisecond (the picture is static)
STANDARD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001), Fraction(30),
                  Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60)]


def standard_rate(r: Fraction) -> Fraction:
    """Variable-frame-rate files report odd averages (e.g. 1567/52); use the nearest normal rate."""
    best = min(STANDARD_RATES, key=lambda s: abs(s - r))
    if abs(best - r) / r <= 0.02:
        return best
    return Fraction(max(1, round(r)))


def parse_rate(rate: str | None) -> Fraction | None:
    """ffprobe frame rates look like '30000/1001'; '0/0' means unknown."""
    if not rate:
        return None
    try:
        r = Fraction(rate)
    except (ValueError, ZeroDivisionError):
        return None
    return standard_rate(r) if 0 < r <= 240 else None


def start_time(stream: dict | None) -> float:
    try:
        return float(stream.get("start_time", 0) or 0) if stream else 0.0
    except ValueError:
        return 0.0


def parse_probe(path: Path, data: dict) -> Probe:
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    if video is None:
        return Probe(path, duration, audio is not None, False)
    fps = parse_rate(video.get("avg_frame_rate")) or parse_rate(video.get("r_frame_rate"))
    offset = start_time(audio) - start_time(video) if audio else 0.0
    return Probe(path, duration, audio is not None, True,
                 int(video.get("width", 0)), int(video.get("height", 0)), fps, round(offset, 4))


def probe(path: Path) -> Probe:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return parse_probe(path, json.loads(out))


def offset_filter(offset: float) -> str:
    """Filter prefix that puts a file's audio on its video's timeline (see Probe.audio_offset)."""
    if offset > 0.0005:
        return f"adelay={offset * 1000:.1f}:all=1,"
    if offset < -0.0005:
        return f"atrim=start={-offset:.4f},asetpts=PTS-STARTPTS,"
    return ""


def decode_audio(path: Path, rate: int = 16000, offset: float = 0.0, channels: int = 1) -> np.ndarray:
    """Decode a file's audio to float32 at `rate` (interleaved if `channels` > 1), aligned to its video."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-af", f"{offset_filter(offset)}anull",
           "-ac", str(channels), "-ar", str(rate), "-f", "f32le", "pipe:1"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()
