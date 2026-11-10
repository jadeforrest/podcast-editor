"""Even out the people: one gain per sound source so everyone's speech sits at the same level.

Each person's level is measured only while *they* are talking (their own transcript
words, so bleed and the other person's turns don't count), as the energy average of
the louder frames inside those words (pauses between syllables are gated out). The
gain brings that level to TARGET_DB. Clipping is prevented separately, by a lookahead
limiter on the mix (render.LIMITER): a gain never has to be lowered for a rare laugh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import FRAME, frame_db
from .media import decode_audio

RATE = 16000
TARGET_DB = -20.0     # speech RMS, dBFS: about -19 LUFS, leaving headroom for normal speech peaks
MAX_BOOST = 24.0      # dB; a near-silent mic is left quieter rather than turned into a noise source
MAX_CUT = 24.0
GATE = 20.0           # dB below the loud end of a person's speech that still counts as speech
MIN_FRAMES = 50       # at least 0.5 s of speech to trust a measurement
FLOOR_DB = -70.0      # quieter than this inside "speech" means the words aren't in this signal


def speech_frames(db: np.ndarray, words: list[dict]) -> np.ndarray:
    """Frame levels inside the given words."""
    mask = np.zeros(len(db), dtype=bool)
    for w in words:
        mask[int(w["start"] / FRAME):int(np.ceil(w["end"] / FRAME))] = True
    return db[mask]


def speech_level(db: np.ndarray, words: list[dict]) -> float | None:
    """Energy-average dBFS of a person's speech, or None if there is too little to measure."""
    vals = speech_frames(db, words)
    if vals.size < MIN_FRAMES:
        return None
    vals = vals[vals > np.percentile(vals, 95) - GATE]
    level = float(10 * np.log10(np.mean(10 ** (vals.astype(np.float64) / 10))))
    return level if level > FLOOR_DB else None


def gain_for(level: float | None, target: float = TARGET_DB) -> float:
    if level is None:
        return 0.0
    return float(np.clip(target - level, -MAX_CUT, MAX_BOOST))


def measure(ep_root: Path, tracks: list[dict], words: list[dict]) -> dict[str, float | None]:
    """Speech level of each sound source, keyed by person."""
    out = {}
    for t in tracks:
        if t.get("audio_source", t.get("has_audio")):
            # Stereo, as it is mixed (ffmpeg upmixes mono 3 dB down); frame_db over the interleaved
            # samples averages the power of both channels.
            audio = decode_audio(ep_root / t["file"], RATE, t.get("audio_offset", 0.0), channels=2)
            out[t["name"]] = speech_level(frame_db(audio, 2 * RATE), [w for w in words if w["speaker"] == t["name"]])
    return out


def describe(levels: dict[str, float | None], gains: dict[str, float]) -> str:
    parts = [f"{name} {'unmeasured' if lv is None else f'{lv:.1f} dB'} -> {gains[name]:+.1f} dB"
             for name, lv in levels.items()]
    return "Audio levels: " + ", ".join(parts)


def track_gains(ep_root: Path, transcript: dict) -> dict[str, float]:
    levels = measure(ep_root, transcript["tracks"], transcript["words"])
    gains = {name: gain_for(lv) for name, lv in levels.items()}
    print(describe(levels, gains))
    return gains
