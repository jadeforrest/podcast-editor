"""Audio-energy analysis: where between words a clean cut is possible.

A boundary between two consecutive words (in time order, across all speakers)
is *clean* when the mixed audio has a quiet stretch of at least `min_quiet`
seconds somewhere between them. The cut point is the middle of that stretch,
so a deletion leaves roughly one natural pause behind. When people talk over
each other, or words run together, there is no quiet stretch and the boundary
is marked as not cuttable.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from .text import norm

FRAME = 0.01  # seconds per analysis frame


def frame_db(samples: np.ndarray, rate: int, frame: float = FRAME) -> np.ndarray:
    """RMS level in dBFS for each `frame`-second window."""
    hop = int(round(rate * frame))
    n = len(samples) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    x = samples[: n * hop].astype(np.float64).reshape(n, hop)
    rms = np.sqrt(np.mean(x * x, axis=1))
    return (20 * np.log10(np.maximum(rms, 1e-10))).astype(np.float32)


@dataclass(frozen=True)
class Levels:
    noise: float
    speech: float


def levels(db: np.ndarray) -> Levels:
    live = db[db > -100]
    if live.size == 0:
        return Levels(-100.0, -100.0)
    return Levels(float(np.percentile(live, 10)), float(np.percentile(live, 95)))


def quiet_threshold(lv: Levels) -> float:
    """dBFS level below which the mix counts as quiet enough to cut."""
    return min(max(lv.noise + 10, lv.speech - 28), lv.speech - 12)


def quiet_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) frame indexes of each True run."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return edges[0::2], edges[1::2]


def best_quiet(starts: np.ndarray, ends: np.ndarray, lo: int, hi: int, min_frames: int) -> tuple[int, int] | None:
    """Longest quiet run clipped to [lo, hi), if it is at least `min_frames` long."""
    first = int(np.searchsorted(ends, lo, side="right"))
    last = int(np.searchsorted(starts, hi, side="left"))
    best = None
    for k in range(first, last):
        s, e = max(int(starts[k]), lo), min(int(ends[k]), hi)
        if e - s >= min_frames and (best is None or e - s > best[1] - best[0]):
            best = (s, e)
    return best


def cut_dict(qs: float, qe: float) -> dict:
    return {"t": round((qs + qe) / 2, 3), "qs": round(qs, 3), "qe": round(qe, 3)}


def boundary_windows(words: list[dict], duration: float, slack: float) -> list[tuple[float, float]]:
    """Time window to search for a cut before word i (i == len(words): after the last)."""
    windows = []
    max_end = 0.0
    for i in range(len(words) + 1):
        lo = 0.0 if i == 0 else max(0.0, max_end - slack)
        hi = duration if i == len(words) else min(duration, words[i]["start"] + slack)
        windows.append((lo, hi))
        if i < len(words):
            max_end = max(max_end, words[i]["end"])
    return windows


def compute_cuts(words: list[dict], mix_db: np.ndarray, duration: float, threshold: float,
                 min_quiet: float = 0.04, slack: float = 0.1) -> list[dict | None]:
    """One entry per boundary: cuts[i] is before words[i]; cuts[len(words)] is after the last word."""
    starts, ends = quiet_runs(mix_db < threshold)
    min_frames = max(1, int(round(min_quiet / FRAME)))
    cuts: list[dict | None] = []
    windows = boundary_windows(words, duration, slack)
    for i, (lo, hi) in enumerate(windows):
        q = best_quiet(starts, ends, int(lo / FRAME), int(np.ceil(hi / FRAME)), min_frames)
        if q is not None:
            cuts.append(cut_dict(q[0] * FRAME, min(q[1] * FRAME, duration)))
        elif i == 0:
            cuts.append(cut_dict(0.0, 0.0))
        elif i == len(words):
            cuts.append(cut_dict(duration, duration))
        else:
            cuts.append(None)
    return cuts


def peak_db(db: np.ndarray, start: float, end: float, pad: float = 0.05) -> float:
    a = max(0, int((start - pad) / FRAME))
    b = min(len(db), int(np.ceil((end + pad) / FRAME)) + 1)
    return float(db[a:b].max()) if b > a else -100.0


def has_twin(word: dict, others: list[dict], other_starts: list[float], tol: float = 0.5) -> bool:
    """Another track has the same word at roughly the same time (i.e. mic bleed)."""
    lo = bisect.bisect_left(other_starts, word["start"] - tol)
    hi = bisect.bisect_right(other_starts, word["end"] + tol)
    target = norm(word["text"])
    return any(norm(o["text"]) == target for o in others[lo:hi])


def drop_reason(word: dict, track: str, dbs: dict[str, np.ndarray], lv: dict[str, Levels],
                words_by_track: dict[str, list[dict]], starts_by_track: dict[str, list[float]]) -> str | None:
    """Why a recognised word should not be in the transcript, or None to keep it.

    'silent': its own track is near-silent there (Whisper hallucination).
    'bleed': another mic picked it up louder and recognised the same word.
    """
    own = peak_db(dbs[track], word["start"], word["end"])
    if own < lv[track].speech - 35:
        return "silent"
    for other, odb in dbs.items():
        if other == track:
            continue
        if peak_db(odb, word["start"], word["end"]) > own + 10 and has_twin(
                word, words_by_track[other], starts_by_track[other]):
            return "bleed"
    return None


def filter_words(words_by_track: dict[str, list[dict]], dbs: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    """Merge all tracks into one time-ordered word list, dropping hallucinations and bleed."""
    lv = {t: levels(db) for t, db in dbs.items()}
    starts = {t: [w["start"] for w in ws] for t, ws in words_by_track.items()}
    kept, dropped = [], []
    for track, ws in words_by_track.items():
        for w in ws:
            reason = drop_reason(w, track, dbs, lv, words_by_track, starts)
            if reason:
                dropped.append({**w, "reason": reason})
            else:
                kept.append(w)
    order = lambda w: (w["start"], w["end"], w["speaker"])  # noqa: E731
    return sorted(kept, key=order), sorted(dropped, key=order)
