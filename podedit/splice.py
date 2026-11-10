"""Stream audio through, keeping only the chosen sample ranges.

Each splice gets a short fade out/in so cuts don't click. Lengths are exact:
the output has precisely sum(end - start) samples, matching the video.
"""

from __future__ import annotations

from typing import BinaryIO, Iterator

import numpy as np

Ranges = list[tuple[int, int]]


def envelope(idx: np.ndarray, start: int, end: int, fade: int, fade_in: bool, fade_out: bool) -> np.ndarray:
    gain = np.ones(len(idx), dtype=np.float32)
    if fade > 0 and fade_in:
        gain = np.minimum(gain, (idx - start + 0.5) / fade)
    if fade > 0 and fade_out:
        gain = np.minimum(gain, (end - idx - 0.5) / fade)
    return np.clip(gain, 0.0, 1.0).astype(np.float32)


def splice_chunk(chunk: np.ndarray, offset: int, ranges: Ranges, total: int, fade: int) -> Iterator[np.ndarray]:
    """Kept parts of `chunk` (frames x channels) whose first frame is sample `offset`."""
    stop = offset + len(chunk)
    for start, end in ranges:
        a, b = max(start, offset), min(end, stop)
        if a >= b:
            continue
        idx = np.arange(a, b)
        gain = envelope(idx, start, end, fade, fade_in=start > 0, fade_out=end < total)
        yield chunk[a - offset:b - offset] * gain[:, None]


def splice_array(audio: np.ndarray, ranges: Ranges, fade: int) -> np.ndarray:
    """Reference (non-streaming) version, used by tests."""
    parts = list(splice_chunk(audio, 0, ranges, len(audio), fade))
    return np.concatenate(parts) if parts else audio[:0]


def splice_stream(src: BinaryIO, dst: BinaryIO, ranges: Ranges, total: int, channels: int, fade: int,
                  chunk_frames: int = 1 << 16, on_progress=None) -> int:
    """Copy float32 interleaved audio from src to dst keeping `ranges`. Returns frames written.

    Ranges must be sorted and non-overlapping. Output is padded with silence if the
    source ends early, so its length always matches the planned video length.
    """
    offset = written = 0
    pending = [r for r in ranges if r[1] > r[0]]
    frame_bytes = 4 * channels
    while pending:
        raw = src.read(chunk_frames * frame_bytes)
        if not raw:
            break
        chunk = np.frombuffer(raw[: len(raw) // frame_bytes * frame_bytes], dtype=np.float32).reshape(-1, channels)
        for part in splice_chunk(chunk, offset, pending, total, fade):
            dst.write(np.clip(part, -1.0, 1.0).astype(np.float32).tobytes())
            written += len(part)
        offset += len(chunk)
        pending = [r for r in pending if r[1] > offset]
        if on_progress:
            on_progress(len(chunk))
    expected = sum(b - a for a, b in ranges if b > a)
    if written < expected:
        dst.write(np.zeros((expected - written, channels), dtype=np.float32).tobytes())
    return expected
