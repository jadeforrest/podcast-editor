"""Active-speaker camera switching.

Who is talking comes from the transcript: every word knows its speaker, and a
speaker's camera is the video track with the same name. Shots are planned on
the *edited* timeline (output frames), so cuts never leave a too-short shot.

Rules:
- Switch to a speaker when they start a turn of at least `min_turn` seconds;
  short interjections ("yeah", "mm-hmm") don't move the camera.
- A shot lasts at least `min_shot` seconds. A switch that would come sooner is
  delayed, and dropped if the new speaker has (nearly) finished by then.
- In silence, or while someone without a camera talks, the shot stays put.
"""

from __future__ import annotations

from fractions import Fraction

Span = tuple[int, int]  # [start, end) output frames

MIN_SHOT = 2.0   # seconds
MIN_TURN = 1.0   # seconds
MERGE_GAP = 0.5  # seconds; pauses shorter than this don't end a turn


def merge(spans: list[Span], gap: int) -> list[Span]:
    out: list[Span] = []
    for a, b in sorted(spans):
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def to_output_frames(intervals: list[tuple[float, float]], kept: list[Span], fps: Fraction) -> list[Span]:
    """Source-time intervals -> spans on the edited timeline (parts that were cut vanish)."""
    out: list[Span] = []
    offset = 0
    for f0, f1 in kept:
        for s, e in intervals:
            a, b = max(round(s * fps), f0), min(round(e * fps), f1)
            if b > a:
                out.append((a - f0 + offset, b - f0 + offset))
        offset += f1 - f0
    return out


def turns(words: list[dict], cameras: list[str], kept: list[Span], fps: Fraction) -> list[tuple[int, int, int]]:
    """(start, end, camera index) for each stretch of speech by someone with a camera, by start."""
    gap = round(MERGE_GAP * fps)
    out = []
    for cam, name in enumerate(cameras):
        spoken = [(w["start"], w["end"]) for w in words if w["speaker"] == name]
        out += [(a, b, cam) for a, b in merge(to_output_frames(spoken, kept, fps), gap)]
    return sorted(out)


def choose_shots(segments: list[tuple[int, int, int]], min_shot: int, min_turn: int) -> list[tuple[int, int]]:
    """(start frame, camera index) for each shot; the first starts at frame 0."""
    long_turns = [s for s in segments if s[1] - s[0] >= min_turn]
    shots = [(0, long_turns[0][2] if long_turns else 0)]
    for a, b, cam in long_turns:
        start, current = shots[-1]
        if cam == current:
            continue
        t = max(a, start + min_shot)
        if b - t >= min_turn // 2:
            shots.append((t, cam))
    return shots


def plan_shots(words: list[dict], cameras: list[str], kept: list[Span], fps: Fraction,
               min_shot: float = MIN_SHOT) -> list[tuple[int, int]]:
    segments = turns(words, cameras, kept, fps)
    return choose_shots(segments, round(min_shot * fps), round(MIN_TURN * fps))


def crop_x_expr(shots: list[tuple[int, int]], width: int, total: int) -> str:
    """ffmpeg expression (of frame number n) for the x offset of the shown camera in a camera strip."""
    terms = []
    for k, (start, cam) in enumerate(shots):
        end = shots[k + 1][0] if k + 1 < len(shots) else total
        if cam and end > start:
            terms.append(f"{cam * width}*between(n,{start},{end - 1})")
    return "+".join(terms) or "0"
