"""The edited transcript on the rendered video's clock (output/<episode>.transcript.md).

transcript.md keeps the original recording's times so it stays a stable anchor for
editing. For writing show notes or YouTube chapters you want the opposite: only the
words that survive, stamped with where they land in the finished video, intro included.
"""

from __future__ import annotations

from bisect import bisect_right
from itertools import accumulate
from fractions import Fraction

from . import doc

Span = tuple[int, int]  # [start, end) source frames that are kept


class Clock:
    """Maps source times to the edited timeline (before any intro)."""

    def __init__(self, kept: list[Span], fps: Fraction):
        self.kept, self.fps = kept, fps
        self.starts = [a for a, _ in kept]
        self.offsets = list(accumulate((b - a for a, b in kept), initial=0))

    def span(self, t: float) -> int | None:
        """Index of the kept span that plays source time `t`, or None if it was cut."""
        k = bisect_right(self.starts, t * self.fps) - 1
        return k if k >= 0 and t * self.fps < self.kept[k][1] else None

    def word(self, word: dict) -> float | None:
        """Output time of a word's start; a word counts as kept when its midpoint is (cut points sit
        in the quiet around words, so a kept word's start is at most a frame before its span)."""
        k = self.span((word["start"] + word["end"]) / 2)
        if k is None:
            return None
        f0 = self.kept[k][0]
        return float((self.offsets[k] + max(word["start"] * self.fps, f0) - f0) / self.fps)


def paragraphs(words: list[dict], kept: list[Span], fps: Fraction) -> list[tuple[str, float, list[str]]]:
    """(speaker, output time, words) for each original paragraph that still has words in it."""
    starts, clock = doc.paragraph_starts(words), Clock(kept, fps)
    out = []
    for a, b in zip(starts, starts[1:] + [len(words)]):
        timed = [(clock.word(w), w) for w in words[a:b]]
        shown = [(t, w) for t, w in timed if t is not None]
        if shown:
            out.append((shown[0][1]["speaker"], shown[0][0], [doc.display_text(w) for _, w in shown]))
    return out


def render(transcript: dict, kept: list[Span], fps: Fraction, title: str, intro: float = 0.0) -> str:
    note = (f"Times are positions in the rendered video, including a {intro:.1f}s intro."
            if intro else "Times are positions in the rendered video (no intro).")
    lines = [f"# {title}", "", note, ""]
    for speaker, t, text in paragraphs(transcript["words"], kept, fps):
        lines += [f"**{speaker}** [{doc.timestamp(t + intro)}]", " ".join(text), ""]
    return "\n".join(lines)
