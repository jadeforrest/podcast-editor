"""Edit decision list: deleted words -> time ranges to keep."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .doc import display_text, timestamp

Interval = tuple[float, float]


def runs_of(indexes: set[int]) -> list[tuple[int, int]]:
    """Consecutive runs (inclusive) of the given word indexes."""
    out: list[tuple[int, int]] = []
    for i in sorted(indexes):
        if out and out[-1][1] == i - 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def clean_left(i: int, cuts: list) -> bool:
    return i == 0 or cuts[i] is not None


def clean_right(j: int, cuts: list) -> bool:
    return j == len(cuts) - 2 or cuts[j + 1] is not None


def smallest_clean(i: int, j: int, cuts: list) -> tuple[int, int]:
    """Grow a run outwards until both of its edges are cuttable."""
    while not clean_left(i, cuts):
        i -= 1
    while not clean_right(j, cuts):
        j += 1
    return i, j


def phrase(words: list[dict], i: int, j: int) -> str:
    return " ".join(display_text(w) for w in words[i:j + 1])


@dataclass
class Problem:
    run: tuple[int, int]
    message: str


def describe_problem(words: list[dict], i: int, j: int, cuts: list) -> str:
    n = len(words)
    sides = [s for s, ok in (("before", clean_left(i, cuts)), ("after", clean_right(j, cuts))) if not ok]
    ctx_i, ctx_j = max(0, i - 3), min(n - 1, j + 3)
    si, sj = smallest_clean(i, j, cuts)
    return (f"[{timestamp(words[i]['start'])}] {words[i]['speaker']}: can't cut "
            f"{phrase(words, i, j)!r} cleanly ({' and '.join(sides)} it runs into the next word)\n"
            f"    context: ...{phrase(words, ctx_i, ctx_j)}...\n"
            f"    smallest clean cut containing it: {phrase(words, si, sj)!r}")


@dataclass
class Plan:
    runs: list[tuple[int, int]] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)


def plan_deletions(words: list[dict], cuts: list, deleted: set[int]) -> Plan:
    plan = Plan()
    for i, j in runs_of(deleted):
        if clean_left(i, cuts) and clean_right(j, cuts):
            plan.runs.append((i, j))
        else:
            plan.problems.append(Problem((i, j), describe_problem(words, i, j, cuts)))
    return plan


def run_removal(i: int, j: int, cuts: list, duration: float, max_gap: float | None) -> Interval:
    last = len(cuts) - 2
    a = 0.0 if i == 0 else cuts[i]["t"]
    b = duration if j == last else cuts[j + 1]["t"]
    if max_gap is not None:
        if i > 0:
            a = min(a, cuts[i]["qs"] + max_gap / 2)
        if j < last:
            b = max(b, cuts[j + 1]["qe"] - max_gap / 2)
    return a, b


def gap_removals(cuts: list, runs: list[tuple[int, int]], max_gap: float) -> list[Interval]:
    """Shorten every remaining pause longer than max_gap to max_gap."""
    touched = {k for i, j in runs for k in range(i, j + 2)}
    out = []
    for k, c in enumerate(cuts):
        if c is not None and k not in touched and c["qe"] - c["qs"] > max_gap:
            out.append((c["qs"] + max_gap / 2, c["qe"] - max_gap / 2))
    return out


def removals(runs: list[tuple[int, int]], cuts: list, duration: float, max_gap: float | None = None) -> list[Interval]:
    out = [run_removal(i, j, cuts, duration, max_gap) for i, j in runs]
    if max_gap is not None:
        out += gap_removals(cuts, runs, max_gap)
    return out


def keep_intervals(duration: float, remove: list[Interval]) -> list[Interval]:
    """The complement of `remove` within [0, duration]."""
    keeps, pos = [], 0.0
    clipped = ((min(max(0.0, a), duration), min(duration, b)) for a, b in remove)
    for a, b in sorted((a, b) for a, b in clipped if b > a):
        if a > pos:
            keeps.append((pos, a))
        pos = max(pos, b)
    if pos < duration:
        keeps.append((pos, duration))
    return keeps


def snap_to_frames(keeps: list[Interval], fps: Fraction) -> list[tuple[int, int]]:
    """Keep ranges as [first, end) frame numbers, so audio and video cut at identical times."""
    out: list[tuple[int, int]] = []
    for a, b in keeps:
        f0, f1 = round(a * fps), round(b * fps)
        if f1 <= f0:
            continue
        if out and out[-1][1] >= f0:
            out[-1] = (out[-1][0], max(out[-1][1], f1))
        else:
            out.append((f0, f1))
    return out
