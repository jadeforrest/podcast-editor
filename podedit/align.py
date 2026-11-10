"""Exact word alignment (longest common subsequence) for edited transcripts.

difflib's greedy matching can pair a kept word with the wrong copy of a
repeated word, which here means keeping the wrong person's audio. Within a
paragraph-sized span we can afford an exact LCS instead.
"""

from __future__ import annotations

import difflib

Opcode = tuple[str, int, int, int, int]
MAX_CELLS = 4_000_000  # beyond this the DP gets slow; fall back to difflib


def lcs_pairs(a: list[str], b: list[str]) -> list[tuple[int, int]]:
    """Matched (i, j) index pairs of a longest common subsequence."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, below = dp[i], dp[i + 1]
        ai = a[i]
        for j in range(m - 1, -1, -1):
            row[j] = below[j + 1] + 1 if ai == b[j] else max(below[j], row[j + 1])
    pairs, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j] and dp[i][j] == dp[i + 1][j + 1] + 1:
            pairs.append((i, j))
            i, j = i + 1, j + 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def gap_opcode(i1: int, i2: int, j1: int, j2: int) -> Opcode | None:
    if i1 < i2 and j1 < j2:
        return ("replace", i1, i2, j1, j2)
    if i1 < i2:
        return ("delete", i1, i2, j1, j2)
    if j1 < j2:
        return ("insert", i1, i2, j1, j2)
    return None


def pairs_to_opcodes(pairs: list[tuple[int, int]], n: int, m: int) -> list[Opcode]:
    ops: list[Opcode] = []
    pi = pj = 0
    for i, j in pairs + [(n, m)]:
        gap = gap_opcode(pi, i, pj, j)
        if gap:
            ops.append(gap)
        if i < n:
            ops.append(("equal", i, i + 1, j, j + 1))
        pi, pj = i + 1, j + 1
    return ops


def common_affixes(a: list[str], b: list[str]) -> tuple[int, int]:
    pre = 0
    while pre < len(a) and pre < len(b) and a[pre] == b[pre]:
        pre += 1
    suf = 0
    while suf < len(a) - pre and suf < len(b) - pre and a[-1 - suf] == b[-1 - suf]:
        suf += 1
    return pre, suf


def shift(ops: list[Opcode], di: int, dj: int) -> list[Opcode]:
    return [(op, i1 + di, i2 + di, j1 + dj, j2 + dj) for op, i1, i2, j1, j2 in ops]


def opcodes(a: list[str], b: list[str]) -> tuple[list[Opcode], bool]:
    """difflib-style opcodes turning a into b, and whether the result is exact (LCS)."""
    pre, suf = common_affixes(a, b)
    mid_a, mid_b = a[pre:len(a) - suf], b[pre:len(b) - suf]
    exact = len(mid_a) * len(mid_b) <= MAX_CELLS
    if exact:
        mid = pairs_to_opcodes(lcs_pairs(mid_a, mid_b), len(mid_a), len(mid_b))
    else:
        mid = difflib.SequenceMatcher(None, mid_a, mid_b, autojunk=False).get_opcodes()
    ops = [("equal", 0, pre, 0, pre)] if pre else []
    ops += shift(mid, pre, pre)
    if suf:
        ops.append(("equal", len(a) - suf, len(a), len(b) - suf, len(b)))
    return ops, exact
