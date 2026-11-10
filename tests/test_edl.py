from fractions import Fraction

from hypothesis import given
from hypothesis import strategies as st

from podedit import edl
from tests.conftest import clean_cut, make_words


def words_and_cuts(glued: list[bool]):
    """len(glued)+1 words one second apart; glued[k] says boundary k+1 is uncuttable."""
    n = len(glued) + 1
    words = make_words([("jade", f"w{i}", i, i + 0.5) for i in range(n)])
    cuts = [clean_cut(0, 0)] + [None if g else clean_cut(i + 0.75, 0.25) for i, g in enumerate(glued)]
    cuts.append(clean_cut(n, 0))
    return words, cuts


@given(st.sets(st.integers(0, 60)))
def test_runs_partition_the_set(ix):
    runs = edl.runs_of(ix)
    assert {i for a, b in runs for i in range(a, b + 1)} == ix
    assert all(runs[k][1] + 1 < runs[k + 1][0] for k in range(len(runs) - 1))


def test_plan_reports_unclean_with_suggestion():
    words, cuts = words_and_cuts([False, True, False, False])  # w1‿w2
    plan = edl.plan_deletions(words, cuts, {1, 4})
    assert plan.runs == [(4, 4)]
    assert len(plan.problems) == 1
    msg = plan.problems[0].message
    assert "can't cut 'w1'" in msg and "after it" in msg and "'w1 w2'" in msg


def test_unclean_on_both_sides():
    words, cuts = words_and_cuts([True, True])
    msg = edl.plan_deletions(words, cuts, {1}).problems[0].message
    assert "before and after" in msg and "'w0 w1 w2'" in msg


def test_first_and_last_words_are_always_cuttable():
    words, cuts = words_and_cuts([False])
    cuts[0] = cuts[-1] = None  # even without quiet at the edges
    assert edl.plan_deletions(words, cuts, {0, 1}).runs == [(0, 1)]


def test_removals_cut_at_quiet_midpoints():
    words, cuts = words_and_cuts([False, False, False])
    assert edl.removals([(1, 1)], cuts, 4.0) == [(0.75, 1.75)]
    assert edl.removals([(0, 0)], cuts, 4.0) == [(0.0, 0.75)]
    assert edl.removals([(3, 3)], cuts, 4.0) == [(2.75, 4.0)]


def test_max_gap_shortens_pauses():
    words, cuts = words_and_cuts([False, False])
    cuts[1] = {"t": 2.0, "qs": 0.5, "qe": 3.5}  # a 3 s pause before w1
    assert edl.removals([], cuts, 3.0, max_gap=1.0) == [(1.0, 3.0)]
    cuts[2] = {"t": 5.0, "qs": 4.0, "qe": 6.0}
    # deleting w1 joins two long pauses; each side is trimmed to half the max gap
    assert edl.removals([(1, 1)], cuts, 7.0, max_gap=1.0) == [(1.0, 5.5)]


@given(st.floats(1, 1000), st.lists(st.tuples(st.floats(-5, 1005), st.floats(0, 50)), max_size=20))
def test_keep_intervals_complement(duration, raw):
    """Property: keeps are sorted, disjoint, inside [0, d], and avoid every removal."""
    remove = [(a, a + w) for a, w in raw]
    keeps = edl.keep_intervals(duration, remove)
    assert all(0 <= a < b <= duration for a, b in keeps)
    assert all(keeps[k][1] < keeps[k + 1][0] for k in range(len(keeps) - 1))
    for a, b in remove:
        if b > a:  # empty removals remove nothing
            assert all(kb <= a or ka >= b for ka, kb in keeps)
    # and nothing else is removed: sample points outside all removals are kept
    for x in [duration * f for f in (0.01, 0.33, 0.5, 0.77, 0.99)]:
        if not any(a <= x <= b for a, b in remove):
            assert any(ka <= x <= kb for ka, kb in keeps)


@given(st.lists(st.tuples(st.floats(0, 100), st.floats(0, 5)), max_size=20),
       st.sampled_from([Fraction(30), Fraction(25), Fraction(30000, 1001), Fraction(60)]))
def test_snap_to_frames(raw, fps):
    keeps = edl.keep_intervals(200.0, [(a, a + w) for a, w in raw])
    frames = edl.snap_to_frames(keeps, fps)
    assert all(a < b for a, b in frames)
    assert all(frames[k][1] < frames[k + 1][0] for k in range(len(frames) - 1))
    kept = sum(b - a for a, b in keeps)
    assert abs(sum(b - a for a, b in frames) / fps - kept) <= len(keeps) / fps


def test_snap_merges_touching():
    assert edl.snap_to_frames([(0, 1.0), (1.01, 2.0)], Fraction(30)) == [(0, 60)]
    assert edl.snap_to_frames([(0, 0.001)], Fraction(30)) == []
