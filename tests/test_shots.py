from fractions import Fraction

from hypothesis import given
from hypothesis import strategies as st

from podedit import shots

FPS = Fraction(25)


def w(speaker, start, end):
    return {"speaker": speaker, "text": "x", "start": start, "end": end}


def test_merge_and_output_mapping():
    assert shots.merge([(10, 20), (0, 5), (22, 30)], gap=2) == [(0, 5), (10, 30)]
    # kept: source frames 0-50 and 100-150; speech at 1.0-3.0 s and 3.6-4.4 s (frames 25-75, 90-110)
    assert shots.to_output_frames([(1.0, 3.0), (3.6, 4.4)], [(0, 50), (100, 150)], FPS) == [(25, 50), (50, 60)]


def test_short_interjections_and_min_shot():
    words = [w("a", 0, 5), w("b", 2, 2.4),          # "yeah": too short to switch
             w("b", 5.2, 6.4), w("a", 6.5, 12)]   # a replies 1.3 s after the switch to b...
    got = shots.plan_shots(words, ["a", "b"], [(0, 300)], FPS, min_shot=2.0)
    # ...so switching back waits until b's shot has lasted 2 s (frame 130 + 50)
    assert got == [(0, 0), (130, 1), (180, 0)]


def test_switch_dropped_when_turn_ends_before_min_shot():
    words = [w("a", 0, 1.2), w("b", 1.3, 2.4), w("a", 2.5, 6)]
    assert shots.plan_shots(words, ["a", "b"], [(0, 150)], FPS, min_shot=3.0) == [(0, 0)]


def test_start_on_first_real_speaker_and_unknown_speakers_ignored():
    words = [w("guest-mic", 0, 4), w("b", 1, 3)]
    assert shots.plan_shots(words, ["a", "b"], [(0, 100)], FPS) == [(0, 1)]
    assert shots.plan_shots([], ["a", "b"], [(0, 100)], FPS) == [(0, 0)]


def test_crop_expression():
    assert shots.crop_x_expr([(0, 0), (50, 1), (90, 0), (120, 2)], 1920, 200) == \
        "1920*between(n,50,89)+3840*between(n,120,199)"
    assert shots.crop_x_expr([(0, 0)], 1920, 10) == "0"


@given(st.lists(st.tuples(st.sampled_from("abc"), st.floats(0, 100), st.floats(0.05, 8)), max_size=40),
       st.floats(0.5, 5))
def test_shot_invariants(raw, min_shot):
    """Property: shots start at 0, are increasing, alternate cameras, and each lasts at least min_shot."""
    words = [w(s, a, a + d) for s, a, d in raw]
    got = shots.plan_shots(words, ["a", "b", "c"], [(0, 3000)], FPS, min_shot)
    assert got[0][0] == 0
    starts = [s for s, _ in got]
    assert starts == sorted(set(starts))
    assert all(got[k][1] != got[k + 1][1] for k in range(len(got) - 1))
    assert all(got[k + 1][0] - got[k][0] >= round(min_shot * FPS) for k in range(len(got) - 1))
    assert all(0 <= cam < 3 for _, cam in got)


def test_output_frames_accumulate_over_several_kept_spans():
    """Mutation gap: each kept span starts where the previous ones end on the edited timeline."""
    got = shots.to_output_frames([(0.0, 100.0)], [(0, 10), (20, 30), (40, 50)], Fraction(1))
    assert got == [(0, 10), (10, 20), (20, 30)]
    assert shots.to_output_frames([(15.0, 15.0)], [(0, 30)], Fraction(1)) == []


def test_crop_expression_when_the_last_shot_is_a_later_camera():
    assert shots.crop_x_expr([(0, 1), (50, 2)], 100, 80) == "100*between(n,0,49)+200*between(n,50,79)"
