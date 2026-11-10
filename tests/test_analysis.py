from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from podedit import analysis
from podedit.analysis import FRAME

RATE = 16000


def tone(seconds, amp=0.3, f=220):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence(seconds, amp=1e-4):
    return (np.random.default_rng(1).standard_normal(int(seconds * RATE)) * amp).astype(np.float32)


def test_frame_db():
    db = analysis.frame_db(tone(1.0, amp=1.0), RATE)
    assert len(db) == 100 and abs(db.mean() - (-3.0)) < 0.2
    assert analysis.frame_db(np.zeros(10, np.float32), RATE).size == 0


@given(st.lists(st.booleans(), max_size=200))
def test_quiet_runs_match_naive(mask):
    starts, ends = analysis.quiet_runs(np.array(mask, dtype=bool))
    rebuilt = [False] * len(mask)
    for s, e in zip(starts, ends):
        assert s < e
        rebuilt[s:e] = [True] * (e - s)
    assert rebuilt == mask


def test_best_quiet_picks_longest_within_window():
    starts, ends = np.array([0, 10, 30]), np.array([5, 25, 32])
    assert analysis.best_quiet(starts, ends, 3, 40, 2) == (10, 25)
    assert analysis.best_quiet(starts, ends, 12, 14, 2) == (12, 14)
    assert analysis.best_quiet(starts, ends, 26, 29, 1) is None
    assert analysis.best_quiet(starts, ends, 30, 32, 3) is None


def test_levels_and_threshold():
    db = np.concatenate([np.full(900, -70.0), np.full(100, -20.0)])
    lv = analysis.levels(db)
    assert lv.noise == -70 and lv.speech == -20
    assert analysis.quiet_threshold(lv) == -48
    assert analysis.quiet_threshold(analysis.Levels(-25, -20)) == -32
    assert analysis.levels(np.full(5, -200.0)).speech == -100


def word(text, start, end, speaker="jade"):
    return {"speaker": speaker, "text": text, "start": start, "end": end}


def test_cuts_clean_at_pauses_glued_without():
    # "um" 0.5-0.8, 0.3 s pause, "the" 1.1-1.4 then "car" immediately at 1.4-1.8
    audio = np.concatenate([silence(0.5), tone(0.3), silence(0.3), tone(0.7), silence(0.5)])
    db = analysis.frame_db(audio, RATE)
    words = [word("um", 0.5, 0.8), word("the", 1.1, 1.4), word("car", 1.4, 1.8)]
    cuts = analysis.compute_cuts(words, db, 2.3, threshold=-40)
    assert cuts[1] is not None and 0.8 <= cuts[1]["t"] <= 1.1
    assert cuts[2] is None
    assert cuts[0]["t"] < 0.5 and cuts[3]["t"] > 1.8


def test_edge_cuts_exist_even_without_silence():
    db = analysis.frame_db(tone(1.0), RATE)
    cuts = analysis.compute_cuts([word("a", 0.0, 1.0)], db, 1.0, threshold=-40)
    assert cuts[0]["t"] == 0.0 and cuts[1]["t"] == 1.0


def test_overlapping_speaker_blocks_cut():
    # B talks over the gap in A's speech; the long B word ends after A's next word starts
    audio = tone(2.0)
    db = analysis.frame_db(audio, RATE)
    words = [word("a", 0.0, 0.5), word("bbb", 0.4, 1.55, "guest"), word("c", 1.0, 1.5)]
    assert analysis.compute_cuts(words, db, 2.0, -40)[2] is None
    windows = analysis.boundary_windows(words, 2.0, 0.1)
    assert windows[3][0] >= 1.45  # window after "c" starts after the longest word so far ends


def test_filter_drops_silent_and_bleed():
    jade = np.concatenate([tone(1.0, 0.3), silence(1.0)])
    guest = np.concatenate([tone(1.0, 0.01), silence(0.5), tone(0.5, 0.3)])
    dbs = {"jade": analysis.frame_db(jade, RATE), "guest": analysis.frame_db(guest, RATE)}
    words_by_track = {
        "jade": [word("hello", 0.1, 0.5), word("thanks", 1.2, 1.4)],                 # thanks: hallucinated in silence
        "guest": [word("Hello", 0.1, 0.5, "guest"), word("hi", 0.1, 0.5, "guest"),  # Hello: bleed; hi: no twin
                  word("yes", 1.6, 1.9, "guest")],
    }
    kept, dropped = analysis.filter_words(words_by_track, dbs)
    assert sorted(w["text"] for w in kept) == ["hello", "hi", "yes"]
    assert {(w["text"], w["reason"]) for w in dropped} == {("thanks", "silent"), ("Hello", "bleed")}


def test_peak_db_out_of_range():
    assert analysis.peak_db(np.zeros(3, np.float32), 10.0, 11.0) == -100.0
    assert FRAME == 0.01


def test_cuts_land_on_frame_boundaries_inside_quiet():
    from fractions import Fraction
    grid = Fraction(24)
    c = analysis.cut_dict(1.0, 1.05, grid)                 # contains 25/24 = 1.0417
    assert c["t"] == float(Fraction(25, 24)) and 1.0 <= c["t"] <= 1.05
    assert analysis.cut_dict(1.005, 1.035, grid) is None    # 30 ms of quiet between two frame boundaries
    assert analysis.cut_dict(1.005, 1.035)["t"] == 1.02     # fine without a grid (unit use)


@given(st.floats(0, 100), st.floats(0.001, 1), st.sampled_from([24, 25, 30, 60, 1000]))
def test_grid_cut_property(qs, width, rate):
    """Property: a grid cut is a frame boundary inside the quiet stretch, and exists whenever one does."""
    import math
    from fractions import Fraction
    grid = Fraction(rate)
    c = analysis.cut_dict(qs, qs + width, grid)
    exists = math.ceil(qs * rate - 1e-9) <= math.floor((qs + width) * rate + 1e-9)
    assert (c is not None) == exists
    if c:
        assert qs - 1e-6 <= c["t"] <= qs + width + 1e-6
        assert abs(c["t"] * rate - round(c["t"] * rate)) < 1e-6


def test_stretched_whisper_end_does_not_hide_a_pause():
    # "so" really ends at 0.6 s, but Whisper says 3.0 s; the 2 s pause after it is real silence
    audio = np.concatenate([tone(0.6), silence(2.0), tone(0.5), silence(0.2)])
    db = analysis.frame_db(audio, RATE)
    words = [word("so", 0.0, 3.0), word("anyway", 2.6, 3.1)]
    cuts = analysis.compute_cuts(words, db, 3.3, threshold=-40)
    assert cuts[1] is not None and 1.2 <= cuts[1]["t"] <= 2.6


def test_bleed_is_caught_on_the_first_track_too():
    """Mutation gap: the bleed check must look at every other track, not stop at its own."""
    jade, guest = tone(1.0, 0.01), tone(1.0, 0.3)  # jade's mic only faintly picked up guest's "hello"
    dbs = {"jade": analysis.frame_db(jade, RATE), "guest": analysis.frame_db(guest, RATE)}
    kept, dropped = analysis.filter_words(
        {"jade": [word("Hello", 0.1, 0.5)], "guest": [word("hello", 0.1, 0.5, "guest")]}, dbs)
    assert [w["speaker"] for w in kept] == ["guest"] and [(w["speaker"], w["reason"]) for w in dropped] == \
        [("jade", "bleed")]


def test_windows_allow_slack_around_word_timings():
    words = [word("a", 0.5, 0.6), word("b", 2.0, 2.5)]
    got = analysis.boundary_windows(words, 3.0, 0.1)
    assert got == [(0.0, pytest.approx(0.6)), (pytest.approx(0.5), pytest.approx(2.1)), (pytest.approx(2.4), 3.0)]


def test_compute_cuts_uses_the_grid_and_min_quiet():
    db = np.full(300, -10.0, dtype=np.float32)
    db[100:130] = -80.0   # 0.3 s pause after "a"
    db[200:202] = -80.0   # 0.02 s gap after "b": shorter than min_quiet
    words = [word("a", 0.2, 0.9), word("b", 1.4, 1.95), word("c", 2.1, 2.8)]
    cuts = analysis.compute_cuts(words, db, 3.0, -40.0, min_quiet=0.04, grid=Fraction(25))
    assert cuts[1] is not None and abs(cuts[1]["t"] * 25 - round(cuts[1]["t"] * 25)) < 1e-9 and cuts[1]["t"] != 1.15
    assert cuts[2] is None


def test_cut_dict_keeps_the_quiet_stretch():
    assert analysis.cut_dict(1.00049, 1.2004) == {"t": 1.1, "qs": 1.0, "qe": 1.2}
