import numpy as np
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
    words = [word("a", 0.0, 0.5), word("bbb", 0.4, 1.8, "guest"), word("c", 1.0, 1.5)]
    assert analysis.compute_cuts(words, db, 2.0, -40)[2] is None
    windows = analysis.boundary_windows(words, 2.0, 0.1)
    assert windows[3][0] >= 1.7  # window after "c" starts after the longest word so far ends


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
