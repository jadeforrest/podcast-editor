import json
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from podedit import loudness, render as rnd
from podedit.media import decode_audio
from tests.conftest import clean_cut, ffmpeg, make_words


def frames(level, n=100):
    return np.full(n, level, dtype=np.float32)


def test_speech_level_gates_pauses_and_ignores_other_times():
    db = np.concatenate([frames(-20.0), frames(-90.0, 20), frames(-20.0), frames(-5.0, 100)])
    words = [{"start": 0.0, "end": 2.2}]  # speech with a pause in it; the -5 dB stretch is someone else
    assert abs(loudness.speech_level(db, words) - -20.0) < 1e-4


def test_speech_level_needs_enough_real_speech():
    assert loudness.speech_level(frames(-20.0), [{"start": 0.0, "end": 0.3}]) is None
    assert loudness.speech_level(frames(-90.0), [{"start": 0.0, "end": 1.0}]) is None  # words not in this signal
    assert loudness.speech_level(frames(-20.0), []) is None


@given(st.lists(st.floats(-60, -10), min_size=60, max_size=300), st.floats(-15, 15))
def test_speech_level_moves_with_gain(levels, shift):
    """Turning a signal up by x dB raises its measured level by exactly x dB."""
    db = np.array(levels, dtype=np.float32)
    words = [{"start": 0.0, "end": len(levels) * 0.01}]
    a, b = loudness.speech_level(db, words), loudness.speech_level(db + np.float32(shift), words)
    assert a is not None and b is not None and abs((b - a) - shift) < 1e-3


@given(st.one_of(st.none(), st.floats(-120, 10)))
def test_gain_reaches_target_within_limits(level):
    g = loudness.gain_for(level)
    assert -loudness.MAX_CUT <= g <= loudness.MAX_BOOST
    if level is None:
        assert g == 0.0
    elif -loudness.MAX_BOOST <= loudness.TARGET_DB - level <= loudness.MAX_CUT:
        assert abs(level + g - loudness.TARGET_DB) < 1e-9


def test_mix_command_applies_gains_and_limits():
    tracks = [{"name": "a", "file": "a.wav", "has_audio": True}, {"name": "b", "file": "b.wav", "has_audio": True}]
    graph = rnd.mix_command(Path("/ep"), tracks, {"a": -6.0, "b": 0.0})
    graph = graph[graph.index("-filter_complex") + 1]
    assert "volume=-6.00dB" in graph and graph.count("volume=") == 1 and rnd.LIMITER in graph


def test_describe():
    assert loudness.describe({"a": -30.0, "b": None}, {"a": 10.0, "b": 0.0}) == \
        "Audio levels: a -30.0 dB -> +10.0 dB, b unmeasured -> +0.0 dB"


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_render_evens_out_people_without_clipping(tmp_path):
    """A loud person and a very quiet one (whose boosted cough would clip) come out at one level, under 0 dBFS."""
    from podedit.paths import Episode
    ep = Episode(tmp_path / "ep")
    ep.create()
    loud = "if(lt(t,2),0.5*sin(2*PI*300*t),0)"
    quiet = "if(between(t,2,3.8),0.02*sin(2*PI*500*t),0)+if(between(t,3.85,3.9),0.9*sin(2*PI*800*t),0)"
    for name, expr in (("loud", loud), ("quiet", quiet)):
        ffmpeg("-f", "lavfi", "-i", f"aevalsrc='{expr}':s=48000:d=4", str(ep.input_dir / f"{name}.wav"))
    tracks = [{"name": n, "file": f"input/{n}.wav", "has_video": False, "has_audio": True, "audio_source": True}
              for n in ("loud", "quiet")]
    words = make_words([("loud", "hello", 0.2, 1.8), ("quiet", "hi", 2.2, 3.7)])
    t = {"duration": 4.0, "tracks": tracks, "words": words, "dropped": [],
         "cuts": [clean_cut(0, 0), clean_cut(2.0), clean_cut(4.0, 0)]}
    ep.transcript_json.write_text(json.dumps(t))
    from podedit.doc import render_markdown
    plan = rnd.plan_edit(t, render_markdown(t, ep.name))
    out = rnd.render(ep, t, plan, layout="speaker", camera=None, intro=None, outro=None, cover=None, fast=False)
    a = decode_audio(out, 48000, channels=2).reshape(-1, 2)
    first, second = rms_db(a[int(0.5 * 48000):int(1.5 * 48000)]), rms_db(a[int(2.5 * 48000):int(3.5 * 48000)])
    assert abs(first - second) < 1.0, (first, second)
    assert abs(first - loudness.TARGET_DB) < 1.0
    assert np.abs(a).max() < 0.95  # the cough was boosted ~20 dB; the limiter holds it near -1 dBFS

    plain = rnd.render(ep, t, plan, layout="speaker", camera=None, intro=None, outro=None, cover=None, fast=False,
                       normalize=False)
    b = decode_audio(plain, 48000, channels=2).reshape(-1, 2)
    assert rms_db(b[int(0.5 * 48000):int(1.5 * 48000)]) - rms_db(b[int(2.5 * 48000):int(3.5 * 48000)]) > 20


def test_measure_matches_stereo_and_mono_sources(tmp_path):
    """The same speech in a mono and a stereo file measures the same level as it will be mixed."""
    ffmpeg("-f", "lavfi", "-i", "sine=f=400:d=2", str(tmp_path / "m.wav"))
    ffmpeg("-f", "lavfi", "-i", "sine=f=400:d=2", "-af", "pan=stereo|c0=0.707*c0|c1=0.707*c0", str(tmp_path / "s.wav"))
    tracks = [{"name": n, "file": f"{n}.wav", "has_audio": True} for n in ("m", "s")]
    words = make_words([("m", "x", 0.1, 1.9), ("s", "x", 0.1, 1.9)])
    lv = loudness.measure(tmp_path, tracks, words)
    assert abs(lv["m"] - lv["s"]) < 0.2, lv
