import shutil
from fractions import Fraction

from hypothesis import given
from hypothesis import strategies as st

from podedit import cli, paths, timeline
from tests.conftest import make_words
from tests.test_render import CAMS, md_without, transcript

FPS = Fraction(25)


def test_clock_maps_across_cuts():
    clock = timeline.Clock([(0, 25), (50, 100)], FPS)  # keep 0-1s and 2-4s
    assert clock.word({"start": 0.2, "end": 0.4}) == 0.2
    assert clock.word({"start": 1.2, "end": 1.6}) is None        # in the cut
    assert clock.word({"start": 2.5, "end": 2.9}) == 1.5        # 1s kept before, 0.5s into the span
    assert clock.word({"start": 1.95, "end": 2.4}) == 1.0       # starts just before its span: clamped
    assert clock.word({"start": 4.0, "end": 4.2}) is None       # past the end
    assert timeline.Clock([], FPS).word({"start": 0.0, "end": 0.1}) is None


def test_render_drops_cut_words_and_adds_intro():
    t = transcript(CAMS)
    # words: um 0.5, the 1.0, car 1.3 (speaker a), yes 2.5 (speaker b); "um" cut: 0.88s gone from the start
    md = timeline.render(t, [(22, 100)], FPS, "ep", intro=10.0)
    assert "um" not in md
    assert "**a** [00:00:10]\nthe car" in md and "**b** [00:00:11]\nyes" in md
    assert "including a 10.0s intro" in md
    assert "(no intro)" in timeline.render(t, [(0, 100)], FPS, "ep")


@st.composite
def edits(draw):
    n = draw(st.integers(1, 30))
    gaps = draw(st.lists(st.floats(0.0, 3.0), min_size=n, max_size=n))
    lens = draw(st.lists(st.floats(0.05, 1.0), min_size=n, max_size=n))
    spec, t = [], 0.0
    for k, (g, length) in enumerate(zip(gaps, lens)):
        t += g
        spec.append((draw(st.sampled_from("ab")), f"w{k}", round(t, 3), round(t + length, 3)))
        t += length
    total = round(t * FPS) + 5
    marks = sorted(set(draw(st.lists(st.integers(0, total), max_size=12))))
    kept = [(a, b) for a, b in zip(marks[::2], marks[1::2]) if b > a]
    return make_words(spec), kept


def reference_time(word, kept):
    """Linear walk over the kept spans: the plain statement of what Clock.word computes."""
    mid, before = (word["start"] + word["end"]) / 2 * FPS, 0
    for a, b in kept:
        if a <= mid < b:
            return float((before + max(word["start"] * FPS, a) - a) / FPS)
        before += b - a
    return None


@given(edits())
def test_clock_agrees_with_reference_and_keeps_order(case):
    words, kept = case
    clock = timeline.Clock(kept, FPS)
    times = [clock.word(w) for w in words]
    assert times == [reference_time(w, kept) for w in words]
    shown = [x for x in times if x is not None]
    assert shown == sorted(shown)
    assert all(0 <= x <= sum(b - a for a, b in kept) / FPS for x in shown)
    paras = timeline.paragraphs(words, kept, FPS)
    assert sum(len(p[2]) for p in paras) == len(shown)  # every kept word appears exactly once
    assert [p[1] for p in paras] == sorted(p[1] for p in paras)


def test_check_writes_timeline_with_intro(make_episode, media_dir, monkeypatch, capsys):
    t = transcript(CAMS)
    ep = make_episode(t, md_without(t, "um "))
    shutil.copy(media_dir / "intro.mp4", ep.root / "intro.mp4")  # 1s
    monkeypatch.setattr(paths, "TEMPLATES", ep.root / "none")
    assert cli.main(["check", str(ep.root)]) == 0
    assert str(ep.output_transcript) in capsys.readouterr().out
    md = ep.output_transcript.read_text()
    assert "**a** [00:00:01]\nthe car" in md and "including a 1.0s intro" in md
    assert cli.main(["check", str(ep.root), "--no-intro"]) == 0
    assert "**a** [00:00:00]\nthe car" in ep.output_transcript.read_text()


def test_check_with_problems_writes_no_timeline(make_episode):
    t = transcript(CAMS)
    ep = make_episode(t, md_without(t, "the‿"))
    assert cli.main(["check", str(ep.root)]) == 1
    assert not ep.output_transcript.exists()
