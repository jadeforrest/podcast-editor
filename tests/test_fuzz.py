"""Fuzz targets for everything that reads untrusted input: the hand/LLM-edited transcript.md,
Whisper's word text, and ffprobe output. Default runs are small (fast pass); set
HYPOTHESIS_PROFILE=fuzz for a long campaign (slow pass)."""

from fractions import Fraction
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from podedit import loudness, render as rnd, shots
from podedit.doc import diff_edit, display_text, edited_tokens, paragraph_starts, render_markdown
from podedit.media import parse_probe, parse_rate
from tests.conftest import clean_cut, make_words
from tests.test_doc import transcripts

# Characters that mean something to the markdown parser, plus ordinary text.
MD_CHARS = st.sampled_from(list("*[]#‿<!->\n :.0123456789ab ") + ["**jade** [00:00:01]\n", "<!--", "-->", "\n\n"])


@st.composite
def mangled(draw, md: str) -> str:
    """md after random character-level damage: insertions, deletions, duplicated or dropped lines."""
    for _ in range(draw(st.integers(0, 8))):
        kind = draw(st.sampled_from(["insert", "delete", "dup_line", "drop_line"]))
        pos = draw(st.integers(0, len(md)))
        if kind == "insert":
            md = md[:pos] + "".join(draw(st.lists(MD_CHARS, min_size=1, max_size=5))) + md[pos:]
        elif kind == "delete":
            md = md[:pos] + md[pos + draw(st.integers(1, 40)):]
        else:
            lines = md.split("\n")
            k = min(pos, len(lines) - 1)
            lines[k:k + 1] = [lines[k], lines[k]] if kind == "dup_line" else []
            md = "\n".join(lines)
    return md


@given(transcripts(), st.data())
def test_fuzz_diff_edit_survives_any_damage(t, data):
    md = data.draw(mangled(render_markdown(t, "T")))
    r = diff_edit(t["words"], md)
    assert r.deleted <= set(range(len(t["words"])))
    assert all(isinstance(w, str) for w in r.warnings)


@given(transcripts(), st.text(max_size=300))
def test_fuzz_diff_edit_arbitrary_text(t, md):
    r = diff_edit(t["words"], md)
    assert r.deleted <= set(range(len(t["words"])))


WORD_TEXT = st.text(st.characters(codec="utf-8", exclude_categories=["Cs"]), min_size=1, max_size=8)


@given(st.lists(WORD_TEXT, max_size=30), st.data())
def test_fuzz_any_whisper_text_round_trips(texts, data):
    """Whatever Whisper writes as a word, an untouched transcript.md reads back as no edits."""
    n = len(texts)
    speakers = data.draw(st.lists(st.sampled_from(["jade", "guest"]), min_size=n, max_size=n))
    words = make_words([(speakers[i], x, i * 1.0, i + 0.5) for i, x in enumerate(texts)])
    t = {"words": words, "cuts": [clean_cut(0.0, 0)] + [clean_cut(i + 0.75) for i in range(n)]}
    md = render_markdown(t, "T")
    assert edited_tokens(md) == [display_text(w) for w in words]
    assert diff_edit(words, md).deleted == set()


@given(transcripts(), st.data(), st.sampled_from([Fraction(25), Fraction(30000, 1001), Fraction(1000)]),
       st.one_of(st.none(), st.floats(0.05, 3)), st.booleans())
def test_fuzz_plan_edit_keeps_valid_ranges(t, data, fps, max_gap, keep_unclean):
    """Any edit, however mangled, plans sorted, disjoint, in-bounds keep ranges whose audio matches."""
    duration = len(t["words"]) * 1.0 + 0.5
    track = {"name": "jade", "has_video": fps != Fraction(1000), "fps": str(fps)}
    tr = {**t, "duration": duration, "tracks": [track]}
    md = data.draw(mangled(render_markdown(tr, "T")))
    plan = rnd.plan_edit(tr, md, keep_unclean=keep_unclean, max_gap=max_gap, allow_big_cut=True)
    end = round(duration * plan.fps)
    assert all(0 <= a < b <= end for a, b in plan.frames)
    assert all(plan.frames[k][1] < plan.frames[k + 1][0] for k in range(len(plan.frames) - 1))
    samples = rnd.sample_ranges(plan.frames, plan.fps)
    assert all(0 <= a < b for a, b in samples)
    assert all(samples[k][1] <= samples[k + 1][0] for k in range(len(samples) - 1))


RATE_STR = st.one_of(st.none(), st.text(max_size=12), st.sampled_from(["0/0", "1/0", "30000/1001", "inf", "nan",
                                                                         "-25/1", "1e400", "25"]))
NUMISH = st.one_of(st.none(), st.integers(-10, 10_000), st.sampled_from(["", "N/A", "nan", "1.5", "-0.02"]))


@given(st.lists(st.fixed_dictionaries(
    {"codec_type": st.sampled_from(["video", "audio", "data", None])},
    optional={"avg_frame_rate": RATE_STR, "r_frame_rate": RATE_STR, "width": st.integers(0, 8000),
              "height": st.integers(0, 8000), "start_time": NUMISH,
              "disposition": st.fixed_dictionaries({}, optional={"attached_pic": st.integers(0, 1)})}),
    max_size=4), st.one_of(st.none(), st.floats(0, 1e5), st.sampled_from(["", "12.5", "N/A", "nan", "inf"])))
def test_fuzz_parse_probe(streams, duration):
    """ffprobe output for odd files (VFR, broken rates, cover art, missing fields) never crashes the probe."""
    data = {"streams": streams, "format": {} if duration is None else {"duration": duration}}
    p = parse_probe(Path("x.mp4"), data)
    assert p.fps is None or 0 < p.fps <= 240
    assert np.isfinite(p.audio_offset) and np.isfinite(p.duration) and p.duration >= 0


@given(RATE_STR)
def test_fuzz_parse_rate(rate):
    r = parse_rate(rate)
    assert r is None or 0 < r <= 240


@given(st.lists(st.floats(-200, 20), max_size=400),
       st.lists(st.tuples(st.floats(0, 5), st.floats(0, 5)), max_size=10))
def test_fuzz_speech_level(db, spans):
    level = loudness.speech_level(np.array(db, dtype=np.float32), [{"start": a, "end": a + d} for a, d in spans])
    assert level is None or (loudness.FLOOR_DB < level <= 20 + 1e-3)
    assert -loudness.MAX_CUT <= loudness.gain_for(level) <= loudness.MAX_BOOST


@given(st.lists(st.tuples(st.sampled_from("abcz"), st.floats(0, 100), st.floats(0.0, 8)), max_size=40),
       st.lists(st.tuples(st.integers(0, 2500), st.integers(1, 400)), min_size=1, max_size=8),
       st.floats(0.5, 5))
def test_fuzz_shots_with_cuts(raw, kept_raw, min_shot):
    """Shots on an edited timeline stay inside it, start at 0, and only name real cameras."""
    kept, pos = [], 0
    for gap, length in sorted(kept_raw):
        start = max(pos, gap)
        kept.append((start, start + length))
        pos = start + length + 1
    total = sum(b - a for a, b in kept)
    words = [{"speaker": s, "start": a, "end": a + d} for s, a, d in raw]
    got = shots.plan_shots(words, ["a", "b", "c"], kept, Fraction(25), min_shot)
    assert got[0][0] == 0 and all(0 <= s < max(total, 1) and 0 <= c < 3 for s, c in got)
    expr = shots.crop_x_expr(got, 1920, total)
    assert "nan" not in expr and "-" not in expr


# A file stem can be almost anything: '*', brackets, spaces, unicode (not '/' or a newline).
SPEAKER = st.one_of(
    st.sampled_from(["a*b", "*x", "x*", "a**b", "**", " x", "x ", "[x]", "x] [y", "jade"]),
    st.text(st.characters(codec="utf-8", exclude_categories=["Cs", "Cc", "Zl", "Zp"], exclude_characters="/"),
            min_size=1, max_size=10).filter(lambda s: s.strip() and s not in (".", "..")))


@given(st.lists(SPEAKER, min_size=1, max_size=3, unique=True), st.data())
def test_fuzz_any_speaker_name_anchors_edits(names, data):
    """Whatever a person's file is called, deleting a paragraph (with its speaker line) deletes exactly its words."""
    n = data.draw(st.integers(1, 12))
    speakers = data.draw(st.lists(st.sampled_from(names), min_size=n, max_size=n))
    words = make_words([(speakers[i], f"w{i}", i * 1.0, i + 0.5) for i in range(n)])
    t = {"words": words, "cuts": [clean_cut(0.0, 0)] + [clean_cut(i + 0.75) for i in range(n)]}
    starts = paragraph_starts(words)
    bounds = starts + [n]
    p = data.draw(st.integers(0, len(starts) - 1))
    drop = set(range(bounds[p], bounds[p + 1]))
    r = diff_edit(words, render_markdown(t, "T", drop))
    assert r.deleted == drop and r.warnings == []
