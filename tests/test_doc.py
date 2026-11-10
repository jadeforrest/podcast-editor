import time

from hypothesis import given, settings
from hypothesis import strategies as st

from podedit import align
from podedit.doc import GLUE, diff_edit, display_text, paragraph_starts, parse_markdown, render_markdown, timestamp
from podedit.text import norm
from tests.conftest import clean_cut, make_words

VOCAB = ["um", "the", "car", "is", "red", "uh", "I", "think", "so", "yes", "Hello,", "world.", "don't", "--"]


def transcript_from(texts, speakers, glue_flags):
    words = make_words([(speakers[i], t, i * 1.0, i * 1.0 + 0.5) for i, t in enumerate(texts)])
    cuts = [clean_cut(0.0, 0)] + [None if g else clean_cut(i + 0.75) for i, g in enumerate(glue_flags)]
    cuts[-1] = clean_cut(len(texts) * 1.0, 0)
    return {"words": words, "cuts": cuts}


@st.composite
def transcripts(draw, max_size=40):
    n = draw(st.integers(0, max_size))
    texts = draw(st.lists(st.sampled_from(VOCAB), min_size=n, max_size=n))
    speakers = draw(st.lists(st.sampled_from(["jade", "guest"]), min_size=n, max_size=n))
    glue = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    return transcript_from(texts, speakers, glue)


def test_norm():
    assert norm("Hello,") == "hello"
    assert norm("Don’t") == "don't"
    assert norm("--") == "--"
    assert norm("'quoted'") == "quoted"


def test_timestamp():
    assert timestamp(3723.9) == "01:02:03"


def test_display_text_removes_internal_space():
    assert display_text({"text": "1, 000"}) == "1,000"
    assert display_text({"text": " "}) == "…"


def test_render_layout():
    t = transcript_from(["um", "the", "car", "yes"], ["jade", "jade", "jade", "guest"], [True, False, True, False])
    md = render_markdown(t, "Ep")
    assert "**jade** [00:00:00]\num‿the car‿\n" in md
    assert "**guest** [00:00:03]\nyes\n" in md


@given(transcripts())
def test_parse_inverts_render(t):
    """Property: reading back an untouched transcript gives exactly the word tokens."""
    assert parse_markdown(render_markdown(t, "T")) == [display_text(w) for w in t["words"]]


def edited_md(t, drop, drop_headers=False):
    """The markdown a user would get by deleting the words in `drop` (and emptied speaker lines)."""
    words = t["words"]
    out = []
    starts = set(paragraph_starts(words))
    for i, w in enumerate(words):
        if i in starts:
            out.append(f"\n**{w['speaker']}** [{timestamp(w['start'])}]\n")
        if i not in drop:
            out.append(display_text(w) + " ")
    md = "".join(out)
    if drop_headers:
        lines = md.split("\n")
        md = "\n".join(l for k, l in enumerate(lines)
                       if not (l.startswith("**") and (k + 1 >= len(lines) or not lines[k + 1].strip())))
    return md


def plain(words):
    return " ".join(words)


@given(transcripts())
def test_untouched_means_no_edits(t):
    r = diff_edit(t["words"], render_markdown(t, "T"))
    assert r.deleted == set() and r.warnings == []


@given(transcripts(), st.data(), st.booleans())
def test_deleting_any_subset(t, data, drop_headers):
    """Property: deleting any words is recovered exactly, with or without their emptied speaker lines."""
    n = len(t["words"])
    drop = data.draw(st.sets(st.integers(0, max(0, n - 1)), max_size=n)) if n else set()
    r = diff_edit(t["words"], edited_md(t, drop, drop_headers))
    kept = [norm(display_text(w)) for i, w in enumerate(t["words"]) if i not in r.deleted]
    assert kept == [norm(display_text(w)) for i, w in enumerate(t["words"]) if i not in drop]
    assert len(r.deleted) == len(drop)
    assert r.warnings == []


@given(transcripts(), st.data())
def test_paragraph_deletes_keep_the_right_speaker(t, data):
    """Property: deleting whole paragraphs (with speaker lines) deletes exactly those words."""
    words = t["words"]
    starts = paragraph_starts(words)
    bounds = starts + [len(words)]
    chosen = data.draw(st.sets(st.integers(0, max(0, len(starts) - 1)))) if starts else set()
    drop = {i for p in chosen for i in range(bounds[p], bounds[p + 1])}
    assert diff_edit(words, edited_md(t, drop, drop_headers=True)).deleted == drop


def test_wrong_copy_of_repeated_word_is_not_kept():
    t = transcript_from(["yes", "yes", "right"], ["jade", "guest", "guest"], [False] * 3)
    md = render_markdown(t, "T").replace("**jade** [00:00:00]\nyes\n", "")
    assert diff_edit(t["words"], md).deleted == {0}


def test_deletion_of_unique_words_is_exact():
    t = transcript_from(["um", "the", "car", "is", "red"], ["jade"] * 5, [False] * 5)
    assert diff_edit(t["words"], "the car red").deleted == {0, 3}


def test_glue_and_punctuation_and_case_are_ignored():
    t = transcript_from(["Hello,", "world."], ["jade"] * 2, [True, False])
    assert diff_edit(t["words"], "hello WORLD").deleted == set()
    assert diff_edit(t["words"], f"Hello,{GLUE}").deleted == {1}


def test_removed_speaker_line_with_text_kept_changes_nothing():
    t = transcript_from(["hi", "there", "yes", "ok"], ["jade", "jade", "guest", "jade"], [False] * 4)
    md = render_markdown(t, "T").replace("**guest** [00:00:02]\n", "")
    r = diff_edit(t["words"], md)
    assert r.deleted == set() and r.warnings == []


def test_insert_and_reword_warn_and_keep_audio():
    t = transcript_from(["the", "car", "is", "red"], ["jade"] * 4, [False] * 4)
    r = diff_edit(t["words"], "the big car is red")
    assert r.deleted == set() and "added text ignored" in r.warnings[0]
    r = diff_edit(t["words"], "the automobile vehicle is red")
    assert r.deleted == set() and "reworded" in r.warnings[0]
    r = diff_edit(t["words"], "the car is red indeed")
    assert "added text" in r.warnings[0]
    r = diff_edit(t["words"], "Preface the car is red")
    assert "added text" in r.warnings[0]


def test_spelling_fix_is_silent():
    t = transcript_from(["the", "colour", "is", "red"], ["jade"] * 4, [False] * 4)
    r = diff_edit(t["words"], "the color is red")
    assert r.deleted == set() and r.warnings == []


def test_huge_unanchored_span_falls_back_with_warning(monkeypatch):
    monkeypatch.setattr(align, "MAX_CELLS", 2)
    t = transcript_from(["a", "b", "c", "d", "e"], ["jade"] * 5, [False] * 5)
    r = diff_edit(t["words"], "a c e")
    assert r.deleted == {1, 3} and "approximate" in r.warnings[0]


@given(st.lists(st.sampled_from("abc"), max_size=12), st.lists(st.sampled_from("abc"), max_size=12))
def test_opcodes_are_a_valid_exact_diff(a, b):
    """Property: opcodes rebuild b from a, and the equal parts are a longest common subsequence."""
    ops, exact = align.opcodes(a, b)
    rebuilt, equal = [], 0
    for op, i1, i2, j1, j2 in ops:
        rebuilt += a[i1:i2] if op == "equal" else b[j1:j2]
        equal += (i2 - i1) if op == "equal" else 0
    assert rebuilt == b and exact
    import difflib
    best = max((m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()), default=0)
    assert equal >= best


def test_parse_ignores_headers_comments_titles():
    md = "# Title\n<!-- delete me\nstill comment -->\n**jade** [00:00:01]\nhello‿there  friend\n\n"
    assert parse_markdown(md) == ["hello", "there", "friend"]


def test_deleting_whole_paragraph():
    t = transcript_from(["hi", "there", "yes", "ok"], ["jade", "jade", "guest", "jade"], [False] * 4)
    md = render_markdown(t, "T").replace("**guest** [00:00:02]\nyes\n", "")
    assert diff_edit(t["words"], md).deleted == {2}


@settings(deadline=None, max_examples=1)
@given(st.just(None))
def test_diff_is_fast_enough_for_long_episodes(_):
    texts = [VOCAB[(i * 7 + i // 13) % len(VOCAB)] for i in range(20000)]
    t = transcript_from(texts, ["jade"] * len(texts), [False] * len(texts))
    start = time.time()
    r = diff_edit(t["words"], edited_md(t, {i for i in range(len(texts)) if i % 50 == 0}))
    assert len(r.deleted) == 400 and r.warnings == []
    assert time.time() - start < 20


def test_paragraph_breaks():
    words = make_words([("jade", "one", 0, 0.5), ("jade", "two.", 0.6, 1), ("jade", "three", 3.5, 4)]
                       + [("jade", "w.", 5 + k, 5.5 + k) for k in range(85)]
                       + [("jade", "x", 100 + k, 100.5 + k) for k in range(250)])
    starts = paragraph_starts(words)
    assert starts[:2] == [0, 2]  # 2.5 s pause
    assert 2 + 80 in starts      # sentence end after the soft limit
    assert any(s - starts[k - 1] == 200 for k, s in enumerate(starts) if k)  # hard limit
