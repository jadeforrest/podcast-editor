"""The editable transcript (transcript.md) and how edits map back to words.

The markdown holds no offsets. Each word in it corresponds, in order, to a word
in work/transcript.json, which holds the timings. To find out what was edited,
the words left in the file are aligned against the original word list; words
that no longer appear are the cuts. So deleting text never "shifts" anything:
the timings live in the untouched JSON and are looked up after the fact.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from . import align
from .text import norm

GLUE = "‿"  # ‿ joins words that cannot be cut apart
HEADER = re.compile(r"^\*\*(?P<speaker>[^*]+)\*\*\s*\[(?P<ts>[\d:.]+)\]\s*$")
SPLIT = re.compile(rf"[\s{GLUE}]+")

PREAMBLE = f"""<!--
HOW TO EDIT
- Delete words, sentences or whole paragraphs to cut them from the video.
- Words joined with {GLUE} can't be cut apart (they run together, or people talk
  over each other). Delete the whole joined group or none of it.
  A {GLUE} at the end of a paragraph means it runs into the next one.
- Don't add or reorder words: new words can't be recorded after the fact.
  Fixing a spelling is fine; the original audio is kept.
- Leave speaker lines (**name** [time]) as they are: they anchor your edits.
  To cut a whole paragraph, delete it together with its speaker line.
- Run `./pod check <episode>` to see what will be cut before rendering.
-->
"""


def timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def display_text(word: dict) -> str:
    """A word as one whitespace-free token."""
    return SPLIT.sub("", word["text"]) or "…"


PAUSE_BREAK = 2.0     # seconds of silence that start a new paragraph
SOFT_LIMIT = 80       # words, after which a paragraph ends at the next sentence end
HARD_LIMIT = 200      # words, after which a paragraph ends regardless


def starts_paragraph(words: list[dict], i: int, length: int) -> bool:
    """Whether word i begins a new paragraph, given the current paragraph's length."""
    if i == 0 or words[i]["speaker"] != words[i - 1]["speaker"]:
        return True
    if words[i]["start"] - words[i - 1]["end"] >= PAUSE_BREAK or length >= HARD_LIMIT:
        return True
    return length >= SOFT_LIMIT and words[i - 1]["text"].rstrip("\"')").endswith((".", "?", "!"))


def paragraph_starts(words: list[dict]) -> list[int]:
    starts: list[int] = []
    for i in range(len(words)):
        if starts_paragraph(words, i, i - starts[-1] if starts else 0):
            starts.append(i)
    return starts


def render_markdown(transcript: dict, title: str) -> str:
    words, cuts = transcript["words"], transcript["cuts"]
    lines = [f"# {title}", "", PREAMBLE]
    para = ""
    new_para = set(paragraph_starts(words))
    for i, w in enumerate(words):
        if i in new_para:
            if para:
                lines += [para.rstrip(), ""]
            lines.append(f"**{w['speaker']}** [{timestamp(w['start'])}]")
            para = ""
        glued = i + 1 < len(words) and cuts[i + 1] is None
        para += display_text(w) + (GLUE if glued else " ")
    if para:
        lines += [para.rstrip(), ""]
    return "\n".join(lines)


def strip_comments(md: str) -> str:
    return re.sub(r"<!--.*?-->", " ", md, flags=re.S)


def parse_markdown(md: str) -> list[str]:
    """The words left in an edited transcript, in order."""
    tokens: list[str] = []
    for line in strip_comments(md).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or HEADER.match(stripped):
            continue
        tokens += [t for t in SPLIT.split(stripped) if t]
    return tokens


@dataclass
class EditResult:
    deleted: set[int] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


def ignored_text(kind: str, words: list[dict], new: list[str]) -> str:
    where = f"[{timestamp(words[0]['start'])}] " if words else ""
    if kind == "insert":
        return f"{where}added text ignored (can't create new audio): {' '.join(new)!r}"
    old = " ".join(display_text(w) for w in words)
    return f"{where}reworded text kept with original audio: {old!r} -> {' '.join(new)!r}"


def header_key(speaker: str, start: float) -> tuple[str, str]:
    return speaker.strip(), timestamp(start)


def parse_paragraphs(md: str) -> list[tuple[tuple[str, str] | None, list[str]]]:
    """(speaker-line key, tokens) per paragraph; key None for text before any speaker line."""
    paras: list[tuple[tuple[str, str] | None, list[str]]] = [(None, [])]
    for line in strip_comments(md).splitlines():
        stripped = line.strip()
        header = HEADER.match(stripped)
        if header:
            paras.append(((header["speaker"].strip(), header["ts"]), []))
        elif stripped and not stripped.startswith("#"):
            paras[-1][1].extend(t for t in SPLIT.split(stripped) if t)
    return paras


def segments(words: list[dict], md: str) -> list[tuple[int, int, list[str]]]:
    """Split the edit into (word_start, word_end, edited_tokens) spans anchored on speaker lines.

    A speaker line still present in the edited file pins its paragraph to the
    original one, so alignment only ever happens within a paragraph (or within
    the paragraphs between two surviving speaker lines).
    """
    starts = paragraph_starts(words)
    keys = [header_key(words[s]["speaker"], words[s]["start"]) for s in starts]
    anchors: list[tuple[int, list[str]]] = [(0, [])]
    last = -1
    for key, tokens in parse_paragraphs(md):
        p = next((k for k in range(last + 1, len(keys)) if keys[k] == key), None) if key else None
        if p is None:
            anchors[-1][1].extend(tokens)
        else:
            anchors.append((starts[p], list(tokens)))
            last = p
    bounds = [a for a, _ in anchors] + [len(words)]
    return [(bounds[k], bounds[k + 1], toks) for k, (_, toks) in enumerate(anchors)]


def apply_opcodes(result: EditResult, words: list[dict], offset: int, a: list[str], b: list[str],
                  edited: list[str], ops) -> None:
    for op, i1, i2, j1, j2 in ops:
        if op == "delete":
            result.deleted.update(range(offset + i1, offset + i2))
        elif op == "insert":
            anchor = words[offset + i1:offset + i1 + 1] or words[max(0, offset + i1 - 1):offset + i1]
            result.warnings.append(ignored_text("insert", anchor, edited[j1:j2]))
        elif op == "replace" and (i2 - i1 != j2 - j1 or not all(
                similar(a[i], b[j]) for i, j in zip(range(i1, i2), range(j1, j2)))):
            result.warnings.append(ignored_text("replace", words[offset + i1:offset + i2], edited[j1:j2]))


def diff_edit(words: list[dict], md: str) -> EditResult:
    """Which words were deleted from the edited transcript markdown.

    Matching ignores case and punctuation. Replaced or inserted text can't be
    turned into audio, so it keeps the original words and produces a warning
    (except pure spelling/punctuation fixes, which are silent).
    """
    result = EditResult()
    for start, end, edited in segments(words, md):
        a = [norm(display_text(w)) for w in words[start:end]]
        b = [norm(t) for t in edited]
        ops, exact = align.opcodes(a, b)
        if not exact:
            result.warnings.append(f"[{timestamp(words[start]['start'])}] a long stretch has no speaker lines; "
                                   "matching there is approximate, check the result")
        apply_opcodes(result, words, start, a, b, edited, ops)
    return result


def similar(x: str, y: str) -> bool:
    """A spelling fix rather than a different word."""
    return difflib.SequenceMatcher(None, x, y).ratio() >= 0.6
