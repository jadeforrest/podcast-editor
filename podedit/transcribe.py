"""Tracks -> work/transcript.json + transcript.md."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import analysis
from .doc import diff_edit, render_markdown
from .media import AUDIO_GRID, Probe, decode_audio, probe
from .paths import Episode, find_tracks

RATE = 16000
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
# Whisper tends to "clean up" disfluencies; a prompt full of them makes it keep ums and uhs.
FILLER_PROMPT = "Umm, so, uh, I was like, you know, hmm... I mean, uh, yeah. Okay, um, so."


def cache_key(path: Path, model: str, language: str | None) -> str:
    st = path.stat()
    raw = f"{path.name}|{st.st_size}|{int(st.st_mtime)}|{model}|{language}|{FILLER_PROMPT}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def run_whisper(audio: np.ndarray, model: str, language: str | None) -> dict:
    import mlx_whisper  # imported lazily: slow to load, and not needed for editing/rendering

    return mlx_whisper.transcribe(
        audio, path_or_hf_repo=model, language=language, word_timestamps=True,
        initial_prompt=FILLER_PROMPT, condition_on_previous_text=False,
        temperature=0.0, hallucination_silence_threshold=2.0, verbose=False,
    )


def whisper_words(result: dict, speaker: str) -> list[dict]:
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            text = w["word"].strip()
            if text and w["end"] >= w["start"]:
                words.append({"speaker": speaker, "text": text,
                              "start": round(float(w["start"]), 3), "end": round(float(w["end"]), 3)})
    return words


def transcribe_track(ep: Episode, path: Path, audio: np.ndarray, model: str, language: str | None) -> list[dict]:
    cache = ep.work_dir / f"whisper-{path.stem}-{cache_key(path, model, language)}.json"
    if cache.is_file():
        print(f"  {path.name}: using cached recognition ({cache.name})")
        return json.loads(cache.read_text())
    print(f"  {path.name}: recognising speech with {model}")
    words = whisper_words(run_whisper(audio, model, language), path.stem)
    cache.write_text(json.dumps(words))
    return words


def track_info(p: Probe, ep: Episode, audio_source: bool) -> dict:
    """`audio_source`: this file's sound is used (transcribed and mixed). A person's
    camera file is muted when they also have an audio-only file (their mic)."""
    info = {"name": p.path.stem, "file": str(p.path.relative_to(ep.root)), "duration": round(p.duration, 3),
            "has_video": p.has_video, "has_audio": p.has_audio, "audio_source": audio_source,
            "audio_offset": p.audio_offset}
    if p.has_video:
        info.update(width=p.width, height=p.height, fps=str(p.fps) if p.fps else None)
    return info


def audio_sources(probes: list[Probe]) -> dict[str, Probe]:
    """Per person (file name without extension), the file whose sound represents them.

    `jade.mp4` alone: its own sound. `jade.mp4` + `jade.wav`: the .wav (a dedicated
    mic), with the camera muted. A camera with no sound and no mic: nobody.
    """
    people: dict[str, list[Probe]] = {}
    for p in probes:
        people.setdefault(p.path.stem, []).append(p)
    sources = {}
    for name, files in people.items():
        cams = [p for p in files if p.has_video]
        mics = [p for p in files if not p.has_video]
        if len(cams) > 1 or len(mics) > 1:
            raise SystemExit(f"{name!r} has {len(files)} files of the same kind; give each person at most one "
                             "video file and one audio file (e.g. jade.mp4 + jade.wav).")
        source = mics[0] if mics else cams[0]
        if source.has_audio:
            sources[name] = source
    return sources


def mix(tracks: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(max(len(t) for t in tracks), dtype=np.float32)
    for t in tracks:
        out[: len(t)] += t
    return out


def check_lengths(probes: list[Probe]) -> list[str]:
    durations = [p.duration for p in probes]
    if max(durations) - min(durations) > 1.0:
        return [f"tracks differ in length by {max(durations) - min(durations):.1f}s; "
                "they should all be from the same session and start together"]
    return []


def probe_tracks(ep: Episode) -> list[Probe]:
    paths = find_tracks(ep)
    if not paths:
        raise SystemExit(f"No media files in {ep.input_dir}")
    probes = [probe(p) for p in paths]
    for warning in check_lengths(probes) + offset_warnings(probes):
        print(f"warning: {warning}")
    return probes


def cut_grid(probes: list[Probe]):
    """Cuts must fall on video frame boundaries (the highest camera frame rate), else on milliseconds."""
    rates = [p.fps for p in probes if p.has_video and p.fps]
    return max(rates) if rates else AUDIO_GRID


def offset_warnings(probes: list[Probe]) -> list[str]:
    return [f"{p.path.name}: audio starts {p.audio_offset * 1000:+.0f} ms from its video; compensating"
            for p in probes if abs(p.audio_offset) >= 0.01]


def build(ep: Episode, model: str = DEFAULT_MODEL, language: str | None = "en",
          min_quiet: float = 0.04, quiet_db: float | None = None) -> dict:
    probes = probe_tracks(ep)
    sources = audio_sources(probes)
    if not sources:
        raise SystemExit("None of the input files has any sound.")
    print(f"Decoding {len(sources)} audio track(s)...")
    audio = {name: decode_audio(p.path, RATE, p.audio_offset) for name, p in sources.items()}
    words_by_track = {name: transcribe_track(ep, p.path, audio[name], model, language) for name, p in sources.items()}
    print("Analysing pauses and overlaps...")
    dbs = {name: analysis.frame_db(a, RATE) for name, a in audio.items()}
    words, dropped = analysis.filter_words(words_by_track, dbs)
    mix_db = analysis.frame_db(mix(list(audio.values())), RATE)
    lv = analysis.levels(mix_db)
    threshold = analysis.quiet_threshold(lv) if quiet_db is None else quiet_db
    duration = max(p.duration for p in probes)
    cuts = analysis.compute_cuts(words, mix_db, duration, threshold, min_quiet=min_quiet, grid=cut_grid(probes))
    for i, w in enumerate(words):
        w["id"] = i
    return {
        "version": 1, "episode": ep.name, "duration": round(duration, 3),
        "tracks": [track_info(p, ep, sources.get(p.path.stem) is p) for p in probes],
        "settings": {"model": model, "language": language, "min_quiet": min_quiet,
                     "quiet_db": round(threshold, 1), "noise_db": round(lv.noise, 1), "speech_db": round(lv.speech, 1)},
        "words": words, "cuts": cuts, "dropped": dropped,
    }


def has_edits(ep: Episode) -> bool:
    return ep.transcript_md.exists() and (
        not ep.original_md.exists() or ep.transcript_md.read_text() != ep.original_md.read_text())


def check_before_run(ep: Episode, model: str, language: str | None, force: bool) -> None:
    """Refuse up front (before any slow work) if re-transcribing would lose edits.

    With the same model, language and files, the words come back identical (from the
    cache), so edits can be carried over and only the cut analysis changes.
    """
    if force or not has_edits(ep):
        return
    old = json.loads(ep.transcript_json.read_text()) if ep.transcript_json.is_file() else None
    same = old is not None and old["settings"]["model"] == model and old["settings"]["language"] == language \
        and sorted(t["file"] for t in old["tracks"]) == sorted(str(p.relative_to(ep.root)) for p in find_tracks(ep))
    if not same:
        raise SystemExit(f"{ep.transcript_md} has edits, and a different model, language or set of files would "
                         "change the words. Re-run with --force to start over (your edits are lost).")


def same_words(a: list[dict], b: list[dict]) -> bool:
    key = lambda w: (w["speaker"], w["text"], w["start"], w["end"])  # noqa: E731
    return [key(w) for w in a] == [key(w) for w in b]


def carried_over_deletions(ep: Episode, transcript: dict, force: bool) -> set[int] | None:
    """The user's deletions to keep in the regenerated transcript.md, or None for a fresh one."""
    if force or not has_edits(ep):
        return None
    old = json.loads(ep.transcript_json.read_text())
    if not same_words(old["words"], transcript["words"]):
        raise SystemExit("The recognised words changed, so your edits can't be carried over. "
                         "Re-run with --force to start over.")
    backup = ep.work_dir / "transcript.before-retranscribe.md"
    backup.write_text(ep.transcript_md.read_text())
    deleted = diff_edit(old["words"], ep.transcript_md.read_text()).deleted
    print(f"Kept your edits ({len(deleted)} words deleted) and updated the \u203f marks. Spelling fixes and other "
          f"text changes were reset; your previous file is saved as {backup}")
    return deleted


def write_outputs(ep: Episode, transcript: dict, force: bool) -> None:
    deleted = carried_over_deletions(ep, transcript, force)
    ep.transcript_json.write_text(json.dumps(transcript, indent=1))
    original = render_markdown(transcript, ep.name)
    ep.original_md.write_text(original)
    ep.transcript_md.write_text(original if deleted is None else render_markdown(transcript, ep.name, deleted))


def summary(transcript: dict) -> str:
    words, cuts = transcript["words"], transcript["cuts"]
    inner = cuts[1:-1]
    glued = sum(c is None for c in inner)
    pct = 100 * glued / len(inner) if inner else 0
    return (f"{len(words)} words from {len(transcript['tracks'])} track(s); "
            f"{glued} of {len(inner)} word boundaries ({pct:.0f}%) can't be cut cleanly; "
            f"{len(transcript['dropped'])} likely bleed/hallucinated words dropped.")
