"""Tracks -> work/transcript.json + transcript.md."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import analysis
from .doc import render_markdown
from .media import Probe, decode_audio, probe
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


def track_info(p: Probe, ep: Episode) -> dict:
    info = {"name": p.path.stem, "file": str(p.path.relative_to(ep.root)), "duration": round(p.duration, 3),
            "has_video": p.has_video, "has_audio": p.has_audio}
    if p.has_video:
        info.update(width=p.width, height=p.height, fps=str(p.fps) if p.fps else None)
    return info


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
    names = [p.stem for p in paths]
    if len(set(names)) != len(names):
        raise SystemExit("Two input files have the same name (before the extension); rename one.")
    probes = [probe(p) for p in paths]
    for warning in check_lengths(probes):
        print(f"warning: {warning}")
    return probes


def build(ep: Episode, model: str = DEFAULT_MODEL, language: str | None = "en",
          min_quiet: float = 0.04, quiet_db: float | None = None) -> dict:
    probes = probe_tracks(ep)
    print(f"Decoding {len(probes)} track(s)...")
    audio = {p.path.stem: decode_audio(p.path, RATE) for p in probes}
    words_by_track = {p.path.stem: transcribe_track(ep, p.path, audio[p.path.stem], model, language) for p in probes}
    print("Analysing pauses and overlaps...")
    dbs = {name: analysis.frame_db(a, RATE) for name, a in audio.items()}
    words, dropped = analysis.filter_words(words_by_track, dbs)
    mix_db = analysis.frame_db(mix(list(audio.values())), RATE)
    lv = analysis.levels(mix_db)
    threshold = analysis.quiet_threshold(lv) if quiet_db is None else quiet_db
    duration = max(p.duration for p in probes)
    cuts = analysis.compute_cuts(words, mix_db, duration, threshold, min_quiet=min_quiet)
    for i, w in enumerate(words):
        w["id"] = i
    return {
        "version": 1, "episode": ep.name, "duration": round(duration, 3),
        "tracks": [track_info(p, ep) for p in probes],
        "settings": {"model": model, "language": language, "min_quiet": min_quiet,
                     "quiet_db": round(threshold, 1), "noise_db": round(lv.noise, 1), "speech_db": round(lv.speech, 1)},
        "words": words, "cuts": cuts, "dropped": dropped,
    }


def write_outputs(ep: Episode, transcript: dict, force: bool) -> None:
    if ep.transcript_md.exists() and not force and (
            not ep.original_md.exists() or ep.transcript_md.read_text() != ep.original_md.read_text()):
        raise SystemExit(f"{ep.transcript_md} has edits. Re-run with --force to overwrite them.")
    ep.transcript_json.write_text(json.dumps(transcript, indent=1))
    md = render_markdown(transcript, ep.name)
    ep.original_md.write_text(md)
    ep.transcript_md.write_text(md)


def summary(transcript: dict) -> str:
    words, cuts = transcript["words"], transcript["cuts"]
    inner = cuts[1:-1]
    glued = sum(c is None for c in inner)
    pct = 100 * glued / len(inner) if inner else 0
    return (f"{len(words)} words from {len(transcript['tracks'])} track(s); "
            f"{glued} of {len(inner)} word boundaries ({pct:.0f}%) can't be cut cleanly; "
            f"{len(transcript['dropped'])} likely bleed/hallucinated words dropped.")
