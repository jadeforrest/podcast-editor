"""Episode directory convention.

    episodes/<episode>/
        input/            drop the source tracks here (one file per person)
        transcript.md     the editable transcript (the only file you edit)
        intro.* outro.*   optional per-episode overrides of templates/
        work/             generated data: timings, caches, scratch audio
        output/           the rendered <episode>.mp4
    templates/intro.*, templates/outro.*, templates/cover.*   shared defaults
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPISODES = ROOT / "episodes"
TEMPLATES = ROOT / "templates"

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
AUDIO_EXT = {".wav", ".aif", ".aiff", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
MEDIA_EXT = VIDEO_EXT | AUDIO_EXT


@dataclass(frozen=True)
class Episode:
    root: Path

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def transcript_md(self) -> Path:
        return self.root / "transcript.md"

    @property
    def transcript_json(self) -> Path:
        return self.work_dir / "transcript.json"

    @property
    def original_md(self) -> Path:
        return self.work_dir / "transcript.original.md"

    @property
    def output_mp4(self) -> Path:
        return self.output_dir / f"{self.name}.mp4"

    def create(self) -> None:
        for d in (self.input_dir, self.work_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)


def resolve_episode(arg: str) -> Episode:
    """An episode is named by its directory name under episodes/, or by a path."""
    p = Path(arg).expanduser()
    if p.is_dir():
        return Episode(p.resolve())
    return Episode(EPISODES / arg)


def find_tracks(ep: Episode) -> list[Path]:
    if not ep.input_dir.is_dir():
        return []
    files = [f for f in ep.input_dir.iterdir() if f.suffix.lower() in MEDIA_EXT and not f.name.startswith(".")]
    return sorted(files, key=lambda f: f.name.lower())


def find_named(dirs: list[Path], stem: str, exts: set[str]) -> Path | None:
    """First file called <stem>.<ext> in the first directory that has one."""
    for d in dirs:
        for ext in sorted(exts):
            for candidate in (d / f"{stem}{ext}", d / f"{stem}{ext.upper()}"):
                if candidate.is_file():
                    return candidate
    return None


def find_template(ep: Episode, stem: str, templates: Path = TEMPLATES) -> Path | None:
    exts = IMAGE_EXT if stem == "cover" else MEDIA_EXT
    return find_named([ep.root, templates], stem, exts)
