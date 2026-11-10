"""Edited transcript -> output/<episode>.mp4 (with optional intro/outro)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from tqdm import tqdm

from . import doc, edl
from .media import Probe, probe
from .paths import Episode
from .splice import splice_stream

AUDIO_RATE = 48000
CHANNELS = 2
FADE = int(AUDIO_RATE * 0.005)  # 5 ms fade at each splice
DEFAULT_FPS = Fraction(30)
DEFAULT_SIZE = (1920, 1080)
AFORMAT = f"aformat=sample_fmts=fltp:sample_rates={AUDIO_RATE}:channel_layouts=stereo"


@dataclass
class EditPlan:
    frames: list[tuple[int, int]]
    fps: Fraction
    deleted: int
    problems: list[edl.Problem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def kept_frames(self) -> int:
        return sum(b - a for a, b in self.frames)

    @property
    def kept_seconds(self) -> float:
        return float(self.kept_frames / self.fps)


def video_fps(transcript: dict) -> Fraction:
    rates = [Fraction(t["fps"]) for t in transcript["tracks"] if t.get("has_video") and t.get("fps")]
    return max(rates) if rates else DEFAULT_FPS


def plan_edit(transcript: dict, md: str, keep_unclean: bool = False, max_gap: float | None = None) -> EditPlan:
    words, cuts, duration = transcript["words"], transcript["cuts"], transcript["duration"]
    result = doc.diff_edit(words, md)
    plan = edl.plan_deletions(words, cuts, result.deleted)
    fps = video_fps(transcript)
    keeps = edl.keep_intervals(duration, edl.removals(plan.runs, cuts, duration, max_gap))
    deleted = sum(j - i + 1 for i, j in plan.runs)
    out = EditPlan(edl.snap_to_frames(keeps, fps), fps, deleted, plan.problems, result.warnings)
    if keep_unclean:
        out.warnings += [f"kept (not cut cleanly): {p.message.splitlines()[0]}" for p in plan.problems]
        out.problems = []
    return out


def sample_ranges(frames: list[tuple[int, int]], fps: Fraction, rate: int = AUDIO_RATE) -> list[tuple[int, int]]:
    return [(round(a * rate / fps), round(b * rate / fps)) for a, b in frames]


# --- ffmpeg graph -----------------------------------------------------------------------------

def even(x: int) -> int:
    return max(2, x - x % 2)


@dataclass
class Segment:
    """An intro/outro clip normalised to the main canvas."""
    path: Path
    probe: Probe


def choose_cameras(tracks: list[dict], layout: str, camera: str | None) -> list[dict]:
    videos = [t for t in tracks if t.get("has_video")]
    if not videos or layout == "side-by-side":
        return videos
    if camera:
        match = [t for t in videos if t["name"] == camera]
        if not match:
            raise SystemExit(f"No video track called {camera!r}; have: {', '.join(t['name'] for t in videos)}")
        return match
    return videos[:1]


def canvas_size(cameras: list[dict], intro: Segment | None) -> tuple[int, int]:
    if cameras:
        return even(max(t["width"] for t in cameras)), even(max(t["height"] for t in cameras))
    if intro and intro.probe.has_video:
        return even(intro.probe.width), even(intro.probe.height)
    return DEFAULT_SIZE


def camera_chain(idx: int, cell_w: int, h: int, fps: Fraction) -> str:
    """Scale a camera to the canvas height and centre-crop it to its cell."""
    return (f"[{idx}:v:0]setpts=PTS-STARTPTS,fps={fps},scale=-2:{h},crop='min(iw,{cell_w})':{h},"
            f"pad={cell_w}:{h}:(ow-iw)/2:0,setsar=1,tpad=stop_mode=clone:stop_duration=5")


def select_expr(frames: list[tuple[int, int]]) -> str:
    return "+".join(f"between(n,{a},{b - 1})" for a, b in frames) or "0"


def fit_chain(w: int, h: int, fps: Fraction) -> str:
    return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},format=yuv420p")


@dataclass
class Graph:
    inputs: list[list[str]] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)

    def add_input(self, *args: str) -> int:
        self.inputs.append(list(args))
        return len(self.inputs) - 1

    def args(self) -> list[str]:
        return [a for inp in self.inputs for a in inp]


def main_video(g: Graph, ep_root: Path, cameras: list[dict], plan: EditPlan, size: tuple[int, int],
               cover: Path | None) -> None:
    w, h = size
    fps, total = plan.fps, plan.kept_frames
    if not cameras:
        seconds = f"{plan.kept_seconds + 1:.3f}"
        if cover:
            k = g.add_input("-loop", "1", "-framerate", str(fps), "-t", seconds, "-i", str(cover))
        else:
            k = g.add_input("-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps}:d={seconds}")
        g.filters.append(f"[{k}:v:0]{fit_chain(w, h, fps)},trim=end_frame={total}[mv]")
        return
    cell = even(w // len(cameras))
    labels = []
    for n, cam in enumerate(cameras):
        k = g.add_input("-i", str(ep_root / cam["file"]))
        g.filters.append(f"{camera_chain(k, cell, h, fps)}[c{n}]")
        labels.append(f"[c{n}]")
    stack = f"hstack=inputs={len(labels)}," if len(labels) > 1 else ""
    pad = f"pad={w}:{h}:(ow-iw)/2:0," if cell * len(cameras) != w else ""
    g.filters.append(f"{''.join(labels)}{stack}{pad}select='{select_expr(plan.frames)}',"
                     f"setpts=N/FRAME_RATE/TB,format=yuv420p,trim=end_frame={total}[mv]")


def segment_chains(g: Graph, seg: Segment, label: str, size: tuple[int, int], fps: Fraction) -> None:
    w, h = size
    k = g.add_input("-i", str(seg.path))
    d = f"{seg.probe.duration:.3f}"
    if seg.probe.has_video:
        g.filters.append(f"[{k}:v:0]{fit_chain(w, h, fps)}[{label}v]")
    else:
        g.filters.append(f"color=c=black:s={w}x{h}:r={fps}:d={d},setsar=1,format=yuv420p[{label}v]")
    if seg.probe.has_audio:
        g.filters.append(f"[{k}:a:0]aresample={AUDIO_RATE},{AFORMAT}[{label}a]")
    else:
        g.filters.append(f"anullsrc=r={AUDIO_RATE}:cl=stereo,atrim=duration={d},{AFORMAT}[{label}a]")


def build_graph(ep_root: Path, transcript: dict, plan: EditPlan, audio_file: Path, layout: str = "side-by-side",
                camera: str | None = None, intro: Segment | None = None, outro: Segment | None = None,
                cover: Path | None = None) -> Graph:
    g = Graph()
    cameras = choose_cameras(transcript["tracks"], layout, camera)
    size = canvas_size(cameras, intro)
    main_video(g, ep_root, cameras, plan, size, cover)
    k = g.add_input("-f", "f32le", "-ar", str(AUDIO_RATE), "-ac", str(CHANNELS), "-i", str(audio_file))
    g.filters.append(f"[{k}:a:0]{AFORMAT}[ma]")
    parts = []
    for label, seg in (("intro", intro), ("m", None), ("outro", outro)):
        if label == "m":
            parts.append("[mv][ma]")
        elif seg is not None:
            segment_chains(g, seg, label, size, plan.fps)
            parts.append(f"[{label}v][{label}a]")
    g.filters.append(f"{''.join(parts)}concat=n={len(parts)}:v=1:a=1[outv][outa]")
    return g


def encoder_args(fast: bool) -> list[str]:
    if fast:
        return ["-c:v", "h264_videotoolbox", "-q:v", "70", "-allow_sw", "1"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]


# --- running ----------------------------------------------------------------------------------

def mix_command(ep_root: Path, tracks: list[dict]) -> list[str]:
    audio = [t for t in tracks if t.get("has_audio")]
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    for t in audio:
        cmd += ["-i", str(ep_root / t["file"])]
    ins = "".join(f"[{i}:a:0]" for i in range(len(audio)))
    mixer = f"amix=inputs={len(audio)}:normalize=0:duration=longest," if len(audio) > 1 else ""
    graph = f"{ins}{mixer}aresample={AUDIO_RATE},aformat=sample_fmts=flt:channel_layouts=stereo[a]"
    return cmd + ["-filter_complex", graph, "-map", "[a]", "-f", "f32le", "pipe:1"]


def cut_audio(ep: Episode, transcript: dict, plan: EditPlan, dest: Path) -> None:
    ranges = sample_ranges(plan.frames, plan.fps)
    total = int(round(transcript["duration"] * AUDIO_RATE))
    proc = subprocess.Popen(mix_command(ep.root, transcript["tracks"]), stdout=subprocess.PIPE)
    with tqdm(total=total, desc="Cutting audio  ", bar_format="{l_bar}{bar}| [{elapsed}<{remaining}]") as bar, \
            open(dest, "wb") as out:
        splice_stream(proc.stdout, out, ranges, total, CHANNELS, FADE,
                      on_progress=lambda frames: bar.update(min(frames, bar.total - bar.n)))
        proc.stdout.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed while decoding the source audio")


def run_ffmpeg(args: list[str], seconds: float, desc: str) -> None:
    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error", "-progress", "pipe:1", "-nostats"] + args
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    with tqdm(total=round(seconds, 1), unit="s", desc=desc, bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]") as bar:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and value.isdigit():
                bar.n = min(bar.total, int(value) / 1e6)
                bar.refresh()
        bar.n = bar.total
        bar.refresh()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed while rendering (see the error above)")


def load_segment(path: Path | None) -> Segment | None:
    return Segment(path, probe(path)) if path else None


def render(ep: Episode, transcript: dict, plan: EditPlan, *, layout: str, camera: str | None, intro: Path | None,
           outro: Path | None, cover: Path | None, fast: bool) -> Path:
    ep.output_dir.mkdir(exist_ok=True)
    audio_file = ep.work_dir / "edited-audio.f32"
    cut_audio(ep, transcript, plan, audio_file)
    segs = load_segment(intro), load_segment(outro)
    g = build_graph(ep.root, transcript, plan, audio_file, layout, camera, segs[0], segs[1], cover)
    script = ep.work_dir / "render-filter.txt"
    script.write_text(";\n".join(g.filters))
    partial = ep.output_dir / f".{ep.name}.partial.mp4"
    total = plan.kept_seconds + sum(s.probe.duration for s in segs if s)
    args = g.args() + ["-filter_complex_script", str(script), "-map", "[outv]", "-map", "[outa]",
                       *encoder_args(fast), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
                       "-movflags", "+faststart", str(partial)]
    run_ffmpeg(args, total, "Rendering video")
    partial.replace(ep.output_mp4)
    audio_file.unlink(missing_ok=True)
    return ep.output_mp4
