import json
import shutil
from fractions import Fraction
from pathlib import Path

import pytest

from podedit import render as rnd
from podedit.media import probe
from tests.conftest import clean_cut, make_words

CAMS = [
    {"name": "a", "file": "input/a.mp4", "has_video": True, "has_audio": True, "width": 320, "height": 240, "fps": "25"},
    {"name": "b", "file": "input/b.mp4", "has_video": True, "has_audio": True, "width": 320, "height": 180, "fps": "25"},
]
AUDIO_ONLY = [{"name": "c", "file": "input/c.wav", "has_video": False, "has_audio": True}]


def transcript(tracks, duration=4.0):
    words = make_words([("a", "um", 0.5, 0.8), ("a", "the", 1.0, 1.3), ("a", "car", 1.3, 1.7),
                        ("b", "yes", 2.5, 2.9)])
    cuts = [clean_cut(0.2), clean_cut(0.9), None, clean_cut(2.1), clean_cut(3.5)]
    return {"duration": duration, "tracks": tracks, "words": words, "cuts": cuts, "dropped": []}


def md_without(t, *texts):
    from podedit.doc import render_markdown
    md = render_markdown(t, "x")
    for text in texts:
        md = md.replace(text, "", 1)
    return md


def test_plan_edit_cuts_and_problems():
    t = transcript(CAMS)
    plan = rnd.plan_edit(t, md_without(t, "um "))
    assert plan.deleted == 1 and not plan.problems
    assert plan.frames == [(22, 100)] and plan.fps == 25  # first word: cut from the very start
    assert abs(plan.kept_seconds - 3.12) < 1e-9
    mid = rnd.plan_edit(t, md_without(t, "yes"))
    assert mid.frames == [(0, 52)]  # last word: cut to the very end
    bad = md_without(t, "the‿")
    assert len(rnd.plan_edit(t, bad).problems) == 1
    lenient = rnd.plan_edit(t, bad, keep_unclean=True)
    assert not lenient.problems and "kept (not cut cleanly)" in lenient.warnings[0]
    assert lenient.frames == [(0, 100)]


def test_fps_and_samples():
    assert rnd.video_fps({"tracks": AUDIO_ONLY}) == 30
    assert rnd.video_fps({"tracks": CAMS + [{**CAMS[0], "fps": "30000/1001"}]}) == Fraction(30000, 1001)
    assert rnd.sample_ranges([(0, 25), (50, 75)], Fraction(25)) == [(0, 48000), (96000, 144000)]


def test_choose_cameras():
    assert rnd.choose_cameras(CAMS, "side-by-side", None) == CAMS
    assert rnd.choose_cameras(CAMS, "single", None) == CAMS[:1]
    assert rnd.choose_cameras(CAMS, "single", "b") == CAMS[1:]
    assert rnd.choose_cameras(AUDIO_ONLY, "single", None) == []
    with pytest.raises(SystemExit, match="No video track called 'z'"):
        rnd.choose_cameras(CAMS, "single", "z")


def test_graph_side_by_side_with_intro(media_dir):
    t = transcript(CAMS)
    plan = rnd.plan_edit(t, md_without(t, "um "))
    intro = rnd.Segment(media_dir / "intro.mp4", probe(media_dir / "intro.mp4"))
    outro = rnd.Segment(media_dir / "outro.mp4", probe(media_dir / "outro.mp4"))
    g = rnd.build_graph(Path("/ep"), t, plan, Path("/ep/work/a.f32"), intro=intro, outro=outro)
    script = ";".join(g.filters)
    assert "hstack=inputs=2" in script and "crop='min(iw,160)':240" in script
    assert "select='between(n,22,99)'" in script
    assert "anullsrc" in script  # outro has no audio track
    assert script.endswith("[introv][introa][mv][ma][outrov][outroa]concat=n=3:v=1:a=1[outv][outa]")
    assert g.args()[:2] == ["-i", "/ep/input/a.mp4"]


def test_graph_audio_only_cover_and_black(media_dir):
    t = transcript(AUDIO_ONLY)
    plan = rnd.plan_edit(t, md_without(t))
    g = rnd.build_graph(Path("/ep"), t, plan, Path("a.f32"), cover=media_dir / "cover.png")
    assert "-loop" in g.args() and "scale=1920:1080" in g.filters[0]
    g = rnd.build_graph(Path("/ep"), t, plan, Path("a.f32"))
    assert any("color=c=black:s=1920x1080" in a for a in g.args())
    assert rnd.canvas_size([], rnd.Segment(media_dir / "intro.mp4", probe(media_dir / "intro.mp4"))) == (640, 360)
    assert rnd.select_expr([]) == "0"


def test_encoder_and_mix_commands():
    assert "libx264" in rnd.encoder_args(False) and "h264_videotoolbox" in rnd.encoder_args(True)
    cmd = rnd.mix_command(Path("/ep"), CAMS + [{"name": "v", "file": "v.mp4", "has_video": True, "has_audio": False}])
    assert cmd.count("-i") == 2 and "amix=inputs=2" in cmd[cmd.index("-filter_complex") + 1]
    single = rnd.mix_command(Path("/ep"), AUDIO_ONLY)
    assert "amix" not in single[single.index("-filter_complex") + 1]


def setup_episode(tmp_path, media_dir, tracks, files):
    from podedit.paths import Episode
    ep = Episode(tmp_path / "ep")
    ep.create()
    for f in files:
        shutil.copy(media_dir / f, ep.input_dir / f)
    t = transcript(tracks)
    ep.transcript_json.write_text(json.dumps(t))
    return ep, t


@pytest.mark.parametrize("tracks,files,layout,expect_size", [
    (CAMS, ["a.mp4", "b.mp4"], "side-by-side", (320, 240)),
    (CAMS, ["a.mp4", "b.mp4"], "single", (320, 240)),
    (AUDIO_ONLY, ["c.wav"], "side-by-side", (640, 360)),
])
def test_render_end_to_end(tmp_path, media_dir, tracks, files, layout, expect_size):
    ep, t = setup_episode(tmp_path, media_dir, tracks, files)
    plan = rnd.plan_edit(t, md_without(t, "um "))
    out = rnd.render(ep, t, plan, layout=layout, camera=None, intro=media_dir / "intro.mp4",
                     outro=media_dir / "outro.mp4", cover=None, fast=False)
    p = probe(out)
    assert (p.width, p.height) == expect_size and p.has_audio
    assert abs(p.duration - (1 + plan.kept_seconds + 1)) < 0.1
    assert not (ep.work_dir / "edited-audio.f32").exists()


def test_failed_ffmpeg_raises(tmp_path):
    with pytest.raises(SystemExit, match="ffmpeg failed"):
        rnd.run_ffmpeg(["-i", str(tmp_path / "missing.mp4"), str(tmp_path / "o.mp4")], 1.0, "x")
