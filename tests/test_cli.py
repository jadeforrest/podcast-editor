import json
import shutil

import numpy as np
import pytest

from podedit import cli, paths, transcribe
from podedit.media import parse_probe, parse_rate, probe
from podedit.paths import Episode, find_template, find_tracks, resolve_episode
from tests.test_render import CAMS, md_without, transcript


@pytest.fixture
def episodes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "EPISODES", tmp_path / "episodes")
    monkeypatch.setattr(cli, "TEMPLATES", tmp_path / "templates")
    (tmp_path / "templates").mkdir()
    return tmp_path / "episodes"


def test_new_and_status(episodes, capsys):
    assert cli.main(["new", "ep1"]) == 0
    assert (episodes / "ep1" / "input").is_dir()
    (episodes / "ep1" / "input" / "jade.wav").write_bytes(b"")
    cli.main(["new", "ep2"])
    (episodes / "ep2" / "transcript.md").write_text("x")
    cli.main(["status"])
    out = capsys.readouterr().out
    assert "ep1" in out and "has input" in out and "transcribed" in out and "templates" in out


def test_check_and_render_refusal(make_episode, capsys):
    t = transcript(CAMS)
    ep = make_episode(t, md_without(t, "the‿"))
    assert cli.main(["check", str(ep.root)]) == 1
    assert "CAN'T CUT" in capsys.readouterr().out
    assert cli.main(["check", str(ep.root), "--keep-unclean"]) == 0
    assert cli.main(["render", str(ep.root)]) == 1
    assert "Not rendering" in capsys.readouterr().err


def test_render_command(make_episode, media_dir, monkeypatch, capsys):
    t = transcript(CAMS)
    ep = make_episode(t, md_without(t, "um "))
    for f in ("a.mp4", "b.mp4"):
        shutil.copy(media_dir / f, ep.input_dir / f)
    shutil.copy(media_dir / "intro.mp4", ep.root / "intro.mp4")
    monkeypatch.setattr(paths, "TEMPLATES", ep.root / "none")
    assert cli.main(["render", str(ep.root), "--no-outro", "--max-gap", "1"]) == 0
    out = capsys.readouterr().out
    assert "intro: " + str(ep.root / "intro.mp4") in out and "outro: none" in out
    assert ep.output_mp4.is_file()


def test_missing_transcript(tmp_path):
    with pytest.raises(SystemExit, match="No transcript yet"):
        cli.main(["check", str(tmp_path)])


def test_paths(tmp_path):
    ep = Episode(tmp_path / "ep")
    assert find_tracks(ep) == []
    ep.create()
    for name in ("b.MP4", "a.wav", ".hidden.wav", "notes.txt"):
        (ep.input_dir / name).write_bytes(b"")
    assert [p.name for p in find_tracks(ep)] == ["a.wav", "b.MP4"]
    tpl = tmp_path / "tpl"
    tpl.mkdir()
    (tpl / "intro.mov").write_bytes(b"")
    (tpl / "cover.png").write_bytes(b"")
    assert find_template(ep, "intro", tpl) == tpl / "intro.mov"
    (ep.root / "intro.mp4").write_bytes(b"")
    assert find_template(ep, "intro", tpl) == ep.root / "intro.mp4"
    assert find_template(ep, "cover", tpl) == tpl / "cover.png"
    assert find_template(ep, "outro", tpl) is None
    assert resolve_episode(str(ep.root)).root == ep.root.resolve()
    assert resolve_episode("nope").root == paths.EPISODES / "nope"
    assert ep.output_mp4.name == "ep.mp4"


def test_probe_parsing(media_dir):
    assert parse_rate("30000/1001") == pytest.approx(29.97, abs=0.01)
    assert parse_rate("0/0") is None and parse_rate(None) is None and parse_rate("x") is None
    p = parse_probe(media_dir / "x", {"streams": [
        {"codec_type": "video", "disposition": {"attached_pic": 1}}, {"codec_type": "audio"}],
        "format": {"duration": "3.5"}})
    assert not p.has_video and p.has_audio and p.duration == 3.5
    real = probe(media_dir / "a.mp4")
    assert real.has_video and real.width == 320 and real.fps == 25


# --- transcribe, with Whisper replaced by a fake ------------------------------------------------

def fake_whisper(audio, model, language):
    return {"segments": [{"words": [{"word": " Hello,", "start": 0.1, "end": 0.5},
                                    {"word": " ", "start": 0.5, "end": 0.6},
                                    {"word": " world.", "start": 2.0, "end": 2.6}]}]}


def test_transcribe_pipeline(tmp_path, media_dir, monkeypatch, capsys):
    monkeypatch.setattr(transcribe, "run_whisper", fake_whisper)
    ep = Episode(tmp_path / "ep")
    ep.create()
    shutil.copy(media_dir / "a.mp4", ep.input_dir / "a.mp4")
    assert cli.main(["transcribe", str(ep.root), "--language", ""]) == 0
    t = json.loads(ep.transcript_json.read_text())
    assert [w["text"] for w in t["words"]] == ["Hello,", "world."]
    assert len(t["cuts"]) == 3 and t["settings"]["language"] is None
    assert "Hello," in ep.transcript_md.read_text()
    # second run uses the cache and refuses to clobber edits
    ep.transcript_md.write_text("edited")
    monkeypatch.setattr(transcribe, "run_whisper", lambda *a: pytest.fail("cache not used"))
    with pytest.raises(SystemExit, match="has edits"):
        cli.main(["transcribe", str(ep.root), "--language", ""])
    assert "cached" in capsys.readouterr().out
    assert cli.main(["transcribe", str(ep.root), "--language", "", "--force", "--quiet-db", "-30"]) == 0


def test_transcribe_errors(tmp_path, media_dir, monkeypatch):
    ep = Episode(tmp_path / "ep")
    ep.create()
    with pytest.raises(SystemExit, match="No media files"):
        transcribe.build(ep)
    shutil.copy(media_dir / "a.mp4", ep.input_dir / "x.mp4")
    shutil.copy(media_dir / "c.wav", ep.input_dir / "x.wav")
    with pytest.raises(SystemExit, match="same name"):
        transcribe.build(ep)


def test_transcribe_helpers(media_dir):
    short = parse_probe(media_dir / "a", {"format": {"duration": "1"}})
    long = parse_probe(media_dir / "b", {"format": {"duration": "5"}})
    assert "differ in length" in transcribe.check_lengths([short, long])[0]
    assert transcribe.check_lengths([long, long]) == []
    m = transcribe.mix([np.ones(3, np.float32), np.ones(1, np.float32)])
    assert m.tolist() == [2, 1, 1]
    s = transcribe.summary({"words": [1], "tracks": [1], "dropped": [], "cuts": [{}, None, {}, {}]})
    assert "1 of 2 word boundaries (50%)" in s
    assert "0 of 0" in transcribe.summary({"words": [], "tracks": [1], "dropped": [], "cuts": [{}]})
