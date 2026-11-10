import json
import shutil

import numpy as np
import pytest

from podedit import cli, paths, transcribe
from fractions import Fraction

from podedit.media import offset_filter, parse_probe, parse_rate, probe, start_time
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
    assert parse_rate("1567/52") == 30 and parse_rate("12/1") == 12 and parse_rate("24000/1001") == Fraction(24000, 1001)
    late = parse_probe(media_dir / "x", {"streams": [
        {"codec_type": "video", "start_time": "0.000"}, {"codec_type": "audio", "start_time": "0.050"}],
        "format": {}})
    assert late.audio_offset == 0.05
    assert offset_filter(0.05) == "adelay=50.0:all=1," and offset_filter(-0.02).startswith("atrim=start=0.0200")
    assert offset_filter(0) == ""
    assert "audio starts +50 ms" in transcribe.offset_warnings([late])[0]
    assert start_time({"start_time": "N/A"}) == 0.0


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
    # editing, then re-running with new analysis settings keeps the edits (words come from the cache)
    md = ep.transcript_md.read_text()
    ep.transcript_md.write_text(md.replace("Hello,", ""))
    monkeypatch.setattr(transcribe, "run_whisper", lambda *a: pytest.fail("cache not used"))
    assert cli.main(["transcribe", str(ep.root), "--language", "", "--quiet-db", "-30"]) == 0
    out = capsys.readouterr().out
    assert "cached" in out and "Kept your edits (1 words deleted)" in out
    assert "Hello," not in ep.transcript_md.read_text() and "world." in ep.transcript_md.read_text()
    assert (ep.work_dir / "transcript.before-retranscribe.md").is_file()
    # a different model would change the words: refused before any slow work
    with pytest.raises(SystemExit, match="different model"):
        cli.main(["transcribe", str(ep.root), "--language", "", "--model", "other"])
    assert cli.main(["transcribe", str(ep.root), "--language", "", "--force"]) == 0
    assert "Hello," in ep.transcript_md.read_text()


def test_carry_over_refuses_if_words_changed(tmp_path, media_dir, monkeypatch):
    monkeypatch.setattr(transcribe, "run_whisper", fake_whisper)
    ep = Episode(tmp_path / "ep")
    ep.create()
    shutil.copy(media_dir / "a.mp4", ep.input_dir / "a.mp4")
    cli.main(["transcribe", str(ep.root)])
    ep.transcript_md.write_text("edited")
    t = json.loads(ep.transcript_json.read_text())
    t["words"][0]["text"] = "Goodbye"
    with pytest.raises(SystemExit, match="words changed"):
        transcribe.write_outputs(ep, t, force=False)


def test_camera_paired_with_mic_and_silent_camera(tmp_path, media_dir, monkeypatch):
    seen = []
    monkeypatch.setattr(transcribe, "run_whisper", lambda audio, m, l: seen.append(len(audio)) or fake_whisper(0, 0, 0))
    ep = Episode(tmp_path / "ep")
    ep.create()
    shutil.copy(media_dir / "a.mp4", ep.input_dir / "jade.mp4")     # camera with its own (scratch) sound
    shutil.copy(media_dir / "c.wav", ep.input_dir / "jade.wav")     # jade's real mic
    shutil.copy(media_dir / "silent.mp4", ep.input_dir / "wide.mp4")  # a camera with no sound at all
    t = transcribe.build(ep)
    roles = {tr["file"]: tr["audio_source"] for tr in t["tracks"]}
    assert roles == {"input/jade.mp4": False, "input/jade.wav": True, "input/wide.mp4": False}
    assert len(seen) == 1  # only the mic is transcribed
    assert {w["speaker"] for w in t["words"]} == {"jade"}
    from podedit import render as rnd
    cmd = rnd.mix_command(ep.root, t["tracks"])
    assert cmd.count("-i") == 1 and str(ep.root / "input/jade.wav") in cmd


def test_only_silent_files(tmp_path, media_dir):
    ep = Episode(tmp_path / "ep")
    ep.create()
    shutil.copy(media_dir / "silent.mp4", ep.input_dir / "cam.mp4")
    with pytest.raises(SystemExit, match="has any sound"):
        transcribe.build(ep)


def test_new_rejects_bad_names(episodes):
    for bad in ("../x", "a/b", ".hidden", ""):
        with pytest.raises(SystemExit, match="Episode names"):
            cli.main(["new", bad])


def test_transcribe_errors(tmp_path, media_dir, monkeypatch):
    ep = Episode(tmp_path / "ep")
    ep.create()
    with pytest.raises(SystemExit, match="No media files"):
        transcribe.build(ep)
    shutil.copy(media_dir / "c.wav", ep.input_dir / "x.wav")
    shutil.copy(media_dir / "c.wav", ep.input_dir / "x.flac")
    with pytest.raises(SystemExit, match="same kind"):
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
