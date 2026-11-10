import os
import json
import subprocess
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

# Slow pass: HYPOTHESIS_PROFILE=fuzz runs every property test as a long fuzz campaign.
settings.register_profile("fuzz", max_examples=20_000, deadline=None,
                          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


def make_words(spec):
    """spec: list of (speaker, text, start, end)."""
    return [{"id": i, "speaker": s, "text": t, "start": a, "end": b} for i, (s, t, a, b) in enumerate(spec)]


def clean_cut(t, half=0.05):
    return {"t": t, "qs": round(t - half, 3), "qe": round(t + half, 3)}


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture
def make_episode(tmp_path):
    """Write an episode dir with given transcript dict and (optionally) markdown."""
    from podedit.doc import render_markdown
    from podedit.paths import Episode

    def _make(transcript, md=None, name="ep"):
        ep = Episode(tmp_path / name)
        ep.create()
        ep.transcript_json.write_text(json.dumps(transcript))
        original = render_markdown(transcript, ep.name)
        ep.original_md.write_text(original)
        ep.transcript_md.write_text(md if md is not None else original)
        return ep

    return _make


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory) -> Path:
    """Tiny synthetic media: two 4s camera tracks, an audio-only track, an intro with and without sound."""
    d = tmp_path_factory.mktemp("media")
    ffmpeg("-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=4", "-f", "lavfi", "-i", "sine=f=300:d=4",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "a.mp4"))
    ffmpeg("-f", "lavfi", "-i", "smptebars=s=320x180:r=25:d=4", "-f", "lavfi", "-i", "sine=f=500:d=4",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "b.mp4"))
    ffmpeg("-f", "lavfi", "-i", "sine=f=700:d=4", str(d / "c.wav"))
    ffmpeg("-f", "lavfi", "-i", "color=c=red:s=640x360:r=30:d=1", "-f", "lavfi", "-i", "sine=f=900:d=1",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "intro.mp4"))
    ffmpeg("-f", "lavfi", "-i", "color=c=green:s=640x360:r=30:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           str(d / "outro.mp4"))
    ffmpeg("-f", "lavfi", "-i", "color=c=white:s=200x200", "-frames:v", "1", str(d / "cover.png"))
    ffmpeg("-f", "lavfi", "-i", "testsrc=s=320x240:r=25:d=4", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           str(d / "silent.mp4"))
    # A/V sync source: every second, a white frame and a 1 kHz beep start together
    flash = "color=c=black:s=160x120:r=25:d=6,geq=lum='if(lt(mod(N,25),1),255,16)':cb=128:cr=128"
    beep = "aevalsrc='if(lt(mod(t,1),0.04),0.8*sin(2*PI*1000*t),0)':s=48000:d=6"
    ffmpeg("-f", "lavfi", "-i", flash, "-f", "lavfi", "-i", beep, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", str(d / "sync.mp4"))
    return d
