# podcast-editor

Transcript-driven podcast video editor. User docs: README.md (workflow, folder convention, transcript format).

- Entry point `./pod` → `podedit/cli.py`. Python 3.12 venv in `.venv/` (mlx-whisper needs ≤3.12-era wheels; system python is 3.14).
- `work/transcript.json` is the source of truth for timings; `transcript.md` holds none. Edits are recovered by aligning
  the markdown against the JSON word list (`doc.diff_edit` → `align.opcodes`), anchored on speaker lines.
- `cuts[i]` is the boundary before `words[i]` (`cuts[len(words)]` after the last); `None` = not cleanly cuttable (`‿` in the markdown).
- Cut points are video frame boundaries inside the quiet stretch, chosen at transcribe time (`analysis.cut_dict`, grid =
  highest camera fps, or 1 ms for audio-only). Keep ranges are frames and audio sample ranges derive from them; keep it
  that way or A/V drifts (`test_av_sync_survives_many_cuts`).
- Tracks are grouped per person by file stem; `audio_source` in transcript.json says whose sound is used
  (`transcribe.audio_sources`). Audio/video start-time offsets are compensated via `media.offset_filter`.
- Default `speaker` layout: cameras are fit to the canvas and hstacked into a strip; after the frame `select`, a
  `crop` whose x is a per-frame expression of `n` picks the camera (`shots.crop_x_expr`). Shots are in output frames.
- `check`/`render` also write `output/<episode>.transcript.md`: kept words timed on the rendered video's clock
  (intro included) via `timeline.Clock`, for show notes/chapters. transcript.md times stay original-recording times.
- Tests: `.venv/bin/python -m pytest` (uses real ffmpeg on tiny lavfi media; Whisper is faked). CRAP gate: `tools/crap.py` (see README).
- Media lives outside the repo: `$PODEDIT_HOME` (default `~/Movies/podcast-editor`) holds `episodes/` and `templates/`
  (`paths.data_home`). Tests get a temp home from an autouse fixture in `tests/conftest.py`; never point a test at the real one.
