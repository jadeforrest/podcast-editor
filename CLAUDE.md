# podcast-editor

Transcript-driven podcast video editor. User docs: README.md (workflow, folder convention, transcript format).

- Entry point `./pod` → `podedit/cli.py`. Python 3.12 venv in `.venv/` (mlx-whisper needs ≤3.12-era wheels; system python is 3.14).
- `work/transcript.json` is the source of truth for timings; `transcript.md` holds none. Edits are recovered by aligning
  the markdown against the JSON word list (`doc.diff_edit` → `align.opcodes`), anchored on speaker lines.
- `cuts[i]` is the boundary before `words[i]` (`cuts[len(words)]` after the last); `None` = not cleanly cuttable (`‿` in the markdown).
- Keep ranges are snapped to video frames (`edl.snap_to_frames`) and audio sample ranges derive from those frames — keep it that way or A/V drifts.
- Default `speaker` layout: cameras are fit to the canvas and hstacked into a strip; after the frame `select`, a
  `crop` whose x is a per-frame expression of `n` picks the camera (`shots.crop_x_expr`). Shots are in output frames.
- Tests: `.venv/bin/python -m pytest` (uses real ffmpeg on tiny lavfi media; Whisper is faked). CRAP gate: `tools/crap.py` (see README).
- Never commit anything under `episodes/` or `templates/` (media; git-ignored).
