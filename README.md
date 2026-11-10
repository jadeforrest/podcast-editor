# podcast-editor

Edit a recorded podcast by editing its transcript. Delete words in a text file;
`./pod render` cuts the same moments out of the video and adds your intro/outro.

Runs locally on Apple Silicon (speech recognition with
[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper),
cutting and encoding with ffmpeg). Nothing is uploaded anywhere.

## Workflow

```bash
./pod new 2026-11-10-alice-smith          # 1. make the episode folder
# 2. drop the recordings into episodes/2026-11-10-alice-smith/input/
./pod transcribe 2026-11-10-alice-smith   # 3. writes transcript.md
# 4. edit episodes/2026-11-10-alice-smith/transcript.md (by hand or with an LLM)
./pod check 2026-11-10-alice-smith        # 5. see what will be cut, and anything that can't be
./pod render 2026-11-10-alice-smith       # 6. writes output/2026-11-10-alice-smith.mp4
```

`./pod status` lists episodes and how far along each is. Every command takes
`--help`. The first run creates `.venv/`. The first `transcribe` downloads the
speech model (about 1.6 GB), once.

## Folder convention

```
episodes/<episode>/
    input/              the recordings: one file per person, all from the same session
    transcript.md       the transcript you edit (the only file you touch)
    intro.mp4 outro.mp4 optional: override the shared templates for this episode
    cover.png           optional: picture for audio-only episodes
    work/               generated: timings, recognition cache, temp files
    output/<episode>.mp4
templates/
    intro.*  outro.*  cover.*   shared defaults, used when present
```

Everything under `episodes/` and `templates/` is ignored by git (except the READMEs).

**Input files.** Name each one after the person (`jade.mp4`, `alice.mp4`): the
file name becomes the speaker name in the transcript. Video (`.mp4 .mov .mkv
.webm …`) and audio (`.wav .m4a .mp3 .flac …`) both work, and they can be mixed,
e.g. two camera files, or one video plus a separate good-mic `.wav`. All files
must start at the same moment (as separate-track exports from Riverside, Zoom,
SquadCast, etc. do). They are cut identically.

**Intro/outro.** If `intro.*`/`outro.*` exists in the episode folder it's used;
otherwise the one in `templates/`; otherwise none. `--no-intro`/`--no-outro`
skip them. They're scaled and letterboxed to the episode's frame size; one
without sound gets silence.

## Editing the transcript

```markdown
**jade** [00:01:12]
Um,‿the car is red and‿I think that's fine.

**alice** [00:01:16]
Yes. It's very red.
```

- **Delete** words, sentences or whole paragraphs to cut them.
- **`‿` joins words that can't be cut apart.** Either there's no pause between
  them, or people are talking over each other. Delete the whole joined group or
  none of it. Here you can cut `the car` alone, but not `Um,` without `the`. A `‿`
  at the end of a paragraph means it runs into the next one.
- **Keep the speaker lines** (`**jade** [00:01:12]`). They hold your edits in
  place. To cut a whole paragraph, delete it along with its speaker line. The
  times are where the paragraph starts in the *original* recording. They're
  there to help you find it, and they don't change as you edit.
- **Don't add, reword or reorder** words. New words can't be recorded after the
  fact, so the original audio is kept and `check` lists what was ignored.
  Spelling and punctuation fixes are fine and stay quiet.

`./pod check <episode>` prints what will be cut and how long the result is, and
lists any deletion that can't be made cleanly. For example:

```
CAN'T CUT: [00:00:02] jade: can't cut 'talk about' cleanly (before it runs into the next word)
    context: ...think we should talk about podcasts today...
    smallest clean cut containing it: 'I think we should talk about'
```

Fix it by putting the words back, or deleting the suggested larger phrase.
`render` refuses to run while problems remain. `render --keep-unclean` renders
anyway and leaves those words in.

### Editing with an LLM

Give it `transcript.md` along with something like:

> This is a podcast transcript. Remove filler words (um, uh, "you know", false
> starts, repeated words) and any stretches I mark. Only delete text: never add,
> reword, reorder, or merge paragraphs. Keep every `**name** [time]` line unless
> you delete its whole paragraph. Words joined with `‿` must be deleted all
> together or not at all. Return the full file.

Then run `./pod check`, because LLMs sometimes reword things, and check will
list that.

## Options worth knowing

| Option | What it does |
|---|---|
| `render --max-gap 1.0` | Also shortens every pause longer than 1 s to 1 s |
| `render --layout single --camera alice` | Show one camera instead of side-by-side |
| `render --fast` | Hardware H.264 encoder: several times faster, larger file, a little lower quality |
| `transcribe --model mlx-community/whisper-large-v3-mlx` | Slower, sometimes more accurate model |
| `transcribe --language ''` | Auto-detect the language (default `en`) |
| `transcribe --min-quiet 0.03` | Shortest pause (s) that counts as a clean cut point. Lower means more cuttable, with more risk of a clipped word |
| `transcribe --quiet-db -45` | Override the auto-detected "quiet" level |
| `transcribe --force` | Re-transcribe even though `transcript.md` has edits (they're lost) |

## Output

- **Video layout.** One camera: its full frame. Several cameras: side-by-side,
  each centre-cropped to an equal slice of a frame as large as the biggest
  camera. Cameras are ordered by file name, so prefix `1-`/`2-` to control the
  order (that also changes the speaker names). Audio-only: `cover.*` if present,
  otherwise black, at the intro's size or 1920×1080.
- **Audio.** All tracks mixed at their recorded levels, 48 kHz stereo, AAC 256k.
  Each cut has a 5 ms fade so it doesn't click.
- **Quality.** H.264 at CRF 18 (visually lossless), at the source resolution
  and frame rate. Cuts land exactly on frame boundaries, so sound and picture
  stay in sync however many cuts there are.

## How it works

1. **transcribe.** Each track is recognised separately with word timestamps
   (Whisper is prompted with disfluencies so it keeps "um"/"uh"). Words that
   another mic picked up louder (bleed), and words recognised in near-silence
   (hallucinations), are dropped. The rest are merged in time order. The mixed
   audio is then measured in 10 ms frames. Between every pair of consecutive
   words, the tool looks for a stretch of at least `min-quiet` seconds that is
   quiet relative to the speech level. If one exists, its middle is the cut
   point for that boundary. If not, the boundary gets a `‿`. Everything is
   saved in `work/transcript.json`. `transcript.md` is generated from it and
   holds no timings at all.
2. **check / render.** The edited `transcript.md` is aligned back to the
   original word list: an exact longest-common-subsequence match, done
   paragraph by paragraph using the speaker lines as anchors. Words with no
   match were deleted. Because timings are looked up in the untouched JSON,
   deleting text never shifts any offsets. Each run of deleted words is removed
   from the cut point before it to the cut point after it, which leaves about
   one natural pause. The kept ranges are snapped to video frames. The audio is
   spliced in a streaming pass, and ffmpeg then trims the video, stacks the
   cameras, adds intro/outro and encodes, all in one pass with a progress bar.

Re-running `transcribe` reuses the cached recognition (`work/whisper-*.json`),
so trying new `--min-quiet`/`--quiet-db` values is quick.

## Limitations

- Only words can be cut. A cough, or an "um" that Whisper didn't write down,
  can't be removed from the text. Use `--max-gap` for long silences.
- Whisper's word timings are approximate (often ±0.1 s). The quiet-point
  search tolerates that. If a cut clips a word, raise `--min-quiet`.
- Repeated identical words in one paragraph ("I I think") can't be told apart
  by text. Either copy may be the one removed. That's normally harmless.

## Development

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest                                    # fast pass (~10 s, uses real ffmpeg)
.venv/bin/python -m pytest --cov=podedit --cov-branch --cov-report=json:coverage.json \
  && .venv/bin/python tools/crap.py coverage.json             # CRAP <= 12 per function
```

Code is in `podedit/`: `transcribe` (recognition + analysis), `analysis` (cut
points, bleed filter), `doc` + `align` (markdown and edit alignment), `edl`
(deleted words → keep ranges), `splice` (audio), `render` (ffmpeg graph),
`cli`.
