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
file name, without its extension, becomes the speaker name in the transcript.
Video (`.mp4 .mov .mkv .webm …`) and audio (`.wav .m4a .mp3 .flac …`) both work.
All files must start at the same moment (as separate-track exports from
Riverside, Zoom, SquadCast, etc. do). They are cut identically.

| A person has | Used for sound | Used for picture |
|---|---|---|
| `jade.mp4` | its own sound | it |
| `jade.mp4` + `jade.wav` (camera plus their own mic) | the `.wav` only. The camera's sound is ignored, so there's no echo | the `.mp4` |
| `jade.wav` | it | none (the camera holds on whoever was last on screen) |
| `wide.mp4` with no sound | none | it, but only with `--layout side-by-side` or `--camera wide` |

A person can have at most one video file and one audio file. If a file's audio
starts slightly before or after its video (some recorders do this), the
difference is read from the file and corrected.

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
  there to help you find it, and they don't change as you edit. During
  crosstalk a speaker line may show tenths (`[00:01:12.4]`), which keeps every
  line unique.
- A line starting with `#` is ordinary speech (Whisper writes things like `#1`).
  Only the title line at the top counts as a heading.
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

Fix it by putting the words back, or deleting the suggested larger phrase. You
can also re-run `transcribe` with a lower `--min-quiet`, which keeps your edits.
`render` refuses to run while problems remain. `render --keep-unclean` renders
anyway and leaves those words in.

`check` and `render` also stop when an edit would remove more than half the
episode, or delete a long stretch at the end (over 10 min, or 20% of the
episode). That is usually an LLM whose reply was cut off partway through the
file. Pass `--allow-big-cut` if you meant it.

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
| `render --min-shot 3` | With the default speaker layout: stay at least 3 s on a camera before switching (default 2) |
| `render --layout side-by-side` | Show all cameras at once instead of switching to whoever is talking |
| `render --camera alice` | Show only Alice's camera throughout |
| `render --no-normalize` | Keep everyone at their recorded level instead of evening out their speech |
| `render --fast` | Hardware H.264 encoder: several times faster, larger file, a little lower quality |
| `transcribe --model mlx-community/whisper-large-v3-mlx` | Slower, sometimes more accurate model |
| `transcribe --language ''` | Auto-detect the language (default `en`) |
| `transcribe --min-quiet 0.03` | Shortest pause (s) that counts as a clean cut point. Lower means more cuttable, with more risk of a clipped word |
| `transcribe --quiet-db -45` | Override the auto-detected "quiet" level |
| `render --allow-big-cut` | Allow removing over half the episode, or a long stretch at the end |
| `transcribe --force` | Start `transcript.md` afresh, discarding your edits. Not needed just to change `--min-quiet`/`--quiet-db` |

## Output

- **Video layout.** By default the picture **switches to whoever is talking**.
  A speaker's camera is the video file with the same name, so `alice` talking
  shows `alice.mp4`. The switch happens when someone starts a turn of at least
  1 s, so a quick "yeah" doesn't move the camera. Every shot also lasts at least
  `--min-shot` seconds (default 2). In silence, or while someone without a
  camera talks, the shot stays where it is. Shots are planned on the edited
  timeline, so cuts never leave a flash of the wrong camera. The frame is the
  size of the biggest camera (by area), and the other cameras are letterboxed
  to fit.
  `--layout side-by-side` shows all cameras at once instead, each centre-cropped
  to an equal slice, ordered by file name. With one camera, or `--layout
  single`, that camera fills the frame throughout. Audio-only episodes show
  `cover.*` if present, otherwise black, at the intro's size or 1920×1080.
- **Audio.** Everyone is evened out to the same speech level. Each person's
  level is measured only while they are talking (their own transcript words,
  so bleed and the other person's turns don't count), and their sound is
  turned up or down to -20 dBFS (about -19 LUFS), by at most 24 dB. `render`
  prints each person's measured level and gain. A lookahead limiter on the mix
  then holds every peak under -1 dBFS, so a boosted laugh or cough never
  clips, while normal speech passes through untouched. `--no-normalize` mixes
  everyone at their recorded level instead (the limiter still applies). The
  intro and outro keep their own levels. Output is 48 kHz stereo, AAC 256k.
  Each cut has a 5 ms fade so it doesn't click.
- **Quality.** H.264 at CRF 18 (visually lossless), at the source resolution
  and frame rate. A variable frame rate (phones, some webcams) is rounded to
  the nearest standard rate. Cuts land exactly on frame boundaries, so sound
  and picture stay in sync however many cuts there are. A test checks this
  with a flash and a beep.

## How it works

1. **transcribe.** Each track is recognised separately with word timestamps
   (Whisper is prompted with disfluencies so it keeps "um"/"uh"). Words that
   another mic picked up louder (bleed), and words recognised in near-silence
   (hallucinations), are dropped. The rest are merged in time order. The mixed
   audio is then measured in 10 ms frames. Between every pair of consecutive
   words, the tool looks for a stretch of at least `min-quiet` seconds that is
   quiet relative to the speech level and contains a video frame boundary. If
   one exists, the frame boundary nearest its middle is the cut point for that
   boundary. If not, the boundary gets a `‿`. (Audio-only episodes cut to the
   millisecond.) Everything is
   saved in `work/transcript.json`. `transcript.md` is generated from it and
   holds no timings at all.
2. **check / render.** The edited `transcript.md` is aligned back to the
   original word list: an exact longest-common-subsequence match, done
   paragraph by paragraph using the speaker lines as anchors. Words with no
   match were deleted. Because timings are looked up in the untouched JSON,
   deleting text never shifts any offsets. Each run of deleted words is removed
   from the cut point before it to the cut point after it, which leaves about
   one natural pause. The kept ranges are on video frames. The audio is
   spliced in a streaming pass, and ffmpeg then trims the video, stacks the
   cameras, adds intro/outro and encodes, all in one pass with a progress bar.

Re-running `transcribe` reuses the cached recognition (`work/whisper-*.json`),
so trying new `--min-quiet`/`--quiet-db` values is quick. If you have already
edited `transcript.md`, your deletions are kept and only the `‿` marks are
updated. Spelling fixes are reset, and the previous file is saved in `work/`.
A different `--model`, `--language` or set of input files would change the
words themselves. That is refused up front unless you pass `--force`.

## Limitations

- Only words can be cut. A cough, or an "um" that Whisper didn't write down,
  can't be removed from the text. Use `--max-gap` for long silences.
- Whisper's word timings are approximate (often ±0.1 s). The quiet-point
  search tolerates that, and caps any "word" longer than 1.2 s so a stretched
  ending can't hide the pause after it. If a cut clips a word, raise
  `--min-quiet`.
- Crosstalk still shows as many short paragraphs, one each time the speaker
  changes.
- Rendering decodes every camera at full size, even for footage that's cut.
  Very long recordings with thousands of cuts render more slowly (see the
  `select` cost in the code).
- Repeated identical words in one paragraph ("I I think") can't be told apart
  by text. Either copy may be the one removed. That's normally harmless.

## Development

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest                                    # fast pass (~10 s, uses real ffmpeg)
.venv/bin/python -m pytest --cov=podedit --cov-branch --cov-report=json:coverage.json \
  && .venv/bin/python tools/crap.py coverage.json             # CRAP <= 12 per function
HYPOTHESIS_PROFILE=fuzz .venv/bin/python -m pytest              # slow pass: 20k examples per property (~15 min)
```

Code is in `podedit/`: `transcribe` (recognition + analysis), `analysis` (cut
points, bleed filter), `doc` + `align` (markdown and edit alignment), `edl`
(deleted words → keep ranges), `loudness` (per-person speech levels), `shots` (active-speaker camera switching),
`splice` (audio), `render` (ffmpeg graph), `cli`.
