"""`pod` command line: new, transcribe, check, render, status."""

from __future__ import annotations

import argparse
import json
import sys

from . import render as rnd
from .doc import timestamp
from .paths import EPISODES, TEMPLATES, Episode, find_template, find_tracks, resolve_episode


def load(ep: Episode) -> tuple[dict, str]:
    if not ep.transcript_json.is_file() or not ep.transcript_md.is_file():
        raise SystemExit(f"No transcript yet for {ep.name}. Run: ./pod transcribe {ep.name}")
    return json.loads(ep.transcript_json.read_text()), ep.transcript_md.read_text()


def cmd_new(args) -> int:
    ep = Episode(EPISODES / args.episode)
    ep.create()
    print(f"Created {ep.root}\nDrop the recordings (one file per person, e.g. jade.mp4, guest.mp4) into:\n  {ep.input_dir}")
    print(f"Then run: ./pod transcribe {ep.name}")
    return 0


def cmd_transcribe(args) -> int:
    from . import transcribe as tr  # heavy imports only when needed

    ep = resolve_episode(args.episode)
    ep.work_dir.mkdir(parents=True, exist_ok=True)
    transcript = tr.build(ep, model=args.model, language=args.language or None,
                          min_quiet=args.min_quiet, quiet_db=args.quiet_db)
    tr.write_outputs(ep, transcript, force=args.force)
    print(tr.summary(transcript))
    print(f"Edit: {ep.transcript_md}\nThen: ./pod check {ep.name}   and   ./pod render {ep.name}")
    return 0


def report(plan: rnd.EditPlan, duration: float) -> None:
    for w in plan.warnings:
        print(f"note: {w}")
    for p in plan.problems:
        print(f"CAN'T CUT: {p.message}")
    print(f"\n{plan.deleted} words cut cleanly; {len(plan.problems)} cut(s) not possible; "
          f"{timestamp(duration)} -> {timestamp(plan.kept_seconds)} "
          f"({duration - plan.kept_seconds:.1f}s removed).")


def plan_from(args) -> tuple[Episode, dict, rnd.EditPlan]:
    ep = resolve_episode(args.episode)
    transcript, md = load(ep)
    plan = rnd.plan_edit(transcript, md, keep_unclean=args.keep_unclean, max_gap=args.max_gap)
    report(plan, transcript["duration"])
    return ep, transcript, plan


def cmd_check(args) -> int:
    _, _, plan = plan_from(args)
    return 1 if plan.problems else 0


def segment(ep: Episode, stem: str, disabled: bool):
    return None if disabled else find_template(ep, stem)


def cmd_render(args) -> int:
    ep, transcript, plan = plan_from(args)
    if plan.problems:
        print("\nNot rendering. Fix the lines above in transcript.md (restore those words, or delete the "
              "suggested larger phrase), or re-run with --keep-unclean to leave them in.", file=sys.stderr)
        return 1
    intro, outro = segment(ep, "intro", args.no_intro), segment(ep, "outro", args.no_outro)
    cover = find_template(ep, "cover")
    for label, path in (("intro", intro), ("outro", outro)):
        print(f"{label}: {path or 'none'}")
    out = rnd.render(ep, transcript, plan, layout=args.layout, camera=args.camera, intro=intro, outro=outro,
                     cover=cover, fast=args.fast, min_shot=args.min_shot)
    print(f"Done: {out}")
    return 0


def cmd_status(args) -> int:
    eps = sorted(d for d in EPISODES.iterdir() if d.is_dir()) if EPISODES.is_dir() else []
    for d in eps:
        ep = Episode(d)
        state = ("rendered" if ep.output_mp4.exists() else
                 "transcribed" if ep.transcript_md.exists() else
                 "has input" if find_tracks(ep) else "empty")
        print(f"{ep.name:40s} {state}")
    print(f"\ntemplates ({TEMPLATES}): " + (", ".join(sorted(f.name for f in TEMPLATES.iterdir()
                                                            if not f.name.startswith("."))) or "none"))
    return 0


def edit_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("episode", help="episode directory name under episodes/, or a path")
    p.add_argument("--keep-unclean", action="store_true",
                   help="leave in deleted words that can't be cut cleanly instead of stopping")
    p.add_argument("--max-gap", type=float, metavar="SECONDS",
                   help="also shorten every pause longer than this (e.g. 1.0)")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pod", description="Edit podcast video by editing its transcript.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("new", help="create an episode directory")
    p.add_argument("episode")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("transcribe", help="recognise speech and write transcript.md")
    p.add_argument("episode")
    p.add_argument("--model", default="mlx-community/whisper-large-v3-turbo")
    p.add_argument("--language", default="en", help="language code, or '' to auto-detect")
    p.add_argument("--min-quiet", type=float, default=0.04, help="shortest pause (s) that allows a clean cut")
    p.add_argument("--quiet-db", type=float, help="override the auto-detected quiet level (dBFS)")
    p.add_argument("--force", action="store_true", help="overwrite an edited transcript.md")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("check", help="show what the edited transcript will cut, and any problems")
    edit_options(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("render", help="render the edited episode to output/<episode>.mp4")
    edit_options(p)
    p.add_argument("--layout", choices=["speaker", "side-by-side", "single"], default="speaker",
                   help="speaker: show whoever is talking (default); side-by-side: all cameras at once; "
                        "single: one camera throughout")
    p.add_argument("--camera", help="with --layout single: which track's video to use")
    p.add_argument("--min-shot", type=float, default=2.0, metavar="SECONDS",
                   help="with --layout speaker: shortest time on one camera before switching (default 2)")
    p.add_argument("--no-intro", action="store_true")
    p.add_argument("--no-outro", action="store_true")
    p.add_argument("--fast", action="store_true", help="hardware encoder: much faster, somewhat larger/lower quality")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("status", help="list episodes and templates")
    p.set_defaults(func=cmd_status)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
