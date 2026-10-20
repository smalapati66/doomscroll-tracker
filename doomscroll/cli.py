"""Command-line entry point:  .venv/bin/python -m doomscroll <command> ...

Stages (each reads the previous stage's files from data/<video stem>/):
  probe      inspect a video's resolution, timing, rotation and timestamps
  calibrate  click phone-screen corners -> data/<stem>/calibration.json
  track      MediaPipe hand tracking     -> data/<stem>/tracking_raw.npz (+ tracking_meta.json)
  detect     hand selection, mapping, smoothing, upward-gesture detection
                                         -> data/<stem>/thumb_track.csv, gestures.json, detect_summary.json
  diagnose   diagnostic overlay video    -> diagnostics/<stem>_diag_<range>.mp4
  render     abstract pigment video      -> renders/<name>/pigment.mp4 (+ exact config + timing)
  synth      SYNTHETIC gestures for renderer testing only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .common import PROJECT_ROOT, PipelineError, data_dir_for, load_config


def _data_dir(args) -> Path:
    return Path(args.data_dir) if getattr(args, "data_dir", None) else data_dir_for(args.video)


def cmd_probe(args):
    from .video import probe, scan_timestamps
    info = probe(args.video)
    t = scan_timestamps(args.video)
    d = np.diff(t)
    report = info.to_dict()
    report["timestamps"] = {
        "n_frames_from_packets": int(len(t)),
        "first_s": float(t[0]), "last_s": float(t[-1]),
        "frame_interval_s": {"min": float(d.min()), "median": float(np.median(d)), "max": float(d.max())},
        "constant_frame_rate": bool(np.ptp(d) < 1e-4),
        "intervals_over_1.5x_median": int((d > 1.5 * np.median(d)).sum()),
    }
    print(json.dumps(report, indent=2))


def cmd_calibrate(args):
    from .calibration import (DEFAULT_ASPECT, interactive_calibrate, make_segment, parse_corners,
                              save_calibration, save_preview_images)
    from .video import probe
    info = probe(args.video)
    out = Path(args.out) if args.out else _data_dir(args) / "calibration.json"
    t = args.time if args.time is not None else (args.append_at if args.append_at is not None else info.duration_s / 2)
    aspect = args.aspect or DEFAULT_ASPECT
    if args.corners:
        corners, frame_t = parse_corners(args.corners), t
    else:
        print("opening calibration window (it may appear behind your terminal)...")
        res = interactive_calibrate(args.video, t, args.margin, args.display_scale, aspect)
        if res is None:
            print("calibration cancelled; nothing saved")
            return
        corners, frame_t = res
    start = args.append_at if args.append_at is not None else 0.0
    seg = make_segment(corners, start, frame_t)
    save_calibration(out, args.video, seg, aspect, append=args.append_at is not None)
    imgs = save_preview_images(args.video, seg, aspect, out.parent / "calibration_previews", frame_t)
    print(f"saved {out}")
    print("corners (TL, TR, BR, BL px): " + "; ".join(f"{x:.1f},{y:.1f}" for x, y in seg["corners_px"]))
    for p in imgs:
        print(f"preview: {p}")


def cmd_track(args):
    from .tracking import run_tracking
    cfg = load_config(args.config)
    run_tracking(args.video, _data_dir(args), cfg, args.start, args.end)


def cmd_detect(args):
    from .pipeline import run_detect
    cfg = load_config(args.config)
    run_detect(args.video, _data_dir(args), cfg, args.calibration)


def cmd_diagnose(args):
    from .diagnostics import make_contact_sheet, make_diagnostic
    cfg = load_config(args.config)
    if args.contact_sheet:
        make_contact_sheet(args.video, _data_dir(args), cfg, args.out, args.calibration,
                           start_s=args.start, end_s=args.end)
        return
    make_diagnostic(args.video, _data_dir(args), cfg, args.start, args.end, args.out,
                    audio=not args.no_audio, calibration_path=args.calibration)


def cmd_render(args):
    from .pigment import render_from_cli
    cfg = load_config(args.config)
    render_from_cli(args, cfg)


def cmd_synth(args):
    from .synthetic import write_synthetic
    write_synthetic(Path(args.out), args.duration, args.interval, args.seed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m doomscroll", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, video=True):
        if video:
            sp.add_argument("--video", required=True, help="source recording, e.g. footage/test.mov")
        sp.add_argument("--data-dir", help="override data/<video stem>/")
        sp.add_argument("--config", help="TOML overriding config/default.toml")

    sp = sub.add_parser("probe", help="inspect video metadata and timestamps")
    sp.add_argument("--video", required=True)
    sp.set_defaults(fn=cmd_probe)

    sp = sub.add_parser("calibrate", help="select phone screen corners")
    common(sp)
    sp.add_argument("--time", type=float, help="source time (s) of the frame to show (default: middle)")
    sp.add_argument("--corners", help="non-interactive: 'x,y;x,y;x,y;x,y' in order TL;TR;BR;BL (image px)")
    sp.add_argument("--aspect", type=float, help="screen width/height (default 9/19.5 for iPhone)")
    sp.add_argument("--append-at", type=float,
                    help="add a calibration segment valid from this source time (phone moved) instead of replacing")
    sp.add_argument("--margin", type=int, default=160, help="gray border (px) so corners can lie outside the frame")
    sp.add_argument("--display-scale", type=float, default=0.75, help="window zoom")
    sp.add_argument("--out", help="calibration path (default data/<stem>/calibration.json)")
    sp.set_defaults(fn=cmd_calibrate)

    sp = sub.add_parser("track", help="run MediaPipe and cache raw hand landmarks")
    common(sp)
    sp.add_argument("--start", type=float, default=0.0, help="source seconds")
    sp.add_argument("--end", type=float, help="source seconds (default: end of video)")
    sp.set_defaults(fn=cmd_track)

    sp = sub.add_parser("detect", help="detect upward gestures from cached tracking (fast, rerunnable)")
    common(sp)
    sp.add_argument("--calibration", help="default data/<stem>/calibration.json")
    sp.set_defaults(fn=cmd_detect)

    sp = sub.add_parser("diagnose", help="render the diagnostic overlay video")
    common(sp)
    sp.add_argument("--start", type=float, default=0.0)
    sp.add_argument("--end", type=float, help="default: end of video")
    sp.add_argument("--out", help="output .mp4")
    sp.add_argument("--no-audio", action="store_true", help="do not copy source audio")
    sp.add_argument("--contact-sheet", action="store_true",
                    help="instead of a video, write one image tile per accepted gesture for quick review")
    sp.add_argument("--calibration", help="default data/<stem>/calibration.json")
    sp.set_defaults(fn=cmd_diagnose)

    sp = sub.add_parser("render", help="render the abstract pigment video from gestures JSON")
    sp.add_argument("--gestures", help="gestures JSON (default data/<stem>/gestures.json via --video)")
    sp.add_argument("--video", help="source video (only used to locate data/<stem>/)")
    sp.add_argument("--config", help="TOML overriding config/default.toml")
    sp.add_argument("--start", type=float, default=0.0, help="source seconds where the output begins")
    sp.add_argument("--end", type=float, help="source seconds where the output ends (default: source end)")
    sp.add_argument("--no-history", action="store_true",
                    help="with --start > 0: begin from a clean blue canvas instead of simulating 0..start")
    sp.add_argument("--width", type=int, help="override render.width (preview)")
    sp.add_argument("--height", type=int, help="override render.height (preview)")
    sp.add_argument("--fps", type=float, help="override render.fps")
    sp.add_argument("--speed", type=float, default=1.0,
                    help="ACCELERATED preview factor (>1 is labeled in file name and on frame)")
    sp.add_argument("--name", help="output folder name under renders/")
    sp.add_argument("--out-dir", help="explicit output folder")
    sp.add_argument("--stills", type=str, default="",
                    help="comma-separated output times (s) to also save as PNG, e.g. '5,30,60'")
    sp.set_defaults(fn=cmd_render)

    sp = sub.add_parser("synth", help="write a SYNTHETIC gestures file to test the renderer (not tracked data)")
    sp.add_argument("--out", default=str(PROJECT_ROOT / "data" / "synthetic" / "gestures_SYNTHETIC.json"))
    sp.add_argument("--duration", type=float, default=1800.0, help="seconds")
    sp.add_argument("--interval", type=float, default=4.5, help="mean seconds between swipes")
    sp.add_argument("--seed", type=int, default=1)
    sp.set_defaults(fn=cmd_synth)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except PipelineError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0
