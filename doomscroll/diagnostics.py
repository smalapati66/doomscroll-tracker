"""Diagnostic video: source footage + calibration + thumb tracking + detector state.

Left: original camera frame with the calibrated screen boundary, detected hands, the raw (hollow)
and smoothed (filled) thumb point, a short trail, and status text.
Right: phone-coordinate panel (rectified screen, dimmed) with the same trail in calibrated
coordinates, so you can check that upward thumb motion really is upward (decreasing y).

Trail colours (final classification, known after the fact):
  gray = idle / not part of any candidate   orange = candidate that was rejected
  green = accepted stroke                   (no line is ever drawn across missing samples)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .calibration import load_calibration, rectify
from .common import PROJECT_ROOT, SWIPE_PROXY_DISCLAIMER, PipelineError, Progress
from .encode import VideoWriter, mux_source_audio
from .gestures import L_ACCEPTED, L_IDLE, L_OUTSIDE, L_REJECTED, S_CANDIDATE, S_IDLE, S_STROKE
from .pipeline import process
from .tracking import load_tracking
from .trackproc import JUMP_REJECTED, NO_HAND, OK, OTHER_HAND_ONLY, UNMAPPABLE
from .video import iter_frames

# BGR colours (we draw on BGR images)
C_IDLE, C_REJ, C_ACC, C_OUT = (150, 150, 150), (0, 165, 255), (60, 220, 60), (200, 120, 200)
C_SCREEN, C_LOST, C_TEXT = (0, 255, 0), (40, 40, 255), (255, 255, 255)
LABEL_COLOR = {L_IDLE: C_IDLE, L_REJECTED: C_REJ, L_ACCEPTED: C_ACC, L_OUTSIDE: C_OUT}
STATE_COLOR = {S_IDLE: C_IDLE, S_CANDIDATE: C_REJ, S_STROKE: C_ACC}
STATE_TEXT = {S_IDLE: "IDLE", S_CANDIDATE: "CANDIDATE", S_STROKE: "STROKE (accepted)"}
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
TRAIL_S = 0.7
PANEL_H = 600


def _text(img, s, org, scale=0.55, color=C_TEXT, thick=1):
    # Outline by offsetting the same-thickness glyphs (Hershey advance width depends on thickness,
    # so a thicker outline would drift out of alignment).
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2), (2, 0), (-2, 0), (0, -2)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _ip(p):
    return int(round(p[0])), int(round(p[1]))


def make_diagnostic(video: str, data_dir: Path, cfg: dict, start_s: float, end_s: float | None,
                    out: str | None, audio: bool = True, calibration_path: str | None = None) -> Path:
    raw, meta = load_tracking(data_dir, video)
    calib_path = Path(calibration_path) if calibration_path else data_dir / "calibration.json"
    calib = load_calibration(calib_path)
    P = process(raw, calib, cfg)
    t_all, pts_all = P["t"], P["pts"]
    idx_of_pts = {int(p): i for i, p in enumerate(pts_all)}
    vi = meta["video_info"]
    end_s = min(end_s, vi["duration_s"]) if end_s is not None else vi["duration_s"]
    if start_s >= end_s:
        raise PipelineError(f"empty range: start {start_s} >= end {end_s}")
    W, Hh = vi["width"], vi["height"]
    panel_w = int(round(PANEL_H * calib.aspect))
    out_w, out_h = W + panel_w + 40, max(Hh, PANEL_H + 80)
    out_w, out_h = out_w + out_w % 2, out_h + out_h % 2
    out_path = Path(out) if out else PROJECT_ROOT / "diagnostics" / f"{Path(video).stem}_diag_{start_s:.0f}-{end_s:.0f}s.mp4"
    tmp_path = out_path.with_name(out_path.stem + "_noaudio.mp4")
    from fractions import Fraction
    tb = Fraction(vi["time_base"])
    writer = VideoWriter(tmp_path if audio and vi["has_audio"] else out_path, out_w, out_h,
                         fps=vi["avg_fps"], time_base=tb)
    accepted_start_times = np.array(sorted(t_all[s.indices[0]] for s in P["strokes"] if s.accepted))
    strokes_acc = [s for s in P["strokes"] if s.accepted]
    stroke_of_idx = {}
    for gid, s in enumerate(strokes_acc, 1):
        for i in s.indices:
            stroke_of_idx[i] = gid
    tol = float(cfg["detect"]["boundary_tolerance"])
    tol_quad = np.array([[-tol, -tol], [1 + tol, -tol], [1 + tol, 1 + tol], [-tol, 1 + tol]])
    start_pts = int(round(start_s / float(tb))) + vi["start_pts"]

    n_expected = int(((t_all >= start_s) & (t_all < end_s)).sum())
    prog = Progress("diagnostic", n_expected or None)
    lm = raw["landmarks_px"]
    for fr in iter_frames(video, start_s, end_s, fmt="bgr24"):
        img = fr.image.copy()
        i = idx_of_pts.get(int(fr.pts))
        canvas = np.full((out_h, out_w, 3), 25, np.uint8)
        # --- calibration overlay
        quad = calib.corners_at(fr.t)
        cv2.polylines(img, [np.round(quad).astype(np.int32)], True, C_SCREEN, 2, cv2.LINE_AA)
        tq = calib.to_image(fr.t, tol_quad)
        cv2.polylines(img, [np.round(tq).astype(np.int32)], True, (0, 120, 0), 1, cv2.LINE_AA)
        # --- phone panel background: rectified screen, dimmed
        seg_cal = {"homography_px_to_phone": calib.H[int(calib.segment_index(fr.t)[0])].tolist()}
        panel = (rectify(fr.image, seg_cal, calib.aspect, PANEL_H).astype(np.float32) * 0.35).astype(np.uint8)
        px = lambda q: (int(round(q[0] * (panel_w - 1))), int(round(q[1] * (PANEL_H - 1))))

        if i is None:
            _text(img, "frame not in tracking cache (run `track` over this range)", (10, Hh - 50), 0.6, C_LOST)
        else:
            # all detected hands, faint; selected hand brighter
            for k in range(lm.shape[1]):
                if not np.isfinite(lm[i, k, 0, 0]):
                    continue
                col = (230, 200, 60) if k == P["sel"][i] else (110, 110, 110)
                for a, b in HAND_EDGES:
                    cv2.line(img, _ip(lm[i, k, a]), _ip(lm[i, k, b]), col, 1, cv2.LINE_AA)
            # trail: consecutive valid samples in the same segment, last TRAIL_S seconds
            j0 = np.searchsorted(t_all, t_all[i] - TRAIL_S)
            for a in range(max(j0, 1), i + 1):
                b = a - 1
                if P["seg"][a] < 0 or P["seg"][b] != P["seg"][a]:
                    continue
                col = LABEL_COLOR.get(int(P["label"][a]), C_IDLE)
                pa_img = calib.to_image(t_all[a], P["phone"][a:a + 1])[0]
                pb_img = calib.to_image(t_all[b], P["phone"][b:b + 1])[0]
                cv2.line(img, _ip(pb_img), _ip(pa_img), col, 3, cv2.LINE_AA)
                cv2.line(panel, px(P["phone"][b]), px(P["phone"][a]), col, 3, cv2.LINE_AA)
            st = int(P["status"][i])
            if st == OK:
                state = int(P["online"][i])
                col = STATE_COLOR.get(state, C_IDLE)
                p_s = calib.to_image(fr.t, P["phone"][i:i + 1])[0]
                cv2.circle(img, _ip(P["thumb_px"][i]), 9, C_TEXT, 1, cv2.LINE_AA)  # raw
                cv2.circle(img, _ip(p_s), 6, col, -1, cv2.LINE_AA)                 # smoothed
                cv2.circle(panel, px(P["phone"][i]), 7, col, -1, cv2.LINE_AA)
                cv2.circle(panel, px(P["phone_raw"][i]), 9, C_TEXT, 1, cv2.LINE_AA)
                if int(P["label"][i]) == L_OUTSIDE:
                    _text(img, "OUTSIDE SCREEN REGION", (10, 130), 0.7, C_OUT, 2)
                state_txt = STATE_TEXT.get(state, "?")
            else:
                state_txt = "TRACKING LOST"
                cv2.rectangle(img, (0, 0), (W - 1, Hh - 1), C_LOST, 6)
                msg = {NO_HAND: "NO HAND DETECTED", OTHER_HAND_ONLY: "OTHER HAND ONLY (intended hand lost)",
                       JUMP_REJECTED: "JUMP REJECTED (implausible move)", UNMAPPABLE: "UNMAPPABLE POINT"}[st]
                _text(img, msg, (10, 130), 0.7, C_LOST, 2)
                if st == JUMP_REJECTED and np.all(np.isfinite(P["thumb_px"][i])):
                    x, y = _ip(P["thumb_px"][i])
                    cv2.line(img, (x - 10, y - 10), (x + 10, y + 10), C_LOST, 3)
                    cv2.line(img, (x - 10, y + 10), (x + 10, y - 10), C_LOST, 3)
            n_acc = int(np.searchsorted(accepted_start_times, fr.t, side="right"))
            gid = stroke_of_idx.get(i)
            _text(img, f"t = {fr.t:9.3f} s (source PTS)   frame {i}", (10, 26), 0.65)
            _text(img, f"detector: {state_txt}", (10, 54), 0.65, STATE_COLOR.get(int(P['online'][i]), C_LOST) if st == OK else C_LOST)
            _text(img, f"accepted gestures so far: {n_acc}" + (f"   [frame belongs to accepted stroke #{gid}]" if gid else ""), (10, 82), 0.65)
            _text(img, f"segment {int(P['seg'][i])}  v_up {P['v_up'][i]:+.2f} screen-h/s" if st == OK else "", (10, 106), 0.5)
        _text(img, "upward motion in screen region = swipe PROXY (not contact, not fingerprints)", (10, Hh - 14), 0.5, (200, 200, 200))

        canvas[:Hh, :W] = img
        x0, y0 = W + 20, 40
        canvas[y0:y0 + PANEL_H, x0:x0 + panel_w] = panel
        cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + panel_w, y0 + PANEL_H), C_SCREEN, 1)
        _text(canvas, "phone coords (y down)", (x0, 25), 0.5)
        cv2.arrowedLine(canvas, (x0 + panel_w - 15, y0 + 60), (x0 + panel_w - 15, y0 + 15), (230, 230, 230), 2, tipLength=0.3)
        _text(canvas, "up", (x0 + panel_w - 45, y0 + 75), 0.45)
        legend = [("idle", C_IDLE), ("rejected cand.", C_REJ), ("accepted", C_ACC)]
        for n, (s, c) in enumerate(legend):
            cv2.line(canvas, (x0, y0 + PANEL_H + 20 + 18 * n), (x0 + 25, y0 + PANEL_H + 20 + 18 * n), c, 3)
            _text(canvas, s, (x0 + 32, y0 + PANEL_H + 25 + 18 * n), 0.45)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), pts=int(fr.pts) - start_pts)
        prog.update()
    prog.done()
    writer.close()
    if audio and vi["has_audio"]:
        if mux_source_audio(tmp_path, video, start_s, end_s, out_path):
            tmp_path.unlink()
        else:
            tmp_path.replace(out_path)
    print(f"wrote {out_path}  ({writer.n} frames, {out_w}x{out_h}; t=0 in this file = source {start_s:.3f} s)")
    print(f"note: {SWIPE_PROXY_DISCLAIMER}")
    return out_path


def make_contact_sheet(video: str, data_dir: Path, cfg: dict, out: str | None = None,
                       calibration_path: str | None = None, cols: int = 6,
                       start_s: float = 0.0, end_s: float | None = None) -> Path:
    """One tile per accepted gesture: the frame at the stroke's end with the whole stroke path drawn
    (green, start dot hollow), cropped around the calibrated screen. For quick visual review."""
    raw, meta = load_tracking(data_dir, video)
    calib = load_calibration(Path(calibration_path) if calibration_path else data_dir / "calibration.json")
    P = process(raw, calib, cfg)
    numbered = [(gid, s) for gid, s in enumerate((s for s in P["strokes"] if s.accepted), 1)]
    numbered = [(gid, s) for gid, s in numbered
                if P["t"][s.indices[0]] >= start_s and (end_s is None or P["t"][s.indices[0]] < end_s)]
    acc = [s for _, s in numbered]
    if not acc:
        raise PipelineError("no accepted gestures to put on a contact sheet")
    tiles = []
    prog = Progress("contact sheet", len(acc))
    for gid, s in numbered:
        i0, i1 = s.indices[0], s.indices[-1]
        t0, t1 = float(P["t"][i0]), float(P["t"][i1])
        fr = next(iter_frames(video, t1, None, fmt="bgr24"))
        img = fr.image.copy()
        quad = calib.corners_at(t1)
        cv2.polylines(img, [np.round(quad).astype(np.int32)], True, C_SCREEN, 1, cv2.LINE_AA)
        path = calib.to_image(t1, P["phone"][s.indices])
        cv2.polylines(img, [np.round(path).astype(np.int32)], False, C_ACC, 3, cv2.LINE_AA)
        cv2.circle(img, _ip(path[0]), 7, C_TEXT, 2, cv2.LINE_AA)
        cv2.circle(img, _ip(path[-1]), 5, C_ACC, -1, cv2.LINE_AA)
        x0, y0 = np.floor(quad.min(0)).astype(int) - 60
        x1, y1 = np.ceil(quad.max(0)).astype(int) + 140
        H_, W_ = img.shape[:2]
        crop = img[max(0, y0):min(H_, y1), max(0, x0):min(W_, x1)]
        crop = cv2.resize(crop, (300, int(300 * crop.shape[0] / crop.shape[1])))
        _text(crop, f"#{gid} {t0:.2f}-{t1:.2f}s", (6, 20), 0.5)
        _text(crop, f"up {s.displacement:.2f}  v {s.peak_speed:.1f}  {s.end_reason}", (6, 40), 0.42)
        tiles.append(crop)
        prog.update()
    prog.done()
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 25))
    sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    rng = "" if (start_s == 0 and end_s is None) else f"_{start_s:.0f}-{end_s:.0f}s" if end_s else f"_{start_s:.0f}s-end"
    out_path = Path(out) if out else PROJECT_ROOT / "diagnostics" / f"{Path(video).stem}_gestures_contact_sheet{rng}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {out_path} ({len(acc)} gestures)")
    return out_path
