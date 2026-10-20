"""Phone-screen calibration: image pixels -> normalized phone coordinates.

Normalized phone coordinates: x = 0 at the phone screen's left edge, 1 at its right edge;
y = 0 at the top edge, 1 at the bottom edge (as the phone's own UI is oriented). Upward swipes
therefore have *decreasing* y.

Corners are clicked in the order top-left, top-right, bottom-right, bottom-left *of the phone
screen* (as the phone's UI is oriented), in display-oriented camera image pixels. Because the
corner order is defined on the phone, any mirroring of the camera image is absorbed into the
homography: the mapping stays correct either way, and we report whether the clicked order looks
mirrored.

The mapping assumes the phone is stationary. If it moves, the fixed mapping is wrong. You can add
manually calibrated segments that take effect from a given source time (`calibrate --append-at T`);
nothing is tracked automatically.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import cv2
import numpy as np

from .common import PipelineError, file_fingerprint, read_json, write_json
from .video import frame_at, probe

CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]
UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
DEFAULT_ASPECT = 9.0 / 19.5  # iPhone screens (width / height); adjust with --aspect
CALIBRATION_VERSION = 1


def validate_corners(corners) -> tuple[np.ndarray, bool]:
    """Return (corners Nx2 float, mirrored?) or raise PipelineError."""
    c = np.asarray(corners, dtype=np.float64)
    if c.shape != (4, 2) or not np.all(np.isfinite(c)):
        raise PipelineError("calibration needs exactly 4 finite corner points (TL, TR, BR, BL)")
    # signed cross products of consecutive edges must all share a sign (convex, no self-intersection)
    cross = []
    for i in range(4):
        a, b, d = c[i], c[(i + 1) % 4], c[(i + 2) % 4]
        e1, e2 = b - a, d - b
        cross.append(e1[0] * e2[1] - e1[1] * e2[0])
    cross = np.array(cross)
    if not (np.all(cross > 0) or np.all(cross < 0)):
        raise PipelineError(
            "calibration corners do not form a convex quadrilateral in the order "
            "top-left, top-right, bottom-right, bottom-left. Redo calibration.")
    area = 0.5 * abs(np.dot(c[:, 0], np.roll(c[:, 1], -1)) - np.dot(c[:, 1], np.roll(c[:, 0], -1)))
    if area < 400:
        raise PipelineError(f"calibration quadrilateral is degenerate (area {area:.0f} px^2). Redo calibration.")
    # In image coords (y down) a clockwise TL->TR->BR->BL order gives positive cross products.
    mirrored = bool(np.all(cross < 0))
    return c, mirrored


def make_segment(corners, start_s: float, frame_t: float) -> dict:
    c, mirrored = validate_corners(corners)
    H = cv2.getPerspectiveTransform(c.astype(np.float32), UNIT_SQUARE.astype(np.float32)).astype(np.float64)
    Hinv = np.linalg.inv(H)
    return {
        "start_s": float(start_s),
        "frame_time_s": float(frame_t),
        "corners_px": c.tolist(),
        "corner_order": CORNER_NAMES,
        "homography_px_to_phone": H.tolist(),
        "homography_phone_to_px": Hinv.tolist(),
        "image_looks_mirrored": mirrored,
    }


class Calibration:
    def __init__(self, data: dict, path: str | Path | None = None):
        self.data, self.path = data, path
        segs = sorted(data["segments"], key=lambda s: s["start_s"])
        if not segs:
            raise PipelineError(f"calibration {path} has no segments")
        self.starts = np.array([s["start_s"] for s in segs])
        self.H = [np.array(s["homography_px_to_phone"]) for s in segs]
        self.Hinv = [np.array(s["homography_phone_to_px"]) for s in segs]
        self.corners = [np.array(s["corners_px"]) for s in segs]
        for s in segs:
            validate_corners(s["corners_px"])
        self.aspect = float(data.get("screen_aspect", DEFAULT_ASPECT))
        self.image_size = tuple(data["image_size"])

    def segment_index(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        return np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, len(self.starts) - 1)

    @staticmethod
    def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        homog = np.c_[pts, np.ones(len(pts))] @ H.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return homog[:, :2] / homog[:, 2:3]

    def to_phone(self, t, pts_px) -> np.ndarray:
        """Map image pixels (N,2) at source times t (N,) to normalized phone coords; NaN stays NaN."""
        pts_px = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        seg = self.segment_index(t) if np.ndim(t) else np.full(len(pts_px), self.segment_index(t)[0])
        out = np.full_like(pts_px, np.nan)
        for k in np.unique(seg):
            m = seg == k
            out[m] = self._apply(self.H[k], pts_px[m])
        return out

    def to_image(self, t: float, pts_phone) -> np.ndarray:
        k = int(self.segment_index(t)[0])
        return self._apply(self.Hinv[k], pts_phone)

    def corners_at(self, t: float) -> np.ndarray:
        return self.corners[int(self.segment_index(t)[0])]

    def summary(self) -> dict:
        return {"path": str(self.path) if self.path else None, "n_segments": len(self.starts),
                "segment_starts_s": self.starts.tolist(), "screen_aspect": self.aspect}


def load_calibration(path: str | Path) -> Calibration:
    data = read_json(path, "calibration file",
                     "run:  .venv/bin/python -m doomscroll calibrate --video footage/<file>.mov")
    if data.get("version") != CALIBRATION_VERSION or "segments" not in data or "image_size" not in data:
        raise PipelineError(f"calibration file {path} has an unexpected format; redo calibration")
    return Calibration(data, path)


def save_calibration(path: Path, video: str, segment: dict, aspect: float, append: bool) -> dict:
    info = probe(video)
    if append and path.is_file():
        data = read_json(path, "calibration file")
        if list(data["image_size"]) != [info.width, info.height]:
            raise PipelineError("existing calibration was made on a different image size; cannot append")
        data["segments"] = [s for s in data["segments"] if abs(s["start_s"] - segment["start_s"]) > 1e-6]
        data["segments"].append(segment)
        data["segments"].sort(key=lambda s: s["start_s"])
    else:
        if append:
            segment["start_s"] = 0.0  # first segment always covers the start of the video
        data = {
            "version": CALIBRATION_VERSION,
            "source_video": file_fingerprint(video),
            "image_size": [info.width, info.height],
            "image_orientation": f"display orientation (stream rotation {info.rotation_deg} deg applied)",
            "coordinate_system": "phone: x 0..1 left->right, y 0..1 top->bottom (phone UI orientation)",
            "screen_aspect": aspect,
            "assumption": "phone stationary within each segment; moving the phone invalidates the mapping",
            "segments": [segment],
        }
    data["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
    if any(s["image_looks_mirrored"] for s in data["segments"]):
        print("note: corner order is counter-clockwise in the image, i.e. the camera image looks mirrored. "
              "The homography accounts for this.")
    write_json(path, data)
    return data


def parse_corners(text: str) -> list[list[float]]:
    try:
        pts = [[float(v) for v in p.split(",")] for p in text.replace(" ", "").split(";") if p]
    except ValueError as e:
        raise PipelineError(f"could not parse --corners '{text}': use 'x,y;x,y;x,y;x,y'") from e
    if len(pts) != 4 or any(len(p) != 2 for p in pts):
        raise PipelineError("--corners needs 4 points 'x,y;x,y;x,y;x,y' in order TL;TR;BR;BL")
    return pts


def rectify(img_bgr: np.ndarray, segment: dict, aspect: float, out_h: int = 640) -> np.ndarray:
    out_w = max(1, int(round(out_h * aspect)))
    H = np.array(segment["homography_px_to_phone"])
    S = np.diag([out_w, out_h, 1.0])
    return cv2.warpPerspective(img_bgr, S @ H, (out_w, out_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))


# ----------------------------------------------------------------------------- interactive UI

HELP_LINES = [
    "Click phone SCREEN corners in order: TL, TR, BR, BL (gray margin = outside the frame)",
    "u: undo   r: reset   , / . : -/+1 s   [ / ] : -/+10 s   (corners stay while you scrub)",
    "Enter or s: save (after checking preview)   q / Esc: quit without saving",
]


def interactive_calibrate(video: str, t: float, margin: int, display_scale: float, aspect: float,
                          initial: list | None = None) -> tuple[list, float] | None:
    """Return (corners, frame_time) or None if cancelled."""
    info = probe(video)
    win, pwin = "calibrate: phone screen corners", "rectified preview (phone coords)"
    corners: list[list[float]] = [list(p) for p in initial] if initial else []
    state = {"t": t}

    def load(tt):
        tt = float(np.clip(tt, 0, max(0.0, info.duration_s - 0.1)))
        fr = frame_at(video, tt, fmt="bgr24")
        state["t"], state["frame"] = fr.t, fr.image

    load(t)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append([x / display_scale - margin, y / display_scale - margin])

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    preview_shown = False
    try:
        while True:
            img = state["frame"]
            canvas = cv2.copyMakeBorder(img, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=(70, 70, 70))
            pts = [(int(round(x + margin)), int(round(y + margin))) for x, y in corners]
            for i, p in enumerate(pts):
                cv2.circle(canvas, p, 6, (0, 255, 255), -1)
                cv2.putText(canvas, ["TL", "TR", "BR", "BL"][i], (p[0] + 8, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if len(pts) >= 2:
                cv2.polylines(canvas, [np.array(pts)], len(pts) == 4, (0, 255, 0), 2)
            nxt = CORNER_NAMES[len(corners)] if len(corners) < 4 else "done - check preview"
            lines = [f"t = {state['t']:.3f} s   next: {nxt}"] + HELP_LINES
            for i, line in enumerate(lines):
                cv2.putText(canvas, line, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(canvas, line, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            if display_scale != 1.0:
                canvas = cv2.resize(canvas, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(win, canvas)

            seg = None
            if len(corners) == 4:
                try:
                    seg = make_segment(corners, 0.0, state["t"])
                    cv2.imshow(pwin, rectify(img, seg, aspect))
                    preview_shown = True
                except Exception as e:  # invalid quad: tell the user, let them undo
                    print(f"\n{e}")
                    corners.pop()
            elif preview_shown:
                cv2.destroyWindow(pwin)
                preview_shown = False

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return None
            if key == ord("u") and corners:
                corners.pop()
            elif key == ord("r"):
                corners.clear()
            elif key in (ord(","), ord(".")):
                load(state["t"] + (-1 if key == ord(",") else 1))
            elif key in (ord("["), ord("]")):
                load(state["t"] + (-10 if key == ord("[") else 10))
            elif key in (13, 10, ord("s")) and seg is not None:
                return corners, state["t"]
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def save_preview_images(video: str, seg: dict, aspect: float, out_dir: Path, t: float) -> list[Path]:
    """Write the annotated frame and rectified preview as PNGs for later inspection."""
    fr = frame_at(video, t, fmt="bgr24")
    img = fr.image.copy()
    c = np.array(seg["corners_px"])
    cv2.polylines(img, [np.round(c).astype(np.int32)], True, (0, 255, 0), 2)
    for name, p in zip(["TL", "TR", "BR", "BL"], c):
        cv2.circle(img, tuple(np.round(p).astype(int)), 5, (0, 255, 255), -1)
        cv2.putText(img, name, (int(p[0]) + 6, int(p[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    out_dir.mkdir(parents=True, exist_ok=True)
    p1, p2 = out_dir / f"calibration_frame_{t:07.2f}s.png", out_dir / f"calibration_rectified_{t:07.2f}s.png"
    cv2.imwrite(str(p1), img)
    cv2.imwrite(str(p2), rectify(fr.image, seg, aspect))
    return [p1, p2]
