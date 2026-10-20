"""Synthetic-trajectory tests for gesture detection and pigment rendering.

Run:  .venv/bin/python -m pytest -q
These use generated thumb paths in phone coordinates; no footage or MediaPipe is involved.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from doomscroll.calibration import Calibration, make_segment, validate_corners
from doomscroll.common import PipelineError, load_config
from doomscroll.gestures import detect_strokes
from doomscroll.pigment import PigmentField, compute_stamps, scaled_config
from doomscroll.trackproc import JUMP_REJECTED, OK, segment_track, smooth_segments
from doomscroll.video import orient

ASPECT = 9 / 19.5
CFG = load_config()


# ----------------------------------------------------------------------------- helpers

def ease(u):
    u = np.clip(u, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def swipe_path(events, x=0.6, y_rest=0.8):
    """Continuous y(t) from a list of (t_start, duration, y_from, y_to) moves; holds in between."""
    def f(t):
        y = np.full_like(t, y_rest, dtype=float)
        for t0, d, a, b in sorted(events):
            y = np.where(t >= t0, a + (b - a) * ease((t - t0) / d), y)
        return np.c_[np.full_like(t, x), y]
    return f


def sample(f, duration, fps, jitter_frac=0.0, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fps)
    if jitter_frac:
        t = np.sort(t + rng.uniform(-jitter_frac, jitter_frac, len(t)) / fps)
    xy = f(t)
    if noise:
        xy = xy + rng.normal(0, noise, xy.shape)
    return t, xy


def run_detect(t, xy, cfg=CFG, drop=None):
    status = np.full(len(t), OK, np.int8)
    xy = xy.copy()
    if drop is not None:
        status[drop] = 1  # NO_HAND
        xy[drop] = np.nan
    seg, status = segment_track(t, xy, status, ASPECT, cfg["detect"])
    sm = smooth_segments(t, xy, seg, ASPECT, cfg["smoothing"])
    strokes, online, label, v = detect_strokes(t, sm, seg, cfg["detect"])
    return [s for s in strokes if s.accepted], strokes, seg, sm, status


def to_gestures(t, sm, strokes):
    return [{"id": k + 1, "points": [{"t": float(t[i]), "x": float(sm[i, 0]), "y": float(sm[i, 1])}
                                     for i in s.indices]} for k, s in enumerate(strokes)]


# ----------------------------------------------------------------------------- detection

def test_upward_gesture_is_accepted_and_keeps_its_beginning():
    # one 0.25 s swipe from y=0.8 to y=0.4 starting at t=1.0
    f = swipe_path([(1.0, 0.25, 0.8, 0.4), (1.6, 0.6, 0.4, 0.8)])
    t, xy = sample(f, 3.0, 30)
    acc, *_ = run_detect(t, xy)
    assert len(acc) == 1
    s = acc[0]
    t0, t1 = t[s.indices[0]], t[s.indices[-1]]
    assert 0.9 <= t0 <= 1.07, f"stroke should start near the true start (1.0 s), got {t0:.3f}"
    assert 1.2 <= t1 <= 1.35
    assert s.displacement > 0.3


def test_downward_return_alone_is_ignored():
    f = swipe_path([(0.5, 0.3, 0.3, 0.85)], y_rest=0.3)
    t, xy = sample(f, 2.0, 30)
    acc, all_s, *_ = run_detect(t, xy)
    assert acc == []


def test_downward_return_after_swipe_adds_no_points():
    f = swipe_path([(1.0, 0.25, 0.8, 0.4), (1.3, 0.3, 0.4, 0.8)])
    t, xy = sample(f, 2.5, 60)
    acc, _, _, sm, _ = run_detect(t, xy)
    assert len(acc) == 1
    ys = sm[acc[0].indices, 1]
    # stroke ends at its highest point; nothing after the peak
    assert np.argmin(ys) == len(ys) - 1


def test_jitter_alone_creates_no_stroke():
    f = swipe_path([])
    for seed in range(5):
        t, xy = sample(f, 10.0, 30, noise=0.012, seed=seed)  # ~ the jitter level seen in real tracking
        acc, *_ = run_detect(t, xy)
        assert acc == [], f"seed {seed}: jitter produced {len(acc)} strokes"


def test_tracking_gap_ends_stroke_and_is_not_bridged():
    # a long upward drag interrupted by a 0.4 s gap (> max_gap_s)
    f = swipe_path([(1.0, 0.8, 0.9, 0.1)])
    t, xy = sample(f, 3.0, 30)
    gap = (t > 1.3) & (t < 1.7)
    acc, all_s, seg, sm, status = run_detect(t, xy, drop=gap)
    # before and after the gap are different segments
    assert seg[(t < 1.3) & (seg >= 0)].max() < seg[(t > 1.7) & (seg >= 0)].min()
    for s in all_s:
        ts = t[s.indices]
        assert not ((ts.min() < 1.3) and (ts.max() > 1.7)), "a stroke spans the gap"
    # rendering: no pigment deposited in the gap region of the path
    gestures = to_gestures(t, sm, acc)
    cfg = copy.deepcopy(CFG)
    stamps = compute_stamps(gestures, cfg, 90, 160, 0.125, ASPECT)
    y_gap_lo, y_gap_hi = f(np.array([1.7]))[0, 1] * 159, f(np.array([1.3]))[0, 1] * 159
    inside = (stamps["y"] > y_gap_lo + 3) & (stamps["y"] < y_gap_hi - 3)
    assert not inside.any(), "stamps were placed across the tracking gap"


def test_short_gap_within_tolerance_is_allowed():
    f = swipe_path([(1.0, 0.3, 0.85, 0.35)])
    t, xy = sample(f, 2.0, 30)
    drop = np.flatnonzero(t > 1.12)[:2]  # 2 dropped frames = 0.067 s < max_gap_s
    acc, *_ = run_detect(t, xy, drop=drop)
    assert len(acc) == 1


def test_rapid_distinct_swipes_stay_separate():
    ev = []
    for k in range(4):  # 4 swipes, one every 0.6 s, with quick returns
        t0 = 0.5 + 0.6 * k
        ev += [(t0, 0.18, 0.8, 0.45), (t0 + 0.25, 0.25, 0.45, 0.8)]
    t, xy = sample(swipe_path(ev), 3.5, 30)
    acc, *_ = run_detect(t, xy)
    assert len(acc) == 4
    starts = np.array([t[s.indices[0]] for s in acc])
    assert np.all(np.diff(starts) > 0.45)


def test_quick_dip_and_snap_back_is_not_a_swipe():
    # thumb rests at 0.7, dips to 0.88 for ~0.1 s and snaps back (tracking glitch / twitch), twice
    ev = [(1.0, 0.07, 0.7, 0.88), (1.07, 0.1, 0.88, 0.7), (3.0, 0.06, 0.7, 0.9), (3.06, 0.08, 0.9, 0.7)]
    t, xy = sample(swipe_path(ev, y_rest=0.7), 4.5, 30)
    acc, *_ = run_detect(t, xy)
    assert acc == []


def test_swipe_after_quick_reposition_is_still_accepted():
    # quick dip (reposition) immediately followed by a real swipe that rises well above the rest level
    ev = [(1.0, 0.07, 0.7, 0.88), (1.07, 0.25, 0.88, 0.35), (1.6, 0.5, 0.35, 0.7)]
    t, xy = sample(swipe_path(ev, y_rest=0.7), 3.0, 30)
    acc, *_ = run_detect(t, xy)
    assert len(acc) == 1


def test_push_from_bottom_edge_is_excluded():
    f = swipe_path([(1.0, 0.2, 1.02, 0.75), (1.5, 0.3, 0.75, 1.02)], y_rest=1.02)
    t, xy = sample(f, 2.5, 30)
    acc, *_ = run_detect(t, xy)
    assert acc == []


def test_implausible_jump_is_rejected():
    f = swipe_path([])
    t, xy = sample(f, 2.0, 30)
    xy[30] = [0.1, 0.1]  # one wild sample
    status = np.full(len(t), OK, np.int8)
    seg, status = segment_track(t, xy, status, ASPECT, CFG["detect"])
    assert status[30] == JUMP_REJECTED and seg[30] == -1
    acc, *_ = run_detect(t, xy)
    assert acc == []


@pytest.mark.parametrize("fps,jitter", [(24, 0.0), (30, 0.0), (60, 0.0), (30, 0.35)])
def test_detection_is_frame_rate_independent(fps, jitter):
    ev = [(1.0, 0.25, 0.8, 0.4), (1.5, 0.4, 0.4, 0.8), (3.0, 0.35, 0.85, 0.5), (3.6, 0.4, 0.5, 0.85)]
    f = swipe_path(ev)
    t, xy = sample(f, 4.5, fps, jitter_frac=jitter, seed=1)
    acc, *_ = run_detect(t, xy)
    assert len(acc) == 2
    for s, (t0, d) in zip(acc, [(1.0, 0.25), (3.0, 0.35)]):
        assert abs(t[s.indices[0]] - t0) < 0.1
        assert abs(t[s.indices[-1]] - (t0 + d)) < 0.1


def test_pigment_deposition_is_frame_rate_independent():
    ev = [(1.0, 0.25, 0.8, 0.4), (1.5, 0.4, 0.4, 0.8), (3.0, 0.35, 0.85, 0.5), (3.6, 0.4, 0.5, 0.85)]
    f = swipe_path(ev)
    totals, fields = [], []
    for fps in (24, 30, 60):
        t, xy = sample(f, 4.5, fps)
        acc, _, _, sm, _ = run_detect(t, xy)
        g = to_gestures(t, sm, acc)
        cfg = scaled_config(CFG, 0.5)
        stamps = compute_stamps(g, cfg, 180, 320, 0.5, ASPECT)
        field = PigmentField(cfg, 180, 320, 0.5)
        for k in range(len(stamps["t"])):
            field.deposit_stamp(stamps, k)
        totals.append(field.total())
        fields.append(field.total_density().copy())
    totals = np.array(totals)
    assert np.ptp(totals) / totals.mean() < 0.06, f"deposited totals differ across fps: {totals}"
    for fl in fields[1:]:
        assert np.corrcoef(fields[0].ravel(), fl.ravel())[0, 1] > 0.9


# ----------------------------------------------------------------------------- pigment field

def _painted_field(tau=20.0, residue=0.12):
    cfg = scaled_config(CFG, 0.25)
    cfg["render"]["bleed"]["bleed_time_s"] = tau
    cfg["render"]["bleed"]["residue_fraction"] = residue
    f = swipe_path([(0.0, 0.3, 0.9, 0.3)])
    t = np.linspace(0, 0.3, 10)
    g = [{"id": 1, "points": [{"t": float(a), "x": float(p[0]), "y": float(p[1])} for a, p in zip(t, f(t))]},
         {"id": 2, "points": [{"t": float(a), "x": 0.02, "y": float(p[1])} for a, p in zip(t, f(t))]}]  # at the border
    st = compute_stamps(g, cfg, 90, 160, 0.125, ASPECT)
    field = PigmentField(cfg, 90, 160, 0.125)
    for k in range(len(st["t"])):
        field.deposit_stamp(st, k)
    return field


def test_bleeding_conserves_pigment_including_at_borders():
    field = _painted_field()
    before = field.total()
    for _ in range(30 * 60):
        field.step(1 / 30)
    assert abs(field.total() - before) / before < 1e-4
    assert field.bleed.min() > -1e-9, "explicit scheme went unstable"
    assert field.bleed.sum() > 0.9 * (before - field.stain.sum())  # wet paint has mostly bled after 3 tau


def test_wet_release_follows_elapsed_time():
    field = _painted_field(tau=20.0)
    w0 = field.wet.sum()
    for _ in range(30 * 20):
        field.step(1 / 30)
    assert abs(field.wet.sum() / w0 - np.exp(-1)) < 1e-3


def test_bleeding_depends_on_elapsed_time_not_step_count():
    a, b = _painted_field(), _painted_field()
    for _ in range(30 * 20):
        a.step(1 / 30)
    for _ in range(24 * 20):
        b.step(1 / 24)
    da, db = a.total_density(), b.total_density()
    assert np.abs(da - db).max() / da.max() < 0.02


def test_no_global_fade_and_stain_is_permanent():
    field = _painted_field()
    tot, stain = field.total(), field.stain.copy()
    for _ in range(300):
        field.step(1 / 30)
    assert abs(field.total() - tot) / tot < 1e-4
    assert np.array_equal(field.stain, stain)


def test_brush_has_bristle_streaks():
    """Across the stroke the deposit should vary (streaks), not be a smooth blob."""
    cfg = scaled_config(CFG, 1.0)
    t = np.linspace(0, 0.3, 20)
    g = [{"id": 1, "points": [{"t": float(a), "x": 0.5, "y": float(0.9 - 0.6 * a / 0.3)} for a in t]}]
    st = compute_stamps(g, cfg, 360, 640, 0.5, ASPECT)
    field = PigmentField(cfg, 360, 640, 0.5)
    for k in range(len(st["t"])):
        field.deposit_stamp(st, k)
    row = field.wet[400, 160:200]       # a horizontal cut through the stroke body
    inside = row[row > 0.2 * row.max()]
    assert len(inside) > 8
    assert inside.std() / inside.mean() > 0.1


# ----------------------------------------------------------------------------- calibration / video

def test_homography_maps_corners_to_unit_square_and_detects_mirroring():
    corners = [[400, 200], [680, 210], [700, 720], [420, 715]]
    seg = make_segment(corners, 0.0, 0.0)
    cal = Calibration({"version": 1, "segments": [seg], "image_size": [1280, 720], "screen_aspect": ASPECT})
    out = cal.to_phone(np.zeros(4), np.array(corners, float))
    assert np.allclose(out, [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-6)
    assert not seg["image_looks_mirrored"]
    # a point moving up in the image (y decreasing) moves up on the phone (y decreasing)
    p = cal.to_phone(np.zeros(2), np.array([[550, 600], [550, 400]], float))
    assert p[1, 1] < p[0, 1]
    # mirrored image: phone TL appears at image right
    mirrored = [[680, 210], [400, 200], [420, 715], [700, 720]]
    _, is_m = validate_corners(mirrored)
    assert is_m
    assert np.isnan(cal.to_phone(np.zeros(1), np.array([[np.nan, np.nan]]))).all()


def test_bad_corner_order_is_rejected():
    with pytest.raises(PipelineError):
        validate_corners([[400, 200], [700, 720], [680, 210], [420, 715]])  # self-intersecting


def test_orient_matches_ffmpeg_convention():
    img = np.zeros((2, 3, 3), np.uint8)
    img[0, 0] = 255  # top-left marker
    # FFmpeg rotation -90 (typical iPhone portrait) -> rotate clockwise: marker goes to top-right
    r = orient(img, 270)
    assert r.shape[:2] == (3, 2) and r[0, 1, 0] == 255
