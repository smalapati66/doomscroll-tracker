"""Stage 3: render accepted gestures as bristle brushstrokes that bleed into the surface (silent video).

Artistic model, not physical paint mixing:
  * Brush: a soft watercolour-like brush. Each stroke is swept by a row of bristles across the brush width. Every bristle has its
    own paint load, runs dry along the stroke (so streaks break up toward the end), and the brush
    tapers in at the start and lifts off at the end.
  * Deposition is pre-computed at fixed ARC-LENGTH spacing along the recorded path, each sample
    carrying its own source timestamp. The pigment laid down therefore depends on the path, never on
    how many frames are decoded or rendered, and strokes are revealed along their path in real time.
    Downward returns are not in the gesture data and add nothing.
  * Three pigment layers, total pigment conserved (nothing is ever removed or faded):
      wet    fresh paint that keeps the brush texture. While wet it creeps slowly through the paper
             fibres (feathered edges), and it releases pigment into `bleed` exponentially with time
             constant bleed_time_s (exact for any time step).
      bleed  released pigment that spreads forever, during strokes and pauses: dB/dt = div(k grad B)
             with a fibrous, spatially varying k; explicit flux-form scheme with zero-flux borders and
             automatic sub-stepping (k*dt <= 0.2 < 0.25 stability limit). Runs on a coarser grid.
      stain  a small permanent residue of each stroke (residue_fraction) that never moves.
  * Colour: the ground colour comes from total density, c = D / (D + density_scale), through a
    peach -> dusty rose -> deep red -> oxblood palette. Wet paint is drawn on top as a translucent rust-red
    wash with darkened rims and granulation, so a fresh stroke reads as a watercolour brushstroke and
    then dissolves into the ground as it bleeds.
  * All randomness comes from a fixed seed; per-gesture streams are seeded by (seed, gesture id),
    so a time-range render matches the full render exactly.
"""
from __future__ import annotations

import copy
import datetime as _dt
import math
from pathlib import Path

import cv2
import numpy as np

from .common import PROJECT_ROOT, PipelineError, Progress, data_dir_for, read_json, sha256_file, write_json
from .encode import VideoWriter

# keys whose values are in output-canvas pixels (scaled when rendering at another width)
PX_KEYS = {"brush": ["width_px", "spacing_px", "skip_scale_px", "paper_grain_scale_px", "path_smoothing_px"],
           "bleed": ["distance_after_60s_px", "fiber_scale_px", "wet_spread_after_60s_px"],
           "wet": ["edge_width_px", "pooling_scale_px"]}


# ----------------------------------------------------------------------------- inputs

def load_gestures(path: str | Path) -> dict:
    doc = read_json(path, "gestures JSON", "run `detect` first, or pass --gestures <file>")
    if "gestures" not in doc:
        raise PipelineError(f"{path} does not look like a gestures file (no 'gestures' key)")
    return doc


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    if len(h) != 6:
        raise PipelineError(f"palette colour '{h}' must be #RRGGBB")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32)


def build_lut(stops, n: int = 1024) -> np.ndarray:
    pos = np.array([float(s[0]) for s in stops])
    if np.any(np.diff(pos) <= 0) or pos[0] != 0.0 or pos[-1] != 1.0:
        raise PipelineError("palette stops must be strictly increasing from 0.0 to 1.0")
    cols = np.stack([hex_to_rgb(s[1]) for s in stops])
    x = np.linspace(0, 1, n)
    return np.stack([np.interp(x, pos, cols[:, c]) for c in range(3)], axis=1).astype(np.float32)


def value_noise(shape, scale: float, rng: np.random.Generator, octaves: int = 4) -> np.ndarray:
    """Smooth fractal noise in [0,1] (bicubic-upsampled random grids)."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = max(scale / (2 ** o), 1.0)
        gh, gw = max(2, int(math.ceil(h / s)) + 2), max(2, int(math.ceil(w / s)) + 2)
        g = rng.random((gh, gw)).astype(np.float32)
        up = cv2.resize(g, (int(gw * s), int(gh * s)), interpolation=cv2.INTER_CUBIC)[:h, :w]
        if up.shape != tuple(shape):
            up = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * up
        total += amp
        amp *= 0.5
    out /= total
    lo, hi = np.percentile(out, [1, 99])
    return np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)


def load_texture(path: str) -> np.ndarray:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise PipelineError(f"brush texture image not found or unreadable: {p}")
    return img.astype(np.float32) / 255.0


def scaled_config(cfg: dict, factor: float) -> dict:
    """Copy of cfg with all output-pixel parameters multiplied by factor (for smaller previews)."""
    eff = copy.deepcopy(cfg)
    for sec, keys in PX_KEYS.items():
        for k in keys:
            eff["render"][sec][k] = float(cfg["render"][sec][k]) * factor
    return eff


# ----------------------------------------------------------------------------- brush samples

def canvas_mapper(rcfg: dict, sim_w: int, sim_h: int, aspect: float):
    """Normalized phone coords -> sim-cell coords."""
    if rcfg["fit"] == "stretch":
        return lambda x, y: (x * (sim_w - 1), y * (sim_h - 1))
    if rcfg["fit"] == "contain":
        if aspect * sim_h <= sim_w:
            w, h = aspect * sim_h, sim_h
        else:
            w, h = sim_w, sim_w / aspect
        ox, oy = (sim_w - w) / 2, (sim_h - h) / 2
        return lambda x, y: (ox + x * (w - 1), oy + y * (h - 1))
    raise PipelineError(f"render.fit must be 'stretch' or 'contain', got {rcfg['fit']!r}")


def _gauss_smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of a 1-D signal (edge-padded), sigma in samples."""
    if sigma < 0.5 or len(a) < 3:
        return a
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(a, r, mode="edge"), k / k.sum(), mode="valid")


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def compute_stamps(gestures: list, cfg: dict, sim_w: int, sim_h: int, scale: float, aspect: float) -> dict:
    """Brush samples for all gestures, sorted by time.

    Samples sit every `spacing_px` along each stroke's arc length (canvas space); times are
    interpolated along the path. Per sample we store position, direction, width factor and the
    paint amount of every bristle (loads x drying x skips x taper).
    """
    rcfg, b = cfg["render"], cfg["render"]["brush"]
    seed, nb = int(rcfg["seed"]), int(b["n_bristles"])
    width = float(b["width_px"]) * scale
    step = max(float(b["spacing_px"]) * scale, 0.2)
    dry_len = max(float(b["dry_length"]) * sim_h, 1.0)
    skip_scale = max(float(b["skip_scale_px"]) * scale, 1.0)
    mapper = canvas_mapper(rcfg, sim_w, sim_h, aspect)
    cols = {k: [] for k in ("t", "x", "y", "angle", "s", "wf", "amp", "gid")}
    bristles = []
    for g in gestures:
        pts = g["points"]
        if not pts:
            continue
        t = np.array([p["t"] for p in pts], np.float64)
        x, y = mapper(np.array([p["x"] for p in pts]), np.array([p["y"] for p in pts]))
        s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        L = float(s[-1])
        sk = np.arange(0.0, L + 1e-9, step)
        su, ui = np.unique(s, return_index=True)
        if len(su) == 1:
            tk, xk, yk = t[:1], x[:1], y[:1]
        else:
            tk, xk, yk = np.interp(sk, su, t[ui]), np.interp(sk, su, x[ui]), np.interp(sk, su, y[ui])
        n = len(tk)
        sk = sk[:n] if len(sk) >= n else np.zeros(n)
        if n > 2:
            # The brush follows a slightly smoothed path, and bristles are oriented by a direction
            # smoothed over ~half a brush width, so tracking wobble does not fan the bristles out.
            xk = _gauss_smooth(xk, float(b["path_smoothing_px"]) * scale / step)
            yk = _gauss_smooth(yk, float(b["path_smoothing_px"]) * scale / step)
            sa = 0.5 * width / step
            ang = np.arctan2(_gauss_smooth(np.gradient(yk), sa), _gauss_smooth(np.gradient(xk), sa))
        else:
            ang = np.full(n, -np.pi / 2)
        rng = np.random.default_rng([seed, int(g["id"])])
        load = np.clip(rng.normal(1.0, float(b["bristle_load_variation"]), nb), 0.15, 1.8)
        dry = dry_len * rng.uniform(0.6, 1.4, nb)
        # smooth random "skip" signal per bristle along the stroke
        nk = int(L / skip_scale) + 3
        knots = rng.random((nb, nk))
        u = sk / skip_scale
        i0 = np.clip(u.astype(int), 0, nk - 2)
        f = _smoothstep(u - i0)
        skip = knots[:, i0] * (1 - f) + knots[:, i0 + 1] * f            # (nb, n)
        remain = np.exp(-sk[None, :] / dry[:, None])                     # paint left on each bristle
        gate = _smoothstep((skip - float(b["dry_break"]) * (1 - remain)) / 0.25 + 0.5)
        amp = load[:, None] * remain * gate                              # (nb, n)
        if L > 0:
            pin = _smoothstep(sk / max(float(b["taper_in"]) * L, 1e-6))
            pout = _smoothstep((L - sk) / max(float(b["taper_out"]) * L, 1e-6))
            press = pin * np.sqrt(pout)
        else:
            press = np.ones(n)
        cols["t"].append(tk); cols["x"].append(xk); cols["y"].append(yk); cols["angle"].append(ang)
        cols["s"].append(sk); cols["wf"].append(0.45 + 0.55 * press)
        cols["amp"].append((amp * (0.5 + 0.5 * press)[None, :]).T)
        cols["gid"].append(np.full(n, int(g["id"])))
        offs = (np.linspace(-0.5, 0.5, nb) + rng.normal(0, 0.25 / nb, nb)) * width
        bristles.append(offs)
    empty = {k: np.zeros((0, nb)) if k == "amp" else np.array([]) for k in cols}
    out = {k: np.concatenate(v) for k, v in cols.items()} if cols["t"] else empty
    out["offs"] = np.concatenate([np.tile(o, (len(a), 1)) for o, a in zip(bristles, cols["t"])]) if bristles \
        else np.zeros((0, nb))
    order = np.argsort(out["t"], kind="stable")
    out = {k: v[order] for k, v in out.items()}
    out.update(step=step, width=width)
    return out


# ----------------------------------------------------------------------------- simulation

class PigmentField:
    def __init__(self, cfg: dict, sim_w: int, sim_h: int, scale: float):
        rcfg = cfg["render"]
        b, bl = rcfg["brush"], rcfg["bleed"]
        self.w, self.h, self.scale = sim_w, sim_h, scale
        self.wet = np.zeros((sim_h, sim_w), np.float32)
        self.stain = np.zeros((sim_h, sim_w), np.float32)
        self.ds = max(1, int(bl["grid_downsample"]))
        self.cw, self.ch = max(4, sim_w // self.ds), max(4, sim_h // self.ds)
        self.bleed = np.zeros((self.ch, self.cw), np.float64)   # pigment MASS per coarse cell
        self.residue = float(bl["residue_fraction"])
        self.tau = float(bl["bleed_time_s"])
        rng = np.random.default_rng(int(rcfg["seed"]))
        # diffusion coefficient in coarse cells^2/s from the spread distance sqrt(2 k t) at t = 60 s
        L = float(bl["distance_after_60s_px"]) * scale / (sim_w / self.cw)
        k0 = L * L / 120.0
        fib = value_noise((self.ch, self.cw), max(float(bl["fiber_scale_px"]) * scale / self.ds, 1.0), rng)
        kmap = k0 * (1 + float(bl["fiber_variation"]) * (2 * fib - 1))
        self.kx = 0.5 * (kmap[:, 1:] + kmap[:, :-1])            # face coefficients
        self.ky = 0.5 * (kmap[1:, :] + kmap[:-1, :])
        self.kmax = float(kmap.max())
        # wet paint also creeps slowly through the fibres while it is still wet (feathered edges)
        Lw = float(bl["wet_spread_after_60s_px"]) * scale
        kw0 = Lw * Lw / 120.0
        fib_s = cv2.resize(fib, (sim_w, sim_h), interpolation=cv2.INTER_LINEAR)
        kwmap = (kw0 * (1 + float(bl["fiber_variation"]) * (2 * fib_s - 1))).astype(np.float32)
        self.kwx = 0.5 * (kwmap[:, 1:] + kwmap[:, :-1])
        self.kwy = 0.5 * (kwmap[1:, :] + kwmap[:-1, :])
        self.kwmax = float(kwmap.max())
        self.paper = value_noise((sim_h, sim_w), max(float(b["paper_grain_scale_px"]) * scale, 1.0), rng)
        self.paper_strength = float(b["paper_grain"])
        self.texture = None if b["texture"] == "procedural" else load_texture(b["texture"])
        self.texture_strength = float(b["texture_strength"])
        self.density_per_pass = float(b["density_per_pass"])
        self.bristle_softness = float(b["bristle_softness"])
        self.deposited = 0.0

    # --- deposition
    def deposit_stamp(self, st: dict, k: int) -> None:
        nb = st["amp"].shape[1]
        wf = float(st["wf"][k])
        offs = st["offs"][k] * wf
        width = st["width"] * wf
        sig_b = max(float(self.bristle_softness) * width / nb, 0.35)
        sig_u = max(0.8 * st["step"], 0.6)
        half = int(math.ceil(width / 2 + 3 * sig_b + 2))
        x, y = float(st["x"][k]), float(st["y"][k])
        cx, cy = int(round(x)), int(round(y))
        x0, y0 = max(cx - half, 0), max(cy - half, 0)
        x1, y1 = min(cx + half + 1, self.w), min(cy + half + 1, self.h)
        if x1 <= x0 or y1 <= y0:
            return
        X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32) - x, np.arange(y0, y1, dtype=np.float32) - y)
        c, s = math.cos(st["angle"][k]), math.sin(st["angle"][k])
        u = X * c + Y * s           # along the stroke
        v = -X * s + Y * c          # across the stroke
        vg = np.linspace(-half, half, 4 * half + 1, dtype=np.float32)
        prof = (st["amp"][k][:, None] * np.exp(-(vg[None, :] - offs[:, None]) ** 2 / (2 * sig_b ** 2))).sum(0)
        norm = math.sqrt(2 * math.pi) * sig_b / (width / nb)   # sum of unit bristles ~ 1 across the brush
        prof /= norm
        if self.texture is not None:  # user texture mapped onto the stroke like a ribbon
            th, tw = self.texture.shape
            row = int((st["s"][k] / max(width, 1e-6)) * tw) % th
            col = np.clip(((vg / max(width, 1e-6)) + 0.5) * (tw - 1), 0, tw - 1).astype(int)
            prof = prof * (1 - self.texture_strength + self.texture_strength * self.texture[row, col])
        amp = self.density_per_pass * st["step"] / (sig_u * math.sqrt(2 * math.pi))
        dep = amp * np.exp(-u * u / (2 * sig_u ** 2)) * np.interp(v, vg, prof).astype(np.float32)
        g = self.paper[y0:y1, x0:x1]
        dep = (dep * (1 - self.paper_strength + self.paper_strength * 2 * g)).astype(np.float32)
        self.stain[y0:y1, x0:x1] += self.residue * dep
        self.wet[y0:y1, x0:x1] += (1 - self.residue) * dep
        self.deposited += float(dep.sum(dtype=np.float64))

    # --- time evolution
    @staticmethod
    def _flux_diffuse(A: np.ndarray, kx: np.ndarray, ky: np.ndarray, kmax: float, dt: float) -> None:
        """In-place explicit diffusion div(k grad A), zero-flux borders, sub-stepped for stability."""
        n = max(1, int(math.ceil(kmax * dt / 0.2)))
        kx, ky = kx * (dt / n), ky * (dt / n)
        for _ in range(n):
            fx = kx * (A[:, 1:] - A[:, :-1])
            fy = ky * (A[1:, :] - A[:-1, :])
            A[:, :-1] += fx
            A[:, 1:] -= fx
            A[:-1, :] += fy
            A[1:, :] -= fy

    def step(self, dt: float) -> None:
        if dt <= 0:
            return
        if self.kwmax > 0 and self.wet.any():
            self._flux_diffuse(self.wet, self.kwx, self.kwy, self.kwmax, dt)
        if self.tau > 0:
            rel = self.wet * np.float32(1 - math.exp(-dt / self.tau))
            total = float(rel.sum(dtype=np.float64))
            if total > 0:
                self.wet -= rel
                coarse = cv2.resize(rel, (self.cw, self.ch), interpolation=cv2.INTER_AREA).astype(np.float64)
                self.bleed += coarse * (total / max(coarse.sum(), 1e-30))   # exact mass transfer
        if self.kmax > 0 and self.bleed.any():
            self._flux_diffuse(self.bleed, self.kx, self.ky, self.kmax, dt)

    # --- readout
    def bleed_density(self) -> np.ndarray:
        cell = (self.w * self.h) / (self.cw * self.ch)
        return cv2.resize((self.bleed / cell).astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_LINEAR)

    def total_density(self) -> np.ndarray:
        return self.stain + self.wet + self.bleed_density()

    def total(self) -> float:
        return float(self.stain.sum(dtype=np.float64) + self.wet.sum(dtype=np.float64) + self.bleed.sum())


class Colorizer:
    def __init__(self, cfg: dict, out_w: int, out_h: int):
        pal, wet = cfg["render"]["palette"], cfg["render"]["wet"]
        self.lut = build_lut(pal["stops"])
        self.scale = float(pal["density_scale"])
        self.curve = pal["curve"]
        if self.curve not in ("hyperbolic", "exp"):
            raise PipelineError(f"render.palette.curve must be 'hyperbolic' or 'exp', got {self.curve!r}")
        self.wet_light, self.wet_dense = hex_to_rgb(wet["color_light"]), hex_to_rgb(wet["color_dense"])
        self.wet_opacity, self.wet_scale = float(wet["opacity"]), float(wet["opacity_scale"])
        self.wet_dense_scale = float(wet["dense_scale"])
        self.edge_darkening = float(wet["edge_darkening"])
        self.edge_sigma = max(float(wet["edge_width_px"]) * float(cfg["render"]["sim_scale"]), 0.5)
        self.granulation = float(wet["granulation"])
        self.pooling = float(wet["pooling"])
        sim_w = max(8, int(round(out_w * float(cfg["render"]["sim_scale"]))))
        sim_h = max(8, int(round(out_h * float(cfg["render"]["sim_scale"]))))
        pool_scale = max(float(wet["pooling_scale_px"]) * float(cfg["render"]["sim_scale"]), 2.0)
        self.pool = value_noise((sim_h, sim_w), pool_scale, np.random.default_rng([int(cfg["render"]["seed"]), 7]), octaves=3)
        gs = float(pal["grain_strength"])
        rng = np.random.default_rng([int(cfg["render"]["seed"]), 99])
        self.grain = (1 + gs * (2 * value_noise((out_h, out_w), 2.0, rng, octaves=3) - 1)).astype(np.float32)[..., None]
        self.w, self.h = out_w, out_h

    def __call__(self, field: PigmentField) -> np.ndarray:
        # composite at pigment-field resolution, then upsample the colour image once
        d = np.maximum(field.total_density(), 0)
        c = d / (d + self.scale) if self.curve == "hyperbolic" else 1 - np.exp(-d / self.scale)
        idx = np.clip((c * (len(self.lut) - 1)).astype(np.int32), 0, len(self.lut) - 1)
        rgb = self.lut[idx]
        w = np.maximum(field.wet, 0)
        # watercolour edge darkening: pigment gathers at the rim of the wet AREA (the stroke outline),
        # independent of how thick the paint is inside
        area = 1 - np.exp(-w / 0.25)
        rim = np.clip(2 * (area - cv2.GaussianBlur(area, (0, 0), self.edge_sigma)), 0, 1)
        rim = self.edge_darkening * rim * area
        alpha = self.wet_opacity * (1 - np.exp(-w / self.wet_scale))
        # granulation (fine) and pooling (blotchy, low frequency): uneven settling of the wash
        alpha *= 1 - self.granulation + self.granulation * 2 * field.paper
        alpha *= 1 - self.pooling + self.pooling * 2 * self.pool
        alpha = np.clip(alpha, 0, 1)[..., None]
        mix = np.clip(w / (w + self.wet_dense_scale) + 0.5 * rim, 0, 1)[..., None]
        wet_rgb = self.wet_light * (1 - mix) + self.wet_dense * mix
        rgb = (rgb * (1 - alpha) + wet_rgb * alpha) * (1 - 0.5 * rim)[..., None]   # darkened rim
        rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_CUBIC) * self.grain
        return np.clip(rgb, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- rendering

def render(gdoc: dict, cfg: dict, out_dir: Path, start_s: float = 0.0, end_s: float | None = None,
           history: bool = True, speed: float = 1.0, stills: list[float] | None = None,
           source_label: str = "") -> dict:
    rcfg = cfg["render"]
    W, H, fps = int(rcfg["width"]), int(rcfg["height"]), float(rcfg["fps"])
    if W <= 0 or H <= 0 or fps <= 0:
        raise PipelineError("render width, height and fps must be positive")
    if speed <= 0:
        raise PipelineError("--speed must be positive")
    scale = float(rcfg["sim_scale"])
    sim_w, sim_h = max(8, int(round(W * scale))), max(8, int(round(H * scale)))
    # pixel sizes are specified for a 720-wide canvas; keep them proportional for smaller previews
    eff = scaled_config(cfg, W / 720.0)

    duration = float(gdoc.get("video_duration_s") or 0.0)
    end_s = duration if end_s is None else min(end_s, duration) if duration else end_s
    if end_s is None or end_s <= start_s:
        raise PipelineError(f"empty render range: start {start_s} s, end {end_s} s")
    aspect = float(gdoc.get("calibration", {}).get("screen_aspect", 9 / 19.5))
    stamps = compute_stamps(gdoc["gestures"], eff, sim_w, sim_h, scale, aspect)
    field = PigmentField(eff, sim_w, sim_h, scale)
    colorize = Colorizer(eff, W, H)
    n_frames = max(1, int(round((end_s - start_s) / speed * fps)))
    accelerated = abs(speed - 1.0) > 1e-9

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("still_*.png"):  # stills from an earlier render in this folder
        old.unlink()
    video_path = out_dir / ("pigment_ACCELERATED_%gx.mp4" % speed if accelerated else "pigment.mp4")
    st_t = stamps["t"]
    ptr = 0

    def advance_to(t_target: float, t_prev: float):
        nonlocal ptr
        field.step(t_target - t_prev)
        while ptr < len(st_t) and st_t[ptr] <= t_target:
            field.deposit_stamp(stamps, ptr)
            ptr += 1

    # history: simulate 0..start without encoding so a range render matches the full render
    t_sim = 0.0 if history else start_s
    if not history:
        ptr = int(np.searchsorted(st_t, start_s, side="right")) if len(st_t) else 0
    elif start_s > 0:
        hist_dt = 1.0 / fps
        steps = int(math.ceil(start_s / hist_dt))
        prog = Progress("history (not encoded)", steps)
        for k in range(1, steps + 1):
            tt = min(k * hist_dt, start_s)
            advance_to(tt, t_sim)
            t_sim = tt
            prog.update()
        prog.done()

    writer = VideoWriter(video_path, W, H, fps)
    stills = sorted(stills or [])
    still_paths = []
    prog = Progress("render", n_frames)
    for k in range(n_frames):
        t_k = start_s + k * speed / fps
        advance_to(t_k, t_sim)
        t_sim = t_k
        rgb = colorize(field)
        if accelerated:
            label = f"ACCELERATED PREVIEW x{speed:g} - NOT REAL TIME - src {t_k:7.1f}s"
            fs = max(0.35, 0.5 * W / 720)
            cv2.rectangle(rgb, (0, H - 34), (W, H), (0, 0, 0), -1)
            cv2.putText(rgb, label, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(rgb)
        out_t = k / fps
        while stills and out_t + 1e-9 >= stills[0]:
            p = out_dir / f"still_out{stills[0]:07.2f}s_src{t_k:07.2f}s.png"
            cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            still_paths.append(str(p))
            stills.pop(0)
        prog.update()
    prog.done()
    writer.close()

    total = field.total()
    stats = {
        "brush_samples_total": int(len(st_t)), "brush_samples_deposited": int(ptr),
        "gestures_in_file": len(gdoc["gestures"]),
        "gestures_started_by_end": int(len(np.unique(stamps["gid"][:ptr]))) if ptr else 0,
        "pigment_deposited": field.deposited, "pigment_in_field": total,
        "conservation_rel_error": (total - field.deposited) / field.deposited if field.deposited else 0.0,
        "pigment_share_wet_bleed_stain": [float(field.wet.sum()) / total if total else 0,
                                          float(field.bleed.sum()) / total if total else 0,
                                          float(field.stain.sum()) / total if total else 0],
        "max_density": float(field.total_density().max()),
        "mean_density": float(field.total_density().mean()),
        "sim_size": [sim_w, sim_h], "bleed_grid": [field.cw, field.ch],
    }
    timing = {
        "time_origin": ("output t = 0 corresponds to source time start_s; source time is measured from the "
                        "first video frame's presentation timestamp of the Mac recording (see gestures.json "
                        "time_origin). source_time = start_s + output_time * speed"),
        "source": source_label,
        "start_s": start_s, "end_s": end_s, "speed": speed, "accelerated_preview": accelerated,
        "fps": fps, "n_frames": n_frames, "output_duration_s": n_frames / fps,
        "history_simulated_from_0": bool(history and start_s > 0),
        "audio": "none (silent). To add phone audio later, align its start to the Mac recording's source time.",
    }
    return {"video": video_path, "stats": stats, "timing": timing, "stills": still_paths, "effective_cfg": eff}


def render_from_cli(args, cfg: dict) -> None:
    if args.width:
        cfg["render"]["width"] = args.width
    if args.height:
        cfg["render"]["height"] = args.height
    if args.fps:
        cfg["render"]["fps"] = args.fps
    if args.gestures:
        gpath = Path(args.gestures)
    elif args.video:
        gpath = data_dir_for(args.video) / "gestures.json"
    else:
        raise PipelineError("pass --gestures <file> or --video <source> to locate gestures.json")
    gdoc = load_gestures(gpath)
    stem = gpath.parent.name if gpath.name == "gestures.json" else gpath.stem
    end_label = f"{args.end:.0f}s" if args.end is not None else "end"
    name = args.name or f"{stem}_{args.start:.0f}s-{end_label}_{cfg['render']['width']}x{cfg['render']['height']}"
    if abs(args.speed - 1) > 1e-9:
        name += f"_ACCELERATED_{args.speed:g}x"
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "renders" / name
    stills = [float(s) for s in args.stills.split(",") if s.strip()] if args.stills else []
    res = render(gdoc, cfg, out_dir, args.start, args.end, history=not args.no_history, speed=args.speed,
                 stills=stills, source_label=str(gpath))
    write_json(out_dir / "render_config.json", {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "gestures_file": str(gpath), "gestures_sha256": sha256_file(gpath),
        "command_args": {k: v for k, v in vars(args).items() if k != "fn"},
        "config": cfg,
        "effective_render_config_after_size_scaling": res["effective_cfg"]["render"],
    })
    write_json(out_dir / "timing.json", res["timing"])
    write_json(out_dir / "render_stats.json", res["stats"])
    s = res["stats"]
    print(f"wrote {res['video']}  ({res['timing']['n_frames']} frames @ {res['timing']['fps']:g} fps = "
          f"{res['timing']['output_duration_s']:.3f} s)")
    print(f"brush samples {s['brush_samples_deposited']}/{s['brush_samples_total']}, pigment conservation error "
          f"{100*s['conservation_rel_error']:.4f}%, mean density {s['mean_density']:.2f}, "
          f"wet/bleed/stain {[round(v, 2) for v in s['pigment_share_wet_bleed_stain']]}")
    for p in res["stills"]:
        print(f"still: {p}")
    print(f"config + timing saved in {out_dir}")
