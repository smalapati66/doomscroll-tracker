"""Shared helpers: errors, configuration loading, progress reporting, project paths."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.toml"

SWIPE_PROXY_DISCLAIMER = (
    "Strokes are upward thumb-tip movements tracked by camera inside the calibrated screen "
    "region. They are a proxy for swipes: they do not prove screen contact, do not show that "
    "the phone actually scrolled, and are not fingerprints."
)


class PipelineError(Exception):
    """An expected, user-facing failure (missing file, bad calibration, encode error...)."""


# ----------------------------------------------------------------------------- config

def _deep_merge(base: dict, override: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        key = f"{path}.{k}" if path else k
        if k not in base:
            raise PipelineError(f"unknown config key '{key}' (check spelling against config/default.toml)")
        if isinstance(v, dict) and isinstance(base[k], dict):
            out[k] = _deep_merge(base[k], v, key)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load config/default.toml, then overlay the user's file (if given)."""
    with open(DEFAULT_CONFIG, "rb") as f:
        cfg = tomllib.load(f)
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise PipelineError(f"config file not found: {p}")
        with open(p, "rb") as f:
            try:
                user = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise PipelineError(f"could not parse {p}: {e}") from e
        cfg = _deep_merge(cfg, user)
    return cfg


# ----------------------------------------------------------------------------- paths

def data_dir_for(video: str | Path) -> Path:
    """data/<video stem>/ holds calibration, tracking cache and gestures for one recording."""
    return PROJECT_ROOT / "data" / Path(video).stem


def require_file(path: str | Path, what: str, hint: str = "") -> Path:
    p = Path(path)
    if not p.is_file():
        msg = f"{what} not found: {p}"
        if hint:
            msg += f"\n  hint: {hint}"
        raise PipelineError(msg)
    return p


def file_fingerprint(path: str | Path) -> dict:
    """Cheap identity of a (possibly huge) file: size + mtime + hash of first/last MiB."""
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if st.st_size > (2 << 20):
            f.seek(-(1 << 20), 2)
            h.update(f.read(1 << 20))
    return {"name": p.name, "size": st.st_size, "mtime": int(st.st_mtime), "partial_sha256": h.hexdigest()}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    tmp.replace(p)


def read_json(path: str | Path, what: str, hint: str = ""):
    p = require_file(path, what, hint)
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise PipelineError(f"{what} is not valid JSON ({p}): {e}") from e


# ----------------------------------------------------------------------------- progress

class Progress:
    """Single-line progress on stderr, updated at most ~2x per second."""

    def __init__(self, label: str, total: int | None):
        self.label, self.total = label, total
        self.t0 = self.last = time.monotonic()
        self.n = 0

    def update(self, n: int = 1, extra: str = "") -> None:
        self.n += n
        now = time.monotonic()
        if now - self.last < 0.5 and (self.total is None or self.n < self.total):
            return
        self.last = now
        el = now - self.t0
        rate = self.n / el if el > 0 else 0.0
        if self.total:
            eta = (self.total - self.n) / rate if rate > 0 else float("inf")
            msg = f"\r{self.label}: {self.n}/{self.total} ({100*self.n/self.total:5.1f}%)  {rate:6.1f}/s  ETA {eta:6.0f}s {extra}"
        else:
            msg = f"\r{self.label}: {self.n}  {rate:6.1f}/s {extra}"
        sys.stderr.write(msg[:160].ljust(100))
        sys.stderr.flush()

    def done(self) -> None:
        el = time.monotonic() - self.t0
        sys.stderr.write(f"\r{self.label}: {self.n} done in {el:.1f}s".ljust(100) + "\n")
        sys.stderr.flush()
