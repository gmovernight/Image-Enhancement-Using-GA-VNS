# src/utils.py
from __future__ import annotations
import os, json, hashlib, time, random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import cv2


# ----------------------------- Seeding ------------------------------------ #
def set_seed(seed: int) -> np.random.Generator:
    """
    Set global seeds for reproducibility and return a numpy Generator.
    """
    if seed is None:
        raise ValueError("seed must be an int")
    np.random.seed(seed)                 # legacy/global
    random.seed(seed)                    # python stdlib
    rng = np.random.default_rng(seed)    # modern per-run RNG
    return rng


# --------------------------- Image I/O & helpers --------------------------- #
def read_image_gray01(path: str | os.PathLike) -> np.ndarray:
    """
    Read an image as grayscale float32 in [0,1]. Shape (H, W).
    """
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = img.astype(np.float32) / 255.0
    return img


def save_image_gray01(path: str | os.PathLike, img01: np.ndarray, make_dirs: bool = True) -> None:
    """
    Save a grayscale float image in [0,1] to disk (8-bit PNG by default).
    """
    p = Path(path)
    if make_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)
    if img01.ndim != 2:
        raise ValueError("save_image_gray01 expects a 2D grayscale array.")
    arr = np.clip(img01, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    ok = cv2.imwrite(str(p), arr)
    if not ok:
        raise IOError(f"cv2.imwrite failed for: {p}")


def clip01(x: np.ndarray) -> np.ndarray:
    """Clip to [0,1] float32 (without modifying input in-place)."""
    return np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)


def ensure_gray01(img: np.ndarray) -> np.ndarray:
    """
    Ensure grayscale float32 [0,1]. Converts if input is uint8 or 3-channel.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return clip01(img.astype(np.float32))


def percentile_contrast_stretch(img01: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """
    Contrast stretch by percentiles. Returns float32 in [0,1].
    """
    if not (0 <= p_low < p_high <= 100):
        raise ValueError("Require 0 <= p_low < p_high <= 100")
    lo = np.percentile(img01, p_low)
    hi = np.percentile(img01, p_high)
    if hi <= lo + 1e-12:
        return clip01(img01.copy())
    out = (img01 - lo) / (hi - lo)
    return clip01(out)


# ------------------------------- Hashing ----------------------------------- #
def stable_hash(obj: Any) -> str:
    """
    Stable SHA1 hash for JSON-serializable objects (used for caching).
    """
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ------------------------------- Timing ------------------------------------ #
@dataclass
class Timer:
    name: str = ""
    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.start
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed:.3f}s")


# ------------------------------- CSV Logger -------------------------------- #
class CSVLogger:
    """
    Minimal CSV logger that writes header on first write.
    Usage:
        log = CSVLogger('results/ga_log.csv', ['gen','best_F','mean_F'])
        log.write({'gen':0, 'best_F':1.23, 'mean_F':2.34})
    """
    def __init__(self, path: str | os.PathLike, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._opened = False

    def write(self, row: Dict[str, Any]) -> None:
        mode = "a"
        write_header = not self._opened and not self.path.exists()
        with self.path.open(mode, encoding="utf-8") as f:
            if write_header:
                f.write(",".join(self.fieldnames) + "\n")
            vals = [row.get(k, "") for k in self.fieldnames]
            f.write(",".join(map(_csv_escape, vals)) + "\n")
        self._opened = True


def _csv_escape(x: Any) -> str:
    s = str(x)
    if any(c in s for c in [",", "\n", '"']):
        s = '"' + s.replace('"', '""') + '"'
    return s
