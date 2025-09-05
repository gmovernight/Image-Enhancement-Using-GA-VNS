# src/metrics.py
from __future__ import annotations
import math
import numpy as np
from skimage.metrics import structural_similarity as _ssim

def _to_float01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    # assume inputs already ~[0,1]; just clamp for safety
    return np.clip(x, 0.0, 1.0)

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _to_float01(a), _to_float01(b)
    return float(np.mean((a - b) ** 2, dtype=np.float64))

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m == 0.0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / m))  # data_range=1

def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _to_float01(a), _to_float01(b)
    # skimage returns mean SSIM over the image; set data_range=1.0 for [0,1] floats
    val = _ssim(a, b, data_range=1.0)
    return float(val)

def all_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Convenience: returns {'MSE':..., 'PSNR':..., 'SSIM':...}."""
    m = mse(a, b)
    return {
        "MSE": m,
        "PSNR": float("inf") if m == 0.0 else float(10.0 * math.log10(1.0 / m)),
        "SSIM": ssim(a, b),
    }