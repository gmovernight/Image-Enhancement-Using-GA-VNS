# src/ops.py
from __future__ import annotations
import numpy as np
import cv2
from skimage import exposure

from .utils import clip01, percentile_contrast_stretch, ensure_gray01

# ------------------------------- Bounds ------------------------------------ #
BOUNDS = {
    "gamma":    {"g": (0.40, 1.80)},
    "gauss":    {"sigma": (0.30, 3.00), "ksize": (3, 9)},  # ksize ∈ {3,5,7,9}
    "unsharp":  {"radius": (0.50, 2.50), "amount": (0.10, 1.50), "thresh": (0.0, 10.0)},
    "cstretch": {"p_low": (0.0, 10.0), "p_high": (90.0, 99.0)},  # enforce p_high ≥ p_low+5
}

def _odd_ksize_from_sigma(sigma: float, kmin: int = 3, kmax: int = 9) -> int:
    if sigma <= 0:
        return 3
    k = int(2 * round(3 * sigma) + 1)  # ~±3σ coverage
    k = max(kmin, min(k, kmax))
    if k % 2 == 0:
        k += 1
    return k

# ------------------------------- Operators --------------------------------- #
def gamma(img01: np.ndarray, g: float) -> np.ndarray:
    g_lo, g_hi = BOUNDS["gamma"]["g"]
    g = float(np.clip(g, g_lo, g_hi))
    x = ensure_gray01(img01)
    x = np.maximum(x, 1e-8)  # avoid 0**g edge cases
    out = x ** g            # g<1 → brighten, g>1 → darken
    return clip01(out)

def gauss(img01: np.ndarray, sigma: float, ksize: int | None = None) -> np.ndarray:
    """Gaussian blur with given sigma (pixels). If ksize None, derive it."""
    s_lo, s_hi = BOUNDS["gauss"]["sigma"]
    sigma = float(np.clip(sigma, s_lo, s_hi))
    if ksize is None:
        ksize = _odd_ksize_from_sigma(sigma)
    k_lo, k_hi = BOUNDS["gauss"]["ksize"]
    ksize = int(np.clip(ksize, k_lo, k_hi))
    if ksize % 2 == 0:
        ksize += 1
    x = ensure_gray01(img01)
    out = cv2.GaussianBlur(x, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return clip01(out)

def unsharp(img01: np.ndarray, radius: float, amount: float, thresh: float = 0.0) -> np.ndarray:
    """
    Unsharp masking: out = img + amount * (img - blur(img, radius))
    - radius: Gaussian sigma (px)
    - amount: 0.1..1.5
    - thresh: ignore edges with |mask| < thresh/255 (on 0..1 scale)
    """
    r_lo, r_hi = BOUNDS["unsharp"]["radius"]
    a_lo, a_hi = BOUNDS["unsharp"]["amount"]
    t_lo, t_hi = BOUNDS["unsharp"]["thresh"]
    radius = float(np.clip(radius, r_lo, r_hi))
    amount = float(np.clip(amount, a_lo, a_hi))
    thresh = float(np.clip(thresh, t_lo, t_hi))

    x = ensure_gray01(img01)
    k = _odd_ksize_from_sigma(radius)
    blur = cv2.GaussianBlur(x, (k, k), sigmaX=radius, sigmaY=radius, borderType=cv2.BORDER_REFLECT)
    mask = x - blur
    if thresh > 0:
        thr = thresh / 255.0
        mask[np.abs(mask) < thr] = 0.0
    out = x + amount * mask
    return clip01(out)

def he(img01: np.ndarray) -> np.ndarray:
    """Global histogram equalization."""
    x = ensure_gray01(img01)
    out = exposure.equalize_hist(x)  # returns float in [0,1]
    return clip01(out.astype(np.float32, copy=False))

def cstretch(img01: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Percentile contrast stretch; enforces p_high ≥ p_low+5."""
    pl_lo, pl_hi = BOUNDS["cstretch"]["p_low"]
    ph_lo, ph_hi = BOUNDS["cstretch"]["p_high"]
    p_low  = float(np.clip(p_low,  pl_lo, pl_hi))
    p_high = float(np.clip(p_high, ph_lo, ph_hi))
    if p_high < p_low + 5.0:
        p_high = min(ph_hi, p_low + 5.0)
    return percentile_contrast_stretch(ensure_gray01(img01), p_low, p_high)