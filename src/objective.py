# src/objective.py
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np

from .dataset import ImagePair, load_pair
from .pipeline import apply_pipeline, Pipeline
from .metrics import mse, psnr, ssim

def _metrics_of(enh: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    m = mse(enh, gt)
    p = psnr(enh, gt)
    s = ssim(enh, gt)
    return {"MSE": m, "PSNR": p, "SSIM": s}

def evaluate_on_pairs(pipe: Pipeline, pairs: List[ImagePair]) -> Dict[str, float]:
    """
    Apply `pipe` to each distorted image and compare to GT.
    Returns mean metrics over the set: {'MSE', 'PSNR', 'SSIM'}.
    """
    mses, psnrs, ssims = [], [], []
    for pair in pairs:
        I, GT = load_pair(pair)
        J = apply_pipeline(I, pipe)
        mses.append(mse(J, GT))
        psnrs.append(psnr(J, GT))
        ssims.append(ssim(J, GT))
    return {
        "MSE": float(np.mean(mses)),
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
    }

def objective(pipe: Pipeline,
              pairs: List[ImagePair],
              mode: str = "ssim",
              weights: Tuple[float, float, float] = (1.0, 0.1, 0.5)) -> Tuple[float, Dict[str, float]]:
    """
    Compute objective value to MINIMIZE and return (F, mean_metrics).
    - mode="ssim": F = 1 - mean(SSIM)
    - mode="weighted": F = α*mean(MSE) - β*mean(PSNR) + γ*(1 - mean(SSIM))
      (α,β,γ) given by `weights`.
    """
    means = evaluate_on_pairs(pipe, pairs)
    if mode == "ssim":
        F = 1.0 - means["SSIM"]
    elif mode == "weighted":
        alpha, beta, gamma = weights
        # Guard: PSNR can be inf if MSE==0; cap for stability.
        psnr_mean = means["PSNR"]
        if np.isinf(psnr_mean):
            psnr_mean = 100.0
        F = alpha * means["MSE"] - beta * psnr_mean + gamma * (1.0 - means["SSIM"])
    else:
        raise ValueError("Unknown objective mode. Use 'ssim' or 'weighted'.")
    return float(F), means
