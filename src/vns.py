# src/vns.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import time

from .utils import set_seed, CSVLogger
from .dataset import ImagePair, list_train_pairs
from .pipeline import (
    Step, Pipeline, random_pipeline,
    pipe_to_dict, pipe_from_dict, pretty
)
from .fitness import evaluate_pipeline, Memo
from . import ops as OPS

# Discrete choices for Gaussian kernel size
GAUSS_KSIZES = (3, 5, 7, 9)

# ------------------------------- Config ------------------------------------ #
@dataclass
class VNSConfig:
    seed: int = 42
    max_iters: int = 300                 # total VNS iterations (shake+local search cycles)
    K: int = 4                           # number of neighbourhoods
    ls_trials: int = 10                  # local search: attempts per shake
    jitter_strength: float = 0.12        # param jitter fraction of range
    L_min: int = 4
    L_max: int = 6
    objective: str = "ssim"              # "ssim" or "weighted"
    weights: Tuple[float, float, float] = (1.0, 0.1, 0.5)
    log_path: str = "results/vns_log.csv"
    best_path: str = "results/vns_best.json"
    verbose: bool = True
    print_every: int = 10  # print every N iterations

# -------------------------- Bounds & sampling ------------------------------ #
def _bounds_for(op: str) -> Dict[str, Tuple[float, float]]:
    b = OPS.BOUNDS
    if op == "gamma":    return {"g": b["gamma"]["g"]}
    if op == "gauss":    return {"sigma": b["gauss"]["sigma"], "ksize": b["gauss"]["ksize"]}
    if op == "unsharp":  return {"radius": b["unsharp"]["radius"], "amount": b["unsharp"]["amount"], "thresh": b["unsharp"]["thresh"]}
    if op == "cstretch": return {"p_low": b["cstretch"]["p_low"], "p_high": b["cstretch"]["p_high"]}
    if op == "he":       return {}
    raise ValueError(f"Unknown op: {op}")

def _sample_params(op: str, rng: np.random.Generator) -> Dict[str, Any]:
    b = OPS.BOUNDS
    if op == "gamma":
        lo, hi = b["gamma"]["g"]; return {"g": float(rng.uniform(lo, hi))}
    if op == "gauss":
        slo, shi = b["gauss"]["sigma"]
        return {"sigma": float(rng.uniform(slo, shi)), "ksize": int(rng.choice(GAUSS_KSIZES))}
    if op == "unsharp":
        rlo, rhi = b["unsharp"]["radius"]; alo, ahi = b["unsharp"]["amount"]; tlo, thi = b["unsharp"]["thresh"]
        return {"radius": float(rng.uniform(rlo, rhi)), "amount": float(rng.uniform(alo, ahi)), "thresh": float(rng.uniform(tlo, thi))}
    if op == "he":
        return {}
    if op == "cstretch":
        pl_lo, pl_hi = b["cstretch"]["p_low"]; ph_lo, ph_hi = b["cstretch"]["p_high"]
        p_low = float(rng.uniform(pl_lo, pl_hi))
        p_high_min = min(ph_hi, max(ph_lo, p_low + 5.0))
        p_high = float(rng.uniform(p_high_min, ph_hi))
        return {"p_low": p_low, "p_high": p_high}
    raise ValueError(op)

# --------------------------- Neighbourhood moves --------------------------- #
def _jitter_param(rng: np.random.Generator, step: Step, strength: float) -> None:
    """Jitter a single parameter of a step."""
    bounds = _bounds_for(step.op)
    if not bounds:
        return
    key = rng.choice(list(bounds.keys()))
    lo, hi = bounds[key]
    if key == "ksize":
        # discrete neighbor
        v = int(step.params.get(key, 5))
        idx = GAUSS_KSIZES.index(v) if v in GAUSS_KSIZES else 1
        idx = int(np.clip(idx + rng.choice([-1, 1]), 0, len(GAUSS_KSIZES)-1))
        step.params[key] = int(GAUSS_KSIZES[idx])
    else:
        v = float(step.params.get(key, (lo + hi) / 2.0))
        scale = (hi - lo) * strength
        nv = float(v + rng.normal(0.0, scale))
        # reflect and clip
        if nv < lo: nv = lo + (lo - nv)
        if nv > hi: nv = hi - (nv - hi)
        step.params[key] = float(np.clip(nv, lo, hi))

    # enforce cstretch constraint
    if step.op == "cstretch":
        pl = float(step.params.get("p_low"))
        ph = float(step.params.get("p_high"))
        if ph < pl + 5.0:
            step.params["p_high"] = min(OPS.BOUNDS["cstretch"]["p_high"][1], pl + 5.0)

def _jitter_all_params(rng: np.random.Generator, step: Step, strength: float) -> None:
    bounds = _bounds_for(step.op)
    for key, (lo, hi) in bounds.items():
        if key == "ksize":
            # small discrete hop
            v = int(step.params.get(key, 5))
            idx = GAUSS_KSIZES.index(v) if v in GAUSS_KSIZES else 1
            idx = int(np.clip(idx + rng.choice([-1, 0, 1]), 0, len(GAUSS_KSIZES)-1))
            step.params[key] = int(GAUSS_KSIZES[idx])
        else:
            v = float(step.params.get(key, (lo + hi) / 2.0))
            nv = float(v + rng.normal(0.0, (hi - lo) * strength))
            if nv < lo: nv = lo + (lo - nv)
            if nv > hi: nv = hi - (nv - hi)
            step.params[key] = float(np.clip(nv, lo, hi))
    if step.op == "cstretch":
        pl = float(step.params["p_low"])
        ph = float(step.params["p_high"])
        if ph < pl + 5.0:
            step.params["p_high"] = min(OPS.BOUNDS["cstretch"]["p_high"][1], pl + 5.0)

def _replace_op(rng: np.random.Generator, pipe: Pipeline, pos: int) -> None:
    new_op = rng.choice(list(OPS.BOUNDS.keys()))
    pipe[pos] = Step(str(new_op), _sample_params(str(new_op), rng))

def _swap_neighbors(rng: np.random.Generator, pipe: Pipeline) -> None:
    if len(pipe) < 2: return
    i = int(rng.integers(0, len(pipe)-1))
    pipe[i], pipe[i+1] = pipe[i+1], pipe[i]

def _insert_step(rng: np.random.Generator, pipe: Pipeline, L_max: int) -> None:
    if len(pipe) >= L_max: return
    pos = int(rng.integers(0, len(pipe)+1))
    op = str(rng.choice(list(OPS.BOUNDS.keys())))
    pipe.insert(pos, Step(op, _sample_params(op, rng)))

def _delete_step(rng: np.random.Generator, pipe: Pipeline, L_min: int) -> None:
    if len(pipe) <= L_min: return
    del pipe[int(rng.integers(0, len(pipe)))]

# ----------------------------- VNS primitives ------------------------------ #
def _shake(rng: np.random.Generator, pipe: Pipeline, k: int, cfg: VNSConfig) -> Pipeline:
    """Generate a neighbor in Nk by applying one k-specific modification."""
    child = [Step(s.op, dict(s.params)) for s in pipe]

    if k == 1:
        # N1: jitter a single param on a random step
        pos = int(rng.integers(0, len(child)))
        _jitter_param(rng, child[pos], cfg.jitter_strength)
    elif k == 2:
        # N2: jitter all params of one step
        pos = int(rng.integers(0, len(child)))
        _jitter_all_params(rng, child[pos], cfg.jitter_strength)
    elif k == 3:
        # N3: structural (swap neighbors OR replace op at pos)
        if rng.random() < 0.5 and len(child) >= 2:
            _swap_neighbors(rng, child)
        else:
            pos = int(rng.integers(0, len(child)))
            _replace_op(rng, child, pos)
    elif k == 4:
        # N4: length change (insert or delete)
        if rng.random() < 0.5:
            _insert_step(rng, child, cfg.L_max)
        else:
            _delete_step(rng, child, cfg.L_min)
    else:
        # fallback to small param jitter
        pos = int(rng.integers(0, len(child)))
        _jitter_param(rng, child[pos], cfg.jitter_strength)

    # enforce length limits
    if len(child) < cfg.L_min:
        _insert_step(rng, child, cfg.L_max)
    if len(child) > cfg.L_max:
        _delete_step(rng, child, cfg.L_min)
    return child

def _local_search(rng: np.random.Generator, cand: Pipeline, F_cand: float,
                  pairs: List[ImagePair], cfg: VNSConfig, memo: Memo) -> Tuple[Pipeline, float, Dict[str, float]]:
    """
    First-improvement hill-climb around `cand` using small param jitters.
    Returns best neighbor found (or the candidate itself).
    """
    best = [Step(s.op, dict(s.params)) for s in cand]
    best_F = F_cand
    best_means = None

    for _ in range(cfg.ls_trials):
        nb = [Step(s.op, dict(s.params)) for s in best]
        pos = int(rng.integers(0, len(nb)))
        _jitter_param(rng, nb[pos], cfg.jitter_strength * 0.75)  # a bit smaller in LS
        F_nb, means_nb = evaluate_pipeline(nb, pairs, mode=cfg.objective, weights=cfg.weights, cache=memo)
        if F_nb < best_F - 1e-9:
            best, best_F, best_means = nb, F_nb, means_nb
            # first improvement: keep going around the new point
    if best_means is None:
        # no improvement; fetch means for the original candidate
        _, best_means = evaluate_pipeline(best, pairs, mode=cfg.objective, weights=cfg.weights, cache=memo)
    return best, best_F, best_means

# --------------------------------- VNS loop -------------------------------- #
@dataclass
class VNSResult:
    best_pipe: Pipeline
    best_F: float
    best_means: Dict[str, float]
    history: List[Dict[str, float]]

def run_vns(train_pairs: List[ImagePair], cfg: VNSConfig, start_pipe: Pipeline | None = None) -> VNSResult:
    rng = set_seed(cfg.seed)
    t0 = time.time()
    memo = Memo()
    log = CSVLogger(cfg.log_path, ["iter","k","best_F","best_SSIM","best_PSNR","best_MSE"])

    # initial solution
    if start_pipe is None:
        curr = random_pipeline(rng, L=None, L_min=cfg.L_min, L_max=cfg.L_max)
    else:
        curr = [Step(s.op, dict(s.params)) for s in start_pipe]

    curr_F, curr_means = evaluate_pipeline(curr, train_pairs, mode=cfg.objective, weights=cfg.weights, cache=memo)
    best = [Step(s.op, dict(s.params)) for s in curr]
    best_F, best_means = curr_F, dict(curr_means)

    history: List[Dict[str, float]] = []
    k = 1
    it = 0
    while it < cfg.max_iters:
        # Shaking in Nk
        cand = _shake(rng, curr, k, cfg)
        F_cand, _ = evaluate_pipeline(cand, train_pairs, mode=cfg.objective, weights=cfg.weights, cache=memo)

        # Local search from shaken solution
        nb, F_nb, means_nb = _local_search(rng, cand, F_cand, train_pairs, cfg, memo)

        # Acceptance
        if F_nb < curr_F - 1e-9:
            curr, curr_F, curr_means = nb, F_nb, means_nb
            if curr_F < best_F - 1e-9:
                best, best_F, best_means = [Step(s.op, dict(s.params)) for s in curr], curr_F, dict(curr_means)
            k = 1  # improvement: restart at first neighborhood
        else:
            k += 1
            if k > cfg.K:
                k = 1  # cycle neighborhoods

        log.write({"iter": it, "k": k, "best_F": best_F,
                   "best_SSIM": best_means["SSIM"], "best_PSNR": best_means["PSNR"], "best_MSE": best_means["MSE"]})
        history.append({"iter": it, "k": k, "best_F": best_F,
                        "best_SSIM": best_means["SSIM"], "best_PSNR": best_means["PSNR"], "best_MSE": best_means["MSE"]})
        
        # ---------- ADDED: progress print right after history.append(...) ----------
        if cfg.verbose and (it % cfg.print_every == 0 or it == cfg.max_iters - 1):
            print(f"[VNS] it {it+1}/{cfg.max_iters} k={k} | "
                  f"bestF={best_F:.4f} SSIM={best_means['SSIM']:.4f} "
                  f"PSNR={best_means['PSNR']:.2f} MSE={best_means['MSE']:.5f}",
                  flush=True)
        # ---------------------------------------------------------------------------
        
        it += 1

    # Save best
    with open(cfg.best_path, "w", encoding="utf-8") as f:
        json.dump(pipe_to_dict(best), f, indent=2)
        
    print(f"[VNS] done in {time.time()-t0:.1f}s | bestF={best_F:.4f} SSIM={best_means['SSIM']:.4f}", flush=True)
    

    return VNSResult(best_pipe=best, best_F=best_F, best_means=best_means, history=history)

# ------------------------------- Entry point ------------------------------ #
if __name__ == "__main__":
    import os, json
    pairs = list_train_pairs()

    # Start from GA best if available
    start = None
    if os.path.exists("results/ga_best.json"):
        with open("results/ga_best.json","r",encoding="utf-8") as f:
            start = pipe_from_dict(json.load(f))

    # Real run defaults (assignment-ready)
    cfg = VNSConfig(
        seed=1,
        max_iters=300,
        K=4,
        ls_trials=10,
        jitter_strength=0.12,
        L_min=4, L_max=6,
        objective="ssim",
        weights=(1.0, 0.1, 0.5),
        log_path="results/vns_log.csv",
        best_path="results/vns_best.json",
        verbose=True, print_every=10,
    )
    res = run_vns(pairs, cfg, start_pipe=start)
    print("Best F:", res.best_F)
    print("Best means:", res.best_means)
    print("Best pipe:", pretty(res.best_pipe))
