# src/ga.py
from __future__ import annotations
import json, math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import time
from pathlib import Path

try:
    from tqdm import trange
except Exception:
    trange = None


from .utils import set_seed
from .dataset import ImagePair, list_train_pairs
from .pipeline import (
    Step, Pipeline, random_pipeline, apply_pipeline,
    pipe_to_dict, pipe_from_dict, pretty
)
from .fitness import evaluate_pipeline, Memo
from . import ops as OPS
from .utils import CSVLogger


# ----------------------------- GA Hyperparams ------------------------------ #
@dataclass
class GAConfig:
    seed: int = 42
    pop_size: int = 40
    generations: int = 60
    tourn_k: int = 3
    pc: float = 0.9          # crossover prob
    pm: float = 0.25         # mutation prob (per individual)
    elitism: int = 2
    L_min: int = 4
    L_max: int = 6
    objective: str = "ssim"  # "ssim" or "weighted"
    weights: Tuple[float, float, float] = (1.0, 0.1, 0.5)  # used if weighted
    log_path: str = "results/ga_log.csv"
    best_path: str = "results/ga_best.json"
    verbose: bool = True
    print_every: int = 1  # print every N generations if tqdm not available


# ----------------------------- Helper: bounds ------------------------------ #
def _bounds_for(op: str) -> Dict[str, Tuple[float, float]]:
    b = OPS.BOUNDS
    if op == "gamma":    return {"g": b["gamma"]["g"]}
    if op == "gauss":    return {"sigma": b["gauss"]["sigma"], "ksize": b["gauss"]["ksize"]}
    if op == "unsharp":  return {"radius": b["unsharp"]["radius"], "amount": b["unsharp"]["amount"], "thresh": b["unsharp"]["thresh"]}
    if op == "cstretch": return {"p_low": b["cstretch"]["p_low"], "p_high": b["cstretch"]["p_high"]}
    if op == "he":       return {}
    raise ValueError(f"Unknown op: {op}")


# ---------------------------- Genetic operators --------------------------- #
GAUSS_KSIZES = (3, 5, 7, 9)

def _tournament_select(rng: np.random.Generator, fitnesses: List[float], k: int) -> int:
    # lower fitness is better
    idxs = rng.integers(0, len(fitnesses), size=k)
    best = int(idxs[0])
    best_f = fitnesses[best]
    for i in idxs[1:]:
        i = int(i)
        if fitnesses[i] < best_f:
            best = i; best_f = fitnesses[i]
    return best

def _one_point_cx(rng: np.random.Generator, a: Pipeline, b: Pipeline, L_min: int, L_max: int) -> Tuple[Pipeline, Pipeline]:
    if len(a) < 2 or len(b) < 2:
        return a.copy(), b.copy()
    ca = int(rng.integers(1, len(a)))  # cut after index ca-1
    cb = int(rng.integers(1, len(b)))
    c1 = a[:ca] + b[cb:]
    c2 = b[:cb] + a[ca:]

    # fix lengths to [L_min, L_max]
    def _fix(L: Pipeline) -> Pipeline:
        if len(L) > L_max:
            # trim from middle
            while len(L) > L_max:
                del L[int(len(L)//2)]
        while len(L) < L_min:
            # insert random steps (reuse op distribution)
            L.insert(int(rng.integers(0, len(L)+1)), _random_step_like(rng))
        return L
    return _fix(c1), _fix(c2)

def _random_step_like(rng: np.random.Generator) -> Step:
    # sample from registry uniformly
    op = rng.choice(list(OPS.BOUNDS.keys()))
    if op == "cstretch":  # BOUNDS includes only implemented ops
        pass
    return Step(op=op, params=_sample_params(op, rng))

def _sample_params(op: str, rng: np.random.Generator) -> Dict[str, Any]:
    b = OPS.BOUNDS
    if op == "gamma":
        lo, hi = b["gamma"]["g"]; return {"g": float(rng.uniform(lo, hi))}
    if op == "gauss":
        slo, shi = b["gauss"]["sigma"]; klo, khi = b["gauss"]["ksize"]
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

def _mutate_params(rng: np.random.Generator, step: Step, strength: float = 0.15) -> None:
    """Gaussian jitter on continuous params; snap to bounds. strength ~ fraction of range."""
    bounds = _bounds_for(step.op)
    for k, (lo, hi) in bounds.items():
        v = step.params.get(k, float(rng.uniform(lo, hi)))
        if k == "ksize":
            # discrete neighbor move
            idx = GAUSS_KSIZES.index(int(v)) if int(v) in GAUSS_KSIZES else 1
            move = int(rng.choice([-1, 0, 1]))
            idx = int(np.clip(idx + move, 0, len(GAUSS_KSIZES)-1))
            step.params[k] = int(GAUSS_KSIZES[idx])
        else:
            scale = (hi - lo) * strength
            nv = float(v + rng.normal(0.0, scale))
            # reflect at bounds
            if nv < lo: nv = lo + (lo - nv)
            if nv > hi: nv = hi - (nv - hi)
            nv = float(np.clip(nv, lo, hi))
            # keep cstretch constraint
            step.params[k] = nv

    # enforce cstretch constraint p_high ≥ p_low + 5
    if step.op == "cstretch":
        pl, ph = step.params["p_low"], step.params["p_high"]
        if ph < pl + 5.0:
            step.params["p_high"] = min(OPS.BOUNDS["cstretch"]["p_high"][1], pl + 5.0)

def _mutate_structure(rng: np.random.Generator, pipe: Pipeline, cfg: GAConfig) -> None:
    """Swap steps OR change op at a random position (resample params)."""
    if len(pipe) >= 2 and rng.random() < 0.5:
        i, j = int(rng.integers(0, len(pipe))), int(rng.integers(0, len(pipe)))
        pipe[i], pipe[j] = pipe[j], pipe[i]
    else:
        pos = int(rng.integers(0, len(pipe)))
        # change op
        new_op = str(rng.choice(list(OPS.BOUNDS.keys())))
        pipe[pos] = Step(new_op, _sample_params(new_op, rng))

def _mutate_length(rng: np.random.Generator, pipe: Pipeline, cfg: GAConfig) -> None:
    if rng.random() < 0.5 and len(pipe) > cfg.L_min:
        # delete
        del pipe[int(rng.integers(0, len(pipe)))]
    elif len(pipe) < cfg.L_max:
        # insert
        pos = int(rng.integers(0, len(pipe)+1))
        op = str(rng.choice(list(OPS.BOUNDS.keys())))
        pipe.insert(pos, Step(op, _sample_params(op, rng)))

def _mutate_individual(rng: np.random.Generator, ind: Pipeline, cfg: GAConfig) -> Pipeline:
    child = [Step(s.op, dict(s.params)) for s in ind]
    # param mutation on one or more steps
    n_changes = max(1, int(round(len(child) * 0.5)))
    for _ in range(n_changes):
        pos = int(rng.integers(0, len(child)))
        _mutate_params(rng, child[pos])
    # occasionally change structure / length
    if rng.random() < 0.5:
        _mutate_structure(rng, child, cfg)
    if rng.random() < 0.3:
        _mutate_length(rng, child, cfg)
    # clip length
    if len(child) < cfg.L_min:
        while len(child) < cfg.L_min:
            _mutate_length(rng, child, cfg)
    if len(child) > cfg.L_max:
        while len(child) > cfg.L_max:
            del child[int(rng.integers(0, len(child)))]
    return child


# --------------------------------- GA loop --------------------------------- #
@dataclass
class GAResult:
    best_pipe: Pipeline
    best_F: float
    best_means: Dict[str, float]
    history: List[Dict[str, float]]

def run_ga(train_pairs: List[ImagePair], cfg: GAConfig) -> GAResult:
    rng = set_seed(cfg.seed)
    memo = Memo()
    log = CSVLogger(cfg.log_path, ["gen","best_F","mean_F","best_SSIM","best_PSNR","best_MSE"])

    # init population
    pop: List[Pipeline] = [random_pipeline(rng, L=None, L_min=cfg.L_min, L_max=cfg.L_max) for _ in range(cfg.pop_size)]
    fits: List[float] = []
    stats: List[Dict[str, float]] = []
    for ind in pop:
        F, means = evaluate_pipeline(ind, train_pairs, mode=cfg.objective, weights=cfg.weights, cache=memo)
        fits.append(F); stats.append(means)

    # best-so-far
    best_idx = int(np.argmin(fits))
    best = [Step(s.op, dict(s.params)) for s in pop[best_idx]]
    best_F = float(fits[best_idx])
    best_means = dict(stats[best_idx])

    history: List[Dict[str, float]] = []

    t0 = time.time()
    gen_iter = trange(cfg.generations, desc="GA", unit="gen") if (trange and cfg.verbose) else range(cfg.generations)

    for gen in gen_iter:
        new_pop: List[Pipeline] = []
        # Elitism
        elite_idxs = list(np.argsort(fits))[:cfg.elitism]
        for ei in elite_idxs:
            new_pop.append([Step(s.op, dict(s.params)) for s in pop[int(ei)]])

        # Offspring
        while len(new_pop) < cfg.pop_size:
            i = _tournament_select(rng, fits, cfg.tourn_k)
            j = _tournament_select(rng, fits, cfg.tourn_k)
            p1 = pop[i]; p2 = pop[j]
            c1, c2 = _one_point_cx(rng, p1, p2, cfg.L_min, cfg.L_max) if rng.random() < cfg.pc else (p1.copy(), p2.copy())
            if rng.random() < cfg.pm: c1 = _mutate_individual(rng, c1, cfg)
            if rng.random() < cfg.pm and len(new_pop)+1 < cfg.pop_size: c2 = _mutate_individual(rng, c2, cfg)
            new_pop.append(c1)
            if len(new_pop) < cfg.pop_size: new_pop.append(c2)

        # Evaluate new pop
        pop = new_pop
        fits, stats = [], []
        for ind in pop:
            F, means = evaluate_pipeline(ind, train_pairs, mode=cfg.objective, weights=cfg.weights, cache=memo)
            fits.append(F); stats.append(means)

        # Gen best
        gi = int(np.argmin(fits))
        gen_best_F = float(fits[gi])
        gen_best_means = dict(stats[gi])
        if gen_best_F < best_F:
            best = [Step(s.op, dict(s.params)) for s in pop[gi]]
            best_F = gen_best_F
            best_means = gen_best_means

        row = {
            "gen": gen,
            "best_F": gen_best_F,
            "mean_F": float(np.mean(fits)),
            "best_SSIM": gen_best_means["SSIM"],
            "best_PSNR": gen_best_means["PSNR"],
            "best_MSE": gen_best_means["MSE"],
        }
        log.write(row)
        history.append(row)

        # Progress output
        if trange and cfg.verbose:
            gen_iter.set_postfix(bestF=gen_best_F, SSIM=gen_best_means["SSIM"], PSNR=gen_best_means["PSNR"], MSE=gen_best_means["MSE"], refresh=False)
        elif cfg.verbose and (gen % cfg.print_every == 0 or gen == cfg.generations-1):
            print(f"[GA] gen {gen+1}/{cfg.generations} | bestF={gen_best_F:.4f} SSIM={gen_best_means['SSIM']:.4f} "
                  f"PSNR={gen_best_means['PSNR']:.2f} MSE={gen_best_means['MSE']:.5f} meanF={row['mean_F']:.4f}", flush=True)

    print(f"[GA] done in {time.time()-t0:.1f}s | bestF={best_F:.4f} SSIM={best_means['SSIM']:.4f}", flush=True)
    
    # Save best pipeline to JSON
    with open(cfg.best_path, "w", encoding="utf-8") as f:
        json.dump(pipe_to_dict(best), f, indent=2)



    return GAResult(best_pipe=best, best_F=best_F, best_means=best_means, history=history)


# ------------------------------ Entry point ------------------------------ #
if __name__ == "__main__":
    pairs = list_train_pairs()
    # Real run defaults (assignment-ready)
    cfg = GAConfig(
        seed=1,
        pop_size=40,
        generations=60,
        tourn_k=3, pc=0.90, pm=0.25, elitism=2,
        log_path="results/ga_log.csv",
        best_path="results/ga_best.json",
        verbose=True, print_every=1,
    )
    res = run_ga(pairs, cfg)
    print("Best F:", res.best_F)
    print("Best means:", res.best_means)
    print("Best pipe:", pretty(res.best_pipe))