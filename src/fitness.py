# src/fitness.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Any

from .dataset import ImagePair
from .pipeline import Pipeline, pipeline_hash
from .objective import objective
from .utils import stable_hash


@dataclass
class Memo:
    """In-memory cache: key -> (F, means)."""
    store: Dict[str, Tuple[float, Dict[str, float]]] | None = None
    def __post_init__(self):
        if self.store is None:
            self.store = {}
    def __len__(self) -> int:
        return len(self.store)  # type: ignore[return-value]


def _eval_key(pipe: Pipeline,
              pairs: List[ImagePair],
              mode: str,
              weights: Tuple[float, float, float]) -> str:
    """Stable cache key over pipeline + dataset IDs + objective settings."""
    return stable_hash({
        "pipe": pipeline_hash(pipe),
        "pairs": tuple(p.id for p in pairs),
        "mode": mode,
        "weights": tuple(float(w) for w in weights),
    })


def evaluate_pipeline(pipe: Pipeline,
                      pairs: List[ImagePair],
                      mode: str = "ssim",
                      weights: Tuple[float, float, float] = (1.0, 0.1, 0.5),
                      cache: Memo | None = None) -> Tuple[float, Dict[str, float]]:
    """
    Compute fitness to MINIMIZE for `pipe` on `pairs` using `objective(...)`.
    If `cache` is provided, re-use previous results for identical evaluations.
    Returns (F, mean_metrics_dict).
    """
    key = _eval_key(pipe, pairs, mode, weights)
    if cache is not None and key in cache.store:           # type: ignore[index]
        return cache.store[key]                            # type: ignore[index]
    F, means = objective(pipe, pairs, mode=mode, weights=weights)
    if cache is not None:
        cache.store[key] = (F, means)                      # type: ignore[index]
    return F, means
