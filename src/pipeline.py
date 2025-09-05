# src/pipeline.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Sequence, Tuple, Any

import numpy as np

from .utils import clip01, stable_hash
from . import ops as OPS  # gamma, gauss, unsharp, he, cstretch + BOUNDS


# ---------------------------- Registry & schema ---------------------------- #
# Each op: function and ordered parameter names for (de)serialization
OP_REGISTRY: Dict[str, Tuple[Callable[..., np.ndarray], Sequence[str]]] = {
    "gamma":    (OPS.gamma,   ("g",)),
    "gauss":    (OPS.gauss,   ("sigma", "ksize")),
    "unsharp":  (OPS.unsharp, ("radius", "amount", "thresh")),
    "he":       (OPS.he,      ()),  # no params
    "cstretch": (OPS.cstretch,("p_low", "p_high")),
}

# Allowed discrete values for some params
GAUSS_KSIZES = (3, 5, 7, 9)


@dataclass
class Step:
    op: str
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "params": self.params}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Step":
        return Step(op=d["op"], params=dict(d.get("params", {})))


Pipeline = List[Step]


# ----------------------------- Random sampling ----------------------------- #
def _rand_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))

def _sample_params(op: str, rng: np.random.Generator) -> Dict[str, Any]:
    b = OPS.BOUNDS
    if op == "gamma":
        g_lo, g_hi = b["gamma"]["g"]
        return {"g": _rand_uniform(rng, g_lo, g_hi)}
    elif op == "gauss":
        s_lo, s_hi = b["gauss"]["sigma"]
        sigma = _rand_uniform(rng, s_lo, s_hi)
        ksize = int(rng.choice(GAUSS_KSIZES))
        return {"sigma": sigma, "ksize": ksize}
    elif op == "unsharp":
        r_lo, r_hi = b["unsharp"]["radius"]
        a_lo, a_hi = b["unsharp"]["amount"]
        t_lo, t_hi = b["unsharp"]["thresh"]
        return {
            "radius": _rand_uniform(rng, r_lo, r_hi),
            "amount": _rand_uniform(rng, a_lo, a_hi),
            "thresh": _rand_uniform(rng, t_lo, t_hi),
        }
    elif op == "he":
        return {}
    elif op == "cstretch":
        pl_lo, pl_hi = b["cstretch"]["p_low"]
        ph_lo, ph_hi = b["cstretch"]["p_high"]
        p_low = _rand_uniform(rng, pl_lo, pl_hi)
        # enforce p_high ≥ p_low + 5 (and within overall bounds)
        min_ph = max(ph_lo, p_low + 5.0)
        min_ph = min(min_ph, ph_hi)
        p_high = _rand_uniform(rng, min_ph, ph_hi)
        return {"p_low": p_low, "p_high": p_high}
    else:
        raise ValueError(f"Unknown op: {op}")


def random_step(rng: np.random.Generator) -> Step:
    op = rng.choice(list(OP_REGISTRY.keys()))
    return Step(op=op, params=_sample_params(op, rng))


def random_pipeline(rng: np.random.Generator, L: int | None = None,
                    L_min: int = 4, L_max: int = 6) -> Pipeline:
    if L is None:
        L = int(rng.integers(L_min, L_max + 1))
    L = int(np.clip(L, L_min, L_max))
    return [random_step(rng) for _ in range(L)]


# ------------------------------ Apply & utils ------------------------------ #
def apply_pipeline(img01: np.ndarray, pipe: Pipeline) -> np.ndarray:
    """Apply steps sequentially; returns clipped float image in [0,1]."""
    out = img01
    for st in pipe:
        func, order = OP_REGISTRY[st.op]
        args = [st.params[k] for k in order] if order else []
        out = func(out, *args) if args else func(out)
        out = clip01(out)
    return out


def pipeline_hash(pipe: Pipeline) -> str:
    """Stable hash for caching."""
    serial = [s.to_dict() for s in pipe]
    return stable_hash(serial)


def pipe_to_dict(pipe: Pipeline) -> List[Dict[str, Any]]:
    return [s.to_dict() for s in pipe]


def pipe_from_dict(d: List[Dict[str, Any]]) -> Pipeline:
    return [Step.from_dict(x) for x in d]


def pretty(pipe: Pipeline) -> str:
    parts = []
    for s in pipe:
        if s.params:
            kv = ", ".join(f"{k}={_fmtv(v)}" for k, v in s.params.items())
            parts.append(f"{s.op}({kv})")
        else:
            parts.append(f"{s.op}()")
    return " -> ".join(parts)


def _fmtv(v: Any) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


# ------------------------------ Self-test ---------------------------------- #
if __name__ == "__main__":
    import json
    from .dataset import list_train_pairs, load_pair
    from .utils import set_seed, save_image_gray01

    rng = set_seed(42)
    pairs = list_train_pairs()
    I, _ = load_pair(pairs[0])

    pipe = random_pipeline(rng)
    print("PIPE:", pretty(pipe))
    J = apply_pipeline(I, pipe)
    save_image_gray01("outputs/_pipe_example.png", J)

    js = json.dumps(pipe_to_dict(pipe), indent=2)
    print(js)