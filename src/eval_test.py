# src/eval_test.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .dataset import list_test_pairs, load_pair
from .pipeline import pipe_from_dict, apply_pipeline, pretty
from .metrics import all_metrics
from .utils import save_image_gray01


def _load_pipe(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return pipe_from_dict(json.load(f))


def main():
    # Expect the real-run artifacts only
    ga_path = Path("results/ga_best.json")
    vns_path = Path("results/vns_best.json")

    if not ga_path.exists() and not vns_path.exists():
        raise SystemExit("No saved best pipelines found. Run GA/VNS first (ga_best.json / vns_best.json).")

    pairs = list_test_pairs()
    if not pairs:
        raise SystemExit("No test images found in data/test.")

    rows: List[Dict[str, object]] = []

    def eval_and_save(algo: str, path: Path):
        pipe = _load_pipe(path)
        out_dir = Path("outputs/test") / algo.lower()
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[EVAL] {algo}: {path}")
        print(f"       Pipeline: {pretty(pipe)}")

        for pair in pairs:
            I, GT = load_pair(pair)
            J = apply_pipeline(I, pipe)
            save_image_gray01(out_dir / f"{pair.id}.png", J)

            mets = all_metrics(J, GT)
            rows.append({
                "pair_id": pair.id,
                "algo": algo,
                "MSE": mets["MSE"],
                "PSNR": mets["PSNR"],
                "SSIM": mets["SSIM"],
            })

    if ga_path.exists():  eval_and_save("GA", ga_path)
    if vns_path.exists(): eval_and_save("VNS", vns_path)

    # Save per-image metrics
    df = pd.DataFrame(rows)
    out_csv = Path("results/test_metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[EVAL] Wrote {out_csv} with {len(df)} rows.")

    # Print per-algorithm means (for your report)
    if not df.empty:
        means = df.groupby("algo")[["MSE", "PSNR", "SSIM"]].mean().reset_index()
        print("\nPer-algorithm means on test set:")
        print(means.to_string(index=False))


if __name__ == "__main__":
    main()
