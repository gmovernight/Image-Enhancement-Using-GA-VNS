# Image Enhancement with GA & VNS 🚀🖼️

_Evolve image-enhancement pipelines that **actually** make pictures look better—automatically._  
This repo trains two search algorithms—**Genetic Algorithm (GA)** and **Variable Neighbourhood Search (VNS)**—to discover the best sequence of classical ops (gamma, Gaussian blur, unsharp masking, histogram equalisation, contrast stretching) for grayscale images.

> **Use case:** You have a small set of distorted ➜ ground-truth pairs for training, and a test set for final reporting. We learn the pipeline that maximizes quality (SSIM / PSNR / MSE).

---

## ✨ Highlights

- 🔌 **Plug-and-play pipeline**: compose `gamma → gauss → unsharp → he → cstretch` with safe bounds & clamping.
- 🧠 **Two optimisers**: GA (global search) + VNS (systematic local neighbourhood jumps).
- 📊 **Built-in metrics**: MSE, PSNR, SSIM (with a weighted objective option).
- 🧪 **Reproducible**: fixed seeds, deterministic pairing, clean logs, and JSON artifacts.
- 🛠️ **CLI-first**: run everything via `python -m src.<module>` from repo root.
- 📁 **Neat outputs**: best pipelines in `results/`, enhanced test images in `outputs/`, metrics as CSV.

> Want to peek at results? Check `results/test_metrics.csv` and the images under `outputs/test` after you run the evaluator.

---

## 🧱 Project Structure

```
.
├─ src/
│  ├─ dataset.py     # strict pairing & loaders
│  ├─ ops.py         # gamma, gauss, unsharp, he, cstretch (+bounds/clamp)
│  ├─ pipeline.py    # pipeline schema, sampler, apply, pretty formatting
│  ├─ metrics.py     # MSE, PSNR, SSIM
│  ├─ objective.py   # SSIM-only or weighted (MSE/PSNR/SSIM) objective
│  ├─ fitness.py     # training loop glue + memoization cache
│  ├─ ga.py          # Genetic Algorithm runner
│  ├─ vns.py         # Variable Neighbourhood Search runner
│  └─ eval_test.py   # Evaluate saved best pipelines on test set
├─ data/
│  ├─ train/         # paired distorted/ground-truth train images
│  └─ test/          # paired distorted/ground-truth test images
├─ results/          # logs + best pipelines (*.json, *.csv)
└─ outputs/          # enhanced test images (by GA/VNS)
```

---

## 🗂️ Data Layout

Place your images like this (extensions can vary: `.png`, `.jpg`, `.tif`, …):

```
data/
├─ train/
│  ├─ img1.png
│  ├─ img1_gt.png
│  ├─ img2.jpg
│  ├─ img2_gt.tif
│  └─ ...
└─ test/
   ├─ img_t1.jpg
   ├─ img_t1_gt.png
   └─ ...
```

- Training: file stems match `imgN` ↔ `imgN_gt`  
- Testing:  file stems match `img_tN` ↔ `img_tN_gt`

---

## ⚙️ Installation

```bash
# From repo root
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -U pip
pip install numpy opencv-python scikit-image pandas tqdm
```

> If you prefer, use `requirements.txt` with these packages and run `pip install -r requirements.txt`.

---

## 🚴 Quickstart (3 commands)

> Always run from **repo root** and use `-m src.<module>` to avoid import issues.

### 1) Train GA
```bash
python -m src.ga
```
- Saves the best individual to `results/ga_best.json` and a log to `results/ga_log.csv`.

### 2) Train VNS
```bash
python -m src.vns
```
- Produces `results/vns_best.json` and `results/vns_log.csv`. If `ga_best.json` exists, VNS may start from it (nice warm-start).

### 3) Evaluate on the Test Set
```bash
python -m src.eval_test
```
- Enhances every test pair with GA and/or VNS best pipelines, saving images under `outputs/test/{GA,VNS}` and per-image metrics in `results/test_metrics.csv`.

---

## 🧰 Configuration Cheatsheet

Tweak defaults by editing the config/dataclass blocks inside the runners:

- **GA (`src/ga.py`)**
  - Common knobs: `seed`, `pop_size`, `generations`, `pc` (crossover), `pm` (mutation), `L_min/L_max` (pipeline length range), `objective` (`"ssim"` or `"weighted"`), and `weights` (α, β, γ) when using weighted.

- **VNS (`src/vns.py`)**
  - Common knobs: `seed`, `max_iters`, neighbourhood count, local-search depth, jitter strength, `L_min/L_max`, `objective`, `weights`.

- **Objective (`src/objective.py`)**
  - `"ssim"`: **F = 1 − mean(SSIM)** (minimise)  
  - `"weighted"`: **F = α·mean(MSE) − β·mean(PSNR) + γ·(1 − mean(SSIM))**

---

## 💡 Tips & Troubleshooting

- **Import error (`attempted relative import with no known parent package`)**  
  Run from repo root and use `python -m src.ga` / `src.vns` / `src.eval_test`.

- **No pairs found**  
  Double‑check the `data/train` & `data/test` naming (`imgN` with `imgN_gt`, `img_tN` with `img_tN_gt`).

- **Kernel sizes & bounds**  
  Gaussian kernel sizes snap to common odd values; percentile contrast stretch ensures `p_high ≥ p_low + margin`; all ops clamp to valid ranges.

- **Reproducibility**  
  Seeds are set for consistent runs. Change `seed` in the configs to explore variability.

---

## 🧪 Why GA? Why VNS?

- **GA** explores broadly with selection + crossover + mutation—great for large, rugged search spaces like pipeline design.  
- **VNS** systematically perturbs the current solution to escape local minima—great for polishing or jumping neighbourhoods once a good solution is found.

Use **GA ➜ VNS** as a strong baseline: GA finds a good region; VNS refines it.

---

## 📦 Artifacts You’ll See

- `results/ga_best.json`, `results/vns_best.json` – human‑readable best pipelines  
- `results/ga_log.csv`, `results/vns_log.csv` – progress over time  
- `results/test_metrics.csv` – per‑image MSE/PSNR/SSIM on the test set  
- `outputs/test/{GA,VNS}` – enhanced test images (for side‑by‑side viewing)

---

## 🙌 Acknowledgements

Built for **COS791 (2025) – Assignment 1**. Classical algorithm ideas draw on foundational work by Holland (GA) and by Mladenović & Hansen (VNS).

---

## 📣 Contributing

Issues and PRs welcome—clean code, small commits, and clear descriptions help a bunch. If you add new ops, keep parameter bounds safe and document them in `ops.py`.

---