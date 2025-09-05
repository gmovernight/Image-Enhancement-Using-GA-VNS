# src/dataset.py
"""
Strict dataset pairing for two naming schemes (case-insensitive):

Train (in data/train):
  - Distorted:     imgN.<ext>
  - Ground truth:  imgN_gt.<ext>

Test (in data/test):
  - Distorted:     img_tN.<ext>
  - Ground truth:  img_tN_gt.<ext>

N is a positive integer (1, 2, ...). Extensions can differ between distorted
and GT. Only files directly in the folder are considered (no recursion).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Pattern, Tuple
import re

from .utils import read_image_gray01

# Extension preference (lowercase)
_IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
_EXT_SET = set(_IMG_EXTS)

# Patterns
_PAT_DIST_TRAIN: Pattern[str] = re.compile(r"^img(\d+)$", re.IGNORECASE)
_PAT_GT_TRAIN:   Pattern[str] = re.compile(r"^img(\d+)_gt$", re.IGNORECASE)

_PAT_DIST_TEST: Pattern[str] = re.compile(r"^img_t(\d+)$", re.IGNORECASE)
_PAT_GT_TEST:   Pattern[str] = re.compile(r"^img_t(\d+)_gt$", re.IGNORECASE)


@dataclass(frozen=True)
class ImagePair:
    """Holds a distorted/ground-truth pair (paths only)."""
    distorted: Path
    gt: Path
    id: str  # e.g., "img1" or "img_t1"


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in _EXT_SET


def _best_by_ext(paths: List[Path]) -> Path:
    """
    If multiple files exist for the same ID with different extensions,
    pick according to _IMG_EXTS preference order (e.g., .png first).
    """
    if len(paths) == 1:
        return paths[0]
    rank: Dict[str, int] = {ext: i for i, ext in enumerate(_IMG_EXTS)}
    return sorted(paths, key=lambda p: rank.get(p.suffix.lower(), 999))[0]


def _list_pairs_strict(root: Path,
                       pat_dist: Pattern[str],
                       pat_gt: Pattern[str],
                       id_fmt: str) -> List[ImagePair]:
    """
    Generic strict pairing under a naming scheme in a single folder.
    - pat_dist extracts N from distorted stem.
    - pat_gt   extracts N from GT stem.
    - id_fmt   formats the id string from N (e.g., "img{}" or "img_t{}").
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    files = [p for p in root.iterdir() if _is_image(p)]

    dist_map: Dict[int, List[Path]] = {}
    gt_map: Dict[int, List[Path]] = {}

    for p in files:
        stem = p.stem
        m_d = pat_dist.match(stem)
        if m_d:
            n = int(m_d.group(1))
            dist_map.setdefault(n, []).append(p)
            continue
        m_g = pat_gt.match(stem)
        if m_g:
            n = int(m_g.group(1))
            gt_map.setdefault(n, []).append(p)
            continue

    Ns = sorted(set(dist_map.keys()) & set(gt_map.keys()))
    pairs: List[ImagePair] = []
    for n in Ns:
        dpath = _best_by_ext(dist_map[n])
        gpath = _best_by_ext(gt_map[n])
        pairs.append(ImagePair(distorted=dpath, gt=gpath, id=id_fmt.format(n)))
    return pairs


def list_train_pairs(root: str | Path = "data/train") -> List[ImagePair]:
    """Return aligned pairs from data/train using imgN/imgN_gt scheme."""
    return _list_pairs_strict(Path(root), _PAT_DIST_TRAIN, _PAT_GT_TRAIN, "img{}")


def list_test_pairs(root: str | Path = "data/test") -> List[ImagePair]:
    """Return aligned pairs from data/test using img_tN/img_tN_gt scheme."""
    return _list_pairs_strict(Path(root), _PAT_DIST_TEST, _PAT_GT_TEST, "img_t{}")


def load_pair(pair: ImagePair):
    """Read images as grayscale float ∈[0,1]. Returns (I_distorted, I_gt)."""
    I = read_image_gray01(pair.distorted)
    GT = read_image_gray01(pair.gt)
    if I.shape != GT.shape:
        raise ValueError(
            f"Shape mismatch for pair {pair.id}: {I.shape} vs {GT.shape} "
            f"({pair.distorted.name} vs {pair.gt.name})"
        )
    return I, GT


# ---------------------- Quick manual self-test (optional) ------------------ #
if __name__ == "__main__":
    tr = list_train_pairs()
    te = list_test_pairs()
    print(f"Found {len(tr)} train pairs:", [p.id for p in tr])
    print(f"Found {len(te)} test  pairs:", [p.id for p in te])
    if tr:
        I, GT = load_pair(tr[0])
        print("Sample train pair:", tr[0].id, I.shape, I.dtype, GT.shape, GT.dtype)