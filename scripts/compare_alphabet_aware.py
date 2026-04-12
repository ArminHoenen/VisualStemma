#!/usr/bin/env python3
"""
Match+prune two clusterings using:
- Centroids from clusters/<name>/cluster_i (many images per cluster)
- Single image vectors from clusters_alphabet/cluster_i (one image per cluster)
Steps:
1) Embed images -> centroid vectors (first clustering) and single vectors (second)
2) Hungarian assignment between centroid vectors and single vectors
3) Rank matched pairs by dissimilarity (distance between centroid and matched single)
4) Iteratively prune worst pair, recompute distance-matrix distance after each prune
5) Stop at 1/3 of original clusters
6) Output best stage (smallest matrix distance)

Embeddings:
- Prefer precomputed .npy next to each image if present.
- Else try to compute embeddings using torchvision ResNet50 (if torch+torchvision installed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_cluster_index(p: Path) -> int:
    """
    Extract integer i from folder name 'cluster_<i>' (or 'cluster-i', etc.).
    """
    m = re.search(r"cluster[_\-]?(\d+)", p.name)
    if not m:
        raise ValueError(f"Cannot parse cluster index from folder name: {p.name}")
    return int(m.group(1))


def list_cluster_dirs(root: Path) -> List[Path]:
    """
    Return cluster dirs sorted by index.
    """
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")
    dirs = [d for d in root.iterdir() if d.is_dir() and re.search(r"cluster", d.name)]
    dirs.sort(key=parse_cluster_index)
    return dirs


def list_images_in_dir(d: Path) -> List[Path]:
    imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    imgs.sort()
    return imgs


# --------------------------
# Embeddings
# --------------------------

_TORCH_BACKEND_READY = False
_torch_model = None
_torch_transform = None


def _try_init_torchvision_backend() -> bool:
    global _TORCH_BACKEND_READY, _torch_model, _torch_transform
    if _TORCH_BACKEND_READY:
        return True
    try:
        import torch
        import torch.nn as nn
        from PIL import Image  # noqa: F401
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = nn.Identity()
        model.eval()

        _torch_model = model
        _torch_transform = weights.transforms()
        _TORCH_BACKEND_READY = True
        return True
    except Exception:
        _TORCH_BACKEND_READY = False
        _torch_model = None
        _torch_transform = None
        return False


def _embed_with_torchvision(img_path: Path) -> np.ndarray:
    import torch
    from PIL import Image

    assert _torch_model is not None and _torch_transform is not None

    img = Image.open(img_path).convert("RGB")
    x = _torch_transform(img).unsqueeze(0)  # [1,3,H,W]
    with torch.no_grad():
        feat = _torch_model(x)  # [1,2048]
    v = feat.squeeze(0).cpu().numpy().astype(np.float32)
    return v


def load_image_embedding(
    img_path: Path,
    prefer_npy: bool = True,
) -> np.ndarray:
    """
    Load embedding vector for image.
    - If prefer_npy: try <image_stem>.npy (same directory) first.
      Example: img_0001.jpg -> img_0001.npy
    - Else fallback to torchvision ResNet50 if available.
    """
    if prefer_npy:
        npy_path = img_path.with_suffix(".npy")
        if npy_path.exists():
            v = np.load(npy_path)
            v = np.asarray(v).astype(np.float32).reshape(-1)
            return v

    if _try_init_torchvision_backend():
        v = _embed_with_torchvision(img_path)
        return v

    raise RuntimeError(
        f"No embedding found for {img_path} (missing .npy) and torchvision backend not available.\n"
        f"Either save embeddings as .npy next to each image, or install torch+torchvision."
    )


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def vector_distance(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        # 1 - cosine similarity
        aa = a / max(np.linalg.norm(a), 1e-12)
        bb = b / max(np.linalg.norm(b), 1e-12)
        return float(1.0 - np.dot(aa, bb))
    elif metric == "euclidean":
        return float(np.linalg.norm(a - b))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def pairwise_distance_matrix(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        Xn = l2_normalize_rows(X)
        D = 1.0 - (Xn @ Xn.T)
        # numerical cleanup
        D = np.clip(D, 0.0, 2.0)
        np.fill_diagonal(D, 0.0)
        return D.astype(np.float32)
    elif metric == "euclidean":
        D = cdist(X, X, metric="euclidean")
        return D.astype(np.float32)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# --------------------------
# Matrix distance (distance between distance matrices)
# --------------------------

def upper_triangle_vec(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return D[iu].astype(np.float64)


def zscore(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = float(np.mean(v))
    sd = float(np.std(v))
    return (v - mu) / max(sd, eps)


def matrix_distance(D1: np.ndarray, D2: np.ndarray, mode: str = "zrmse") -> float:
    """
    Compute a scalar similarity/distance between two distance matrices.

    Modes:
        - "zrmse": sqrt(mean((z1 - z2)^2))   (smaller = better)
        - "fro": Frobenius norm              (smaller = better)
        - "pearson": Pearson correlation     (larger = better)
    """
    if D1.shape != D2.shape:
        raise ValueError(f"Shape mismatch: {D1.shape} vs {D2.shape}")

    if D1.shape[0] < 2:
        return 0.0

    v1 = upper_triangle_vec(D1)
    v2 = upper_triangle_vec(D2)

    if mode == "zrmse":
        v1z = zscore(v1)
        v2z = zscore(v2)
        return float(np.sqrt(np.mean((v1z - v2z) ** 2)))

    elif mode == "fro":
        return float(np.linalg.norm(D1 - D2, ord="fro"))

    elif mode == "pearson":
        # Pearson correlation between upper triangles
        v1m = v1 - np.mean(v1)
        v2m = v2 - np.mean(v2)

        denom = (np.linalg.norm(v1m) * np.linalg.norm(v2m))
        if denom < 1e-12:
            return 0.0

        rho = float(np.dot(v1m, v2m) / denom)
        return rho

    else:
        raise ValueError(f"Unknown matrix distance mode: {mode}")


# --------------------------
# Main pipeline
# --------------------------

@dataclass
class ClusterItem:
    cluster_idx: int
    dir_path: Path
    image_paths: List[Path]
    vector: np.ndarray  # centroid for first set; single vector for second set


def build_centroid_items(
    clusters_root: Path,
    name: str,
    metric_for_centroid: str,
    prefer_npy: bool,
) -> List[ClusterItem]:
    """
    For clusters/<name>/cluster_i, compute centroid = mean of image embeddings (optionally normalized).
    """
    root = clusters_root / name
    cluster_dirs = list_cluster_dirs(root)

    items: List[ClusterItem] = []
    for d in cluster_dirs:
        imgs = list_images_in_dir(d)
        if len(imgs) == 0:
            raise ValueError(f"No images found in centroid cluster dir: {d}")

        vecs = []
        for img in imgs:
            vecs.append(load_image_embedding(img, prefer_npy=prefer_npy))
        X = np.stack(vecs, axis=0)

        centroid = np.mean(X, axis=0).astype(np.float32)

        # For cosine, normalizing centroids is usually sensible.
        if metric_for_centroid == "cosine":
            centroid = (centroid / max(np.linalg.norm(centroid), 1e-12)).astype(np.float32)

        items.append(
            ClusterItem(
                cluster_idx=parse_cluster_index(d),
                dir_path=d,
                image_paths=imgs,
                vector=centroid,
            )
        )
    items.sort(key=lambda x: x.cluster_idx)
    return items


def build_single_items(
    alphabet_root: Path,
    prefer_npy: bool,
) -> List[ClusterItem]:
    """
    For clusters_alphabet/cluster_i, load the single image vector (must contain exactly one image file).
    """
    cluster_dirs = list_cluster_dirs(alphabet_root)

    items: List[ClusterItem] = []
    for d in cluster_dirs:
        imgs = list_images_in_dir(d)
        if len(imgs) != 1:
            raise ValueError(f"Expected exactly 1 image in {d}, found {len(imgs)}")
        v = load_image_embedding(imgs[0], prefer_npy=prefer_npy).astype(np.float32)
        items.append(
            ClusterItem(
                cluster_idx=parse_cluster_index(d),
                dir_path=d,
                image_paths=imgs,
                vector=v,
            )
        )
    items.sort(key=lambda x: x.cluster_idx)
    return items


def hungarian_match(
    A: np.ndarray,  # centroids [K,d]
    B: np.ndarray,  # singles  [K,d]
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (row_ind, col_ind) from Hungarian solving min cost assignment for C[i,j]=dist(Ai,Bj).
    """
    if metric == "cosine":
        A2 = l2_normalize_rows(A)
        B2 = l2_normalize_rows(B)
        C = cdist(A2, B2, metric="cosine")
    elif metric == "euclidean":
        C = cdist(A, B, metric="euclidean")
    else:
        raise ValueError(f"Unknown metric: {metric}")

    row_ind, col_ind = linear_sum_assignment(C)
    return row_ind, col_ind


def export_pruned_sets(
    export_dir: Path,
    kept_centroid_items: List[ClusterItem],
    kept_single_items: List[ClusterItem],
    mapping: List[Tuple[int, int]],
):
    export_dir.mkdir(parents=True, exist_ok=True)
    out_a = export_dir / "centroids"
    out_b = export_dir / "alphabet"
    out_a.mkdir(exist_ok=True)
    out_b.mkdir(exist_ok=True)

    # Copy directories (cluster folders) into export targets
    for item in kept_centroid_items:
        dst = out_a / item.dir_path.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(item.dir_path, dst)

    for item in kept_single_items:
        dst = out_b / item.dir_path.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(item.dir_path, dst)

    summary = {
        "kept_centroid_clusters": [it.cluster_idx for it in kept_centroid_items],
        "kept_alphabet_clusters": [it.cluster_idx for it in kept_single_items],
        "mapping_centroid_to_alphabet": [{"centroid": a, "alphabet": b} for a, b in mapping],
    }
    with open(export_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Your <myname> inside clusters/<myname>/cluster_i")
    ap.add_argument("--clusters_root", default="clusters", type=str)
    ap.add_argument("--alphabet_root", default="clusters_alphabet", type=str)
    ap.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"],
                    help="Distance metric for vectors and matrices.")
    ap.add_argument("--matrix_metric", default="zrmse", choices=["zrmse", "fro","pearson"],
                    help="How to compute distance between distance matrices.")
    ap.add_argument("--prefer_npy", action="store_true",
                    help="Prefer loading <image>.npy embeddings; fallback to torchvision if not found.")
    ap.add_argument("--min_frac", default=1/3, type=float,
                    help="Stop pruning when remaining clusters <= ceil(K*min_frac). Default=1/3.")
    ap.add_argument("--rematch_each_step", action="store_true",
                    help="If set, recompute Hungarian after each prune (slower, sometimes better).")
    ap.add_argument("--export_dir", default=None, type=str,
                    help="If set, copy pruned clusters into this folder + write summary.json.")
    args = ap.parse_args()

    clusters_root = Path(args.clusters_root)
    alphabet_root = Path(args.alphabet_root)

    centroid_items = build_centroid_items(
        clusters_root=clusters_root,
        name=args.name,
        metric_for_centroid=args.metric,
        prefer_npy=args.prefer_npy,
    )
    single_items = build_single_items(
        alphabet_root=alphabet_root,
        prefer_npy=args.prefer_npy,
    )

    K1 = len(centroid_items)
    K2 = len(single_items)
    if K1 != K2:
        raise ValueError(f"Cluster count mismatch: centroids={K1} vs alphabet={K2}. They must be equal.")

    K = K1
    min_keep = int(math.ceil(K * float(args.min_frac)))
    min_keep = max(1, min_keep)

    A = np.stack([it.vector for it in centroid_items], axis=0)
    B = np.stack([it.vector for it in single_items], axis=0)

    # Initial assignment (or re-assigned each step if requested)
    def compute_assignment(Acur: np.ndarray, Bcur: np.ndarray) -> np.ndarray:
        rows, cols = hungarian_match(Acur, Bcur, metric=args.metric)
        # rows should be 0..n-1 in some order; convert to permutation p where B[p[i]] matches A[i]
        p = np.empty_like(cols)
        p[rows] = cols
        return p

    # We will keep a fixed assignment on full K by default.
    perm_full = compute_assignment(A, B)
    # reorder B + single_items according to centroid order
    B_perm = B[perm_full]
    single_items_perm = [single_items[j] for j in perm_full]

    # dissimilarity per matched pair (for pruning order)
    pair_dissim = np.array(
        [vector_distance(A[i], B_perm[i], metric=args.metric) for i in range(K)],
        dtype=np.float64,
    )
    prune_order = np.argsort(-pair_dissim)  # most dissimilar first

    # Evaluate stages
    best_dist = float("inf")
    best_keep_indices: List[int] = list(range(K))
    best_stage_size = K

    keep_mask = np.ones(K, dtype=bool)

    def eval_stage(keep_mask_local: np.ndarray) -> float:
        idx = np.where(keep_mask_local)[0]
        if len(idx) <= 1:
            return 0.0
        Acur = A[idx]
        Bcur = B[idx]
        if args.rematch_each_step:
            p = compute_assignment(Acur, Bcur)
            Bcur = Bcur[p]
        else:
            # keep B_perm aligned with A; if using fixed assignment, Bcur should come from B_perm
            Bcur = B_perm[idx]

        D1 = pairwise_distance_matrix(Acur, metric=args.metric)
        D2 = pairwise_distance_matrix(Bcur, metric=args.metric)
        return matrix_distance(D1, D2, mode=args.matrix_metric)

    # stage 0 (no pruning)
    d0 = eval_stage(keep_mask)
    best_dist = d0
    best_keep_indices = list(np.where(keep_mask)[0])
    best_stage_size = len(best_keep_indices)

    # Prune until min_keep
    pruned = 0
    for j in prune_order:
        if keep_mask.sum() <= min_keep:
            break
        keep_mask[j] = False
        pruned += 1

        d = eval_stage(keep_mask)
        if args.matrix_metric == "pearson":
            if d > best_dist:
                best_dist = d
                best_keep_indices = list(np.where(keep_mask)[0])
                best_stage_size = len(best_keep_indices)
        else:
            if d < best_dist:
                best_dist = d
                best_keep_indices = list(np.where(keep_mask)[0])
                best_stage_size = len(best_keep_indices)

    kept_centroids = [centroid_items[i] for i in best_keep_indices]
    kept_singles = [single_items_perm[i] for i in best_keep_indices]
    mapping = [(centroid_items[i].cluster_idx, single_items_perm[i].cluster_idx) for i in best_keep_indices]

    # Output
    print("=== Best pruning stage ===")
    print(f"Total clusters K: {K}")
    print(f"Stop threshold (min_keep): {min_keep}")
    print(f"Best stage size: {best_stage_size}")
    print(f"Distance between distance matrices ({args.matrix_metric}): {best_dist:.6f}")
    print()
    print("Kept centroid clusters (clusters/<name>/...):")
    for it in kept_centroids:
        print(f"  cluster_{it.cluster_idx} -> {it.dir_path}")
    print()
    print("Kept alphabet clusters (clusters_alphabet/...):")
    for it in kept_singles:
        print(f"  cluster_{it.cluster_idx} -> {it.dir_path}")
    print()
    print("Bijective mapping (centroid_cluster -> alphabet_cluster) for kept set:")
    for a_idx, b_idx in mapping:
        print(f"  cluster_{a_idx}  ->  cluster_{b_idx}")

    if args.export_dir:
        export_dir = Path(args.export_dir)
        export_pruned_sets(export_dir, kept_centroids, kept_singles, mapping)
        print()
        print(f"Exported pruned clusters + summary.json to: {export_dir.resolve()}")


if __name__ == "__main__":
    main()
