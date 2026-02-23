from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass


@dataclass
class PatchConfig:
    patch_size: int = 64
    stride: int = 32
    min_plume_frac: float = 0.01
    keep_negative_ratio: float = 0.2


def make_patches(out_dir: str, no2: np.ndarray, delta: np.ndarray, u: np.ndarray, v: np.ndarray, plume_mask: np.ndarray, cfg: PatchConfig):
    '''
    Save patches as NPZ for ML training.

    Each patch contains:
      - X: stacked channels [no2, delta, u, v]
      - Y: plume mask (0/1)
    '''
    os.makedirs(out_dir, exist_ok=True)
    h, w = no2.shape
    ps, st = cfg.patch_size, cfg.stride

    patches, labels = [], []
    for i in range(0, max(1, h - ps + 1), st):
        for j in range(0, max(1, w - ps + 1), st):
            m = plume_mask[i:i+ps, j:j+ps]
            x = np.stack([no2[i:i+ps, j:j+ps], delta[i:i+ps, j:j+ps], u[i:i+ps, j:j+ps], v[i:i+ps, j:j+ps]], axis=0)
            patches.append(x.astype(np.float32))
            labels.append(m.astype(np.uint8))

    X = np.stack(patches, axis=0)
    Y = np.stack(labels, axis=0)

    plume_frac = Y.mean(axis=(1, 2))
    pos_idx = np.where(plume_frac >= cfg.min_plume_frac)[0]
    neg_idx = np.where(plume_frac < cfg.min_plume_frac)[0]

    keep_neg = int(len(pos_idx) * cfg.keep_negative_ratio)
    if keep_neg < len(neg_idx):
        rng = np.random.default_rng(0)
        neg_idx = rng.choice(neg_idx, size=keep_neg, replace=False)

    keep = np.concatenate([pos_idx, neg_idx]) if len(pos_idx) else neg_idx
    keep = np.sort(keep)

    out_path = os.path.join(out_dir, "patches.npz")
    np.savez_compressed(out_path, X=X[keep], Y=Y[keep])
    return out_path, int(len(pos_idx)), int(len(neg_idx)), int(len(keep))
