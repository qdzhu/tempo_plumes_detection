from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops

from .background import robust_background_upwind
from .geo import wind_to_dir_deg


def detect_plume(
    delta_field,
    dx_km,
    dy_km,
    u,
    v,
    k_mad: float = 4.0,
    smooth_sigma: float = 1.0,
    min_area_px: int = 20,
    max_upwind_fraction: float = 0.25,
    align_cos_thresh: float = 0.7,
):
    '''
    Rule-based plume detection:
      1) Smooth anomaly
      2) Robust threshold with upwind background + k*MAD
      3) Connected components
      4) Keep components connected to source vicinity AND aligned with wind AND predominantly downwind
    '''
    u0 = float(np.nanmedian(u))
    v0 = float(np.nanmedian(v))
    wind_dir = wind_to_dir_deg(u0, v0)

    dn = gaussian_filter(np.array(delta_field, dtype=np.float32), sigma=smooth_sigma)

    bg, mad = robust_background_upwind(dn, dx_km, dy_km, wind_dir)
    thr = bg + k_mad * (mad + 1e-12)

    binary = (dn > thr) & np.isfinite(dn)
    lab = label(binary, connectivity=2)
    props = regionprops(lab)

    if len(props) == 0:
        return np.zeros_like(binary, dtype=bool), {"detected": False, "reason": "no_components"}

    dist2 = dx_km**2 + dy_km**2
    w = np.array([np.sin(np.deg2rad(wind_dir)), np.cos(np.deg2rad(wind_dir))], dtype=np.float32)

    best = None
    best_score = -np.inf

    for p in props:
        if p.area < min_area_px:
            continue

        rr, cc = p.coords[:, 0], p.coords[:, 1]

        if not np.any(dist2[rr, cc] <= 10.0**2):
            continue

        X = np.column_stack([dx_km[rr, cc], dy_km[rr, cc]])
        X = X[np.all(np.isfinite(X), axis=1)]
        if X.shape[0] < 10:
            continue

        Xc = X - X.mean(axis=0, keepdims=True)
        C = np.cov(Xc.T)
        evals, evecs = np.linalg.eigh(C)
        main = evecs[:, np.argmax(evals)]

        align_cos = float(np.abs(np.dot(main, w)))
        if align_cos < align_cos_thresh:
            continue

        proj = X @ w
        down = dn[rr, cc][proj > 0]
        up = dn[rr, cc][proj < 0]
        down_sum = float(np.nansum(down))
        up_sum = float(np.nansum(up))
        frac_up = up_sum / (down_sum + up_sum + 1e-12)
        if frac_up > max_upwind_fraction:
            continue

        score = float((np.nanmax(dn[rr, cc]) - bg) * align_cos * (1.0 - frac_up))
        if score > best_score:
            best_score = score
            best = p

    if best is None:
        return np.zeros_like(binary, dtype=bool), {"detected": False, "reason": "no_component_pass_filters"}

    plume_mask = (lab == best.label)
    rr, cc = best.coords[:, 0], best.coords[:, 1]
    total_enh = float(np.nansum((dn[rr, cc] - bg).clip(min=0)))

    metrics = {
        "detected": True,
        "wind_dir_deg": float(wind_dir),
        "u_med": float(u0),
        "v_med": float(v0),
        "bg": float(bg),
        "mad": float(mad),
        "thr": float(thr),
        "area_px": int(best.area),
        "peak_enhancement": float(np.nanmax(dn[rr, cc] - bg)),
        "total_enhancement": float(total_enh),
        "score": float(best_score),
    }
    return plume_mask, metrics
