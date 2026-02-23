from __future__ import annotations
import numpy as np


def robust_background_upwind(field, dx_km, dy_km, wind_dir_deg: float, upwind_half_angle: float = 60.0):
    '''
    Background as the median value in an upwind sector (relative to downwind direction).
    Returns (background_median, background_MAD).
    '''
    ang_math = (np.degrees(np.arctan2(dy_km, dx_km)) + 360.0) % 360.0  # 0°=East, 90°=North
    ang = (90.0 - ang_math) % 360.0                                  # 0°=North, 90°=East
    upwind_dir = (wind_dir_deg + 180.0) % 360.0
    dtheta = np.abs(((ang - upwind_dir + 180) % 360) - 180)

    upwind_mask = dtheta <= upwind_half_angle
    vals = field[upwind_mask]
    vals = vals[np.isfinite(vals)]

    if vals.size < 50:
        bg = np.nanpercentile(field, 20)
        mad = np.nanmedian(np.abs(field - np.nanmedian(field)))
        return float(bg), float(mad)

    bg = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - bg))
    return float(bg), float(mad)
