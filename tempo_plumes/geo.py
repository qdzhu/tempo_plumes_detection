from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def wind_to_dir_deg(u: float, v: float) -> float:
    '''
    Convert (eastward u, northward v) to downwind direction in degrees.

    Meteorological convention (downwind direction):
      - 0° = North
      - 90° = East
      - 180° = South
      - 270° = West
    '''
    angle_math = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0  # 0°=East, 90°=North
    angle_met = (90.0 - angle_math) % 360.0                     # rotate so 0°=North
    return float(angle_met)


def build_hrrr_tree(hrrr_lon, hrrr_lat) -> cKDTree:
    '''
    Build a cKDTree from HRRR grid lon/lat for use with regrid_hrrr_to_tempo_nn.
    Pre-building and reusing the tree avoids rebuilding it for every plant.
    '''
    pts = np.column_stack([hrrr_lon.ravel(), hrrr_lat.ravel()])
    return cKDTree(pts)


def regrid_hrrr_to_tempo_nn(hrrr_u, hrrr_v, hrrr_lon, hrrr_lat, tempo_lon, tempo_lat,
                             tree: cKDTree | None = None):
    '''
    Nearest-neighbor regridding of HRRR winds onto TEMPO grid in lon/lat space.
    Pass a pre-built tree (from build_hrrr_tree) to avoid rebuilding it each call.
    '''
    if tree is None:
        tree = build_hrrr_tree(hrrr_lon, hrrr_lat)

    q = np.column_stack([tempo_lon.ravel(), tempo_lat.ravel()])
    _, idx = tree.query(q, k=1)

    u_on = hrrr_u.ravel()[idx].reshape(tempo_lon.shape).astype(np.float32)
    v_on = hrrr_v.ravel()[idx].reshape(tempo_lon.shape).astype(np.float32)
    return u_on, v_on


def crop_window(field, lon, lat, plant_lon, plant_lat, half_size_km: float = 80.0):
    '''
    Crop a square window centered at (plant_lon, plant_lat) using an approximate km/deg conversion.
    Returns cropped field/lon/lat plus dx_km/dy_km relative to plant location.
    '''
    lat0 = float(plant_lat)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(lat0))

    dx_km = (lon - plant_lon) * km_per_deg_lon
    dy_km = (lat - plant_lat) * km_per_deg_lat

    mask = (np.abs(dx_km) <= half_size_km) & (np.abs(dy_km) <= half_size_km)
    ii, jj = np.where(mask)
    if ii.size < 100:
        raise ValueError("Too few pixels in crop window. Increase half_size_km or check plant lon/lat.")

    i0, i1 = ii.min(), ii.max() + 1
    j0, j1 = jj.min(), jj.max() + 1

    return (
        field[i0:i1, j0:j1],
        lon[i0:i1, j0:j1],
        lat[i0:i1, j0:j1],
        dx_km[i0:i1, j0:j1],
        dy_km[i0:i1, j0:j1],
        (i0, i1, j0, j1),
    )
