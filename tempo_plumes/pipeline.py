from __future__ import annotations

from scipy.spatial import cKDTree

from .io import load_tempo_no2_l3, load_hrrr_uv_mean5
from .geo import regrid_hrrr_to_tempo_nn, crop_window, wind_to_dir_deg
from .background import robust_background_upwind
from .detect import detect_plume


def load_scene(
    tempo_nc: str,
    hrrr_grib: str,
    tempo_var: str = "vertical_column_troposphere",
    hrrr_engine: str | None = None,
):
    '''
    Load raw TEMPO NO2 and HRRR wind arrays for a single file pair.
    Returns (no2, tlon, tlat, hu, hv, hlon, hlat) — full grids, no cropping or regridding.
    Use detect_for_plant() to process individual plants from these arrays.
    '''
    no2, tlon, tlat = load_tempo_no2_l3(tempo_nc, var_name=tempo_var)
    hu, hv, hlon, hlat = load_hrrr_uv_mean5(hrrr_grib, engine=hrrr_engine)
    return no2, tlon, tlat, hu, hv, hlon, hlat


def detect_for_plant(
    no2, tlon, tlat, hu, hv, hlon, hlat,
    plant_lon: float,
    plant_lat: float,
    half_size_km: float = 80.0,
    hrrr_tree: cKDTree | None = None,
):
    '''
    Run plume detection for one plant given pre-loaded full-grid arrays.
    Crops the TEMPO window first, then regrids HRRR onto the small cropped
    window (~6400 points vs ~4M for the full grid).

    Pass hrrr_tree (from build_hrrr_tree) to avoid rebuilding it each call.
    Returns (plume_mask, metrics, context).
    '''
    no2_c, lon_c, lat_c, dx_c, dy_c, idxbox = crop_window(
        no2, tlon, tlat, plant_lon, plant_lat, half_size_km=half_size_km
    )

    # Regrid HRRR onto the small cropped TEMPO window (fast path)
    u_c, v_c = regrid_hrrr_to_tempo_nn(hu, hv, hlon, hlat, lon_c, lat_c, tree=hrrr_tree)

    wind_dir = wind_to_dir_deg(float(u_c.mean()), float(v_c.mean()))
    bg, _ = robust_background_upwind(no2_c, dx_c, dy_c, wind_dir)
    delta = no2_c - bg

    plume_mask, metrics = detect_plume(delta, dx_c, dy_c, u_c, v_c)

    context = {
        "no2": no2_c,
        "delta": delta,
        "lon": lon_c,
        "lat": lat_c,
        "u": u_c,
        "v": v_c,
        "plant_lon": plant_lon,
        "plant_lat": plant_lat,
    }
    return plume_mask, metrics, context


def run_one_plume_detection(
    tempo_nc: str,
    hrrr_grib: str,
    plant_lon: float,
    plant_lat: float,
    tempo_var: str = "vertical_column_troposphere",
    hrrr_engine: str | None = None,
    half_size_km: float = 80.0,
):
    '''
    End-to-end plume detection for one TEMPO file and one HRRR file.
    Returns:
      plume_mask (2D bool, cropped window)
      metrics (dict)
      context (dict with cropped fields)
    '''
    no2, tlon, tlat, hu, hv, hlon, hlat = load_scene(
        tempo_nc, hrrr_grib, tempo_var=tempo_var, hrrr_engine=hrrr_engine
    )
    return detect_for_plant(
        no2, tlon, tlat, hu, hv, hlon, hlat,
        plant_lon, plant_lat, half_size_km=half_size_km,
    )
