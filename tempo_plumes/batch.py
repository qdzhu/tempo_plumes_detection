from __future__ import annotations

import os
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

import netCDF4 as ncdf
import pandas as pd
import numpy as np
import xarray as xr
from datetime import timezone, datetime

from .pipeline import load_scene, detect_for_plant
from .geo import build_hrrr_tree
from .export import save_netcdf, try_save_geotiff
from .patches import PatchConfig, make_patches
from .match import (
    parse_tempo_time_utc,
    build_hrrr_time_index,
    match_hrrr_for_tempo_floor_hour,
    parse_hrrr_valid_time_utc,
)


def _to_dataset(context: dict, plume_mask: np.ndarray, metrics: dict) -> xr.Dataset:
    ds = xr.Dataset(
        data_vars=dict(
            no2=(("y", "x"), context["no2"]),
            delta=(("y", "x"), context["delta"]),
            u=(("y", "x"), context["u"]),
            v=(("y", "x"), context["v"]),
            plume_mask=(("y", "x"), plume_mask.astype(np.uint8)),
        ),
        coords=dict(
            lon=(("y", "x"), context["lon"]),
            lat=(("y", "x"), context["lat"]),
        ),
        attrs={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()},
    )
    ds.attrs["plant_lon"] = float(context["plant_lon"])
    ds.attrs["plant_lat"] = float(context["plant_lat"])
    return ds


def run_batch(
    plants_csv: str,
    tempo_glob: str,
    hrrr_glob: str,
    out_dir: str,
    tempo_var: str = "vertical_column_troposphere",
    hrrr_engine: str | None = None,
    half_size_km: float = 80.0,
    max_dt_hours: float = 2.0,
    require_exact_hour: bool = False,
    make_geotiff: bool = False,
    make_patches_flag: bool = True,
    patch_size: int = 64,
    stride: int = 32,
    max_workers: int = 1,
):
    os.makedirs(out_dir, exist_ok=True)

    plants = pd.read_csv(plants_csv)
    required = {"plant_id", "plant_lon", "plant_lat"}
    if not required.issubset(set(plants.columns)):
        raise ValueError(f"plants_csv must include columns: {sorted(list(required))}")

    tempo_files = sorted(glob.glob(tempo_glob, recursive=True))
    if not tempo_files:
        raise ValueError(f"No TEMPO files matched: {tempo_glob}")

    hrrr_items = build_hrrr_time_index(hrrr_glob)

    rows = []
    hrrr_tree = None
    last_hrrr_path = None

    for tempo_path in tempo_files:
        tempo_time = parse_tempo_time_utc(tempo_path)
        hrrr_path = match_hrrr_for_tempo_floor_hour(
            tempo_time, hrrr_items, max_dt_hours=max_dt_hours, require_exact_hour=require_exact_hour
        )
        if hrrr_path is None:
            for _, p in plants.iterrows():
                rows.append(
                    dict(
                        plant_id=str(p["plant_id"]),
                        tempo_file=os.path.basename(tempo_path),
                        tempo_time_utc=tempo_time.isoformat(),
                        hrrr_file="",
                        hrrr_time_utc="",
                        matched=False,
                        detected=False,
                        reason="no_hrrr_match",
                    )
                )
            continue

        hrrr_time = parse_hrrr_valid_time_utc(hrrr_path)
        tstr = tempo_time.strftime("%Y%m%dT%H%M%SZ")
        plant_rows = list(plants.to_dict("records"))

        # Check which plants already have output — skip loading entirely if all done
        def _out_nc_path(plant_id):
            return os.path.join(out_dir, "netcdf", plant_id, f"{plant_id}_{tstr}.nc")

        pending = [p for p in plant_rows if not os.path.exists(_out_nc_path(str(p["plant_id"])))]
        done    = [p for p in plant_rows if     os.path.exists(_out_nc_path(str(p["plant_id"])))]

        for p in done:
            plant_id = str(p["plant_id"])
            rows.append(dict(
                plant_id=plant_id,
                tempo_file=os.path.basename(tempo_path),
                tempo_time_utc=tempo_time.isoformat(),
                hrrr_file=os.path.basename(hrrr_path),
                hrrr_time_utc=hrrr_time.isoformat(),
                matched=True,
                detected=False,
                reason="skipped:already_exists",
                out_nc=_out_nc_path(plant_id),
            ))

        if not pending:
            continue

        # Load TEMPO + HRRR once per file pair (not once per plant)
        try:
            no2, tlon, tlat, hu, hv, hlon, hlat = load_scene(
                tempo_path, hrrr_path, tempo_var=tempo_var, hrrr_engine=hrrr_engine
            )
        except Exception as e:
            for p in pending:
                rows.append(
                    dict(
                        plant_id=str(p["plant_id"]),
                        tempo_file=os.path.basename(tempo_path),
                        tempo_time_utc=tempo_time.isoformat(),
                        hrrr_file=os.path.basename(hrrr_path),
                        hrrr_time_utc=hrrr_time.isoformat(),
                        matched=True,
                        detected=False,
                        reason=f"error:{type(e).__name__}:{e}",
                    )
                )
            continue

        # Build HRRR KDTree once per unique HRRR file (fixed grid; reuse across TEMPO files)
        if hrrr_path != last_hrrr_path:
            hrrr_tree = build_hrrr_tree(hlon, hlat)
            last_hrrr_path = hrrr_path

        def _process_plant(p):
            plant_id = str(p["plant_id"])
            plon = float(p["plant_lon"])
            plat = float(p["plant_lat"])

            try:
                plume_mask, metrics, context = detect_for_plant(
                    no2, tlon, tlat, hu, hv, hlon, hlat,
                    plon, plat,
                    half_size_km=half_size_km,
                    hrrr_tree=hrrr_tree,
                )

                ds = _to_dataset(context, plume_mask, metrics)

                out_nc = _out_nc_path(plant_id)
                save_netcdf(out_nc, ds)

                tif_ok = False
                if make_geotiff:
                    out_tif = os.path.join(out_dir, "geotiff", plant_id, f"{plant_id}_{tstr}_delta.tif")
                    tif_ok = try_save_geotiff(out_tif, context["delta"], context["lon"], context["lat"])

                patch_path = ""
                npos = nneg = nkeep = 0
                if make_patches_flag:
                    cfg = PatchConfig(patch_size=patch_size, stride=stride)
                    out_patch_dir = os.path.join(out_dir, "patches", plant_id, tstr)
                    patch_path, npos, nneg, nkeep = make_patches(
                        out_patch_dir, context["no2"], context["delta"], context["u"], context["v"], plume_mask, cfg
                    )

                return dict(
                    plant_id=plant_id,
                    tempo_file=os.path.basename(tempo_path),
                    tempo_time_utc=tempo_time.isoformat(),
                    hrrr_file=os.path.basename(hrrr_path),
                    hrrr_time_utc=hrrr_time.isoformat(),
                    matched=True,
                    detected=bool(metrics.get("detected", False)),
                    reason=metrics.get("reason", ""),
                    out_nc=out_nc,
                    geotiff_written=bool(tif_ok),
                    patches=patch_path,
                    n_pos_patches=int(npos),
                    n_neg_patches=int(nneg),
                    n_keep_patches=int(nkeep),
                    score=float(metrics.get("score", np.nan)) if metrics.get("detected", False) else np.nan,
                )

            except Exception as e:
                return dict(
                    plant_id=plant_id,
                    tempo_file=os.path.basename(tempo_path),
                    tempo_time_utc=tempo_time.isoformat(),
                    hrrr_file=os.path.basename(hrrr_path),
                    hrrr_time_utc=hrrr_time.isoformat(),
                    matched=True,
                    detected=False,
                    reason=f"error:{type(e).__name__}:{e}",
                )

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_plant, p) for p in pending]
                for future in as_completed(futures):
                    rows.append(future.result())
        else:
            for p in pending:
                rows.append(_process_plant(p))

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "batch_summary.csv")
    summary.to_csv(out_csv, index=False)
    return out_csv


def _read_nc_attrs(nc_path: str) -> dict:
    """Read global attributes from a single NetCDF file. Module-level for multiprocessing."""
    plant_id = os.path.basename(os.path.dirname(nc_path))
    stem = os.path.splitext(os.path.basename(nc_path))[0]
    tstr = stem[len(plant_id) + 1:] if stem.startswith(plant_id + "_") else stem

    try:
        tempo_time_utc = datetime.strptime(tstr, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        tempo_time_utc = tstr

    try:
        ds = ncdf.Dataset(nc_path, "r")
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        ds.close()
    except Exception as e:
        attrs = {"error": str(e)}

    row = {"plant_id": plant_id, "tempo_time_utc": tempo_time_utc, "out_nc": nc_path}
    row.update(attrs)
    return row


def summarize_from_netcdf(out_dir: str, max_workers: int = 8) -> str:
    """
    Rebuild batch_summary.csv from existing NetCDF files under out_dir/netcdf/.

    Each .nc file's global attributes contain detection metrics plus plant_lon/plant_lat.
    plant_id and tempo_time_utc are parsed from the filename convention:
        <plant_id>_<YYYYmmddTHHMMSSZ>.nc

    Uses netCDF4 directly (faster than xarray for attribute-only reads) and
    reads files in parallel with multiprocessing.Pool (avoids netCDF4/HDF5 thread-safety issues).
    """
    nc_paths = sorted(glob.glob(os.path.join(out_dir, "netcdf", "*", "*.nc")))
    if not nc_paths:
        raise FileNotFoundError(f"No NetCDF files found under {os.path.join(out_dir, 'netcdf')}")

    with Pool(processes=max_workers) as pool:
        rows = pool.map(_read_nc_attrs, nc_paths)

    summary = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "batch_summary.csv")
    summary.to_csv(out_csv, index=False)
    return out_csv
