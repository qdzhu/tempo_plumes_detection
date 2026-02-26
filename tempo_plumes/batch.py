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


def _scan_completed_nc(out_dir: str, plant_ids: list) -> set:
    """
    Scan every plant's netcdf directory ONCE before the main loop and return a
    set of (plant_id, tstr) pairs for files that exist and are non-empty.
    One directory read per plant — no file opens.
    """
    done = set()
    netcdf_dir = os.path.join(out_dir, "netcdf")
    for plant_id in plant_ids:
        plant_dir = os.path.join(netcdf_dir, plant_id)
        if not os.path.isdir(plant_dir):
            continue
        prefix = plant_id + "_"
        with os.scandir(plant_dir) as it:
            for entry in it:
                if not entry.name.endswith(".nc") or not entry.is_file():
                    continue
                try:
                    if entry.stat().st_size == 0:
                        continue   # skip empty stubs; they'll be overwritten
                except OSError:
                    continue
                stem = entry.name[:-3]   # strip .nc
                tstr = stem[len(prefix):] if stem.startswith(prefix) else stem
                done.add((plant_id, tstr))
    return done


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

    plant_ids = [str(p) for p in plants["plant_id"]]
    print("Scanning existing output files …")
    completed_nc = _scan_completed_nc(out_dir, plant_ids)
    print(f"  {len(completed_nc):,} valid plant×time pairs already on disk.")

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

        # Check which plants already have output — O(1) set lookup, no file I/O.
        def _out_nc_path(plant_id):
            return os.path.join(out_dir, "netcdf", plant_id, f"{plant_id}_{tstr}.nc")

        pending = [p for p in plant_rows if (str(p["plant_id"]), tstr) not in completed_nc]
        done    = [p for p in plant_rows if (str(p["plant_id"]), tstr) in     completed_nc]

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

        # Pre-create all plant output directories in the main thread to avoid
        # NFS propagation races when threads call os.makedirs concurrently.
        for p in pending:
            os.makedirs(os.path.join(out_dir, "netcdf", str(p["plant_id"])), exist_ok=True)

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



def _summarize_plant_dir(plant_dir: str) -> str:
    """
    Read all NetCDF attrs in one plant directory and write a per-plant summary.csv.
    Returns the path to the written CSV, or empty string if no files found.
    Module-level for multiprocessing.
    """
    plant_id = os.path.basename(plant_dir)
    rows = []
    with os.scandir(plant_dir) as it:
        for entry in it:
            if not entry.name.endswith(".nc") or not entry.is_file():
                continue
            stem = entry.name[:-3]
            tstr = stem[len(plant_id) + 1:] if stem.startswith(plant_id + "_") else stem
            try:
                tempo_time_utc = datetime.strptime(tstr, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                tempo_time_utc = tstr
            try:
                ds = ncdf.Dataset(entry.path, "r")
                attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
                ds.close()
            except Exception as e:
                attrs = {"error": str(e)}
            row = {"plant_id": plant_id, "tempo_time_utc": tempo_time_utc, "out_nc": entry.path}
            row.update(attrs)
            rows.append(row)

    if not rows:
        return ""
    out_csv = os.path.join(plant_dir, "summary.csv")
    pd.DataFrame(rows).sort_values("tempo_time_utc").to_csv(out_csv, index=False)
    return out_csv


def summarize_by_plant(out_dir: str, max_workers: int = 8) -> list[str]:
    """
    Write a summary.csv inside each plant's netcdf directory.
    Parallelizes over plant directories — each worker handles one plant at a time.
    Returns a list of written CSV paths.
    """
    netcdf_dir = os.path.join(out_dir, "netcdf")
    if not os.path.isdir(netcdf_dir):
        raise FileNotFoundError(f"No netcdf directory found at {netcdf_dir}")

    plant_dirs = [e.path for e in os.scandir(netcdf_dir) if e.is_dir()]
    if not plant_dirs:
        raise FileNotFoundError(f"No plant directories found under {netcdf_dir}")

    print(f"Found {len(plant_dirs):,} plant directories. Processing with {max_workers} workers...")

    with Pool(processes=max_workers) as pool:
        csv_paths = pool.map(_summarize_plant_dir, plant_dirs)

    written = [p for p in csv_paths if p]
    print(f"Done. Wrote {len(written):,} per-plant summary CSVs.")
    return written
