from __future__ import annotations

import argparse
import os
import xarray as xr
import numpy as np

from .pipeline import run_one_plume_detection
from .batch import run_batch, summarize_from_netcdf


def _cmd_single(args):
    plume_mask, metrics, context = run_one_plume_detection(
        args.tempo,
        args.hrrr,
        args.plant_lon,
        args.plant_lat,
        tempo_var=args.tempo_var,
        hrrr_engine=args.engine,
        half_size_km=args.half_size_km,
    )

    print("Detected:", metrics.get("detected", False))
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.out_nc:
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
            attrs=metrics,
        )
        os.makedirs(os.path.dirname(args.out_nc), exist_ok=True)
        ds.to_netcdf(args.out_nc)
        print("Wrote:", args.out_nc)


def _cmd_batch(args):
    out_csv = run_batch(
        plants_csv=args.plants_csv,
        tempo_glob=args.tempo_glob,
        hrrr_glob=args.hrrr_glob,
        out_dir=args.out_dir,
        tempo_var=args.tempo_var,
        hrrr_engine=args.engine,
        half_size_km=args.half_size_km,
        max_dt_hours=args.max_dt_hours,
        require_exact_hour=args.require_exact_hour,
        make_geotiff=args.geotiff,
        make_patches_flag=not args.no_patches,
        patch_size=args.patch_size,
        stride=args.stride,
        max_workers=args.workers,
    )
    print("Wrote summary:", out_csv)


def main():
    p = argparse.ArgumentParser(prog="tempo-plume-detect")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("single", help="Run plume detection for one TEMPO + one HRRR file.")
    p1.add_argument("--tempo", required=True)
    p1.add_argument("--hrrr", required=True)
    p1.add_argument("--plant-lon", type=float, required=True)
    p1.add_argument("--plant-lat", type=float, required=True)
    p1.add_argument("--tempo-var", default="vertical_column_troposphere")
    p1.add_argument("--engine", default=None, help="xarray engine for GRIB, e.g., cfgrib")
    p1.add_argument("--half-size-km", type=float, default=80.0)
    p1.add_argument("--out-nc", default="")
    p1.set_defaults(func=_cmd_single)

    p2 = sub.add_parser("batch", help="Batch run: many plants x many TEMPO files; HRRR matched by floored TEMPO hour.")
    p2.add_argument("--plants-csv", required=True)
    p2.add_argument("--tempo-glob", required=True)
    p2.add_argument("--hrrr-glob", required=True)
    p2.add_argument("--out-dir", required=True)
    p2.add_argument("--tempo-var", default="vertical_column_troposphere")
    p2.add_argument("--engine", default=None)
    p2.add_argument("--half-size-km", type=float, default=80.0)
    p2.add_argument("--max-dt-hours", type=float, default=2.0)
    p2.add_argument("--require-exact-hour", action="store_true")
    p2.add_argument("--geotiff", action="store_true")
    p2.add_argument("--no-patches", action="store_true")
    p2.add_argument("--patch-size", type=int, default=64)
    p2.add_argument("--stride", type=int, default=32)
    p2.add_argument("--workers", type=int, default=1,
                    help="Number of parallel workers for plant processing (default: 1, serial).")
    p2.set_defaults(func=_cmd_batch)

    p3 = sub.add_parser("summarize", help="Rebuild batch_summary.csv from existing NetCDF files.")
    p3.add_argument("--out-dir", required=True, help="Same out_dir used in the batch run.")
    p3.set_defaults(func=lambda a: print("Wrote summary:", summarize_from_netcdf(a.out_dir)))

    args = p.parse_args()
    args.func(args)
