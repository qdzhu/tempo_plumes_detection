"""
Driver script for tempo-plumes batch detection.
Intended to be called from the SGE submission script.
Uses Python-level glob patterns to avoid shell expansion issues.
"""
import os
from tempo_plumes.batch import run_batch
PLANTS_CSV  = "oris_coords_tz.csv"
TEMPO_GLOB  = "/scratch/sao/qzhu/tempo/TEMPO_NO2_L3_V03/2024/**/*.nc"
HRRR_GLOB   = "/scratch/sao/qzhu/HRRR/hrrr/**/*.grib2"
OUT_DIR     = "/scratch/sao/qzhu/AI-projs/Plume-detection/CEMS"


HRRR_ENGINE        = "cfgrib"
HALF_SIZE_KM       = 80.0
MAX_DT_HOURS       = 2.0
REQUIRE_EXACT_HOUR = False
MAKE_GEOTIFF       = False
MAKE_PATCHES       = True
PATCH_SIZE         = 64
STRIDE             = 32
MAX_WORKERS        = 10   # set to 1 for serial; increase up to available cores
START_DATE         = "2024-10-01"   # only process TEMPO files on or after this date; set to None for all

if __name__ == "__main__":
    print(f"Starting batch run → {OUT_DIR}")
    out_csv = run_batch(
        plants_csv         = PLANTS_CSV,
        tempo_glob         = TEMPO_GLOB,
        hrrr_glob          = HRRR_GLOB,
        out_dir            = OUT_DIR,
        hrrr_engine        = HRRR_ENGINE,
        half_size_km       = HALF_SIZE_KM,
        max_dt_hours       = MAX_DT_HOURS,
        require_exact_hour = REQUIRE_EXACT_HOUR,
        make_geotiff       = MAKE_GEOTIFF,
        make_patches_flag  = MAKE_PATCHES,
        patch_size         = PATCH_SIZE,
        stride             = STRIDE,
        max_workers        = MAX_WORKERS,
        start_date         = START_DATE,
    )
    print(f"Done. Summary written to: {out_csv}")
