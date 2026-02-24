"""
Write a summary.csv inside each plant's netcdf directory.
Run this after a batch detection to regenerate or repair per-plant summaries.
Output: netcdf/<plant_id>/summary.csv for each plant.
"""
from tempo_plumes.batch import summarize_by_plant

# ── Paths ──────────────────────────────────────────────────────────────────
OUT_DIR     = "/scratch/sao/qzhu/AI-projs/Plume-detection/CEMS"
MAX_WORKERS = 10

# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Writing per-plant summaries under: {OUT_DIR}/netcdf/")
    csv_paths = summarize_by_plant(OUT_DIR, max_workers=MAX_WORKERS)
    print(f"Done. {len(csv_paths):,} per-plant summary CSVs written.")
