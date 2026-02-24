"""
Write a summary.csv inside each plant's netcdf directory.
Parallelizes over plant directories for efficiency with large file counts.
Output: netcdf/<plant_id>/summary.csv for each plant.
"""
from tempo_plumes.batch import summarize_by_plant
import pandas as pd
import os
# ── Paths ──────────────────────────────────────────────────────────────────
OUT_DIR     = "/scratch/sao/qzhu/AI-projs/Plume-detection/CEMS"
MAX_WORKERS = 10

# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Writing per-plant summaries under: {OUT_DIR}/netcdf/")
    #csv_paths = summarize_by_plant(OUT_DIR, max_workers=MAX_WORKERS)
    #print(f"Done. {len(csv_paths):,} per-plant summary CSVs written.")

    #aggregate all files together
    plants = pd.read_csv('oris_coords_tz.csv')
    plant_ids = plants['plant_id']
    summary = pd.DataFrame()
    for plant_id in plant_ids:
        if os.path.exists(OUT_DIR + '/netcdf/{}/summary.csv'.format(plant_id)):
            this_summary = pd.read_csv(OUT_DIR + '/netcdf/{}/summary.csv'.format(plant_id))
            summary = pd.concat([summary, this_summary])
    summary.to_csv(OUT_DIR + '/netcdf/summary.csv')
