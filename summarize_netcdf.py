"""
Rebuild batch_summary.csv from existing NetCDF output files.
Run this after a batch detection to regenerate or repair the summary CSV
without re-running any detection.
"""
from tempo_plumes.batch import summarize_from_netcdf

# ── Paths ──────────────────────────────────────────────────────────────────
OUT_DIR = "Outputs/batch"

# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Scanning NetCDF files under: {OUT_DIR}/netcdf/")
    out_csv = summarize_from_netcdf(OUT_DIR)
    print(f"Done. Summary written to: {out_csv}")
