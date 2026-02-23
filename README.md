# tempo-plumes

A practical baseline package to detect point-source plumes using TEMPO L3 columns and HRRR winds.

## Install
```bash
unzip tempo_plumes_package.zip
cd tempo_plumes
pip install -e .
```

Optional GeoTIFF:
```bash
pip install -e ".[geotiff]"
```

## Notes
- TEMPO time is taken from filename and floored to hour (e.g., 23:28 -> 23Z).
- HRRR valid time is parsed from `hrrr.t??z.wrfnatf??` plus the YYYYMMDD date token in the path.
- Batch mode matches TEMPO hour to HRRR valid hour; if missing, falls back to nearest within `--max-dt-hours`.
