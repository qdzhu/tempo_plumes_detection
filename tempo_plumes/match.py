from __future__ import annotations

import os
import re
import glob
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

_TEMPO_TIME_RE = re.compile(r"_(\d{8}T\d{6})Z_")
_HRRR_CYCLE_RE = re.compile(r"\.t(\d{2})z\.wrfnatf(\d{2})\b")


def parse_tempo_time_utc(path: str) -> datetime:
    base = os.path.basename(path)
    m = _TEMPO_TIME_RE.search(base)
    if not m:
        raise ValueError(f"Cannot parse TEMPO UTC time from filename: {base}")
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def floor_to_hour_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.replace(minute=0, second=0, microsecond=0)


def parse_hrrr_valid_time_utc(path: str) -> datetime:
    base = os.path.basename(path)
    m = _HRRR_CYCLE_RE.search(base)
    if not m:
        raise ValueError(f"Cannot parse HRRR cycle/fxx from filename: {base}")

    cycle_hour = int(m.group(1))
    fxx = int(m.group(2))

    tokens = re.findall(r"(20\d{6})", path)
    if not tokens:
        raise ValueError(f"Cannot infer HRRR date (YYYYMMDD) from path: {path}")
    yyyymmdd = tokens[-1]
    date0 = datetime.strptime(yyyymmdd, "%Y%m%d").replace(tzinfo=timezone.utc)

    return date0 + timedelta(hours=cycle_hour + fxx)


def build_hrrr_time_index(hrrr_glob: str) -> List[Tuple[datetime, str]]:
    files = sorted(glob.glob(hrrr_glob, recursive=True))
    if not files:
        raise ValueError(f"No HRRR files matched: {hrrr_glob}")

    items: List[Tuple[datetime, str]] = []
    for f in files:
        try:
            t = parse_hrrr_valid_time_utc(f)
            items.append((t, f))
        except Exception:
            continue

    items.sort(key=lambda x: x[0])
    if not items:
        raise ValueError("No HRRR files were parseable into valid times.")
    return items


def match_hrrr_for_tempo_floor_hour(
    tempo_time: datetime,
    hrrr_items: List[Tuple[datetime, str]],
    max_dt_hours: float = 2.0,
    require_exact_hour: bool = False,
) -> str | None:
    target = floor_to_hour_utc(tempo_time)

    lo, hi = 0, len(hrrr_items)
    while lo < hi:
        mid = (lo + hi) // 2
        if hrrr_items[mid][0] < target:
            lo = mid + 1
        else:
            hi = mid

    if lo < len(hrrr_items) and hrrr_items[lo][0] == target:
        return hrrr_items[lo][1]

    if require_exact_hour:
        return None

    candidates = []
    if lo < len(hrrr_items):
        candidates.append(hrrr_items[lo])
    if lo - 1 >= 0:
        candidates.append(hrrr_items[lo - 1])

    best, best_dt = None, None
    for t, f in candidates:
        dt = abs((t - target).total_seconds())
        if best is None or dt < best_dt:
            best, best_dt = f, dt

    if best_dt is not None and best_dt <= max_dt_hours * 3600:
        return best
    return None
