from __future__ import annotations

import os
import numpy as np
import xarray as xr


def save_netcdf(out_nc: str, ds: xr.Dataset) -> None:
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    ds.to_netcdf(out_nc)


def try_save_geotiff(out_tif: str, arr2d: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray, nodata: float = -9999.0) -> bool:
    '''
    Save a 2D array to GeoTIFF using an approximate georeference derived from lon/lat grid.
    Requires rasterio; if not installed, returns False.
    '''
    try:
        import rasterio
        from rasterio.transform import from_origin
    except Exception:
        return False

    os.makedirs(os.path.dirname(out_tif), exist_ok=True)

    a = np.array(arr2d, dtype=np.float32)
    a = np.where(np.isfinite(a), a, nodata)

    lon00 = float(lon2d[0, 0])
    lat00 = float(lat2d[0, 0])

    dx = float(np.nanmedian(np.diff(lon2d[0, :])))
    dy = float(np.nanmedian(np.diff(lat2d[:, 0])))

    if dy > 0:
        a = np.flipud(a)
        lat00 = float(lat2d[-1, 0])
        dy = -dy

    transform = from_origin(lon00, lat00, dx, -dy)

    with rasterio.open(
        out_tif,
        "w",
        driver="GTiff",
        height=a.shape[0],
        width=a.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(a, 1)

    return True
