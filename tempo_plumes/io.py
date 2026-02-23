from __future__ import annotations

import numpy as np
import netCDF4 as ncdf
import xarray as xr


def load_tempo_no2_l3(
    nc_path: str,
    var_name: str = "vertical_column_troposphere",
    qa_flag_name: str = "main_data_quality_flag",
    cloud_name: str = "eff_cloud_fraction",
    cloud_thresh: float = 0.2,
    require_qa0: bool = True,
):
    """
    Load TEMPO NO2 L3 and apply QC:
      - main_data_quality_flag == 0
      - amf_cloud_fraction < cloud_thresh
    Returns (no2, lon, lat) with bad pixels set to np.nan.
    """
    ds = ncdf.Dataset(nc_path)

    # --- NO2 field ---
    no2 = ds.groups["product"][var_name][:].astype(np.float32)
    no2 = np.squeeze(no2)
    if no2.ndim == 3:
        no2 = no2[0, :, :]

    # --- Lon/Lat (root variables) ---
    lat = np.array(ds.variables["latitude"][:])
    lon = np.array(ds.variables["longitude"][:])
    lat = np.squeeze(lat)
    lon = np.squeeze(lon)


    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
        lon = lon.astype(np.float32)
        lat = lat.astype(np.float32)

    # --- QC fields ---
    # quality flag (product group)
    flag = ds.groups["product"][qa_flag_name][:] if qa_flag_name in ds.groups["product"].variables else None
    if flag is not None:
        flag = np.squeeze(flag)
        if flag.ndim == 3:
            flag = flag[0, :, :]
        flag = flag.astype(np.int16)

    # cloud fraction (support_data group)
    cld = ds.groups["support_data"][cloud_name][:] if cloud_name in ds.groups["support_data"].variables else None
    if cld is not None:
        cld = np.squeeze(cld)
        if cld.ndim == 3:
            cld = cld[0, :, :]
        cld = cld.astype(np.float32)

    ds.close()

    # --- Apply filters ---
    m = np.isfinite(no2)
    if require_qa0 and flag is not None:
        m &= (flag == 0)
    if cld is not None:
        m &= (cld < cloud_thresh)

    no2 = np.where(m, no2, np.nan).astype(np.float32)
    return no2, lon, lat
    



def load_hrrr_uv_mean5(
    grib_path: str,
    u_name: str = "u",
    v_name: str = "v",
    engine: str | None = None,
):
    '''
    HRRR subset:
      - u/v dims: ("hybrid", "y", "x")
      - average the lowest 5 hybrid levels to represent near-surface winds
      - convert longitudes to [-180, 180] to match TEMPO
    '''
    ds = xr.open_dataset(grib_path, engine=engine) if engine else xr.open_dataset(grib_path)

    u = ds[u_name].isel(hybrid=slice(0, 5)).mean("hybrid").values.astype(np.float32)
    v = ds[v_name].isel(hybrid=slice(0, 5)).mean("hybrid").values.astype(np.float32)

    hrrr_lon = ds["longitude"][:].values.astype(np.float32)
    hrrr_lon[hrrr_lon > 180] -= 360
    hrrr_lat = ds["latitude"][:].values.astype(np.float32)

    return u, v, hrrr_lon, hrrr_lat
