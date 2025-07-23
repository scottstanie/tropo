from __future__ import annotations

from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from opera_utils import get_dates
from opera_utils.disp import open_file
from scipy.interpolate import RegularGridInterpolator

# ---------- time utilities --------------------------------------------------
TROPO_INTERVAL = timedelta(hours=6)


def get_dem_url(frame_id: int) -> str:
    """Generate the URL for DEM data for a given frame ID."""
    return (
        f"s3://opera-adt/disp/disp-s1-static-layers/F{frame_id:05d}/dem_warped_utm.tif"
    )


def get_los_url(frame_id: int) -> str:
    """Generate the URL for LOS data for a given frame ID."""
    return f"s3://opera-adt/disp/disp-s1-static-layers/F{frame_id:05d}/los_enu.tif"


def _open_2d(filename: str) -> xr.DataArray:
    raster = rxr.open_rasterio(filename)
    if isinstance(raster, list):
        return raster[0].squeeze(drop=True)
    else:
        return raster.squeeze(drop=True)


def _build_tropo_index(urls: list[str]) -> pd.Series:
    """Return Series(url, index=datetime UTC)."""
    times = [get_dates(u, fmt="%Y%m%dT%H%M%S")[0] for u in urls]
    return pd.Series(urls, index=pd.to_datetime(times, utc=True))


def _bracket(url_index: pd.Series, ts: pd.Timestamp) -> tuple[str, str]:
    """Return (earlier, later) urls within ±6 h; raises if missing."""
    early = url_index.loc[:ts].iloc[-1]  # backward
    late = url_index.loc[ts:].iloc[0]  # forward
    if (ts - early.name) > TROPO_INTERVAL or (late.name - ts) > TROPO_INTERVAL:
        raise ValueError(f"No tropo product within ±6 h of {ts}")
    return early, late


# ---------- I/O + cropping --------------------------------------------------
@lru_cache(maxsize=16)
def _open_crop(
    url: str,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    h_max: float,
    h_margin: float = 500,
) -> xr.Dataset:
    """Lazy-open a single L4 file and subset to bbox+height."""
    if Path(url).exists():
        ds = xr.open_dataset(url, engine="h5netcdf")
    else:
        ds = open_file(url)
    lat_max, lat_min = lat_bounds  # note south-to-north ordering in slice
    lon_min, lon_max = lon_bounds
    ds = ds.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
        height=slice(None, h_max + h_margin),
    )
    return ds.load()  # pull the small cube into memory


def _interp_in_time(
    ds0: xr.Dataset,
    ds1: xr.Dataset,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    t: pd.Timestamp,
) -> xr.Dataset:
    """Linear time interpolation of total_delay cube."""
    w = (t - t0) / (t1 - t0)
    td0 = ds0.hydrostatic_delay + ds0.wet_delay
    td1 = ds1.hydrostatic_delay + ds1.wet_delay
    return (1.0 - w) * td0 + w * td1  # keeps (height, lat, lon)


def _height_to_utm_surface(td_3d: xr.DataArray, dem_utm: xr.DataArray) -> xr.DataArray:
    """Use RegularGridInterpolator exactly like your apply-tropo logic."""
    td_3d = td_3d.rename(latitude="y", longitude="x")  # rioxarray expects y/x
    td_utm = td_3d.rio.write_crs("epsg:4326").rio.reproject(
        dem_utm.rio.crs, resampling="cubic"
    )
    td_utm = td_utm.isel(x=slice(2, -2), y=slice(2, -2))  # trim edges

    rgi = RegularGridInterpolator(
        (td_utm.height.values, td_utm.y.values, td_utm.x.values),
        td_utm.values,
        method="cubic",
        bounds_error=False,
        fill_value=np.nan,
    )
    yy, xx = np.meshgrid(dem_utm.y, dem_utm.x, indexing="ij")
    interp = rgi((dem_utm.values.ravel(), yy.ravel(), xx.ravel()))
    out = dem_utm.copy()
    out.values[:] = interp.reshape(dem_utm.shape).astype("float32")
    return out
