"""Create tropospheric displacement corrections.

Accepts an OPERA frame ID, or list of DISP products (which
can be local, or filenames/urls)

One GeoTIFF per DISP secondary scene is written to --out-dir.
"""

from pathlib import Path

import pandas as pd
import tyro
from loguru import logger
from opera_utils.disp import DispProductStack, search
from shapely.geometry import box

from ._helpers import (
    _bracket,
    _build_tropo_index,
    _height_to_utm_surface,
    _interp_in_time,
    _open_2d,
    _open_crop,
    get_dem_url,
)


def main(
    frame_id: int | None = None,
    disp_files: list[Path] | None = None,
    tropo_urls_file: Path = Path("tropo_urls.txt"),
    out_dir: Path = Path("corrections"),
    margin_deg: float = 0.3,
):
    """Create tropospheric corrections for Displacement products."""
    if (frame_id is None) == (disp_files is None):
        raise ValueError("Specify *either* --frame-id or --disp-files")

    # load DISP stack
    if disp_files is None:
        products = search(frame_id=frame_id)
        stack = DispProductStack(products=products)
    else:
        stack = DispProductStack.from_file_list(
            [p.read_text().strip() for p in disp_files]
        )

    df = stack.to_dataframe()
    ref_dt = pd.to_datetime(df.reference_datetime.iloc[0], utc=True)
    sec_dts = pd.to_datetime(df.secondary_datetime, utc=True)

    dem_utm = _open_2d(get_dem_url(df.frame_id.iloc[0]))
    bbox = box(*dem_utm.rio.transform_bounds("epsg:4326")).buffer(margin_deg).bounds
    lat_bounds = (bbox[3], bbox[1])  # north, south
    lon_bounds = (bbox[0], bbox[2])  # west , east
    h_max = float(dem_utm.max())

    tropo_urls = Path(tropo_urls_file).read_text().splitlines()
    tropo_idx = _build_tropo_index(tropo_urls)

    # cache delays for every unique timestamp
    delay_2d = {}
    for ts in {ref_dt, *sec_dts}:
        early_u, late_u = _bracket(tropo_idx, ts)
        ds0 = _open_crop(early_u, lat_bounds, lon_bounds, h_max)
        ds1 = _open_crop(late_u, lat_bounds, lon_bounds, h_max)

        td_interp = _interp_in_time(ds0, ds1, ds0.time.item(), ds1.time.item(), ts)
        delay_2d[ts] = _height_to_utm_surface(td_interp.total_delay, dem_utm)

    # ---------- write corrections ------------------------------------------
    out_dir.mkdir(exist_ok=True, parents=True)
    for _, row in df.iterrows():
        sec_ts = pd.to_datetime(row.secondary_datetime, utc=True)
        corr = delay_2d[sec_ts] - delay_2d[ref_dt]
        out_name = f"tropo_corr_F{row.frame_id:05d}_{sec_ts:%Y%m%dT%H%M%S}Z.tif"
        corr.rio.to_raster(out_dir / out_name)
        logger.info(f"Wrote {out_name}")


if __name__ == "__main__":
    tyro.cli(main)
