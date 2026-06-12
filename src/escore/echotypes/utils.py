import pandas as pd
import xarray as xr
from echoregions.regions2d import Regions2D


def _select_region_row(regions: Regions2D, region_id: int, close: bool):
    if close:
        region_data = regions.close_region(region_id)
    else:
        region_data = regions.select_region(region_id)
    if not len(region_data) == 1:
        raise ValueError(f"More than 1 row with region_id = {region_id}")
    region_row = region_data.iloc[0]
    return region_row


def _select_bbox_data(ds: xr.Dataset, var: str, region_row: pd.Series):
    return ds[var].sel(
        ping_time=slice(region_row["region_bbox_left"], region_row["region_bbox_right"]),
        depth=slice(region_row["region_bbox_top"], region_row["region_bbox_bottom"]),
    )
