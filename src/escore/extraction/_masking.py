"""ROI masking for echotype extraction."""

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from echoregions.regions2d import Regions2D


class MaskingError(Exception):
    """A region could not be masked on the acoustic data grid."""


def _covering_slice(coord: xr.DataArray, lower: object, upper: object) -> slice:
    values = coord.values
    start = max(np.searchsorted(values, lower, side="right") - 1, 0)  # type: ignore
    stop = min(np.searchsorted(values, upper, side="left"), len(values) - 1)  # type: ignore
    return slice(start, stop + 1)


def select_in_region(
    sv: xr.DataArray,
    regions: Regions2D,
    metadata: pd.Series,
    roi_id: int,
) -> xr.DataArray:
    """Mask a bounding-box subset of Sv using the Echoregions polygon."""
    try:
        time = metadata.at["time"]
        depth = metadata.at["depth"]
        time_lower = (
            pd.Timestamp(metadata.at["region_bbox_left"]).to_datetime64()
            if "region_bbox_left" in metadata
            else time.min()
        )
        time_upper = (
            pd.Timestamp(metadata.at["region_bbox_right"]).to_datetime64()
            if "region_bbox_right" in metadata
            else time.max()
        )
        depth_lower = (
            metadata.at["region_bbox_top"] if "region_bbox_top" in metadata else depth.min()
        )
        depth_upper = (
            metadata.at["region_bbox_bottom"] if "region_bbox_bottom" in metadata else depth.max()
        )
        subset = sv.isel(
            ping_time=_covering_slice(sv.ping_time, time_lower, time_upper),
            depth=_covering_slice(sv.depth, depth_lower, depth_upper),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MaskingError(f"Unable to select bounding box for region {roi_id}: {exc}") from exc

    template = xr.DataArray(
        np.zeros((subset.sizes["ping_time"], subset.sizes["depth"])),
        coords={"ping_time": subset.ping_time, "depth": subset.depth},
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All rows in Regions DataFrame")
            result = regions.region_mask(template, region_id=roi_id)
        if result is None:
            raise MaskingError(f"No mask returned for region {roi_id}.")
        mask = result[0]["mask_3d"].sel(region_id=roi_id, drop=False) == 1
    except KeyError as exc:
        raise MaskingError(f"Region {roi_id} has no mask (likely outside the data range).") from exc

    return xr.where(mask, subset, np.nan, keep_attrs=True)
