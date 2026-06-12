from typing import Tuple

import numpy as np
import xarray as xr


def stack_for_sklearn(
    da_Sv: xr.DataArray,
    drop_na: bool = True,
) -> xr.DataArray:

    # Stack (ping_time, depth) into a flat sample dimension -> (n_samples, channel)
    da_stacked = da_Sv.stack(sample=("ping_time", "depth")).transpose("sample", "channel")

    # Drop NaN samples
    if drop_na:
        return da_stacked.dropna("sample", how="any")

    return da_stacked


def unstack_sklearn_preds(
    preds: np.ndarray,
    da_stacked: xr.DataArray,
    name: str | None = None,
) -> xr.DataArray:

    da_preds = xr.DataArray(
        preds,
        coords={"sample": da_stacked.coords["sample"]},
        dims=["sample"],
        name=name,
    ).unstack("sample")

    return da_preds


def classes_to_segments(da_Sv: xr.DataArray, da_classes: xr.DataArray):
    """
    Decompose da_Sv into a (time, depth, channel, segment) array.
    Segment i contains Sv data if this (time, depth) point was assigned class i,
    and NaN if not.

    Parameters
    ----------
    da_Sv : xr.DataArray
        Acoustic data array of typical dimensions (time, depth, channel).
    da_classes : xr.DataArray
        Classification results, indexed by a subset of da_Sv's dimensions, typically (time, depth).
    """

    # Create a decomposition with 'segment' dimension
    segment_values = np.unique(da_classes.values)
    segment_values = segment_values[~np.isnan(segment_values)].astype(int)

    # Create a list of masked arrays, one per segment
    segmented_data = []
    for seg in segment_values:
        mask = da_classes == seg
        masked_sv = da_Sv.where(mask)
        segmented_data.append(masked_sv)

    # Stack into a new dimension
    da_segments = xr.concat(segmented_data, dim="segment")
    da_segments["segment"] = segment_values

    return da_segments
