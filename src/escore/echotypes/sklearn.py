import numpy as np
import xarray as xr


def stack_for_sklearn(
    da_Sv: xr.DataArray,
    drop_na: bool = True,
    time_var: str = "ping_time",
    depth_var: str = "depth",
    channel_var: str = "channel",
) -> xr.DataArray:
    """Stack an acoustic DataArray for sklearn prediction."""

    # Stack (ping_time, depth) into a flat sample dimension -> (n_samples, channel)
    da_stacked = da_Sv.stack(sample=(time_var, depth_var)).transpose("sample", channel_var)

    # Drop NaN samples
    if drop_na:
        return da_stacked.dropna("sample", how="any")

    return da_stacked


def unstack_sklearn_preds(
    preds: np.ndarray,
    da_stacked: xr.DataArray,
    name: str | None = None,
) -> xr.DataArray:
    """Unstack the sklearn prediction using the da_stacked array that was used as prediction
    input."""

    da_preds = da_stacked.isel(channel=0).drop_vars("channel")
    da_preds.data = preds
    da_preds = da_preds.unstack("sample")
    da_preds.name = name

    return da_preds


def classes_to_segments(da_Sv: xr.DataArray, da_classes: xr.DataArray) -> xr.DataArray:
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
    da_segments = da_segments.assign_coords({"segment": ("segment", segment_values)})

    return da_segments
