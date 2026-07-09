import re

import numpy as np
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

    # Find indices directly without using sel() with method parameter
    ping_left = pd.Timestamp(region_row["region_bbox_left"])
    ping_right = pd.Timestamp(region_row["region_bbox_right"])
    depth_top = float(region_row["region_bbox_top"])
    depth_bottom = float(region_row["region_bbox_bottom"])

    # Use searchsorted to find indices for non-unique coordinates
    ping_idx_left = (ds.ping_time.values >= ping_left).argmax()
    ping_idx_right = (ds.ping_time.values <= ping_right)[::-1].argmax()
    ping_idx_right = len(ds.ping_time) - 1 - ping_idx_right

    depth_idx_top = (ds.depth.values >= depth_top).argmax()
    depth_idx_bottom = (ds.depth.values <= depth_bottom)[::-1].argmax()
    depth_idx_bottom = len(ds.depth) - 1 - depth_idx_bottom

    return ds[var].isel(
        ping_time=slice(ping_idx_left, ping_idx_right + 1),
        depth=slice(depth_idx_top, depth_idx_bottom + 1),
    )


def _format_feature_dataframe(
    df: pd.DataFrame,
    ref_channel_idx: int | None = None,
) -> pd.DataFrame:
    """Format feature dataframe data. Channel columns are renamed
    (`channel_{i}_Sv` -> `channel_{i}`). If ref_channel_idx is provided,
    the `channel_{i}` value will contain Sv difference between the original
    `channel_{i}_Sv` and `channel_{ref_channel_idx}_Sv`.

    Parameters
    ----------
    df : pd.DataFrame
        Echotype dataframe containing per-channel Sv data in columns named `channel_{i}_Sv`.
    ref_channel_idx : int | None, optional
        Index of the reference channel to subtract to all channels, by default None

    Returns
    -------
    pd.DataFrame
        Dataframe with formatted channel columns.
    """

    # Rename all channels 'channel_{i}_Sv' -> 'channel_{i}' and fetch ids
    channels_idx: list[int] = []

    def rename_channels(col: str) -> str:
        m = re.fullmatch(r"channel_(\d+)_Sv", col)
        if m:
            i = int(m.group(1))
            channels_idx.append(i)
            return f"channel_{i}"
        return col

    df = df.rename(rename_channels, axis=1)

    if ref_channel_idx is not None:
        ref_col = f"channel_{int(ref_channel_idx)}"

        # Check that reference channel exists in dataframe
        if ref_col not in df.columns:
            raise KeyError(f"Reference channel column {ref_col} not found in dataframe")

        # Fetch reference column data
        ref_values = df[ref_col].copy()

        # Subtract reference channel from all channels
        for i in channels_idx:
            col = f"channel_{i}"
            df[col] = df[col] - ref_values

    return df


def _stack_for_sklearn(
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


def _unstack_sklearn_preds(
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


def _classes_to_segments(da_Sv: xr.DataArray, da_classes: xr.DataArray) -> xr.DataArray:
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
