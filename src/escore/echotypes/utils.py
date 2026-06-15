import re

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


def _format_echotype_dataframe(
    df: pd.DataFrame,
    ref_channel_idx: int | None = None,
) -> pd.DataFrame:
    """Format echotype dataframe data. Channel columns are renamed
    (channel_{i}\_Sv -> channel\_{i}). If ref_channel_idx is provided,
    the channel\_{i} value will contain Sv difference between the original
    channel_{i}\_Sv and channel_{ref_channel_idx}\_Sv.

    Parameters
    ----------
    df : pd.DataFrame
        Echotype dataframe containing per-channel Sv data in columns named channel_{i}\_Sv.
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
