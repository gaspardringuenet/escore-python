from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.dates import ConciseDateFormatter
from matplotlib.figure import Figure


def plot_channels(
    da_Sv: xr.DataArray,
    plot_api: Literal["hvplot", "plot"],
    region_row: pd.Series | None = None,
    **plot_kwrgs,
):
    # Default kwargs & plotting API depending on backend
    if plot_api == "hvplot":
        default_kwrgs = {
            "x": "ping_time",
            "y": "depth",
            "clim": (-90.0, -50.0),
            "flip_yaxis": True,
            "rasterize": False,
            "cmap": "viridis",
        }

        # Build echogram QuadMesh
        echogram = da_Sv.hvplot.quadmesh(**(default_kwrgs | plot_kwrgs))

        # Build region's polygon & return HoloViews overlay
        if region_row is not None:
            df = pd.DataFrame({"ping_time": region_row["time"], "depth": region_row["depth"]})
            polygon = df.hvplot.paths(
                x="ping_time", y="depth", color="red", rasterize=default_kwrgs["rasterize"]
            )
            return echogram * polygon

        # Return simple echogram
        return echogram

    elif plot_api == "plot":
        default_kwrgs = {
            "x": "ping_time",
            "y": "depth",
            "col": "channel",
            "col_wrap": 2,
            "vmin": -90.0,
            "vmax": -50.0,
            "yincrease": False,
            "cmap": "viridis",
        }

        # Build echogram plot (FaceGrid)
        echogram = da_Sv.plot.pcolormesh(**(default_kwrgs | plot_kwrgs))

        # Plot region over all facets
        if region_row is not None:
            for ax in echogram.axes.flat:
                ax.plot(
                    region_row["time"],
                    region_row["depth"],
                    fillstyle="full",
                    markersize=0.5,
                    color="red",
                )

        return echogram

    else:
        raise ValueError(f"Unsupported plot API '{plot_api}'. Chose one from ['hvplot', 'plot'].")


def plot_rgb(
    da_Sv: xr.DataArray,
    channel_idx: List[int],
    region_row: pd.Series | None = None,
    ax: Axes | None = None,
    **plot_kwrgs,
) -> Figure | Axes:

    default_plot_kwrgs = {
        "vmin": -90,
        "vmax": -50,
        "shading": "nearest",
        "figsize": (8, 4),
    }
    plot_kwrgs = default_plot_kwrgs | plot_kwrgs

    # Create meshgrid
    time = da_Sv["ping_time"].values
    depth = da_Sv["depth"].values
    X, Y = np.meshgrid(time, depth)

    # Create Z values
    da_Sv = da_Sv.isel(channel=channel_idx).transpose("depth", "ping_time", "channel")
    Z = (da_Sv.values - plot_kwrgs["vmin"]) / (plot_kwrgs["vmax"] - plot_kwrgs["vmin"])
    Z = Z.clip(0, 1)
    Z = np.nan_to_num(Z, nan=0.0)

    # Mask invalid data (mostly bottom)
    valid_data_mask = ~np.all(np.isnan(da_Sv.values), axis=-1)
    Z = np.ma.array(Z, mask=np.dstack([~valid_data_mask] * 3))

    # Create figure (if ax was not passed)
    if ax is not None:
        fig = None
        plot_kwrgs.pop("figsize")
    else:
        fig, ax = plt.subplots(figsize=plot_kwrgs.pop("figsize"))
    ax.pcolormesh(X, Y, Z, **plot_kwrgs)

    # Format
    ax.set_ylabel("depth")
    ax.set_xlabel("ping_time")
    ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.yaxis.set_inverted(True)

    # Overlay polygon
    if region_row is not None:
        ax.plot(
            region_row["time"],
            region_row["depth"],
            fillstyle="full",
            markersize=0.5,
            color="red",
        )

    # Return ax if it was passed else fig
    return fig if fig else ax
