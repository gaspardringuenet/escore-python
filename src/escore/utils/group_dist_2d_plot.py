import holoviews as hv
import hvplot.pandas
import pandas as pd


def plot_echotypes_dist_2D(
    df: pd.DataFrame,
    x_channel: str,
    y_channel: str,
    ref_channel: str,
    group_channel: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "viridis",
    subplots_width: int = 500,
):
    """Show interactive figure with a 2D scatterplot of echogram samples.

    Parameters
    ----------
    df : pd.DataFrame
        Contains echotypes samples (typically output of EchotypeWorkflow.export). Must contain
        x_channnel, y_channel, ref_channel and group_channel.
    x_channel : str
        Channel to plot on the x axis.
    y_channel : str
        Channel to plot on the y axis.
    ref_channel : str
        Channel to subtract to the others.
    group_channel : str
        Grouping channel for group-specific subplot.
    xlabel : str | None, optional
        Label of the x axis, by default None
    ylabel : str | None, optional
        Label of the y axis, by default None
    cmap : str, optional
        Colormap for bivariate and hexbin plots, by default "viridis"
    subplots_width : int, optional
        Width of each of the two subplots, by default 500
    """

    hv.output(widget_location="bottom")

    # Compute Delta MVBS
    df = df.copy()
    ref_values = df[ref_channel].copy()
    df[x_channel] = df[x_channel] - ref_values
    df[y_channel] = df[y_channel] - ref_values

    # 2D scatter plot of samples (rasterize & use cnorm='eq_hist' for better viz of large dataset)
    scatterplot = df.hvplot.scatter(
        x=x_channel,
        y=y_channel,
        cmap="dimgray_r",
        rasterize=True,
        cnorm="eq_hist",
        grid=True,
    ).opts(
        clabel="Sample count",
    )

    # Bivariate distribution contours for each echotype (switch with selector widget)
    biv = df.hvplot.bivariate(
        x=x_channel,
        y=y_channel,
        groupby=group_channel,
        levels=4,
        cmap=cmap,
        filled=True,
        fill_alpha=0.5,
        legend=False,
    )

    # Overlay plots
    global_plot = (scatterplot * biv).opts(
        xlabel=xlabel,
        ylabel=ylabel,
        show_legend=False,
        width=subplots_width,
    )

    # Hexbin of the selected echotype's sampless
    specific_plot = df.hvplot.hexbin(
        x=x_channel,
        y=y_channel,
        groupby=group_channel,
        cmap=cmap,
        grid=True,
        clabel="Sample count",
    ).opts(
        xlabel=xlabel,
        ylabel=ylabel,
        show_legend=False,
        width=subplots_width,
    )

    # Overlay plots
    plot = (global_plot + specific_plot).opts(shared_axes=True)

    return plot
