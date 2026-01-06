import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skimage.draw import polygon

from escore.visualize import normalize_sv_array

from .processing import *



def overlay_mask(mask, alpha_in, alpha_out):
    h, w = mask.shape
    overlay = np.zeros((h, w, 4))
    overlay[mask] = [255., 0, 0, alpha_in]
    overlay[~mask] = [0, 0, 0, alpha_out]

    return overlay



def get_RGB_fig(
    sv, 
    shape,
    vmin=-90.,
    vmax=-50., 
    window_size:tuple[int, int]=None, 
    padding=10, 
    frequencies=[38, 70, 120],
    show_mask=True,
    mask_alpha_in=0.3,
    mask_alpha_out=0.
):

    # Fetch shape points
    points = np.array(shape["points"])
    bbox = shape["it_min"], shape["it_max"], shape["iz_min"], shape["iz_max"]

    # Window
    xmin, xmax, ymin, ymax = get_window(bbox, 
                                        array_shape=(len(sv.time), len(sv.depth)),
                                        window_shape=window_size,
                                        padding=padding)
    # Compute shape for aspect ratio rendering in app
    #print(f"Window: {xmin, xmax, ymin, ymax}")
    window_shape = xmax - xmin + 1, ymax - ymin + 1

    # Slice sv using bbox
    roi_sv = sv.isel(time=slice(xmin, xmax+1), depth=slice(ymin, ymax+1)).sel(channel=frequencies)

    # Turn into image format array
    sv_array = roi_sv.values
    sv_array = normalize_sv_array(sv_array, vmin=vmin, vmax=vmax)

    # RGB plot
    fig = px.imshow(sv_array, aspect="auto")
    fig.layout.xaxis.title.text = "ESDU"
    fig.layout.yaxis.title.text = "Depth sample"

    # Add points
    xs, ys = [p[0]-xmin for p in points], [p[1]-ymin for p in points]
    fig.add_trace(
        go.Scatter(x=xs, y=ys,
                   mode='markers',
                   marker=dict(color='red', size=5, symbol='circle'))
    )

    # Add mask of selected pixels
    if show_mask:
        mask = get_mask(window=(xmin, xmax, ymin, ymax), points=points)
        mask = mask.T           # shape (W, H) -> (H, W)
        overlay = overlay_mask(mask, mask_alpha_in, mask_alpha_out)

        fig.add_trace(
            go.Image(z=overlay, colormodel="rgba")
        )

    fig.update_xaxes(range=[0, window_shape[0]], autorange=False)
    fig.update_yaxes(range=[window_shape[1], 0], autorange=False)

    return fig, window_shape



def get_clustering_labels_fig(labels_da: xr.DataArray):

    fig = px.imshow(labels_da.values.T)

    return fig



def mean_and_sd_lineplot(df, line_name, line_color="red"):

    data = [
        go.Scatter(
            name=line_name,
            x=df["channel"],
            y=df["mean"],
            marker=dict(color=line_color)
        ),
        go.Scatter(
            name="Upper Bound",
            x=df["channel"],
            y=df["mean"]+df["sd"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name="Lower Bound",
            x=df["channel"],
            y=df["mean"]-df["sd"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor='rgba(68, 68, 68, 0.3)',
            showlegend=False
        )
    ]

    return data



def echotype_deltaSv_histograms(
    roi_sv,
    labels_da,
    cluster_id,
    ref_frequency=38.
):
    
    # Filter Sv based on clustering & select 38 kHz channel
    mask_da = (labels_da == cluster_id)
    echotype_sv = roi_sv.where(mask_da)

    # Compute delta Sv
    echotype_delta_sv = compute_delta_sv(echotype_sv, ref_frequency)

    # Stack spatial dimensions + to numpy
    X = stack_pixels(
        echotype_delta_sv.squeeze("reference_frequency", drop=True)
    ).values # shape (n_pixels, n_channels)

    # Create fig with subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        specs= [[{"colspan": 2}, None],
                [{}, {}]]
    )

    # Add traces
    # Histograms of delta Sv
    for c in range(X.shape[1]):

        fig.add_trace(
            go.Histogram(
            x=X[:, c],
            xbins=dict(start=-50., end=50., size=0.5),
            histnorm="probability",
            opacity=0.5,
            name=f"ΔSv {echotype_delta_sv.channel.values[c]} kHz - {echotype_delta_sv.reference_frequency.values[0]} kHz"
            ),
            row=1,
            col="all"
        )

    # Relative frequency response curve
    df = pd.DataFrame({
        "channel": echotype_delta_sv.channel.values,
        "mean": X.mean(axis=0),
        "sd": X.std(axis=0)
    })
    ref_freq_row = pd.DataFrame([[ref_frequency, 0, 0]], columns=df.columns)
    df = pd.concat([ref_freq_row, df], ignore_index=True)  

    fig.add_traces(
        data=mean_and_sd_lineplot(df, 
                                  line_name=f"Frequency response rel. {ref_frequency}"),
        rows=2,
        cols=1
    )

    # Frequency response curve
    X = stack_pixels(echotype_sv).values
    df = pd.DataFrame({
        "channel": echotype_sv.channel.values,
        "mean": X.mean(axis=0),
        "sd": X.std(axis=0)
    })

    fig.add_traces(
        data=mean_and_sd_lineplot(df, 
                                  line_name=f"Absolute frequency response",
                                  line_color="blue"),
        rows=2,
        cols=2
    )

    # Layout
    fig.update_layout(
        barmode="overlay",
        #title = dict(text='Delta Sv distribution',
        #             x=0.5,
        #             xanchor='center'),
        xaxis1=dict(title='ΔSv [dB]',
                    range=[-30., 30.]),
        yaxis1=dict(title='Probability'),
        xaxis2=dict(title='Sampling frequency [kHz]',
                    range=[38-10, 200+10]),
        yaxis2=dict(title='ΔSv [dB]',
                    range=[-30, 30]),
        xaxis3=dict(title='Sampling frequency [kHz]',
                    range=[38-10, 200+10]),
        yaxis3=dict(title='Sv [dB]',
                    range=[-90, -55]),
        showlegend=True
    )
    
    #print(fig)

    return fig