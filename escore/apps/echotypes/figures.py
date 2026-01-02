import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    # Slice sv using bbox
    roi_sv = sv.isel(time=slice(xmin, xmax), depth=slice(ymin, ymax)).sel(channel=frequencies)

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
                   marker=dict(line_color='red', size=10, line_width=3, symbol='x-thin'))
    )

    # Add mask of selected pixels
    if show_mask:
        mask = get_mask(window=(xmin, xmax, ymin, ymax), points=points)
        mask = mask.T           # shape (W, H) -> (H, W)
        overlay = overlay_mask(mask, mask_alpha_in, mask_alpha_out)

        fig.add_trace(
            go.Image(z=overlay, colormodel="rgba")
        )

    return fig




def get_Kmeans_labels_fig(
    sv,
    shape,
    frequencies,
    n_clusters,
    random_state
):
    roi_sv = get_roi_Sv(sv, shape, frequencies)
    labels_da, kmeans = kmeans_roi_sv(roi_sv, n_clusters, random_state)

    fig = px.imshow(labels_da.values.T)

    return fig