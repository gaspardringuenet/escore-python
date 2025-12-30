import numpy as np
import plotly.express as px

from escore.visualize import normalize_sv_array


def get_RGB_fig(sv, shape, vmin=-50., vmax=-90., padding=10, frequencies=[38, 70, 120]):

    # Fetch shape points
    points = np.array(shape["points"])

    # Window
    xmin, xmax, ymin, ymax = shape["it_min"]-padding, shape["it_max"]+padding, shape["iz_min"]-padding, shape["iz_max"]+padding
    
    # Make sur the window isn't out of boundaries
    xmin, ymin = max(xmin, 0), max(ymin, 0)
    xmax, ymax = min(xmax, len(sv.time)), min(ymax, len(sv.depth))

    # Slice sv using bbox
    roi_sv = sv.isel(time=slice(xmin, xmax), depth=slice(ymin, ymax)).sel(channel=frequencies)

    # Turn into image format array
    sv_array = roi_sv.values
    sv_array = normalize_sv_array(sv_array, vmin=-90, vmax=-50)

    fig = px.imshow(sv_array)

    fig.layout.xaxis.title.text = "ESDU"
    fig.layout.yaxis.title.text = "Depth sample"

    fig.layout.title = dict(text = "RGB plot of ROI + padding",
                            x=0.5,
                            xanchor='center')

    return fig