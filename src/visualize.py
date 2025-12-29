import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sqlite3


def sv2array(sv:xr.DataArray, time_idx_slice=slice(0, 100), depth_idx_slice=slice(0, 100), channels:int|tuple[int, int, int]=(38, 70, 120)):
    """Not sure about this one.
    """
    channels = channels if (type(channels)==int) else list(channels)
    sv_array = sv.isel(time=time_idx_slice, depth=depth_idx_slice).sel(channel=channels).values
    return sv_array


def normalize_sv_array(sv_array:np.ndarray, vmin:float=-90., vmax:float=-50.):
    a = (sv_array - vmin) / (vmax - vmin)
    a = np.clip(a, 0, 1)
    a = np.nan_to_num(a, nan=0)
    a = a.transpose(2, 1, 0)
    return a


def plot_sv_rgb_image(ax, a, title, outfile):
    ax.imshow(a, aspect='auto', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('ESDU')
    ax.set_ylabel('Depth sample')


def get_mask_from_points(a, points, xmin, ymin):
    points[:, 0] -= xmin
    points[:, 1] -= ymin

    # Assume points represent a rectangle


def plot_shape(sv, shape, outfile, padding=10, frequencies=[38, 70, 120]):
    """Save the RGB echogram of an ROI shape.
    """
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

    fig, ax = plt.subplots(layout='constrained')

    # Plot RGB image
    plot_sv_rgb_image(ax, sv_array, title=shape["id"], outfile=outfile)

    # Overlay mask showing selected points
    #mask = get_mask_from_points(sv_array, points, xmin, ymin)
    ax.plot(points[:, 0]-xmin, points[:, 1]-ymin)

    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_ROI_summaries():
    pass