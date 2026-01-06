import numpy as np
import xarray as xr

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from skimage.draw import polygon


# Selecting Sv values from ROI shape and sv xr.DataArray

def get_offset(imin, imax, array_length):
    """Compute the offset of the [imin, imax] window in array, i.e. how much this window is out of boundaries.
    Offset is negative if imin < 0 and positive if imax > array_length.
    Assumptions are imin < imax and imax-imin < array_length.
    """
    if imin < 0:
        return imin
    else:
        return max(imax-array_length, 0)
    


def get_window(
    bbox: tuple[int, int, int, int],
    array_shape: tuple[int, int],
    padding: int | None = None,
    window_shape: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """
    Given a bounding box, returns a window contained within an array of shape
    `array_shape`, and containing `bbox`.

    Two methods:
    - padding: expands bbox by a fixed number of pixels (cropped at boundaries)
    - window_shape: fixed-size window centered on bbox (shifted if needed)

    Args:
        bbox: (xmin, xmax, ymin, ymax)
        array_shape: (xlen, ylen)
        padding: number of pixels around bbox
        window_shape: (width, height) of the rigid window

    Returns:
        (xmin, xmax, ymin, ymax)
    """

    if (padding is None) == (window_shape is None):
        raise ValueError("Exactly one of `padding` or `window_shape` must be provided.")

    xmin, xmax, ymin, ymax = bbox
    xlen, ylen = array_shape

    if window_shape is not None:
        win_w, win_h = window_shape

        # Center of bbox
        x_center = (xmin + xmax) // 2
        y_center = (ymin + ymax) // 2

        # Compute window extents (correct for odd sizes)
        half_w_left = win_w // 2
        half_w_right = win_w - half_w_left
        half_h_top = win_h // 2
        half_h_bottom = win_h - half_h_top

        #print(f"win_w%2: {win_w%2} | win_h%2: {win_h%2}")
        xmin = x_center - half_w_left + (1 - win_w % 2)
        xmax = x_center + half_w_right - win_w % 2
        ymin = y_center - half_h_top + (1 - win_h % 2)
        ymax = y_center + half_h_bottom - win_h % 2

        # Shift window to stay inside array
        if xmin < 0:
            xmax -= xmin
            xmin = 0
        if xmax > xlen:
            xmin -= xmax - xlen
            xmax = xlen

        if ymin < 0:
            ymax -= ymin
            ymin = 0
        if ymax > ylen:
            ymin -= ymax - ylen
            ymax = ylen

    else:
        xmin -= padding
        xmax += padding
        ymin -= padding
        ymax += padding

        # Crop to array bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xlen, xmax)
        ymax = min(ylen, ymax)

    return xmin, xmax, ymin, ymax




def mask_from_rectangle(mask_shape, points):
    mask = np.zeros(mask_shape)

    mask[points[:, 0].min():points[:, 0].max()+1, 
         points[:, 1].min():points[:, 1].max()+1] = 1
    
    return (mask == 1)



def mask_from_polygon(mask_shape, points):
    mask = np.zeros(mask_shape)

    rr, cc = polygon(
        points[:, 0],  # rows
        points[:, 1],  # cols
        shape=mask.shape
    )
    mask[rr, cc] = 1

    return (mask == 1)



def get_mask(
    window: tuple[int, int, int, int],
    points: list
):
    """Returns a boolean mask of pixels contained within a shape delimited by an ordered list of points.

    Args:
        window (tuple[int, int, int, int]): coordinates of a rectangular windows on witch to draw the mask. Usually the bbox of the shape in a larger array. 
        points (list): list of [x, y] coordinates of ordered to create a convex polygon. Rectangles are represented by 2 points of one of their diagonal.

    Returns:
        numpy.ndarray: boolean mask of shape to overlay on top of window.
    """

    # Get the coordinates of the visualization window
    xmin, xmax, ymin, ymax = window

    # Convert points in the window coordinates system
    points = np.array([[p[0]-xmin, p[1]-ymin] for p in points])

    # Create mask
    mask_shape = (xmax-xmin+1, ymax-ymin+1)
    if len(points)==2:
        mask = mask_from_rectangle(mask_shape, points)
        
    elif len(points) > 2:
        mask = mask_from_polygon(mask_shape, points)
    
    else:
        pass # 1 or 0 points are ignored: return an empty mask

    return mask



def get_roi_Sv(
    sv: xr.DataArray,
    shape: dict, 
    frequencies=[38, 70, 120, 200]
):
    # Fetch shape points
    points = np.array(shape["points"])
    bbox = shape["it_min"], shape["it_max"], shape["iz_min"], shape["iz_max"]
    xmin, xmax, ymin, ymax = bbox

    # Slice sv using bbox (avoids hard loading all the sv values)
    bbox_sv = sv.isel(time=slice(xmin, xmax+1), depth=slice(ymin, ymax+1)).sel(channel=frequencies)

    # Get mask
    mask = get_mask(window=bbox, points=points)

    # Convert mask to DataArray
    mask_da = xr.DataArray(
        mask,
        dims=('time', 'depth'),
        coords = {'time':bbox_sv.time, 'depth':bbox_sv.depth}
    )

    # Apply mask
    roi_sv = bbox_sv.where(mask_da)

    return roi_sv



def compute_delta_sv(
    sv: xr.DataArray,
    ref_frequency: float
) -> xr.DataArray:
    """Subtract a reference channel to volume backscattering (Sv) values contained in the other channels of sv.
    Computes ΔSv_ref = Sv(channel) - Sv(ref) for all channels.

    Args:
        sv (xr.DataArray): multi-frequency Sv data with a 'channel' dimension.
        ref_frequency (float): reference channel coordinate.

    Returns:
        xr.DataArray: DataArray of same shape as sv, except for one less on the channel dimension. channel coords are renamed.
    """
    
    if "channel" not in sv.dims:
        raise ValueError("Input DataArray must have a 'channel' dimension.")
    
    if sv.sizes["channel"] < 2:
        raise ValueError(
            f"At least 2 frequency channels are required for r(f) clustering: {sv.sizes['channel']} available."
        )

    try:
        sv_ref = sv.sel(channel=ref_frequency)
    except KeyError:
        raise ValueError(
            f"Reference frequency {ref_frequency} kHz does not match available frequencies:"
            f"{sv.channel.values}"
        )
    
    sv_other = sv.drop_sel(channel=ref_frequency)

    delta_sv = sv_other - sv_ref

    # Rename variable
    delta_sv = delta_sv.rename("Delta Sv")

    # Add a dimension for reference frequency (channels are kept identical)
    delta_sv = delta_sv.expand_dims(
        reference_frequency= np.array([ref_frequency], dtype=np.float64)
    )

    # Add metadata
    delta_sv.reference_frequency.attrs.update({
        "units": "kHz",
        "long_name": "Reference frequency"
    })

    return delta_sv



def stack_pixels(da: xr.DataArray):

    # Stack spatial dimensions
    stacked = da.stack(pixel=("time", "depth"))

    # Drop pixels with NaNs
    stacked = stacked.dropna(dim="pixel", how="any")

    # shape to (n_pixels, n_channels) as expected by most clustering algorithms
    stacked = stacked.transpose("pixel", "channel")

    return stacked # Contains the necessary information for unstacking



def cluster_roi(
    roi_sv: xr.DataArray,
    features: str,
    method: str,
    n_clusters: int, 
    ref_frequency: float,
    random_state: int=0,
):
    
    # Create clustering model
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    elif method == "GMM":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    else:
        raise ValueError(f"Clustering method must be one of ['KMeans', 'GMM']. Current input: '{method}'")

    # Get the right variables (Sv or Delta Sv)
    if features == "Sv":
        data = roi_sv
    
    elif features == "Delta Sv":
        data = compute_delta_sv(roi_sv, ref_frequency)
        data = data.squeeze("reference_frequency", drop=True) # remove reference frequency dim for use in stack_pixels
    
    else:
        raise ValueError(f"Clustering features must be one of ['Sv', 'Delta Sv']. Current input: '{features}'")

    # Stack pixels of data into clustering compatible format
    X = stack_pixels(data)

    # Run clustering
    labels = model.fit_predict(X.values)

    # Create label DataArray
    labels_pixel = xr.DataArray(
        labels,
        dims="pixel",
        coords={"pixel": X.pixel}
    )

    # Unstack to (time, depth)
    labels_da = labels_pixel.unstack("pixel")
    
    return labels_da, model