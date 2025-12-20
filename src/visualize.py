import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def sv2array(sv:xr.DataArray, time_idx_slice=slice(0, 100), depth_idx_slice=slice(0, 100), channels:int|tuple[int, int, int]=(38, 70, 120)):
    channels = channels if (type(channels)==int) else list(channels)
    sv_array = sv.isel(time=time_idx_slice, depth=depth_idx_slice).sel(channel=channels).values
    return sv_array



