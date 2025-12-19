from datetime import datetime
import numpy as np
import xarray as xr

from .data_config import DATA_DICT, EI_DICT

# Import function allowing to import and combine legs as a single `xarray.Dataset`
def load_survey_ds(survey, chunks={"time": 1000, "depth": 100}):
    leg_list = []
    
    for leg_name in EI_DICT.get(survey):
        leg_id = leg_name.split("_")[-1]

        leg_path = DATA_DICT.get(leg_name)

        leg_ds = xr.open_dataset(leg_path, chunks=chunks)

        # Add a 'leg' variable to each dataset
        leg_ds = leg_ds.assign_coords(leg=(('time',), [leg_id]*leg_ds.sizes['time']))

        leg_list.append(leg_ds)

    # Concatenate and sort by time to ensure order
    combined = xr.concat(leg_list, dim='time', data_vars='all')
    
    if 'time' in combined:
        ds = combined.sortby('time')
    else:
        ds = combined

    return ds

# Get basic information on the survey
# Get start and end time as datetimes
def get_start_end_time_str(ds: xr.Dataset, 
                           dest_format: str = '%d %b %Y, %H:%M',
                           leg: str = None):
    if leg :
        mask = ds.leg == leg
        time_dates = ds["time"].where(mask, drop=True)
    else:
        time_dates = ds["time"]
    
    start_dt = time_dates.values[0].astype('M8[ms]').astype(datetime)
    start = start_dt.strftime(dest_format)
    end_dt = time_dates.values[-1].astype('M8[ms]').astype(datetime)
    end = end_dt.strftime(dest_format)

    return start, end

# Summary function
def print_file_infos(ds: xr.Dataset, tz="UTC"):
    # Name of the cruise
    print(f"* Title:\t{ds.title}")

    n_legs = len(np.unique(ds.leg.values))
    print(f"* N legs:\t{n_legs}")
    
    for leg in np.unique(ds.leg.values):
        start, end = get_start_end_time_str(ds, leg=leg)
        print(f"* Dates ({leg}):\t{str(start)} - {end} ({tz})")
    
    print(f"* Resolution:\t{ds.data_ping_axis_interval_value} {ds.data_ping_axis_interval_type} x {ds.data_range_axis_interval_value} {ds.data_range_axis_interval_type}")
    print(f"* Dimensions:\t({ds.sizes['time']}, {ds.sizes['depth']}, {ds.sizes['channel']}) - (time, depth, channel)")
    print(f"* N pixels:\t{ds.sizes['time'] * ds.sizes['depth']}")