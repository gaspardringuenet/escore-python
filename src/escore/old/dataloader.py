from pathlib import Path
import xarray as xr
import glob
import requests
from typing import Sequence



def download_test_data(filepath: Path, url:str) -> None:

    print(f'Downloading netCDF file: {filepath.name}...')
    r  = requests.get(url)

    with open(filepath, 'wb') as f:
        f.write(r.content)

    print(f'Saved to {filepath}')

    return None



def ensure_time_monotony(ds: xr.Dataset):
    return ds.sortby('time')



def check_no_time_overlap(ds_list: list[xr.Dataset]):
    times = [(ds.time.min().data, ds.time.max().data) for ds in ds_list]
    times = sorted(times)

    for (_, end), (start, _) in zip(times[:-1], times[1:]):
        if start <= end:
            raise ValueError("Overlapping time coordinates detected.")
        

def load_dataset(
        dir_path:Path,
        chunks:dict | None = None,
        backup_url:str = 'https://www.seanoe.org/data/00602/71379/data/70042.nc',
        backup_fname:str = '70042.nc'
) -> xr.Dataset:
    """Load all .nc files, as long as they have the same 'cruise_name' attr, and their time coordinates do not overlap. 
    Makes it straightforward to output images, but future versions should be able to handle more cases (differents cruises, overlapping time coordinates).

    Args:
        dir_path (Path): folder containing .nc file. Data from subfolders is not accounted for.
        chunks (dict, optional): time and depth chunks for xr.Dataset lazy loading. Defaults to {"time": 1000, "depth": 100}.
        backup_url (str, optional): URL from which to download test data in case no file is present in directory. Defaults to 'https://www.seanoe.org/data/00602/71379/data/70042.nc'.
        backup_fname (str, optional): Name of the file accessed via backup_url. Defaults to '70042.nc'.

    Returns:
        xr.Dataset: The desired dataset.
    """

    if not dir_path.is_dir():
        raise ValueError(f"Input directory does not exist.")

    # Fetch .nc file in directory
    files = [Path(p) for p in glob.glob(str(dir_path / '*.nc'))]

    # If no file is present, download test data from the internet
    if not files:
        filepath = dir_path / backup_fname
        download_test_data(filepath, backup_url)

        files = [filepath] # Add the new data to the list of files

    return load_dataset_from_files(files, chunks)



def load_dataset_from_files(
        files: Sequence[Path],
        chunks: dict | None = None,
) -> xr.Dataset:
    # TODO Add an echointegration parameters check !
    
    if not files:
        raise ValueError("No netCDF files provided.")
    
    # Defaults chunks size
    if chunks is None:
        chunks = {"time": 1000, "depth": 100}

    # Open xr.Dataserts from all file (use lazy loading)
    ds_list = [xr.open_dataset(file, chunks=chunks) for file in files]

    # Fetch cruise names and validate
    cruise_names = set([ds.attrs.get('cruise_name') for ds in ds_list])

    if None in cruise_names:
        raise ValueError("Missing 'cruise_name' attribute in one or more datasets")

    if len(cruise_names) > 1: 
        raise ValueError(f"Datasets having different cruise_name attributes are not handled. Cruise names: {cruise_names}")
    
    
    ds_list = [ensure_time_monotony(ds) for ds in ds_list]          #TODO Temporary fix (non monotony issue affects ABRAÇOS01 data)
    ds_list = sorted(ds_list, key=lambda ds: ds.time.values[0])     # Sort Datasets with regards to time
    check_no_time_overlap(ds_list)

    # Concatenate datasets
    try:
        ds_combined  = xr.concat(
            ds_list,
            dim='time', 
            data_vars='all',
            combine_attrs="drop_conflicts"
        )
    except Exception as e:
        files_list_str = '\n-' + '\n-'.join(files)
        raise ValueError(f"Unable to concatenate Datasets along the time dimensions. netCDF files:{files_list_str}") from e
    
    return ds_combined



if __name__ == "__main__":

    here = Path(__file__).parent.parent
    ds = load_dataset(dir_path=here/"data-perso/test/empty")
    
    print(f"Cruise name: {ds.attrs.get('cruise_name')}")
    print(f"Ping axis EI: {ds.attrs.get('data_ping_axis_interval_value')} x {ds.attrs.get('data_ping_axis_interval_type')}")
    print(f"Ping axis EI: {ds.attrs.get('data_range_axis_interval_value')} x {ds.attrs.get('data_range_axis_interval_type')}")