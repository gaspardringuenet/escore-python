import argparse

from escore.config import load_config
from escore.io import load_survey_ds


if __name__ == '__main__':

    # Parse config argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/config.yml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load Abracos ds
    ds = load_survey_ds(survey="abracos1_3pings1m",
                        config=config)
    
    ds_subset = ds.isel(time=slice(5000, 6500))

    ds_subset.to_netcdf("data/input/netCDF-echointegrations/abracos1_3pings1m_subset.nc")