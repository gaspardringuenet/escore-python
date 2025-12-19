from pathlib import Path
from src.builder import DatasetConfig, build_dataset

import argparse
from src.config import load_config


if __name__ == '__main__':

    # Parse config argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/config.yml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    img_config = config["image_dataset"]

    # Set the DatasetConfig before building the Dataset
    dataset_config = DatasetConfig(time_frame_size=img_config["time_frame_size"],
                                    vmin=img_config["vmin"], 
                                    vmax=img_config["vmax"],
                                    z_min_idx=img_config["z_min_idx"],
                                    z_max_idx=img_config["z_max_idx"], 
                                    frequencies=img_config["frequencies"], 
                                    echogram_cmap=img_config["echogram_cmap"])

    # Build Dataset
    build_dataset(
        dataset_config=dataset_config, 
        global_config=config,
        ei_list=img_config["ei_list"], 
        root_path=Path(config["paths"]["echogram_images_dir"])
    )