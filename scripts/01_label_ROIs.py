from pathlib import Path
import subprocess
from datetime import datetime
import argparse
import sqlite3
import json

from tqdm import tqdm

from src.config import load_config
from src.io import load_survey_ds
from src.registry import add_shape_ids, ROIRegistry
from src.visualize import plot_shape


def main(config):
    # Create paths
    # User inputs
    images_dir_path = Path(config["session"]["images_dir"])
    interim_dir_path = Path(config["paths"]["interim_dir"])
    session_name = config["session"]["name"]

    # Derived paths
    roi_json_folder_name = 'ROI_' + session_name
    roi_json_path =   images_dir_path / roi_json_folder_name
    work_dir_path = interim_dir_path / config["session"]["ei"] / session_name
    work_dir_plot_path = work_dir_path / 'plots'
    roi_plot_dir = work_dir_plot_path / 'ROIs_RGB'
    registry_path = work_dir_path / "roi_registry.db"

    # Ensure folder paths exist
    roi_json_path.mkdir(parents=True, exist_ok=True)
    work_dir_plot_path.mkdir(parents=True, exist_ok=True)
    roi_plot_dir.mkdir(parents=True, exist_ok=True)

    # Create new ROI labelling session id
    labelling_session_id = datetime.today().strftime('%Y-%m-%d_%H%M') # for ROI ids
    print(f"\n**** Current ROI labelling session ****")
    print(f" - Id:\t\t{labelling_session_id}")
    print(f" - Name:\t{config['session']['name']}")
    print(f" - EI:\t\t{config['session']['ei']}")
    print(f" - RGB images:\t{config['session']['images_dir']}")

    # Launch labelme as subprocess
    subprocess.run([
        "labelme",
        str(images_dir_path),
        '--output',
        str(roi_json_path),
        '--nodata'  # avoids encoding the image in the json file
    ])

    # Add id's and update geometry hash
    #print("\nCreating id's for new ROIs.")
    add_shape_ids(json_dir=roi_json_path, session_id=labelling_session_id, start_id=0)

    # Update registry
    print(f"\nUpdating ROI registry file at: {registry_path}")
    with ROIRegistry(db_path=registry_path, root_path=HERE) as registry:
        registry.update(json_dir=roi_json_path)     # Update the registry
        registry.print_update()                     # Print an update of new/modified/deleted ROIs
        roi_shapes = registry.fetch_for_plots(config, plot_dir=roi_plot_dir) # Fetch data from registry for plots
        registry.remove_deleted()                   # Remove deleted shapes from registry
            
    # Save ROI plots in work_dir
    # MODIFY : Delete when deleted.
    ds = load_survey_ds(survey=config["session"]["ei"], config=config)
    sv = ds["Sv"]

    print(f"\nPlotting {len(roi_shapes)} ROIs to - {roi_plot_dir}")
    for shape in tqdm(roi_shapes, desc="ROIs"):
        outfile = roi_plot_dir / f"{shape['id']}.png"

        plot_shape(sv, shape, outfile, 
                   padding=config["session"]["roi_plots"]["padding"],
                   frequencies=config["session"]["roi_plots"]["frequencies"])


if __name__ == '__main__':

    HERE = Path(__file__).resolve().parent.parent

    # Parse config argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/config.yml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Execute main
    main(config)


# Plot new and modified shapes
# - Loop trough shapes df
# - Where status in ['new', 'modified']:
#   - Plot RGB echogram of bbox + padding
#   - Plot contour line of the shape (make it work for non rectangle shapes)

# Remove plots of deleted shapes