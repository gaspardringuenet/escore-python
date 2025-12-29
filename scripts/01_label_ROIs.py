from pathlib import Path
import subprocess
from datetime import datetime
import argparse
import sqlite3
import json

from tqdm import tqdm

from src.config import load_config
from src.io import load_survey_ds
from src.registry import add_shape_ids, open_db, update_registry, print_update, remove_deleted, fetch_ROIs_for_plots
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
    registry_path = work_dir_path / "roi_registry.db"

    # Ensure folder paths exist
    roi_json_path.mkdir(parents=True, exist_ok=True)
    work_dir_plot_path.mkdir(parents=True, exist_ok=True)

    # Create new ROI labelling session id
    labelling_session_id = datetime.today().strftime('%Y-%m-%d_%H%M') # for ROI ids
    print(f"Current ROI labelling session id: {labelling_session_id}")

    # Launch labelme as subprocess
    subprocess.run([
        "labelme",
        str(images_dir_path),
        '--output',
        str(roi_json_path),
        '--nodata'  # avoids encoding the image in the json file
    ])

    # Add id's and update geometry hash
    print("Creating id's for new ROIs.")
    add_shape_ids(json_dir=roi_json_path, session_id=labelling_session_id, start_id=0)
    #update_geom_hash_json(json_dir=roi_json_path)

    # Update registry
    print(f"Updating ROI registry file at: {registry_path}")

    # Open registry (create if it does not exist)
    conn = open_db(registry_path)
    update_registry(json_dir=roi_json_path, conn=conn, root_path=HERE)
    print_update(conn)
   
    # Remove deleted shapes from registry
    remove_deleted(conn)

    # Save ROI plots in work_dir
    # MODIFY : save only for new and modified. Delete when deleted.
    ds = load_survey_ds(survey=config["session"]["ei"], config=config)
    sv = ds["Sv"]

    plot_dir = work_dir_plot_path / 'ROIs_RGB'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Get ROIs for plots
    shapes = fetch_ROIs_for_plots(conn, config, plot_dir)

    print(f"Plotting {len(shapes)} ROI's...")
    for shape in tqdm(shapes, desc="ROI's"):
        outfile = plot_dir / f"{shape['id']}.png"

        plot_shape(sv, shape, outfile, 
                   padding=config["session"]["roi_plots"]["padding"],
                   frequencies=config["session"]["roi_plots"]["frequencies"])

    # Close connection to db
    conn.close()


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