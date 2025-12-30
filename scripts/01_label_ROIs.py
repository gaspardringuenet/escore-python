from pathlib import Path
import subprocess
from datetime import datetime
import argparse

from tqdm import tqdm

from escore.config import load_config
from escore.io import load_survey_ds
from escore.registry import add_shape_ids, ROIRegistry
from escore.visualize import plot_shape


def main(config):
    # Create paths
    # User inputs
    images_dir = Path(config["session"]["images_dir"])
    interim_dir = Path(config["paths"]["interim_dir"])
    session_name = config["session"]["name"]
    ei = config["session"]["ei"]

    # Derived paths
    json_dir = images_dir / ('ROI_' + session_name)
    work_dir = interim_dir / ei / session_name
    plot_dir = work_dir / 'plots' / 'ROIs_RGB'
    registry_path = work_dir / "roi_registry.db"

    # Ensure folder paths exist
    json_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create new ROI labelling session id
    labelling_session_id = datetime.today().strftime('%Y-%m-%d_%H%M') # for ROI ids

    print(f"=== ROI labelling session ===")
    print(f" - Id:\t\t{labelling_session_id}")
    print(f" - Name:\t{session_name}")
    print(f" - EI:\t\t{ei}")
    print(f" - RGB images:\t{images_dir}")
    
    # Launch labelme as subprocess
    subprocess.run([
        "labelme",
        str(images_dir),
        '--output',
        str(json_dir),
        '--nodata'  # avoids encoding the image in the json file
    ])

    # Add id's and update geometry hash
    #print("\nCreating id's for new ROIs.")
    add_shape_ids(json_dir=json_dir, session_id=labelling_session_id, start_id=0)

    # Update registry
    print(f"\nUpdating ROI registry file at: {registry_path}")
    with ROIRegistry(db_path=registry_path, root_path=HERE) as registry:
        registry.update(json_dir=json_dir)     # Update the registry
        registry.print_update()                     # Print an update of new/modified/deleted ROIs
        roi_shapes = registry.fetch_for_plots(config, plot_dir=plot_dir) # Fetch data from registry for plots
        registry.remove_deleted()                   # Remove deleted shapes from registry
            
    # Save ROI plots in work_dir
    # MODIFY : Delete when deleted.
    ds = load_survey_ds(survey=config["session"]["ei"], config=config)
    sv = ds["Sv"]

    print(f"\nPlotting {len(roi_shapes)} ROIs to - {plot_dir}")
    for shape in tqdm(roi_shapes, desc="ROIs"):
        outfile = plot_dir / f"{shape['id']}.png"

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