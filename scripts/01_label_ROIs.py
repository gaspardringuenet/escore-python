from pathlib import Path
import json
import shutil
import subprocess
from datetime import datetime

import argparse
from src.config import load_config


def add_shape_ids(json_dir: Path, session_id: str, start_id: int = 0):
    """
    Add unique 'id' to each shape in all JSONs in json_dir.
    IDs are prefixed with session_id and are unique per session.
    """
    counter = start_id

    for json_file in json_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            if "id" not in shape:  # newly created shape
                shape["id"] = f"{session_id}_{counter:03d}"
                counter += 1

        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

    print(f"{counter} new ROI(s) have been created during sessions. (Note that modified ROIs are not counted)")



def copy_files(in_dir: Path, out_dir: Path):
    """
    Copy JSON files from in_dir to out_dir.
    """
    for json_file in in_dir.glob("*.json"):
        shutil.copyfile(json_file, out_dir/json_file.name)



if __name__ == '__main__':

    # Parse config argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/config.yml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create paths
    # User inputs
    images_dir_path = Path(config["session"]["images_dir"])
    interim_dir_path = Path(config["paths"]["interim_dir"])
    library = config["session"]["library_name"]

    # Derived paths
    roi_json_folder_name = 'ROI_' + library
    roi_json_path =   images_dir_path / roi_json_folder_name
    work_dir_path = interim_dir_path / library
    work_dir_json_path = work_dir_path / 'ROI_json_files'

    # Ensure paths exist
    roi_json_path.mkdir(parents=True, exist_ok=True)
    work_dir_json_path.mkdir(parents=True, exist_ok=True)

    # Create new session_id
    session_id = datetime.today().strftime('%Y-%m-%d_%H%M') # for ROI ids
    print(f"Current session id: {session_id}")

    # Launch labelme as subprocess
    subprocess.run([
        "labelme",
        str(images_dir_path),
        '--output',
        str(roi_json_path),
        '--nodata'  # avoids encoding the image in the json file
    ])

    # Add id's to new shapes
    add_shape_ids(json_dir=roi_json_path,
                session_id=session_id, start_id=0)

    # Copy all JSONs to work_dir for further use
    copy_files(in_dir=roi_json_path, out_dir=work_dir_json_path)