from pathlib import Path
import json
import shutil
import subprocess
from datetime import datetime
import argparse
import glob
import re
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from src.config import load_config
from src.io import load_survey_ds



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


def format_shape_coord(json_dir:Path):
    """Formats the points coordinates of shapes in the JSON files in json_dir:
        - Adds the time frame start index to the time coordinates (convert from image coordinates to survey coordinates)
        - Rounds coordinates and converts to ints.

    Args:
        json_dir (Path): path of the directory contaning labelme JSON files.
    """
    for filename in glob.glob(str(json_dir/'*.json')):

        with open(filename, 'r') as f:
            roi_data = json.load(f)
        
        match = re.search(r'_T(\d+)', roi_data['imagePath'])
        if match is None:
            raise ValueError(f"Could not extract T index from {roi_data['imagePath']}")
        t0 = int(match.group(1))

        for shp in roi_data['shapes']:
            for p in shp['points']:
                p[0] += t0
                p[0], p[1] = int(p[0]), int(p[1])

        with open(filename, "w") as f:
            json.dump(roi_data, f, indent=2)


def normalize_sv_array(sv_array:np.ndarray, vmin:float=-90., vmax:float=-50.):
    a = (sv_array - vmin) / (vmax - vmin)
    a = np.clip(a, 0, 1)
    a = np.nan_to_num(a, nan=0)
    a = a.transpose(2, 1, 0)
    return a

def plot_sv_rgb_image(a, title, outfile):
    fig, ax = plt.subplots(layout='constrained')
    ax.imshow(a, aspect='auto', interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('ESDU')
    ax.set_ylabel('Depth sample')
    plt.savefig(outfile, dpi=300)
    plt.close()

def plot_shape(sv, shape, frequencies, outfile):
    """Save the RGB echogram of an ROI shape.
    """
    # Fetch shape points
    points = np.array(shape["points"])
    # Compute bounding box
    xmin, xmax = points[:, 0].min(), points[:, 0].max()
    ymin, ymax = points[:, 1].min(), points[:, 1].max()
    # Slice sv using bbox
    roi_sv = sv.isel(time=slice(xmin, xmax), depth=slice(ymin, ymax)).sel(channel=frequencies)
    # Turn into image format array
    sv_array = roi_sv.values
    sv_array = normalize_sv_array(sv_array, vmin=-90, vmax=-50)
    plot_sv_rgb_image(sv_array, title=shape["id"], outfile=outfile)


def plot_ROIs(sv, plot_dir:Path, json_dir:Path, frequencies:list=[38, 70, 120]):
    # Ensure plot dir exists
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Loop through JSON files
    for filename in glob.glob(str(json_dir/'*.json')):
        # Load ROI JSON file
        with open(filename, 'r') as f:
            roi_data = json.load(f)
        # Loop through shapes
        for shape in roi_data['shapes']:
            # Create file name for the plot
            outfile = plot_dir / shape['id']
            # Plot
            plot_shape(sv, shape, frequencies, outfile)


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
    session_name = config["session"]["name"]

    # Derived paths
    roi_json_folder_name = 'ROI_' + session_name
    roi_json_path =   images_dir_path / roi_json_folder_name
    work_dir_path = interim_dir_path / config["session"]["ei"] / session_name
    work_dir_json_path = work_dir_path / 'ROI_json_files'

    # Ensure paths exist
    roi_json_path.mkdir(parents=True, exist_ok=True)
    work_dir_json_path.mkdir(parents=True, exist_ok=True)

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

    # Add id's to new shapes
    add_shape_ids(json_dir=roi_json_path,
                  session_id=labelling_session_id,
                  start_id=0)

    # Copy all JSONs to work_dir for further use
    copy_files(in_dir=roi_json_path, out_dir=work_dir_json_path)

    # Format coordinates
    format_shape_coord(json_dir=work_dir_json_path)

    # Save ROI plots in work_dir
    ds = load_survey_ds(survey=config["session"]["ei"], config=config)
    sv = ds["Sv"]
    plot_ROIs(sv, plot_dir=work_dir_path, json_dir=work_dir_json_path, frequencies=[38, 70, 120])