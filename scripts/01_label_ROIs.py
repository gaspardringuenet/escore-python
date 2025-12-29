import os
from pathlib import Path
import json
import shutil
import subprocess
from datetime import datetime
import argparse
import glob
import re
import numpy as np
import hashlib
import sqlite3

from tqdm import tqdm

import matplotlib.pyplot as plt

from src.config import load_config
from src.io import load_survey_ds


def add_shape_ids(json_dir: Path, session_id: str, start_id: int = 0):
    """
    Add unique 'id' to each shape in all JSONs in json_dir.
    IDs are prefixed with session_id and are unique per session.

    --> Add a field indicating if shape is new / has been modified ?
    (I don't know how to do that if shape has been modified).
    """
    counter = start_id

    for json_file in json_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            if "id" not in shape:  # newly created shape
                shape["id"] = f"{session_id}_{counter:04d}"
                counter += 1

        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

    print(f"{counter} new ROI(s) have been created during sessions. (Note that modified ROIs are not counted)")


def geometry_hash(shape):
    payload = {
        "shape_type": shape["shape_type"],
        "points": shape["points"],
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()


def update_geom_hash_json(json_dir: Path):
    for json_file in json_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        for shape in data.get("shapes", []):
            shape["geom_hash"] = geometry_hash(shape)

        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)


def get_t_offset(image_name:str):
        match = re.search(r'_T(\d+)', str(image_name))
        if match is None:
            raise ValueError(f"Could not extract T index from name: {image_name}")
        offset = int(match.group(1))

        return offset


def clean_points(points:list, t_offset:int):
    for p in points:
        p[0], p[1] = int(p[0]) + t_offset, int(p[1])
    return points


def get_bbox(points: list):
    points_array = np.array(points)
    it_min = int(points_array[:, 0].min())
    it_max = int(points_array[:, 0].max())
    iz_min = int(points_array[:, 1].min())
    iz_max = int(points_array[:, 1].max())
    return it_min, it_max, iz_min, iz_max


def open_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS roi_registry (
            id TEXT PRIMARY KEY,
            image TEXT NOT NULL,
            geom_hash TEXT NOT NULL,
            points TEXT NOT NULL,         -- JSON
            it_min INTEGER NOT NULL,
            it_max INTEGER NOT NULL,
            iz_min INTEGER NOT NULL,
            iz_max INTEGER NOT NULL,
            shape_type TEXT NOT NULL,
            created TEXTE NOT NULL,
            modified TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def update_registry(json_dir, conn, root_path):
    now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    cur = conn.cursor()

    shape_ids = []

    for json_file in json_dir.glob("*.json"):

        with open(json_file, "r") as f:
            data = json.load(f)

        image_path = (json_file/data["imagePath"]).resolve().relative_to(root_path)
        t_offset = get_t_offset(image_name=image_path.name)
        
        for shape in data.get("shapes", []):
            shape_id = shape.get("id")

            if shape_id is None:
                continue # ignore untracked shapes
            
            shape_ids.append(shape_id)

            geom_hash = shape.get("geom_hash", geometry_hash(shape))
            points = clean_points(shape["points"], t_offset)
            points_json = json.dumps(shape["points"])
            it_min, it_max, iz_min, iz_max = get_bbox(points)
            shape_type = shape["shape_type"]
            
            cur.execute(
                "SELECT geom_hash FROM roi_registry WHERE id = ?",
                (shape_id, )
            )
            row = cur.fetchone()

            if row is None:
                # New shape
                cur.execute("""
                    INSERT INTO roi_registry
                    (id, image, geom_hash, points, it_min, it_max, iz_min, iz_max, shape_type, created, modified, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    shape_id, str(image_path), geom_hash, points_json, it_min, it_max, iz_min, iz_max, shape_type, now, now, "new"
                ))

            elif row[0] != geom_hash:
                # Modified shape
                cur.execute("""
                    UPDATE roi_registry
                    SET geom_hash = ?, points = ?, it_min = ?, it_max = ?, iz_min = ?, iz_max = ?, modified = ?, status = ?
                    WHERE id = ?
                """, (
                    geom_hash, points_json, it_min, it_max, iz_min, iz_max, now, "modified", shape_id
                ))

            else:
                # Unchanged
                cur.execute("""
                    UPDATE roi_registry
                    SET status = ?
                    WHERE id = ?
                """, (
                    "unchanged", shape_id
                ))

    # Handle deleted shapes
    shape_ids = [shape["id"] for json_file in json_dir.glob("*.json")
                for shape in json.load(open(json_file))["shapes"]]

    placeholders = ",".join("?" for _ in shape_ids)
    sql = f"""
        UPDATE roi_registry
        SET status = ?
        WHERE id NOT IN ({placeholders})
    """
    cur.execute(sql, ["deleted", *shape_ids])

    conn.commit()


def remove_deleted(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM roi_registry WHERE status == 'deleted'")
    conn.commit()


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
    """
    WORK IN PROGRESSS
    Current issues:
        - Progress bar for JSONs, but i'd like for shapes.
        - Plot only new / modified shape --> could add a 'new' flag to shapes.
    """
    # Ensure plot dir exists
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Loop through JSON files
    for filename in tqdm(glob.glob(str(json_dir/'*.json'))):
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

    HERE = Path(__file__).resolve().parent.parent

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
    registry_path = work_dir_path / "roi_registry.db"

    # Ensure folder paths exist
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

    # Add id's and update geometry hash
    print(f"Formatting and copying JSON files to {work_dir_json_path}")
    add_shape_ids(json_dir=roi_json_path, session_id=labelling_session_id, start_id=0)
    update_geom_hash_json(json_dir=roi_json_path)

    # Open registry (create if it does not exist)
    conn = open_db(registry_path)
    update_registry(json_dir=roi_json_path, conn=conn, root_path=HERE)

    # Load new and modified shapes
    import pandas as pd
    df = pd.read_sql("SELECT * FROM roi_registry WHERE status != 'unchanged'", conn)

    n_new = len(df[df['status'] == "new"].index)
    n_mod = len(df[df['status'] == "modified"].index)
    n_del = len(df[df['status'] == "deleted"].index)

    print(f"Session tracks update: {n_new} new, {n_mod} modified, {n_del} deleted.")

    # Remove deleted shapes from registry
    remove_deleted(conn)

    # Save ROI plots in work_dir
    # MODIFY : save only for new and modified. Delete when deleted.
    ds = load_survey_ds(survey=config["session"]["ei"], config=config)
    sv = ds["Sv"]
    #print("Plotting ROI summary figures")
    #plot_ROIs(sv, plot_dir=work_dir_path, json_dir=work_dir_json_path, frequencies=[38, 70, 120])



## Copy all JSONs to work_dir for further use
#copy_files(in_dir=roi_json_path, out_dir=work_dir_json_path)
#
# Format coordinates
#format_shape_coord(json_dir=work_dir_json_path)