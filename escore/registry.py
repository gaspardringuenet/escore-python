from pathlib import Path
import sqlite3
from datetime import datetime
import json
import hashlib
import re


# Track shapes ids in the LABELME JSON files
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
                shape["id"] = f"{session_id}_{counter:04d}"
                counter += 1

        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)


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


# Cleaning helper functions
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
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)


# SQLite statements
create_roi_registry_sql = """
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
    """

insert_new_roi_sql = """
    INSERT INTO roi_registry
    (id, image, geom_hash, points, it_min, it_max, iz_min, iz_max, shape_type, created, modified, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

update_roi_sql = """
    UPDATE roi_registry
    SET geom_hash = ?, points = ?, it_min = ?, it_max = ?, iz_min = ?, iz_max = ?, shape_type = ?, modified = ?, status = ?
    WHERE id = ?
"""


# sqlite3 helper functions
def open_db(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute(create_roi_registry_sql)
    conn.commit()
    return conn


def add_new_roi(conn, shape, image_path, t_offset, now):
    cur = conn.cursor()

    shape_id = shape.get("id")
    geom_hash = geometry_hash(shape)
    points = clean_points(shape["points"], t_offset)
    points_json = json.dumps(points)
    it_min, it_max, iz_min, iz_max = get_bbox(points)
    shape_type = shape["shape_type"]

    # New shape
    cur.execute(insert_new_roi_sql,
                (shape_id, str(image_path), geom_hash, points_json, it_min, it_max, iz_min, iz_max, shape_type, now, now, "new"))


def modify_roi(conn, shape, t_offset, now):
    cur = conn.cursor()

    shape_id = shape.get("id")
    geom_hash = geometry_hash(shape)
    points = clean_points(shape["points"], t_offset)
    points_json = json.dumps(points)
    it_min, it_max, iz_min, iz_max = get_bbox(points)
    shape_type = shape["shape_type"]

    # New shape
    cur.execute(update_roi_sql,
                (geom_hash, points_json, it_min, it_max, iz_min, iz_max, shape_type, now, "modified", shape_id))
    

def set_unchanged(conn, shape_id):
    cur = conn.cursor()
    cur.execute("UPDATE roi_registry SET status = ? WHERE id = ?", ("unchanged", shape_id))


def set_deleted(conn, json_dir):
    cur = conn.cursor()

    shape_ids = [shape["id"] for json_file in json_dir.glob("*.json") for shape in json.load(open(json_file))["shapes"]]
    placeholders = ",".join("?" for _ in shape_ids)
    sql = f"UPDATE roi_registry SET status = ? WHERE id NOT IN ({placeholders})"

    cur.execute(sql, ["deleted", *shape_ids])


def update_shapes_from_json(conn, json_file, root_path, now):

    with open(json_file, "r") as f:
        data = json.load(f)

    image_path = (json_file/data["imagePath"]).resolve().relative_to(root_path)
    t_offset = get_t_offset(image_name=image_path.name)
    
    for shape in data.get("shapes", []):
        shape_id = shape.get("id")

        if shape_id is None:
            continue # ignore untracked shapes
        
        cur = conn.cursor()
        cur.execute("SELECT geom_hash FROM roi_registry WHERE id = ?", (shape_id, ))
        row = cur.fetchone()

        if row is None:
            add_new_roi(conn, shape, image_path, t_offset, now) # New shape

        elif row[0] != geometry_hash(shape):
            modify_roi(conn, shape, t_offset, now) # Modified shape

        else:
            set_unchanged(conn, shape_id) # Unchanged


def update_registry(json_dir, conn, root_path):
    now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

    for json_file in json_dir.glob("*.json"):
        try:
            with conn:
                update_shapes_from_json(conn, json_file, root_path, now)
        except sqlite3.OperationalError as e:
            print(f"Failed update for registry for JSON file - {json_file.name}\n",)
        
    # Handle deleted shapes
    set_deleted(conn, json_dir)
    conn.commit()


def print_update(conn):
    flags = ['new', 'modified', 'deleted', 'unchanged']
    counts = {}

    try:
        with conn:
            cur = conn.cursor()
            for flag in flags:
                cur.execute("SELECT COUNT(*) FROM roi_registry WHERE status == ?", (flag,))
                counts[flag] = cur.fetchone()[0]
    except sqlite3.OperationalError as e:
        print(e)

    n_tot = counts["new"] + counts["modified"] + counts["unchanged"]
    print("\nSession ROI registry update:",
          f"\n * {counts['new']} new",
          f"\n * {counts['modified']} modified",
          f"\n * {counts['deleted']} deleted",
          f"\n * Total number of ROIs in registry: {n_tot}")


def remove_deleted(conn):
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM roi_registry WHERE status == 'deleted'")
    except sqlite3.OperationalError as e:
        print(e)


def fetch_ROIs_for_plots(conn, config, plot_dir):
    try:
        with  conn:
            cur = conn.cursor()

            if config["session"]["roi_plots"]["force_plot"]:
                sql = "SELECT id, points, it_min, it_max, iz_min, iz_max, status FROM roi_registry"
                cur.execute(sql)
            else:
                id_list = [file.stem for file in plot_dir.glob("*.png")] # list the id's of ROI that have already been plotted
                placeholders = ','.join('?' for _ in id_list)
                sql = f"""SELECT id, points, it_min, it_max, iz_min, iz_max, status FROM roi_registry
                        WHERE (status IN ('new', 'modified')) OR 
                                (status == 'unchanged' AND (id NOT IN ({placeholders})))""" 
                cur.execute(sql, id_list)
            
            # Fetch table rows
            rows = cur.fetchall()

            # Convert to list of shapes (represented by dicts)
            shapes = []
            for row in rows:
                shape = {}
                keys = "id", "points", "it_min", "it_max", "iz_min", "iz_max", "status"
                for i in range(len(row)):
                    shape[keys[i]] = row[i]
                shape["points"] = json.loads(shape["points"])

                shapes.append(shape)

            return shapes

    except sqlite3.OperationalError as e:
        print(e)


def list_valid_ROI_ids(conn):
    cur = conn.cursor()
    cur.execute("SELECT id FROM roi_registry WHERE status != 'deleted'")
    return [id for (id,) in cur.fetchall()]


class ROIRegistry:
    def __init__(self, db_path: Path, root_path: Path):
        self.db_path = db_path
        self.root_path = root_path
        self.conn = self.__open_db__()

    def __open_db__(self):
        return open_db(self.db_path)

    def update(self, json_dir: Path):
        update_registry(
            json_dir=json_dir,
            conn=self.conn,
            root_path=self.root_path
        )
    
    def print_update(self):
        print_update(self.conn)

    def remove_deleted(self):
        remove_deleted(self.conn)

    def fetch_for_plots(self, config, plot_dir: Path):
        return fetch_ROIs_for_plots(self.conn, config, plot_dir)
    
    def close(self):
        self.conn.close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def list_ids(self):
        return(list_valid_ROI_ids(self.conn))
    

def registry_row_to_shape(row):
    shape = {}
    keys = "id", "points", "it_min", "it_max", "iz_min", "iz_max", "status"
    for i in range(len(row)):
        shape[keys[i]] = row[i]
    shape["points"] = json.loads(shape["points"])

    return shape


def get_shape(registry, id):
    cur = registry.conn.cursor()
    sql = "SELECT id, points, it_min, it_max, iz_min, iz_max FROM roi_registry WHERE id == ?"
    cur.execute(sql, (id,))
    row = cur.fetchone()
    return registry_row_to_shape(row)


if __name__ == "__main__":

    registry_path = Path("data/interim/amazomix_3pings1m/test_01/roi_registry.db")
    root_path = Path(__file__).resolve().parent.parent.parent

    with ROIRegistry(registry_path, root_path) as registry:
        print(registry.list_ids())
