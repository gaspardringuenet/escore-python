from pathlib import Path
import json
from dash import Dash, dcc, html, callback, Input, Output
import plotly.express as px
import numpy as np

from escore.config import load_config
from escore.io import load_survey_ds
from escore.registry import ROIRegistry
from escore.visualize import normalize_sv_array


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


def get_RGB_fig(sv, shape, vmin=-50., vmax=-90., padding=10, frequencies=[38, 70, 120]):

    # Fetch shape points
    points = np.array(shape["points"])

    # Window
    xmin, xmax, ymin, ymax = shape["it_min"]-padding, shape["it_max"]+padding, shape["iz_min"]-padding, shape["iz_max"]+padding
    
    # Make sur the window isn't out of boundaries
    xmin, ymin = max(xmin, 0), max(ymin, 0)
    xmax, ymax = min(xmax, len(sv.time)), min(ymax, len(sv.depth))

    # Slice sv using bbox
    roi_sv = sv.isel(time=slice(xmin, xmax), depth=slice(ymin, ymax)).sel(channel=frequencies)

    # Turn into image format array
    sv_array = roi_sv.values
    sv_array = normalize_sv_array(sv_array, vmin=-90, vmax=-50)

    fig = px.imshow(sv_array)

    return fig


def main(config):

    # Fetch config for pathz
    interim_dir = Path(config["paths"]["interim_dir"])
    session_name = config["session"]["name"]
    ei = config["session"]["ei"]

    # Derive paths
    work_dir = interim_dir / ei / session_name
    registry_path = work_dir / "roi_registry.db"

    # Fetch one ROI from registry
    with ROIRegistry(db_path=registry_path, root_path=HERE) as registry:
        ids = registry.list_ids()

    # Load sv
    ds = load_survey_ds(survey=config["session"]["ei"], config=config)
    sv = ds["Sv"]

    markdown_text = f"""
    ### Test

    Now we try to plot the RGB image of the first ROI.

    Next things we'd like to do:
    - [ ]   Be able to change the ROI with a Dropdown menu of ROIs.
    """

    app = Dash(external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"])

    app.layout = html.Div([
        html.H1(children="Echo-types Selection App"),
        dcc.Dropdown(ids, ids[0], id='dropdown-roi-selection'),
        dcc.Markdown(children=markdown_text),
        dcc.Graph(id='rgb-plot-fig')
    ])

    # Interactions
    @callback(
        Output(component_id='rgb-plot-fig', component_property='figure'),
        Input(component_id='dropdown-roi-selection', component_property='value')
    )
    def update_fig(roi_id):
        with ROIRegistry(db_path=registry_path, root_path=HERE) as registry:
            roi_shape = get_shape(registry, id=roi_id)
        
        fig = get_RGB_fig(sv, roi_shape, vmin=-50., vmax=-90., padding=10, frequencies=[38, 70, 120])
        return fig

    app.run(debug=True)

if __name__ == '__main__':

    HERE = Path(__file__).resolve().parent.parent

    # Load config
    config = load_config("scripts/config.yml")

    # Execute main
    main(config)
