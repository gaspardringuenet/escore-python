from dash import Dash

from .layout import make_layout
from .callbacks import register_callbacks

def create_app(sv, registry_path, root_path, roi_ids):

    app = Dash(
        external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
        ]
    )

    intro_text = f"""
    Next things we'd like to do:
    - [x]   Be able to change the ROI with a Dropdown menu of ROIs.
    - [ ]   Show the ROI on the RGB plot.
    """

    app.layout = make_layout(roi_ids, intro_text)

    register_callbacks(app, sv, registry_path, root_path)

    return app