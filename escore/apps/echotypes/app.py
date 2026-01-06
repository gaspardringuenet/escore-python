from pathlib import Path
from dash import Dash

#from .layout_main import make_layout
from .layout_main_test import make_layout
from .callbacks import register_callbacks


def create_app(sv, registry_path, root_path, roi_ids):

    here = Path(__file__).parent

    app = Dash(
        assets_folder = str(here / "assets"),
        external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"]
    )

    intro_text = f"""
    This app is designed for the interactive selection of *echo-types*' pixels in pre-selected *regions of interest* (ROI).
    """

    app.layout = make_layout(roi_ids, intro_text)

    #register_callbacks(app, sv, registry_path, root_path)

    return app


#external_stylesheets=[
#    str(here / "assets/style.css"),
#    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
#]