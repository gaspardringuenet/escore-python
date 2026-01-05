import dash
from dash import dcc, html, callback, Input, Output
import plotly.graph_objects as go
import plotly.express as px

import numpy as np
from pathlib import Path
from PIL import Image

app = dash.Dash(__name__)

# Load image
here = Path(__file__).parent.parent.parent.parent
filepath = here/"data/echogram_images/RGB_38_70_120kHz_TF10000_Z0--1_Sv-90--50dB/amazomix_3pings1m/amazomix_3pings1m_T300000-310000_Z0--1_Sv-90--50.png"
img = Image.open(filepath)
img = np.asarray(img)

# Figure
#fig = go.Figure(
#    go.Image(z=img, aspect=3)
#)

h, w, c = img.shape
ratio = 1/10
h_px, w_px = 600, 600 // ratio

fig = px.imshow(img)

fig = fig.update_layout({
    'paper_bgcolor': 'yellow',
    'plot_bgcolor': 'pink',
    'margin': {"b": 0, "t": 10, "l": 0, "r": 10},
    'yaxis': {
        'scaleanchor': 'x',
        'scaleratio': w/h * ratio,
        'constrain': 'domain'
    }
})


app.layout = html.Div([
    dcc.Graph(
        id="graph",
        figure=fig,
        style={"width": f"{w_px}px", "height": f"{h_px}px"},
        config={"responsive": True},
    )
])


""" @app.callback(
    Output("aspect-output", "children"),
    Input("graph", "relayoutData")
)
def report_aspect(relayoutData):
    if relayoutData is None:
        return "Aspect: unknown"
    
    # relayoutData may contain width and height of the plot area
    width = relayoutData.get("width", None)
    height = relayoutData.get("height", None)
    
    if width is None or height is None:
        return "Aspect: unknown"
    
    aspect = width / height
    return f"Aspect ratio: {aspect:.3f}" """


if __name__ == "__main__":
    app.run(debug=True)
