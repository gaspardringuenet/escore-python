from dash import html, dcc
import dash_bootstrap_components as dbc

from .layout_utils import *

GRAPH_ASPECT = 4/3


def make_left_pannel():

    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(24, 1fr)",
            "gridTemplateRows": "repeat(24, 1fr)",
            "height": "100%",
            "width": "100%",
            "gap": "10px",
            "overflow": "auto",
        },
        children=[
            html.Div(
                generate_ROI_visual_params_bar(),
                style={
                    "gridColumn": "1 / -1",
                    "gridRow": "1 / 4",
                    "background": "#ddd"
                }
            ),

            html.Div(
                dcc.Graph(id="rgb-plot-fig",
                          config={'responsive': True}),
                style={
                    "gridColumn": "1 / 23",
                    "gridRow": "4 / 14",
                    "background": "#ddd"
                }
            ),

            html.Div(
                generate_dB_slider(),
                style={
                    "gridColumn": "23 / -1",
                    "gridRow": "4 / 14",
                    "background": "#ddd",
                }
            ),

            html.Div(
                "Dist plots",
                style={
                    "gridColumn": "1 / -1",
                    "gridRow": "14 / -1",
                    "background": "#ddd"
                }
            ),

        ]
    )


def make_right_pannel():

    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(12, 1fr)",
            "gridTemplateRows": "repeat(24, 1fr)",
            "height": "100%",
            "width": "100%",
            "gap": "10px",
            "overflow": "auto",
        },
        children=[
            html.Div(
                generate_clustering_params_bar(),
                style={
                    "gridColumn": "1 / -1",
                    "gridRow": "1 / 4",
                    "background": "#ddd"
                }
            ),

            html.Div(
                dcc.Graph(id="clustering-plot-fig"),
                style={
                    "gridColumn": "1 / -1",
                    "gridRow": "4 / 13",
                    "background": "#ddd"
                }
            ),

            html.Div(
                dcc.Graph(id="echo-type-valid-fig"),
                style={
                    "gridColumn": "1 / -1",
                    "gridRow": "13 / 21",
                    "background": "#ddd"
                }
            ),

            html.Div(
                "Save Buttons",
                style={
                    "gridColumn": "1 / -1",
                    "gridRow": "21 / -1",
                    "background": "#ddd"
                }
            ),
        ]
    )


def make_layout(roi_ids, intro_text):

    return html.Div(
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(100, 1fr)",
            "gridTemplateRows": "repeat(100, 1fr)",
            "height": "100vh",
            "gap": "10px",
        },
        children=[

            # Store data in memory
            dcc.Store(id='active-channels-store', storage_type='memory', data={'values': [38., 70., 120., 200.]}),
            dcc.Store(id='labels-da-store', storage_type='memory'),


            # Main layout

            # Banner
            html.Div(
                html.H3("Echo-types Selection App"),
                style={
                    "gridColumn": "1 / -1",
                    "gridRow": "1 / 5",
                    "background": "#ddd",
                },
            ),

            # Working session info pannel
            html.Div(
                [
                    html.B("Session Params."),
                    dcc.Markdown("""..."""),
                    generate_frequencies_pannel()
                ],
                style={
                    "gridColumn": "1 / 30",
                    "gridRow": "5 / 25",
                    "background": "#ddd",
                    "minHeight": 0,
                },
            ),

            # Overflowing data table for ROI selection / Echo-types tracking
            html.Div(
                ["ROI/Echo-types table", dcc.Dropdown(roi_ids, roi_ids[0], id="dropdown-roi-selection")],
                style={
                    "gridColumn": "30 / -1",
                    "gridRow": "5 / 25",
                    "background": "#ddd",
                },
            ),

            # Left pannel containing ROI information
            html.Div(
                make_left_pannel(),
                style={
                    "gridColumn": "1 / 50",
                    "gridRow": "25 / -1",
                    "background": "#bbb",
                    "padding": "10px"
                },
            ),


            # Right pannel containing clustering and echo-type info, as well as save buttons
            html.Div(
                make_right_pannel(),
                style={
                    "gridColumn": "50 / -1",
                    "gridRow": "25 / -1",
                    "background": "#bbb",
                    "padding": "10px"
                },
            ),

        ],
    )
