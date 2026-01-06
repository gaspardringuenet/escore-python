from dash import html, dcc

from .layout_utils import *

GRAPH_ASPECT = 2/1


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
                "ROI Params",
                style={
                    "gridColumn": "1 / -1",
                    "gridRow": "1 / 4",
                    "background": "#ddd"
                }
            ),

            html.Div(
                "ROI Plot",
                style={
                    "gridColumn": "1 / 23",
                    "gridRow": "4 / 14",
                    "background": "#ddd"
                }
            ),

            html.Div(
                "Plot Slider",
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
                "Clustering Params",
                style={
                    "gridColumn": "1 / -1",
                    "gridRow": "1 / 4",
                    "background": "#ddd"
                }
            ),

            html.Div(
                "Clustering Plot",
                style={
                    "gridColumn": "1 / -1",
                    "gridRow": "4 / 13",
                    "background": "#ddd"
                }
            ),

            html.Div(
                "Dist Plots",
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
            html.Div(
                "Banner",
                style={
                    "gridColumn": "1 / -1",
                    "gridRow": "1 / 5",
                    "background": "#ddd",
                },
            ),

            html.Div(
                "Session info etc.",
                style={
                    "gridColumn": "1 / 30",
                    "gridRow": "5 / 25",
                    "background": "#ddd",
                    "minHeight": 0,
                },
            ),

            html.Div(
                "ROI/Echo-types table",
                style={
                    "gridColumn": "30 / -1",
                    "gridRow": "5 / 25",
                    "background": "#ddd",
                },
            ),

            html.Div(
                make_left_pannel(),
                style={
                    "gridColumn": "1 / 50",
                    "gridRow": "25 / -1",
                    "background": "#bbb",
                    "padding": "10px"
                },
            ),

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
