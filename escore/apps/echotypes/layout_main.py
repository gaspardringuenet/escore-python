from dash import html, dcc

from .layout_utils import *

GRAPH_ASPECT = 2/1

def make_layout(roi_ids, intro_text):
    layout = html.Div(
        id = "app-container",
        children = [
            # Store data
            #dcc.Store(id='sv-shape-store', storage_type='memory'),
            dcc.Store(id='labels-da-store', storage_type='memory'),
            # Banner
            html.Div(
                id = "banner",
                children = [html.H2("Echo-types Selection App"),]
            ),
            html.Div(
                id = "columns",
                children = [
                    # Left column
                    html.Div(
                        id = "left-column",
                        children = [
                            dcc.Markdown(intro_text),
                            dcc.Dropdown(
                                roi_ids,
                                roi_ids[0],
                                id="dropdown-roi-selection"
                            )
                        ]
                    ),
                    # Right column
                    html.Div(
                        id = "right-column",
                        children = [
                            # RGB plot of ROI
                            html.Div(
                                id = "rgb",
                                children = [
                                    html.B("RGB Plot"),
                                    generate_ROI_visual_params_bar(),
                                    html.Hr(),
                                    dcc.Graph(id="rgb-plot-fig",
                                              style={"aspect-ratio": str(GRAPH_ASPECT)}),
                                    generate_dB_slider(),
                                ]
                            ),
                            # Clustering result
                            html.Div(
                                id = "clustering",
                                children = [
                                    html.B("Clustering of ESU by r(f)"),
                                    generate_clustering_params_bar(),
                                    html.Hr(),
                                    dcc.Graph(id="clustering-plot-fig"),
                                ],
                            ),
                            # Validation
                            html.Div(
                                id = "echo-type validation",
                                children = [
                                    html.B("Echo-type validation"),
                                    generate_validation_fig_params_bar(),
                                    html.Hr(),
                                    dcc.Graph(id="echo-type-valid-fig"),
                                ],
                            )
                        ]
                    )
                ]
            ),
        ]
    )

    return layout