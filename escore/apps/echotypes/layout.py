from dash import html, dcc


def generate_ROI_visual_params_bar():
    visual_type_options = ["ROI mask in context",
                           "ROI data only"]
    
    #mask_alpha_in
    #mask_alpha_out
    #context_window_size

    return html.Div(
        id = "roi-visual-params",
        children = [
            html.Div(
                id = "param-visual-type",
                className= "visual-param",
                children = [
                    html.B("Visualization type"),
                    dcc.Dropdown(id = "dd-visual-type", options=visual_type_options, value=visual_type_options[0])
                ]
            ),
            html.Div(
                id = "param-visual-alpha",
                className= "visual-param",
                children = [
                    html.B("Mask params"),
                    html.Div(
                        children=[html.B("Alpha in"),  dcc.Input(id = "input-alpha-in", value=0.3)],
                        style = {"display": "flex", "flex-direction": "row"}
                    ),
                    html.Div(
                        children=[html.B("Alpha out"),  dcc.Input(id = "input-alpha-out", value=0.)],
                        style = {"display": "flex", "flex-direction": "row"}
                    )
                ]
            ),
        ],
        style = {"display": "flex", "flex-direction": "row"}
    )


def generate_dB_slider():
    return html.Div(
        id = "slider-db",
        className= "acou-plot-param",
        children = [
            html.B("dB thresholds"),
            dcc.RangeSlider(id = "db-slider",
                            min=-120, max=0, step=1, 
                            marks={i:f"{i} dB" for i in range(-120, 0, 10)},
                            count=1,
                            value=[-90, -50],
                            tooltip={"placement": "bottom", "always_visible": True})
        ]
    )


def generate_clustering_params_bar():
    return html.Div(
        id = "clustering-params",
        children = [
            html.Div(
                id = "param-k",
                className= "clustering-param",
                children = [
                    html.B("Number of classes"),
                    dcc.Input(id = "input-k", value=2, type='number')
                ]
            ),
            html.Div(
                id = "param-method",
                className= "clustering-param",
                children = [
                    html.B("Method"),
                    dcc.Dropdown(id = "input-method", options=["K-means"], value="K-means")
                ]
            ),
        ],
        style = {"display": "flex", "flex-direction": "row"}
    )



def make_layout(roi_ids, intro_text):
    layout = html.Div(
        id = "app-container",
        children = [
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
                                    dcc.Graph(id="rgb-plot-fig"),
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