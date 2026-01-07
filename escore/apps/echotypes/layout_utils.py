from dash import html, dcc

import dash_bootstrap_components as dbc



def generate_frequencies_pannel(
        freqs_kHz=[38., 70., 120., 200.],
        dropout_depth_dict_m={38.:800, 70.:500, 120.:300, 200.:150}, 
        min_depth_m=0, 
        max_depth_m=800, 
        ei_zstep_m=1
):

    def dropout_input(c):
        key = freqs_kHz[c]

        return dbc.InputGroup(
            [
                dbc.InputGroupText(
                    f"Channel {c} - {int(key)} kHz", class_name='input-button'),
                dbc.Input(
                    id=f"input-dropout-depth-{c}",
                    type="number",
                    value=dropout_depth_dict_m.get(key, 0),
                    min=min_depth_m,
                    max=max_depth_m,
                    step=ei_zstep_m,
                    className="input-button",
                    size='sm'
                ),
            ],
            className='mb-2'
        )
        
    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Label("Channels dropout depths (m)")] +
                        [dropout_input(c) for c in range(len(freqs_kHz))]
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Create echo-types for freqs:"),
                            dbc.Checklist(
                                id='checklist-freqs',
                                options=[{'label': f'{freq} kHz', 'value': freq} for freq in freqs_kHz],
                                value=freqs_kHz,
                            ),
                            dbc.Label("Resulting range:"),
                        ]
                    )
                ]
            )
        ),
        className="mb-3",
    )



def generate_dB_slider():
    return html.Div(
        children = [
            dcc.RangeSlider(id = "db-slider",
                            min=-120, max=0, step=1, 
                            marks={i:str(i) for i in range(-120, 0, 10)},
                            count=1,
                            value=[-90, -50],
                            vertical = True,
                            tooltip={"placement": "bottom", "always_visible": True, "template": "{value} dB"})
        ]
    )





def generate_ROI_visual_params_bar():

    visual_type_options = [
        "ROI mask in context",
        "ROI data only",
    ]

    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    # --- Visualization type ---
                    dbc.Col(
                        [
                            dbc.Label("Visualization type"),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Mode", class_name='input-button'),
                                    dbc.Select(
                                        id="dd-visual-type",
                                        options=visual_type_options,
                                        value=visual_type_options[0],
                                        class_name='input-button'
                                    )
                                ],
                                class_name="mb-2",
                            ),
                        ],
                        width=4,
                    ),

                    # --- Mask params ---
                    dbc.Col(
                        [
                            dbc.Label("Mask"),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Alpha in", className="input-button"),
                                    dbc.Input(
                                        id="input-alpha-in",
                                        type="number",
                                        value=0.3,
                                        min=0,
                                        max=1,
                                        step=0.01,
                                        className="input-button"
                                    ),
                                ],
                                className="mb-2",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Alpha out", className="input-button"),
                                    dbc.Input(
                                        id="input-alpha-out",
                                        type="number",
                                        value=0.0,
                                        min=0,
                                        max=1,
                                        step=0.01,
                                        className="input-button"
                                    ),
                                ],
                            ),
                        ],
                        width=4,
                    ),

                    # --- Context window ---
                    dbc.Col(
                        [
                            dbc.Label("Context window"),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("ESDUs", className="input-button"),
                                    dbc.Input(
                                        id="input-win-esdu",
                                        type="number",
                                        value=400,
                                        min=100,
                                        max=10_000,
                                        className="input-button"
                                    ),
                                ],
                                className="mb-2",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("Depth", className="input-button"),
                                    dbc.Input(
                                        id="input-win-depth-samples",
                                        type="number",
                                        value=300,
                                        min=100,
                                        max=700,
                                        className="input-button"
                                    ),
                                ],
                            ),
                        ],
                        width=4,
                    ),
                ],
                className="g-3",
            )
        ),
        className="mb-3",
        style={"height": "100%"}
    )



def generate_clustering_params_bar():
    return dbc.Card(
        dbc.CardBody(
            dbc.Row([
                dbc.Col(
                    [
                        dbc.Label("Clustering Parameters"),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("N clusters", className="input-button"),
                                dbc.Input(
                                    id="input-k",
                                    type="number",
                                    value=2,
                                    min=1,
                                    max=20,
                                    step=1,
                                    className="input-button"
                                )
                            ],
                            class_name='mb-2'
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Method", className="input-button"),
                                dbc.Select(
                                    id="dropdown-method",
                                    options=["KMeans", "GMM"], 
                                    value="KMeans",
                                    className="input-button"
                                )
                            ],
                            class_name='mb-2'
                        )
                    ],
                    width=4
                ),
                dbc.Col(
                    [
                        dbc.Label("Features"),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Variables", className="input-button"),
                                dbc.Select(
                                    id="dropdown-features",
                                    options=["Delta Sv", "Sv"], 
                                    value="Delta Sv",
                                    className="input-button"
                                )
                            ]
                        )
                    ],
                    width=4
                ),
                dbc.Col(
                    [
                        dbc.Label("Selected cluster"),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Cluster id", className="input-button"),
                                dbc.Select(
                                    id="dropdown-cluster-id",
                                    options=[0], 
                                    value=0,
                                    className="input-button"
                                )
                            ]
                        )
                    ],
                    width=4
                )
            ])
        ),
        className="mb-3",
        style={"height": "100%"}
    )




"""
dcc.Dropdown(
    id = "dropdown-method", 
    options=["KMeans", "GMM"], 
    value="KMeans",
    className="input-button"
)
dbc.DropdownMenu(
    children=[dbc.DropdownMenuItem("KMeans"), dbc.DropdownMenuItem("GMM")]
)
"""


def generate_clustering_params_bar_former():
    return html.Div(
        id = "clustering-params",
        children = [
            html.Div(
                id = "param-k",
                className= "clustering-param",
                children = [
                    html.B("Number of classes"),
                    dcc.Input(id = "input-k", value=2, min=1, type='number')
                ]
            ),
            html.Div(
                className= "clustering-param",
                children = [
                    html.B("Method"),
                    dcc.Dropdown(id = "dropdown-method", options=["KMeans", "GMM"], value="KMeans")
                ]
            ),
            html.Div(
                className= "clustering-param",
                children = [
                    html.B("Features"),
                    dcc.Dropdown(id = "dropdown-features", options=["Sv", "Delta Sv"], value="Delta Sv")
                ]
            ),
            html.Div(
                id = "param-freqs",
                className = "clustering-param",
                children = [
                    html.B("Frequencies"),
                    dcc.Checklist(
                        id="checklist-freqs",
                        options=[
                            {"label": "38 kHz", "value": 38},
                            {"label": "70 kHz", "value": 70},
                            {"label": "120 kHz", "value": 120},
                            {"label": "200 kHz", "value": 200},
                        ],
                        value=[38, 70],
                        inline=True
                    )
                ]
            ),
        ],
        style = {"display": "flex", "flex-direction": "row"}
    )



def generate_validation_fig_params_bar():
    return html.Div(
        id = "valid-params",
        children = [
            html.Div(
                children = [
                    html.B("Select cluster"),
                    dcc.Dropdown(id = "input-cluster-id", value=0, options=[0])
                ]
            )
        ]
    )
