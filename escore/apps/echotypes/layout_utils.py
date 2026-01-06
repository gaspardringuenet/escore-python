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
            html.Div(
                id = "param-visual-window",
                className= "visual-param",
                children = [
                    html.B("Context window"),
                    html.Div(
                        children=[html.B("ESDUs"),  dcc.Input(id = "input-win-esdu", value=400, min=100, max=10_000, type="number")],
                        style = {"display": "flex", "flex-direction": "row"}
                    ),
                    html.Div(
                        children=[html.B("Depth samples"),  dcc.Input(id = "input-win-depth-samples", value=300, min=100, max=700, type="number")],
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

