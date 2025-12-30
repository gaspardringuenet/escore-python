from dash import html, dcc

def make_layout(roi_ids, intro_text):
    return html.Div([

        html.H1("Echo-types Selection App"),
        dcc.Markdown(intro_text),
        dcc.Dropdown(
            roi_ids,
            roi_ids[0],
            id="dropdown-roi-selection"
        ),
        dcc.Graph(id="rgb-plot-fig"),
    ])