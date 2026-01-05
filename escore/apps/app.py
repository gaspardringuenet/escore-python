from dash import html, Dash, callback, Input, Output, dcc
from time import sleep

app = Dash(__name__)
app.layout = html.Div(
    [
        dcc.Store(id='store-trigger'),
        html.Button("Pause for a while", id="button-pause"),
        dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="pause-nclicks", children=['0']),
        ),
        html.Button("Disabled during pause", id="button-dependent"),
        html.Div(id="dependent-nclicks", children=['0']),
        html.Button("Enabled during pause", id="button-independent"),
        html.Div(id="independent-nclicks", children=['0']),
    ]
)

# -- Pause button clicked: Disable 'pause' and 'dependent' buttons and update store
@callback(
    Output("store-trigger", "data"),
    Output("button-pause","disabled", allow_duplicate=True),
    Output("button-dependent","disabled", allow_duplicate=True),
    Input("button-pause", "n_clicks"),
    prevent_initial_call=True
)
def  start_pause(nclicks):
    return nclicks, True, True

# -- Store updated: do the long calculation. Enable 'pause' and 'dependent' button when it is finished
@callback(
    Output("pause-nclicks", "children"),
    Output("button-pause","disabled", allow_duplicate=True),
    Output("button-dependent","disabled", allow_duplicate=True),
    Input("store-trigger", "data"),
    prevent_initial_call=True
)
def  do_pause(nclicks):
    sleep(5)
    return str(nclicks), False, False

# -- 'Dependent' button clicked
@callback(
    Output("dependent-nclicks", "children"),
    Input("button-dependent", "n_clicks"),
    prevent_initial_call=True
)
def  update_dependent(nclicks):
    return str(nclicks)

# -- 'Independent' button clicked
@callback(
    Output("independent-nclicks", "children"),
    Input("button-independent", "n_clicks"),
    prevent_initial_call=True
)
def  update_independent(nclicks):
    return str(nclicks)

if __name__ == '__main__':
    app.run(debug=False,  host='0.0.0.0')
