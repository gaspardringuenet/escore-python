from dash import Input, Output

from escore.registry import ROIRegistry, get_shape
from .figures import get_RGB_fig


def register_callbacks(app, sv, registry_path, root_path):
    @app.callback(
        Output(component_id='rgb-plot-fig', component_property='figure'),
        Input(component_id='dropdown-roi-selection', component_property='value')
    )
    def update_fig(roi_id):
        with ROIRegistry(db_path=registry_path, root_path=root_path) as registry:
            roi_shape = get_shape(registry, id=roi_id)
        
        fig = get_RGB_fig(sv, roi_shape, vmin=-50., vmax=-90., padding=10, frequencies=[38, 70, 120])
        return fig