from dash import Input, Output

from escore.registry import ROIRegistry, get_shape
from .figures import get_RGB_fig, get_Kmeans_labels_fig


def register_callbacks(app, sv, registry_path, root_path):

    @app.callback(
        Output(component_id='input-alpha-in', component_property='value'),
        Output(component_id='input-alpha-out', component_property='value'),
        Input(component_id='dd-visual-type', component_property='value')
    )
    def update_alpha_in(type):
        if type == "ROI mask in context":
            return 0.3, 0.
        if type == "ROI data only":
            return 0., 1
        


    @app.callback(
        Output(component_id='rgb-plot-fig', component_property='figure'),
        Input(component_id='dropdown-roi-selection', component_property='value'),
        Input(component_id='db-slider', component_property='value'),
        Input(component_id='input-alpha-in', component_property='value'),
        Input(component_id='input-alpha-out', component_property='value'),
        Input(component_id='dd-visual-type', component_property='value')
    )
    def update_fig(roi_id, db_range, mask_alpha_in, mask_alpha_out, type):
        with ROIRegistry(db_path=registry_path, root_path=root_path) as registry:
            roi_shape = get_shape(registry, id=roi_id)
        
        vmin, vmax = db_range

        if type == "ROI mask in context":
            window_size = (400, 300)
            padding = None
        if type == "ROI data only":
            window_size = None
            padding = 0
        
        fig = get_RGB_fig(
            sv, 
            roi_shape, 
            vmin=vmin, 
            vmax=vmax, 
            window_size=window_size,
            padding=padding, 
            frequencies=[38, 70, 120],
            show_mask=True,
            mask_alpha_in=mask_alpha_in,
            mask_alpha_out=mask_alpha_out
        )

        return fig
    

    @app.callback(
        Output(component_id='clustering-plot-fig', component_property='figure'),
        Input(component_id='dropdown-roi-selection', component_property='value'),
        Input(component_id='input-k', component_property='value'),
    )
    def update_fig(roi_id, n_clusters):
        with ROIRegistry(db_path=registry_path, root_path=root_path) as registry:
            roi_shape = get_shape(registry, id=roi_id)
        
        fig = get_Kmeans_labels_fig(
            sv, 
            roi_shape,
            frequencies=[38, 70, 120],
            n_clusters=n_clusters,
            random_state=42
        )

        return fig
