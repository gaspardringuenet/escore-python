from dash import Input, Output, State

import numpy as np
import xarray as xr

from escore.registry import ROIRegistry, get_shape
from .processing import get_roi_Sv, cluster_roi
from .figures import get_RGB_fig, get_clustering_labels_fig, echotype_deltaSv_histograms
from .layout_main import GRAPH_ASPECT


def register_callbacks(app, sv, registry_path, root_path):

    @app.callback(
        Output(component_id='input-alpha-in', component_property='value'),
        Output(component_id='input-alpha-out', component_property='value'),
        Output(component_id='input-win-esdu', component_property='value'),
        Output(component_id='input-win-esdu', component_property='disabled'),
        Output(component_id='input-win-depth-samples', component_property='value'),
        Output(component_id='input-win-depth-samples', component_property='disabled'),
        Input(component_id='dd-visual-type', component_property='value'),
        #Input(component_id='input-win-esdu', component_property='value'),
        #Input(component_id='input-win-depth-samples', component_property='value'),
    )
    def update_visual_params(type):
        if type == "ROI mask in context":
            return 0., 0.5, 600, False, 300, False
        if type == "ROI data only":
            return 0., 1,  600, True, 300, True
        

    @app.callback(
        Output(component_id='rgb-plot-fig', component_property='figure'),
        Input(component_id='dropdown-roi-selection', component_property='value'),
        Input(component_id='db-slider', component_property='value'),
        Input(component_id='input-alpha-in', component_property='value'),
        Input(component_id='input-alpha-out', component_property='value'),
        Input(component_id='dd-visual-type', component_property='value'),
        Input(component_id='input-win-esdu', component_property='value'),
        Input(component_id='input-win-depth-samples', component_property='value'),
    )
    def update_fig(roi_id, db_range, mask_alpha_in, mask_alpha_out, type, win_esdu, win_depth):
        with ROIRegistry(db_path=registry_path, root_path=root_path) as registry:
            roi_shape = get_shape(registry, id=roi_id)
        
        vmin, vmax = db_range

        if type == "ROI mask in context":
            window_size = win_esdu, win_depth
            padding = None
        if type == "ROI data only":
            window_size = None
            padding = 0
        
        fig, (w, h) = get_RGB_fig(
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

        # Make sur the image is properly stretched to the dcc.Graph aspect ratio
        fig.update_layout({
            'paper_bgcolor': 'yellow',
            'plot_bgcolor': 'pink',
            'margin': {"b": 0, "t": 0, "l": 0, "r": 0},
            'xaxis': {
                'scaleanchor': 'y',
                'scaleratio': h/w * GRAPH_ASPECT,
                'constrain': 'domain'
            }
        })

        return fig
    


    # Update the clustering figure based on current ROI and clusterings parameters
    @app.callback(
        Output('clustering-plot-fig', 'figure'),
        Output('labels-da-store', 'data'),
        Input('dropdown-roi-selection', 'value'),
        Input('input-k', 'value'),
        Input('checklist-freqs', 'value'),
        Input('dropdown-method', 'value'),
        Input('dropdown-features', 'value'),
    )
    def update_fig(roi_id, n_clusters, frequencies, method, features):
        with ROIRegistry(db_path=registry_path, root_path=root_path) as registry:
            roi_shape = get_shape(registry, id=roi_id)

        # Get ROI Sv data from bbox sv and shape
        roi_sv = get_roi_Sv(sv, roi_shape, frequencies)

        # Perform clustering and fetch labels as DataArray
        labels_da, model = cluster_roi(
            roi_sv, 
            features, 
            method, 
            n_clusters,
            ref_frequency=38, 
            random_state=42
        )
        
        # Create figure
        fig = get_clustering_labels_fig(labels_da)

        # Serialize labels to store in memory
        payload = {
            "values": labels_da.values.tolist(),
            "shape": labels_da.shape
        }
        #labels_da_dict = labels_da.to_dict()

        return fig, payload
    

    # Create an histogram of Sv38 values for the selected cluster, given an ROI and clustering parameters.
    @app.callback(
        Output('echo-type-valid-fig', 'figure'),
        Input('dropdown-roi-selection', 'value'),
        Input('labels-da-store', 'data'),
        Input('input-cluster-id', 'value'),
        Input('checklist-freqs', 'value'),
    )
    def update_fig(roi_id, labels_payload, cluster_id, frequencies):
        with ROIRegistry(db_path=registry_path, root_path=root_path) as registry:
            roi_shape = get_shape(registry, id=roi_id)

        # Get ROI Sv data from bbox sv and shape
        roi_sv = get_roi_Sv(sv, roi_shape, frequencies)

        # Load labels as DataArray from dcc.Store()
        values = np.array(labels_payload["values"]).reshape(labels_payload["shape"])
        labels_da = xr.DataArray(values, dims=['time', 'depth'])

        fig = echotype_deltaSv_histograms(roi_sv, labels_da, cluster_id)

        return fig
    

    # Limit the options for cluster selection to [0, K-1] where K is the number of clusters
    @app.callback(
        Output('input-cluster-id', 'options'),
        Input('input-k', 'value')
    )
    def update_cluster_options(k):
        return [_ for _ in range(k)]
    

    # Update the cluster id base on click on the cluster image
    @app.callback(
        Output('input-cluster-id', 'value'),
        Input('clustering-plot-fig', 'clickData'),
        prevent_initial_call=True,
    )
    def cluster_id_on_click(clickData):
        point= clickData["points"][0]
        cluster_id = point["z"]

        return cluster_id