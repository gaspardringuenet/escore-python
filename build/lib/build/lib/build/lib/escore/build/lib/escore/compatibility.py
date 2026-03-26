import dask.array as da
from functools import wraps
import numpy as np 
from typing import Callable, Any, Literal
import xarray as xr

def multiarrays(
    features_dim_action: Literal["keep", "drop", "replace"],
    out_features_dim_size: int | None = None,
    in_features_dim_name: str | None = None,
    out_features_dim_name: str | None = None,
    output_var_name: str | None = None,
    out_features_coords: str | None = None,
    dtype: type | None = None
):
    """Handle type logic enabling func to work on numpy arrays and xarrays (Dask-backed or not)
    seemlessly.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(X, *args, **kwargs):
            if isinstance(X, xr.DataArray):
                if isinstance(X.data, da.Array):

                    # Dask-backed (map_blocks + wrap output)
                    X_transposed = features_to_last_dim(X, in_features_dim_name)
                    mapped = map_to_xarray_blocks(
                        func=func,
                        X=X_transposed,
                        features_dim_action=features_dim_action,
                        out_features_dim_size=out_features_dim_size,
                        out_dtype=dtype,
                        func_args=args,
                        func_kwargs=kwargs
                    )
                    return wrap_output(
                        mapped,
                        X_transposed,
                        features_dim_action,
                        out_features_dim_name,
                        output_var_name,
                        out_features_coords,
                    )

                else:
                    # Regular xarray (run func on .values and wrap)
                    X_transposed = features_to_last_dim(X, in_features_dim_name)
                    output = func(X_transposed.values, *args, **kwargs)
                    return wrap_output(
                        output,
                        X_transposed,
                        features_dim_action,
                        out_features_dim_name,
                        output_var_name,
                        out_features_coords,
                    )

            # Numpy array (just run func)
            elif isinstance(X, np.ndarray):
                return func(X, *args, **kwargs)
            else:
                raise TypeError(f"Unsupported type: {type(X)}")
        
        return wrapper
    return decorator



def features_to_last_dim(
    X: xr.DataArray,
    features_dim_name: str | None
) -> xr.DataArray:
    """Transpose X to get features dimension last."""

    if features_dim_name is None:
        return X

    other_dims = [d for d in X.dims if d != features_dim_name]
    return X.transpose(*other_dims, features_dim_name)



def wrap_output(
    output: np.ndarray | da.Array,
    X: xr.DataArray,
    features_dim_action: Literal["keep", "drop", "replace"],
    out_features_dim_name: str | None = None,
    output_var_name: str | None = None,
    out_features_coords: np.ndarray | None = None,
) -> xr.DataArray:
    """Wrap output array with X's coords, replacing the last dimension.
    output and X are assumed to share all dimensions but the last.
    """

    # Build new xarray dims and coords depending on actions on features dim
    match features_dim_action:
        case "keep":
            new_dim_names = X.dims
            new_coords = X.coords
        case "drop":
            new_dim_names = X.dims[:-1]
            new_coords = {k: X.coords[k] for k in new_dim_names if k in X.coords}
        case "replace":
            if out_features_dim_name is None:
                raise ValueError("Cannot create new xarray dimension because no name was given.")
            new_dim_names = X.dims[:-1] + (out_features_dim_name,)
            new_coords = {k: X.coords[k] for k in X.dims[:-1] if k in X.coords}
            if out_features_coords is not None:
                new_coords[out_features_dim_name] = out_features_coords
        case _:
            raise ValueError(f"Invalid features dimension action: {features_dim_action = }." 
                              "Must be one of ['keep', 'drop', 'replace']")
        
    # Wrap metadata around output
    output_da = xr.DataArray(
        name=output_var_name,
        data=output,
        dims=new_dim_names,
        coords=new_coords
    )

    return output_da



def map_to_xarray_blocks(
    func: Callable[[np.ndarray, Any], Any],
    X: xr.DataArray,
    features_dim_action: Literal["keep", "drop", "replace"],
    out_features_dim_size: int | None = None,
    out_dtype: type | None = None,
    func_args: tuple = (),
    func_kwargs: dict = {}
) -> da.Array:
    """Maps func to Dask-backed xarray X.
    If in_features_dim_name is not provided, it is assumed that the features dimension is the last one.
    """
    
    match features_dim_action:
        # No change to array dimensions (e.g. scalar -> scalar)
        case "keep":
            mapped = X.data.map_blocks(
                lambda x: func(x, *func_args, **func_kwargs),
                dtype=out_dtype or X.dtype
            )
            return mapped
        # Dropping last axis (e.g. classes scores -> best score)
        case "drop":
            mapped = X.data.map_blocks(
                lambda x: func(x, *func_args, **func_kwargs),
                drop_axis=-1,    # drop in features axis (the last axis)
                dtype=out_dtype or X.dtype
            )
            return mapped
        # Drop last axis + build new axis with unit chunks
        case "replace":
            if (out_features_dim_size is None) or (out_features_dim_size == 0):
                raise ValueError("Dask mapping can only replace features dim by one with size > 0." 
                                 "Use features_dim_action='drop' to drop features dimension")
            
            output_chunks = X.chunks[:-1] + ((out_features_dim_size,),)
            n_axes = X.ndim

            mapped = X.data.map_blocks(
                lambda x: func(x, *func_args, **func_kwargs),
                drop_axis=-1,   # drop in features axis
                new_axis=n_axes-1,    # add out features axis in last position
                chunks=output_chunks,  # new chunking similar to X + unit for out features
                dtype=out_dtype or X.dtype
            )
            return mapped
        case _:
            raise ValueError(f"Invalid features dimension action: {features_dim_action = }." 
                             "Must be one of ['keep', 'drop', 'replace']")