import numpy as np
import pandas as pd
from scipy.stats import chi2
import xarray as xr


def make_ellipses(
    lib_ds: xr.Dataset,
    labels: np.ndarray,
    echotypes: np.ndarray,
    frequencies,
    ref_freq,
    weight_method,
    confidence
) -> pd.DataFrame:
    
    unique_classes = np.unique(labels)
    ellipses_list = []

    for class_id in unique_classes:
        class_echotypes = echotypes[labels == class_id]

        data, weights = _collect_all_observations_weighted(
            lib_ds,
            class_echotypes,
            frequencies,
            ref_freq,
            weight_method
        )

        if data.size == 0:
            print(f"Class {class_id = } empty, skipping.")

        ellipses_params = _fit_ellipse_weighted(data, weights, class_id, confidence)

        if ellipses_params is not None:
            ellipses_list.append(ellipses_params)

    return pd.DataFrame(ellipses_list)


def _collect_all_observations_weighted(
    lib_ds: xr.Dataset,
    echotypes: np.ndarray,
    frequencies: list,
    ref_freq: float,
    weight_method: str = "none"
) -> tuple[np.ndarray, np.ndarray]:
    
    all_freqs = sorted(list(frequencies) + [ref_freq])
    obs_list = []
    weights_list = []

    for e in echotypes:

        echotype_sv: xr.DataArray = (
            lib_ds
            .Sv
            .where(lib_ds.echotype == e, drop=True)
            .sel(channel=all_freqs)
            .dropna(dim="obs", how="any")
        )

        if echotype_sv.sizes["obs"] == 0:
            continue

        echotype_sv_diff = echotype_sv.sel(channel=frequencies) - echotype_sv.sel(channel=ref_freq)
        sv_diff_array = echotype_sv_diff.transpose("obs", "channel").values
        n_obs = sv_diff_array.shape[0]

        obs_list.append(sv_diff_array)

        # Compute echotype weight
        if weight_method == "none":
            w = 1.0
        elif weight_method == "sqrt":
            w = 1. / np.sqrt(n_obs)
        elif weight_method == "log":
            w = 1. / np.log1p(n_obs)
        elif weight_method == "equal":
            w = 1. / n_obs
        else:
            raise ValueError(f"Unknown method: {weight_method = }")
        
        # Same weight to all observations
        weights_list.append(np.full(n_obs, w))

    if not obs_list:
        return np.array([]).reshape(0, len(frequencies)), np.array([])
    
    data = np.vstack(obs_list)
    weights = np.concatenate(weights_list)

    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    return data, weights



def _fit_ellipse_weighted(
    data,
    weights,
    class_id,
    confidence
) -> dict:
    
    if data.shape[0] < 2:
        print(f"Class {class_id} has insufficient data: {data.shape = }")
        return None

    # Weighted mean (center)
    center = np.average(data, axis=0, weights=weights)

    # Weighted covariance
    cov = np.cov(data.T, aweights=weights)

    # Handle 1D case
    if cov.ndim == 0:
        cov = np.ndarray([cov])

    # Number of dimensions (frequencies)
    n_dims = data.shape[1]

    # Chi-squared value for confidence interval
    chi2_val = chi2.ppf(confidence, df=n_dims)

    # Eigendecomposition
    eigen_vals, eigen_vecs = np.linalg.eig(cov)
    sorted_idx = np.argsort(eigen_vals)[::-1]

    # Sort eigenvalues and eigenvectors
    eigen_vals_sorted = eigen_vals[sorted_idx]
    eigen_vecs_sorted = eigen_vecs[:, sorted_idx]
        
    # Semi-axes (sorted with largest first)
    semi_axes = np.sqrt(eigen_vals_sorted * chi2_val)

    # General data
    result = {
        "class": class_id,
        "center": center,
        "eigen_vals": eigen_vals_sorted,
        "eigen_vects": eigen_vecs_sorted,
        "semi_axes": semi_axes,
        "cov": cov,
        "n_obs": data.shape[0],
        "n_dims": n_dims,
        "chi2_scale": chi2_val
    }

    # 2D-specific parameters
    if n_dims == 2:
        major_eigenvec = eigen_vecs_sorted[:, 0].real
        angle = np.degrees(np.arctan2(major_eigenvec[1], major_eigenvec[0]))
        result.update({
            "center_x": center[0],
            "center_y": center[1],
            "semi_major": semi_axes[0],
            "semi_minor": semi_axes[1],
            "angle": angle,
            "cov_xx": cov[0, 0],
            "cov_yy": cov[1, 1],
            "cov_xy": cov[0, 1]
        })

    return result




def test():
    """Build mock data for testing the ellipses module."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Parameters
    n_classes = 5
    n_echotypes = 70
    freqs = [70., 120.]
    ref_freq = 38.
    all_freqs = sorted([ref_freq] + freqs)
    
    # Generate observations
    labels = []
    obs_list = []
    echotype_list = []
    lat_list = []
    lon_list = []
    time_list = []
    depth_list = []
    
    base_time = datetime(2017, 4, 10, 19, 52, 6)
    
    for echotype_id in range(n_echotypes):
        class_id = echotype_id % n_classes
        n_obs = np.random.randint(100, 1000)  # Fewer obs for faster testing
        
        # Gaussian distribution centered around 40 dB
        mu = 40. + 10. * np.random.random(size=len(all_freqs))
        sigma = 5. * np.random.random(size=len(all_freqs))
        sv_data = mu + sigma * np.random.randn(n_obs, len(all_freqs))
        
        labels.append(class_id)
        obs_list.append(sv_data)
        echotype_list.extend([echotype_id] * n_obs)
        lat_list.extend(np.random.uniform(-9.5, -8.5, n_obs))
        lon_list.extend(np.random.uniform(-35.0, -34.5, n_obs))
        time_list.extend([base_time + timedelta(seconds=i) for i in range(n_obs)])
        depth_list.extend(np.random.uniform(130, 200, n_obs))
    
    # Concatenate all observations
    sv_array = np.vstack(obs_list)
    total_obs = sv_array.shape[0]
    
    # Create xarray Dataset
    lib_ds = xr.Dataset(
        data_vars={
            "Sv": (["channel", "obs"], sv_array.T),
            "echotype": (["obs"], np.array(echotype_list, dtype=np.int32)),
        },
        coords={
            "channel": (["channel"], all_freqs),
            "obs": np.arange(total_obs),
            "latitude": (["obs"], np.array(lat_list)),
            "longitude": (["obs"], np.array(lon_list)),
            "time": (["obs"], pd.to_datetime(time_list).values),
            "depth": (["obs"], np.array(depth_list)),
        },
        attrs={
            "echotype_library_name": "test_lib",
            "description": "Mock echotype Sv observations",
        }
    )
    
    print(lib_ds)
    
    # Test the make_ellipses function
    all_echotypes = np.arange(n_echotypes)
    
    ellipses_df = make_ellipses(
        lib_ds=lib_ds,
        labels=labels,
        echotypes=all_echotypes,
        frequencies=freqs,
        ref_freq=ref_freq,
        weight_method="none",
        confidence=0.95
    )
    
    print("\nEllipses DataFrame:")
    print(ellipses_df)


if __name__ == "__main__":
    test()