"""Build feature vector for each echotype."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple, Sequence
import xarray as xr
from scipy.interpolate import make_smoothing_spline, BSpline
from sklearn.decomposition import PCA


def extract_mean_sv_diff(
    lib_ds: xr.Dataset, 
    frequencies: List[float], 
    ref_freq: float
) -> pd.DataFrame:
    """Compute mean Sv - Sv(ref_freq) for each echotype in the library.

    Args:
        lib_ds (xr.Dataset): Echotype library. Typically an 'EchotypesApp' output.
            Minimal required schema:
            Dimensions:     (obs, echotype, channel)
            Coordinates:
                echotype    (obs) int
                channel     (channel) float
            Data variables:
                Sv          (channel, obs) float
        frequencies (List[float]): List of frequencies for which to compute a value.
        ref_freq (float): Reference frequency channel to subtract.

    Returns:
        pd.DataFrame: Dataframe with index 'echotype' and one column for each frequency f in
            frequencies containing the mean Sv(f)-Sv(ref_freq) value for the echotype. Also
            include a 'n_obs' column representing the count of observations in the echotype.
            
    """

    all_freqs = sorted(frequencies+[ref_freq])

    echotypes = np.unique(lib_ds.echotype.values)
    features_list = []

    # Loop through the library
    for e in echotypes:

        # Select echotype Sv data
        echotype_sv: xr.DataArray = _get_echotype_sv(lib_ds, e, all_freqs)

        # Ignore non valid echotypes
        n_obs = echotype_sv.sizes["obs"]
        if n_obs == 0:
            print(f"Skipped echotype {e} containing no valid obs for channels {all_freqs}")
            continue

        # Compute Sv difference
        echotype_sv_diff = echotype_sv.sel(channel=frequencies) - echotype_sv.sel(channel=ref_freq)

        # Compute echotype features
        features = {
            "echotype": e,
            **{
                f"mean_SvDiff_{int(c)}-{int(ref_freq)}": float(echotype_sv_diff.sel(channel=c).mean()) 
                for c in frequencies
            },
            "n_obs": n_obs
        }

        # Add to list
        features_list.append(features)

    return pd.DataFrame(features_list).set_index('echotype')


class FDAExtractor:

    def __init__(self, lib_ds: xr.Dataset, frequencies: Sequence[float], ref_freq: float):

        # Data and base parameters
        self.lib_ds = lib_ds
        self.frequencies = frequencies
        self.ref_freq = ref_freq

        # B-splines output and logging
        self.splines_results: dict | None = None
        self.splines_log: dict | None = None

        # PCA
        self.pca: PCA | None = None
        self.pc_score: np.ndarray | None = None

        # Final features matrix
        self.features_df: pd.DataFrame | None = None


    def fit_splines(
        self,
        smoothing_lambda: float = 1.,
        min_n_obs: int = 0,
        n_bins: int = 30,
        pdf_range: Tuple[float, float] = (-50., 50.)
    ) -> dict:
        
        # fit B-splines on Sv diff distributions using heler
        self.splines_results, self.splines_log = _fit_splines(
            lib_ds=self.lib_ds,
            frequencies=self.frequencies,
            ref_freq=self.ref_freq,
            smoothing_lambda=smoothing_lambda,
            min_n_obs=min_n_obs,
            n_bins=n_bins,
            pdf_range=pdf_range
        )

        # clear downstream data
        self.pca, self.pc_score = None, None
        self.features_df = None

        return self.splines_results
    

    def fit_pca(self, n_components: int = 3) -> PCA:
        
        # raise error if no upstream data
        if self.splines_results is None:
            raise ValueError("Fit B-splines first.")
        
        # fetch B-splines coefficient - shape (n_echotypes, n_coefs)
        splines_coefs_list = self.splines_results["spline_coefs"]
        spline_coeffs_array = np.array(splines_coefs_list)

        # fit PCA, transform data and store
        self.pca: PCA = PCA(n_components=min(n_components, spline_coeffs_array.shape[1] - 1))
        self.pc_score = self.pca.fit_transform(spline_coeffs_array)

        # clear downstream data
        self.features_df = None

        return self.pca
    

    def build_features(self) -> pd.DataFrame:
        
        # raise errors if no upstream data
        if self.splines_results is None:
            raise ValueError("Fit B-splines first.")
        if self.pc_score is None:
            raise ValueError("Fit PCA first.")

        # loop through echotypes and gather PCs and n_obs
        echotypes_list = self.splines_results["echotypes"]
        features_list = []

        for idx, e in enumerate(echotypes_list):
            features = {
                "echotype": e,
                **{
                    f"PC{i+1}": float(self.pc_score[idx, i])
                    for i in range(self.pc_score.shape[1])
                }
            }
            features_list.append(features)

        # return features matrix as dataframe
        return pd.DataFrame(features_list).set_index("echotype")
    

    def fit_and_build_features(
        self,
        n_components: int = 2,
        smoothing_lambda: float = 1.,
        min_n_obs: int = 0,
        n_bins: int = 30,
        pdf_range: Tuple[float, float] = (-50., 50.)
    ) -> pd.DataFrame:
        
        print("Step 1 - Fit B-splines on Sv difference distributions per echotype")
        _ = self.fit_splines(
            smoothing_lambda,
            min_n_obs,
            n_bins,
            pdf_range
        )

        print("Step 2 - Global PCA on concatenated spline coefficient")
        _ = self.fit_pca(n_components)

        print("Step 3 - Build output dataframe")
        return self.build_features()
    
    
    def plot_spline_diagnostics(
        self,
        echotype: int,
        figsize: Tuple[int, int] = (15, 4),
        ncols: int = 2,
        layout: str = "tight",
        include_pca_reconstruction: bool = False
    ) -> None:
        if self.splines_log is None:
            raise ValueError("Fit splines first.")
        if echotype not in self.splines_results["echotypes"]:
            raise ValueError(f"Echotype {echotype} was skipped during fitting.")
        if include_pca_reconstruction and self.pca is None:
            raise ValueError(f"Fit PCA to display PCA reconstruction.")
        
        all_freqs = sorted(list(self.frequencies + [self.ref_freq]))

        # Get echotype data
        echotype_sv: xr.DataArray = _get_echotype_sv(self.lib_ds, echotype, all_freqs)
        echotype_sv_diff = echotype_sv.sel(channel=self.frequencies) - echotype_sv.sel(channel=self.ref_freq)
        sv_diff_array = echotype_sv_diff.transpose("channel", "obs").values

        # Get echotype index (useful to catch spline evaluation or pca data)
        echotype_idx = self.splines_log["echotype_to_valid_idx"][echotype]

        # Get stored B-splines
        spl = self.splines_log["splines"]
        splines_params = self.splines_log["params"]

        # Pre-compute values
        x_spline = np.linspace(*splines_params["pdf_range"], 200)
        y_spline = spl(x_spline)[:, :, echotype_idx]  # shape (len(x_spline), n_frequencies)

        # Get PCA reconstruction if required
        if include_pca_reconstruction:

            # Get reconstructed coefficient from PCA
            pc_scores = self.pc_score[echotype_idx:echotype_idx+1]          # shape (1, n_components)
            pca_inverse = self.pca.inverse_transform(pc_scores)
            reconstructed_coefs = self.pca.inverse_transform(pc_scores)[0]  # shape (n_freqs * n_coefs_per_freq)

            # Compute n_coefs_per_freq
            n_coefs = len(reconstructed_coefs)
            n_freqs = len(self.frequencies) 
            if n_coefs % n_freqs != 0:
                raise ValueError(f"Number of spline coefficient should be a multiple of target frequencies. {n_coefs = }, {len(self.frequencies) = }")
            n_coefs_per_freq = n_coefs // len(self.frequencies)

        else:
            reconstructed_coefs = None

        # Create figure (one subplot per frequency)
        n_freqs = len(self.frequencies)
        nrows = n_freqs // ncols + (n_freqs % ncols > 0)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout=layout)

        # Flatten axes to 1D for easier iteration
        axes = np.atleast_1d(axes).flatten()
        
        for freq_idx, (freq, ax) in enumerate(zip(self.frequencies, axes)):

            # Compute and plot histogram
            dist, bin_edges = np.histogram(
                a=sv_diff_array[freq_idx],
                bins=splines_params["n_bins"],
                range=splines_params["pdf_range"],
                density=True
            )
            ax.bar(bin_edges[:-1], dist, width=np.diff(bin_edges), align="edge", alpha=0.5, label="Data histogram", edgecolor="black")

            # Plot fitted spline
            ax.plot(x_spline, y_spline[:, freq_idx], 'r-', linewidth=2, label="B-spline fit")

            # PLot PCA-reconstructed spline if requested
            if include_pca_reconstruction:

                # Reconstruct spline for this frequency
                coef_start = freq_idx * n_coefs_per_freq     # Assumes all freqs have the same number of spline coefs (correct)
                coef_end = coef_start + n_coefs_per_freq

                t, c, k = spl.t, reconstructed_coefs[coef_start:coef_end], spl.k
                reconstructed_spl: BSpline = BSpline(t, c, k)
                
                # Compute values and plot
                y_reconstructed = reconstructed_spl(x_spline)
                ax.plot(x_spline, y_reconstructed, 'g--', linewidth=2, label=f"PCA reconstruction ({self.pca.n_components_} PCs)")

            # Set axes properties
            ax.set_xlabel(fr'$\Delta$Sv {freq:.0f}-{self.ref_freq:.0f} kHz [dB]')
            ax.set_ylabel("Probability density")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Clean unused axes
        for i in range(n_freqs, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f"Spline fit diagnostics\nEchotype {echotype} in library {self.lib_ds.attrs.get('echotype_library_name')} (n_obs={len(echotype_sv.obs)}) ")


def _fit_splines(
    lib_ds: xr.Dataset,
    frequencies: List[float],
    ref_freq: float,
    smoothing_lambda: float = 1.,
    min_n_obs: int = 0,
    n_bins: int = 30,
    pdf_range: Tuple[float, float] = (-50., 50.)
) -> Tuple[dict, dict]:
    
    args = locals()
    all_freqs = sorted(frequencies + [ref_freq])
    echotypes = np.unique(lib_ds.echotype.values)
    freq_array = np.array(frequencies)

    # Collect histograms for all echotypes and frequencies
    print("Step 0 - Collect probability density distributions")
    echotype_histograms = {}  # echotype -> (n_frequencies, n_bins)
    valid_echotypes = []
    n_obs_dict = {}

    for e in echotypes:
        echotype_sv = _get_echotype_sv(lib_ds, e, all_freqs)
        n_obs = echotype_sv.sizes["obs"]
        
        if n_obs < min_n_obs:
            print(f"Skipped echotype {e} containing too few valid obs. {n_obs = }.")
            continue

        echotype_sv_diff = echotype_sv.sel(channel=frequencies) - echotype_sv.sel(channel=ref_freq)
        sv_diff_array = echotype_sv_diff.transpose("channel", "obs").values
        
        # Compute histograms for all frequencies
        histograms = []
        for c in range(len(freq_array)):
            dist, bin_edges = np.histogram(
                a=sv_diff_array[c],
                bins=n_bins,
                range=pdf_range,
                density=True
            )
            histograms.append(dist)
        
        echotype_histograms[e] = np.array(histograms)  # shape (n_frequencies, n_bins)
        valid_echotypes.append(int(e))
        n_obs_dict[e] = n_obs

    # Step 1 - Fit B-splines with shared basis using batched operation
    print("Step 1 - Fit B-splines with shared basis")

    # Prepare y data for batch smoothing - stack all echotypes
    y_batch = np.stack([echotype_histograms[e] for e in valid_echotypes], axis=0)  # shape (n_valid_echotypes, n_frequencies, n_bins)
    y_batch = y_batch.T # reshape to (n_bins, n_frequencies, n_valid_echotypes)

    # Compute bin centers (same for all)
    _, bin_edges = np.histogram(np.array([0]), bins=n_bins, range=pdf_range, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
    # Fit all splines at once - interpolate on the first dimension (histogram bins)
    spl = make_smoothing_spline(bin_centers, y_batch, lam=smoothing_lambda)
        
    # Extract coefficients - shape (n_coefs, n_frequencies, n_valid_echotypes)
    spline_coefs_dict = {}
    for i, e in enumerate(valid_echotypes):
        # Flatten coefficients for this echotype across all frequencies
        coefs = spl.c[:, :, i].T # shape (n_frequencies, n_coefs)
        spline_coefs_dict[e] = coefs.flatten().tolist()  # length n_coefs * n_frequencies

    # Build results dict
    results = {
        "echotypes": valid_echotypes,
        "spline_coefs": [spline_coefs_dict[e] for e in valid_echotypes],
        "n_obs": [n_obs_dict[e] for e in valid_echotypes],
    }

    # Build log dict
    log = {
        "params": args,
        "splines": spl,
        "echotype_to_valid_idx": {e: i for i, e in enumerate(valid_echotypes)},
    }

    return results, log




def _get_echotype_sv(lib_ds: xr.Dataset, e: int, channels: Sequence[float]) -> xr.DataArray:

    return (
        lib_ds
        .Sv
        .where(lib_ds.echotype == e, drop=True)
        .sel(channel=channels)
        .dropna(dim="obs", how="any")
    )