import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Tuple

import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.xarray  # noqa
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from echoregions.regions2d import Regions2D
from sklearn.pipeline import Pipeline

from escore.echotypes.paths import ResultsPathManager
from escore.echotypes.utils import _select_bbox_data, _select_region_row
from escore.utils.plot import plot_channels, plot_rgb
from escore.utils.sklearn import classes_to_segments, stack_for_sklearn, unstack_sklearn_preds


class EchotypeWorkflow:
    def __init__(
        self,
        ds_MVBS: xr.Dataset,
        regions: Regions2D,
        results_dir: str,
        acoustic_var: str = "Sv",
    ):

        # Paths
        self.paths = ResultsPathManager(results_dir)

        # Visualization toolkit
        self.plot = WorkflowDataVisualizer(self)

        # Data
        self.data = ds_MVBS
        self.acoustic_var = acoustic_var
        self.regions = regions

        # Current region_id
        self.current_: CurrentRegion | None = None

    #### Main worflow methods ####

    def status_report(self):
        """Loads workflow state file and prints status information including:
        -   the amount of processed regions (by completion status)
        -   the last loaded region
        """
        state_dict = self._load_state()
        region_status_dict = state_dict["status"]

        # Compute stats about the workflow
        total, n_pending, n_completed, n_rejected = (0,) * 4
        for _, status in region_status_dict.items():
            total += 1
            if status == "pending":
                n_pending += 1
            if status == "completed":
                n_completed += 1
            if status == "rejected":
                n_rejected += 1
        n_processed = n_completed + n_rejected

        # Print status report
        print(
            "Workflow status reports:\n"
            f"Out of {total} regions, {n_processed} ({n_processed / total:.1%}) have been processed.\n"
            f" * {n_completed} marked as completed (echotypes have been extracted)\n"
            f" * {n_rejected} marked as rejected (skipped without extracting)\n"
            f" * {n_pending} remain pending\n"
            f"Last loaded region id: {state_dict['last']}"
        )

    # Step 1 - Initialize by setting current region to process

    def set_current(
        self,
        region_id: int,
    ) -> None:
        """Change the current region / echotype being processed. Load processing data if
        there is some.

        Parameters
        ----------
        region_id : int
            Id of the region in the Regions2D annotation object.
        """

        # TODO: Update "last" field in state dict.

        if region_id not in self.regions.data["region_id"].values:
            raise ValueError(
                f"Unkown {region_id = }. "
                "(region_id must reference the region_id attribute of the regions.data DataFrame)"
            )

        # Display message if a current id already exists
        if self.current_ is not None:
            if self.current_.region_id != region_id:
                print(
                    f"Switching current id from {self.current_.region_id} to {region_id}. "
                    "Unsaved changes for the previous echotype are dropped. "
                )
            else:
                print(f"Current already set to {self.current_.region_id}. Reloading recipe...")

        # Create new current instance, pre-load data subset and set edit mode
        self.current_ = CurrentRegion(region_id, self._get_region_data(region_id))

        # Load recipe, pipeline, and results (if they exist)
        self._load_current()

        # Display warning about the status of region
        state_dict = self._load_state()
        status: str = state_dict["status"][region_id]
        print(f"\nWARNING: Current region has status {status.upper()}.")

        # Update workflow state
        state_dict["last"] = region_id
        self._dump_state(state_dict)

        print("Ready for processing!")

    def reset_current(self):
        """Resets current echotype state as blank. Only region_id and region_data
        fields are preserved.

        Raises
        ------
        ValueError
            If no current region has been set.
        """

        if self.current_ is None:
            raise ValueError("No current region in process (use .set_current)")

        print("Resetting current echotype state as blank...")
        self.current_.segmenter = None
        self.current_.segments = None
        self.current_.segment_id = None
        self.current_.echotype_data = None
        print("Ready for processing!")

    # Step 2 - Set segmentation pipeline & perform segmentation

    def set_segmenter(self, segmenter: Pipeline):
        """Set the segmentation pipeline for echotype extraction.
        Pipeline ingests np.ndarray's in shape (n_samples, n_channels).
        Segmenter is stored as .segmenter field in `current_`.

        Parameters
        ----------
        segmenter : Pipeline
            scikit-learn Pipeline for echotype extraction.

        Raises
        ------
        ValueError
            If no current region has been set.
        ValueError
            If segmenter is not a scikit-learn Pipeline.
        """

        # Checks
        if self.current_ is None:
            raise ValueError("No current region in process (use .set_current)")
        if not isinstance(segmenter, Pipeline):
            raise ValueError("Segmenter must be a sklearn Pipeline")
        # Set
        self.current_.segmenter = segmenter

    def segment(self) -> xr.DataArray:
        """Run segmentation pipeline on current region data.

        Returns
        -------
        xr.DataArray
            Region data split with an additional segment dimension.
            For each segment value, contains the region data where the
            semgmentation mask for this segment is True.

        Raises
        ------
        ValueError
            If no current region has been set.
        ValueError
            If no segmenter has been set.
        """

        # Checks
        if self.current_ is None:
            raise ValueError("No current region in process (use .set_current)")
        if self.current_.segmenter is None:
            raise ValueError("No segmenter (use .set_segmenter)")
        if self.current_.region_data is None:
            self.current_.region_data = self._get_region_data(self.current_.region_id)

        # Preprocessing
        da_Sv = self.current_.region_data
        da_stacked = stack_for_sklearn(da_Sv, drop_na=True)
        X = da_stacked.values

        # Fit model & predict
        preds: np.ndarray = self.current_.segmenter.fit_predict(X)  # type: ignore

        # Formatting predictions
        da_preds = unstack_sklearn_preds(preds, da_stacked)
        da_segments = classes_to_segments(da_Sv, da_preds)

        # Save to CurrentRegion
        self.current_.segments = da_segments

        return da_segments

    def select_segment(self, segment_id: int) -> xr.DataArray | None:
        """Select echotype data by indicating the segment id to chose.
        Select a segment from segmented data, modify current echotype
        data and metadata, and return echotype DataArray.

        Parameters
        ----------
        segment_id : int
            id of the segment corresponding to the desired echotype.

        Returns
        -------
        xr.DataArray
            echotype DataArray (time, depth, channel) dimensions.
        """
        # Get segment from segmented DataArray
        try:
            segment = self._get_segment(segment_id)
        except ValueError as e:
            print(f"Unable to fetch segment {segment_id} - {e}")
            return None

        # Save segment_id & echotype_data to CurrentRegion
        print(f"Using segment {segment_id} as echotype data.")
        self.current_.segment_id = segment_id  # type: ignore (check already performed by _get_segment)
        self.current_.echotype_data = segment  # type: ignore

        return segment

    # Step 3 - Save results

    def save_current(self, overwrite: bool = False):
        """Save echotype data and recipe in results directory.
        Automatically change the region's status to 'completed'.

        Parameters
        ----------
        overwrite : bool, optional
            Allow overwriting an existing echotype, by default False

        Raises
        ------
        ValueError
            If no current region has been set.
        ValueError
            If processing is incomplete (None fields remaining in the CurrentRegion object).
        ValueError
            If user attemps to save an already saved echotype with overwrite = False.
        """

        if self.current_ is None:
            raise ValueError("No current region to dump data from.")

        if any(asdict(self.current_).values()) is None:
            raise ValueError("Cannot dump incomplete echotype.")

        region_id = self.current_.region_id
        dir = self.paths.region_results(region_id)
        failsafe_dir = dir.parent / "failsafe"

        # If the region's results directory exists: check for overwriting permission
        if dir.exists():
            if not overwrite:
                raise ValueError(f"Results dir already exists and {overwrite = }.")

            # Move existing results to the failsafe directory
            shutil.move(dir, failsafe_dir)

        # Create region's results directory
        dir.mkdir(parents=True, exist_ok=True)

        # Select paths
        segmenter_path = self.paths.segmenter(region_id)
        recipe_path = self.paths.recipe(region_id)
        echotype_path = self.paths.echotype_data(region_id)

        # Format recipe
        recipe: dict = {
            "segmenter": str(self.current_.segmenter),
            "segment_id": self.current_.segment_id,
        }

        # Dump from CurrentRegion
        try:
            joblib.dump(self.current_.segmenter, segmenter_path)
            yaml.safe_dump(recipe, open(recipe_path, "w"))
            self.current_.echotype_data.to_netcdf(echotype_path, engine="netcdf4")  # type: ignore
        except Exception as e:
            print(f"Dumping echotype failed: {e}")
            if overwrite:
                print("Going back to previous version of results directory")
                shutil.move(failsafe_dir / dir.name, dir.parent)
            else:
                print("Removing results dir")
                shutil.rmtree(dir)
        finally:
            if failsafe_dir.exists():
                shutil.rmtree(failsafe_dir)
            self._mark_completed()
            print("Echotype data & recipe saved successfully!")

    # Step 4 - Mark the region as processed ("completed" or "rejected")

    def _mark_completed(self):
        """Set current region's status as completed. Called by save_current.
        Should not be called by user directly."""

        if self.current_ is None:
            raise ValueError("No current region to dump data from.")

        state_dict = self._load_state()
        state_dict["status"][self.current_.region_id] = "completed"
        self._dump_state(state_dict)

        print(f"Region {self.current_.region_id} worflow status updated to 'completed'.")

    def mark_rejected(self):
        """Set current region's status as rejected in workflow state
        file."""

        if self.current_ is None:
            raise ValueError("No current region to dump data from.")

        state_dict = self._load_state()
        state_dict["status"][self.current_.region_id] = "rejected"
        self._dump_state(state_dict)

        print(f"Region {self.current_.region_id} worflow status updated to 'rejected'.")

    # Bonus Step - Get Echotype Data for direct inspection

    def get_echotype_dataarray(self) -> xr.DataArray:
        """Return selected echotype DataArray if is exists.

        Returns
        -------
        xr.DataArray
            Echotype DataArray.

        Raises
        ------
        ValueError
            If no current region has been set.
        ValueError
            If no echotype data is available.
        """
        # Checks
        if self.current_ is None:
            raise ValueError("No current region in process (use .set_current)")
        if self.current_.echotype_data is None:
            raise ValueError("No echotype data (use .segment & .select_segment)")

        return self.current_.echotype_data

    def get_echotype_dataframe(
        self,
        use_frequency_var: xr.DataArray | None = None,
    ) -> pd.DataFrame:
        """Return selected echotype as DataFrame if is exists."""

        da_echotype = self.get_echotype_dataarray()

        if use_frequency_var is not None:
            da_echotype = da_echotype.assign_coords(channel=use_frequency_var)

        df_echotype = (
            da_echotype.to_dataframe(
                name="Sv",
                dim_order=["depth", "ping_time", "channel"],
            )
            .reset_index()
            .pivot_table(
                index=["depth", "ping_time"],
                columns="channel",
                values="Sv",
            )
            .reset_index(drop=True)
        )

        return df_echotype

    #### Workflow I/O utilities ####

    def _dump_state(self, state_dict: dict):
        path = self.paths.workflow_state
        yaml.safe_dump(state_dict, open(path, "w"))

    def _init_state(self) -> dict:
        """Initialize the state dict and save it as yaml file.

        Returns
        -------
        dict
            Worflow state dict
        """
        state_dict = {
            "last": None,
            "status": {r_id: "pending" for r_id in self.regions.data["region_id"]},
        }
        self._dump_state(state_dict)
        return state_dict

    def _load_state(self) -> dict:
        """Load and return the state file as dict. If no file exists, call _init_state.

        Returns
        -------
        dict
            Worflow state dict
        """
        path = self.paths.workflow_state
        if path.is_file():
            return yaml.safe_load(open(path, "r"))
        else:
            return self._init_state()

    def _load_current(self) -> None:
        """Load a region / echotype processing data. Update CurrentRegion object.
        Directory state assumption: existence of a region's result folder is equivalent
        to this folder containing all necessary files (This assumption is enforced by
        dumping all files at once and deleting the directory in case of failure -
        see .dump_current).

        Raises
        ------
        ValueError
            If worklow does not contain a CurrentRegion object.
        """

        if self.current_ is None:
            raise ValueError("No current region to load data on.")
        region_id = self.current_.region_id

        # If the region's results directory exists: we assume it contains data
        if self.paths.region_results(region_id).exists():
            print("Loading recipe for current region...")
            # Select paths
            segmenter_path = self.paths.segmenter(region_id)
            recipe_path = self.paths.recipe(region_id)
            echotype_path = self.paths.echotype_data(region_id)

            # Load to CurrentRegion dataclass
            self.current_.segmenter = joblib.load(segmenter_path)
            self.current_.segment_id = yaml.safe_load(open(recipe_path, "r"))["segment_id"]
            self.current_.echotype_data = xr.open_dataarray(echotype_path)

        # Else: we assume the directory is empty and there is no data to load
        else:
            print("No saved recipe to load.")
            pass

    #### Data selection utilities ####

    def _get_region_data(self, region_id: int) -> xr.DataArray:

        # Select only bbox for speed
        region_row = _select_region_row(self.regions, region_id, close=True)
        bbox_Sv = _select_bbox_data(self.data, self.acoustic_var, region_row)

        # Use the built in mask function to create a mask
        da_template = bbox_Sv.isel(channel=0).compute()
        mask_result = self.regions.region_mask(da_template, region_id)

        if mask_result is None:
            raise ValueError(f"{Regions2D.region_mask} result is None.")

        region_mask_ds: xr.Dataset = mask_result[0]  # type: ignore (possible signature error in region_mask)

        # Select where is 1 and masked region's region id is 1
        region_acoustic_data = xr.where(
            region_mask_ds["mask_3d"].sel(region_id=region_id, drop=False) == 1,
            bbox_Sv,
            np.nan,
        )

        return region_acoustic_data.drop_vars("region_id")

    def _get_segment(self, segment_id: int) -> xr.DataArray:
        """Get a single segment from the segmentation results.

        Parameters
        ----------
        segment_id : int
            Segment id to select on the segment coordinate.

        Returns
        -------
        xr.DataArray
            Segment data: region data where the segmentation predicted segment_id.
            Array padded with NA elsewhere.
            Shape (time, depth, channel).

        Raises
        ------
        ValueError
            If no current region has been set.
        ValueError
            If no segmented data is available in workflow.
        ValueError
            If segment_id is out of bounds.
        """
        # Checks
        if self.current_ is None:
            raise ValueError("No current region in process (use .set_current)")
        if self.current_.segments is None:
            raise ValueError("No segmented data (use .segment)")
        n = len(self.current_.segments[segment_id])
        if segment_id >= n:
            raise ValueError(f"segment_id out of bound for results with {n} segments.")

        # Select segment from segments in CurrentRegion
        return self.current_.segments.isel(segment=segment_id).drop_vars("segment")


@dataclass
class CurrentRegion:
    # Id & input
    region_id: int
    region_data: xr.DataArray

    # Worflow outputs
    segmenter: Pipeline | None = None
    segments: xr.DataArray | None = None
    segment_id: int | None = None
    echotype_data: xr.DataArray | None = None


class WorkflowDataVisualizer:
    def __init__(self, parent: EchotypeWorkflow):
        self.parent = parent

    #### Echograms ####

    # Input data

    def input(
        self,
        how: Literal["bbox", "exact"] = "exact",
        plot_api: Literal["hvplot", "plot"] = "hvplot",
        **plot_kwrgs,
    ):
        if self.parent.current_ is None:
            raise ValueError("No current region in process (use .set_current).")
        region_id = self.parent.current_.region_id

        if how == "bbox":
            region_row = _select_region_row(self.parent.regions, region_id, close=True)
            da_Sv = _select_bbox_data(self.parent.data, self.parent.acoustic_var, region_row)
            return plot_channels(da_Sv, plot_api, region_row, **plot_kwrgs)
        elif how == "exact":
            da_Sv = self.parent.current_.region_data
            return plot_channels(da_Sv, plot_api, **plot_kwrgs)
        else:
            raise ValueError(f"Invalid argument {how = }. Expected on of ['bbox', 'exact']")

    def input_rgb(
        self,
        how: Literal["bbox", "exact"] = "bbox",
        channel_idx: Tuple[int, int, int] = (0, 1, 2),
        **plot_kwrgs,
    ):
        # Checks
        if self.parent.current_ is None:
            raise ValueError("No current region in process (use .set_current).")
        region_id = self.parent.current_.region_id

        if not len(channel_idx) == 3:
            raise ValueError(f"channel_idx should be of length 3 for RGB plot. {channel_idx = }.")
        channels = list(channel_idx)

        if how == "bbox":
            region_row = _select_region_row(self.parent.regions, region_id, close=True)
            da_Sv = _select_bbox_data(self.parent.data, self.parent.acoustic_var, region_row)
            plot_rgb(da_Sv, channels, region_row, **plot_kwrgs)
        elif how == "exact":
            da_Sv = self.parent.current_.region_data
            plot_rgb(da_Sv, channels, **plot_kwrgs)

    # Segments

    def segments(
        self,
        plot_api: Literal["hvplot", "plot"] = "hvplot",
        **plot_kwrgs,
    ):
        if self.parent.current_ is None:
            raise ValueError("No current region in process (use .set_current).")
        if self.parent.current_.segments is None:
            raise ValueError("No segmented data (use .segment)")

        if plot_api == "plot":
            plot_kwrgs_overrides = {
                "row": "segment",
                "col": "channel",
                "col_wrap": None,
            }
            plot_kwrgs = plot_kwrgs | plot_kwrgs_overrides

        da_Sv = self.parent.current_.segments
        return plot_channels(da_Sv, plot_api, **plot_kwrgs)

    def segments_rgb(
        self,
        channel_idx: Tuple[int, int, int] = (0, 1, 2),
        ncols: int = 2,
        figsize: Tuple[float, float] = (15, 15),
        **plot_kwrgs,
    ):
        if self.parent.current_ is None:
            raise ValueError("No current region in process (use .set_current).")
        if self.parent.current_.segments is None:
            raise ValueError("No segmented data (use .segment)")
        if not len(channel_idx) == 3:
            raise ValueError(f"channel_idx should be of length 3 for RGB plot. {channel_idx = }.")
        channels = list(channel_idx)

        segments_values = self.parent.current_.segments["segment"].values
        n_segments = len(segments_values)
        nrows = n_segments // ncols + n_segments % ncols
        _, axes = plt.subplots(
            ncols,
            nrows,
            figsize=figsize,
            sharex=True,
            sharey=True,
        )
        for seg, (i, ax) in zip(segments_values, enumerate(axes.flat)):
            da_Sv = self.parent._get_segment(i)
            _ = plot_rgb(da_Sv, channels, ax=ax, **plot_kwrgs)
            ax.set_title(f"segment {i}")

    # Echotype echograms

    def echotype(
        self,
        plot_api: Literal["hvplot", "plot"] = "hvplot",
        **plot_kwrgs,
    ):
        if self.parent.current_ is None:
            raise ValueError("No current region in process (use .set_current).")
        if self.parent.current_.echotype_data is None:
            raise ValueError("No echotype_data (use .select_segment)")

        da_Sv = self.parent.current_.echotype_data
        return plot_channels(da_Sv, plot_api, **plot_kwrgs)

    def echotype_rgb(
        self,
        how: Literal["bbox", "exact"] = "bbox",
        channel_idx: Tuple[int, int, int] = (0, 1, 2),
        **plot_kwrgs,
    ):
        # Checks
        if self.parent.current_ is None:
            raise ValueError("No current region in process (use .set_current).")
        if self.parent.current_.echotype_data is None:
            raise ValueError("No echotype_data (use .select_segment)")
        if not len(channel_idx) == 3:
            raise ValueError(f"channel_idx should be of length 3 for RGB plot. {channel_idx = }.")
        channels = list(channel_idx)

        da_Sv = self.parent.current_.echotype_data
        plot_rgb(da_Sv, channels, **plot_kwrgs)

    #### Other plots ####

    def echotype_frequency_response(
        self,
        relative_to_channel: int | None = None,
        **hvplot_kwrgs,
    ):

        da_echotype = self.parent.get_echotype_dataarray()
        freqs = self.parent.data["frequency_nominal"]

        # Replace channel coordinate with frequency_nominal and rename dimension
        da_echotype = da_echotype.assign_coords(channel=freqs).rename(
            {"channel": "frequency_nominal"}
        )

        # Absolute frequency response
        if relative_to_channel is None:
            ds_afr = xr.Dataset(
                {
                    "mean": da_echotype.mean(dim=["depth", "ping_time"]),
                    "std": da_echotype.std(dim=["depth", "ping_time"]),
                }
            )

            hvplot_kwrgs_defaults = {
                "title": "Absolute Frequency Response",
                "xlabel": "Channel Frequency [Hz]",
                "ylabel": "Mean SV [dB]",
            }
            hvplot_kwrgs |= hvplot_kwrgs_defaults

            plot_afr = (
                ds_afr.hvplot.line(x="frequency_nominal", y="mean")
                * ds_afr.hvplot.scatter(x="frequency_nominal", y="mean")
                * ds_afr.hvplot.errorbars(x="frequency_nominal", y="mean", yerr1="std")
            ).opts(**hvplot_kwrgs)

            return plot_afr

        # Relative frequency response
        ref = relative_to_channel
        da_diff = da_echotype - da_echotype.isel(frequency_nominal=ref)
        ds_rfr = xr.Dataset(
            {
                "mean": da_diff.mean(dim=["depth", "ping_time"]),
                "std": da_diff.std(dim=["depth", "ping_time"]),
            }
        )

        hvplot_kwrgs_defaults = {
            "title": "Relative Frequency Response",
            "xlabel": "Channel Frequency [Hz]",
            "ylabel": f"Mean ΔSV (relative to ch. {ref}) [dB]",
        }
        hvplot_kwrgs |= hvplot_kwrgs_defaults

        plot_rfr = (
            ds_rfr.hvplot.line(x="frequency_nominal", y="mean")
            * ds_rfr.hvplot.scatter(x="frequency_nominal", y="mean")
            * ds_rfr.hvplot.errorbars(x="frequency_nominal", y="mean", yerr1="std")
        ).opts(**hvplot_kwrgs)

        return plot_rfr
