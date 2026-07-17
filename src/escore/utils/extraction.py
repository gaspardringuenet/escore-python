import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

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

from .extraction_utils.echograms import plot_channels, plot_rgb
from .extraction_utils.paths import ResultsPathManager
from .extraction_utils.wrangler import (
    _classes_to_segments,
    _format_feature_dataframe,
    _select_bbox_data,
    _select_region_row,
    _stack_for_sklearn,
    _unstack_sklearn_preds,
)


@dataclass
class CurrentRegion:
    """
    Data class storing the currently selected region, and its various processing levels.
    - `region_id`: Region id in the parent ExtractionWorkflow's region field
    - `region_data`: Acoustic data contained within the region's shape
    - `segmenter`: Segmentation function applied to the `region_data` samples to produce a segmentation
    - `uses_time_depth_features`: Whether segmenter take (time, depth, Sv_c1, ..., Sv_cn) as input instead of (Sv_c1, ..., Sv_cn)
    - `semgents`: DataArray similar to `region_data`, but with data split along a 'segment' dimension, corresponding to the segmentation produced by the segmenter
    - `segment_id`: Id of the selected segment (segment containing the desired feature)
    - `feature_data`: DataArray containing only the selected segment's data (padded with NA)
    """

    # Id & input
    region_id: int
    region_data: xr.DataArray

    # Worflow outputs
    segmenter: Pipeline | None = None
    uses_time_depth_features: bool | None = None
    segments: xr.DataArray | None = None
    segment_id: int | None = None
    feature_data: xr.DataArray | None = None


class ExtractionWorkflow:
    def __init__(
        self,
        ds_MVBS: xr.Dataset,
        regions: Regions2D,
        results_dir: str | Path,
        acoustic_var: str = "Sv",
    ):
        """
        Container class for the interactive post-processing of Regions2D annotations.

        Parameters
        ----------
        ds_MVBS : xr.Dataset
            Acoustic dataset. Should contain 'ping_time', 'depth' and 'channel' coordinates, as well as an acoustic variable (see `acoustic_var`).
        regions : Regions2D
            2D polygonal regions selected on the acoustic dataset (regions outside the dataset's bounds will cause errors).
        results_dir : str | Path
            Result directory for processing. Stored data include workflow status and extracted feature and segmentation recipe for each region.
        acoustic_var : str, optional
            Name of the acoustic variable in dataset, by default "Sv"
        """

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
            f" * {n_completed} marked as completed (features have been extracted)\n"
            f" * {n_rejected} marked as rejected (skipped without extracting)\n"
            f" * {n_pending} remain pending\n"
            f"Last loaded region id: {state_dict['last']}"
        )

    # Step 1 - Initialize by setting current region to process

    def set_current(
        self,
        region_id: int,
        verbose: bool = True,
    ) -> None:
        """Change the current region / feature being processed. Load processing data if
        there is some.

        Parameters
        ----------
        region_id : int
            Id of the region in the Regions2D annotation object.
        verbose : bool, optional
            Print processing info, by default True

        Raises
        ------
        ValueError
            If requested region id does not exist in the region.data DataFrame.
        """

        if region_id not in self.regions.data["region_id"].values:
            raise ValueError(
                f"Unkown {region_id = }. "
                "(region_id must reference the region_id attribute of the regions.data DataFrame)"
            )

        # Display message if a current id already exists
        if self.current_ is not None and verbose:
            if self.current_.region_id != region_id:
                print(
                    f"Switching current id from {self.current_.region_id} to {region_id}. "
                    "Unsaved changes for the previous feature are dropped. "
                )
            else:
                print(
                    f"Current already set to {self.current_.region_id}. Reloading recipe..."
                )

        # Create new current instance, pre-load data subset and set edit mode
        self.current_ = CurrentRegion(region_id, self._get_region_data(region_id))

        # Load recipe, pipeline, and results (if they exist)
        self._load_current(verbose)

        # Display warning about the status of region
        state_dict = self._load_state()
        status: str = state_dict["status"][region_id]
        print(
            f"WARNING: Current region has status {status.upper()}."
        ) if verbose else None

        # Update workflow state
        state_dict["last"] = region_id
        self._dump_state(state_dict)

        print("Ready for processing!") if verbose else None

    def reset_current(self):
        """Resets current feature state as blank. Only region_id and region_data
        fields are preserved.

        Raises
        ------
        ValueError
            If no current region has been set.
        """

        if self.current_ is None:
            raise ValueError("No current region in process (use .set_current)")

        print("Resetting current feature state as blank...")
        self.current_.segmenter = None
        self.current_.segments = None
        self.current_.segment_id = None
        self.current_.feature_data = None
        print("Ready for processing!")

    # Step 2 - Set segmentation pipeline & perform segmentation

    def set_segmenter(self, segmenter: Pipeline):
        """Set the segmentation pipeline for feature extraction.
        Pipeline ingests np.ndarray's in shape (n_samples, n_channels).
        Segmenter is stored as .segmenter field in `current_`.

        Parameters
        ----------
        segmenter : Pipeline
            scikit-learn Pipeline for feature extraction.

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
        # TODO: Add possibility to use ping_time and depth as features

        # Checks
        if self.current_ is None:
            raise ValueError("No current region in process (use .set_current)")
        if self.current_.segmenter is None:
            raise ValueError("No segmenter (use .set_segmenter)")
        if self.current_.region_data is None:
            self.current_.region_data = self._get_region_data(self.current_.region_id)

        # Preprocessing
        da_Sv = self.current_.region_data
        da_stacked = _stack_for_sklearn(da_Sv, drop_na=True)
        X = da_stacked.values  # (n_samples, n_channels)

        # Fit model & predict
        preds: np.ndarray = self.current_.segmenter.fit_predict(X)  # type: ignore

        # Formatting predictions
        da_preds = _unstack_sklearn_preds(preds, da_stacked)
        da_segments = _classes_to_segments(da_Sv, da_preds)

        # Save to CurrentRegion
        self.current_.segments = da_segments

        return da_segments

    def select_segment(
        self,
        segment_id: int,
        verbose: bool = True,
    ) -> xr.DataArray | None:
        """Select feature data by indicating the segment id to chose.
        Select a segment from segmented data, modify current feature
        data and metadata, and return feature DataArray.

        Parameters
        ----------
        segment_id : int
            id of the segment corresponding to the desired feature.
        verbose: bool
            Whether to print info to the user.

        Returns
        -------
        xr.DataArray
            feature DataArray (time, depth, channel) dimensions.
        """
        # Get segment from segmented DataArray
        try:
            segment = self._get_segment(segment_id)
            segment.name = self.acoustic_var
        except ValueError as e:
            print(f"Unable to fetch segment {segment_id} - {e}")
            return None

        # Save segment_id & feature_data to CurrentRegion
        print(f"Using segment {segment_id} as feature data.") if verbose else None
        self.current_.segment_id = segment_id  # type: ignore (check already performed by _get_segment)
        self.current_.feature_data = segment  # type: ignore

        return segment

    # Step 3 - Save results

    def save_current(
        self,
        overwrite: bool = False,
        verbose: bool = True,
    ):
        """Save feature data and recipe in results directory.
        Automatically change the region's status to 'completed'.

        Parameters
        ----------
        overwrite : bool, optional
            Allow overwriting an existing feature, by default False
        verbose: bool
            Whether to print info to the user.

        Raises
        ------
        ValueError
            If no current region has been set.
        ValueError
            If processing is incomplete (None fields remaining in the CurrentRegion object).
        ValueError
            If user attemps to save an already saved feature with overwrite = False.
        """

        if self.current_ is None:
            raise ValueError("No current region to dump data from.")

        if any(asdict(self.current_).values()) is None:
            raise ValueError("Cannot dump incomplete feature.")

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
        feature_path = self.paths.feature_data(region_id)

        # Format recipe
        recipe: dict = {
            "segmenter": str(self.current_.segmenter),
            "segment_id": self.current_.segment_id,
        }

        # Dump from CurrentRegion
        try:
            joblib.dump(self.current_.segmenter, segmenter_path)
            yaml.safe_dump(recipe, open(recipe_path, "w"))
            self.current_.feature_data.to_netcdf(feature_path, engine="netcdf4")  # type: ignore
        except Exception as e:
            print(f"Dumping feature failed: {e}") if verbose else None
            if overwrite:
                print("Back to previous version of results dir") if verbose else None
                shutil.move(failsafe_dir / dir.name, dir.parent)
            else:
                print("Removing results dir") if verbose else None
                shutil.rmtree(dir)
        finally:
            if failsafe_dir.exists():
                shutil.rmtree(failsafe_dir)
            self._mark_completed(verbose)
            print(
                "Selected feature data & recipe saved successfully!"
            ) if verbose else None

    # Step 4 - Mark the region as processed ("completed" or "rejected")

    def _mark_completed(self, verbose: bool = True):
        """Set current region's status as completed. Called by save_current.
        Should not be called by user directly."""

        if self.current_ is None:
            raise ValueError("No current region to dump data from.")

        state_dict = self._load_state()
        state_dict["status"][self.current_.region_id] = "completed"
        self._dump_state(state_dict)

        if verbose:
            print(
                f"Region {self.current_.region_id} worflow status updated to 'completed'."
            )

    def mark_rejected(self):
        """Set current region's status as rejected in workflow state
        file."""

        if self.current_ is None:
            raise ValueError("No current region to dump data from.")

        state_dict = self._load_state()
        state_dict["status"][self.current_.region_id] = "rejected"
        self._dump_state(state_dict)

        print(f"Region {self.current_.region_id} worflow status updated to 'rejected'.")

    # Bonus Step - Get Selected feature data for direct inspection

    def get_feature_dataarray(self) -> xr.DataArray:
        """Return selected feature DataArray if is exists.

        Returns
        -------
        xr.DataArray
            Selected feature DataArray.

        Raises
        ------
        ValueError
            If no current region has been set.
        ValueError
            If no feature data is available.
        """
        # Checks
        if self.current_ is None:
            raise ValueError("No current region in process (use .set_current)")
        if self.current_.feature_data is None:
            raise ValueError("No feature data (use .segment & .select_segment)")

        return self.current_.feature_data

    def get_feature_dataframe(
        self,
        add_region_cols: List[str] | Literal["default"] = "default",
    ) -> pd.DataFrame:
        """Return selected feature as DataFrame if is exists

        Parameters
        ----------
        add_region_cols : List[str] | Literal[&quot;default&quot;], optional
            Region data to append to each feature row. If "default", columns
            "region_id" and "region_class" are used. To add new columns, specify
            column names as a list. By default "default".

        Returns
        -------
        pd.DataFrame
            DataFrame containing feature data with default columns
            (depth, time, channel_0_Sv, ..., channel_n_Sv, region_id, region_class)
            and requested additional region columns (if they exist).

        Raises
        ------
        ValueError
            If add_region_cols is neither "default" nor a list of strings.
        """

        da_feature = self.get_feature_dataarray()

        # Format to DataFrame with columns: depth, time, channel_0_Sv, ..., channel_n_Sv
        df = (
            da_feature.to_dataframe(
                name="Sv",
                dim_order=["depth", "ping_time", "channel"],
            )
            .reset_index()
            .pivot_table(
                index=["depth", "ping_time", "latitude", "longitude"],
                columns="channel",
                values="Sv",
            )
        )
        df.columns.name = None

        # Rename channel columns
        df = df.rename(
            columns={col: f"channel_{i}_Sv" for i, col in enumerate(df.columns)}
        ).reset_index()

        # Add region attributes
        # Select region row
        region_row = _select_region_row(
            self.regions,
            self.current_.region_id,  # type: ignore (check performed by .get_feature_datarray)
            close=False,
        )

        # Append default columns
        default_cols = ["region_id", "region_class"]
        if isinstance(add_region_cols, list) and len(add_region_cols) > 0:
            # Keep defaults and add additional
            cols = default_cols + [
                col for col in add_region_cols if col not in default_cols
            ]
        elif add_region_cols == "default":
            cols = default_cols
        else:
            raise ValueError(
                f'add_region_cols argument must be either "default" or list of strings.Got {add_region_cols}.'
            )

        # Add new columns for all samples in feature
        for col in cols:
            # Avoid potential confusion between sample and region attribute (e.g. "depth")
            colname = col if col not in df.columns else f"region_{col}"

            # Add new column and print warning if column does not exist
            try:
                df[colname] = [region_row[col]] * len(df)
            except KeyError:
                print(
                    f"Warning: column {col} does not exist in regions data. Available columns {list(region_row.index)}."
                )

        return df

    # Step 5 - Export entire features dataset as .csv file
    def export_completed(
        self,
        add_region_cols: List[str] | Literal["default"] = "default",
    ) -> pd.DataFrame:
        """Export all features as a single DataFrame. Loop through
        "completed" regions and convert to dataframes using the
        `.get_feature_dataframe` method. Echotypes are assigned unique
        int ids.

        Parameters
        ----------
        add_region_cols : List[str] | Literal[&quot;default&quot;], optional
            Regions data to insert. Passed to get_feature_dataframe, by default "default"

        Returns
        -------
        pd.DataFrame
            DataFrame containing feature data with default columns
            (feature_id, depth, time, channel_0_Sv, ..., channel_n_Sv, region_id, region_class)
            and requested additional region columns (if they exist).
        """

        state_dict = self._load_state()
        region_status_dict = state_dict["status"]

        df_list = []

        for i, (region_id, status) in enumerate(region_status_dict.items()):
            if status != "completed":
                continue
            self.set_current(region_id, verbose=False)
            try:
                df = self.get_feature_dataframe(add_region_cols)
            except Exception as e:
                print(f"Failed with region {region_id} - {e}")
                continue
            df.insert(0, "feature_id", i)
            df_list.append(df)

        df_tot = pd.concat(df_list)

        return df_tot

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

    def _load_current(self, verbose: bool) -> None:
        """Load a region / feature processing data. Update CurrentRegion object.
        Directory state assumption: existence of a region's result folder is equivalent
        to this folder containing all necessary files (This assumption is enforced by
        dumping all files at once and deleting the directory in case of failure -
        see .dump_current).

        Parameters
        ----------
        verbose : bool
            Print loading info.

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
            print("Loading recipe for current region...") if verbose else None
            # Select paths
            segmenter_path = self.paths.segmenter(region_id)
            recipe_path = self.paths.recipe(region_id)
            feature_path = self.paths.feature_data(region_id)

            # Load to CurrentRegion dataclass
            self.current_.segmenter = joblib.load(segmenter_path)
            self.current_.segment_id = yaml.safe_load(open(recipe_path, "r"))[
                "segment_id"
            ]
            self.current_.feature_data = xr.open_dataarray(feature_path)

        # Else: we assume the directory is empty and there is no data to load
        else:
            print("No saved recipe to load.") if verbose else None
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
            keep_attrs=True,
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


class WorkflowDataVisualizer:
    def __init__(self, parent: ExtractionWorkflow):
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
            da_Sv = _select_bbox_data(
                self.parent.data, self.parent.acoustic_var, region_row
            )
            return plot_channels(da_Sv, plot_api, region_row, **plot_kwrgs)
        elif how == "exact":
            da_Sv = self.parent.current_.region_data
            return plot_channels(da_Sv, plot_api, **plot_kwrgs)
        else:
            raise ValueError(
                f"Invalid argument {how = }. Expected on of ['bbox', 'exact']"
            )

    def input_rgb(
        self,
        how: Literal["bbox", "exact"] = "bbox",
        channel_idx: Tuple[int, int, int] = (0, 1, 2),
        **plot_kwrgs,
    ):
        """RGB echogram of the selected region."""

        # Checks
        if self.parent.current_ is None:
            raise ValueError("No current region in process (use .set_current).")
        region_id = self.parent.current_.region_id

        if not len(channel_idx) == 3:
            raise ValueError(
                f"channel_idx should be of length 3 for RGB plot. {channel_idx = }."
            )
        channels = list(channel_idx)

        if how == "bbox":
            region_row = _select_region_row(self.parent.regions, region_id, close=True)
            da_Sv = _select_bbox_data(
                self.parent.data, self.parent.acoustic_var, region_row
            )
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
        """RGB echogram of each segment produced by ExtractionWorkflow.segment().

        Parameters
        ----------
        channel_idx : Tuple[int, int, int], optional
            Index of the 3 channels to map to red, green, and blue, by default (0, 1, 2)
        ncols : int, optional
            Number of subplot columns, by default 2
        figsize : Tuple[float, float], optional
            Size of the figure in inches. Passed to matplotlib.pyplot.subplots, by default (15, 15)
        **plot_kwrgs :
            Arguments passed to matplotlib.pyplot.pcolormesh.

        Raises
        ------
        ValueError
            If no current region has been set.
        ValueError
            If no segmented data is available in workflow.
        ValueError
            If number of channel indices is different from 3.
        """
        if self.parent.current_ is None:
            raise ValueError("No current region in process (use .set_current).")
        if self.parent.current_.segments is None:
            raise ValueError("No segmented data (use .segment)")
        if not len(channel_idx) == 3:
            raise ValueError(
                f"channel_idx should be of length 3 for RGB plot. {channel_idx = }."
            )
        channels = list(channel_idx)

        segments_values = self.parent.current_.segments["segment"].values
        n_segments = len(segments_values)
        nrows = n_segments // ncols + (n_segments % ncols > 0)
        _, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            sharex=True,
            sharey=True,
        )
        for _, (i, ax) in zip(segments_values, enumerate(axes.flat)):
            da_Sv = self.parent._get_segment(i)
            _ = plot_rgb(da_Sv, channels, ax=ax, **plot_kwrgs)
            ax.set_title(f"segment #{i}")

    # Selected feature echograms

    def feature(
        self,
        plot_api: Literal["hvplot", "plot"] = "hvplot",
        **plot_kwrgs,
    ):
        if self.parent.current_ is None:
            raise ValueError("No current region in process (use .set_current).")
        if self.parent.current_.feature_data is None:
            raise ValueError("No feature_data (use .select_segment)")

        da_Sv = self.parent.current_.feature_data
        return plot_channels(da_Sv, plot_api, **plot_kwrgs)

    def feature_rgb(
        self,
        how: Literal["bbox", "exact"] = "bbox",
        channel_idx: Tuple[int, int, int] = (0, 1, 2),
        **plot_kwrgs,
    ):
        # Checks
        if self.parent.current_ is None:
            raise ValueError("No current region in process (use .set_current).")
        if self.parent.current_.feature_data is None:
            raise ValueError("No feature_data (use .select_segment)")
        if not len(channel_idx) == 3:
            raise ValueError(
                f"channel_idx should be of length 3 for RGB plot. {channel_idx = }."
            )
        channels = list(channel_idx)

        da_Sv = self.parent.current_.feature_data
        plot_rgb(da_Sv, channels, **plot_kwrgs)

    #### Other plots ####

    def feature_frequency_response(
        self,
        relative_to_channel: int | None = None,
        channel_values: List[float] | None = None,
        channel_varname: str = "channel",
    ):
        # Get feature data as dataframe
        df = self.parent.get_feature_dataframe()

        # Format dataframe (rename channel cols + compute Sv diff if necessary)
        df = _format_feature_dataframe(df, relative_to_channel)

        # Fetch channel columns
        channel_cols = df.filter(regex=r"^channel_\d+").columns.to_list()

        # Build summary dataframe
        summary = pd.DataFrame(
            {"mean": df[channel_cols].mean(), "sd": df[channel_cols].std()}
        )
        summary.index.name = "channel"
        summary = summary.reset_index()

        # Use custom variable name & values
        if channel_values is not None:
            summary[channel_varname] = channel_values
        else:
            summary[channel_varname] = summary["channel"]

        # Build hvplot elements
        curve = summary.hvplot.line(x=channel_varname, y="mean", line_width=2)
        points = summary.hvplot.scatter(x=channel_varname, y="mean", size=6)
        errors = summary.hvplot.errorbars(x=channel_varname, y="mean", yerr1="sd")

        # Overlay
        plot = curve * points * errors

        # Ensure margin around x values
        try:
            xmin, xmax = min(channel_values), max(channel_values)  # type: ignore
            xrange = xmax - xmin
            plot = plot.opts(xlim=(xmin - 0.05 * xrange, xmax + 0.05 * xrange))
        except Exception:
            pass

        # Default opts
        varname = self.parent.acoustic_var
        absolute = relative_to_channel is None
        ylabel = varname if absolute else "Δ" + varname
        title = "Frequency Response"
        prefix = "Absolute " if absolute else "Relative "
        title = prefix + title
        plot = plot.opts(
            ylim=(-85, -55) if absolute else (-15, 15),
            ylabel=ylabel,
            title=title,
            show_grid=True,
            width=400,
        )

        return plot

    def feature_frequency_response_hist(
        self,
        relative_to_channel: int | None = None,
        channel_values: Sequence[float] | None = None,
        channel_varname: str = "channel",
        **hvplot_kwrgs,
    ) -> hv.element.Histogram:
        """Histogram of the feature's Sv (or Delta Sv) values, by channel.
        Delta Sv if computed if 'relative_to_channel' is specified.

        Parameters
        ----------
        relative_to_channel : int | None, optional
            Index of the reference channel to subtract to all channels, by default None
        channel_values : Sequence[float] | None, optional
            Channel values to use instead of default 'channel_{i}' labels.
            Must match the order of channel in the workflow's acoustic dataset.
            By default None
        channel_varname : str, optional
            Name of the channel variable, by default "channel"

        Returns
        -------
        hv.element.Histogram
            Histogram plot using holoviews.

        Raises
        ------
        ValueError
            If channel_values length does not match the number of channel columns.
        """

        # Get feature data as dataframe
        df = self.parent.get_feature_dataframe()

        # Format dataframe (rename channel cols + compute Sv diff if necessary)
        df = _format_feature_dataframe(df, relative_to_channel)

        # Fetch channel columns
        channel_cols = df.filter(regex=r"^channel_\d+").columns.to_list()

        # Use custom variable name & values
        if channel_values is not None:
            if not len(channel_values) == len(channel_cols):
                raise ValueError(
                    f"Number of provided channel values ({len(channel_values)})"
                    f"does not match number of channels: {len(channel_cols)}."
                )
            # Rename channel cols to contain channel values
            df = df.rename(
                columns={col: val for col, val in zip(channel_cols, channel_values)}
            )

            # Use provided values as new channel columns
            channel_values_final = list(channel_values)
        else:
            # Use default channel columns
            channel_values_final = channel_cols

        # Drop reference channel (all zeros)
        if relative_to_channel is not None:
            ref_channel_value = channel_values_final[relative_to_channel]
            df = df.drop(ref_channel_value, axis=1)  # Drop
            # Remove reference channel from channel values
            channel_values_final.remove(ref_channel_value)  # type: ignore

        # Melt channel columns to long format
        df = df.melt(
            id_vars=["depth", "ping_time"],
            value_vars=channel_values_final,
            var_name=channel_varname,
            value_name="value",
        )

        # Build hvplot histogram
        defaults = {"normed": True, "bins": 40, "alpha": 0.7}
        hvplot_kwrgs = defaults | hvplot_kwrgs

        hist = df.hvplot.hist(y="value", by=channel_varname, **hvplot_kwrgs)

        return hist  # type: ignore
