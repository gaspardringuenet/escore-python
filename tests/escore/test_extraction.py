import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import echoregions as er
import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from escore import (
    ExtractionConfig,
    ExtractionResults,
    ReviewDecision,
    ReviewDecisions,
    apply_decisions,
    assess_selection,
    extract_echotypes,
    plot_review,
    summarize_candidates,
)
from escore.extraction._processing import stack_data


@pytest.fixture
def sv():
    time = pd.date_range("2020-01-01", periods=8, freq="1min")
    depth = np.arange(8, dtype=float)
    values = np.empty((3, 8, 8))
    values[0] = -60 + np.arange(8)[:, None] * 0.1
    values[1] = values[0] + np.where(np.arange(8)[:, None] < 4, 1, 10)
    values[2] = values[0] + np.where(np.arange(8)[:, None] < 4, 2, 15)
    values[:, 3, 3] = np.nan
    return xr.DataArray(
        values,
        dims=("channel", "ping_time", "depth"),
        coords={"channel": ["ref", "f1", "f2"], "ping_time": time, "depth": depth},
    )


@pytest.fixture
def regions(sv):
    time = sv.ping_time.values
    data = pd.DataFrame(
        [
            {
                "region_id": 1,
                "region_name": "inside",
                "time": np.array([time[1], time[6], time[6], time[1]]),
                "depth": np.array([1.0, 1.0, 6.0, 6.0]),
            },
            {
                "region_id": 2,
                "region_name": "outside",
                "time": np.array([time[1], time[6], time[6], time[1]]),
                "depth": np.array([50.0, 50.0, 60.0, 60.0]),
            },
        ]
    )
    return er.read_regions_csv(data)


@pytest.mark.parametrize("channels", [("ref", "f1"), ("ref", "f1", "f2")])
def test_extract_real_region_and_raw_selection(sv, regions, channels):
    config = ExtractionConfig(
        channels=channels,
        kernel=np.ones((3, 3)),
        clusterer="kmeans",
        kmeans_clusters=2,
    )
    results = extract_echotypes(sv, regions, config=config)

    assert set(results.candidates) == {1}
    assert results.failures[2].stage == "mask"
    candidate = results.candidates[1]
    assert candidate.metadata["region_name"] == "inside"
    assert candidate.cluster_ids == (0, 1)
    assert candidate.proposed_cluster in candidate.cluster_ids
    assert not candidate.raw_sv.identical(candidate.smoothed_sv)
    assert candidate.labels.sel(ping_time=sv.ping_time[3], depth=3).item() == -1
    for cluster in candidate.cluster_ids:
        xr.testing.assert_equal(
            candidate.selected_sv(cluster), candidate.raw_sv.where(candidate.labels == cluster)
        )
        expected = (candidate.labels == cluster).sum().item() / (candidate.labels >= 0).sum().item()
        assert candidate.retained_share(cluster) == pytest.approx(expected)
    with pytest.raises(ValueError, match="Cluster 9"):
        candidate.mask(9)

    mode = "1D" if len(channels) == 2 else "2D"
    _, existing_data = stack_data(candidate.smoothed_sv, mode=mode)
    expected_labels = np.full(len(existing_data), -1)
    X = existing_data.reshape(len(existing_data), -1)
    valid = np.isfinite(X).all(axis=1)
    expected_labels[valid] = candidate.model.predict(X[valid])
    np.testing.assert_array_equal(
        candidate.labels.transpose("ping_time", "depth").values.ravel(), expected_labels
    )


def test_csv_input_and_confidence_filter(sv, regions, tmp_path):
    path = tmp_path / "regions.csv"
    regions.to_csv(path, index=False)
    config = ExtractionConfig(
        channels=("ref", "f1", "f2"),
        gmm_components=(1,),
        confidence_threshold=1.0,
    )
    results = extract_echotypes(sv, path, config=config)
    candidate = results.candidates[1]
    assert candidate.model.n_components == 1
    xr.testing.assert_equal(candidate.labels, candidate.filtered_labels)
    assert candidate.retained_share(candidate.proposed_cluster) == 1


def test_standalone_import_and_extraction():
    package_parent = Path(__file__).resolve().parents[2] / "packages" / "escore" / "src"
    script = dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[1])
        import echoregions as er
        import numpy as np
        import pandas as pd
        import xarray as xr
        import escore

        time = pd.date_range("2020-01-01", periods=5, freq="1min").to_numpy()
        sv = xr.DataArray(
            np.ones((2, 5, 5)),
            dims=("channel", "ping_time", "depth"),
            coords={"channel": ["ref", "f1"], "ping_time": time, "depth": np.arange(5.)},
        )
        regions = er.read_regions_csv(pd.DataFrame([{
            "region_id": 1,
            "time": np.array([time[0], time[4], time[4], time[0]]),
            "depth": np.array([0., 0., 4., 4.]),
        }]))
        config = escore.ExtractionConfig(channels=("ref", "f1"), gmm_components=(1,))
        results = escore.extract_echotypes(sv, regions, config=config)
        assert 1 in results.candidates, results.failures
        assert not any(name == "experiment" or name.startswith("experiment.")
                       for name in sys.modules)
        """
    )
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(package_parent)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_mask_respects_explicit_bounds_and_polygon(sv, regions):
    regions.data["region_bbox_left"] = sv.ping_time.values[0]
    regions.data["region_bbox_right"] = sv.ping_time.values[-1]
    regions.data["region_bbox_top"] = 0.0
    regions.data["region_bbox_bottom"] = 7.0
    config = ExtractionConfig(channels=("ref", "f1"), gmm_components=(1,))
    raw = extract_echotypes(sv, regions, config=config).candidates[1].raw_sv
    times = np.arange(8)[:, None]
    depths = np.arange(8)[None, :]
    inside = (times > 1) & (times <= 6) & (depths > 1) & (depths <= 6)
    expected = np.where(inside[None, :, :], sv.sel(channel=["ref", "f1"]).values, np.nan)
    np.testing.assert_array_equal(raw.transpose("channel", "ping_time", "depth").values, expected)


def test_multicomponent_confidence_filter_uses_fitted_probabilities(sv, regions):
    sv = sv.copy()
    sv.loc[dict(channel="f1")] = sv.sel(channel="ref") + np.arange(8)[:, None] * 0.3
    config = ExtractionConfig(
        channels=("ref", "f1"),
        gmm_components=(2,),
        confidence_threshold=1.0,
    )
    candidate = extract_echotypes(sv, regions, config=config).candidates[1]
    _, data = stack_data(candidate.smoothed_sv, mode="1D")
    X = data.reshape(-1, 1)
    labels = candidate.labels.transpose("ping_time", "depth").values.ravel()
    valid = labels >= 0
    probabilities = candidate.model.predict_proba(X[valid])
    expected = labels.copy()
    expected[valid] = np.where(
        probabilities[np.arange(valid.sum()), labels[valid]] >= 1.0, labels[valid], -1
    )
    assert np.any(expected[valid] == -1)
    np.testing.assert_array_equal(
        candidate.filtered_labels.transpose("ping_time", "depth").values.ravel(), expected
    )


def test_chunked_input_can_be_smoothed(sv, regions):
    config = ExtractionConfig(channels=("ref", "f1"), kernel=np.ones((3, 3)))
    chunked = extract_echotypes(sv.chunk({"ping_time": 2, "depth": 2}), regions, config=config)
    eager = extract_echotypes(sv, regions, config=config)
    xr.testing.assert_equal(chunked.candidates[1].labels, eager.candidates[1].labels)
    xr.testing.assert_allclose(
        chunked.candidates[1].smoothed_sv.compute(), eager.candidates[1].smoothed_sv
    )


def test_extract_more_than_two_delta_features(sv, regions):
    extra = (sv.sel(channel="f2") + 3).expand_dims(channel=["f3"])
    four_channels = xr.concat([sv, extra], dim="channel")
    config = ExtractionConfig(channels=("ref", "f1", "f2", "f3"), gmm_components=(1,))
    candidate = extract_echotypes(four_channels, regions, config=config).candidates[1]
    assert candidate.model.means_.shape == (1, 3)
    assert candidate.selected_sv(0).sizes["channel"] == 4


def test_adjacent_clustering_and_sv_bounds(sv, regions):
    from escore.extraction import explore_kmeans

    config = ExtractionConfig(
        channels=("ref", "f1", "f2"), clusterer="kmeans", kmeans_clusters=2,
        clustering_features="adjacent", clustering_channels=("f1", "ref", "f2"),
        min_sv=-59.9,
    )
    candidate = extract_echotypes(sv, regions, config=config).candidates[1]
    data = candidate.smoothed_sv.sel(channel=["f1", "ref", "f2"]).diff("channel")
    values = data.stack(pixel=("ping_time", "depth")).transpose("pixel", "channel").values
    valid = np.isfinite(values).all(axis=1)
    labels = candidate.labels.transpose("ping_time", "depth").values.ravel()
    np.testing.assert_array_equal(labels[valid], candidate.model.predict(values[valid]))
    assert np.isnan(candidate.raw_sv.where(sv.sel(channel="ref") < -59.9).values).all()
    scores = explore_kmeans(candidate, config, [2])
    assert np.isfinite(scores.loc[2, "calinski_harabasz"])


def test_no_finite_pixels_is_reported_per_roi(sv, regions):
    results = extract_echotypes(
        xr.full_like(sv, np.nan),
        regions,
        config=ExtractionConfig(channels=("ref", "f1")),
    )
    assert not results.candidates
    assert results.failures[1].stage == "cluster"
    assert "no pixels" in results.failures[1].message
    assert results.failures[2].stage == "mask"


def test_insufficient_pixels_is_reported(sv, regions):
    sparse = xr.full_like(sv, np.nan)
    sparse.loc[dict(ping_time=sv.ping_time[3], depth=4)] = [-60, -55, -50]
    results = extract_echotypes(
        sparse, regions, config=ExtractionConfig(channels=("ref", "f1"))
    )
    assert results.failures[1].stage == "cluster"
    assert "Not enough valid data" in results.failures[1].message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channels": ("ref",)},
        {"channels": ("ref", "ref")},
        {"channels": ("ref", "f1"), "kernel": [[1, -1]]},
        {"channels": ("ref", "f1"), "kernel": [[0, 0]]},
        {"channels": ("ref", "f1"), "confidence_threshold": float("nan")},
        {"channels": ("ref", "f1"), "clusterer": "kmeans", "confidence_threshold": 0.5},
        {"channels": ("ref", "f1"), "random_state": -1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        ExtractionConfig(**kwargs)


def test_missing_channels_and_unsorted_coordinates_fail(sv, regions):
    with pytest.raises(ValueError, match="Channels not found"):
        extract_echotypes(sv, regions, config=ExtractionConfig(channels=("ref", "missing")))
    with pytest.raises(ValueError, match="sorted"):
        extract_echotypes(
            sv.isel(depth=slice(None, None, -1)),
            regions,
            config=ExtractionConfig(channels=("ref", "f1")),
        )


@pytest.fixture
def extracted(sv, regions):
    return extract_echotypes(
        sv,
        regions,
        config=ExtractionConfig(
            channels=("ref", "f1", "f2"), kernel=np.ones((3, 3)),
            clusterer="kmeans", kmeans_clusters=2,
        ),
    )


def test_review_roundtrip_and_raw_means(extracted, tmp_path):
    review = ReviewDecisions.new(extracted)
    candidate = extracted.candidates[1]
    cluster = candidate.proposed_cluster
    with pytest.raises(ValueError, match="pending"):
        apply_decisions(extracted, review)
    summary = summarize_candidates(extracted, review, min_share=1)
    assert set(summary["status"]) == {"pending", "failed"}
    assert "low_share" in summary.loc[summary.roi_id == 1, "flags"].iloc[0]

    review.set(ReviewDecision(1, cluster, "accepted", "reviewed", ("low_share",)))
    path = tmp_path / "review.json"
    review.save(path)
    loaded = ReviewDecisions.load(path)
    outcome = apply_decisions(extracted, loaded)
    assert outcome.failures[2].stage == "mask"
    assert outcome.echotypes.items[0].decision.reason == "reviewed"
    assert outcome.echotypes.items[0].decision.overridden_flags == ("low_share",)
    raw = candidate.selected_sv(cluster).transpose("ping_time", "depth", "channel").values
    raw = raw.reshape(-1, 3)
    raw = raw[np.isfinite(raw).all(axis=1)]
    expected = (raw[:, 1:] - raw[:, :1]).mean(axis=0)
    np.testing.assert_allclose(outcome.echotypes.means.iloc[0].values, expected)
    assert list(outcome.echotypes.means.columns) == ["f1 - ref", "f2 - ref"]
    assert outcome.echotypes.means.index.tolist() == [(1, cluster)]
    other = replace(extracted, extraction_id="different")
    with pytest.raises(ValueError, match="different extraction"):
        apply_decisions(other, loaded)


def test_review_selection_multiple_and_missing(extracted):
    review = ReviewDecisions.new(extracted)
    candidate = extracted.candidates[1]
    first, second = candidate.cluster_ids
    review.select(1, second)
    assert review.decisions == (ReviewDecision(1, second, "pending"),)
    review.set(ReviewDecision(1, second, "accepted"))
    review.set(ReviewDecision(1, first, "accepted"))
    with pytest.raises(ValueError, match="existing decision"):
        review.select(1, first)
    outcome = apply_decisions(extracted, review)
    assert set(outcome.echotypes.means.index) == {(1, first), (1, second)}
    with pytest.raises(ValueError, match="accepted"):
        review.reject_roi(1)
    review.remove(1, first)
    review.remove(1, second)
    with pytest.raises(ValueError, match="pending or missing"):
        apply_decisions(extracted, review)
    partial = apply_decisions(extracted, review, allow_pending=True)
    assert partial.pending == (ReviewDecision(1, candidate.proposed_cluster, "pending"),)
    assert partial.echotypes.means.empty
    review.reject_roi(1, "unsuitable")
    assert apply_decisions(extracted, review).rejected[0].reason == "unsuitable"


def test_review_rejects_invalid_or_empty_selections(extracted):
    review = ReviewDecisions.new(extracted)
    review.select(1, 999)
    with pytest.raises(ValueError, match="not fitted"):
        apply_decisions(extracted, review)
    review.select(1, extracted.candidates[1].proposed_cluster)
    cluster = review.decisions[0].cluster_id
    extracted.candidates[1].filtered_labels = xr.full_like(extracted.candidates[1].labels, -1)
    assert "empty_selection" in assess_selection(extracted, 1, cluster).flags
    review.set(ReviewDecision(1, cluster, "accepted"))
    with pytest.raises(ValueError, match="empty"):
        apply_decisions(extracted, review)


def test_extraction_snapshot_is_materialized(sv, regions, tmp_path):
    results = extract_echotypes(
        sv.chunk({"ping_time": 2, "depth": 2}),
        regions,
        config=ExtractionConfig(channels=("ref", "f1"), kernel=np.ones((3, 3))),
    )
    path = tmp_path / "extraction.joblib"
    results.save(path)
    restored = ExtractionResults.load(path)
    assert restored.extraction_id == results.extraction_id
    assert restored.candidates[1].raw_sv.chunks is None
    xr.testing.assert_equal(restored.candidates[1].raw_sv, results.candidates[1].raw_sv.compute())
    assert ReviewDecisions.new(restored).extraction_id == results.extraction_id


@pytest.mark.parametrize("channels", [("ref", "f1"), ("ref", "f1", "f2")])
def test_plot_review_draws_seven_views(sv, regions, channels):
    results = extract_echotypes(
        sv, regions, config=ExtractionConfig(channels=channels, gmm_components=(1,))
    )
    fig = plot_review(results, 1, cluster_id=0)
    fig.canvas.draw()
    assert [ax.get_title() for ax in fig.axes[:7]] == [
        "Raw echogram", "Raw ΔMVBS", "Smoothed echogram", "Smoothed ΔMVBS",
        "Smoothed clusters", "Selected raw ΔMVBS", "Selected raw pixels",
    ]
    assert fig.axes[1].get_xlim() == (-20, 20)
    assert fig.axes[5].get_xlim() == (-20, 20)
    plt.close(fig)


def test_pipeline_extract_review_resume_and_apply(sv, regions, tmp_path, monkeypatch):
    from experiment.config.extract import ExtractConfig, ModeConfig
    from experiment.pipelines import extract as pipeline

    zarr = tmp_path / "sv.zarr"
    sv.assign_coords(channel=[38000.0, 70000.0, 120000.0]).to_dataset(name="Sv").to_zarr(zarr)
    csv = tmp_path / "rois.csv"
    regions.to_csv(csv, index=False)
    paths = SimpleNamespace(
        input=SimpleNamespace(echointegration=zarr),
        interm=SimpleNamespace(rois=csv),
        output=SimpleNamespace(reporting=tmp_path),
    )
    config = ExtractConfig(
        mode="2D",
        modes={"2D": ModeConfig(depth_slice=slice(None), channels=[0, 1, 2])},
        kernel=np.ones((3, 3)),
        model="kmeans",
        kmeans={"n_clusters": 2, "n_init": 2, "random_state": 0},
        min_share=0.5,
        overrides={},
    )
    monkeypatch.setattr(pipeline.PathsConfig, "from_yml", classmethod(lambda cls, path: paths))
    monkeypatch.setattr(pipeline.ExtractConfig, "from_yml", classmethod(lambda cls, path: config))
    base_args = ["extract", "--config", "unused.yml", "--savepath", "reports"]
    monkeypatch.setattr(sys, "argv", [*base_args, "--roi", "1"])
    pipeline.main()
    output = tmp_path / "reports" / "2D"
    results = ExtractionResults.load(output / "extraction.joblib")
    review = ReviewDecisions.load(output / "review.json")
    assert review.decisions[0].status == "pending"
    assert list((output / "review-figures" / results.extraction_id).glob("*.svg"))
    assert set(pd.read_csv(output / "review-summary.csv")["status"]) == {"pending", "failed"}

    config.overrides = {1: 0}
    monkeypatch.setattr(
        sys, "argv",
        ["extract", "--config", "unused.yml", "--savepath", "subset", "--override", "--no-plots"],
    )
    pipeline.main()
    subset_dir = tmp_path / "subset" / "2D"
    subset = ExtractionResults.load(subset_dir / "extraction.joblib")
    assert set(subset.candidates) == {1}
    assert not subset.failures
    assert ReviewDecisions.load(subset_dir / "review.json").decisions == (ReviewDecision(1, 0, "pending"),)
    config.overrides = {}

    monkeypatch.setattr(
        sys, "argv", ["extract", "--resume", "--review-dir", str(output), "--apply", "--no-plots"]
    )
    with pytest.raises(ValueError, match="pending"):
        pipeline.main()
    assert not list(output.glob("echotype-means-*.csv"))

    candidate = results.candidates[1]
    review.set(ReviewDecision(1, candidate.proposed_cluster, "accepted"))
    review.save(output / "review.json")
    monkeypatch.setattr(
        pipeline, "extract_echotypes",
        lambda *args, **kwargs: pytest.fail("Resume must not rerun extraction"),
    )
    monkeypatch.setattr(
        pipeline.PathsConfig, "from_yml",
        classmethod(lambda cls, path: pytest.fail("Resume must not access input paths")),
    )
    monkeypatch.setattr(
        sys, "argv", ["extract", "--resume", "--review-dir", str(output), "--apply", "--no-plots"]
    )
    pipeline.main()
    outputs = list(output.glob("echotype-means-*.csv"))
    assert len(outputs) == 1
    table = pd.read_csv(outputs[0])
    assert table.loc[0, "roi_id"] == 1
    assert set(table.columns) == {
        "roi_id", "cluster_id", "70000.0 - 38000.0", "120000.0 - 38000.0",
    }
