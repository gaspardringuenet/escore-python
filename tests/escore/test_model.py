import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

from escore import (
    Echotype, EchotypeCollection, EscoreClassifier, ReviewDecision, ReviewDecisions,
    WardExploration, classify_sv, sensitivity,
)


@pytest.fixture
def means():
    return pd.DataFrame(
        [[-11, -1], [-10, 0], [-9, -1], [-10, 1],
         [9, 0], [10, 1], [11, 0], [10, -1]],
        columns=["f1 - ref", "f2 - ref"],
        index=pd.MultiIndex.from_tuples([(i, 0) for i in range(8)], names=["roi_id", "cluster_id"]),
    )


@pytest.fixture
def model(means):
    return EscoreClassifier(n_classes=2).fit(means, channels=("ref", "f1", "f2"))


def test_ward_exploration_independent_of_covariance(means):
    explore = WardExploration(means)
    labels = explore.labels(2)
    assert len(set(labels[:4])) == 1
    assert len(set(labels[4:])) == 1
    assert labels[0] != labels[4]
    scores = explore.scores([1, 2, 8])
    assert scores.loc[2, "silhouette"] > 0.8
    assert "require" in scores.loc[1, "error"]
    assert "require" in scores.loc[8, "error"]
    fitted = EscoreClassifier(n_classes=2).fit(means)
    assert adjusted_rand_score(labels, fitted.labels_) == 1


def test_mahalanobis_and_roundtrip(model, means, tmp_path):
    x = means.iloc[:2]
    distances = model.transform(x)
    for k in range(2):
        diff = x.to_numpy() - model.means_[k]
        expected = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(model.covariances_[k]), diff)
        assert_allclose(distances[:, k], expected)
        eigenvalues, eigenvectors = np.linalg.eigh(model.covariances_[k])
        assert_allclose(distances[:, k], ((diff @ eigenvectors) ** 2 / eigenvalues).sum(axis=1))
    model.save(tmp_path / "model.joblib")
    restored = EscoreClassifier.load(tmp_path / "model.joblib")
    assert_array_equal(restored.predict(x), model.predict(x))
    assert restored.channels_ == ("ref", "f1", "f2")
    with pytest.raises(ValueError, match="columns"):
        model.transform(x.iloc[:, ::-1])


def test_class_thresholds_and_boundary(model):
    model.max_squared_distance = (1.0, 3.0)
    assert_array_equal(model._classify_distances(np.array([[1.0, 2.0], [1.0, 3.0]])), [1, -1])


def test_covariance_rejects_collinear_clusters():
    data = pd.DataFrame({"a": [0., 1., 2., 3.], "b": [0., 1., 2., 3.]})
    assert WardExploration(data).scores([2]).loc[2, "silhouette"] > 0
    with pytest.raises(ValueError, match="covariance"):
        EscoreClassifier(n_classes=1).fit(data)


def test_classify_sv_chunked_and_masks(model, tmp_path):
    sv = xr.DataArray(
        [[[-60, -70, -60], [-60, -50, -60], [np.nan, -60, -60], [-100, -100, -100]]],
        dims=("ping_time", "depth", "channel"),
        coords={"channel": ["ref", "f1", "f2"], "ping_time": [1], "depth": range(4)},
    )
    expected = classify_sv(model, sv, min_sv=-85)
    lazy = classify_sv(model, sv.chunk({"depth": 2, "channel": 1}), min_sv=-85)
    xr.testing.assert_equal(lazy.compute(), expected)
    assert_array_equal(expected.label.values[0, 2:], [-2, -3])
    assert np.isnan(expected.squared_distance.values[0, 2:]).all()
    path = tmp_path / "classification.nc"
    expected.to_netcdf(path)
    with xr.open_dataset(path) as restored:
        assert_array_equal(restored.label.values, expected.label.values)
        assert_allclose(restored.squared_distance.values, expected.squared_distance.values, equal_nan=True)
    with pytest.raises(ValueError, match="channels"):
        classify_sv(model, sv, channels=("ref", "f2", "f1"))


def test_classify_sv_reports_eligible_winner_distance(model):
    features = pd.DataFrame([[-9.5, 0.3]], columns=model.feature_names_in_)
    distances = model.transform(features)[0]
    nearest = int(np.argmin(distances))
    other = 1 - nearest
    thresholds = distances + 1
    thresholds[nearest] = distances[nearest] / 2
    model.max_squared_distance = tuple(thresholds)
    sv = xr.DataArray([[-60.0, -69.5, -59.7]], dims=("depth", "channel"),
                      coords={"channel": ["ref", "f1", "f2"], "depth": [0]})
    result = classify_sv(model, sv)
    assert result.label.item() == other
    assert result.best_squared_distance.item() == pytest.approx(distances[nearest])
    assert result.assigned_squared_distance.item() == pytest.approx(distances[other])


def test_sensitivity_counts_and_weights(model, means):
    from escore.model import plot_sensitivity
    sv = xr.DataArray(
        [[[-60, -70, -60], [-60, -50, -60], [np.nan, -60, -60]]],
        dims=("ping_time", "depth", "channel"),
        coords={"channel": ["ref", "f1", "f2"], "ping_time": [0], "depth": [0, 1, 2]},
    )
    item = Echotype(0, 0, sv, None, means.iloc[0], ReviewDecision(0, 0, "accepted"), pd.Series())
    collection = EchotypeCollection((item,), tuple(means.columns))
    target = int(model.predict(means.iloc[[0]])[0])
    expected = pd.Series([target], index=pd.MultiIndex.from_tuples([(0, 0)]))
    weights = {(0, 0): xr.DataArray([[1.0, 3.0, 0.0]], dims=("ping_time", "depth"),
                                   coords={"ping_time": [0], "depth": [0, 1, 2]})}
    detail, group = sensitivity(model, collection, expected, [1000], weights=weights)
    assert detail.iloc[0].valid_pixels == 2
    assert detail.iloc[0].correct_fraction == 0.5
    assert detail.iloc[0].correct_weight_fraction == 0.25
    assert group.iloc[0].incorrect_weight_fraction == 0.75
    fig = plot_sensitivity(group, weighted=True)
    fig.canvas.draw()
    plt.close(fig)
    zeros = {(0, 0): xr.zeros_like(weights[(0, 0)])}
    detail, _ = sensitivity(model, collection, expected, [1], weights=zeros)
    assert np.isnan(detail.iloc[0].correct_weight_fraction)
    wrong = Echotype(0, 0, sv.sel(channel=["ref", "f2", "f1"]), None, means.iloc[0],
                     ReviewDecision(0, 0, "accepted"), pd.Series())
    with pytest.raises(ValueError, match="channel"):
        sensitivity(model, EchotypeCollection((wrong,), tuple(means.columns)), expected, [1])


def test_multiple_pending_clusters_can_be_accepted():
    decisions = ReviewDecisions("test", (ReviewDecision(1, 0, "pending"),))
    decisions.select(1, 1, append=True)
    decisions.accept_roi(1)
    assert [(d.cluster_id, d.status) for d in decisions.decisions] == [(0, "accepted"), (1, "accepted")]


def test_model_plots_and_depth_summary(model, means):
    from escore.model import depth_summary, plot_classes, plot_classification, plot_depth_summary, plot_responses

    fig = plot_classes(model, means)
    fig.canvas.draw()
    plt.close(fig)
    fig = plot_responses(model)
    fig.canvas.draw()
    plt.close(fig)
    labels = xr.DataArray([[0, -1], [1, -3]], dims=("ping_time", "depth"),
                          coords={"ping_time": [0, 1], "depth": [1., 2.]})
    period = xr.DataArray(["day", "night"], dims=("ping_time",), coords={"ping_time": [0, 1]})
    summary = depth_summary(labels, [0, 1.5, 3], period=period)
    assert summary.cells.sum() == 4
    fig = plot_depth_summary(summary)
    fig.canvas.draw()
    plt.close(fig)
    ax = plot_classification(labels)
    plt.close(ax.figure)


def test_three_dimensional_ellipsoid():
    from escore.model import plot_classes

    data = pd.DataFrame(
        [[-2, 0, 0], [0, -2, 0], [0, 0, -2], [1, 1, 1], [1, 0, 0]],
        columns=["a", "b", "c"],
    )
    model = EscoreClassifier(n_classes=1).fit(data)
    fig = plot_classes(model, data, features=(0, 1, 2))
    fig.canvas.draw()
    plt.close(fig)
