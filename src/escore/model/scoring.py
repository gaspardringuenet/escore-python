import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from ..review.outcome import EchotypeCollection
from .classifier import EscoreClassifier, _labels_for_distances


def score_echotypes(model: EscoreClassifier, echotypes: EchotypeCollection) -> pd.DataFrame:
    """Classify the raw pixels of each echotype and report the class fractions.

    Only pixels with finite ΔMVBS for every channel are valid. Fractions are
    relative to the valid pixels of each echotype.

    Returns
    -------
    DataFrame
        One row per echotype, indexed by ``(roi_id, cluster_id)``, with columns
        ``valid_pixels``, ``class_0`` ... ``class_{n_classes - 1}`` and
        ``unclassified``.

    Raises
    ------
    ValueError
        If the echotype features do not match the features the model was fitted on.
    """
    model._check_fitted()
    if tuple(echotypes.feature_names) != tuple(model.feature_names_in_):
        raise ValueError(
            f"Echotype features {echotypes.feature_names} do not match the model features "
            f"{tuple(model.feature_names_in_)}."
        )
    rows = []
    for item in echotypes.items:
        _check_channels(model, item.raw_sv, item.roi_id)
        pixels = pd.DataFrame(_delta_pixels(item.raw_sv), columns=list(model.feature_names_in_))
        counts = np.bincount(model.predict(pixels) + 1, minlength=model.n_classes + 1)
        valid = len(pixels)
        fractions = counts / valid if valid else np.full(model.n_classes + 1, np.nan)
        rows.append(
            {
                "valid_pixels": valid,
                **{f"class_{k}": fractions[k + 1] for k in range(model.n_classes)},
                "unclassified": fractions[0],
            }
        )
    index = pd.MultiIndex.from_tuples(
        [(item.roi_id, item.cluster_id) for item in echotypes.items],
        names=["roi_id", "cluster_id"],
    )
    columns = [
        "valid_pixels",
        *(f"class_{k}" for k in range(model.n_classes)),
        "unclassified",
    ]
    return pd.DataFrame(rows, index=index, columns=columns)


def _delta_pixels(raw_sv: xr.DataArray) -> NDArray[np.float64]:
    values = raw_sv.transpose("ping_time", "depth", "channel").values
    values = np.asarray(values).reshape(-1, raw_sv.sizes["channel"])
    values = values[np.isfinite(values).all(axis=1)]
    return values[:, 1:] - values[:, :1]


def _check_channels(model: EscoreClassifier, sv: xr.DataArray, roi_id: int) -> None:
    if "channel" not in sv.indexes:
        raise ValueError(f"ROI {roi_id} has no channel coordinates.")
    channels = tuple(sv.channel.values)
    if model.channels_ is not None and channels != model.channels_:
        raise ValueError(f"ROI {roi_id} channel order does not match the model.")
    names = tuple(f"{channel} - {channels[0]}" for channel in channels[1:])
    if names != model.feature_names_in_:
        raise ValueError(f"ROI {roi_id} channels do not match model features.")


def sensitivity(
    model: EscoreClassifier,
    echotypes: EchotypeCollection,
    expected: pd.Series,
    thresholds: list[float] | tuple[float, ...],
    *,
    weights: dict[tuple[int, int], xr.DataArray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep global thresholds; return per-echotype and per-class agreement.

    ``expected`` maps each (roi_id, cluster_id) to its reference class. Optional
    weights are aligned per-cell acoustic energy (e.g. sA in a chosen frequency).
    Fractions are over finite Sv pixels; zero-total weights yield NaN. Expected
    classes must be obtained independently for a held-out evaluation.
    """
    model._check_fitted()
    if tuple(echotypes.feature_names) != tuple(model.feature_names_in_):
        raise ValueError("Echotype features do not match model features.")
    if not expected.index.is_unique:
        raise ValueError("Expected-class index must be unique.")
    values = np.asarray(thresholds, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("thresholds must be a non-empty sequence of positive finite values.")
    rows = []
    for item in echotypes.items:
        key = (item.roi_id, item.cluster_id)
        if key not in expected.index:
            raise ValueError(f"No expected class for echotype {key}.")
        target = expected.loc[key]
        if isinstance(target, (bool, np.bool_)) or not isinstance(target, (int, np.integer)):
            raise ValueError(f"Expected class for {key} must be an integer.")
        if not 0 <= target < model.n_classes:
            raise ValueError(f"Expected class for {key} is outside the fitted classes.")
        sv = item.raw_sv.transpose("ping_time", "depth", "channel")
        _check_channels(model, sv, item.roi_id)
        raw = np.asarray(sv.values).reshape(-1, sv.sizes["channel"])
        valid = np.isfinite(raw).all(axis=1)
        delta = raw[valid, 1:] - raw[valid, :1]
        distances = model.transform(pd.DataFrame(delta, columns=model.feature_names_in_))
        weight = None
        if weights is not None:
            if key not in weights:
                raise ValueError(f"No weights for echotype {key}.")
            template = sv.isel(channel=0, drop=True)
            aligned, _ = xr.align(weights[key], template, join="exact")
            weight = np.asarray(aligned.transpose("ping_time", "depth").values).reshape(-1)[valid]
            if not np.isfinite(weight).all() or np.any(weight < 0):
                raise ValueError(
                    f"Weights for valid pixels in {key} must be finite and non-negative."
                )
        for threshold in values:
            labels = _labels_for_distances(distances, threshold)
            masks = (labels == target, (labels != target) & (labels >= 0), labels == -1)
            row = {
                "roi_id": key[0],
                "cluster_id": key[1],
                "expected_class": int(target),
                "threshold": float(threshold),
                "valid_pixels": int(valid.sum()),
            }
            for name, mask in zip(("correct", "incorrect", "unclassified"), masks):
                row[name + "_count"] = int(mask.sum())
                if weight is not None:
                    row[name + "_weight"] = float(weight[mask].sum())
            if weight is not None:
                row["total_weight"] = float(weight.sum())
            rows.append(row)
    columns = [
        "roi_id", "cluster_id", "expected_class", "threshold", "valid_pixels",
        "correct_count", "incorrect_count", "unclassified_count",
    ]
    if weights is not None:
        columns += ["correct_weight", "incorrect_weight", "unclassified_weight", "total_weight"]
    detail = pd.DataFrame(rows, columns=columns)
    group = detail.groupby(["threshold", "expected_class"], sort=True)[columns[4:]].sum()
    for frame in (detail, group):
        for name in ("correct", "incorrect", "unclassified"):
            frame[name + "_fraction"] = (
                frame[name + "_count"] / frame["valid_pixels"].replace(0, np.nan)
            )
            if weights is not None:
                frame[name + "_weight_fraction"] = (
                    frame[name + "_weight"] / frame["total_weight"].replace(0, np.nan)
                )
    return detail.set_index(["threshold", "roi_id", "cluster_id"]), group
