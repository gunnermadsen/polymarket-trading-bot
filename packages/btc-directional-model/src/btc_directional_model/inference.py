from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .evaluation import sigmoid


def artifact_probability(
    artifact: Mapping[str, Any], features: Mapping[str, float | None] | Sequence[float | None]
) -> float:
    names = artifact["feature_names"]
    if isinstance(features, Mapping):
        missing = [name for name in names if name not in features]
        if missing:
            raise ValueError(f"missing artifact features: {', '.join(missing)}")
        values = [features[name] for name in names]
    else:
        values = list(features)
        if len(values) != len(names):
            raise ValueError(f"expected {len(names)} features, received {len(values)}")

    matrix = np.asarray(
        [np.nan if value is None else float(value) for value in values], dtype=np.float64
    )
    medians = np.asarray(artifact["imputation_medians"], dtype=np.float64)
    means = np.asarray(artifact["standardization_means"], dtype=np.float64)
    scales = np.asarray(artifact["standardization_scales"], dtype=np.float64)
    coefficients = np.asarray(artifact["coefficients"], dtype=np.float64)
    filled = np.where(np.isfinite(matrix), matrix, medians)
    standardized = (filled - means) / scales
    logit = float(standardized @ coefficients + artifact["intercept"])
    calibrated_logit = logit * float(artifact["calibration_slope"]) + float(
        artifact["calibration_intercept"]
    )
    return float(sigmoid(np.asarray([calibrated_logit]))[0])


def artifact_prediction(
    artifact: Mapping[str, Any], features: Mapping[str, float | None] | Sequence[float | None]
) -> dict[str, float | int | bool]:
    probability_up = artifact_probability(artifact, features)
    predicted_up = int(probability_up >= 0.5)
    confidence = max(probability_up, 1 - probability_up)
    return {
        "probability_up": probability_up,
        "predicted_up": predicted_up,
        "confidence": confidence,
        "accepted": confidence >= float(artifact["confidence_threshold"]),
    }
