from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from .artifact import LinearFeatureTransform, LinearLogitArtifact, build_artifact
from .dataset import SnapshotRow, build_dataset_manifest, read_jsonl


def fit_logistic_residual(
    rows: Sequence[SnapshotRow],
    *,
    model_version: str,
    training_cutoff_ms: int,
    iterations: int = 500,
    learning_rate: float = 0.05,
    l2: float = 0.01,
) -> LinearLogitArtifact:
    """Fit a small residual-logit candidate using market-balanced gradient descent.

    This function creates a research candidate only. It does not promote the artifact or give it
    trading authority.
    """
    if not rows:
        raise ValueError("training rows must not be empty")
    if iterations < 1 or learning_rate <= 0.0 or l2 < 0.0:
        raise ValueError("invalid optimizer configuration")
    for row in rows:
        row.validate()
        if row.task != "settlement_probability_residual":
            raise ValueError("residual trainer only accepts settlement_probability_residual labels")
    schema_versions = {row.schema_version for row in rows}
    feature_orders = {row.feature_names for row in rows}
    if len(schema_versions) != 1 or len(feature_orders) != 1:
        raise ValueError("training rows must share one feature schema and order")
    feature_names = next(iter(feature_orders))
    manifest = build_dataset_manifest(rows, training_cutoff_ms=training_cutoff_ms)
    weights = _market_balanced_weights(rows)
    weight_sum = sum(weights)

    means = []
    scales = []
    for feature_index in range(len(feature_names)):
        mean = sum(
            weight * row.feature_values[feature_index]
            for row, weight in zip(rows, weights)
        ) / weight_sum
        variance = sum(
            weight * (row.feature_values[feature_index] - mean) ** 2
            for row, weight in zip(rows, weights)
        ) / weight_sum
        means.append(mean)
        scales.append(max(math.sqrt(variance), 1.0e-12))

    coefficients = [0.0] * len(feature_names)
    intercept = 0.0
    for _ in range(iterations):
        intercept_gradient = 0.0
        coefficient_gradients = [0.0] * len(feature_names)
        for row, weight in zip(rows, weights):
            standardized = [
                (value - mean) / scale
                for value, mean, scale in zip(row.feature_values, means, scales)
            ]
            residual = intercept + sum(
                coefficient * value
                for coefficient, value in zip(coefficients, standardized)
            )
            probability = _sigmoid(_logit(row.prior_probability) + residual)
            error = probability - row.label
            intercept_gradient += weight * error
            for index, value in enumerate(standardized):
                coefficient_gradients[index] += weight * error * value
        intercept -= learning_rate * intercept_gradient / weight_sum
        for index in range(len(coefficients)):
            gradient = coefficient_gradients[index] / weight_sum + l2 * coefficients[index]
            coefficients[index] -= learning_rate * gradient

    return build_artifact(
        model_version=model_version,
        task="settlement_probability_residual",
        feature_schema_version=next(iter(schema_versions)),
        features=(
            LinearFeatureTransform(name, mean, scale, coefficient)
            for name, mean, scale, coefficient in zip(
                feature_names, means, scales, coefficients
            )
        ),
        intercept=intercept,
        dataset_manifest_sha256=manifest.manifest_sha256,
        trained_through_ms=training_cutoff_ms,
    )


def _market_balanced_weights(rows: Sequence[SnapshotRow]) -> list[float]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.market_id] = counts.get(row.market_id, 0) + 1
    return [1.0 / counts[row.market_id] for row in rows]


def _logit(probability: float) -> float:
    bounded = min(max(probability, 1.0e-9), 1.0 - 1.0e-9)
    return math.log(bounded / (1.0 - bounded))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a shadow-only residual logistic candidate")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--training-cutoff-ms", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()
    artifact = fit_logistic_residual(
        read_jsonl(args.dataset),
        model_version=args.model_version,
        training_cutoff_ms=args.training_cutoff_ms,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        l2=args.l2,
    )
    args.output.write_text(artifact.to_json() + "\n", encoding="utf-8")
    print(json.dumps({"artifact_sha256": artifact.artifact_sha256, "stage": "candidate"}))


if __name__ == "__main__":
    main()
