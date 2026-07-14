from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .artifact import LinearLogitArtifact
from .dataset import SnapshotRow


@dataclass(frozen=True)
class ProbabilityMetrics:
    row_count: int
    market_count: int
    brier: float
    log_loss: float
    prior_brier: float
    prior_log_loss: float


def score_probability(artifact: LinearLogitArtifact, row: SnapshotRow) -> float:
    artifact.validate()
    row.validate()
    if artifact.feature_schema_version != row.schema_version:
        raise ValueError("artifact and row schema versions differ")
    artifact_names = tuple(feature.name for feature in artifact.features)
    if artifact_names != row.feature_names:
        raise ValueError("artifact and row feature order differ")
    residual = artifact.intercept + sum(
        feature.coefficient * (value - feature.mean) / feature.scale
        for feature, value in zip(artifact.features, row.feature_values)
    )
    return _sigmoid(_logit(row.prior_probability) + residual)


def evaluate_probability_artifact(
    artifact: LinearLogitArtifact, rows: Sequence[SnapshotRow]
) -> ProbabilityMetrics:
    if not rows:
        raise ValueError("evaluation rows must not be empty")
    predictions = [score_probability(artifact, row) for row in rows]
    labels = [row.label for row in rows]
    priors = [row.prior_probability for row in rows]
    return ProbabilityMetrics(
        row_count=len(rows),
        market_count=len({row.market_id for row in rows}),
        brier=_brier(predictions, labels),
        log_loss=_log_loss(predictions, labels),
        prior_brier=_brier(priors, labels),
        prior_log_loss=_log_loss(priors, labels),
    )


def _brier(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    return sum((probability - label) ** 2 for probability, label in zip(probabilities, labels)) / len(
        labels
    )


def _log_loss(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    losses = []
    for probability, label in zip(probabilities, labels):
        bounded = min(max(probability, 1.0e-12), 1.0 - 1.0e-12)
        losses.append(-(label * math.log(bounded) + (1 - label) * math.log(1.0 - bounded)))
    return sum(losses) / len(losses)


def _logit(probability: float) -> float:
    bounded = min(max(probability, 1.0e-9), 1.0 - 1.0e-9)
    return math.log(bounded / (1.0 - bounded))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)

