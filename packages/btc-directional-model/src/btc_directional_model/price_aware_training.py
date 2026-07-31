from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from .continuous_context_features import CONTINUOUS_CONTEXT_MODEL_FEATURES
from .core_config import CoreTrainingConfig
from .core_features import (
    CORE_BOUNDARY_ENRICHED_FEATURES,
    CORE_ORACLE_FEATURES,
)
from .core_training import (
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    CandidateSpec,
    FittedCoreModel,
    ProbabilityCalibrator,
    fit_probability_calibrator,
    tune_and_fit_model,
)
from .offline_challengers import STRICT_BOOK_FIVE_SHARE_V2_FEATURES

BOUNDARY_CONTROL_CANDIDATE = "retrained_boundary_feature_control"
CORE_ORACLE_OUTCOME_CANDIDATE = "boundary_oracle_outcome"
CONTEXT_OUTCOME_CANDIDATE = "boundary_oracle_continuous_context_outcome"
ACCURACY_ANCHORED_OUTCOME_CANDIDATE = CONTEXT_OUTCOME_CANDIDATE
ACCURACY_ANCHORED_ADMISSION_CANDIDATE = "accuracy_anchored_admission"

BOUNDARY_CONTROL_FEATURES = tuple(CORE_BOUNDARY_ENRICHED_FEATURES)
CORE_ORACLE_OUTCOME_FEATURES = tuple(
    dict.fromkeys(
        (
            *CORE_BOUNDARY_ENRICHED_FEATURES,
            *CORE_ORACLE_FEATURES,
        )
    )
)
CONTEXT_OUTCOME_FEATURES = tuple(
    dict.fromkeys((*CORE_ORACLE_OUTCOME_FEATURES, *CONTINUOUS_CONTEXT_MODEL_FEATURES))
)
OUTCOME_FEATURES = CONTEXT_OUTCOME_FEATURES
PRICE_AWARE_FEATURES = tuple(
    dict.fromkeys(
        (
            *CORE_ORACLE_OUTCOME_FEATURES,
            *CONTINUOUS_CONTEXT_MODEL_FEATURES,
            *STRICT_BOOK_FIVE_SHARE_V2_FEATURES,
        )
    )
)
OUTCOME_DERIVED_FEATURES = (
    "outcome_probability_up",
    "outcome_predicted_up",
    "outcome_probability_selected",
    "outcome_confidence_margin",
    "outcome_selected_ask_vwap_5",
    "outcome_selected_entry_debit_per_share",
    "outcome_raw_net_edge_per_share",
)
ADMISSION_FEATURES = tuple(
    dict.fromkeys(
        (
            *PRICE_AWARE_FEATURES,
            *OUTCOME_DERIVED_FEATURES,
        )
    )
)
OUTCOME_DIRECTION_CORRECT_COLUMN = "outcome_direction_correct"
OUTCOME_SIGNAL_BLOCK_COLUMN = "outcome_signal_block"


@dataclass
class AdmissionBundle:
    model: FittedCoreModel
    calibrator: ProbabilityCalibrator
    candidate_spec: CandidateSpec

    @property
    def name(self) -> str:
        return self.model.candidate_name

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        """Return calibrated P(the locked outcome direction is correct)."""

        return self.calibrator.probability(self.model.raw_logit(frame))


def price_aware_classifier_spec(
    name: str,
    *,
    features: tuple[str, ...],
    recency_half_life_days: float | None,
) -> CandidateSpec:
    return CandidateSpec(
        name=name,
        family="histogram",
        feature_names=features,
        row_weight_policy=MARKET_EQUAL_ROW_WEIGHT_POLICY,
        recency_half_life_days=recency_half_life_days,
    )


def attach_five_share_economic_targets(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "label_up",
        "fee_rate",
        "up_ask_vwap_5",
        "down_ask_vwap_5",
        "strict_both_side_eligible",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("economic target frame is missing columns: " + ", ".join(missing))
    invalid = frame.filter(
        ~pl.col("strict_both_side_eligible").fill_null(False)
        | pl.col("fee_rate").is_null()
        | ~pl.col("fee_rate").is_finite()
        | (pl.col("fee_rate") <= 0)
        | pl.any_horizontal(
            [
                pl.col(column).is_null()
                | ~pl.col(column).is_finite()
                | (pl.col(column) <= 0)
                | (pl.col(column) > 1)
                for column in ("up_ask_vwap_5", "down_ask_vwap_5")
            ]
        )
    )
    if invalid.height:
        raise RuntimeError("price-aware targets require known fees and strict positive VWAP5")
    with_costs = frame.with_columns(
        (pl.col("fee_rate") * pl.col("up_ask_vwap_5") * (1.0 - pl.col("up_ask_vwap_5"))).alias(
            "up_fee_per_share"
        ),
        (pl.col("fee_rate") * pl.col("down_ask_vwap_5") * (1.0 - pl.col("down_ask_vwap_5"))).alias(
            "down_fee_per_share"
        ),
    ).with_columns(
        (pl.col("up_ask_vwap_5") + pl.col("up_fee_per_share")).alias("up_entry_debit_per_share"),
        (pl.col("down_ask_vwap_5") + pl.col("down_fee_per_share")).alias(
            "down_entry_debit_per_share"
        ),
    )
    return with_costs.with_columns(
        (pl.col("label_up").cast(pl.Float64) - pl.col("up_entry_debit_per_share")).alias(
            "realized_up_net_per_share"
        ),
        (1.0 - pl.col("label_up").cast(pl.Float64) - pl.col("down_entry_debit_per_share")).alias(
            "realized_down_net_per_share"
        ),
    )


def attach_outcome_signals(
    frame: pl.DataFrame,
    probability_up: np.ndarray,
    *,
    block_name: str,
) -> pl.DataFrame:
    """Attach causal outcome signals and the direction-correct admission target.

    ``probability_up`` must have been produced without using labels from
    ``block_name``.  The named block is persisted as lineage; it is deliberately
    excluded from the admission feature set.
    """

    if not block_name.strip():
        raise ValueError("outcome signal block name must be non-empty")
    required = {
        "label_up",
        "up_ask_vwap_5",
        "down_ask_vwap_5",
        "up_entry_debit_per_share",
        "down_entry_debit_per_share",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("outcome signal frame is missing columns: " + ", ".join(missing))
    label_up = _validated_binary_column(frame, "label_up")
    probability = _validated_probability_vector(probability_up, frame.height)
    predicted_up = probability >= 0.5
    probability_selected = np.where(predicted_up, probability, 1.0 - probability)
    selected_vwap = np.where(
        predicted_up,
        frame["up_ask_vwap_5"].to_numpy(),
        frame["down_ask_vwap_5"].to_numpy(),
    )
    selected_debit = np.where(
        predicted_up,
        frame["up_entry_debit_per_share"].to_numpy(),
        frame["down_entry_debit_per_share"].to_numpy(),
    )
    return frame.with_columns(
        pl.Series("outcome_probability_up", probability),
        pl.Series("outcome_predicted_up", predicted_up.astype(np.int8)),
        pl.Series("outcome_probability_selected", probability_selected),
        pl.Series("outcome_confidence_margin", probability_selected - 0.5),
        pl.Series("outcome_selected_ask_vwap_5", selected_vwap),
        pl.Series("outcome_selected_entry_debit_per_share", selected_debit),
        pl.Series(
            "outcome_raw_net_edge_per_share",
            probability_selected - selected_debit,
        ),
        pl.Series(OUTCOME_DIRECTION_CORRECT_COLUMN, predicted_up == label_up.astype(bool)),
        pl.lit(block_name).alias(OUTCOME_SIGNAL_BLOCK_COLUMN),
    )


def admission_fit_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Return an isolated training view with direction correctness as label_up."""

    required = {
        OUTCOME_DIRECTION_CORRECT_COLUMN,
        OUTCOME_SIGNAL_BLOCK_COLUMN,
        *ADMISSION_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("admission frame is missing columns: " + ", ".join(missing))
    target = _validated_binary_column(frame, OUTCOME_DIRECTION_CORRECT_COLUMN)
    return frame.with_columns(pl.Series("label_up", target))


def fit_admission_bundle(
    fit_frame: pl.DataFrame,
    calibration_frame: pl.DataFrame,
    *,
    core_config: CoreTrainingConfig,
    recency_half_life_days: float | None,
) -> tuple[AdmissionBundle, dict[str, Any]]:
    """Fit and calibrate the direction-correct admission classifier."""

    spec = price_aware_classifier_spec(
        ACCURACY_ANCHORED_ADMISSION_CANDIDATE,
        features=ADMISSION_FEATURES,
        recency_half_life_days=recency_half_life_days,
    )
    isolated_fit = admission_fit_frame(fit_frame)
    isolated_calibration = admission_fit_frame(calibration_frame)
    model, tuning = tune_and_fit_model(isolated_fit, spec, core_config)
    calibrator = fit_probability_calibrator(
        model,
        isolated_calibration,
        core_config,
        spec,
    )
    return AdmissionBundle(model, calibrator, spec), tuning


def score_probability_actions(
    frame: pl.DataFrame,
    probability_up: np.ndarray,
    *,
    candidate: str,
    select_by_value: bool,
) -> pl.DataFrame:
    if select_by_value:
        raise ValueError(
            "value-based direction selection is retired; outcome direction must stay locked"
        )
    probability = _validated_probability_vector(probability_up, frame.height)
    up_cost = frame["up_entry_debit_per_share"].to_numpy()
    down_cost = frame["down_entry_debit_per_share"].to_numpy()
    up_value = probability - up_cost
    down_value = (1.0 - probability) - down_cost
    predicted_up = probability >= 0.5
    economic_score = np.where(predicted_up, up_value, down_value)
    selected_value = np.where(predicted_up, up_value, down_value)
    return _scored_action_frame(
        frame,
        candidate=candidate,
        predicted_up=predicted_up,
        action_probability_up=probability,
        outcome_probability_up=probability,
        economic_score=economic_score,
        predicted_net_per_share=selected_value,
    )


def score_admission_actions(
    frame: pl.DataFrame,
    bundle: AdmissionBundle,
) -> pl.DataFrame:
    required = {
        "outcome_probability_up",
        "outcome_predicted_up",
        "outcome_selected_entry_debit_per_share",
        *ADMISSION_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("admission scoring frame is missing columns: " + ", ".join(missing))
    probability_correct = _validated_probability_vector(
        bundle.probability(frame),
        frame.height,
    )
    locked_direction = _validated_binary_column(
        frame,
        "outcome_predicted_up",
    ).astype(bool)
    outcome_probability_up = _validated_probability_vector(
        frame["outcome_probability_up"].to_numpy(),
        frame.height,
    )
    if not np.array_equal(locked_direction, outcome_probability_up >= 0.5):
        raise ValueError("outcome_predicted_up must match outcome_probability_up >= 0.5")
    selected_debit = frame["outcome_selected_entry_debit_per_share"].to_numpy()
    expected_net = probability_correct - selected_debit
    scored = _scored_action_frame(
        frame,
        candidate=bundle.name,
        predicted_up=locked_direction,
        action_probability_up=outcome_probability_up,
        outcome_probability_up=outcome_probability_up,
        economic_score=expected_net,
        predicted_net_per_share=expected_net,
    )
    return scored.with_columns(
        pl.Series("admission_probability_correct", probability_correct),
    )


def mark_first_economic_action(
    frame: pl.DataFrame,
    *,
    threshold: float,
    score_column: str = "economic_score",
) -> pl.DataFrame:
    if not math_is_finite_nonnegative(threshold):
        raise ValueError("economic action threshold must be finite and nonnegative")
    if score_column not in frame.columns:
        raise ValueError(f"missing economic action score: {score_column}")
    selected_keys = (
        frame.filter(pl.col(score_column) > threshold)
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
        .select("market_id", "observed_at", "seconds_elapsed")
        .with_columns(pl.lit(True).alias("policy_selected"))
    )
    return frame.join(
        selected_keys,
        on=["market_id", "observed_at", "seconds_elapsed"],
        how="left",
        validate="1:1",
    ).with_columns(pl.col("policy_selected").fill_null(False))


def mark_first_confident_action(
    frame: pl.DataFrame,
    *,
    confidence_threshold: float,
) -> pl.DataFrame:
    if not 0.5 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0.5, 1]")
    selected_keys = (
        frame.filter(pl.col("confidence") >= confidence_threshold)
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
        .select("market_id", "observed_at", "seconds_elapsed")
        .with_columns(pl.lit(True).alias("policy_selected"))
    )
    return frame.join(
        selected_keys,
        on=["market_id", "observed_at", "seconds_elapsed"],
        how="left",
        validate="1:1",
    ).with_columns(pl.col("policy_selected").fill_null(False))


def math_is_finite_nonnegative(value: float) -> bool:
    return bool(np.isfinite(value) and value >= 0)


def _validated_probability_vector(
    values: np.ndarray,
    expected_length: int,
) -> np.ndarray:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 1 or len(probability) != expected_length:
        raise ValueError("probability vector must be one-dimensional and match the frame")
    if not np.isfinite(probability).all():
        raise ValueError("probability vector must contain only finite values")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("probability vector must stay inside [0, 1]")
    return probability


def _validated_binary_column(frame: pl.DataFrame, column: str) -> np.ndarray:
    values = frame[column].cast(pl.Float64, strict=False).to_numpy()
    if (
        values.ndim != 1
        or len(values) != frame.height
        or not np.isfinite(values).all()
        or not np.isin(values, (0.0, 1.0)).all()
    ):
        raise ValueError(f"{column} must contain only non-null binary values")
    return values.astype(np.int8)


def _scored_action_frame(
    frame: pl.DataFrame,
    *,
    candidate: str,
    predicted_up: np.ndarray,
    action_probability_up: np.ndarray,
    outcome_probability_up: np.ndarray,
    economic_score: np.ndarray,
    predicted_net_per_share: np.ndarray,
) -> pl.DataFrame:
    predicted = np.asarray(predicted_up, dtype=bool)
    action_probability = np.asarray(action_probability_up, dtype=np.float64)
    label = frame["label_up"].to_numpy().astype(bool)
    selected_price = np.where(
        predicted,
        frame["up_ask_vwap_5"].to_numpy(),
        frame["down_ask_vwap_5"].to_numpy(),
    )
    selected_fee = np.where(
        predicted,
        frame["up_fee_per_share"].to_numpy(),
        frame["down_fee_per_share"].to_numpy(),
    )
    realized_net = np.where(
        predicted,
        frame["realized_up_net_per_share"].to_numpy(),
        frame["realized_down_net_per_share"].to_numpy(),
    )
    expressions: list[pl.Expr] = [
        pl.lit(candidate).alias("candidate"),
        pl.Series("predicted_up", predicted.astype(np.int8)),
        pl.Series("probability_up", action_probability),
        pl.Series(
            "confidence",
            np.maximum(action_probability, 1.0 - action_probability),
        ),
        pl.Series("correct", predicted == label),
        pl.Series("outcome_probability_up", outcome_probability_up),
        pl.Series("economic_score", economic_score),
        pl.Series("direct_net_edge_per_share", predicted_net_per_share),
        pl.Series("predicted_net_per_share", predicted_net_per_share),
        pl.Series("selected_ask_vwap_5", selected_price),
        pl.Series("selected_fee_per_share", selected_fee),
        pl.Series("realized_selected_net_per_share", realized_net),
        pl.lit(True).alias("model_eligible"),
    ]
    return frame.with_columns(*expressions)
