from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

BENCHMARK_SCHEMA_VERSION = "btc-directional-model-benchmark-v1"
FIXED_CHECKPOINTS = (60, 90, 120, 180, 240)
TIME_BANDS = (
    ("60-89", 60, 89),
    ("90-119", 90, 119),
    ("120-179", 120, 179),
    ("180-240", 180, 240),
)
REQUIRED_PREDICTION_COLUMNS = {
    "candidate",
    "market_id",
    "observed_at",
    "seconds_elapsed",
    "label_up",
    "predicted_up",
    "probability_up",
    "confidence",
    "correct",
}


@dataclass(frozen=True)
class CandidatePolicy:
    confidence_threshold: float | None
    deployment_compatible: bool
    native_p99_milliseconds: float | None = None
    runtime_model_bytes: int | None = None
    selection_mode: Literal[
        "fixed_confidence",
        "chronological_preselected",
        "time_band_preselected",
    ] = "fixed_confidence"
    confidence_threshold_min: float | None = None
    confidence_threshold_max: float | None = None

    def __post_init__(self) -> None:
        if self.selection_mode == "fixed_confidence":
            if (
                self.confidence_threshold is None
                or not 0.5 <= self.confidence_threshold <= 1.0
            ):
                raise ValueError(
                    "fixed confidence_threshold must be between 0.5 and 1.0"
                )
            if (
                self.confidence_threshold_min is not None
                or self.confidence_threshold_max is not None
            ):
                raise ValueError(
                    "fixed confidence policies cannot configure a threshold range"
                )
        elif self.selection_mode in {
            "chronological_preselected",
            "time_band_preselected",
        }:
            if self.confidence_threshold is not None:
                raise ValueError(
                    "preselected policies cannot use one fixed threshold"
                )
            if (
                self.confidence_threshold_min is None
                or self.confidence_threshold_max is None
                or not 0.5
                <= self.confidence_threshold_min
                <= self.confidence_threshold_max
                <= 1.0
            ):
                raise ValueError(
                    "preselected policies require a valid threshold range"
                )
        else:
            raise ValueError(f"unsupported policy selection_mode: {self.selection_mode}")
        if (
            self.native_p99_milliseconds is not None
            and self.native_p99_milliseconds <= 0
        ):
            raise ValueError("native_p99_milliseconds must be positive")
        if self.runtime_model_bytes is not None and self.runtime_model_bytes <= 0:
            raise ValueError("runtime_model_bytes must be positive")


@dataclass(frozen=True)
class BenchmarkEvidence:
    label: str
    kind: Literal["development", "holdout"]
    independent: bool

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("evidence label cannot be empty")
        if self.kind not in {"development", "holdout"}:
            raise ValueError("evidence kind must be development or holdout")
        if self.kind == "development" and self.independent:
            raise ValueError("development evidence cannot be marked independent")


@dataclass(frozen=True)
class AdvancementCriteria:
    minimum_accuracy: float = 0.0
    minimum_balanced_accuracy: float = 0.0
    minimum_direction_recall: float = 0.0
    minimum_wilson_lower_95: float = 0.0
    maximum_expected_calibration_error: float = 1.0
    minimum_coverage: float = 0.0
    minimum_coverage_uplift: float = 0.0
    maximum_accuracy_regression: float = 0.0
    maximum_balanced_accuracy_regression: float = 0.0
    maximum_direction_recall_regression: float = 0.0
    maximum_median_entry_seconds_regression: float = -1.0
    minimum_mean_direct_edge_per_share: float = 0.0
    minimum_realized_net_per_share: float = 0.0
    minimum_common_time_markets: int = 1
    maximum_native_p99_milliseconds: float = 1.0
    maximum_runtime_model_bytes: int = 32 * 1024 * 1024


def benchmark_predictions(
    candidate_frames: Mapping[str, pl.DataFrame] | Sequence[pl.DataFrame],
    *,
    policies: Mapping[str, CandidatePolicy],
    control_candidate: str,
    evidence: BenchmarkEvidence,
    eligible_market_ids: Sequence[str] | None = None,
    minimum_samples: int = 100,
    minimum_executable_samples: int | None = None,
    quantity: float = 5.0,
    criteria: AdvancementCriteria | None = None,
) -> dict[str, Any]:
    """Compare real model predictions without fitting or generating model output.

    Each input frame is point-in-time model output. The benchmark applies each
    candidate's own confidence policy, then separately compares unfiltered predictions
    on identical market timestamps at the fixed checkpoints.
    """

    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    if minimum_executable_samples is None:
        minimum_executable_samples = minimum_samples
    if minimum_executable_samples <= 0:
        raise ValueError("minimum_executable_samples must be positive")
    if not math.isclose(quantity, 5.0):
        raise ValueError("the benchmark contract requires a fixed quantity of 5 shares")
    if criteria is None:
        criteria = AdvancementCriteria()

    frames = _coerce_candidate_frames(candidate_frames)
    if control_candidate not in frames:
        raise ValueError(f"control candidate {control_candidate!r} is not present")
    missing_policies = sorted(set(frames) - set(policies))
    if missing_policies:
        raise ValueError(f"missing candidate policies: {', '.join(missing_policies)}")

    candidate_order = [control_candidate, *sorted(set(frames) - {control_candidate})]
    normalized = {
        name: _normalize_prediction_frame(frames[name], name) for name in candidate_order
    }
    _validate_cross_candidate_labels(normalized)

    if eligible_market_ids is None:
        universe = sorted(
            {
                market_id
                for frame in normalized.values()
                for market_id in frame["market_id"].to_list()
            }
        )
    else:
        universe = sorted({str(market_id) for market_id in eligible_market_ids})
    if not universe:
        raise ValueError("eligible market universe cannot be empty")
    observed_market_ids = {
        market_id
        for frame in normalized.values()
        for market_id in frame["market_id"].to_list()
    }
    outside_universe = sorted(observed_market_ids - set(universe))
    if outside_universe:
        preview = ", ".join(outside_universe[:5])
        raise ValueError(f"prediction markets fall outside the eligible universe: {preview}")

    candidates: dict[str, Any] = {}
    for name in candidate_order:
        policy = policies[name]
        frame = normalized[name]
        selected = _select_policy_rows(frame, policy)
        own_policy = _own_policy_metrics(
            frame,
            selected,
            eligible_market_ids=universe,
            quantity=quantity,
        )
        candidates[name] = {
            "policy": asdict(policy),
            "own_policy": own_policy,
            "time_bands": _time_band_metrics(selected, eligible_markets=len(universe)),
            "checkpoints": _checkpoint_metrics(
                frame,
                eligible_markets=len(universe),
            ),
        }

    control_metrics = candidates[control_candidate]["own_policy"]
    for name in candidate_order:
        if name == control_candidate:
            candidates[name]["advance"] = {
                "is_control": True,
                "benchmark_passed": None,
                "deployment_qualified": None,
                "checks": [],
            }
            continue
        candidates[name]["advance"] = _advance_checks(
            candidates[name]["own_policy"],
            control_metrics,
            policy=policies[name],
            minimum_samples=minimum_samples,
            minimum_executable_samples=minimum_executable_samples,
            evidence=evidence,
            criteria=criteria,
            quantity=quantity,
        )

    common_comparisons = {
        name: _common_checkpoint_comparison(
            normalized[control_candidate],
            normalized[name],
            control_name=control_candidate,
            candidate_name=name,
        )
        for name in candidate_order
        if name != control_candidate
    }
    for name, comparison in common_comparisons.items():
        common_counts = [
            int(row["common_markets"]) for row in comparison["checkpoints"]
        ]
        minimum_common = min(common_counts) if common_counts else 0
        advance = candidates[name]["advance"]
        advance["checks"].append(
            _check(
                "minimum common exact-time samples",
                minimum_common,
                ">=",
                criteria.minimum_common_time_markets,
                minimum_common >= criteria.minimum_common_time_markets,
            )
        )
        advance["benchmark_passed"] = all(
            check["passed"] for check in advance["checks"]
        )
        advance["deployment_qualified"] = bool(
            advance["benchmark_passed"]
            and advance["evidence_allows_deployment_qualification"]
        )
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "evaluation": {
            **asdict(evidence),
            "development_only": not (
                evidence.kind == "holdout" and evidence.independent
            ),
        },
        "control_candidate": control_candidate,
        "quantity": quantity,
        "eligible_markets": len(universe),
        "minimum_samples": minimum_samples,
        "minimum_executable_samples": minimum_executable_samples,
        "advancement_criteria": asdict(criteria),
        "fixed_checkpoints": list(FIXED_CHECKPOINTS),
        "time_bands": [
            {"name": name, "start": start, "end": end}
            for name, start, end in TIME_BANDS
        ],
        "candidate_order": candidate_order,
        "candidates": candidates,
        "common_comparisons": common_comparisons,
        "benchmark_passed_candidates": [
            name
            for name in candidate_order
            if candidates[name]["advance"]["benchmark_passed"] is True
        ],
        "deployment_qualified_candidates": [
            name
            for name in candidate_order
            if candidates[name]["advance"]["deployment_qualified"] is True
        ],
    }


def select_own_policy_rows(
    frame: pl.DataFrame,
    confidence_threshold: float,
) -> pl.DataFrame:
    if not 0.5 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.5 and 1.0")
    eligible = frame
    if "model_eligible" in frame.columns:
        eligible = eligible.filter(pl.col("model_eligible"))
    return (
        eligible.filter(pl.col("confidence") >= confidence_threshold)
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
        .sort(["observed_at", "market_id"])
    )


def select_chronological_policy_rows(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"policy_selected", "selected_confidence_threshold"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "chronological preselected policy is missing columns: "
            + ", ".join(missing)
        )
    eligible = frame
    if "model_eligible" in frame.columns:
        eligible = eligible.filter(pl.col("model_eligible"))
    if eligible.is_empty():
        return eligible
    unstable_thresholds = (
        eligible.group_by("market_id")
        .agg(
            pl.col("selected_confidence_threshold")
            .n_unique()
            .alias("thresholds")
        )
        .filter(pl.col("thresholds") != 1)
    )
    if not unstable_thresholds.is_empty():
        raise ValueError(
            "chronological selected confidence threshold must be stable per market"
        )
    return _validated_preselected_policy_rows(eligible)


def select_time_band_policy_rows(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "policy_selected",
        "selected_confidence_threshold",
        "policy_threshold_band",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "time-band preselected policy is missing columns: "
            + ", ".join(missing)
        )
    eligible = frame
    if "model_eligible" in frame.columns:
        eligible = eligible.filter(pl.col("model_eligible"))
    if eligible.is_empty():
        return eligible
    if eligible["policy_threshold_band"].null_count():
        raise ValueError("time-band preselected policy cannot contain an unassigned row")
    unstable_thresholds = (
        eligible.group_by("policy_threshold_band")
        .agg(
            pl.col("selected_confidence_threshold")
            .n_unique()
            .alias("thresholds")
        )
        .filter(pl.col("thresholds") != 1)
    )
    if not unstable_thresholds.is_empty():
        raise ValueError(
            "time-band selected confidence threshold must be stable per band"
        )
    return _validated_preselected_policy_rows(eligible)


def _validated_preselected_policy_rows(
    eligible: pl.DataFrame,
) -> pl.DataFrame:
    expected = (
        eligible.filter(
            pl.col("confidence") >= pl.col("selected_confidence_threshold")
        )
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
        .sort(["observed_at", "market_id"])
    )
    selected = eligible.filter(pl.col("policy_selected")).sort(
        ["observed_at", "market_id"]
    )
    duplicate_markets = (
        selected.group_by("market_id")
        .len()
        .filter(pl.col("len") != 1)
    )
    if not duplicate_markets.is_empty():
        raise ValueError(
            "chronological preselected policy must select at most one row per market"
        )
    keys = ["market_id", "observed_at", "seconds_elapsed"]
    if (
        expected.select(keys)
        .join(selected.select(keys), on=keys, how="anti")
        .height
        or selected.select(keys)
        .join(expected.select(keys), on=keys, how="anti")
        .height
    ):
        raise ValueError(
            "policy_selected does not match the chronological first confidence crossing"
        )
    return selected


def _select_policy_rows(
    frame: pl.DataFrame,
    policy: CandidatePolicy,
) -> pl.DataFrame:
    if policy.selection_mode == "chronological_preselected":
        return select_chronological_policy_rows(frame)
    if policy.selection_mode == "time_band_preselected":
        return select_time_band_policy_rows(frame)
    if policy.confidence_threshold is None:
        raise RuntimeError("fixed confidence policy lost its confidence threshold")
    return select_own_policy_rows(frame, policy.confidence_threshold)


def _coerce_candidate_frames(
    candidate_frames: Mapping[str, pl.DataFrame] | Sequence[pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    if isinstance(candidate_frames, Mapping):
        frames = dict(candidate_frames)
    else:
        frames = {}
        for frame in candidate_frames:
            if "candidate" not in frame.columns:
                raise ValueError("each prediction frame must contain candidate")
            names = frame["candidate"].unique().to_list()
            if len(names) != 1:
                raise ValueError("each prediction frame must contain exactly one candidate")
            name = str(names[0])
            if name in frames:
                raise ValueError(f"candidate {name!r} appears in multiple frames")
            frames[name] = frame
    if not frames:
        raise ValueError("at least one candidate prediction frame is required")
    return frames


def _normalize_prediction_frame(frame: pl.DataFrame, candidate: str) -> pl.DataFrame:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{candidate}: missing prediction columns: {', '.join(missing)}")
    if frame.is_empty():
        raise ValueError(f"{candidate}: prediction frame cannot be empty")
    names = {str(value) for value in frame["candidate"].unique().to_list()}
    if names != {candidate}:
        raise ValueError(
            f"{candidate}: candidate column does not match the candidate mapping key"
        )
    seconds = frame["seconds_elapsed"].cast(pl.Float64, strict=False).to_numpy()
    if (
        not np.isfinite(seconds).all()
        or not np.equal(seconds, np.floor(seconds)).all()
        or np.any((seconds < 0) | (seconds >= 300))
    ):
        raise ValueError(
            f"{candidate}: seconds_elapsed must be integral and within [0, 300)"
        )
    normalized = frame.with_columns(
        pl.col("market_id").cast(pl.String),
        pl.col("seconds_elapsed").cast(pl.Int32),
        pl.col("label_up").cast(pl.Int8),
        pl.col("predicted_up").cast(pl.Int8),
        pl.col("probability_up").cast(pl.Float64),
        pl.col("confidence").cast(pl.Float64),
        pl.col("correct").cast(pl.Boolean),
    )
    if "model_eligible" in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("model_eligible").fill_null(False).cast(pl.Boolean)
        )
    if "policy_selected" in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("policy_selected").fill_null(False).cast(pl.Boolean)
        )
    if "selected_confidence_threshold" in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("selected_confidence_threshold").cast(pl.Float64)
        )
    for column in REQUIRED_PREDICTION_COLUMNS:
        if normalized[column].null_count():
            raise ValueError(f"{candidate}: {column} cannot contain null values")
    if normalized.filter(pl.col("market_id").str.len_chars() == 0).height:
        raise ValueError(f"{candidate}: market_id cannot be empty")
    if normalized["observed_at"].dtype.base_type() != pl.Datetime:
        raise ValueError(f"{candidate}: observed_at must be a timezone-aware datetime")
    if normalized["observed_at"].dtype.time_zone is None:
        raise ValueError(f"{candidate}: observed_at must include a timezone")
    normalized = normalized.with_columns(
        pl.col("observed_at").dt.convert_time_zone("UTC")
    )
    duplicate_count = (
        normalized.group_by("market_id", "observed_at", "seconds_elapsed")
        .len()
        .filter(pl.col("len") != 1)
        .height
    )
    if duplicate_count:
        raise ValueError(f"{candidate}: duplicate market timestamps are not allowed")
    for column in ("label_up", "predicted_up"):
        values = set(normalized[column].unique().to_list())
        if not values.issubset({0, 1}):
            raise ValueError(f"{candidate}: {column} must be binary")
    probabilities = normalized["probability_up"].to_numpy()
    confidence = normalized["confidence"].to_numpy()
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError(f"{candidate}: probability_up must be finite and in [0, 1]")
    if not np.isfinite(confidence).all() or np.any(
        (confidence < 0.5) | (confidence > 1.0)
    ):
        raise ValueError(f"{candidate}: confidence must be finite and in [0.5, 1]")
    if "selected_confidence_threshold" in normalized.columns:
        thresholds = normalized["selected_confidence_threshold"].to_numpy()
        if (
            normalized["selected_confidence_threshold"].null_count()
            or not np.isfinite(thresholds).all()
            or np.any((thresholds < 0.5) | (thresholds > 1.0))
        ):
            raise ValueError(
                f"{candidate}: selected_confidence_threshold must be finite "
                "and in [0.5, 1]"
            )
    expected_prediction = (probabilities >= 0.5).astype(np.int8)
    if not np.array_equal(expected_prediction, normalized["predicted_up"].to_numpy()):
        raise ValueError(f"{candidate}: predicted_up is inconsistent with probability_up")
    expected_confidence = np.maximum(probabilities, 1.0 - probabilities)
    if not np.allclose(
        expected_confidence,
        confidence,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{candidate}: confidence is inconsistent with probability_up")
    expected_correct = (
        normalized["predicted_up"].to_numpy()
        == normalized["label_up"].to_numpy()
    )
    if not np.array_equal(expected_correct, normalized["correct"].to_numpy()):
        raise ValueError(f"{candidate}: correct is inconsistent with labels")
    label_counts = (
        normalized.group_by("market_id")
        .agg(pl.col("label_up").n_unique().alias("labels"))
        .filter(pl.col("labels") != 1)
        .height
    )
    if label_counts:
        raise ValueError(f"{candidate}: each market must have one stable label")
    return normalized.sort(["market_id", "seconds_elapsed", "observed_at"])


def _validate_cross_candidate_labels(frames: Mapping[str, pl.DataFrame]) -> None:
    labels: dict[str, int] = {}
    for candidate, frame in frames.items():
        for row in frame.select("market_id", "label_up").unique().iter_rows(
            named=True
        ):
            market_id = row["market_id"]
            label = int(row["label_up"])
            if market_id in labels and labels[market_id] != label:
                raise ValueError(
                    f"{candidate}: label differs from another candidate for {market_id}"
                )
            labels[market_id] = label


def _own_policy_metrics(
    frame: pl.DataFrame,
    selected: pl.DataFrame,
    *,
    eligible_market_ids: Sequence[str],
    quantity: float,
) -> dict[str, Any]:
    eligible_markets = len(eligible_market_ids)
    available = frame
    if "model_eligible" in frame.columns:
        available = available.filter(pl.col("model_eligible"))
    available_markets = available["market_id"].n_unique()
    metrics = _classification_metrics(selected, eligible_markets=eligible_markets)
    markets = metrics["markets"]
    metrics.update(
        {
            "available_markets": available_markets,
            "data_unavailable_markets": eligible_markets - available_markets,
            "confidence_no_trade_markets": available_markets - markets,
            "confidence_no_trade_rate": (
                (available_markets - markets) / available_markets
                if available_markets
                else 0.0
            ),
            "no_trade_markets": eligible_markets - markets,
            "no_trade_rate": (
                (eligible_markets - markets) / eligible_markets
                if eligible_markets
                else 0.0
            ),
            "median_seconds_elapsed": _quantile_or_none(
                selected, "seconds_elapsed", 0.5
            ),
            "p90_seconds_elapsed": _quantile_or_none(
                selected, "seconds_elapsed", 0.9
            ),
            "execution": _execution_metrics(
                selected,
                quantity=quantity,
            ),
        }
    )
    return metrics


def _classification_metrics(
    rows: pl.DataFrame,
    *,
    eligible_markets: int,
) -> dict[str, Any]:
    if rows.is_empty():
        return {
            "markets": 0,
            "eligible_markets": eligible_markets,
            "coverage": 0.0,
            "correct": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "recall": 0.0,
            "up_recall": 0.0,
            "down_recall": 0.0,
            "wilson_lower_95": 0.0,
            "wilson_upper_95": 0.0,
            "expected_calibration_error": None,
            "brier_score": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "maximum_consecutive_losses": 0,
        }
    y_true = rows["label_up"].to_numpy().astype(np.int8)
    y_pred = rows["predicted_up"].to_numpy().astype(np.int8)
    probability = rows["probability_up"].to_numpy().astype(np.float64)
    correct = y_true == y_pred
    markets = len(y_true)
    true_down = y_true == 0
    true_up = y_true == 1
    down_recall = _safe_ratio(np.sum(true_down & (y_pred == 0)), np.sum(true_down))
    up_recall = _safe_ratio(np.sum(true_up & (y_pred == 1)), np.sum(true_up))
    lower, upper = _wilson_interval(int(correct.sum()), markets)
    return {
        "markets": markets,
        "eligible_markets": eligible_markets,
        "coverage": markets / eligible_markets if eligible_markets else 0.0,
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": (up_recall + down_recall) / 2.0,
        "recall": (up_recall + down_recall) / 2.0,
        "up_recall": up_recall,
        "down_recall": down_recall,
        "wilson_lower_95": lower,
        "wilson_upper_95": upper,
        "expected_calibration_error": _expected_calibration_error(
            y_true,
            probability,
        ),
        "brier_score": float(np.mean((probability - y_true) ** 2)),
        "confusion_matrix": [
            [
                int(np.sum(true_down & (y_pred == 0))),
                int(np.sum(true_down & (y_pred == 1))),
            ],
            [
                int(np.sum(true_up & (y_pred == 0))),
                int(np.sum(true_up & (y_pred == 1))),
            ],
        ],
        "maximum_consecutive_losses": _maximum_loss_streak(
            rows.sort(["observed_at", "market_id"])["correct"].to_numpy(),
            loss_when=False,
        ),
    }


def _execution_metrics(rows: pl.DataFrame, *, quantity: float) -> dict[str, Any]:
    selected = _with_execution_columns(rows, quantity=quantity)
    if selected.is_empty():
        return _empty_execution_metrics(quantity)
    evidence_cohort = (
        selected.filter(pl.col("execution_evidence_available"))
        if "execution_evidence_available" in selected.columns
        else selected
    )
    executable = selected.filter(pl.col("_execution_available"))
    economic = executable.filter(
        pl.col("_fee_per_share").is_not_null()
        & pl.col("_fee_per_share").is_finite()
        & (pl.col("_fee_per_share") >= 0.0)
        & pl.col("_direct_edge_per_share").is_not_null()
        & pl.col("_direct_edge_per_share").is_finite()
        & pl.col("_realized_net_pnl").is_not_null()
        & pl.col("_realized_net_pnl").is_finite()
    )
    execution_prices = executable["_execution_price"].to_numpy()
    result = _empty_execution_metrics(quantity)
    result.update(
        {
            "execution_price_source": _execution_price_source(rows),
            "fee_source": _fee_source(rows),
            "selected_markets": selected.height,
            "execution_evidence_markets": evidence_cohort.height,
            "execution_evidence_coverage": evidence_cohort.height / selected.height,
            "executable_markets": executable.height,
            "executable_coverage": executable.height / selected.height,
            "executable_coverage_all_selected": executable.height / selected.height,
            "executable_coverage_within_evidence": (
                executable.height / evidence_cohort.height
                if evidence_cohort.height
                else 0.0
            ),
            "median_selected_ask_vwap_5": (
                float(np.median(execution_prices)) if len(execution_prices) else None
            ),
            "p90_selected_ask_vwap_5": (
                float(np.quantile(execution_prices, 0.9))
                if len(execution_prices)
                else None
            ),
            "economics_available": not economic.is_empty(),
            "economic_markets": economic.height,
        }
    )
    if economic.is_empty():
        return result
    fee_per_share = economic["_fee_per_share"].to_numpy()
    direct_edge = economic["_direct_edge_per_share"].to_numpy()
    net_pnl = economic["_realized_net_pnl"].to_numpy()
    ordered_pnl = economic.sort(["observed_at", "market_id"])[
        "_realized_net_pnl"
    ].to_numpy()
    result.update(
        {
            "mean_fee_per_share": float(np.mean(fee_per_share)),
            "total_fees": float(np.sum(fee_per_share) * quantity),
            "mean_direct_edge_per_share": float(np.mean(direct_edge)),
            "positive_direct_edge_markets": int(np.sum(direct_edge > 0)),
            "positive_direct_edge_rate": float(np.mean(direct_edge > 0)),
            "realized_net_pnl_total": float(np.sum(net_pnl)),
            "realized_net_expectancy_per_trade": float(np.mean(net_pnl)),
            "realized_net_expectancy_per_selected_market": float(
                np.sum(net_pnl) / selected.height
            ),
            "maximum_net_loss_streak": _maximum_loss_streak(
                ordered_pnl,
                loss_when=lambda value: value < 0,
            ),
            "maximum_drawdown": _maximum_drawdown(ordered_pnl),
        }
    )
    return result


def _with_execution_columns(rows: pl.DataFrame, *, quantity: float) -> pl.DataFrame:
    if rows.is_empty():
        return rows.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("_execution_price"),
            pl.lit(False).alias("_execution_available"),
            pl.lit(None, dtype=pl.Float64).alias("_fee_per_share"),
            pl.lit(None, dtype=pl.Float64).alias("_direct_edge_per_share"),
            pl.lit(None, dtype=pl.Float64).alias("_realized_net_pnl"),
        )
    price_expression = _execution_price_expression(rows)
    executable_expression = _execution_available_expression(rows)
    fee_expression = _fee_per_share_expression(rows, quantity=quantity)
    selected_probability = (
        pl.when(pl.col("predicted_up") == 1)
        .then(pl.col("probability_up"))
        .otherwise(1.0 - pl.col("probability_up"))
    )
    with_price = rows.with_columns(
        price_expression.cast(pl.Float64, strict=False).alias("_execution_price"),
        fee_expression.cast(pl.Float64, strict=False).alias("_fee_per_share"),
    )
    with_price = with_price.with_columns(
        (
            executable_expression
            & pl.col("_execution_price").is_not_null()
            & pl.col("_execution_price").is_finite()
            & pl.col("_execution_price").is_between(0.0, 1.0, closed="right")
        )
        .fill_null(False)
        .alias("_execution_available"),
        (
            selected_probability
            - pl.col("_execution_price")
            - pl.col("_fee_per_share")
        ).alias("_derived_direct_edge_per_share"),
    )
    direct_edge_candidates = [
        pl.col(column).cast(pl.Float64, strict=False)
        for column in (
            "direct_net_edge_per_share",
            "expected_net_per_share",
            "direct_edge",
        )
        if column in rows.columns
    ]
    direct_edge_candidates.append(pl.col("_derived_direct_edge_per_share"))
    return with_price.with_columns(
        pl.coalesce(direct_edge_candidates).alias("_direct_edge_per_share"),
        (
            quantity
            * (
                pl.col("correct").cast(pl.Int8)
                - pl.col("_execution_price")
                - pl.col("_fee_per_share")
            )
        ).alias("_realized_net_pnl"),
    )


def _execution_price_expression(rows: pl.DataFrame) -> pl.Expr:
    for column in (
        "selected_ask_vwap_5",
        "execution_price",
        "executable_price",
    ):
        if column in rows.columns:
            return pl.col(column)
    if {"up_ask_vwap_5", "down_ask_vwap_5"}.issubset(rows.columns):
        return (
            pl.when(pl.col("predicted_up") == 1)
            .then(pl.col("up_ask_vwap_5"))
            .otherwise(pl.col("down_ask_vwap_5"))
        )
    return pl.lit(None, dtype=pl.Float64)


def _execution_available_expression(rows: pl.DataFrame) -> pl.Expr:
    if "selected_side_executable" in rows.columns:
        return pl.col("selected_side_executable").fill_null(False).cast(pl.Boolean)
    if {"up_executable", "down_executable"}.issubset(rows.columns):
        return (
            pl.when(pl.col("predicted_up") == 1)
            .then(pl.col("up_executable"))
            .otherwise(pl.col("down_executable"))
            .fill_null(False)
            .cast(pl.Boolean)
        )
    return pl.lit(True)


def _fee_per_share_expression(rows: pl.DataFrame, *, quantity: float) -> pl.Expr:
    for column in (
        "direct_taker_fee_per_share",
        "fee_per_share",
        "estimated_fee_per_share",
    ):
        if column in rows.columns:
            return pl.col(column)
    for column in ("fee_total", "fees_total", "fees"):
        if column in rows.columns:
            return pl.col(column) / quantity
    if "fee_rate" in rows.columns:
        price = _execution_price_expression(rows)
        return (
            pl.col("fee_rate").cast(pl.Float64, strict=False)
            * price
            * (1.0 - price)
        )
    return pl.lit(None, dtype=pl.Float64)


def _execution_price_source(rows: pl.DataFrame) -> str | None:
    for column in (
        "selected_ask_vwap_5",
        "execution_price",
        "executable_price",
    ):
        if column in rows.columns:
            return column
    if {"up_ask_vwap_5", "down_ask_vwap_5"}.issubset(rows.columns):
        return "selected(up_ask_vwap_5,down_ask_vwap_5)"
    return None


def _fee_source(rows: pl.DataFrame) -> str | None:
    for column in (
        "direct_taker_fee_per_share",
        "fee_per_share",
        "estimated_fee_per_share",
        "fee_total",
        "fees_total",
        "fees",
        "fee_rate",
    ):
        if column in rows.columns:
            return column
    return None


def _empty_execution_metrics(quantity: float) -> dict[str, Any]:
    return {
        "quantity": quantity,
        "execution_price_source": None,
        "fee_source": None,
        "selected_markets": 0,
        "execution_evidence_markets": 0,
        "execution_evidence_coverage": 0.0,
        "executable_markets": 0,
        "executable_coverage": 0.0,
        "executable_coverage_all_selected": 0.0,
        "executable_coverage_within_evidence": 0.0,
        "median_selected_ask_vwap_5": None,
        "p90_selected_ask_vwap_5": None,
        "economics_available": False,
        "economic_markets": 0,
        "mean_fee_per_share": None,
        "total_fees": None,
        "mean_direct_edge_per_share": None,
        "positive_direct_edge_markets": 0,
        "positive_direct_edge_rate": None,
        "realized_net_pnl_total": None,
        "realized_net_expectancy_per_trade": None,
        "realized_net_expectancy_per_selected_market": None,
        "maximum_net_loss_streak": None,
        "maximum_drawdown": None,
    }


def _time_band_metrics(
    selected: pl.DataFrame,
    *,
    eligible_markets: int,
) -> list[dict[str, Any]]:
    output = []
    for name, start, end in TIME_BANDS:
        rows = selected.filter(
            pl.col("seconds_elapsed").is_between(start, end, closed="both")
        )
        output.append(
            {
                "band": name,
                "start": start,
                "end": end,
                **_classification_metrics(rows, eligible_markets=eligible_markets),
            }
        )
    return output


def _checkpoint_metrics(
    frame: pl.DataFrame,
    *,
    eligible_markets: int,
) -> list[dict[str, Any]]:
    output = []
    for checkpoint in FIXED_CHECKPOINTS:
        rows = frame.filter(pl.col("seconds_elapsed") == checkpoint)
        if "model_eligible" in rows.columns:
            rows = rows.filter(pl.col("model_eligible"))
        output.append(
            {
                "seconds_elapsed": checkpoint,
                **_classification_metrics(
                    rows,
                    eligible_markets=eligible_markets,
                ),
            }
        )
    return output


def _common_checkpoint_comparison(
    control: pl.DataFrame,
    candidate: pl.DataFrame,
    *,
    control_name: str,
    candidate_name: str,
) -> dict[str, Any]:
    comparisons = []
    keys = ["market_id", "observed_at", "seconds_elapsed"]
    for checkpoint in FIXED_CHECKPOINTS:
        control_rows = control.filter(pl.col("seconds_elapsed") == checkpoint)
        candidate_rows = candidate.filter(pl.col("seconds_elapsed") == checkpoint)
        if "model_eligible" in control_rows.columns:
            control_rows = control_rows.filter(pl.col("model_eligible"))
        if "model_eligible" in candidate_rows.columns:
            candidate_rows = candidate_rows.filter(pl.col("model_eligible"))
        control_at_time = (
            control_rows
            .select(
                *keys,
                pl.col("label_up").alias("control_label_up"),
                pl.col("predicted_up").alias("control_predicted_up"),
                pl.col("probability_up").alias("control_probability_up"),
                pl.col("confidence").alias("control_confidence"),
                pl.col("correct").alias("control_correct"),
            )
        )
        candidate_at_time = (
            candidate_rows
            .select(
                *keys,
                pl.col("label_up").alias("candidate_label_up"),
                pl.col("predicted_up").alias("candidate_predicted_up"),
                pl.col("probability_up").alias("candidate_probability_up"),
                pl.col("confidence").alias("candidate_confidence"),
                pl.col("correct").alias("candidate_correct"),
            )
        )
        common = control_at_time.join(candidate_at_time, on=keys, how="inner")
        if common.filter(
            pl.col("control_label_up") != pl.col("candidate_label_up")
        ).height:
            raise ValueError(
                f"{candidate_name}: labels differ from control on common timestamps"
            )
        control_rows = _comparison_side_rows(common, "control")
        candidate_rows = _comparison_side_rows(common, "candidate")
        control_metrics = _classification_metrics(
            control_rows,
            eligible_markets=common.height,
        )
        candidate_metrics = _classification_metrics(
            candidate_rows,
            eligible_markets=common.height,
        )
        comparisons.append(
            {
                "seconds_elapsed": checkpoint,
                "common_markets": common.height,
                "control": control_metrics,
                "candidate": candidate_metrics,
                "accuracy_delta": (
                    candidate_metrics["accuracy"] - control_metrics["accuracy"]
                ),
                "balanced_accuracy_delta": (
                    candidate_metrics["balanced_accuracy"]
                    - control_metrics["balanced_accuracy"]
                ),
                "up_recall_delta": (
                    candidate_metrics["up_recall"] - control_metrics["up_recall"]
                ),
                "down_recall_delta": (
                    candidate_metrics["down_recall"]
                    - control_metrics["down_recall"]
                ),
            }
        )
    return {
        "control_candidate": control_name,
        "candidate": candidate_name,
        "cohort": "same market_id and exact observed_at at fixed seconds_elapsed",
        "checkpoints": comparisons,
    }


def _comparison_side_rows(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "market_id": pl.String,
                "observed_at": pl.Datetime(time_zone="UTC"),
                "seconds_elapsed": pl.Int32,
                "label_up": pl.Int8,
                "predicted_up": pl.Int8,
                "probability_up": pl.Float64,
                "confidence": pl.Float64,
                "correct": pl.Boolean,
            }
        )
    return frame.select(
        "market_id",
        "observed_at",
        "seconds_elapsed",
        pl.col(f"{prefix}_label_up").alias("label_up"),
        pl.col(f"{prefix}_predicted_up").alias("predicted_up"),
        pl.col(f"{prefix}_probability_up").alias("probability_up"),
        pl.col(f"{prefix}_confidence").alias("confidence"),
        pl.col(f"{prefix}_correct").alias("correct"),
    )


def _advance_checks(
    candidate: dict[str, Any],
    control: dict[str, Any],
    *,
    policy: CandidatePolicy,
    minimum_samples: int,
    minimum_executable_samples: int,
    evidence: BenchmarkEvidence,
    criteria: AdvancementCriteria,
    quantity: float,
) -> dict[str, Any]:
    candidate_execution = candidate["execution"]
    checks = [
        _check(
            "minimum accepted samples",
            candidate["markets"],
            ">=",
            minimum_samples,
            candidate["markets"] >= minimum_samples,
        ),
        _check(
            "minimum accuracy",
            candidate["accuracy"],
            ">=",
            criteria.minimum_accuracy,
            candidate["accuracy"] >= criteria.minimum_accuracy,
        ),
        _check(
            "minimum balanced accuracy",
            candidate["balanced_accuracy"],
            ">=",
            criteria.minimum_balanced_accuracy,
            candidate["balanced_accuracy"]
            >= criteria.minimum_balanced_accuracy,
        ),
        _check(
            "minimum UP recall",
            candidate["up_recall"],
            ">=",
            criteria.minimum_direction_recall,
            candidate["up_recall"] >= criteria.minimum_direction_recall,
        ),
        _check(
            "minimum DOWN recall",
            candidate["down_recall"],
            ">=",
            criteria.minimum_direction_recall,
            candidate["down_recall"] >= criteria.minimum_direction_recall,
        ),
        _check(
            "minimum Wilson lower bound",
            candidate["wilson_lower_95"],
            ">=",
            criteria.minimum_wilson_lower_95,
            candidate["wilson_lower_95"]
            >= criteria.minimum_wilson_lower_95,
        ),
        _check(
            "maximum expected calibration error",
            candidate["expected_calibration_error"],
            "<=",
            criteria.maximum_expected_calibration_error,
            candidate["expected_calibration_error"] is not None
            and candidate["expected_calibration_error"]
            <= criteria.maximum_expected_calibration_error,
        ),
        _check(
            "minimum eligible-market coverage",
            candidate["coverage"],
            ">=",
            criteria.minimum_coverage,
            candidate["coverage"] >= criteria.minimum_coverage,
        ),
        _check(
            "accuracy does not regress beyond tolerance",
            candidate["accuracy"],
            ">=",
            control["accuracy"] - criteria.maximum_accuracy_regression,
            candidate["accuracy"]
            >= control["accuracy"] - criteria.maximum_accuracy_regression,
        ),
        _check(
            "balanced accuracy does not regress beyond tolerance",
            candidate["balanced_accuracy"],
            ">=",
            (
                control["balanced_accuracy"]
                - criteria.maximum_balanced_accuracy_regression
            ),
            candidate["balanced_accuracy"]
            >= (
                control["balanced_accuracy"]
                - criteria.maximum_balanced_accuracy_regression
            ),
        ),
        _check(
            "UP recall does not regress beyond tolerance",
            candidate["up_recall"],
            ">=",
            control["up_recall"] - criteria.maximum_direction_recall_regression,
            candidate["up_recall"]
            >= control["up_recall"]
            - criteria.maximum_direction_recall_regression,
        ),
        _check(
            "DOWN recall does not regress beyond tolerance",
            candidate["down_recall"],
            ">=",
            (
                control["down_recall"]
                - criteria.maximum_direction_recall_regression
            ),
            candidate["down_recall"]
            >= (
                control["down_recall"]
                - criteria.maximum_direction_recall_regression
            ),
        ),
        _check(
            "Wilson lower bound does not regress",
            candidate["wilson_lower_95"],
            ">=",
            control["wilson_lower_95"],
            candidate["wilson_lower_95"] >= control["wilson_lower_95"],
        ),
        _check(
            "eligible-market coverage improves",
            candidate["coverage"] - control["coverage"],
            ">",
            criteria.minimum_coverage_uplift,
            candidate["coverage"] - control["coverage"]
            > criteria.minimum_coverage_uplift,
        ),
        _check(
            "median decision time is earlier",
            _difference_or_none(
                candidate["median_seconds_elapsed"],
                control["median_seconds_elapsed"],
            ),
            "<=",
            criteria.maximum_median_entry_seconds_regression,
            _difference_at_most(
                candidate["median_seconds_elapsed"],
                control["median_seconds_elapsed"],
                criteria.maximum_median_entry_seconds_regression,
            ),
        ),
        _check(
            "minimum executable economics samples",
            candidate_execution["economic_markets"],
            ">=",
            minimum_executable_samples,
            candidate_execution["economic_markets"]
            >= minimum_executable_samples,
        ),
        _check(
            "minimum mean direct edge per share",
            candidate_execution["mean_direct_edge_per_share"],
            ">",
            criteria.minimum_mean_direct_edge_per_share,
            _greater_than(
                candidate_execution["mean_direct_edge_per_share"],
                criteria.minimum_mean_direct_edge_per_share,
            ),
        ),
        _check(
            "minimum realized net per share",
            _per_share(
                candidate_execution["realized_net_expectancy_per_trade"],
                quantity,
            ),
            ">",
            criteria.minimum_realized_net_per_share,
            _greater_than(
                _per_share(
                    candidate_execution["realized_net_expectancy_per_trade"],
                    quantity,
                ),
                criteria.minimum_realized_net_per_share,
            ),
        ),
        _check(
            "runtime deployment contract is compatible",
            policy.deployment_compatible,
            "=",
            True,
            policy.deployment_compatible,
        ),
        _check(
            "native inference p99 is within budget",
            policy.native_p99_milliseconds,
            "<=",
            criteria.maximum_native_p99_milliseconds,
            _at_most(
                policy.native_p99_milliseconds,
                criteria.maximum_native_p99_milliseconds,
            ),
        ),
        _check(
            "runtime model size is within budget",
            policy.runtime_model_bytes,
            "<=",
            criteria.maximum_runtime_model_bytes,
            _at_most(
                policy.runtime_model_bytes,
                criteria.maximum_runtime_model_bytes,
            ),
        ),
    ]
    benchmark_passed = all(check["passed"] for check in checks)
    independent_holdout = evidence.kind == "holdout" and evidence.independent
    return {
        "is_control": False,
        "benchmark_passed": benchmark_passed,
        "deployment_qualified": benchmark_passed and independent_holdout,
        "evidence_allows_deployment_qualification": independent_holdout,
        "checks": checks,
    }


def _check(
    name: str,
    observed: Any,
    operator: str,
    required: Any,
    passed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "operator": operator,
        "required": required,
        "passed": bool(passed),
    }


def _expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    order = np.argsort(probability)
    chunks = np.array_split(order, min(bins, len(order)))
    total = len(order)
    error = 0.0
    for indices in chunks:
        if len(indices):
            error += (
                len(indices)
                / total
                * abs(
                    float(probability[indices].mean())
                    - float(y_true[indices].mean())
                )
            )
    return float(error)


def _wilson_interval(
    correct: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * total)) / total
    )
    return (centre - margin) / denominator, (centre + margin) / denominator


def _quantile_or_none(
    rows: pl.DataFrame,
    column: str,
    quantile: float,
) -> float | None:
    if rows.is_empty():
        return None
    return float(np.quantile(rows[column].to_numpy(), quantile))


def _safe_ratio(numerator: int | np.integer, denominator: int | np.integer) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _maximum_loss_streak(
    values: np.ndarray,
    *,
    loss_when: bool | Any,
) -> int:
    longest = 0
    current = 0
    for value in values:
        is_loss = value == loss_when if isinstance(loss_when, bool) else loss_when(value)
        if is_loss:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _maximum_drawdown(net_pnl: np.ndarray) -> float:
    if len(net_pnl) == 0:
        return 0.0
    cumulative = np.cumsum(net_pnl)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    drawdown = running_peak[1:] - cumulative
    return float(np.max(drawdown))


def _difference_or_none(
    left: float | None,
    right: float | None,
) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _difference_at_most(
    left: float | None,
    right: float | None,
    maximum: float,
) -> bool:
    difference = _difference_or_none(left, right)
    return difference is not None and difference <= maximum


def _per_share(value: float | None, quantity: float) -> float | None:
    if value is None:
        return None
    return value / quantity


def _greater_than(value: float | None, minimum: float) -> bool:
    return value is not None and value > minimum


def _at_most(value: float | None, maximum: float) -> bool:
    return value is not None and value <= maximum
