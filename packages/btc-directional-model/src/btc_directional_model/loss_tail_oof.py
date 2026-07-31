from __future__ import annotations

import hashlib
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .core_config import CoreTrainingConfig
from .core_extract import file_sha256, write_json_atomic
from .core_features import (
    CORE_BOUNDARY_ENRICHED_FEATURES,
    CORE_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
)
from .core_training import (
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    MARKET_EQUAL_ROW_WEIGHT_SCHEDULE,
    CandidateSpec,
    ProbabilityCalibrator,
    chronological_inner_split,
    fit_model,
    fit_probability_calibrator,
    range_frame,
)

LOSS_TAIL_OOF_SCHEMA_VERSION = "btc-loss-tail-oof-signals-v1"
LOSS_TAIL_OOF_HISTORY_START = datetime(2026, 3, 21, tzinfo=UTC)
LOSS_TAIL_OOF_MAXIMUM_FIRST_MARKET_START = LOSS_TAIL_OOF_HISTORY_START + timedelta(minutes=5)
LOSS_TAIL_OOF_KEYS = (
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
)
LOSS_TAIL_OOF_CALIBRATION_WEIGHTING = "equal_total_per_market"
LOSS_TAIL_OOF_HGB_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 160,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 0.10,
}


@dataclass(frozen=True)
class LossTailOofBlock:
    name: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class LossTailOofCalibrationBand:
    name: str
    start_second: int
    end_second_exclusive: int


@dataclass(frozen=True)
class LossTailOofHead:
    name: str
    prefix: str
    feature_names: tuple[str, ...]
    calibration_kind: str
    no_trade_confidence: float
    recency_half_life_days: float | None
    locked_direction: bool = False


LOSS_TAIL_OOF_BLOCKS = (
    LossTailOofBlock(
        "book_history_apr13",
        datetime(2026, 4, 13, tzinfo=UTC),
        datetime(2026, 5, 26, tzinfo=UTC),
    ),
    LossTailOofBlock(
        "calibration_may26",
        datetime(2026, 5, 26, tzinfo=UTC),
        datetime(2026, 6, 2, tzinfo=UTC),
    ),
    LossTailOofBlock(
        "calibration_jun02",
        datetime(2026, 6, 2, tzinfo=UTC),
        datetime(2026, 6, 9, tzinfo=UTC),
    ),
    LossTailOofBlock(
        "evaluation_jun09",
        datetime(2026, 6, 9, tzinfo=UTC),
        datetime(2026, 7, 3, tzinfo=UTC),
    ),
    LossTailOofBlock(
        "evaluation_jul03",
        datetime(2026, 7, 3, tzinfo=UTC),
        datetime(2026, 7, 14, tzinfo=UTC),
    ),
    LossTailOofBlock(
        "confirmation_jul14",
        datetime(2026, 7, 14, tzinfo=UTC),
        datetime(2026, 7, 29, tzinfo=UTC),
    ),
)

LOSS_TAIL_OOF_CALIBRATION_BANDS = (
    LossTailOofCalibrationBand("60-89", 60, 90),
    LossTailOofCalibrationBand("90-119", 90, 120),
    LossTailOofCalibrationBand("120-179", 120, 180),
    LossTailOofCalibrationBand("180-240", 180, 241),
)

LOSS_TAIL_OOF_HEADS = (
    LossTailOofHead(
        name="enriched_58",
        prefix="oof_enriched",
        feature_names=tuple(CORE_ENRICHED_FEATURES),
        calibration_kind="global_platt",
        no_trade_confidence=0.89,
        recency_half_life_days=None,
    ),
    LossTailOofHead(
        name="boundary_68",
        prefix="oof_boundary",
        feature_names=tuple(CORE_BOUNDARY_ENRICHED_FEATURES),
        calibration_kind="time_banded_platt",
        no_trade_confidence=0.89,
        recency_half_life_days=None,
        locked_direction=True,
    ),
    LossTailOofHead(
        name="mature_reversal_71",
        prefix="oof_mature_reversal",
        feature_names=tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES),
        calibration_kind="global_platt",
        no_trade_confidence=0.87,
        recency_half_life_days=28.0,
    ),
)

OOF_SOURCE_SIGNAL_FEATURES = (
    "oof_enriched_probability_up",
    "oof_enriched_calibrated_logit",
    "oof_enriched_confidence",
    "oof_enriched_predicted_up",
    "oof_enriched_no_trade",
    "oof_enriched_probability_delta_5s",
    "oof_enriched_confidence_delta_5s",
    "oof_boundary_probability_up",
    "oof_boundary_calibrated_logit",
    "oof_boundary_confidence",
    "oof_boundary_predicted_up",
    "oof_boundary_no_trade",
    "oof_boundary_probability_delta_5s",
    "oof_boundary_confidence_delta_5s",
    "oof_mature_reversal_probability_up",
    "oof_mature_reversal_calibrated_logit",
    "oof_mature_reversal_confidence",
    "oof_mature_reversal_predicted_up",
    "oof_mature_reversal_no_trade",
    "oof_mature_reversal_probability_delta_5s",
    "oof_mature_reversal_confidence_delta_5s",
    "oof_probability_mean",
    "oof_probability_std",
    "oof_probability_range",
    "oof_confidence_mean",
    "oof_confidence_std",
    "oof_confidence_range",
    "oof_up_vote_count",
    "oof_no_trade_count",
    "oof_direction_agreement",
    "oof_boundary_vs_consensus",
)

ORACLE_BOOK_CONTRADICTION_FEATURES = (
    "seconds_elapsed_scaled",
    "oracle_gap_to_opening_boundary_bps",
    "oracle_return_from_window_open_bps",
    "binance_oracle_basis_bps",
    "binance_oracle_basis_change_30s_bps",
    "oracle_binance_direction_agreement_30s",
    "oracle_boundary_binance_path_agreement",
    "oracle_return_60s_binance_volatility_z",
    "book_mid_difference",
    "book_mid_complement_residual",
    "book_ask_complement_residual",
    "book_up_imbalance",
    "book_down_imbalance",
    "book_up_ask_vwap_5",
    "book_down_ask_vwap_5",
    "book_mid_difference_delta_5s",
    "book_up_imbalance_delta_5s",
    "book_down_imbalance_delta_5s",
    "book_up_ask_vwap_5_delta_5s",
    "book_down_ask_vwap_5_delta_5s",
    "boundary_oracle_gap_alignment_bps",
    "boundary_oracle_return_alignment_bps",
    "boundary_book_mid_alignment",
    "boundary_book_imbalance_alignment",
    "boundary_book_vwap_cost_gap",
    "boundary_book_vwap_delta_5s",
    "boundary_selected_debit_per_share",
    "boundary_selected_debit_severity",
)

# This is deliberately explicit. Any change to the correctness-model input is a
# reviewable schema change rather than a dynamically expanding feature family.
BOUNDARY_CORRECTNESS_FEATURES = (
    "oof_enriched_probability_up",
    "oof_enriched_calibrated_logit",
    "oof_enriched_confidence",
    "oof_enriched_predicted_up",
    "oof_enriched_no_trade",
    "oof_enriched_probability_delta_5s",
    "oof_enriched_confidence_delta_5s",
    "oof_boundary_probability_up",
    "oof_boundary_calibrated_logit",
    "oof_boundary_confidence",
    "oof_boundary_predicted_up",
    "oof_boundary_no_trade",
    "oof_boundary_probability_delta_5s",
    "oof_boundary_confidence_delta_5s",
    "oof_mature_reversal_probability_up",
    "oof_mature_reversal_calibrated_logit",
    "oof_mature_reversal_confidence",
    "oof_mature_reversal_predicted_up",
    "oof_mature_reversal_no_trade",
    "oof_mature_reversal_probability_delta_5s",
    "oof_mature_reversal_confidence_delta_5s",
    "oof_probability_mean",
    "oof_probability_std",
    "oof_probability_range",
    "oof_confidence_mean",
    "oof_confidence_std",
    "oof_confidence_range",
    "oof_up_vote_count",
    "oof_no_trade_count",
    "oof_direction_agreement",
    "oof_boundary_vs_consensus",
    "seconds_elapsed_scaled",
    "oracle_gap_to_opening_boundary_bps",
    "oracle_return_from_window_open_bps",
    "binance_oracle_basis_bps",
    "binance_oracle_basis_change_30s_bps",
    "oracle_binance_direction_agreement_30s",
    "oracle_boundary_binance_path_agreement",
    "oracle_return_60s_binance_volatility_z",
    "book_mid_difference",
    "book_mid_complement_residual",
    "book_ask_complement_residual",
    "book_up_imbalance",
    "book_down_imbalance",
    "book_up_ask_vwap_5",
    "book_down_ask_vwap_5",
    "book_mid_difference_delta_5s",
    "book_up_imbalance_delta_5s",
    "book_down_imbalance_delta_5s",
    "book_up_ask_vwap_5_delta_5s",
    "book_down_ask_vwap_5_delta_5s",
    "boundary_oracle_gap_alignment_bps",
    "boundary_oracle_return_alignment_bps",
    "boundary_book_mid_alignment",
    "boundary_book_imbalance_alignment",
    "boundary_book_vwap_cost_gap",
    "boundary_book_vwap_delta_5s",
    "boundary_selected_debit_per_share",
    "boundary_selected_debit_severity",
)

_STRICT_REQUIRED_COLUMNS = (
    *ORACLE_BOOK_CONTRADICTION_FEATURES[:20],
    "up_entry_debit_per_share",
    "down_entry_debit_per_share",
)
_LEAKAGE_FORBIDDEN_FEATURES = {
    "label_up",
    "official_outcome",
    "final_price",
    "boundary_direction_correct",
    "realized_up_net_per_share",
    "realized_down_net_per_share",
    "realized_selected_net_per_share",
    "correct",
}


def generate_loss_tail_oof_signals(
    universal_core_path: Path,
    strict_rows_path: Path,
    destination: Path,
    *,
    core_config: CoreTrainingConfig,
    force: bool = False,
) -> dict[str, Any]:
    """Generate fixed, block-causal source signals for the correctness model.

    Every scored B0-B5 row is produced by a model whose fit and probability
    calibration data end before the scored block begins. Deployed artifacts are
    never loaded or scored by this module.
    """

    universal_core_path = Path(universal_core_path).resolve()
    strict_rows_path = Path(strict_rows_path).resolve()
    destination = Path(destination).resolve()
    metadata_path = oof_metadata_path(destination)
    _require_input_file(universal_core_path, "universal core")
    _require_input_file(strict_rows_path, "strict-row")
    build_contract = _build_contract(
        universal_core_path,
        strict_rows_path,
        core_config,
    )
    if destination.exists() or metadata_path.exists():
        if not destination.exists() or not metadata_path.exists():
            raise RuntimeError("OOF cache parquet and metadata must exist together")
        if not force:
            metadata = _read_metadata(metadata_path)
            if metadata.get("build_contract") != build_contract:
                raise RuntimeError("OOF cache build contract changed")
            load_loss_tail_oof_signals(destination)
            return metadata

    universal = pl.read_parquet(universal_core_path).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    strict = pl.read_parquet(strict_rows_path).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    _validate_universal_input(universal)
    _validate_strict_input(strict)
    strict_keys = strict.select(*LOSS_TAIL_OOF_KEYS, "label_up")

    source_frames: list[pl.DataFrame] = []
    training_lineage: dict[str, list[dict[str, Any]]] = {}
    for head in LOSS_TAIL_OOF_HEADS:
        frame, lineage = _generate_head_signals(
            universal,
            strict_keys,
            head,
            core_config,
        )
        source_frames.append(frame)
        training_lineage[head.name] = lineage

    combined = _combine_head_signals(source_frames)
    output = _attach_strict_context(strict, combined)
    _validate_output_frame(output)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    output.write_parquet(temporary, compression="zstd", statistics=True)
    os.replace(temporary, destination)
    metadata: dict[str, Any] = {
        "schema_version": LOSS_TAIL_OOF_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "build_contract": build_contract,
        "training_lineage": training_lineage,
        "rows": output.height,
        "markets": output["market_id"].n_unique(),
        "columns": output.columns,
        "feature_names": list(BOUNDARY_CORRECTNESS_FEATURES),
        "target": "boundary_direction_correct",
        "locked_direction_source": "boundary_68",
        "key_sha256": _key_fingerprint(output),
        "output_sha256": file_sha256(destination),
        "blocks": {
            block.name: _cohort_record(output.filter(pl.col("oof_block") == block.name))
            for block in LOSS_TAIL_OOF_BLOCKS
        },
        "deployment": {
            "authorized": False,
            "runtime_artifact_loaded": False,
            "runtime_artifact_exported": False,
        },
    }
    write_json_atomic(metadata_path, metadata)
    return metadata


def build_causal_oof_signals(
    *,
    core_path: Path,
    strict_path: Path,
    output_path: Path,
    config: Any,
    core_config: CoreTrainingConfig,
    force: bool = False,
) -> dict[str, Any]:
    """Compatibility entry point for the frozen loss-tail benchmark contract."""

    configured_blocks = tuple(
        (block.name, block.start, block.end) for block in config.walk_forward.blocks
    )
    fixed_blocks = tuple((block.name, block.start, block.end) for block in LOSS_TAIL_OOF_BLOCKS)
    if configured_blocks != fixed_blocks:
        raise ValueError("loss-tail OOF blocks do not match the frozen benchmark")
    if config.walk_forward.history_start != LOSS_TAIL_OOF_HISTORY_START:
        raise ValueError("loss-tail OOF history start does not match the frozen benchmark")
    return generate_loss_tail_oof_signals(
        core_path,
        strict_path,
        output_path,
        core_config=core_config,
        force=force,
    )


def load_loss_tail_oof_signals(path: Path) -> pl.DataFrame:
    """Load and fully validate a persisted loss-tail OOF signal cache."""

    path = Path(path).resolve()
    metadata_path = oof_metadata_path(path)
    _require_input_file(path, "OOF signal")
    _require_input_file(metadata_path, "OOF metadata")
    metadata = _read_metadata(metadata_path)
    if metadata.get("schema_version") != LOSS_TAIL_OOF_SCHEMA_VERSION:
        raise RuntimeError("unsupported loss-tail OOF schema")
    if file_sha256(path) != metadata.get("output_sha256"):
        raise RuntimeError("loss-tail OOF parquet checksum changed")
    frame = pl.read_parquet(path).sort(["window_start", "market_id", "seconds_elapsed"])
    _validate_output_frame(frame)
    if frame.height != int(metadata.get("rows", -1)):
        raise RuntimeError("loss-tail OOF row count changed")
    if frame["market_id"].n_unique() != int(metadata.get("markets", -1)):
        raise RuntimeError("loss-tail OOF market count changed")
    if frame.columns != metadata.get("columns"):
        raise RuntimeError("loss-tail OOF column order changed")
    if _key_fingerprint(frame) != metadata.get("key_sha256"):
        raise RuntimeError("loss-tail OOF row keys changed")
    if tuple(metadata.get("feature_names", ())) != BOUNDARY_CORRECTNESS_FEATURES:
        raise RuntimeError("loss-tail OOF feature contract changed")
    return frame


def oof_metadata_path(path: Path) -> Path:
    return Path(path).with_suffix(".metadata.json")


def _generate_head_signals(
    universal: pl.DataFrame,
    strict_keys: pl.DataFrame,
    head: LossTailOofHead,
    core_config: CoreTrainingConfig,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    block_frames: list[pl.DataFrame] = []
    lineage: list[dict[str, Any]] = []
    for block in LOSS_TAIL_OOF_BLOCKS:
        history = range_frame(
            universal,
            LOSS_TAIL_OOF_HISTORY_START,
            block.start,
        ).sort(["window_start", "market_id", "seconds_elapsed"])
        fit, calibration = chronological_inner_split(
            history,
            validation_fraction=0.20,
        )
        score = range_frame(universal, block.start, block.end).sort(
            ["window_start", "market_id", "seconds_elapsed"]
        )
        _validate_causal_cohorts(fit, calibration, score, head, block)
        probability, fit_record = _fit_head_probability(
            fit,
            calibration,
            score,
            head,
            core_config,
        )
        signaled = _source_signal_frame(score, probability, head, block.name)
        strict_signaled = signaled.join(
            strict_keys,
            on=[*LOSS_TAIL_OOF_KEYS, "label_up"],
            how="semi",
        )
        block_frames.append(strict_signaled)
        lineage.append(
            {
                "block": block.name,
                "fit": _cohort_record(fit),
                "calibration": _cohort_record(calibration),
                "score_universal": _cohort_record(score),
                "score_strict": _cohort_record(strict_signaled),
                "fit_end_before_score_start": bool(
                    fit["window_start"].max() < score["window_start"].min()
                ),
                "calibration_end_before_score_start": bool(
                    calibration["window_start"].max() < score["window_start"].min()
                ),
                **fit_record,
            }
        )
    return (
        pl.concat(block_frames, how="vertical").sort(
            ["window_start", "market_id", "seconds_elapsed"]
        ),
        lineage,
    )


def _fit_head_probability(
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
    score: pl.DataFrame,
    head: LossTailOofHead,
    core_config: CoreTrainingConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    spec = CandidateSpec(
        name=f"loss_tail_oof_{head.name}",
        family="histogram",
        feature_names=head.feature_names,
        row_weight_policy=MARKET_EQUAL_ROW_WEIGHT_POLICY,
        row_weight_schedule=MARKET_EQUAL_ROW_WEIGHT_SCHEDULE,
        recency_half_life_days=head.recency_half_life_days,
    )
    model = fit_model(
        fit,
        spec,
        dict(LOSS_TAIL_OOF_HGB_PARAMETERS),
        core_config,
    )
    raw_logit = model.raw_logit(score)
    if head.calibration_kind == "global_platt":
        calibrator = fit_probability_calibrator(
            model,
            calibration,
            core_config,
            spec,
        )
        _validate_calibrator(calibrator, head.name)
        probability = calibrator.probability(raw_logit)
        calibration_record: dict[str, Any] = {
            "kind": "global_platt",
            "weighting": LOSS_TAIL_OOF_CALIBRATION_WEIGHTING,
            "calibrator": asdict(calibrator),
        }
    elif head.calibration_kind == "time_banded_platt":
        probability = np.full(score.height, np.nan, dtype=np.float64)
        elapsed = score["seconds_elapsed"].to_numpy()
        bands: list[dict[str, Any]] = []
        for band in LOSS_TAIL_OOF_CALIBRATION_BANDS:
            calibration_band = calibration.filter(
                pl.col("seconds_elapsed").is_between(
                    band.start_second,
                    band.end_second_exclusive,
                    closed="left",
                )
            )
            _validate_binary_rows(calibration_band, f"{head.name} {band.name} calibration")
            calibrator = fit_probability_calibrator(
                model,
                calibration_band,
                core_config,
                spec,
            )
            _validate_calibrator(calibrator, f"{head.name} {band.name}")
            mask = (elapsed >= band.start_second) & (elapsed < band.end_second_exclusive)
            probability[mask] = calibrator.probability(raw_logit[mask])
            bands.append(
                {
                    **asdict(band),
                    "rows": calibration_band.height,
                    "markets": calibration_band["market_id"].n_unique(),
                    "calibrator": asdict(calibrator),
                }
            )
        calibration_record = {
            "kind": "time_banded_platt",
            "weighting": LOSS_TAIL_OOF_CALIBRATION_WEIGHTING,
            "bands": bands,
        }
    else:
        raise ValueError(f"unsupported OOF calibration kind: {head.calibration_kind}")
    _validate_probability(probability, score.height, head.name)
    return probability, {
        "head": head.name,
        "feature_count": len(head.feature_names),
        "features": list(head.feature_names),
        "target": "official_direction",
        "fixed_hyperparameters": dict(LOSS_TAIL_OOF_HGB_PARAMETERS),
        "training_weighting": LOSS_TAIL_OOF_CALIBRATION_WEIGHTING,
        "recency_half_life_days": head.recency_half_life_days,
        "calibration": calibration_record,
        "no_trade_confidence": head.no_trade_confidence,
        "deployed_artifact_used": False,
    }


def _source_signal_frame(
    score: pl.DataFrame,
    probability: np.ndarray,
    head: LossTailOofHead,
    block_name: str,
) -> pl.DataFrame:
    _validate_probability(probability, score.height, head.name)
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    calibrated_logit = np.log(clipped / (1.0 - clipped))
    predicted_up = (probability >= 0.5).astype(np.int8)
    confidence = np.maximum(probability, 1.0 - probability)
    no_trade = (confidence < head.no_trade_confidence).astype(np.int8)
    prefix = head.prefix
    frame = (
        score.select(*LOSS_TAIL_OOF_KEYS, "label_up")
        .with_columns(
            pl.lit(block_name).alias("oof_block"),
            pl.Series(f"{prefix}_probability_up", probability),
            pl.Series(f"{prefix}_calibrated_logit", calibrated_logit),
            pl.Series(f"{prefix}_confidence", confidence),
            pl.Series(f"{prefix}_predicted_up", predicted_up, dtype=pl.Int8),
            pl.Series(f"{prefix}_no_trade", no_trade, dtype=pl.Int8),
        )
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .with_columns(
            pl.col("seconds_elapsed").shift(1).over("market_id").alias("_oof_previous_second"),
            pl.col(f"{prefix}_probability_up")
            .shift(1)
            .over("market_id")
            .alias("_oof_previous_probability"),
            pl.col(f"{prefix}_confidence")
            .shift(1)
            .over("market_id")
            .alias("_oof_previous_confidence"),
        )
    )
    exact_previous = (pl.col("seconds_elapsed") - pl.col("_oof_previous_second")) == 5
    return (
        frame.with_columns(
            pl.when(exact_previous)
            .then(pl.col(f"{prefix}_probability_up") - pl.col("_oof_previous_probability"))
            .otherwise(None)
            .alias(f"{prefix}_probability_delta_5s"),
            pl.when(exact_previous)
            .then(pl.col(f"{prefix}_confidence") - pl.col("_oof_previous_confidence"))
            .otherwise(None)
            .alias(f"{prefix}_confidence_delta_5s"),
        )
        .drop(
            "_oof_previous_second",
            "_oof_previous_probability",
            "_oof_previous_confidence",
        )
        .sort(["window_start", "market_id", "seconds_elapsed"])
    )


def _combine_head_signals(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if len(frames) != len(LOSS_TAIL_OOF_HEADS):
        raise RuntimeError("all three fixed OOF heads are required")
    combined = frames[0]
    for frame in frames[1:]:
        right = frame.drop("oof_block")
        previous_height = combined.height
        combined = combined.join(
            right,
            on=[*LOSS_TAIL_OOF_KEYS, "label_up"],
            how="inner",
            validate="1:1",
        )
        if combined.height != previous_height or combined.height != frame.height:
            raise RuntimeError("OOF heads do not share one exact strict-row universe")

    probabilities = [pl.col(f"{head.prefix}_probability_up") for head in LOSS_TAIL_OOF_HEADS]
    confidences = [pl.col(f"{head.prefix}_confidence") for head in LOSS_TAIL_OOF_HEADS]
    votes = [pl.col(f"{head.prefix}_predicted_up").cast(pl.Int8) for head in LOSS_TAIL_OOF_HEADS]
    no_trades = [pl.col(f"{head.prefix}_no_trade").cast(pl.Int8) for head in LOSS_TAIL_OOF_HEADS]
    probability_mean = sum(probabilities, start=pl.lit(0.0)) / len(probabilities)
    confidence_mean = sum(confidences, start=pl.lit(0.0)) / len(confidences)
    vote_count = sum(votes, start=pl.lit(0, dtype=pl.Int8))
    no_trade_count = sum(no_trades, start=pl.lit(0, dtype=pl.Int8))
    combined = combined.with_columns(
        probability_mean.alias("oof_probability_mean"),
        confidence_mean.alias("oof_confidence_mean"),
        pl.max_horizontal(probabilities)
        .sub(pl.min_horizontal(probabilities))
        .alias("oof_probability_range"),
        pl.max_horizontal(confidences)
        .sub(pl.min_horizontal(confidences))
        .alias("oof_confidence_range"),
        vote_count.alias("oof_up_vote_count"),
        no_trade_count.alias("oof_no_trade_count"),
    )
    probability_variance = sum(
        [
            (pl.col(f"{head.prefix}_probability_up") - pl.col("oof_probability_mean")) ** 2
            for head in LOSS_TAIL_OOF_HEADS
        ],
        start=pl.lit(0.0),
    ) / len(LOSS_TAIL_OOF_HEADS)
    confidence_variance = sum(
        [
            (pl.col(f"{head.prefix}_confidence") - pl.col("oof_confidence_mean")) ** 2
            for head in LOSS_TAIL_OOF_HEADS
        ],
        start=pl.lit(0.0),
    ) / len(LOSS_TAIL_OOF_HEADS)
    return combined.with_columns(
        probability_variance.sqrt().alias("oof_probability_std"),
        confidence_variance.sqrt().alias("oof_confidence_std"),
        pl.col("oof_up_vote_count")
        .is_in([0, len(LOSS_TAIL_OOF_HEADS)])
        .cast(pl.Float64)
        .alias("oof_direction_agreement"),
        (pl.col("oof_boundary_predicted_up") != (pl.col("oof_up_vote_count") >= 2).cast(pl.Int8))
        .cast(pl.Float64)
        .alias("oof_boundary_vs_consensus"),
    )


def _attach_strict_context(
    strict: pl.DataFrame,
    signals: pl.DataFrame,
) -> pl.DataFrame:
    strict_context = strict.select(
        *LOSS_TAIL_OOF_KEYS,
        "label_up",
        *_STRICT_REQUIRED_COLUMNS,
    )
    output = strict_context.join(
        signals,
        on=[*LOSS_TAIL_OOF_KEYS, "label_up"],
        how="inner",
        validate="1:1",
    )
    if output.height != strict.height or output.height != signals.height:
        raise RuntimeError("strict rows and OOF signals do not share one exact key universe")
    direction_sign = pl.col("oof_boundary_predicted_up").cast(pl.Float64) * 2.0 - 1.0
    selected_debit = (
        pl.when(pl.col("oof_boundary_predicted_up") == 1)
        .then(pl.col("up_entry_debit_per_share"))
        .otherwise(pl.col("down_entry_debit_per_share"))
    )
    selected_vwap_delta = (
        pl.when(pl.col("oof_boundary_predicted_up") == 1)
        .then(pl.col("book_up_ask_vwap_5_delta_5s"))
        .otherwise(pl.col("book_down_ask_vwap_5_delta_5s"))
    )
    output = output.with_columns(
        (direction_sign * pl.col("oracle_gap_to_opening_boundary_bps")).alias(
            "boundary_oracle_gap_alignment_bps"
        ),
        (direction_sign * pl.col("oracle_return_from_window_open_bps")).alias(
            "boundary_oracle_return_alignment_bps"
        ),
        (direction_sign * pl.col("book_mid_difference")).alias("boundary_book_mid_alignment"),
        (direction_sign * (pl.col("book_up_imbalance") - pl.col("book_down_imbalance"))).alias(
            "boundary_book_imbalance_alignment"
        ),
        (direction_sign * (pl.col("book_up_ask_vwap_5") - pl.col("book_down_ask_vwap_5"))).alias(
            "boundary_book_vwap_cost_gap"
        ),
        selected_vwap_delta.alias("boundary_book_vwap_delta_5s"),
        selected_debit.alias("boundary_selected_debit_per_share"),
        (selected_debit / pl.max_horizontal(1.0 - selected_debit, pl.lit(0.05)))
        .clip(1.0, 10.0)
        .alias("boundary_selected_debit_severity"),
        (pl.col("oof_boundary_predicted_up") == pl.col("label_up").cast(pl.Int8))
        .cast(pl.Int8)
        .alias("boundary_direction_correct"),
    )
    return output.select(
        *LOSS_TAIL_OOF_KEYS,
        "label_up",
        "oof_block",
        pl.col("oof_block").alias("walk_forward_block"),
        *BOUNDARY_CORRECTNESS_FEATURES,
        "boundary_direction_correct",
    ).sort(["window_start", "market_id", "seconds_elapsed"])


def _validate_universal_input(frame: pl.DataFrame) -> None:
    required = {
        *LOSS_TAIL_OOF_KEYS,
        "label_up",
        *(feature for head in LOSS_TAIL_OOF_HEADS for feature in head.feature_names),
    }
    _require_columns(frame, required, "universal core")
    _validate_unique_keys(frame, "universal core")
    if frame["window_start"].min() > LOSS_TAIL_OOF_MAXIMUM_FIRST_MARKET_START:
        raise RuntimeError("universal core begins after the permitted causal warm-up market")
    if frame["window_start"].max() >= LOSS_TAIL_OOF_BLOCKS[-1].end:
        raise RuntimeError("universal core contains rows after the frozen OOF range")
    _validate_binary_rows(frame, "universal core")


def _validate_strict_input(frame: pl.DataFrame) -> None:
    required = {
        *LOSS_TAIL_OOF_KEYS,
        "label_up",
        *_STRICT_REQUIRED_COLUMNS,
    }
    _require_columns(frame, required, "strict-row")
    _validate_unique_keys(frame, "strict-row")
    outside = frame.filter(
        (pl.col("window_start") < LOSS_TAIL_OOF_BLOCKS[0].start)
        | (pl.col("window_start") >= LOSS_TAIL_OOF_BLOCKS[-1].end)
    )
    if outside.height:
        raise RuntimeError("strict rows extend outside the frozen B0-B5 range")
    _validate_binary_rows(frame, "strict-row")
    numeric = frame.select(pl.col(list(_STRICT_REQUIRED_COLUMNS)).cast(pl.Float64))
    invalid = numeric.select(
        pl.any_horizontal(
            [
                pl.col(column).is_null() | ~pl.col(column).is_finite()
                for column in _STRICT_REQUIRED_COLUMNS
            ]
        ).alias("invalid")
    )["invalid"].sum()
    if invalid:
        raise RuntimeError("strict context contains null or non-finite feature values")
    for column in ("up_entry_debit_per_share", "down_entry_debit_per_share"):
        if frame.filter((pl.col(column) <= 0.0) | (pl.col(column) > 1.0)).height:
            raise RuntimeError(f"{column} must stay inside (0, 1]")


def _validate_causal_cohorts(
    fit: pl.DataFrame,
    calibration: pl.DataFrame,
    score: pl.DataFrame,
    head: LossTailOofHead,
    block: LossTailOofBlock,
) -> None:
    for cohort, name in (
        (fit, "fit"),
        (calibration, "calibration"),
        (score, "score"),
    ):
        _validate_binary_rows(cohort, f"{head.name} {block.name} {name}")
    if fit["window_start"].max() >= calibration["window_start"].min():
        raise RuntimeError(f"{head.name} {block.name} fit/calibration overlap")
    if calibration["window_start"].max() >= score["window_start"].min():
        raise RuntimeError(f"{head.name} {block.name} calibration/score overlap")
    if score["window_start"].min() < block.start or score["window_start"].max() >= block.end:
        raise RuntimeError(f"{head.name} {block.name} score range changed")


def _validate_output_frame(frame: pl.DataFrame) -> None:
    required = {
        *LOSS_TAIL_OOF_KEYS,
        "label_up",
        "oof_block",
        "walk_forward_block",
        *BOUNDARY_CORRECTNESS_FEATURES,
        "boundary_direction_correct",
    }
    _require_columns(frame, required, "loss-tail OOF output")
    _validate_unique_keys(frame, "loss-tail OOF output")
    _validate_binary_rows(frame, "loss-tail OOF output")
    expected_blocks = {block.name for block in LOSS_TAIL_OOF_BLOCKS}
    if set(frame["oof_block"].unique().to_list()) != expected_blocks:
        raise RuntimeError("loss-tail OOF output does not contain every frozen block")
    if frame.filter(pl.col("walk_forward_block") != pl.col("oof_block")).height:
        raise RuntimeError("loss-tail OOF walk-forward block does not match its lineage block")
    if frame.filter(
        pl.col("boundary_direction_correct")
        != (pl.col("oof_boundary_predicted_up") == pl.col("label_up").cast(pl.Int8)).cast(pl.Int8)
    ).height:
        raise RuntimeError("boundary correctness target does not match the locked direction")
    for head in LOSS_TAIL_OOF_HEADS:
        probability = frame[f"{head.prefix}_probability_up"].to_numpy()
        _validate_probability(probability, frame.height, head.name)
        confidence = frame[f"{head.prefix}_confidence"].to_numpy()
        if (
            not np.isfinite(confidence).all()
            or np.any(confidence < 0.5)
            or np.any(confidence > 1.0)
        ):
            raise RuntimeError(f"{head.name} confidence is invalid")
        predicted = frame[f"{head.prefix}_predicted_up"].to_numpy()
        if not np.array_equal(predicted, (probability >= 0.5).astype(np.int8)):
            raise RuntimeError(f"{head.name} direction does not match probability")
        no_trade = frame[f"{head.prefix}_no_trade"].to_numpy()
        if not np.array_equal(
            no_trade,
            (confidence < head.no_trade_confidence).astype(np.int8),
        ):
            raise RuntimeError(f"{head.name} NoTrade state does not match its frozen threshold")


def _validate_binary_rows(frame: pl.DataFrame, role: str) -> None:
    if frame.is_empty():
        raise RuntimeError(f"{role} is empty")
    labels = frame["label_up"].drop_nulls().unique().to_list()
    if set(labels) != {0, 1}:
        raise RuntimeError(f"{role} must contain both outcome classes")


def _validate_calibrator(calibrator: ProbabilityCalibrator, role: str) -> None:
    if (
        not calibrator.converged
        or not math.isfinite(calibrator.slope)
        or not math.isfinite(calibrator.intercept)
        or calibrator.slope <= 0.0
    ):
        raise RuntimeError(f"{role} probability calibration is invalid")


def _validate_probability(values: np.ndarray, expected: int, role: str) -> None:
    probability = np.asarray(values, dtype=np.float64)
    if probability.ndim != 1 or len(probability) != expected:
        raise RuntimeError(f"{role} probability shape changed")
    if not np.isfinite(probability).all() or np.any(probability < 0.0) or np.any(probability > 1.0):
        raise RuntimeError(f"{role} probability is outside [0, 1]")


def _validate_unique_keys(frame: pl.DataFrame, role: str) -> None:
    if frame.select(*LOSS_TAIL_OOF_KEYS).is_duplicated().any():
        raise RuntimeError(f"{role} contains duplicate point-in-time keys")


def _require_columns(frame: pl.DataFrame, required: set[str], role: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{role} is missing columns: " + ", ".join(missing))


def _require_input_file(path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{role} file is missing: {path}")


def _build_contract(
    universal_core_path: Path,
    strict_rows_path: Path,
    core_config: CoreTrainingConfig,
) -> dict[str, Any]:
    source_path = getattr(core_config, "source_path", None)
    source_hash = (
        file_sha256(source_path)
        if isinstance(source_path, Path) and source_path.is_file()
        else None
    )
    random_seed = int(core_config.model.random_seed)
    return {
        "schema_version": LOSS_TAIL_OOF_SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "universal_core_sha256": file_sha256(universal_core_path),
        "strict_rows_sha256": file_sha256(strict_rows_path),
        "core_config_sha256": source_hash,
        "random_seed": random_seed,
        "history_start": LOSS_TAIL_OOF_HISTORY_START.isoformat(),
        "blocks": [
            {
                "name": block.name,
                "start": block.start.isoformat(),
                "end": block.end.isoformat(),
            }
            for block in LOSS_TAIL_OOF_BLOCKS
        ],
        "fit_fraction": 0.80,
        "calibration_fraction": 0.20,
        "fixed_hgb_parameters": dict(LOSS_TAIL_OOF_HGB_PARAMETERS),
        "calibration_weighting": LOSS_TAIL_OOF_CALIBRATION_WEIGHTING,
        "calibration_bands": [asdict(band) for band in LOSS_TAIL_OOF_CALIBRATION_BANDS],
        "heads": [
            {
                **asdict(head),
                "feature_names": list(head.feature_names),
            }
            for head in LOSS_TAIL_OOF_HEADS
        ],
        "strict_key_columns": list(LOSS_TAIL_OOF_KEYS),
        "correctness_features": list(BOUNDARY_CORRECTNESS_FEATURES),
        "target": "boundary_direction_correct",
        "deployed_artifacts_allowed": False,
    }


def _cohort_record(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "window_start_min": frame["window_start"].min().isoformat(),
        "window_start_max": frame["window_start"].max().isoformat(),
        "key_sha256": _key_fingerprint(frame),
    }


def _key_fingerprint(frame: pl.DataFrame) -> str:
    keys = frame.select(*LOSS_TAIL_OOF_KEYS).sort(["window_start", "market_id", "seconds_elapsed"])
    digest = hashlib.sha256()
    digest.update(str(keys.height).encode())
    digest.update(keys.hash_rows(seed=0).to_numpy().tobytes())
    return digest.hexdigest()


def _read_metadata(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("loss-tail OOF metadata must be a JSON object")
    return payload


def _validate_static_contract() -> None:
    counts = tuple(len(head.feature_names) for head in LOSS_TAIL_OOF_HEADS)
    if counts != (58, 68, 71):
        raise RuntimeError(f"fixed OOF source feature counts changed: {counts}")
    if sum(head.locked_direction for head in LOSS_TAIL_OOF_HEADS) != 1:
        raise RuntimeError("exactly one OOF source must lock the direction")
    if len(set(BOUNDARY_CORRECTNESS_FEATURES)) != len(BOUNDARY_CORRECTNESS_FEATURES):
        raise RuntimeError("boundary correctness features contain duplicates")
    if tuple(BOUNDARY_CORRECTNESS_FEATURES[: len(OOF_SOURCE_SIGNAL_FEATURES)]) != (
        OOF_SOURCE_SIGNAL_FEATURES
    ):
        raise RuntimeError("OOF source signal feature order changed")
    if tuple(BOUNDARY_CORRECTNESS_FEATURES[len(OOF_SOURCE_SIGNAL_FEATURES) :]) != (
        ORACLE_BOOK_CONTRADICTION_FEATURES
    ):
        raise RuntimeError("oracle/book contradiction feature order changed")
    leaked = sorted(set(BOUNDARY_CORRECTNESS_FEATURES) & _LEAKAGE_FORBIDDEN_FEATURES)
    if leaked:
        raise RuntimeError(
            "boundary correctness feature contract leaks targets: " + ", ".join(leaked)
        )


_validate_static_contract()
