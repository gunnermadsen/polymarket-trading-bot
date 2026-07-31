from __future__ import annotations

import hashlib
import html
import json
import math
import tomllib
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from .core_benchmark import _execution_metrics
from .core_config import (
    CORE_ORACLE_SOURCE_CONTRACT,
    CoreTrainingConfig,
    load_core_config,
    parse_utc_day,
)
from .core_evaluation import (
    choose_threshold,
    classification_metrics,
    first_crossing_timing,
    scored_prediction_rows,
    threshold_table,
)
from .core_execution import (
    EXECUTION_CONTEXT_SECONDS,
    EXECUTION_DECISION_SECONDS,
    EXECUTION_EVIDENCE_CONTRACT,
    EXECUTION_EVIDENCE_SCHEMA_VERSION,
    STRICT_BOTH_SIDE_TEN_SHARE_QUALITY_MASK,
    ExecutionEvidenceConfig,
    load_execution_evidence_manifest,
)
from .core_extract import file_sha256, write_json_atomic
from .core_features import (
    CORE_MATURE_REVERSAL_ORACLE_FEATURES,
    feature_destination,
    validate_core_feature_cache,
)
from .core_training import (
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    CandidateSpec,
    FrozenTrainingBundle,
    fit_probability_calibrator,
    range_frame,
    tune_and_fit_model,
)
from .offline_challengers import (
    STRICT_BOOK_V2_FEATURES,
    derive_strict_book_feature_frame,
)

ORACLE_BOOK_BENCHMARK_SCHEMA_VERSION = "btc-oracle-book-matched-benchmark-v1"
FIXED_DECISION_SECONDS = (120, 125)
TIMING_DECISION_SECONDS = EXECUTION_DECISION_SECONDS
EXPECTED_RECENCY_HALF_LIFE_DAYS = 28.0
EXPECTED_SPLITS = {
    "fit": (
        datetime(2026, 5, 26, tzinfo=UTC),
        datetime(2026, 7, 1, tzinfo=UTC),
    ),
    "calibration": (
        datetime(2026, 7, 1, tzinfo=UTC),
        datetime(2026, 7, 5, tzinfo=UTC),
    ),
    "threshold": (
        datetime(2026, 7, 5, tzinfo=UTC),
        datetime(2026, 7, 9, tzinfo=UTC),
    ),
    "evaluation": (
        datetime(2026, 7, 14, tzinfo=UTC),
        datetime(2026, 7, 21, tzinfo=UTC),
    ),
}


@dataclass(frozen=True)
class OracleBookSplitConfig:
    fit_start: datetime
    fit_end: datetime
    calibration_start: datetime
    calibration_end: datetime
    threshold_start: datetime
    threshold_end: datetime
    evaluation_start: datetime
    evaluation_end: datetime


@dataclass(frozen=True)
class OracleBookModelConfig:
    confidence_min: float
    confidence_max: float
    confidence_step: float
    recency_half_life_days: float


@dataclass(frozen=True)
class OracleBookPathConfig:
    execution_evidence: Path
    runs: Path


@dataclass(frozen=True)
class OracleBookBenchmarkConfig:
    source_path: Path
    package_root: Path
    core_config: Path
    core_config_sha256: str
    execution_manifest_sha256: str
    evaluation_note: str
    split: OracleBookSplitConfig
    model: OracleBookModelConfig
    paths: OracleBookPathConfig


def load_oracle_book_benchmark_config(path: Path) -> OracleBookBenchmarkConfig:
    source_path = path.resolve()
    package_root = source_path.parent.parent
    with source_path.open("rb") as handle:
        raw = tomllib.load(handle)
    benchmark = raw["benchmark"]
    split = raw["split"]
    model = raw["model"]
    paths = raw["paths"]
    config = OracleBookBenchmarkConfig(
        source_path=source_path,
        package_root=package_root,
        core_config=package_root / str(benchmark["core_config"]),
        core_config_sha256=str(benchmark["core_config_sha256"]).lower(),
        execution_manifest_sha256=str(
            benchmark["execution_manifest_sha256"]
        ).lower(),
        evaluation_note=str(benchmark["evaluation_note"]).strip(),
        split=OracleBookSplitConfig(
            fit_start=parse_utc_day(split["fit_start"]),
            fit_end=parse_utc_day(split["fit_end"]),
            calibration_start=parse_utc_day(split["calibration_start"]),
            calibration_end=parse_utc_day(split["calibration_end"]),
            threshold_start=parse_utc_day(split["threshold_start"]),
            threshold_end=parse_utc_day(split["threshold_end"]),
            evaluation_start=parse_utc_day(split["evaluation_start"]),
            evaluation_end=parse_utc_day(split["evaluation_end"]),
        ),
        model=OracleBookModelConfig(
            confidence_min=float(model["confidence_min"]),
            confidence_max=float(model["confidence_max"]),
            confidence_step=float(model["confidence_step"]),
            recency_half_life_days=float(model["recency_half_life_days"]),
        ),
        paths=OracleBookPathConfig(
            execution_evidence=package_root
            / str(paths["execution_evidence"]),
            runs=package_root / str(paths["runs"]),
        ),
    )
    validate_oracle_book_benchmark_config(config)
    return config


def validate_oracle_book_benchmark_config(
    config: OracleBookBenchmarkConfig,
) -> None:
    if not config.core_config.is_file():
        raise ValueError(f"core config is missing: {config.core_config}")
    if not config.evaluation_note:
        raise ValueError("evaluation_note must be non-empty")
    for name, checksum in (
        ("core_config_sha256", config.core_config_sha256),
        ("execution_manifest_sha256", config.execution_manifest_sha256),
    ):
        if len(checksum) != 64 or any(
            character not in "0123456789abcdef" for character in checksum
        ):
            raise ValueError(
                f"{name} must be 64 lowercase hexadecimal digits"
            )
    if file_sha256(config.core_config) != config.core_config_sha256:
        raise ValueError("pinned core training config hash mismatch")
    observed_splits = {
        "fit": (config.split.fit_start, config.split.fit_end),
        "calibration": (
            config.split.calibration_start,
            config.split.calibration_end,
        ),
        "threshold": (
            config.split.threshold_start,
            config.split.threshold_end,
        ),
        "evaluation": (
            config.split.evaluation_start,
            config.split.evaluation_end,
        ),
    }
    if observed_splits != EXPECTED_SPLITS:
        raise ValueError(
            "matched oracle/book benchmark requires the frozen May 26–July 21 "
            "chronological split contract"
        )
    if (
        config.model.confidence_min,
        config.model.confidence_max,
        config.model.confidence_step,
    ) != (0.80, 0.95, 0.01):
        raise ValueError(
            "matched book benchmark requires the frozen 0.80-0.95 "
            "confidence search"
        )
    if not math.isclose(
        config.model.recency_half_life_days,
        EXPECTED_RECENCY_HALF_LIFE_DAYS,
    ):
        raise ValueError("matched candidates require a 28-day recency half-life")
    if config.paths.execution_evidence == config.paths.runs:
        raise ValueError("execution evidence and run output paths must be isolated")


def run_oracle_book_benchmark(
    config: OracleBookBenchmarkConfig,
) -> tuple[Path, dict[str, Any]]:
    core_config = _load_and_validate_core_config(config)
    feature_metadata = validate_core_feature_cache(
        core_config,
        "pre_holdout",
    )
    feature_path = feature_destination(core_config, "pre_holdout")
    core_frame = pl.read_parquet(feature_path).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    _validate_core_frame(core_frame)

    execution_config = ExecutionEvidenceConfig(
        range_start=core_config.data.range_start,
        range_end=core_config.data.range_end,
        output_dir=config.paths.execution_evidence,
        sample_interval_seconds=5,
        min_seconds_after_open=EXECUTION_CONTEXT_SECONDS[0],
        max_seconds_after_open=EXECUTION_CONTEXT_SECONDS[-1],
        freshness_seconds=2,
        quantity=5.0,
    )
    manifest_path = execution_config.output_dir / "manifest.json"
    if file_sha256(manifest_path) != config.execution_manifest_sha256:
        raise RuntimeError("pinned execution-evidence manifest hash mismatch")
    execution_manifest = load_execution_evidence_manifest(execution_config)
    if (
        execution_manifest["source_contract"] != EXECUTION_EVIDENCE_CONTRACT
        or execution_manifest["source_schema_version"]
        != EXECUTION_EVIDENCE_SCHEMA_VERSION
    ):
        raise RuntimeError("matched benchmark requires v2 execution evidence")
    execution_frame = _load_execution_evidence(
        execution_config,
        execution_manifest,
    )
    _validate_strict_execution_contract(execution_frame)

    strict_features = derive_strict_book_feature_frame(
        core_frame,
        execution_frame,
        require_ten_share=True,
        include_deltas=True,
    ).sort(["window_start", "market_id", "seconds_elapsed"])
    _validate_model_features(strict_features)
    complete_book_market_ids = _complete_context_market_ids(execution_frame)
    complete_timing = _complete_timing_frame(
        strict_features,
        complete_book_market_ids,
    )
    economics = _execution_economics_frame(execution_frame)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    experiments: dict[str, Any] = {}
    for second in FIXED_DECISION_SECONDS:
        name = f"fixed_{second}s"
        frame = strict_features.filter(pl.col("seconds_elapsed") == second)
        experiments[name] = _run_matched_experiment(
            name=name,
            frame=frame,
            expected_seconds=(second,),
            economics=economics,
            config=config,
            core_config=core_config,
            run_dir=run_dir,
            universe_markets=_universe_markets(
                core_frame,
                config.split.evaluation_start,
                config.split.evaluation_end,
                expected_seconds=(second,),
            ),
        )
    experiments["complete_11_point_timing"] = _run_matched_experiment(
        name="complete_11_point_timing",
        frame=complete_timing,
        expected_seconds=TIMING_DECISION_SECONDS,
        economics=economics,
        config=config,
        core_config=core_config,
        run_dir=run_dir,
        universe_markets=_universe_markets(
            core_frame,
            config.split.evaluation_start,
            config.split.evaluation_end,
            expected_seconds=TIMING_DECISION_SECONDS,
        ),
    )

    payload: dict[str, Any] = {
        "schema_version": ORACLE_BOOK_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "development_diagnostic_complete",
        "evaluation": {
            "kind": "development",
            "independent": False,
            "note": config.evaluation_note,
        },
        "data_contract": {
            "core_source_contract": core_config.data.source_contract,
            "range_start": core_config.data.range_start.isoformat(),
            "range_end": core_config.data.range_end.isoformat(),
            "context_seconds": list(EXECUTION_CONTEXT_SECONDS),
            "fixed_decision_seconds": list(FIXED_DECISION_SECONDS),
            "timing_decision_seconds": list(TIMING_DECISION_SECONDS),
            "strict_point_qualification": (
                "strict causal fresh two-sided ten-share book at t-5 and t"
            ),
            "complete_timing_qualification": (
                "strict causal fresh two-sided ten-share book at all 11 exact "
                "90-140 second context points"
            ),
            "quality_or_missingness_features": False,
            "imputation_of_book_features": False,
        },
        "model_contract": {
            "family": "histogram",
            "base_feature_count": len(CORE_MATURE_REVERSAL_ORACLE_FEATURES),
            "base_features": list(CORE_MATURE_REVERSAL_ORACLE_FEATURES),
            "book_feature_count": len(STRICT_BOOK_V2_FEATURES),
            "book_features": list(STRICT_BOOK_V2_FEATURES),
            "challenger_feature_count": (
                len(CORE_MATURE_REVERSAL_ORACLE_FEATURES)
                + len(STRICT_BOOK_V2_FEATURES)
            ),
            "recency_half_life_days": config.model.recency_half_life_days,
            "row_weight_policy": MARKET_EQUAL_ROW_WEIGHT_POLICY,
        },
        "splits": _split_payload(config.split),
        "input_provenance": {
            "benchmark_config": str(config.source_path),
            "benchmark_config_sha256": file_sha256(config.source_path),
            "core_config": str(config.core_config),
            "core_config_sha256": file_sha256(config.core_config),
            "feature_path": str(feature_path),
            "feature_sha256": file_sha256(feature_path),
            "feature_metadata": feature_metadata,
            "execution_manifest": str(manifest_path),
            "execution_manifest_sha256": file_sha256(manifest_path),
            "execution_manifest_totals": execution_manifest["totals"],
        },
        "availability": {
            "point_qualified_rows": strict_features.height,
            "point_qualified_markets": strict_features[
                "market_id"
            ].n_unique(),
            "book_complete_11_point_markets": len(
                complete_book_market_ids
            ),
            "common_oracle_book_complete_11_point_markets": complete_timing[
                "market_id"
            ].n_unique(),
        },
        "experiments": experiments,
        "deployment": {
            "authorized": False,
            "runtime_exported": False,
            "trading_process_created": False,
            "reason": (
                "matched book experiments use consumed development evidence and "
                "book features are outside the current runtime contract"
            ),
        },
    }
    write_json_atomic(run_dir / "benchmark.json", payload)
    _write_text_atomic(run_dir / "report.html", _render_report(payload))
    return run_dir, payload


def control_candidate_spec(name: str, half_life_days: float) -> CandidateSpec:
    return CandidateSpec(
        name=name,
        family="histogram",
        feature_names=tuple(CORE_MATURE_REVERSAL_ORACLE_FEATURES),
        row_weight_policy=MARKET_EQUAL_ROW_WEIGHT_POLICY,
        recency_half_life_days=half_life_days,
    )


def book_candidate_spec(name: str, half_life_days: float) -> CandidateSpec:
    return CandidateSpec(
        name=name,
        family="histogram",
        feature_names=tuple(
            CORE_MATURE_REVERSAL_ORACLE_FEATURES + STRICT_BOOK_V2_FEATURES
        ),
        row_weight_policy=MARKET_EQUAL_ROW_WEIGHT_POLICY,
        recency_half_life_days=half_life_days,
    )


def _run_matched_experiment(
    *,
    name: str,
    frame: pl.DataFrame,
    expected_seconds: tuple[int, ...],
    economics: pl.DataFrame,
    config: OracleBookBenchmarkConfig,
    core_config: CoreTrainingConfig,
    run_dir: Path,
    universe_markets: int,
) -> dict[str, Any]:
    cohorts = _split_experiment_frame(frame, expected_seconds, config.split)
    control_spec = control_candidate_spec(
        f"{name}_core_oracle_control",
        config.model.recency_half_life_days,
    )
    challenger_spec = book_candidate_spec(
        f"{name}_core_oracle_book_challenger",
        config.model.recency_half_life_days,
    )
    threshold_config = replace(
        core_config,
        model=replace(
            core_config.model,
            confidence_min=config.model.confidence_min,
            confidence_max=config.model.confidence_max,
            confidence_step=config.model.confidence_step,
        ),
    )
    arms: dict[str, Any] = {}
    prediction_frames: dict[str, pl.DataFrame] = {}
    for arm_name, spec in (
        ("core_oracle", control_spec),
        ("core_oracle_book", challenger_spec),
    ):
        training, bundle = _fit_matched_candidate(
            cohorts,
            spec,
            threshold_config,
        )
        scored = _score_evaluation(cohorts["evaluation"], bundle)
        scored = _attach_execution_economics(scored, economics)
        selected = _first_crossing(scored, bundle.confidence_threshold)
        evaluation_markets = cohorts["evaluation"]["market_id"].n_unique()
        prediction_path = run_dir / f"{name}-{arm_name}-predictions.parquet"
        _write_parquet_atomic(scored, prediction_path)
        arms[arm_name] = {
            "training": training,
            "evaluation": _evaluation_payload(
                scored,
                selected,
                expected_seconds=expected_seconds,
                eligible_markets=evaluation_markets,
                universe_markets=universe_markets,
            ),
            "predictions": {
                "path": prediction_path.name,
                "sha256": file_sha256(prediction_path),
                "rows": scored.height,
            },
        }
        prediction_frames[arm_name] = scored
    _require_identical_keys(prediction_frames)
    return {
        "cohorts": {
            cohort_name: _cohort_payload(cohort, expected_seconds)
            for cohort_name, cohort in cohorts.items()
        },
        "same_rows_per_arm": True,
        "evaluation_universe_markets": universe_markets,
        "arms": arms,
        "paired_evaluation": _paired_evaluation(
            prediction_frames["core_oracle"],
            prediction_frames["core_oracle_book"],
            expected_seconds,
        ),
    }


def _fit_matched_candidate(
    cohorts: dict[str, pl.DataFrame],
    spec: CandidateSpec,
    config: CoreTrainingConfig,
) -> tuple[dict[str, Any], FrozenTrainingBundle]:
    model, tuning = tune_and_fit_model(cohorts["fit"], spec, config)
    calibrator = fit_probability_calibrator(
        model,
        cohorts["calibration"],
        config,
        spec,
    )
    threshold_probability = calibrator.probability(
        model.raw_logit(cohorts["threshold"])
    )
    history = threshold_table(
        cohorts["threshold"],
        threshold_probability,
        config.model,
    )
    minimum_markets = max(
        100,
        math.ceil(
            cohorts["threshold"]["market_id"].n_unique()
            * config.gates.minimum_coverage
        ),
    )
    threshold, qualified = choose_threshold(
        history,
        config.gates,
        minimum_markets=minimum_markets,
    )
    bundle = FrozenTrainingBundle(
        model=model,
        calibrator=calibrator,
        confidence_threshold=threshold,
    )
    return (
        {
            "candidate": spec.name,
            "family": spec.family,
            "feature_count": len(spec.feature_names),
            "features": list(spec.feature_names),
            "recency_half_life_days": spec.recency_half_life_days,
            "row_weight_policy": spec.row_weight_policy,
            "tuning": tuning,
            "calibrator": asdict(calibrator),
            "confidence_threshold": threshold,
            "threshold_qualified": qualified,
            "threshold_minimum_markets": minimum_markets,
            "threshold_history": history,
        },
        bundle,
    )


def _score_evaluation(
    frame: pl.DataFrame,
    bundle: FrozenTrainingBundle,
) -> pl.DataFrame:
    probability = bundle.probability(frame)
    return scored_prediction_rows(frame, probability).with_columns(
        pl.lit(bundle.model.candidate_name).alias("candidate"),
        pl.lit(True).alias("model_eligible"),
    )


def _first_crossing(
    scored: pl.DataFrame,
    confidence_threshold: float,
) -> pl.DataFrame:
    return (
        scored.filter(pl.col("confidence") >= confidence_threshold)
        .sort(["market_id", "seconds_elapsed", "observed_at"])
        .group_by("market_id", maintain_order=True)
        .first()
        .sort(["observed_at", "market_id"])
    )


def _evaluation_payload(
    scored: pl.DataFrame,
    selected: pl.DataFrame,
    *,
    expected_seconds: tuple[int, ...],
    eligible_markets: int,
    universe_markets: int,
) -> dict[str, Any]:
    policy = classification_metrics(
        selected,
        eligible_markets=eligible_markets,
    )
    return {
        "eligible_matched_markets": eligible_markets,
        "universal_core_oracle_markets": universe_markets,
        "book_availability_coverage": (
            eligible_markets / universe_markets if universe_markets else 0.0
        ),
        "policy": policy,
        "hard_confident_errors": _hard_confident_error_metrics(
            selected,
            eligible_markets=eligible_markets,
            confidence_floor=0.95,
        ),
        "policy_universal_coverage": (
            selected.height / universe_markets if universe_markets else 0.0
        ),
        "timing": first_crossing_timing(
            selected,
            eligible_markets=eligible_markets,
        ),
        "checkpoints": {
            str(second): classification_metrics(
                scored.filter(pl.col("seconds_elapsed") == second),
                eligible_markets=eligible_markets,
            )
            for second in expected_seconds
        },
        "execution_by_size": {
            "vwap5_five_share": _execution_metrics(
                selected,
                quantity=5.0,
                vwap_depth=5,
            ),
            "vwap10_ten_share": _execution_metrics(
                selected,
                quantity=10.0,
                vwap_depth=10,
            ),
        },
    }


def _hard_confident_error_metrics(
    selected: pl.DataFrame,
    *,
    eligible_markets: int,
    confidence_floor: float,
) -> dict[str, Any]:
    errors = selected.filter(
        ~pl.col("correct") & (pl.col("confidence") >= confidence_floor)
    )
    return {
        "confidence_floor": confidence_floor,
        "eligible_markets": eligible_markets,
        "selected_markets": selected.height,
        "hard_confident_error_markets": errors.height,
        "hard_confident_error_exposure_rate": (
            errors.height / eligible_markets if eligible_markets else 0.0
        ),
        "hard_confident_error_rate_selected": (
            errors.height / selected.height if selected.height else 0.0
        ),
        "maximum_incorrect_confidence": (
            float(
                selected.filter(~pl.col("correct"))["confidence"].max()
            )
            if selected.filter(~pl.col("correct")).height
            else None
        ),
    }


def _paired_evaluation(
    control: pl.DataFrame,
    challenger: pl.DataFrame,
    expected_seconds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "same_row_key_sha256": _row_key_sha256(control),
        "checkpoints": {
            str(second): _paired_checkpoint(
                control.filter(pl.col("seconds_elapsed") == second),
                challenger.filter(pl.col("seconds_elapsed") == second),
            )
            for second in expected_seconds
        },
    }


def _paired_checkpoint(
    control: pl.DataFrame,
    challenger: pl.DataFrame,
) -> dict[str, Any]:
    keys = ["market_id", "observed_at", "seconds_elapsed"]
    joined = control.select(
        *keys,
        pl.col("correct").alias("control_correct"),
        pl.col("predicted_up").alias("control_predicted_up"),
    ).join(
        challenger.select(
            *keys,
            pl.col("correct").alias("challenger_correct"),
            pl.col("predicted_up").alias("challenger_predicted_up"),
        ),
        on=keys,
        how="inner",
        validate="1:1",
    )
    if joined.height != control.height or joined.height != challenger.height:
        raise RuntimeError("paired checkpoint lost exact matched rows")
    return {
        "markets": joined.height,
        "control_accuracy": float(joined["control_correct"].mean()),
        "challenger_accuracy": float(joined["challenger_correct"].mean()),
        "accuracy_delta": float(
            joined["challenger_correct"].cast(pl.Int8).mean()
            - joined["control_correct"].cast(pl.Int8).mean()
        ),
        "challenger_only_correct": joined.filter(
            pl.col("challenger_correct") & ~pl.col("control_correct")
        ).height,
        "control_only_correct": joined.filter(
            pl.col("control_correct") & ~pl.col("challenger_correct")
        ).height,
        "prediction_disagreements": joined.filter(
            pl.col("control_predicted_up")
            != pl.col("challenger_predicted_up")
        ).height,
    }


def _split_experiment_frame(
    frame: pl.DataFrame,
    expected_seconds: tuple[int, ...],
    split: OracleBookSplitConfig,
) -> dict[str, pl.DataFrame]:
    ranges = {
        "fit": (split.fit_start, split.fit_end),
        "calibration": (split.calibration_start, split.calibration_end),
        "threshold": (split.threshold_start, split.threshold_end),
        "evaluation": (split.evaluation_start, split.evaluation_end),
    }
    cohorts = {
        name: range_frame(frame, start, end).sort(
            ["window_start", "market_id", "seconds_elapsed"]
        )
        for name, (start, end) in ranges.items()
    }
    for name, cohort in cohorts.items():
        _validate_exact_market_seconds(
            cohort,
            expected_seconds,
            cohort_name=name,
        )
        if cohort["label_up"].n_unique() != 2:
            raise RuntimeError(f"{name} cohort does not contain both labels")
    return cohorts


def _validate_exact_market_seconds(
    frame: pl.DataFrame,
    expected_seconds: tuple[int, ...],
    *,
    cohort_name: str,
) -> None:
    expected = set(expected_seconds)
    invalid = [
        str(row["market_id"])
        for row in (
            frame.group_by("market_id")
            .agg(
                pl.len().alias("rows"),
                pl.col("seconds_elapsed").n_unique().alias("unique_seconds"),
                pl.col("seconds_elapsed").unique().alias("seconds"),
            )
            .to_dicts()
        )
        if int(row["rows"]) != len(expected)
        or int(row["unique_seconds"]) != len(expected)
        or set(row["seconds"]) != expected
    ]
    if invalid:
        raise RuntimeError(
            f"{cohort_name} cohort has incomplete market seconds: {invalid[0]}"
        )


def _complete_context_market_ids(
    execution: pl.DataFrame,
) -> list[str]:
    expected = set(EXECUTION_CONTEXT_SECONDS)
    rows = (
        execution.filter(pl.col("strict_both_side_eligible_10"))
        .filter(pl.col("seconds_elapsed").is_in(EXECUTION_CONTEXT_SECONDS))
        .group_by("market_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("unique_seconds"),
            pl.col("seconds_elapsed").unique().alias("seconds"),
        )
        .to_dicts()
    )
    return sorted(
        str(row["market_id"])
        for row in rows
        if int(row["rows"]) == len(expected)
        and int(row["unique_seconds"]) == len(expected)
        and set(row["seconds"]) == expected
    )


def _complete_timing_frame(
    strict_features: pl.DataFrame,
    complete_book_market_ids: list[str],
) -> pl.DataFrame:
    candidate = strict_features.filter(
        pl.col("market_id").is_in(complete_book_market_ids)
        & pl.col("seconds_elapsed").is_in(TIMING_DECISION_SECONDS)
    )
    complete_ids = [
        str(row["market_id"])
        for row in (
            candidate.group_by("market_id")
            .agg(
                pl.len().alias("rows"),
                pl.col("seconds_elapsed").n_unique().alias("unique_seconds"),
                pl.col("seconds_elapsed").unique().alias("seconds"),
            )
            .to_dicts()
        )
        if int(row["rows"]) == len(TIMING_DECISION_SECONDS)
        and int(row["unique_seconds"]) == len(TIMING_DECISION_SECONDS)
        and set(row["seconds"]) == set(TIMING_DECISION_SECONDS)
    ]
    return candidate.filter(pl.col("market_id").is_in(complete_ids)).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )


def _load_and_validate_core_config(
    config: OracleBookBenchmarkConfig,
) -> CoreTrainingConfig:
    core_config = load_core_config(config.core_config)
    if core_config.data.source_contract != CORE_ORACLE_SOURCE_CONTRACT:
        raise RuntimeError("matched benchmark requires btc_core_oracle_v1")
    if (
        core_config.data.sample_interval_seconds != 5
        or core_config.data.min_seconds_after_open != 120
        or 300 - core_config.data.min_seconds_before_close != 140
    ):
        raise RuntimeError(
            "matched benchmark requires exact 120-140 second oracle features"
        )
    return core_config


def _validate_core_frame(frame: pl.DataFrame) -> None:
    required = {
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
        "binance_sign_up",
        *CORE_MATURE_REVERSAL_ORACLE_FEATURES,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            "core oracle feature cache is missing columns: " + ", ".join(missing)
        )
    duplicate = (
        frame.group_by("market_id", "observed_at")
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise RuntimeError("core oracle feature cache has duplicate row keys")
    _require_finite_features(frame, CORE_MATURE_REVERSAL_ORACLE_FEATURES)


def _validate_model_features(frame: pl.DataFrame) -> None:
    _require_finite_features(
        frame,
        CORE_MATURE_REVERSAL_ORACLE_FEATURES + STRICT_BOOK_V2_FEATURES,
    )


def _require_finite_features(
    frame: pl.DataFrame,
    features: list[str],
) -> None:
    invalid = frame.filter(
        pl.any_horizontal(
            [
                pl.col(feature).is_null() | ~pl.col(feature).is_finite()
                for feature in features
            ]
        )
    )
    if invalid.height:
        raise RuntimeError("matched benchmark contains missing or non-finite features")


def _load_execution_evidence(
    config: ExecutionEvidenceConfig,
    manifest: dict[str, Any],
) -> pl.DataFrame:
    columns = [
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "fee_rate",
        "quality_flags",
        "up_provider_received_at",
        "up_best_bid",
        "up_best_ask",
        "up_best_bid_size",
        "up_best_ask_size",
        "up_bid_depth",
        "up_ask_depth",
        "up_ask_vwap_5",
        "up_ask_vwap_10",
        "up_imbalance",
        "down_provider_received_at",
        "down_best_bid",
        "down_best_ask",
        "down_best_bid_size",
        "down_best_ask_size",
        "down_bid_depth",
        "down_ask_depth",
        "down_ask_vwap_5",
        "down_ask_vwap_10",
        "down_imbalance",
        "up_side_fresh",
        "down_side_fresh",
        "strict_both_side_eligible",
        "strict_both_side_eligible_10",
    ]
    paths = [
        config.output_dir / partition["path"]
        for partition in manifest["partitions"]
    ]
    return (
        pl.scan_parquet(paths)
        .select(columns)
        .collect()
        .sort(["window_start", "market_id", "seconds_elapsed"])
    )


def _validate_strict_execution_contract(frame: pl.DataFrame) -> None:
    duplicate = (
        frame.group_by("market_id", "observed_at")
        .len()
        .filter(pl.col("len") != 1)
    )
    if duplicate.height:
        raise RuntimeError("execution evidence has duplicate global row keys")
    best_price_columns = [
        "up_best_bid",
        "up_best_ask",
        "down_best_bid",
        "down_best_ask",
    ]
    vwap_columns = [
        "up_ask_vwap_5",
        "up_ask_vwap_10",
        "down_ask_vwap_5",
        "down_ask_vwap_10",
    ]
    size_columns = [
        "up_best_bid_size",
        "up_best_ask_size",
        "up_bid_depth",
        "up_ask_depth",
        "down_best_bid_size",
        "down_best_ask_size",
        "down_bid_depth",
        "down_ask_depth",
    ]
    numeric_columns = [
        *best_price_columns,
        *vwap_columns,
        *size_columns,
        "up_imbalance",
        "down_imbalance",
    ]
    strict = frame.filter(pl.col("strict_both_side_eligible_10"))
    invalid = strict.filter(
        ~pl.col("strict_both_side_eligible")
        | ~pl.col("up_side_fresh")
        | ~pl.col("down_side_fresh")
        | (
            (
                pl.col("quality_flags")
                & STRICT_BOTH_SIDE_TEN_SHARE_QUALITY_MASK
            )
            != 0
        )
        | pl.any_horizontal(
            [
                pl.col(column).is_null() | ~pl.col(column).is_finite()
                for column in numeric_columns
            ]
        )
        | pl.any_horizontal(
            [
                (pl.col(column) < 0) | (pl.col(column) > 1)
                for column in best_price_columns
            ]
        )
        | pl.any_horizontal(
            [
                (pl.col(column) <= 0) | (pl.col(column) > 1)
                for column in vwap_columns
            ]
        )
        | pl.any_horizontal(
            [pl.col(column) < 0 for column in size_columns]
        )
        | (pl.col("up_best_bid") > pl.col("up_best_ask"))
        | (pl.col("down_best_bid") > pl.col("down_best_ask"))
        | pl.col("up_provider_received_at").is_null()
        | pl.col("down_provider_received_at").is_null()
        | (pl.col("up_provider_received_at") > pl.col("observed_at"))
        | (pl.col("down_provider_received_at") > pl.col("observed_at"))
        | (
            pl.col("up_provider_received_at")
            < pl.col("observed_at") - pl.duration(seconds=2)
        )
        | (
            pl.col("down_provider_received_at")
            < pl.col("observed_at") - pl.duration(seconds=2)
        )
    )
    if invalid.height:
        raise RuntimeError(
            "strict ten-share execution evidence violates freshness, quality, "
            "causality, or VWAP completeness"
        )


def _execution_economics_frame(execution: pl.DataFrame) -> pl.DataFrame:
    return execution.select(
        "market_id",
        "observed_at",
        "fee_rate",
        "up_ask_vwap_5",
        "down_ask_vwap_5",
        "up_ask_vwap_10",
        "down_ask_vwap_10",
        "strict_both_side_eligible",
        "strict_both_side_eligible_10",
        pl.lit(True).alias("execution_evidence_available"),
    )


def _attach_execution_economics(
    scored: pl.DataFrame,
    economics: pl.DataFrame,
) -> pl.DataFrame:
    joined = scored.join(
        economics,
        on=["market_id", "observed_at"],
        how="left",
        validate="1:1",
    ).with_columns(
        pl.col("strict_both_side_eligible").fill_null(False),
        pl.col("strict_both_side_eligible_10").fill_null(False),
        pl.col("execution_evidence_available").fill_null(False),
    )
    if joined.filter(~pl.col("strict_both_side_eligible_10")).height:
        raise RuntimeError("matched prediction lost strict ten-share execution")
    return joined


def _universe_markets(
    core_frame: pl.DataFrame,
    start: datetime,
    end: datetime,
    *,
    expected_seconds: tuple[int, ...],
) -> int:
    selected = core_frame.filter(
        (pl.col("window_start") >= start)
        & (pl.col("window_start") < end)
        & pl.col("seconds_elapsed").is_in(expected_seconds)
    )
    _validate_exact_market_seconds(
        selected,
        expected_seconds,
        cohort_name="universal evaluation",
    )
    return selected["market_id"].n_unique()


def _cohort_payload(
    frame: pl.DataFrame,
    expected_seconds: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "label_up_markets": frame.filter(pl.col("label_up") == 1)[
            "market_id"
        ].n_unique(),
        "label_down_markets": frame.filter(pl.col("label_up") == 0)[
            "market_id"
        ].n_unique(),
        "expected_seconds": list(expected_seconds),
        "minimum_window_start": frame["window_start"].min().isoformat(),
        "maximum_window_start": frame["window_start"].max().isoformat(),
        "row_key_sha256": _row_key_sha256(frame),
        "core_oracle_matrix_sha256": _matrix_sha256(
            frame,
            CORE_MATURE_REVERSAL_ORACLE_FEATURES,
        ),
        "core_oracle_book_matrix_sha256": _matrix_sha256(
            frame,
            CORE_MATURE_REVERSAL_ORACLE_FEATURES + STRICT_BOOK_V2_FEATURES,
        ),
    }


def _row_key_sha256(frame: pl.DataFrame) -> str:
    return _frame_sha256(
        frame,
        ["market_id", "observed_at", "seconds_elapsed", "label_up"],
    )


def _matrix_sha256(frame: pl.DataFrame, features: list[str]) -> str:
    return _frame_sha256(
        frame,
        [
            "market_id",
            "observed_at",
            "seconds_elapsed",
            "label_up",
            *features,
        ],
    )


def _frame_sha256(frame: pl.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    ordered = frame.select(columns).sort(
        ["market_id", "observed_at", "seconds_elapsed"]
    )
    for row in ordered.iter_rows(named=False):
        normalized = [
            value.isoformat() if isinstance(value, datetime) else value
            for value in row
        ]
        digest.update(
            json.dumps(
                normalized,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _require_identical_keys(
    frames: dict[str, pl.DataFrame],
) -> None:
    keys = ["market_id", "observed_at", "seconds_elapsed"]
    iterator = iter(frames.items())
    reference_name, reference_frame = next(iterator)
    reference = reference_frame.select(keys).sort(keys)
    for name, frame in iterator:
        candidate = frame.select(keys).sort(keys)
        if (
            candidate.height != reference.height
            or reference.join(candidate, on=keys, how="anti").height
            or candidate.join(reference, on=keys, how="anti").height
        ):
            raise RuntimeError(
                f"matched row keys differ between {reference_name} and {name}"
            )


def _split_payload(split: OracleBookSplitConfig) -> dict[str, Any]:
    return {
        "fit": {
            "start": split.fit_start.isoformat(),
            "end": split.fit_end.isoformat(),
        },
        "calibration": {
            "start": split.calibration_start.isoformat(),
            "end": split.calibration_end.isoformat(),
        },
        "threshold": {
            "start": split.threshold_start.isoformat(),
            "end": split.threshold_end.isoformat(),
        },
        "evaluation": {
            "start": split.evaluation_start.isoformat(),
            "end": split.evaluation_end.isoformat(),
        },
    }


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text)
    temporary.replace(path)


def _render_report(payload: dict[str, Any]) -> str:
    experiment_sections = []
    for name, experiment in payload["experiments"].items():
        cohort_rows = "".join(
            "<tr>"
            f"<td>{html.escape(cohort_name)}</td>"
            f"<td>{cohort['markets']:,}</td>"
            f"<td>{cohort['rows']:,}</td>"
            f"<td><code>{cohort['row_key_sha256']}</code></td>"
            "</tr>"
            for cohort_name, cohort in experiment["cohorts"].items()
        )
        arm_rows = ""
        for arm_name, arm in experiment["arms"].items():
            evaluation = arm["evaluation"]
            policy = evaluation["policy"]
            five = evaluation["execution_by_size"]["vwap5_five_share"]
            ten = evaluation["execution_by_size"]["vwap10_ten_share"]
            arm_rows += (
                "<tr>"
                f"<td>{html.escape(arm_name)}</td>"
                f"<td>{policy['markets']:,}</td>"
                f"<td>{policy['coverage']:.2%}</td>"
                f"<td>{policy['accuracy']:.2%}</td>"
                f"<td>{policy['wilson_lower_95']:.2%}</td>"
                f"<td>{_format_number(five['realized_net_expectancy_per_trade'])}</td>"
                f"<td>{_format_number(ten['realized_net_expectancy_per_trade'])}</td>"
                "</tr>"
            )
        experiment_sections.append(
            "<section>"
            f"<h2>{html.escape(name)}</h2>"
            "<h3>Frozen matched cohorts</h3>"
            "<table><thead><tr><th>Cohort</th><th>Markets</th><th>Rows</th>"
            "<th>Row-key SHA-256</th></tr></thead>"
            f"<tbody>{cohort_rows}</tbody></table>"
            "<h3>Evaluation</h3>"
            "<table><thead><tr><th>Arm</th><th>Trades</th><th>Matched coverage</th>"
            "<th>Accuracy</th><th>Wilson lower</th><th>VWAP5 / 5-share net</th>"
            "<th>VWAP10 / 10-share net</th></tr></thead>"
            f"<tbody>{arm_rows}</tbody></table>"
            "</section>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>BTC oracle/book matched benchmark</title>"
        "<style>"
        "body{font:14px system-ui;margin:32px;max-width:1400px;color:#17202a}"
        "table{border-collapse:collapse;width:100%;margin:12px 0 28px}"
        "th,td{border:1px solid #ccd1d1;padding:8px;text-align:left}"
        "th{background:#f4f6f7}code{font-size:11px;word-break:break-all}"
        ".notice{padding:12px;background:#fff3cd;border:1px solid #ffe69c}"
        "</style></head><body>"
        "<h1>BTC core + oracle versus core + oracle + book</h1>"
        "<p class='notice'>Development diagnostic only. No runtime export, deployment, "
        "or trading process was authorized.</p>"
        f"<p>{html.escape(payload['evaluation']['note'])}</p>"
        f"<p>Base features: {payload['model_contract']['base_feature_count']}; "
        f"challenger features: {payload['model_contract']['challenger_feature_count']}; "
        "recency half-life: 28 days.</p>"
        + "".join(experiment_sections)
        + "</body></html>"
    )


def _format_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"
