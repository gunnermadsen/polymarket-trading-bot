from __future__ import annotations

import html
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from .chainlink_oi_config import (
    CHAINLINK_FULL_CANDIDATE,
    CHAINLINK_FULL_OI_CANDIDATE,
    LONG_HISTORY_CANDLE_CANDIDATE,
    ChainlinkOiBenchmarkConfig,
)
from .chainlink_oi_features import (
    BINANCE_OI_FEATURES,
    CHAINLINK_CANDLE_FEATURES,
    CHAINLINK_EXTERNAL_FEATURES,
    POINT_KEY_COLUMNS,
    ExternalSourceFrames,
    derive_chainlink_candle_feature_frame,
    derive_chainlink_oi_feature_frames,
    extract_external_source_frames,
)
from .champion_vwap_benchmark import (
    _execution_metrics,
    _first_champion_crossings,
    _price_diagnostics,
)
from .core_config import CoreTrainingConfig, load_core_config
from .core_evaluation import (
    FIRST_CROSSING_TIME_BANDS,
    classification_metrics,
    scored_prediction_rows,
)
from .core_extract import (
    configure_read_only_connection,
    database_connection,
    file_sha256,
    write_json_atomic,
)
from .core_features import CORE_BOUNDARY_ENRICHED_FEATURES, CORE_ORACLE_FEATURES
from .core_training import (
    MARKET_EQUAL_ROW_WEIGHT_POLICY,
    CandidateSpec,
    FittedCoreModel,
    ProbabilityCalibrator,
    fit_model,
    fit_probability_calibrator,
    range_frame,
)

CONTROL = "frozen_boundary_alignment_champion"
CANDIDATES = (
    CHAINLINK_FULL_CANDIDATE,
    CHAINLINK_FULL_OI_CANDIDATE,
    LONG_HISTORY_CANDLE_CANDIDATE,
)
ALL_MODELS = (CONTROL, *CANDIDATES)
SCHEMA_VERSION = "btc-chainlink-oi-champion-benchmark-v1"
EXTERNAL_CACHE_SCHEMA_VERSION = "btc-chainlink-oi-external-cache-v1"


@dataclass
class ChainlinkOiCandidateBundle:
    name: str
    feature_names: tuple[str, ...]
    model: FittedCoreModel
    calibrators: dict[str, ProbabilityCalibrator]
    confidence_threshold: float

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        raw_logit = self.model.raw_logit(frame)
        elapsed = frame["seconds_elapsed"].to_numpy()
        output = np.full(frame.height, np.nan, dtype=np.float64)
        for name, start, end in FIRST_CROSSING_TIME_BANDS:
            mask = (elapsed >= start) & (elapsed < end)
            output[mask] = self.calibrators[name].probability(raw_logit[mask])
        if not np.isfinite(output).all():
            raise RuntimeError(f"{self.name} produced uncalibrated decision rows")
        return output


def run_chainlink_oi_benchmark(
    config: ChainlinkOiBenchmarkConfig,
) -> tuple[Path, dict[str, Any]]:
    print("chainlink-oi: validating core, champion, and execution artifacts", flush=True)
    core_config = load_core_config(config.paths.core_config)
    core, core_lineage = _load_core_features(config, core_config)
    execution, execution_lineage = _load_execution_evidence(config)

    print("chainlink-oi: loading immutable external-source cache", flush=True)
    external, external_lineage = _load_or_extract_external_sources(config)

    print("chainlink-oi: deriving strict causal candidate cohorts", flush=True)
    short_core = _range(core, config.windows.short_fit_start, config.windows.source_range_end)
    chainlink_frame, chainlink_oi_frame = derive_chainlink_oi_feature_frames(
        short_core,
        external.refprice,
        external.candles,
        external.open_interest,
        refprice_max_age_seconds=config.staleness.refprice_seconds,
        candle_max_age_seconds=config.staleness.candles_seconds,
        open_interest_max_age_seconds=config.staleness.open_interest_seconds,
    )
    candle_frame = derive_chainlink_candle_feature_frame(
        core,
        external.candles,
        candle_max_age_seconds=config.staleness.candles_seconds,
    )
    _validate_common_short_cohort(chainlink_frame, chainlink_oi_frame)

    frames = {
        CHAINLINK_FULL_CANDIDATE: chainlink_frame,
        CHAINLINK_FULL_OI_CANDIDATE: chainlink_oi_frame,
        LONG_HISTORY_CANDLE_CANDIDATE: candle_frame,
    }
    feature_sets = _candidate_feature_sets()

    print("chainlink-oi: fitting three frozen-parameter candidate models", flush=True)
    bundles: dict[str, ChainlinkOiCandidateBundle] = {}
    training: dict[str, Any] = {}
    for name in CANDIDATES:
        bundle, evidence = _fit_candidate(
            name,
            frames[name],
            feature_sets[name],
            config,
            core_config,
        )
        bundles[name] = bundle
        training[name] = evidence
        print(
            f"chainlink-oi: fitted {name}; rows={evidence['fit']['rows']:,}; "
            f"markets={evidence['fit']['markets']:,}",
            flush=True,
        )

    common_evaluation = _range(
        chainlink_frame,
        config.windows.economic_confirmation_start,
        config.windows.directional_stress_end,
    )
    common_keys = common_evaluation.select(*POINT_KEY_COLUMNS)
    evaluation_frames = {
        CHAINLINK_FULL_CANDIDATE: common_evaluation,
        CHAINLINK_FULL_OI_CANDIDATE: _match_keys(
            chainlink_oi_frame,
            common_keys,
            "Chainlink-plus-OI evaluation",
        ),
        LONG_HISTORY_CANDLE_CANDIDATE: _match_keys(
            candle_frame,
            common_keys,
            "long-history candle evaluation",
        ),
    }
    _validate_evaluation_key_identity(evaluation_frames)

    print("chainlink-oi: scoring frozen control and native first crossings", flush=True)
    champion_model = json.loads(config.paths.champion_model.read_text())
    champion_features = tuple(champion_model["features"]["names"])
    champion_first = _champion_first_crossings(common_evaluation, champion_model, champion_features)
    scored_rows = {
        name: scored_prediction_rows(frame, bundles[name].probability(frame))
        for name, frame in evaluation_frames.items()
    }
    native_first = {
        name: _first_crossings_from_scored(rows, config.model.confidence_threshold)
        for name, rows in scored_rows.items()
    }

    windows = {
        "economic_confirmation": (
            config.windows.economic_confirmation_start,
            config.windows.economic_confirmation_end,
        ),
        "directional_stress": (
            config.windows.directional_stress_start,
            config.windows.directional_stress_end,
        ),
        "post_calibration_combined": (
            config.windows.economic_confirmation_start,
            config.windows.directional_stress_end,
        ),
    }
    directional: dict[str, Any] = {}
    native_economics: dict[str, Any] = {}
    native_pairing: dict[str, Any] = {}
    matched_proposals: dict[str, Any] = {}
    matched_frames: dict[str, dict[str, pl.DataFrame]] = {}
    for window_name, (start, end) in windows.items():
        eligible = _range(common_evaluation, start, end)["market_id"].n_unique()
        directional[window_name] = {
            CONTROL: _direction_metrics(_range(champion_first, start, end), eligible),
            **{
                name: _direction_metrics(_range(native_first[name], start, end), eligible)
                for name in CANDIDATES
            },
        }
        native_economics[window_name] = {
            CONTROL: _native_economic_metrics(
                _range(champion_first, start, end),
                execution,
                eligible,
                config.model.quantity,
            ),
            **{
                name: _native_economic_metrics(
                    _range(native_first[name], start, end),
                    execution,
                    eligible,
                    config.model.quantity,
                )
                for name in CANDIDATES
            },
        }
        native_pairing[window_name] = {
            name: _native_champion_market_pairing(
                _range(champion_first, start, end),
                _range(native_first[name], start, end),
                execution,
                config.model.quantity,
            )
            for name in CANDIDATES
        }
        proposal_result, proposal_frames = _matched_proposal_window(
            _range(champion_first, start, end),
            {name: _range(scored_rows[name], start, end) for name in CANDIDATES},
            execution,
            config,
        )
        matched_proposals[window_name] = proposal_result
        matched_frames[window_name] = proposal_frames

    confirmation_folds: dict[str, Any] = {}
    native_confirmation_folds: dict[str, Any] = {}
    for name, start, end in (
        (
            "2026-07-16_to_2026-07-18",
            config.windows.economic_confirmation_start,
            datetime(2026, 7, 18, tzinfo=UTC),
        ),
        (
            "2026-07-18_to_2026-07-21",
            datetime(2026, 7, 18, tzinfo=UTC),
            config.windows.economic_confirmation_end,
        ),
    ):
        fold, _ = _matched_proposal_window(
            _range(champion_first, start, end),
            {candidate: _range(scored_rows[candidate], start, end) for candidate in CANDIDATES},
            execution,
            config,
        )
        confirmation_folds[name] = fold
        eligible = _range(common_evaluation, start, end)["market_id"].n_unique()
        native_confirmation_folds[name] = {
            CONTROL: _native_economic_metrics(
                _range(champion_first, start, end),
                execution,
                eligible,
                config.model.quantity,
            ),
            **{
                candidate: _native_economic_metrics(
                    _range(native_first[candidate], start, end),
                    execution,
                    eligible,
                    config.model.quantity,
                )
                for candidate in CANDIDATES
            },
        }

    promotion = _promotion_decision(
        matched_proposals["economic_confirmation"],
        native_economics["economic_confirmation"],
        native_pairing["economic_confirmation"],
        directional["directional_stress"],
        confirmation_folds,
        native_confirmation_folds,
        config,
    )
    selected_candidate = promotion["selected_candidate"]

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.paths.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_artifacts: dict[str, Any] = {}
    for name, bundle in bundles.items():
        path = run_dir / f"{name}-training-model.joblib"
        joblib.dump(bundle, path, compress=3)
        model_artifacts[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "paper_promotion_qualified": name == selected_candidate,
            "runtime_exported": False,
        }
        native_first[name].write_parquet(
            run_dir / f"{name}-native-first-crossings.parquet",
            compression="zstd",
        )
    champion_first.write_parquet(
        run_dir / "frozen-champion-native-first-crossings.parquet",
        compression="zstd",
    )
    for name, frame in matched_frames["economic_confirmation"].items():
        frame.write_parquet(
            run_dir / f"{name}-matched-confirmation-proposals.parquet",
            compression="zstd",
        )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "challenger_selected" if selected_candidate else "champion_retained",
        "selected_candidate": selected_candidate,
        "evaluation_note": config.evaluation_note,
        "evidence_classification": (
            "consumed chronological development evidence; not independent live-capital qualification"
        ),
        "runtime_changed": False,
        "deployment_changed": False,
        "lineage": {
            "configuration": str(config.source_path),
            "configuration_sha256": file_sha256(config.source_path),
            "source_schema_revision": config.source_schema_revision,
            "core": core_lineage,
            "external": external_lineage,
            "execution": execution_lineage,
            "champion_model": str(config.paths.champion_model),
            "champion_model_sha256": config.champion.model_sha256,
            "champion_manifest_sha256": config.champion.manifest_sha256,
        },
        "data_contract": {
            "source_range": _window_payload(
                config.windows.source_range_start,
                config.windows.source_range_end,
            ),
            "strict_common_source_range": _window_payload(
                config.windows.short_fit_start,
                config.windows.source_range_end,
            ),
            "fit_short": _window_payload(
                config.windows.short_fit_start,
                config.windows.short_fit_end,
            ),
            "fit_long": _window_payload(
                config.windows.source_range_start,
                config.windows.short_fit_end,
            ),
            "calibration": _window_payload(
                config.windows.calibration_start,
                config.windows.calibration_end,
            ),
            "economic_confirmation": _window_payload(
                config.windows.economic_confirmation_start,
                config.windows.economic_confirmation_end,
            ),
            "directional_stress": _window_payload(
                config.windows.directional_stress_start,
                config.windows.directional_stress_end,
            ),
            "pmxt_role": "evaluation_only",
            "vwap_imputation": False,
            "missingness_features": False,
            "same_short_rows_for_chainlink_ab": True,
            "same_post_calibration_rows_for_all_challengers": True,
            "quantity": config.model.quantity,
            "fee_formula": "fee_rate * price * (1 - price)",
        },
        "cohorts": {
            "core": _cohort(core),
            "short_chainlink_common": _cohort(chainlink_frame),
            "long_history_candle": _cohort(candle_frame),
            "post_calibration_common": _cohort(common_evaluation),
            "strict_execution_evidence": _cohort(execution),
        },
        "features": {name: list(values) for name, values in feature_sets.items()},
        "training": training,
        "directional": directional,
        "native_first_crossing_economics": native_economics,
        "native_champion_market_pairing": native_pairing,
        "matched_champion_proposals": matched_proposals,
        "confirmation_folds": confirmation_folds,
        "native_confirmation_folds": native_confirmation_folds,
        "promotion": promotion,
        "model_artifacts": model_artifacts,
    }
    write_json_atomic(run_dir / "benchmark.json", result)
    (run_dir / "report.html").write_text(_report_html(result), encoding="utf-8")
    print(
        f"chainlink-oi: complete; selected={selected_candidate or 'none'}; "
        "runtime/deployment unchanged",
        flush=True,
    )
    return run_dir, result


def _load_core_features(
    config: ChainlinkOiBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    path = core_config.paths.development_feature_data
    metadata_path = path.with_suffix(".metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise RuntimeError("core feature cache and metadata are required")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("feature_file_sha256") != file_sha256(path):
        raise RuntimeError("core feature cache hash mismatch")
    contract = metadata.get("build_contract", {})
    if (
        _utc(contract.get("range_start")) != config.windows.source_range_start
        or _utc(contract.get("range_end")) != config.windows.source_range_end
        or contract.get("source_contract") != "btc_core_oracle_v1"
    ):
        raise RuntimeError("core feature cache does not match the benchmark range/contract")
    champion = json.loads(config.paths.champion_model.read_text())
    needed = tuple(
        dict.fromkeys(
            (
                "market_id",
                "window_start",
                "observed_at",
                "seconds_elapsed",
                "label_up",
                "binance_sign_up",
                "btc_close",
                "opening_boundary",
                "btc_return_30s_bps",
                "btc_path_from_window_open_bps",
                "oracle_model_eligible",
                *champion["features"]["names"],
                *CORE_BOUNDARY_ENRICHED_FEATURES,
                *CORE_ORACLE_FEATURES,
            )
        )
    )
    core = (
        pl.scan_parquet(path)
        .filter(pl.col("oracle_model_eligible"))
        .select(*needed)
        .collect()
        .sort(["window_start", "market_id", "seconds_elapsed"])
    )
    if core.select(pl.struct(POINT_KEY_COLUMNS).n_unique()).item() != core.height:
        raise RuntimeError("core feature cache contains duplicate point keys")
    return core, {
        "path": str(path),
        "sha256": file_sha256(path),
        "metadata": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "rows": core.height,
        "markets": core["market_id"].n_unique(),
    }


def _load_or_extract_external_sources(
    config: ChainlinkOiBenchmarkConfig,
) -> tuple[ExternalSourceFrames, dict[str, Any]]:
    cache = config.paths.shared_cache / "external"
    manifest_path = cache / "manifest.json"
    files = {
        "refprice": cache / "chainlink-refprice.parquet",
        "candles": cache / "chainlink-one-minute-candles.parquet",
        "open_interest": cache / "binance-five-minute-open-interest.parquet",
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        _validate_external_manifest(config, manifest, files)
        return (
            ExternalSourceFrames(
                refprice=pl.read_parquet(files["refprice"]),
                candles=pl.read_parquet(files["candles"]),
                open_interest=pl.read_parquet(files["open_interest"]),
            ),
            {
                **manifest,
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
            },
        )
    if any(path.exists() for path in files.values()):
        raise RuntimeError("external source cache is partial and has no immutable manifest")
    cache.mkdir(parents=True, exist_ok=True)
    with database_connection() as connection:
        configure_read_only_connection(connection)
        frames = extract_external_source_frames(
            connection,
            config.package_root,
            range_start=config.windows.source_range_start,
            range_end=config.windows.source_range_end,
            refprice_feed_id=config.sources.refprice_feed_id,
            candle_symbol=config.sources.candle_symbol,
            open_interest_symbol=config.sources.open_interest_symbol,
        )
    for name, frame in (
        ("refprice", frames.refprice),
        ("candles", frames.candles),
        ("open_interest", frames.open_interest),
    ):
        if frame.is_empty():
            raise RuntimeError(f"external source {name} is empty")
        frame.write_parquet(files[name], compression="zstd")
    manifest = {
        "schema_version": EXTERNAL_CACHE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "range_start": config.windows.source_range_start.isoformat(),
        "range_end": config.windows.source_range_end.isoformat(),
        "source_schema_revision": config.source_schema_revision,
        "refprice_feed_id": config.sources.refprice_feed_id,
        "sources": {
            name: {
                "path": path.name,
                "rows": getattr(frames, name).height,
                "sha256": file_sha256(path),
            }
            for name, path in files.items()
        },
        "sql_sha256": {
            "refprice": file_sha256(config.paths.refprice_source_sql),
            "candles": file_sha256(config.paths.candles_source_sql),
            "open_interest": file_sha256(config.paths.open_interest_source_sql),
        },
    }
    write_json_atomic(manifest_path, manifest)
    return frames, {
        **manifest,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }


def _validate_external_manifest(
    config: ChainlinkOiBenchmarkConfig,
    manifest: dict[str, Any],
    files: dict[str, Path],
) -> None:
    if (
        manifest.get("schema_version") != EXTERNAL_CACHE_SCHEMA_VERSION
        or _utc(manifest.get("range_start")) != config.windows.source_range_start
        or _utc(manifest.get("range_end")) != config.windows.source_range_end
        or manifest.get("source_schema_revision") != config.source_schema_revision
        or manifest.get("refprice_feed_id") != config.sources.refprice_feed_id
    ):
        raise RuntimeError("external source cache contract mismatch")
    expected_sql = {
        "refprice": file_sha256(config.paths.refprice_source_sql),
        "candles": file_sha256(config.paths.candles_source_sql),
        "open_interest": file_sha256(config.paths.open_interest_source_sql),
    }
    if manifest.get("sql_sha256") != expected_sql:
        raise RuntimeError("external source SQL changed after cache creation")
    for name, path in files.items():
        record = manifest.get("sources", {}).get(name, {})
        if not path.is_file() or record.get("sha256") != file_sha256(path):
            raise RuntimeError(f"external source cache hash mismatch: {name}")


def _load_execution_evidence(
    config: ChainlinkOiBenchmarkConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    manifest_path = config.paths.execution_evidence / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("execution evidence manifest is required")
    manifest = json.loads(manifest_path.read_text())
    if (
        _utc(manifest.get("range_start")) != config.windows.short_fit_start
        or _utc(manifest.get("range_end")) != config.windows.source_range_end
        or int(manifest.get("freshness_seconds", -1)) != config.staleness.execution_book_seconds
    ):
        raise RuntimeError("execution evidence range/freshness contract mismatch")
    partitions: list[Path] = []
    for record in manifest.get("partitions", []):
        path = config.paths.execution_evidence / record["path"]
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"execution evidence hash mismatch: {path.name}")
        partitions.append(path)
    if not partitions:
        raise RuntimeError("execution evidence has no verified partitions")
    execution = (
        pl.scan_parquet(partitions)
        .filter(pl.col("strict_both_side_eligible_10"))
        .select(
            "market_id",
            "window_start",
            "observed_at",
            "seconds_elapsed",
            "fee_rate",
            "up_ask_vwap_5",
            "up_ask_vwap_10",
            "down_ask_vwap_5",
            "down_ask_vwap_10",
        )
        .collect()
        .unique(subset=list(POINT_KEY_COLUMNS))
        .sort(["window_start", "market_id", "seconds_elapsed"])
    )
    return execution, {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "partitions_verified": len(partitions),
        "rows": execution.height,
        "markets": execution["market_id"].n_unique(),
    }


def _candidate_feature_sets() -> dict[str, tuple[str, ...]]:
    base = tuple(dict.fromkeys((*CORE_BOUNDARY_ENRICHED_FEATURES, *CORE_ORACLE_FEATURES)))
    return {
        CHAINLINK_FULL_CANDIDATE: tuple(dict.fromkeys((*base, *CHAINLINK_EXTERNAL_FEATURES))),
        CHAINLINK_FULL_OI_CANDIDATE: tuple(
            dict.fromkeys((*base, *CHAINLINK_EXTERNAL_FEATURES, *BINANCE_OI_FEATURES))
        ),
        LONG_HISTORY_CANDLE_CANDIDATE: tuple(dict.fromkeys((*base, *CHAINLINK_CANDLE_FEATURES))),
    }


def _fit_candidate(
    name: str,
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
    config: ChainlinkOiBenchmarkConfig,
    core_config: CoreTrainingConfig,
) -> tuple[ChainlinkOiCandidateBundle, dict[str, Any]]:
    fit_start = (
        config.windows.source_range_start
        if name == LONG_HISTORY_CANDLE_CANDIDATE
        else config.windows.short_fit_start
    )
    fit = range_frame(frame, fit_start, config.windows.short_fit_end)
    calibration = range_frame(
        frame,
        config.windows.calibration_start,
        config.windows.calibration_end,
    )
    _assert_disjoint_markets(fit, calibration, name)
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} is missing model features: " + ", ".join(missing))
    _validate_model_feature_values(frame, feature_names, name)
    spec = CandidateSpec(
        name=name,
        family="histogram",
        feature_names=feature_names,
        row_weight_policy=MARKET_EQUAL_ROW_WEIGHT_POLICY,
    )
    parameters = {
        "learning_rate": config.model.learning_rate,
        "max_iter": config.model.max_iter,
        "max_leaf_nodes": config.model.max_leaf_nodes,
        "min_samples_leaf": config.model.min_samples_leaf,
        "l2_regularization": config.model.l2_regularization,
    }
    started = time.perf_counter()
    model = fit_model(fit, spec, parameters, core_config)
    fit_seconds = time.perf_counter() - started
    calibrators: dict[str, ProbabilityCalibrator] = {}
    calibration_evidence: list[dict[str, Any]] = []
    for band_name, start, end in FIRST_CROSSING_TIME_BANDS:
        rows = calibration.filter(pl.col("seconds_elapsed").is_between(start, end, closed="left"))
        if rows.height < 100 or set(rows["label_up"].unique().to_list()) != {0, 1}:
            raise RuntimeError(f"{name} calibration band {band_name} is not statistically usable")
        calibrator = fit_probability_calibrator(model, rows, core_config, spec)
        if not calibrator.converged or calibrator.slope <= 0:
            raise RuntimeError(f"{name} calibration band {band_name} failed monotonic convergence")
        calibrators[band_name] = calibrator
        calibration_evidence.append(
            {
                "band": band_name,
                "seconds": [start, end],
                "rows": rows.height,
                "markets": rows["market_id"].n_unique(),
                "positive_rate": float(rows["label_up"].mean()),
                **asdict(calibrator),
            }
        )
    bundle = ChainlinkOiCandidateBundle(
        name=name,
        feature_names=feature_names,
        model=model,
        calibrators=calibrators,
        confidence_threshold=config.model.confidence_threshold,
    )
    return bundle, {
        "fit": _cohort(fit),
        "calibration": _cohort(calibration),
        "fit_seconds": fit_seconds,
        "feature_count": len(feature_names),
        "hyperparameters": parameters,
        "market_equal_row_weighting": True,
        "calibration_bands": calibration_evidence,
    }


def _champion_first_crossings(
    frame: pl.DataFrame,
    model: dict[str, Any],
    feature_names: tuple[str, ...],
) -> pl.DataFrame:
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise RuntimeError("frozen champion features are missing: " + ", ".join(missing))
    selected = _first_champion_crossings(
        frame.select(
            "market_id",
            "window_start",
            "observed_at",
            "seconds_elapsed",
            "label_up",
            *feature_names,
        ).sort(["market_id", "seconds_elapsed"]),
        model,
        feature_names,
    )
    return selected.with_columns(
        pl.col("champion_selected_up").cast(pl.Int8).alias("predicted_up"),
        pl.col("champion_probability_up").alias("probability_up"),
        pl.col("champion_selected_probability").alias("confidence"),
        (pl.col("champion_selected_up") == pl.col("label_up").cast(pl.Boolean)).alias("correct"),
    ).sort(["window_start", "market_id"])


def _direction_metrics(rows: pl.DataFrame, eligible_markets: int) -> dict[str, Any]:
    metrics = classification_metrics(rows, eligible_markets=eligible_markets)
    return {
        **metrics,
        "wins": int(rows["correct"].sum()) if rows.height else 0,
        "losses": int(rows.height - rows["correct"].sum()) if rows.height else 0,
        "median_entry_second": (float(rows["seconds_elapsed"].median()) if rows.height else None),
    }


def _native_economic_metrics(
    first_crossings: pl.DataFrame,
    execution: pl.DataFrame,
    eligible_markets: int,
    quantity: float,
) -> dict[str, Any]:
    joined = first_crossings.join(
        execution,
        on=list(POINT_KEY_COLUMNS),
        how="inner",
        validate="1:1",
    ).with_columns(pl.lit(True).alias("policy_selected"))
    attached = _attach_economics(joined, "predicted_up", quantity)
    metrics = _economic_metrics(attached, eligible_markets)
    return {
        **metrics,
        "native_trades": first_crossings.height,
        "book_qualified_trades": attached.height,
        "book_qualification_rate": (
            attached.height / first_crossings.height if first_crossings.height else 0.0
        ),
    }


def _native_champion_market_pairing(
    champion: pl.DataFrame,
    candidate: pl.DataFrame,
    execution: pl.DataFrame,
    quantity: float,
) -> dict[str, Any]:
    control = _attach_economics(
        champion.join(
            execution,
            on=list(POINT_KEY_COLUMNS),
            how="inner",
            validate="1:1",
        ).with_columns(pl.lit(True).alias("policy_selected")),
        "predicted_up",
        quantity,
    )
    paired = control.select(
        "market_id",
        pl.col("correct").alias("champion_correct"),
        pl.col("realized_net_pnl_5").alias("champion_pnl_5"),
    ).join(
        candidate.select(
            "market_id",
            pl.col("correct").alias("candidate_correct"),
        ),
        on="market_id",
        how="left",
        validate="1:1",
    )
    losses = paired.filter(~pl.col("champion_correct"))
    wins = paired.filter(pl.col("champion_correct"))
    rejected_losses = losses.filter(pl.col("candidate_correct").is_null())
    corrected_losses = losses.filter(pl.col("candidate_correct"))
    repeated_losses = losses.filter(~pl.col("candidate_correct"))
    retained_wins = wins.filter(pl.col("candidate_correct"))
    forgone_wins = wins.filter(pl.col("candidate_correct").is_null() | ~pl.col("candidate_correct"))
    saved_loss = (
        float((-rejected_losses["champion_pnl_5"]).sum()) if rejected_losses.height else 0.0
    )
    forgone_win = float(forgone_wins["champion_pnl_5"].sum()) if forgone_wins.height else 0.0
    return {
        "book_qualified_champion_markets": paired.height,
        "candidate_market_overlap": int(paired["candidate_correct"].is_not_null().sum()),
        "champion_losses": losses.height,
        "champion_wins": wins.height,
        "champion_losses_rejected_as_no_trade": rejected_losses.height,
        "champion_losses_corrected_by_direction": corrected_losses.height,
        "champion_losses_repeated": repeated_losses.height,
        "champion_loss_no_trade_rate": (
            rejected_losses.height / losses.height if losses.height else 0.0
        ),
        "champion_loss_avoidance_rate": (
            (rejected_losses.height + corrected_losses.height) / losses.height
            if losses.height
            else 0.0
        ),
        "champion_wins_retained": retained_wins.height,
        "champion_win_retention_rate": (retained_wins.height / wins.height if wins.height else 0.0),
        "saved_loss_dollars_from_no_trade": saved_loss,
        "forgone_win_dollars": forgone_win,
        "saved_loss_to_forgone_win_ratio": (
            saved_loss / forgone_win if forgone_win > 0 else (math.inf if saved_loss > 0 else 0.0)
        ),
    }


def _matched_proposal_window(
    champion: pl.DataFrame,
    challenger_scores: dict[str, pl.DataFrame],
    execution: pl.DataFrame,
    config: ChainlinkOiBenchmarkConfig,
) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    qualified_champion = champion.join(
        execution,
        on=list(POINT_KEY_COLUMNS),
        how="inner",
        validate="1:1",
    )
    if qualified_champion.is_empty():
        return {
            "eligible_champion_proposals": 0,
            "confirmation_days": 0,
            "models": {name: _empty_economic_metrics(0) for name in ALL_MODELS},
            "loss_rejection": {name: _empty_loss_rejection() for name in CANDIDATES},
        }, {}
    eligible = qualified_champion.height
    control = _attach_economics(
        qualified_champion.with_columns(pl.lit(True).alias("policy_selected")),
        "predicted_up",
        config.model.quantity,
    )
    models: dict[str, Any] = {CONTROL: _economic_metrics(control, eligible)}
    frames: dict[str, pl.DataFrame] = {CONTROL: control}
    rejection: dict[str, Any] = {}
    proposal_keys = qualified_champion.select(*POINT_KEY_COLUMNS)
    for name, scores in challenger_scores.items():
        candidate = proposal_keys.join(
            scores.select(
                *POINT_KEY_COLUMNS,
                "label_up",
                "predicted_up",
                "probability_up",
                "confidence",
                "correct",
            ),
            on=list(POINT_KEY_COLUMNS),
            how="inner",
            validate="1:1",
        )
        if candidate.height != eligible:
            raise RuntimeError(f"{name} cannot score every matched champion proposal")
        candidate = candidate.join(
            execution,
            on=list(POINT_KEY_COLUMNS),
            how="inner",
            validate="1:1",
        ).with_columns(
            (pl.col("confidence") >= config.model.confidence_threshold).alias("policy_selected")
        )
        attached = _attach_economics(candidate, "predicted_up", config.model.quantity)
        models[name] = _economic_metrics(attached, eligible)
        rejection[name] = _loss_rejection_metrics(control, attached)
        frames[name] = attached
    return {
        "eligible_champion_proposals": eligible,
        "confirmation_days": qualified_champion.select(
            pl.col("window_start").dt.date().n_unique()
        ).item(),
        "models": models,
        "loss_rejection": rejection,
    }, frames


def _attach_economics(
    frame: pl.DataFrame,
    direction_column: str,
    quantity: float,
) -> pl.DataFrame:
    return (
        frame.with_columns(
            pl.when(pl.col(direction_column).cast(pl.Boolean))
            .then(pl.col("up_ask_vwap_5"))
            .otherwise(pl.col("down_ask_vwap_5"))
            .alias("selected_ask_vwap_5"),
            pl.when(pl.col(direction_column).cast(pl.Boolean))
            .then(pl.col("up_ask_vwap_10"))
            .otherwise(pl.col("down_ask_vwap_10"))
            .alias("selected_ask_vwap_10"),
            (pl.col(direction_column).cast(pl.Int8) == pl.col("label_up")).alias("correct"),
        )
        .with_columns(
            (
                pl.col("fee_rate")
                * pl.col("selected_ask_vwap_5")
                * (1.0 - pl.col("selected_ask_vwap_5"))
            ).alias("fee_per_share_5"),
            (
                pl.col("fee_rate")
                * pl.col("selected_ask_vwap_10")
                * (1.0 - pl.col("selected_ask_vwap_10"))
            ).alias("fee_per_share_10"),
        )
        .with_columns(
            (
                (
                    pl.col("correct").cast(pl.Float64)
                    - pl.col("selected_ask_vwap_5")
                    - pl.col("fee_per_share_5")
                )
                * quantity
            ).alias("realized_net_pnl_5"),
            (
                (
                    pl.col("correct").cast(pl.Float64)
                    - pl.col("selected_ask_vwap_10")
                    - pl.col("fee_per_share_10")
                )
                * 10.0
            ).alias("realized_net_pnl_10"),
        )
    )


def _economic_metrics(frame: pl.DataFrame, eligible_markets: int) -> dict[str, Any]:
    selected = frame.filter(pl.col("policy_selected"))
    five = _execution_metrics(selected, "realized_net_pnl_5")
    ten = _execution_metrics(selected, "realized_net_pnl_10")
    return {
        "eligible_markets": eligible_markets,
        "trades": selected.height,
        "coverage": selected.height / eligible_markets if eligible_markets else 0.0,
        "wins": int(selected["correct"].sum()) if selected.height else 0,
        "losses": int(selected.height - selected["correct"].sum()) if selected.height else 0,
        "accuracy": float(selected["correct"].mean()) if selected.height else None,
        "expected_calibration_error": _selected_ece(selected),
        "five_share": {
            **five,
            "net_pnl_per_eligible_market": (
                five["net_pnl"] / eligible_markets if eligible_markets else 0.0
            ),
        },
        "ten_share_reporting_only": {
            **ten,
            "net_pnl_per_eligible_market": (
                ten["net_pnl"] / eligible_markets if eligible_markets else 0.0
            ),
        },
        "price": _price_diagnostics(selected),
        "fees_5": float((selected["fee_per_share_5"] * 5.0).sum()) if selected.height else 0.0,
    }


def _selected_ece(frame: pl.DataFrame) -> float | None:
    if frame.is_empty() or "probability_up" not in frame.columns:
        return None
    probability = frame["probability_up"].to_numpy()
    labels = frame["label_up"].to_numpy()
    order = np.argsort(probability)
    error = 0.0
    for indices in np.array_split(order, min(10, len(order))):
        if len(indices):
            error += (
                len(indices)
                / len(order)
                * abs(float(probability[indices].mean()) - float(labels[indices].mean()))
            )
    return float(error)


def _loss_rejection_metrics(
    control: pl.DataFrame,
    candidate: pl.DataFrame,
) -> dict[str, Any]:
    joined = control.select(
        *POINT_KEY_COLUMNS,
        pl.col("correct").alias("control_correct"),
        pl.col("predicted_up").alias("control_predicted_up"),
        pl.col("realized_net_pnl_5").alias("control_pnl_5"),
    ).join(
        candidate.select(
            *POINT_KEY_COLUMNS,
            pl.col("policy_selected").alias("candidate_selected"),
            pl.col("correct").alias("candidate_correct"),
            pl.col("predicted_up").alias("candidate_predicted_up"),
            pl.col("realized_net_pnl_5").alias("candidate_pnl_5"),
        ),
        on=list(POINT_KEY_COLUMNS),
        how="inner",
        validate="1:1",
    )
    losses = joined.filter(~pl.col("control_correct"))
    wins = joined.filter(pl.col("control_correct"))
    rejected_losses = losses.filter(~pl.col("candidate_selected"))
    rejected_wins = wins.filter(~pl.col("candidate_selected"))
    retained_wins = wins.filter(pl.col("candidate_selected") & pl.col("candidate_correct"))
    avoided_losses = losses.filter(
        ~pl.col("candidate_selected")
        | (pl.col("candidate_predicted_up") != pl.col("control_predicted_up"))
    )
    saved_loss = float((-rejected_losses["control_pnl_5"]).sum()) if rejected_losses.height else 0.0
    forgone_win = float(rejected_wins["control_pnl_5"].sum()) if rejected_wins.height else 0.0
    return {
        "champion_losses": losses.height,
        "champion_wins": wins.height,
        "rejected_champion_losses": rejected_losses.height,
        "rejected_champion_wins": rejected_wins.height,
        "loss_rejection_rate": rejected_losses.height / losses.height if losses.height else 0.0,
        "loss_avoidance_rate_including_direction_change": (
            avoided_losses.height / losses.height if losses.height else 0.0
        ),
        "win_retention_rate": retained_wins.height / wins.height if wins.height else 0.0,
        "saved_loss_dollars_from_rejection": saved_loss,
        "forgone_win_dollars_from_rejection": forgone_win,
        "saved_loss_to_forgone_win_ratio": (
            saved_loss / forgone_win if forgone_win > 0 else (math.inf if saved_loss > 0 else 0.0)
        ),
    }


def _promotion_decision(
    matched_confirmation: dict[str, Any],
    native_confirmation: dict[str, dict[str, Any]],
    native_pairing: dict[str, dict[str, Any]],
    stress: dict[str, dict[str, Any]],
    matched_folds: dict[str, Any],
    native_folds: dict[str, dict[str, Any]],
    config: ChainlinkOiBenchmarkConfig,
) -> dict[str, Any]:
    control = native_confirmation[CONTROL]
    decisions: dict[str, Any] = {}
    for name in CANDIDATES:
        candidate = native_confirmation[name]
        pairing = native_pairing[name]
        exact_rejection = matched_confirmation["loss_rejection"][name]
        exact_win_rejection_rate = (
            exact_rejection["rejected_champion_wins"] / exact_rejection["champion_wins"]
            if exact_rejection["champion_wins"]
            else 0.0
        )
        gross_loss_reduction = _relative_reduction(
            control["five_share"]["gross_loss"],
            candidate["five_share"]["gross_loss"],
        )
        checks = [
            _check(
                "economic confirmation spans enough UTC days",
                matched_confirmation["confirmation_days"],
                config.promotion.minimum_confirmation_days,
                ">=",
            ),
            _check(
                "economic confirmation has enough selected trades",
                candidate["trades"],
                config.promotion.minimum_economic_confirmation_trades,
                ">=",
            ),
            _check(
                "champion proposal coverage retained",
                candidate["coverage"] / control["coverage"] if control["coverage"] else 0.0,
                config.promotion.minimum_champion_coverage_ratio,
                ">=",
            ),
            _check(
                "accuracy remains within champion tolerance",
                candidate["accuracy"] if candidate["accuracy"] is not None else -1.0,
                (control["accuracy"] or 0.0) - config.promotion.maximum_accuracy_regression,
                ">=",
            ),
            _check(
                "champion losses are rejected",
                pairing["champion_loss_no_trade_rate"],
                config.promotion.minimum_loss_capture_rate,
                ">=",
            ),
            _check(
                "champion wins are retained",
                pairing["champion_win_retention_rate"],
                config.promotion.minimum_win_retention_rate,
                ">=",
            ),
            _check(
                "gross loss is reduced",
                gross_loss_reduction,
                config.promotion.minimum_gross_loss_reduction,
                ">=",
            ),
            _check(
                "saved loss outweighs forgone wins",
                pairing["saved_loss_to_forgone_win_ratio"],
                config.promotion.minimum_saved_loss_to_forgone_win_ratio,
                ">=",
            ),
            _check(
                "exact champion-key rejection is loss-selective",
                exact_rejection["loss_rejection_rate"],
                exact_win_rejection_rate,
                ">",
            ),
            _check(
                "net PnL per eligible champion proposal improves",
                candidate["five_share"]["net_pnl_per_eligible_market"],
                control["five_share"]["net_pnl_per_eligible_market"]
                + config.promotion.minimum_net_expectancy_delta,
                ">",
            ),
            _check(
                "profit factor improves",
                _comparable_profit_factor(candidate["five_share"]),
                _comparable_profit_factor(control["five_share"])
                + config.promotion.minimum_profit_factor_delta,
                ">",
            ),
            _check(
                "selected probability calibration is acceptable",
                candidate["expected_calibration_error"]
                if candidate["expected_calibration_error"] is not None
                else math.inf,
                config.promotion.maximum_expected_calibration_error,
                "<=",
            ),
            _check(
                "directional stress cohort is large enough",
                stress[name]["eligible_markets"],
                config.promotion.minimum_directional_stress_markets,
                ">=",
            ),
            _check(
                "directional stress accuracy remains within tolerance",
                stress[name]["accuracy"],
                stress[CONTROL]["accuracy"] - config.promotion.maximum_accuracy_regression,
                ">=",
            ),
            _check(
                "directional stress coverage is retained",
                stress[name]["coverage"],
                stress[CONTROL]["coverage"] * config.promotion.minimum_champion_coverage_ratio,
                ">=",
            ),
            _check(
                "net PnL per eligible market improves in each confirmation fold",
                sum(
                    fold[name]["five_share"]["net_pnl_per_eligible_market"]
                    > fold[CONTROL]["five_share"]["net_pnl_per_eligible_market"]
                    for fold in native_folds.values()
                ),
                len(native_folds),
                ">=",
            ),
        ]
        if config.promotion.require_nonworse_worst_trade:
            checks.append(
                _check(
                    "worst trade does not worsen",
                    candidate["five_share"]["worst_trade"],
                    control["five_share"]["worst_trade"],
                    ">=",
                )
            )
        if config.promotion.require_nonworse_worst_one_percent:
            checks.append(
                _check(
                    "worst one-percent mean does not worsen",
                    candidate["five_share"]["worst_one_percent_mean"],
                    control["five_share"]["worst_one_percent_mean"],
                    ">=",
                )
            )
        fold_summary = {
            fold_name: {
                "candidate_net_pnl_per_eligible_market": fold["models"][name]["five_share"][
                    "net_pnl_per_eligible_market"
                ],
                "control_net_pnl_per_eligible_market": fold["models"][CONTROL]["five_share"][
                    "net_pnl_per_eligible_market"
                ],
            }
            for fold_name, fold in matched_folds.items()
        }
        decisions[name] = {
            "passed": all(check["passed"] for check in checks),
            "gross_loss_reduction": gross_loss_reduction,
            "native_champion_market_pairing": pairing,
            "exact_champion_key_diagnostic": {
                **exact_rejection,
                "win_rejection_rate": exact_win_rejection_rate,
            },
            "checks": checks,
            "confirmation_folds": fold_summary,
            "native_confirmation_folds": {
                fold_name: {
                    "candidate_net_pnl_per_eligible_market": fold[name]["five_share"][
                        "net_pnl_per_eligible_market"
                    ],
                    "control_net_pnl_per_eligible_market": fold[CONTROL]["five_share"][
                        "net_pnl_per_eligible_market"
                    ],
                }
                for fold_name, fold in native_folds.items()
            },
        }
    qualified = [name for name in CANDIDATES if decisions[name]["passed"]]
    selected = (
        max(
            qualified,
            key=lambda name: native_confirmation[name]["five_share"]["net_pnl_per_eligible_market"],
        )
        if qualified
        else None
    )
    return {
        "selected_candidate": selected,
        "champion_retained": selected is None,
        "paper_only": True,
        "runtime_changed": False,
        "deployment_changed": False,
        "fallback_winner_allowed": False,
        "candidates": decisions,
    }


def _check(name: str, observed: float, required: float, operator: str) -> dict[str, Any]:
    if operator == ">=":
        passed = observed >= required
    elif operator == ">":
        passed = observed > required
    elif operator == "<=":
        passed = observed <= required
    else:
        raise ValueError(f"unsupported comparison operator: {operator}")
    return {
        "name": name,
        "observed": _json_number(observed),
        "required": _json_number(required),
        "operator": operator,
        "passed": bool(passed),
    }


def _comparable_profit_factor(metrics: dict[str, Any]) -> float:
    value = metrics["profit_factor"]
    return float(value) if value is not None else math.inf


def _relative_reduction(control: float, candidate: float) -> float:
    if control <= 0:
        return 0.0
    return (control - candidate) / control


def _validate_common_short_cohort(left: pl.DataFrame, right: pl.DataFrame) -> None:
    if left.is_empty() or right.is_empty():
        raise RuntimeError("strict Chainlink/OI common cohort is empty")
    if not left.select(POINT_KEY_COLUMNS).equals(
        right.select(POINT_KEY_COLUMNS),
        null_equal=True,
    ):
        raise RuntimeError("strict Chainlink and Chainlink-plus-OI rows diverged")


def _validate_model_feature_values(
    frame: pl.DataFrame,
    feature_names: tuple[str, ...],
    role: str,
) -> None:
    valid = frame.select(
        pl.all_horizontal(
            pl.col(name).is_not_null() & pl.col(name).is_finite() for name in feature_names
        ).all()
    ).item()
    if not valid:
        raise RuntimeError(
            f"{role} contains missing/non-finite model features; imputation is disabled"
        )


def _first_crossings_from_scored(frame: pl.DataFrame, threshold: float) -> pl.DataFrame:
    return (
        frame.filter(pl.col("confidence") >= threshold)
        .sort(["observed_at", "market_id"])
        .group_by("market_id", maintain_order=True)
        .first()
    )


def _validate_evaluation_key_identity(frames: dict[str, pl.DataFrame]) -> None:
    anchor = frames[CHAINLINK_FULL_CANDIDATE].select(POINT_KEY_COLUMNS)
    for name, frame in frames.items():
        if not anchor.equals(frame.select(POINT_KEY_COLUMNS), null_equal=True):
            raise RuntimeError(f"evaluation keys diverged for {name}")


def _match_keys(frame: pl.DataFrame, keys: pl.DataFrame, role: str) -> pl.DataFrame:
    matched = keys.join(frame, on=list(POINT_KEY_COLUMNS), how="inner", validate="1:1")
    if matched.height != keys.height:
        raise RuntimeError(f"{role} cannot cover the common evaluation keys")
    return matched.sort(["window_start", "market_id", "seconds_elapsed"])


def _assert_disjoint_markets(left: pl.DataFrame, right: pl.DataFrame, role: str) -> None:
    overlap = left.select("market_id").join(
        right.select("market_id"),
        on="market_id",
        how="inner",
    )
    if overlap.height:
        raise RuntimeError(f"{role} fit and calibration markets overlap")


def _range(frame: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    return frame.filter((pl.col("window_start") >= start) & (pl.col("window_start") < end))


def _cohort(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {"rows": 0, "markets": 0, "days": 0, "start": None, "end": None}
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "days": frame.select(pl.col("window_start").dt.date().n_unique()).item(),
        "start": frame["window_start"].min().isoformat(),
        "end": frame["window_start"].max().isoformat(),
    }


def _window_payload(start: datetime, end: datetime) -> dict[str, str]:
    return {"start": start.isoformat(), "end_exclusive": end.isoformat()}


def _utc(value: str | None) -> datetime:
    if value is None:
        raise RuntimeError("required timestamp is missing")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise RuntimeError("timestamp is not timezone aware")
    return parsed.astimezone(UTC)


def _json_number(value: float) -> float | str:
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _empty_economic_metrics(eligible: int) -> dict[str, Any]:
    return {
        "eligible_markets": eligible,
        "trades": 0,
        "coverage": 0.0,
        "wins": 0,
        "losses": 0,
        "accuracy": None,
        "expected_calibration_error": None,
        "five_share": _execution_metrics(pl.DataFrame({"pnl": []}), "pnl"),
    }


def _empty_loss_rejection() -> dict[str, Any]:
    return {
        "champion_losses": 0,
        "champion_wins": 0,
        "rejected_champion_losses": 0,
        "rejected_champion_wins": 0,
        "loss_rejection_rate": 0.0,
        "win_retention_rate": 0.0,
        "saved_loss_dollars_from_rejection": 0.0,
        "forgone_win_dollars_from_rejection": 0.0,
        "saved_loss_to_forgone_win_ratio": 0.0,
    }


def _report_html(result: dict[str, Any]) -> str:
    confirmation = result["native_first_crossing_economics"]["economic_confirmation"]
    stress = result["directional"]["directional_stress"]
    rows = []
    for name in ALL_MODELS:
        economic = confirmation[name]
        direction = stress[name]
        pairing = (
            result["native_champion_market_pairing"]["economic_confirmation"].get(name, {})
            if name != CONTROL
            else {}
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{economic['trades']:,}</td>"
            f"<td>{_percent(economic['coverage'])}</td>"
            f"<td>{_percent(economic['accuracy'])}</td>"
            f"<td>{economic['five_share']['net_pnl']:.2f}</td>"
            f"<td>{_number(economic['five_share']['profit_factor'])}</td>"
            f"<td>{_number(economic['five_share']['worst_trade'])}</td>"
            f"<td>{_percent(pairing.get('champion_loss_no_trade_rate'))}</td>"
            f"<td>{direction['markets']:,}</td>"
            f"<td>{_percent(direction['accuracy'])}</td>"
            "</tr>"
        )
    check_sections = []
    for name, candidate in result["promotion"]["candidates"].items():
        checks = "".join(
            "<li class='{}'>{}: observed {}, required {} {}</li>".format(
                "pass" if check["passed"] else "fail",
                html.escape(check["name"]),
                html.escape(str(check["observed"])),
                html.escape(check["operator"]),
                html.escape(str(check["required"])),
            )
            for check in candidate["checks"]
        )
        check_sections.append(f"<h3>{html.escape(name)}</h3><ul>{checks}</ul>")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>BTC Chainlink and OI champion benchmark</title>
<style>
body{{font:15px system-ui;margin:32px;max-width:1400px;color:#182230}}table{{border-collapse:collapse;width:100%}}
th,td{{padding:8px;border-bottom:1px solid #d8dee8;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.card{{padding:18px;border:1px solid #d8dee8;border-radius:10px;margin:16px 0}}.pass{{color:#08783e}}.fail{{color:#b42318}}
code{{background:#f4f6f8;padding:2px 5px;border-radius:4px}}
</style></head><body>
<h1>BTC Chainlink + open-interest benchmark</h1>
<div class="card"><b>Outcome:</b> {html.escape(result["status"])}; selected candidate:
<code>{html.escape(result["selected_candidate"] or "none")}</code>. Runtime and deployment were unchanged.</div>
<p>{html.escape(result["evaluation_note"])}</p>
<p><b>Evidence:</b> {html.escape(result["evidence_classification"])}.</p>
<h2>Native first-crossing economics and directional stress</h2>
<table><thead><tr><th>Model</th><th>Trades</th><th>Coverage</th><th>Accuracy</th><th>5-share PnL</th>
<th>Profit factor</th><th>Worst trade</th><th>Champion-loss NoTrade</th><th>Stress trades</th><th>Stress accuracy</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<h2>Paper-promotion checks</h2>{"".join(check_sections)}
<h2>Interpretation contract</h2>
<ul><li>RefPrice and OI are joined strictly before each decision; candles must be closed.</li>
<li>No unavailable value is imputed and missingness is not a feature.</li>
<li>PMXT is used only for executable VWAP5/VWAP10, fees, and economic diagnostics.</li>
<li>Native first-crossing behavior controls paper promotion because each challenger is a replacement decision model, not a champion gate.</li>
<li>Exact champion-proposal keys remain a separate loss-signature diagnostic and must reject losses more selectively than wins.</li></ul>
</body></html>"""


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"
