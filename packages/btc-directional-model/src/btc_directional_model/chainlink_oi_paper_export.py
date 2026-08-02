from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from .chainlink_oi_benchmark import (
    CANDIDATES,
    EXTERNAL_CACHE_SCHEMA_VERSION,
    ChainlinkOiCandidateBundle,
    _candidate_feature_sets,
    _load_core_features,
    _load_or_extract_external_sources,
    _range,
)
from .chainlink_oi_benchmark import SCHEMA_VERSION as BENCHMARK_SCHEMA_VERSION
from .chainlink_oi_config import (
    CHAINLINK_FULL_CANDIDATE,
    CHAINLINK_FULL_OI_CANDIDATE,
    LONG_HISTORY_CANDLE_CANDIDATE,
    ChainlinkOiBenchmarkConfig,
)
from .chainlink_oi_features import (
    derive_chainlink_candle_feature_frame,
    derive_chainlink_oi_feature_frames,
)
from .core_config import load_core_config
from .core_evaluation import FIRST_CROSSING_TIME_BANDS
from .core_extract import file_sha256, write_json_atomic
from .core_training import (
    CORE_FREEZE_SCHEMA_VERSION,
    TRAINING_MODEL_FILENAME,
    FrozenCalibrationBand,
    FrozenTimeBandedTrainingBundle,
    model_candidate_spec,
    row_weight_schedule_payload,
)
from .paper_candidate import write_time_banded_golden_feature_sample
from .provenance import runtime_provenance
from .runtime_export import (
    export_runtime_model,
    frozen_calibration_bands_payload,
    verify_frozen_bundle,
)

PAPER_EXPORT_SCHEMA_VERSION = "btc-chainlink-oi-paper-candidate-v1"
PAPER_ONLY_AUTHORIZATION = "explicit-chainlink-oi-paper-only-forward-evaluation"
GOLDEN_FEATURES_FILENAME = "golden-features.parquet"


@dataclass(frozen=True)
class ChainlinkOiPaperExportSpec:
    candidate: str
    feature_schema_version: str
    model_key: str


@dataclass(frozen=True)
class ChainlinkOiPaperExportResult:
    candidate: str
    freeze_dir: Path
    runtime_dir: Path
    model_key: str
    model_sha256: str
    manifest_sha256: str
    feature_schema_version: str
    feature_schema_sha256: str


PAPER_EXPORT_SPECS = (
    ChainlinkOiPaperExportSpec(
        candidate=CHAINLINK_FULL_CANDIDATE,
        feature_schema_version=(
            "btc-5m-directional-boundary-oracle-chainlink-refprice-candle-features-v1"
        ),
        model_key="btc-5m-directional-chainlink-full-20260703-20260713-paper-v1",
    ),
    ChainlinkOiPaperExportSpec(
        candidate=CHAINLINK_FULL_OI_CANDIDATE,
        feature_schema_version=(
            "btc-5m-directional-boundary-oracle-chainlink-refprice-candle-oi-features-v1"
        ),
        model_key="btc-5m-directional-chainlink-full-oi-20260703-20260713-paper-v1",
    ),
    ChainlinkOiPaperExportSpec(
        candidate=LONG_HISTORY_CANDLE_CANDIDATE,
        feature_schema_version=(
            "btc-5m-directional-boundary-oracle-chainlink-candle-features-v1"
        ),
        model_key=(
            "btc-5m-directional-long-history-candle-20260321-20260713-paper-v1"
        ),
    ),
)


def export_chainlink_oi_paper_candidates(
    *,
    config: ChainlinkOiBenchmarkConfig,
    benchmark_run: Path,
    freeze_root: Path,
    runtime_output_root: Path,
    authorization: str,
) -> tuple[ChainlinkOiPaperExportResult, ...]:
    """Freeze and export the three already-fitted challengers for paper use only."""

    if authorization != PAPER_ONLY_AUTHORIZATION:
        raise RuntimeError(
            "Chainlink/OI paper export requires explicit paper-only authorization"
        )
    benchmark_run = benchmark_run.resolve()
    benchmark, benchmark_path = _load_benchmark_evidence(config, benchmark_run)
    source_bundles = _load_source_bundles(config, benchmark_run, benchmark)
    feature_frames = _calibration_feature_frames(config, benchmark)
    provenance = runtime_provenance(config.package_root)

    results: list[ChainlinkOiPaperExportResult] = []
    for spec in PAPER_EXPORT_SPECS:
        source_bundle = source_bundles[spec.candidate]
        runtime_bundle = time_banded_runtime_bundle(source_bundle)
        feature_frame = feature_frames[spec.candidate]
        _validate_feature_frame(
            spec,
            source_bundle,
            feature_frame,
            benchmark,
        )
        freeze_dir = _freeze_candidate(
            config=config,
            benchmark=benchmark,
            benchmark_path=benchmark_path,
            benchmark_run=benchmark_run,
            spec=spec,
            source_bundle=source_bundle,
            runtime_bundle=runtime_bundle,
            feature_frame=feature_frame,
            freeze_root=freeze_root,
            provenance=provenance,
        )
        runtime_dir = export_runtime_model(
            freeze_dir=freeze_dir,
            golden_features=freeze_dir / GOLDEN_FEATURES_FILENAME,
            output_root=runtime_output_root,
            model_key=spec.model_key,
        )
        runtime_manifest_path = runtime_dir / "manifest.json"
        runtime_manifest = _read_json_object(runtime_manifest_path)
        results.append(
            ChainlinkOiPaperExportResult(
                candidate=spec.candidate,
                freeze_dir=freeze_dir,
                runtime_dir=runtime_dir,
                model_key=spec.model_key,
                model_sha256=str(runtime_manifest["model_sha256"]),
                manifest_sha256=file_sha256(runtime_manifest_path),
                feature_schema_version=spec.feature_schema_version,
                feature_schema_sha256=str(
                    runtime_manifest["feature_schema_sha256"]
                ),
            )
        )
    return tuple(results)


def time_banded_runtime_bundle(
    source: ChainlinkOiCandidateBundle,
) -> FrozenTimeBandedTrainingBundle:
    if source.name not in CANDIDATES:
        raise ValueError(f"unsupported Chainlink/OI candidate: {source.name}")
    if source.model.candidate_name != source.name:
        raise RuntimeError("candidate bundle and fitted model names do not match")
    if tuple(source.feature_names) != tuple(source.model.feature_names):
        raise RuntimeError("candidate bundle and fitted model feature orders do not match")
    if not math.isclose(source.confidence_threshold, 0.89, abs_tol=1e-12):
        raise RuntimeError("candidate confidence threshold is not the benchmark threshold")

    expected_names = tuple(name for name, _, _ in FIRST_CROSSING_TIME_BANDS)
    if tuple(source.calibrators) != expected_names:
        raise RuntimeError("candidate calibration bands are missing or reordered")
    bands: list[FrozenCalibrationBand] = []
    for name, start, end in FIRST_CROSSING_TIME_BANDS:
        calibrator = source.calibrators[name]
        if (
            not calibrator.converged
            or not math.isfinite(calibrator.slope)
            or calibrator.slope <= 0.0
            or not math.isfinite(calibrator.intercept)
        ):
            raise RuntimeError(f"candidate calibration band is invalid: {name}")
        bands.append(
            FrozenCalibrationBand(
                name=name,
                start_second=start,
                end_second_exclusive=end,
                calibrator=calibrator,
                confidence_threshold=source.confidence_threshold,
            )
        )
    return FrozenTimeBandedTrainingBundle(
        model=source.model,
        target_kind="outcome_up",
        bands=tuple(bands),
    )


def _load_benchmark_evidence(
    config: ChainlinkOiBenchmarkConfig,
    benchmark_run: Path,
) -> tuple[dict[str, Any], Path]:
    benchmark_path = benchmark_run / "benchmark.json"
    if not benchmark_path.is_file():
        raise RuntimeError("Chainlink/OI benchmark evidence is missing")
    benchmark = _read_json_object(benchmark_path)
    if (
        benchmark.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or benchmark.get("run_id") != benchmark_run.name
        or benchmark.get("runtime_changed") is not False
        or benchmark.get("deployment_changed") is not False
    ):
        raise RuntimeError("Chainlink/OI benchmark evidence contract is invalid")
    lineage = benchmark.get("lineage", {})
    if (
        lineage.get("configuration_sha256") != file_sha256(config.source_path)
        or lineage.get("source_schema_revision")
        != config.source_schema_revision
        or lineage.get("core", {}).get("sha256")
        != file_sha256(load_core_config(config.paths.core_config).paths.development_feature_data)
    ):
        raise RuntimeError("Chainlink/OI benchmark lineage does not match the configuration")
    if set(benchmark.get("model_artifacts", {})) != set(CANDIDATES):
        raise RuntimeError("benchmark does not contain exactly the three candidate artifacts")
    if set(benchmark.get("features", {})) != set(CANDIDATES):
        raise RuntimeError("benchmark does not contain exactly the three feature contracts")
    return benchmark, benchmark_path


def _load_source_bundles(
    config: ChainlinkOiBenchmarkConfig,
    benchmark_run: Path,
    benchmark: dict[str, Any],
) -> dict[str, ChainlinkOiCandidateBundle]:
    expected_features = _candidate_feature_sets()
    bundles: dict[str, ChainlinkOiCandidateBundle] = {}
    for candidate in CANDIDATES:
        record = benchmark["model_artifacts"][candidate]
        path = benchmark_run / f"{candidate}-training-model.joblib"
        recorded_path = Path(str(record.get("path", "")))
        if recorded_path.name != path.name or not path.is_file():
            raise RuntimeError(f"candidate training artifact is missing: {candidate}")
        if record.get("sha256") != file_sha256(path):
            raise RuntimeError(f"candidate training artifact hash mismatch: {candidate}")
        bundle = joblib.load(path)
        if type(bundle) is not ChainlinkOiCandidateBundle:
            raise TypeError(f"unsupported candidate training bundle: {candidate}")
        if (
            bundle.name != candidate
            or tuple(bundle.feature_names) != expected_features[candidate]
            or tuple(bundle.model.feature_names) != expected_features[candidate]
            or benchmark["features"][candidate] != list(expected_features[candidate])
        ):
            raise RuntimeError(f"candidate feature contract mismatch: {candidate}")
        if bundle.model.family != "histogram":
            raise RuntimeError(f"candidate is not a histogram model: {candidate}")
        expected_parameters = {
            "learning_rate": config.model.learning_rate,
            "max_iter": config.model.max_iter,
            "max_leaf_nodes": config.model.max_leaf_nodes,
            "min_samples_leaf": config.model.min_samples_leaf,
            "l2_regularization": config.model.l2_regularization,
        }
        if bundle.model.hyperparameters != expected_parameters:
            raise RuntimeError(f"candidate hyperparameter contract mismatch: {candidate}")
        bundles[candidate] = bundle
    return bundles


def _calibration_feature_frames(
    config: ChainlinkOiBenchmarkConfig,
    benchmark: dict[str, Any],
) -> dict[str, pl.DataFrame]:
    core_config = load_core_config(config.paths.core_config)
    core, _ = _load_core_features(config, core_config)
    external_manifest = config.paths.shared_cache / "external" / "manifest.json"
    if not external_manifest.is_file():
        raise RuntimeError("immutable external-source cache is required for paper export")
    external_manifest_payload = _read_json_object(external_manifest)
    if (
        external_manifest_payload.get("schema_version")
        != EXTERNAL_CACHE_SCHEMA_VERSION
        or file_sha256(external_manifest)
        != benchmark.get("lineage", {}).get("external", {}).get("manifest_sha256")
    ):
        raise RuntimeError("external-source cache does not match benchmark evidence")
    external, _ = _load_or_extract_external_sources(config)
    calibration_core = _range(
        core,
        config.windows.calibration_start,
        config.windows.calibration_end,
    )
    chainlink, chainlink_oi = derive_chainlink_oi_feature_frames(
        calibration_core,
        external.refprice,
        external.candles,
        external.open_interest,
        refprice_max_age_seconds=config.staleness.refprice_seconds,
        candle_max_age_seconds=config.staleness.candles_seconds,
        open_interest_max_age_seconds=config.staleness.open_interest_seconds,
    )
    candle = derive_chainlink_candle_feature_frame(
        calibration_core,
        external.candles,
        candle_max_age_seconds=config.staleness.candles_seconds,
    )
    return {
        CHAINLINK_FULL_CANDIDATE: chainlink,
        CHAINLINK_FULL_OI_CANDIDATE: chainlink_oi,
        LONG_HISTORY_CANDLE_CANDIDATE: candle,
    }


def _validate_feature_frame(
    spec: ChainlinkOiPaperExportSpec,
    source_bundle: ChainlinkOiCandidateBundle,
    frame: pl.DataFrame,
    benchmark: dict[str, Any],
) -> None:
    required = {
        "market_id",
        "observed_at",
        "seconds_elapsed",
        *source_bundle.feature_names,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{spec.candidate} golden feature frame is missing columns: "
            + ", ".join(missing)
        )
    training = benchmark["training"][spec.candidate]["calibration"]
    if (
        frame.height != int(training["rows"])
        or frame["market_id"].n_unique() != int(training["markets"])
        or frame.select(pl.col("window_start").dt.date().n_unique()).item()
        != int(training["days"])
    ):
        raise RuntimeError(
            f"{spec.candidate} calibration feature cohort changed after training"
        )
    matrix = frame.select(source_bundle.feature_names).cast(pl.Float64).to_numpy()
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{spec.candidate} golden features are not finite")


def _freeze_candidate(
    *,
    config: ChainlinkOiBenchmarkConfig,
    benchmark: dict[str, Any],
    benchmark_path: Path,
    benchmark_run: Path,
    spec: ChainlinkOiPaperExportSpec,
    source_bundle: ChainlinkOiCandidateBundle,
    runtime_bundle: FrozenTimeBandedTrainingBundle,
    feature_frame: pl.DataFrame,
    freeze_root: Path,
    provenance: dict[str, Any],
) -> Path:
    source_record = benchmark["model_artifacts"][spec.candidate]
    source_model_sha256 = str(source_record["sha256"])
    freeze_id = (
        f"{benchmark['run_id']}-{spec.candidate}-"
        f"{source_model_sha256[:12]}-paper"
    )
    destination = freeze_root.resolve() / freeze_id
    if destination.exists():
        _validate_existing_freeze(
            destination,
            spec,
            source_model_sha256,
        )
        return destination

    freeze_root = freeze_root.resolve()
    freeze_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{freeze_id}.partial-", dir=freeze_root)
    )
    try:
        model_path = staging / TRAINING_MODEL_FILENAME
        joblib.dump(runtime_bundle, model_path, compress=3)
        calibration_bands = frozen_calibration_bands_payload(runtime_bundle.bands)
        model_spec = model_candidate_spec(runtime_bundle.model)
        summary = {
            "schema_version": "btc-core-training-model-summary-v1",
            "paper_candidate_schema_version": PAPER_EXPORT_SCHEMA_VERSION,
            "training_only": True,
            "deployment_scope": "paper_only",
            "production_qualified": False,
            "live_capital_allowed": False,
            "candidate": runtime_bundle.model.candidate_name,
            "family": runtime_bundle.model.family,
            "row_weight_policy": model_spec.row_weight_policy,
            "row_weight_schedule": row_weight_schedule_payload(model_spec),
            "feature_names": list(runtime_bundle.model.feature_names),
            "hyperparameters": runtime_bundle.model.hyperparameters,
            "imputation_medians": runtime_bundle.model.imputation_medians.tolist(),
            "calibration_kind": "time_banded_platt",
            "calibration_bands": calibration_bands,
            "target_kind": runtime_bundle.target_kind,
        }
        summary_path = staging / "model-summary.json"
        write_json_atomic(summary_path, summary)

        golden_path = staging / GOLDEN_FEATURES_FILENAME
        probability_up = runtime_bundle.probability_up(feature_frame)
        if not np.isfinite(probability_up).all():
            raise RuntimeError(f"{spec.candidate} produced non-finite golden probabilities")
        write_time_banded_golden_feature_sample(
            feature_frame,
            probability_up,
            runtime_bundle,
            spec.feature_schema_version,
            golden_path,
        )

        core_config = load_core_config(config.paths.core_config)
        core_feature_path = core_config.paths.development_feature_data
        core_metadata_path = core_feature_path.with_suffix(".metadata.json")
        calibration_training = benchmark["training"][spec.candidate]
        fit_start = (
            config.windows.source_range_start
            if spec.candidate == LONG_HISTORY_CANDLE_CANDIDATE
            else config.windows.short_fit_start
        )
        created_at = _parse_datetime(str(benchmark["created_at"]))
        manifest: dict[str, Any] = {
            "schema_version": CORE_FREEZE_SCHEMA_VERSION,
            "paper_candidate_schema_version": PAPER_EXPORT_SCHEMA_VERSION,
            "freeze_id": freeze_id,
            "created_at": created_at.isoformat(),
            "status": "paper_candidate_frozen",
            "deployment_status": "paper_only_forward_evaluation",
            "deployment_scope": "paper_only",
            "production_qualified": False,
            "live_capital_allowed": False,
            "paper_only_authorization": PAPER_ONLY_AUTHORIZATION,
            "model_file": TRAINING_MODEL_FILENAME,
            "model_sha256": file_sha256(model_path),
            "model_summary_sha256": file_sha256(summary_path),
            "source_training_model_file": (
                f"{spec.candidate}-training-model.joblib"
            ),
            "source_training_model_sha256": source_model_sha256,
            "candidate": runtime_bundle.model.candidate_name,
            "family": runtime_bundle.model.family,
            "target_kind": runtime_bundle.target_kind,
            "row_weight_policy": model_spec.row_weight_policy,
            "row_weight_schedule": row_weight_schedule_payload(model_spec),
            "feature_schema_version": spec.feature_schema_version,
            "feature_names": list(runtime_bundle.model.feature_names),
            "hyperparameters": runtime_bundle.model.hyperparameters,
            "calibration_kind": "time_banded_platt",
            "calibration_bands": calibration_bands,
            "prediction_policy": {
                "type": "first_confidence_crossing",
                "minimum_seconds_after_open": config.model.minimum_seconds_after_open,
                "maximum_seconds_after_open": config.model.maximum_seconds_after_open,
                "cadence_seconds": config.model.cadence_seconds,
            },
            "configuration_sha256": file_sha256(config.source_path),
            "core_configuration_sha256": file_sha256(core_config.source_path),
            "development_feature_sha256": file_sha256(core_feature_path),
            "development_feature_metadata_sha256": file_sha256(
                core_metadata_path
            ),
            "external_source_manifest_sha256": benchmark["lineage"]["external"][
                "manifest_sha256"
            ],
            "golden_feature_file": golden_path.name,
            "golden_feature_sha256": file_sha256(golden_path),
            "golden_feature_metadata_sha256": file_sha256(
                golden_path.with_suffix(".metadata.json")
            ),
            "source_tree_sha256": provenance["source_tree_sha256"],
            "git": provenance["git"],
            "runtime_provenance": provenance,
            "random_seed": config.model.random_seed,
            "training_ranges": {
                "fit": {
                    "start": fit_start.isoformat(),
                    "end": config.windows.short_fit_end.isoformat(),
                },
                "probability_calibration": {
                    "start": config.windows.calibration_start.isoformat(),
                    "end": config.windows.calibration_end.isoformat(),
                },
            },
            "holdout_range": {
                "start": config.windows.directional_stress_start.isoformat(),
                "end": config.windows.directional_stress_end.isoformat(),
            },
            "forward_paper_evaluation": {
                "start": created_at.isoformat(),
                "end": None,
                "status": "pending",
            },
            "benchmark_evidence": {
                "run": str(benchmark_run),
                "benchmark_sha256": file_sha256(benchmark_path),
                "evidence_classification": benchmark["evidence_classification"],
                "paper_promotion_qualified": bool(
                    source_record["paper_promotion_qualified"]
                ),
                "fit": calibration_training["fit"],
                "calibration": calibration_training["calibration"],
            },
            "production_blocking_reasons": [
                "artifact is explicitly authorized only for forward paper evaluation",
                "benchmark evidence is consumed chronological development evidence",
                "no independent post-freeze forward cohort has been evaluated",
            ],
        }
        manifest_path = staging / "freeze-manifest.json"
        write_json_atomic(manifest_path, manifest)
        (staging / "freeze-manifest.sha256").write_text(
            file_sha256(manifest_path) + "\n",
            encoding="utf-8",
        )
        try:
            staging.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            _validate_existing_freeze(
                destination,
                spec,
                source_model_sha256,
            )
        verify_frozen_bundle(destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _validate_existing_freeze(
    freeze_dir: Path,
    spec: ChainlinkOiPaperExportSpec,
    source_model_sha256: str,
) -> None:
    _, manifest, _ = verify_frozen_bundle(freeze_dir)
    if (
        manifest.get("paper_candidate_schema_version")
        != PAPER_EXPORT_SCHEMA_VERSION
        or manifest.get("candidate") != spec.candidate
        or manifest.get("feature_schema_version")
        != spec.feature_schema_version
        or manifest.get("source_training_model_sha256")
        != source_model_sha256
        or manifest.get("deployment_scope") != "paper_only"
        or manifest.get("production_qualified") is not False
        or manifest.get("live_capital_allowed") is not False
    ):
        raise RuntimeError("existing Chainlink/OI paper freeze is not identical")


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise RuntimeError("benchmark created_at must be timezone-aware")
    return parsed


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload
