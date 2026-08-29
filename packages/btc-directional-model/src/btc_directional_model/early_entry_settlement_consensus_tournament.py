"""Frozen early-entry settlement-consensus training tournament.

This module is intentionally training-only.  It consumes existing immutable
source caches and frozen constituent specifications, regenerates chronological
OOF predictions on the exact 60..180 second grid, and never writes to a
database, runtime-model directory, or trading-process configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import beta, norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "btc-early-entry-settlement-consensus-tournament-v2"
ARTIFACT_SCHEMA_VERSION = "btc-early-entry-settlement-consensus-model-v2"
SOURCE_PANEL_SCHEMA_VERSION = "btc-early-entry-settlement-consensus-panel-v1"
CHECKPOINT_SCHEMA_VERSION = "btc-early-entry-settlement-consensus-checkpoint-v1"
WEIGHT_NORMALIZATION = "mean_one_market_band_equal_v1"

ENTRY_SECONDS = tuple(range(60, 181, 5))
CANDIDATE_NAMES = (
    "frozen_bridge_control",
    "bridge_latent_equal_logit_pool",
    "three_family_nonnegative_logit_stack",
    "bridge_latent_uncertainty_margin_stack",
)
HISTORY_ARMS = (
    "authentic_only",
    "chainlink_reconstructed",
    "binance_synthetic_extension",
    "uncertainty_weighted_hybrid",
)

KEY_COLUMNS = ("market_id", "window_start", "observed_at", "seconds_elapsed")
SUPERVISION_COLUMNS = (
    "label_up",
    "target_margin_bps",
    "base_label_weight",
    "label_source",
    "authentic_label_up",
    "authentic_margin_bps",
    "proxy_label_up",
    "proxy_margin_bps",
    "binance_raw_margin_bps",
    "binance_corrected_margin_bps",
    "binance_margin_correction_bps",
    "estimated_synthetic_label_error",
    "raw_corrected_synthetic_margin_disagreement_bps",
    "official_outcome",
)

BRIDGE_FEATURES = (
    "seconds_elapsed_scaled",
    "seconds_remaining_scaled",
    "btc_path_from_window_open_bps",
    "btc_cross_venue_boundary_gap_bps",
    "btc_window_open_cross_venue_basis_bps",
    "btc_return_1s_bps",
    "btc_return_5s_bps",
    "btc_return_15s_bps",
    "btc_return_30s_bps",
    "btc_return_60s_bps",
    "btc_realized_volatility_5s_bps",
    "btc_realized_volatility_15s_bps",
    "btc_realized_volatility_30s_bps",
    "btc_realized_volatility_60s_bps",
    "btc_range_30s_bps",
    "btc_range_60s_bps",
    "btc_path_efficiency_30s",
    "btc_path_efficiency_60s",
    "btc_range_position_60s",
    "btc_signed_flow_5s",
    "btc_signed_flow_30s",
    "btc_signed_flow_60s",
    "btc_path_cross_count",
    "btc_boundary_cross_count",
    "btc_seconds_since_boundary_cross",
    "btc_boundary_distance_velocity_5s_bps",
    "btc_momentum_multihorizon_score",
    "btc_momentum_acceleration_5_vs_30",
    "btc_reversal_5_vs_30",
)

CAUSAL_FEATURES = (
    "chainlink_ref_return_1s_bps",
    "chainlink_ref_return_5s_bps",
    "chainlink_ref_return_15s_bps",
    "chainlink_ref_return_30s_bps",
    "chainlink_ref_return_60s_bps",
    "chainlink_ref_boundary_gap_bps",
    "chainlink_ref_realized_volatility_30s_bps",
    "chainlink_ref_realized_volatility_60s_bps",
    "chainlink_ref_path_efficiency_60s",
    "chainlink_ref_boundary_cross_count_60s",
    "chainlink_ref_age_seconds",
    "chainlink_ref_max_gap_60s",
)

LATENT_SENSORS = (
    "twap30_margin_bps",
    "twap60_margin_bps",
    "refprice_margin_bps",
)

AVAILABILITY_COLUMNS = (
    "chainlink_twap_max_available_at",
    "binance_twap_max_available_at",
    "twap_feature_max_available_at",
    "disagreement_feature_max_available_at",
    "disagreement_opening_feature_max_available_at",
    "core_source_max_available_at",
)

ECONOMIC_COLUMNS = (
    "fee_rate",
    "quality_flags",
    "up_provider_received_at",
    "down_provider_received_at",
    "up_best_ask",
    "down_best_ask",
    "up_ask_depth",
    "down_ask_depth",
    "up_ask_vwap_5",
    "down_ask_vwap_5",
)

PANEL_COLUMNS = tuple(
    dict.fromkeys(
        (
            *KEY_COLUMNS,
            "window_end",
            *SUPERVISION_COLUMNS,
            *BRIDGE_FEATURES,
            *CAUSAL_FEATURES,
            "twap30_gap_to_opening_twap60_bps",
            "twap60_gap_to_opening_twap60_bps",
            *AVAILABILITY_COLUMNS,
            "refprice_causal_eligible",
            "oracle_model_eligible",
            "early_oracle_eligible",
            *ECONOMIC_COLUMNS,
        )
    )
)


@dataclass(frozen=True)
class Fold:
    name: str
    test_start: datetime
    test_end: datetime
    official: bool


@dataclass(frozen=True)
class TreeSpec:
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_iter: int
    max_bins: int
    calibration_c: float


@dataclass(frozen=True)
class LatentSpec:
    velocity_decay: float
    process_margin_scale: float
    process_velocity_scale: float
    initial_variance_scale: float
    regime_stickiness: float


@dataclass(frozen=True)
class Policy:
    name: str
    minimum_stressed_edge: float
    maximum_debit: float
    slippage_reserve: float
    uncertainty_reserve_scale: float
    require_margin_excludes_zero: bool
    minimum_consensus: float
    maximum_loss_recovery_wins: float


@dataclass(frozen=True)
class TournamentConfig:
    source_path: Path
    package_root: Path
    repository_root: Path
    raw: dict[str, Any]
    profile: str
    model_family: str
    random_seed: int
    data_watermark: datetime
    candidate_freeze: datetime
    historical_source_end: datetime
    prospective_end: datetime
    regimes: dict[str, datetime]
    folds: tuple[Fold, ...]
    policies: tuple[Policy, ...]
    runs: Path
    committed_results: Path
    historical_source_worktree: Path
    historical_source_cache: Path
    prospective_source_worktree: Path
    prospective_source_cache: Path
    source_code_worktree: Path
    source_code_module: Path
    history_reference_worktree: Path
    history_reference: Path
    bridge_spec: TreeSpec
    causal_spec: TreeSpec
    latent_spec: LatentSpec


@dataclass
class TemperatureCalibrator:
    slope: float
    support_markets: int

    def transform(self, probability: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
        return expit(self.slope * logit(values))


@dataclass
class TreeBundle:
    name: str
    feature_names: tuple[str, ...]
    all_missing_indices: tuple[int, ...]
    classifier: HistGradientBoostingClassifier
    lower: HistGradientBoostingRegressor
    median: HistGradientBoostingRegressor
    upper: HistGradientBoostingRegressor
    calibrator: LogisticRegression
    calibrator_uses_margin: bool
    spec: TreeSpec
    fit_end: datetime
    history_arm: str
    fit_weight_audit: dict[str, Any]
    calibration_weight_audit: dict[str, Any]

    def score_arrays(self, frame: pl.DataFrame) -> dict[str, np.ndarray]:
        matrix = _matrix(frame, self.feature_names, self.all_missing_indices)
        raw = np.clip(self.classifier.predict_proba(matrix)[:, 1], 1e-7, 1 - 1e-7)
        lower = self.lower.predict(matrix)
        median = self.median.predict(matrix)
        upper = self.upper.predict(matrix)
        width = np.maximum(upper, lower) - np.minimum(upper, lower)
        calibration = np.column_stack((logit(raw), median, width)) if self.calibrator_uses_margin else logit(raw).reshape(-1, 1)
        probability = self.calibrator.predict_proba(calibration)[:, 1]
        return {
            "probability_up": np.asarray(probability),
            "margin_lower_bps": np.minimum(lower, upper),
            "margin_median_bps": np.asarray(median),
            "margin_upper_bps": np.maximum(lower, upper),
            "uncertainty_bps": np.maximum(width / (2 * norm.ppf(0.95)), 1e-6),
        }


@dataclass
class ProbabilityTreeBundle:
    feature_names: tuple[str, ...]
    all_missing_indices: tuple[int, ...]
    classifier: HistGradientBoostingClassifier
    calibrator: LogisticRegression
    spec: TreeSpec
    fit_end: datetime
    history_arm: str
    fit_weight_audit: dict[str, Any]
    calibration_weight_audit: dict[str, Any]

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        matrix = _matrix(frame, self.feature_names, self.all_missing_indices)
        raw = np.clip(self.classifier.predict_proba(matrix)[:, 1], 1e-7, 1 - 1e-7)
        return self.calibrator.predict_proba(logit(raw).reshape(-1, 1))[:, 1]


@dataclass
class LatentParameters:
    sensor_intercepts: tuple[float, ...]
    sensor_loadings: tuple[float, ...]
    sensor_variances: tuple[float, ...]
    transition_phi: float
    process_margin_variance: float
    process_velocity_variance: float
    initial_margin_variance: float
    initial_velocity_variance: float
    regime_transition: tuple[tuple[float, ...], ...]
    reconstruction_variance_bps2: float
    fit_markets: int
    fit_rows: int
    chronological_log_likelihood: float


@dataclass
class LatentBundle:
    parameters: LatentParameters
    calibrator: TemperatureCalibrator
    fit_end: datetime
    history_arm: str

    def score_arrays(self, frame: pl.DataFrame) -> dict[str, np.ndarray]:
        ordered = frame.with_row_index("_original_row").sort(
            ["window_start", "market_id", "seconds_elapsed"]
        )
        output = {
            "probability_up": np.empty(ordered.height),
            "margin_lower_bps": np.empty(ordered.height),
            "margin_median_bps": np.empty(ordered.height),
            "margin_upper_bps": np.empty(ordered.height),
            "uncertainty_bps": np.empty(ordered.height),
        }
        offset = 0
        for market in ordered.partition_by("market_id", maintain_order=True):
            sequence = market.select(LATENT_SENSORS).to_numpy()
            result = _latent_filter(sequence, self.parameters)
            count = market.height
            output["probability_up"][offset : offset + count] = self.calibrator.transform(result["probability_up"])
            output["margin_lower_bps"][offset : offset + count] = result["margin_lower_bps"]
            output["margin_median_bps"][offset : offset + count] = result["margin_median_bps"]
            output["margin_upper_bps"][offset : offset + count] = result["margin_upper_bps"]
            output["uncertainty_bps"][offset : offset + count] = result["uncertainty_bps"]
            offset += count
        inverse = np.argsort(ordered["_original_row"].to_numpy())
        return {name: values[inverse] for name, values in output.items()}


@dataclass
class NonnegativeLogitModel:
    intercept: float
    coefficients: tuple[float, ...]

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return expit(self.intercept + np.asarray(matrix) @ np.asarray(self.coefficients))


@dataclass
class NonnegativeMarginModel:
    intercept: float
    coefficients: tuple[float, ...]

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self.intercept + np.asarray(matrix) @ np.asarray(self.coefficients)


@dataclass
class CandidateBundle:
    name: str
    probability_model: Any | None
    probability_scaler: StandardScaler | None
    calibrator: TemperatureCalibrator | None
    margin_model: Any | None
    margin_scaler: StandardScaler | None
    residual_lower: float
    residual_upper: float
    fit_end: datetime
    fit_weight_audit: dict[str, Any] | None = None
    calibration_weight_audit: dict[str, Any] | None = None

    def score_arrays(self, frame: pl.DataFrame) -> dict[str, np.ndarray]:
        constituent = _constituent_arrays(frame)
        if self.name == "frozen_bridge_control":
            probability = constituent["bridge_probability"]
            median = constituent["bridge_median"]
            lower = constituent["bridge_lower"]
            upper = constituent["bridge_upper"]
        elif self.name == "bridge_latent_equal_logit_pool":
            probability = expit(
                0.5 * logit(constituent["bridge_probability"])
                + 0.5 * logit(constituent["latent_probability"])
            )
            median = 0.5 * (constituent["bridge_median"] + constituent["latent_median"])
            lower = 0.5 * (constituent["bridge_lower"] + constituent["latent_lower"])
            upper = 0.5 * (constituent["bridge_upper"] + constituent["latent_upper"])
        elif self.name == "three_family_nonnegative_logit_stack":
            probability_matrix = _three_family_logits(constituent)
            probability = self.probability_model.predict(probability_matrix)
            margin_matrix = np.column_stack(
                (constituent["bridge_median"], constituent["latent_median"], constituent["causal_median"])
            )
            median = self.margin_model.predict(margin_matrix)
            lower = median + self.residual_lower
            upper = median + self.residual_upper
        elif self.name == "bridge_latent_uncertainty_margin_stack":
            features = _uncertainty_stack_features(constituent)
            probability = self.probability_model.predict_proba(
                self.probability_scaler.transform(features)
            )[:, 1]
            margin_features = _uncertainty_margin_features(constituent)
            median = self.margin_model.predict(self.margin_scaler.transform(margin_features))
            lower = median + self.residual_lower
            upper = median + self.residual_upper
        else:
            raise ValueError(f"unknown candidate: {self.name}")
        if self.calibrator is not None:
            probability = self.calibrator.transform(probability)
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        disagreement = _directional_disagreement(constituent)
        probability_dispersion = np.std(
            np.column_stack(
                (
                    constituent["bridge_probability"],
                    constituent["latent_probability"],
                    constituent["causal_probability"],
                )
            ),
            axis=1,
        )
        uncertainty = np.maximum((upper - lower) / (2 * norm.ppf(0.95)), 1e-6)
        uncertainty += probability_dispersion * 10.0 + disagreement.astype(float)
        return {
            "probability_up": np.clip(probability, 1e-7, 1 - 1e-7),
            "margin_lower_bps": lower,
            "margin_median_bps": median,
            "margin_upper_bps": upper,
            "uncertainty_bps": uncertainty,
        }


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _repository_root(package_root: Path) -> Path:
    common = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=package_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    common_path = Path(common)
    if not common_path.is_absolute():
        common_path = (package_root / common_path).resolve()
    return common_path.parent


def load_config(path: Path) -> TournamentConfig:
    source = path.resolve()
    package_root = source.parent.parent
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    training = raw["training"]
    paths = raw["paths"]
    repository_root = _repository_root(package_root)
    config = TournamentConfig(
        source_path=source,
        package_root=package_root,
        repository_root=repository_root,
        raw=raw,
        profile=str(training["profile"]),
        model_family=str(training["model_family"]),
        random_seed=int(training["random_seed"]),
        data_watermark=_utc(training["data_watermark"]),
        candidate_freeze=_utc(training["candidate_freeze"]),
        historical_source_end=_utc(training["historical_source_end"]),
        prospective_end=_utc(training["prospective_end"]),
        regimes={name: _utc(value) for name, value in raw["regimes"].items()},
        folds=tuple(
            Fold(
                str(row["name"]),
                _utc(row["test_start"]),
                _utc(row["test_end"]),
                bool(row["official"]),
            )
            for row in raw["oof_folds"]
        ),
        policies=tuple(Policy(**row) for row in raw["policies"]),
        runs=package_root / paths["runs"],
        committed_results=package_root / paths["committed_results"],
        historical_source_worktree=repository_root / paths["historical_source_worktree"],
        historical_source_cache=(repository_root / paths["historical_source_worktree"] / paths["historical_source_cache"]),
        prospective_source_worktree=repository_root / paths["prospective_source_worktree"],
        prospective_source_cache=(repository_root / paths["prospective_source_worktree"] / paths["prospective_source_cache"]),
        source_code_worktree=repository_root / paths["source_code_worktree"],
        source_code_module=(repository_root / paths["source_code_worktree"] / paths["source_code_module"]),
        history_reference_worktree=repository_root / paths["history_reference_worktree"],
        history_reference=(repository_root / paths["history_reference_worktree"] / paths["history_reference"]),
        bridge_spec=_tree_spec(raw["constituents"]["bridge"]),
        causal_spec=_tree_spec(raw["constituents"]["causal"]),
        latent_spec=LatentSpec(
            **{
                name: raw["constituents"]["latent"][name]
                for name in (
                    "velocity_decay",
                    "process_margin_scale",
                    "process_velocity_scale",
                    "initial_variance_scale",
                    "regime_stickiness",
                )
            }
        ),
    )
    _validate_config(config)
    return config


def _tree_spec(row: dict[str, Any]) -> TreeSpec:
    return TreeSpec(
        learning_rate=float(row["learning_rate"]),
        max_leaf_nodes=int(row["max_leaf_nodes"]),
        min_samples_leaf=int(row["min_samples_leaf"]),
        l2_regularization=float(row["l2_regularization"]),
        max_iter=int(row["max_iter"]),
        max_bins=int(row["max_bins"]),
        calibration_c=float(row["calibration_c"]),
    )


def _validate_config(config: TournamentConfig) -> None:
    training = config.raw["training"]
    if config.profile != "btc_5m_early_entry_settlement_consensus":
        raise ValueError("unexpected early-entry settlement-consensus profile")
    if not (
        training.get("training_only") is True
        and training.get("paper_only") is True
        and training.get("live_capital_allowed") is False
        and training.get("runtime_exported") is False
    ):
        raise ValueError("tournament must remain training-only and non-runtime")
    if tuple(int(value) for value in training["observation_seconds"]) != ENTRY_SECONDS:
        raise ValueError("frozen 25-point observation schedule changed")
    if tuple(training["candidates"]) != CANDIDATE_NAMES:
        raise ValueError("frozen four-candidate roster changed")
    if tuple(training["history_arms"]) != HISTORY_ARMS:
        raise ValueError("frozen history-arm roster changed")
    if training["primary_history_arm"] != "uncertainty_weighted_hybrid":
        raise ValueError("primary history arm changed")
    if training.get("weight_normalization") != WEIGHT_NORMALIZATION:
        raise ValueError("frozen training-weight normalization changed")
    if config.bridge_spec.max_bins != 127 or config.causal_spec.max_bins != 255:
        raise ValueError("frozen constituent estimator bin specifications changed")
    if not (
        config.regimes["binance_start"]
        < config.regimes["chainlink_start"]
        < config.regimes["authentic_counterfactual_start"]
        < config.regimes["twap30_transition_start"]
        < config.regimes["official_twap60_start"]
        < config.candidate_freeze
        < config.prospective_end
        == config.data_watermark
    ):
        raise ValueError("training regimes or freeze boundaries changed")
    if config.historical_source_end != config.candidate_freeze:
        raise ValueError("historical source must stop exactly at the freeze")
    if len(config.folds) != 8 or sum(fold.official for fold in config.folds) != 6:
        raise ValueError("expected two meta folds and six official folds")
    if tuple(sorted(config.folds, key=lambda fold: fold.test_start)) != config.folds:
        raise ValueError("OOF folds are not chronological")
    if any(left.test_end != right.test_start for left, right in zip(config.folds, config.folds[1:])):
        raise ValueError("OOF folds are not contiguous")
    if config.folds[-1].test_end != config.candidate_freeze:
        raise ValueError("OOF folds must consume all evidence through August 26")
    expected_policy_names = (
        "probability_edge_control",
        "edge_maximum_debit",
        "edge_uncertainty_margin",
        "edge_loss_compensation_tail_risk",
    )
    if tuple(policy.name for policy in config.policies) != expected_policy_names:
        raise ValueError("bounded four-policy grid changed")
    for path in (
        config.historical_source_cache,
        config.prospective_source_cache,
        config.source_code_module,
        config.history_reference,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    feature_names = {*BRIDGE_FEATURES, *CAUSAL_FEATURES, *LATENT_SENSORS}
    forbidden = feature_names & set(SUPERVISION_COLUMNS)
    if forbidden:
        raise ValueError("supervision reached inference features: " + ", ".join(sorted(forbidden)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    ).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    temporary.replace(path)


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    frame.write_parquet(temporary, compression="zstd", statistics=True)
    temporary.replace(path)


def _write_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.partial")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


class CheckpointStore:
    def __init__(self, root: Path, identity: str) -> None:
        self.root = root
        self.identity = identity
        root.mkdir(parents=True, exist_ok=True)

    def value(self, name: str, stage_identity: Any, builder: Callable[[], Any]) -> Any:
        path = self.root / f"{name}.joblib"
        sidecar = self.root / f"{name}.json"
        expected = _json_hash(
            {"run_identity": self.identity, "stage": name, "stage_identity": stage_identity}
        )
        if path.is_file() and sidecar.is_file():
            metadata = json.loads(sidecar.read_text())
            if metadata.get("identity") != expected:
                raise RuntimeError(f"checkpoint identity mismatch: {name}")
            if metadata.get("sha256") != file_sha256(path):
                raise RuntimeError(f"checkpoint hash mismatch: {name}")
            payload = joblib.load(path)
            if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
                raise RuntimeError(f"checkpoint schema mismatch: {name}")
            return payload["value"]
        value = builder()
        _write_joblib(
            path,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity": expected,
                "value": value,
            },
        )
        _write_json(
            sidecar,
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity": expected,
                "sha256": file_sha256(path),
                "completed_at": datetime.now(UTC).isoformat(),
            },
        )
        return value


def _git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_dirty(root: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _tag_commit(root: Path, tag: str) -> str:
    return subprocess.run(
        ["git", "rev-list", "-n", "1", tag],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _matrix(
    frame: pl.DataFrame,
    features: tuple[str, ...],
    all_missing_indices: tuple[int, ...] = (),
) -> np.ndarray:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError("missing inference features: " + ", ".join(missing))
    values = frame.select(pl.col(name).cast(pl.Float64) for name in features).to_numpy()
    if all_missing_indices:
        values[:, tuple(all_missing_indices)] = 0.0
    return values


def _verify_manifest_partitions(cache: Path, manifest_name: str) -> dict[str, Any]:
    manifest_path = cache / manifest_name
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("partitions", {})
    groups = {"binance": rows} if isinstance(rows, list) else rows
    verified = 0
    total_bytes = 0
    for group, partitions in sorted(groups.items()):
        for row in partitions:
            path = cache / row["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = file_sha256(path)
            if actual != row["sha256"]:
                raise RuntimeError(f"source hash mismatch: {path}")
            verified += 1
            total_bytes += path.stat().st_size
    return {
        "manifest": manifest_name,
        "manifest_sha256": file_sha256(manifest_path),
        "verified_partitions": verified,
        "verified_bytes": total_bytes,
    }


def _verify_frozen_inputs(config: TournamentConfig) -> dict[str, Any]:
    identity = config.raw["source_identity"]
    checks: dict[str, Any] = {}
    direct_files = (
        (
            "historical_source_manifest",
            config.historical_source_cache / "source-manifest.json",
            identity["historical_manifest_sha256"],
        ),
        (
            "historical_binance_manifest",
            config.historical_source_cache / "binance-manifest.json",
            identity["historical_binance_manifest_sha256"],
        ),
        (
            "historical_frame",
            config.historical_source_cache / "tournament-frame.parquet",
            identity["historical_frame_sha256"],
        ),
        (
            "historical_label_audit",
            config.historical_source_cache / "label-audit.parquet",
            identity["historical_label_audit_sha256"],
        ),
        (
            "prospective_source_manifest",
            config.prospective_source_cache / "source-manifest.json",
            identity["prospective_manifest_sha256"],
        ),
        (
            "prospective_binance_manifest",
            config.prospective_source_cache / "binance-manifest.json",
            identity["prospective_binance_manifest_sha256"],
        ),
        (
            "prospective_frame",
            config.prospective_source_cache / "tournament-frame.parquet",
            identity["prospective_frame_sha256"],
        ),
        ("source_code", config.source_code_module, identity["source_code_sha256"]),
        ("history_reference", config.history_reference, identity["history_reference_sha256"]),
    )
    for name, path, expected in direct_files:
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen {name} hash mismatch: {actual} != {expected}")
        checks[name] = {"path": str(path), "sha256": actual, "passed": True}

    for name, row in config.raw["constituents"].items():
        worktree = config.repository_root / row["artifact_worktree"]
        artifact = worktree / row["artifact_path"]
        source = worktree / row["source_path"]
        tag_commit = _tag_commit(config.repository_root, row["tag"])
        abandoned = subprocess.run(
            ["git", "tag", "--points-at", tag_commit, "abandoned/*"],
            cwd=config.repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if tag_commit != row["recording_commit"]:
            raise RuntimeError(f"{name} tag commit mismatch")
        if abandoned:
            raise RuntimeError(f"{name} constituent is marked abandoned: {abandoned}")
        artifact_sha = file_sha256(artifact)
        source_sha = file_sha256(source)
        if artifact_sha != row["artifact_sha256"] or source_sha != row["source_sha256"]:
            raise RuntimeError(f"{name} constituent artifact/source hash mismatch")
        checks[f"constituent_{name}"] = {
            "tag": row["tag"],
            "recording_commit": tag_commit,
            "artifact_path": str(artifact),
            "artifact_sha256": artifact_sha,
            "source_path": str(source),
            "source_sha256": source_sha,
            "abandoned": False,
            "passed": True,
        }
        reference_metrics_path = row.get("reference_metrics_path")
        if reference_metrics_path is not None:
            reference_metrics = worktree / reference_metrics_path
            reference_sha = file_sha256(reference_metrics)
            if reference_sha != row["reference_metrics_sha256"]:
                raise RuntimeError(f"{name} reference metrics hash mismatch")
            checks[f"constituent_{name}"]["reference_metrics_path"] = str(
                reference_metrics
            )
            checks[f"constituent_{name}"]["reference_metrics_sha256"] = (
                reference_sha
            )

    historical_contract = json.loads(
        (config.historical_source_cache / "source-manifest.json").read_text()
    )["contract"]
    prospective_contract = json.loads(
        (config.prospective_source_cache / "source-manifest.json").read_text()
    )["contract"]
    for name, contract in (
        ("historical", historical_contract),
        ("prospective", prospective_contract),
    ):
        if contract.get("read_only") is not True or contract.get("database_mutations") is not False:
            raise RuntimeError(f"{name} source cache is not read-only")
        if contract.get("completed_artifacts_only") is not True:
            raise RuntimeError(f"{name} source cache is not restricted to completed artifacts")
    if _utc(historical_contract["range_end"]) != config.candidate_freeze:
        raise RuntimeError("historical cache crosses the frozen boundary")
    if _utc(prospective_contract["range_end"]) != config.prospective_end:
        raise RuntimeError("prospective cache does not end at the data watermark")
    checks["source_contracts"] = {
        "historical": historical_contract,
        "prospective": prospective_contract,
        "passed": True,
    }
    checks["historical_partitions"] = _verify_manifest_partitions(
        config.historical_source_cache, "source-manifest.json"
    )
    checks["historical_binance_partitions"] = _verify_manifest_partitions(
        config.historical_source_cache, "binance-manifest.json"
    )
    checks["passed"] = True
    return checks


def _load_external_source_module(config: TournamentConfig) -> Any:
    package_path = config.source_code_module.parent
    import btc_directional_model

    value = str(package_path)
    if value not in btc_directional_model.__path__:
        btc_directional_model.__path__.append(value)
    from btc_directional_model import counterfactual_twap_state_data as source_module

    loaded_path = Path(source_module.__file__).resolve()
    if loaded_path != config.source_code_module.resolve():
        raise RuntimeError(f"unexpected frozen source module loaded: {loaded_path}")
    return source_module


def _load_partition_group(cache: Path, group: str, *, date: str | None = None) -> pl.DataFrame:
    paths = [cache / row["path"] for row in json.loads((cache / "source-manifest.json").read_text())["partitions"][group] if row["rows"]]
    if date is not None:
        paths = [path for path in paths if path.stem == date]
    if not paths:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed", rechunk=True)


def _load_binance(cache: Path, *, date: str | None = None) -> pl.DataFrame:
    manifest = json.loads((cache / "binance-manifest.json").read_text())
    rows = manifest.get("partitions", [])
    if isinstance(rows, dict):
        rows = rows.get("binance", [])
    paths = [cache / row["path"] for row in rows if row["rows"]]
    if date is not None:
        paths = [path for path in paths if path.stem == date]
    if not paths:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(path) for path in paths], how="diagonal_relaxed", rechunk=True)


def _exact_core_at_180(
    raw: pl.DataFrame,
    boundaries: pl.DataFrame,
    prepared_oracle: pl.DataFrame,
    source_module: Any,
) -> pl.DataFrame:
    frame = (
        raw.drop("opening_boundary")
        .join(boundaries.select("market_id", "opening_twap60"), on="market_id", how="inner")
        .rename({"opening_twap60": "opening_boundary"})
        .sort(["market_id", "seconds_elapsed"])
    )
    complete = (
        frame.group_by("market_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("seconds"),
            pl.col("seconds_elapsed").min().alias("minimum"),
            pl.col("seconds_elapsed").max().alias("maximum"),
        )
        .filter(
            (pl.col("rows") == 300)
            & (pl.col("seconds") == 300)
            & (pl.col("minimum") == 0)
            & (pl.col("maximum") == 299)
        )
        .select("market_id")
    )
    frame = frame.join(complete, on="market_id", how="inner")
    frame = source_module.derive_core_point_in_time_features(frame)
    frame = source_module.attach_causal_oracle_rounds(frame, prepared_oracle)
    frame = source_module.derive_oracle_point_in_time_features(frame)
    return frame.filter(
        (pl.col("seconds_elapsed") == 180) & pl.col("oracle_model_eligible")
    ).drop(
        "official_outcome",
        "final_price",
        "btc_path_positive",
        "btc_path_crossed",
        "btc_last_path_cross_second",
        "btc_boundary_positive",
        "btc_boundary_crossed",
        "btc_last_boundary_cross_second",
        "oracle_price",
        "oracle_source_timestamp",
        "oracle_block_timestamp",
        "oracle_phase_id",
        "oracle_round_id",
        "oracle_block_number",
        "oracle_log_index",
        "oracle_window_open_price",
        "oracle_round_changed",
        strict=False,
    )


def _attach_execution_exact(frame: pl.DataFrame, execution: pl.DataFrame, source_module: Any) -> pl.DataFrame:
    selected = execution.filter(
        (pl.col("seconds_elapsed") == 180)
        & ((pl.col("quality_flags") & 63) == 0)
        & pl.col("up_provider_received_at").is_not_null()
        & pl.col("down_provider_received_at").is_not_null()
        & (pl.col("up_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("down_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("up_provider_received_at") >= pl.col("observed_at") - pl.duration(seconds=10))
        & (pl.col("down_provider_received_at") >= pl.col("observed_at") - pl.duration(seconds=10))
    ).unique(["market_id", "observed_at"], keep="last")
    execution_globals = source_module.attach_execution.__globals__
    book_columns = list(execution_globals["BOOK_RAW_FEATURES"])
    columns = [
        "market_id",
        "observed_at",
        "fee_rate",
        "up_provider_received_at",
        "down_provider_received_at",
        "up_best_ask",
        "down_best_ask",
        "up_ask_depth",
        "down_ask_depth",
        "quality_flags",
        *book_columns,
    ]
    joined = frame.join(
        selected.select(*columns), on=["market_id", "observed_at"], how="left", validate="m:1"
    )
    return execution_globals["attach_book_features"](joined)


def _label_map(existing_frame: Path, start: datetime, end: datetime) -> pl.DataFrame:
    needed = (
        "market_id",
        "window_start",
        "window_end",
        *SUPERVISION_COLUMNS,
        "opening_twap60",
        "proxy_open_price",
        "proxy_open_twap30",
        "opening_refprice",
        "binance_open_twap60",
        "binance_open_twap30",
        "opening_chainlink_max_available_at",
        "opening_binance_max_available_at",
    )
    schema = pl.read_parquet_schema(existing_frame)
    missing = sorted(set(needed) - set(schema))
    if missing:
        raise RuntimeError("source frame is missing label-map fields: " + ", ".join(missing))
    return (
        pl.scan_parquet(existing_frame)
        .filter(pl.col("window_start").is_between(start, end, closed="left"))
        .select(*needed)
        .unique("market_id", keep="last")
        .collect()
        .sort(["window_start", "market_id"])
    )


def _build_180_rows(
    cache: Path,
    existing_frame: Path,
    start: datetime,
    end: datetime,
    source_module: Any,
) -> pl.DataFrame:
    labels = _label_map(existing_frame, start, end)
    boundaries = labels.select("market_id", "opening_twap60")
    oracle = _load_partition_group(cache, "oracle")
    prepared_oracle = source_module.prepare_causal_oracle_rounds(
        oracle.select(source_module.CORE_ORACLE_ROUND_SCHEMA.names)
    )
    core_rows: list[pl.DataFrame] = []
    for path in sorted((cache / "core_current").glob("*.parquet")):
        try:
            day = datetime.fromisoformat(path.stem).replace(tzinfo=UTC)
        except ValueError:
            continue
        if not (start <= day < end):
            continue
        raw = pl.read_parquet(path)
        day_boundaries = boundaries.filter(pl.col("market_id").is_in(raw["market_id"].unique()))
        if day_boundaries.is_empty():
            continue
        core_rows.append(_exact_core_at_180(raw, day_boundaries, prepared_oracle, source_module))
    if not core_rows:
        raise RuntimeError("no exact 180-second core rows were generated")
    frame = pl.concat(core_rows, how="diagonal_relaxed", rechunk=True)
    selected = labels.select(
        "market_id",
        *SUPERVISION_COLUMNS,
        "opening_twap60",
        "proxy_open_price",
        "binance_open_twap60",
    )
    frame = frame.drop("label_up", strict=False).join(selected, on="market_id", how="inner", validate="m:1")
    frame = source_module.ensure_oracle_eligibility_compatibility(frame)
    execution = _load_partition_group(cache, "execution").filter(
        pl.col("window_start").is_between(start, end, closed="left")
    )
    frame = _attach_execution_exact(frame, execution, source_module)
    candles = _load_partition_group(cache, "candles").unique("close_timestamp", keep="last")
    frame = source_module.attach_candle_context(frame, candles)
    refprice = _load_partition_group(cache, "refprice")
    frame = source_module.attach_causal_refprice_features(frame, refprice)
    refprice_columns = [name for name in frame.columns if name.startswith("chainlink_ref_")]
    frame = frame.with_columns(
        pl.when(pl.col("refprice_causal_eligible")).then(pl.col(name)).otherwise(None).alias(name)
        for name in refprice_columns
    )
    binance = _load_binance(cache)
    frame = source_module.attach_causal_twap_features(frame, labels, refprice, binance)
    return frame.sort(["window_start", "market_id", "seconds_elapsed"])


def _select_panel_columns(frame: pl.DataFrame) -> pl.DataFrame:
    result = frame.with_columns(
        pl.col("twap30_gap_to_opening_twap60_bps").alias("twap30_margin_bps"),
        pl.col("twap60_gap_to_opening_twap60_bps").alias("twap60_margin_bps"),
        pl.col("chainlink_ref_boundary_gap_bps").alias("refprice_margin_bps"),
    )
    columns = tuple(dict.fromkeys((*PANEL_COLUMNS, *LATENT_SENSORS)))
    missing = sorted(set(columns) - set(result.columns))
    if missing:
        raise RuntimeError("exact source panel is missing columns: " + ", ".join(missing))
    return result.select(*columns)


def _build_exact_panel(
    cache: Path,
    start: datetime,
    end: datetime,
    source_module: Any,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    existing_path = cache / "tournament-frame.parquet"
    existing_schema = pl.read_parquet_schema(existing_path)
    needed = tuple(dict.fromkeys((*PANEL_COLUMNS,)))
    missing = sorted(set(needed) - set(existing_schema))
    if missing:
        raise RuntimeError("immutable source frame is missing fields: " + ", ".join(missing))
    existing = (
        pl.scan_parquet(existing_path)
        .filter(
            pl.col("window_start").is_between(start, end, closed="left")
            & pl.col("seconds_elapsed").is_in(ENTRY_SECONDS[:-1])
        )
        .select(*needed)
        .collect()
    )
    at_180 = _select_panel_columns(
        _build_180_rows(cache, existing_path, start, end, source_module)
    )
    existing = _select_panel_columns(existing)
    combined = pl.concat([existing, at_180], how="vertical_relaxed", rechunk=True).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    duplicate = combined.group_by(KEY_COLUMNS).len().filter(pl.col("len") != 1)
    if duplicate.height:
        raise RuntimeError("exact source panel contains duplicate observation keys")
    complete = (
        combined.group_by("market_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("seconds_elapsed").n_unique().alias("seconds"),
            pl.col("seconds_elapsed").sort().alias("schedule"),
        )
        .filter(
            (pl.col("rows") == len(ENTRY_SECONDS))
            & (pl.col("seconds") == len(ENTRY_SECONDS))
            & (pl.col("schedule") == pl.lit(list(ENTRY_SECONDS)))
        )
        .select("market_id")
    )
    eligible = combined.join(complete, on="market_id", how="inner")
    exclusions = combined["market_id"].n_unique() - eligible["market_id"].n_unique()
    return eligible, {
        "schema_version": SOURCE_PANEL_SCHEMA_VERSION,
        "rows": eligible.height,
        "markets": eligible["market_id"].n_unique(),
        "excluded_incomplete_markets": exclusions,
        "observation_seconds": list(ENTRY_SECONDS),
        "source_frame_sha256": file_sha256(existing_path),
        "database_mutations": False,
        "new_tables": False,
        "new_data_sources": False,
        "new_ingesters": False,
    }


def _source_frame_checkpoint(
    config: TournamentConfig,
    run_root: Path,
    source_module: Any,
    source_preflight: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    path = run_root / "historical-training-panel.parquet"
    metadata_path = run_root / "historical-training-panel.json"
    expected_identity = _json_hash(
        {
            "schema_version": SOURCE_PANEL_SCHEMA_VERSION,
            "config_sha256": file_sha256(config.source_path),
            "source_manifest_sha256": source_preflight["historical_source_manifest"]["sha256"],
            "source_frame_sha256": source_preflight["historical_frame"]["sha256"],
            "source_code_sha256": source_preflight["source_code"]["sha256"],
            "start": config.regimes["binance_start"],
            "end": config.candidate_freeze,
            "seconds": ENTRY_SECONDS,
        }
    )
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("identity") != expected_identity:
            raise RuntimeError("source-panel checkpoint identity mismatch")
        if metadata.get("sha256") != file_sha256(path):
            raise RuntimeError("source-panel checkpoint hash mismatch")
        return pl.read_parquet(path), metadata["manifest"]
    frame, manifest = _build_exact_panel(
        config.historical_source_cache,
        config.regimes["binance_start"],
        config.candidate_freeze,
        source_module,
    )
    _write_parquet(path, frame)
    metadata = {
        "identity": expected_identity,
        "sha256": file_sha256(path),
        "manifest": manifest,
    }
    _write_json(metadata_path, metadata)
    return frame, manifest


def _market_schedule_audit(frame: pl.DataFrame) -> dict[str, Any]:
    grouped = frame.group_by("market_id").agg(
        pl.len().alias("rows"),
        pl.col("seconds_elapsed").n_unique().alias("unique_seconds"),
        pl.col("seconds_elapsed").sort().alias("schedule"),
        pl.col("label_up").n_unique().alias("labels"),
        pl.col("target_margin_bps").n_unique().alias("margins"),
    )
    invalid = grouped.filter(
        (pl.col("rows") != len(ENTRY_SECONDS))
        | (pl.col("unique_seconds") != len(ENTRY_SECONDS))
        | (pl.col("schedule") != pl.lit(list(ENTRY_SECONDS)))
        | (pl.col("labels") != 1)
        | (pl.col("margins") != 1)
    )
    return {
        "passed": invalid.is_empty(),
        "markets": grouped.height,
        "rows": frame.height,
        "invalid_markets": invalid.height,
        "observation_seconds": list(ENTRY_SECONDS),
    }


def _fold_audit(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    rows = []
    seen: set[str] = set()
    passed = True
    for fold in config.folds:
        test_ids = set(
            frame.filter(
                pl.col("window_start").is_between(fold.test_start, fold.test_end, closed="left")
            )["market_id"].unique().to_list()
        )
        train_ids = set(frame.filter(pl.col("window_start") < fold.test_start)["market_id"].unique().to_list())
        overlap = train_ids & test_ids
        repeated_test = seen & test_ids
        passed &= not overlap and not repeated_test
        rows.append(
            {
                "fold": fold.name,
                "train_end_exclusive": fold.test_start.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "train_markets": len(train_ids),
                "test_markets": len(test_ids),
                "train_test_overlap": len(overlap),
                "prior_test_overlap": len(repeated_test),
                "official": fold.official,
            }
        )
        seen |= test_ids
    return {"passed": bool(passed), "folds": rows}


def _availability_audit(frame: pl.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    passed = True
    for name in AVAILABILITY_COLUMNS:
        violations = frame.filter(
            pl.col(name).is_not_null() & (pl.col(name) > pl.col("observed_at"))
        ).height
        rows[name] = {"violations": violations, "passed": violations == 0}
        passed &= violations == 0
    provider_violations = frame.filter(
        (pl.col("up_provider_received_at").is_not_null() & (pl.col("up_provider_received_at") > pl.col("observed_at")))
        | (pl.col("down_provider_received_at").is_not_null() & (pl.col("down_provider_received_at") > pl.col("observed_at")))
    ).height
    rows["execution_provider_receipts"] = {
        "violations": provider_violations,
        "passed": provider_violations == 0,
    }
    passed &= provider_violations == 0
    return {"passed": bool(passed), "columns": rows}


def _official_label_audit(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    official = frame.filter(
        (pl.col("window_start") >= config.regimes["official_twap60_start"])
        & pl.col("label_source").str.starts_with("authentic_official")
    )
    disagreement = official.filter(
        pl.col("label_up") != (pl.col("official_outcome") == pl.lit("up")).cast(pl.Int8)
    )
    return {
        "passed": disagreement.is_empty(),
        "markets": official["market_id"].n_unique(),
        "disagreements": disagreement["market_id"].n_unique(),
    }


def _pre_training_integrity(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    inference_features = {*BRIDGE_FEATURES, *CAUSAL_FEATURES, *LATENT_SENSORS}
    forbidden = sorted(inference_features & set(SUPERVISION_COLUMNS))
    checks = {
        "exact_market_schedule": _market_schedule_audit(frame),
        "market_disjoint_chronological_folds": _fold_audit(frame, config),
        "causal_availability": _availability_audit(frame),
        "official_label_agreement": _official_label_audit(frame, config),
        "inference_feature_registry": {
            "passed": not forbidden,
            "features": sorted(inference_features),
            "forbidden_features": forbidden,
            "supervision_only": list(SUPERVISION_COLUMNS),
        },
        "database_mutations": False,
        "new_tables": False,
        "new_ingesters": False,
        "new_data_sources": False,
        "runtime_exported": False,
        "trading_processes_changed": False,
    }
    failed = [name for name, row in checks.items() if isinstance(row, dict) and "passed" in row and row["passed"] is not True]
    checks["passed"] = not failed
    checks["failed_checks"] = failed
    if failed:
        raise RuntimeError("pre-training integrity failure: " + ", ".join(failed))
    return checks


def _entry_band_expression() -> pl.Expr:
    return (
        pl.when(pl.col("seconds_elapsed") < 90)
        .then(pl.lit("early_60_89"))
        .when(pl.col("seconds_elapsed") < 120)
        .then(pl.lit("early_middle_90_119"))
        .when(pl.col("seconds_elapsed") < 150)
        .then(pl.lit("middle_120_149"))
        .otherwise(pl.lit("later_middle_150_180"))
    )


def _market_band_weights(frame: pl.DataFrame, base_column: str = "history_weight") -> np.ndarray:
    weighted = frame.with_columns(_entry_band_expression().alias("_entry_band"))
    counts = weighted.group_by("market_id", "_entry_band").len().rename({"len": "_band_rows"})
    joined = weighted.join(counts, on=["market_id", "_entry_band"], how="left")
    band_count = joined.group_by("market_id").agg(pl.col("_entry_band").n_unique().alias("_bands"))
    joined = joined.join(band_count, on="market_id", how="left")
    base = joined[base_column].to_numpy().astype(float) if base_column in joined.columns else np.ones(joined.height)
    weights = base / (joined["_band_rows"].to_numpy() * joined["_bands"].to_numpy())
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("market/band weights have no positive support")
    weights *= joined.height / total
    _assert_training_weights(joined, weights)
    return weights


def _assert_training_weights(frame: pl.DataFrame, weights: np.ndarray) -> None:
    values = np.asarray(weights, dtype=float)
    if len(values) != frame.height:
        raise RuntimeError("training weight row count does not match the training frame")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise RuntimeError("training weights must be finite and nonnegative")
    expected = float(frame.height)
    tolerance = max(1e-9, expected * 1e-10)
    if not math.isclose(float(values.sum()), expected, rel_tol=1e-10, abs_tol=tolerance):
        raise RuntimeError(
            "training weights violate the frozen mean-one normalization contract"
        )


def _training_weight_audit(frame: pl.DataFrame, weights: np.ndarray) -> dict[str, Any]:
    values = np.asarray(weights, dtype=float)
    _assert_training_weights(frame, values)
    audited = frame.with_columns(
        _entry_band_expression().alias("_entry_band"),
        pl.Series("_training_weight", values),
    )
    total = float(values.sum())
    squared = float(np.square(values).sum())

    def shares(column: str) -> list[dict[str, Any]]:
        if column not in audited.columns:
            return []
        rows = (
            audited.group_by(column)
            .agg(
                pl.len().alias("rows"),
                pl.col("market_id").n_unique().alias("markets"),
                pl.col("_training_weight").sum().alias("weight"),
            )
            .sort(column)
        )
        return [
            {
                column: row[column],
                "rows": int(row["rows"]),
                "markets": int(row["markets"]),
                "weight": float(row["weight"]),
                "weight_share": float(row["weight"]) / total,
            }
            for row in rows.iter_rows(named=True)
        ]

    return {
        "normalization": WEIGHT_NORMALIZATION,
        "passed": True,
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "total": total,
        "effective_sample_size": total * total / squared,
        "by_label_source": shares("label_source"),
        "by_entry_band": shares("_entry_band"),
    }


def _reliability_feature_matrix(frame: pl.DataFrame) -> np.ndarray:
    return frame.select(
        pl.col("binance_corrected_margin_bps").abs().fill_null(0.0),
        pl.col("raw_corrected_synthetic_margin_disagreement_bps").fill_null(0.0),
        pl.col("estimated_synthetic_label_error").fill_null(0.5),
        pl.col("chainlink_ref_realized_volatility_60s_bps").fill_null(0.0),
    ).to_numpy()


@dataclass
class ReliabilityModel:
    scaler: StandardScaler
    estimator: LogisticRegression
    fit_end: datetime
    support_markets: int

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(self.scaler.transform(_reliability_feature_matrix(frame)))[:, 1]


def _reliability_training_rows(frame: pl.DataFrame, end: datetime) -> pl.DataFrame:
    return (
        frame.filter(
            (pl.col("window_start") < end)
            & pl.col("binance_corrected_margin_bps").is_not_null()
            & (
                pl.col("authentic_label_up").is_not_null()
                | pl.col("proxy_label_up").is_not_null()
            )
        )
        .unique("market_id", keep="last")
        .with_columns(
            pl.when(pl.col("authentic_label_up").is_not_null())
            .then(pl.col("authentic_label_up"))
            .otherwise(pl.col("proxy_label_up"))
            .cast(pl.Int8)
            .alias("_reliability_truth")
        )
        .with_columns(
            (
                (pl.col("binance_corrected_margin_bps") >= 0).cast(pl.Int8)
                == pl.col("_reliability_truth")
            )
            .cast(pl.Int8)
            .alias("_synthetic_correct")
        )
    )


def _fit_reliability_model(frame: pl.DataFrame, end: datetime, config: TournamentConfig) -> ReliabilityModel:
    rows = _reliability_training_rows(frame, end)
    minimum = int(config.raw["reliability"]["minimum_fit_markets"])
    if rows.height < minimum or rows["_synthetic_correct"].n_unique() < 2:
        raise RuntimeError(f"insufficient cross-fitted synthetic reliability support before {end.isoformat()}")
    scaler = StandardScaler().fit(_reliability_feature_matrix(rows))
    estimator = LogisticRegression(
        C=0.25,
        max_iter=2000,
        class_weight="balanced",
        random_state=config.random_seed,
    ).fit(scaler.transform(_reliability_feature_matrix(rows)), rows["_synthetic_correct"].to_numpy())
    return ReliabilityModel(scaler, estimator, end, rows.height)


def _synthetic_reliability_preflight(frame: pl.DataFrame, config: TournamentConfig) -> dict[str, Any]:
    boundaries = (
        _utc("2026-07-01T00:00:00+00:00"),
        _utc("2026-07-16T00:00:00+00:00"),
        _utc("2026-08-01T00:00:00+00:00"),
        _utc("2026-08-07T00:00:00+00:00"),
        _utc("2026-08-14T00:00:00+00:00"),
        config.candidate_freeze,
    )
    ledgers: list[pl.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for index in range(1, len(boundaries) - 1):
        start, end = boundaries[index], boundaries[index + 1]
        model = _fit_reliability_model(frame, start, config)
        test = _reliability_training_rows(frame, end).filter(
            pl.col("window_start").is_between(start, end, closed="left")
        )
        if test.is_empty():
            continue
        probability = model.predict(test)
        scored = test.select("market_id", "window_start", "_synthetic_correct").with_columns(
            pl.Series("reliability_probability", probability),
            pl.lit(start).alias("reliability_train_end"),
            pl.lit(start).alias("prediction_block_start"),
        )
        ledgers.append(scored)
        rows.append(
            {
                "block_start": start.isoformat(),
                "block_end": end.isoformat(),
                "fit_markets": model.support_markets,
                "test_markets": test.height,
                "brier": float(np.mean((probability - test["_synthetic_correct"].to_numpy()) ** 2)),
                "accuracy": float(np.mean((probability >= 0.5) == test["_synthetic_correct"].to_numpy())),
                "strictly_earlier": model.fit_end <= start,
            }
        )
    if not ledgers:
        raise RuntimeError("synthetic reliability preflight produced no OOF evidence")
    ledger = pl.concat(ledgers)
    if ledger["market_id"].n_unique() != ledger.height:
        raise RuntimeError("synthetic reliability OOF ledger contains repeated markets")
    if not all(row["strictly_earlier"] for row in rows):
        raise RuntimeError("synthetic reliability used future markets")
    return {
        "passed": True,
        "features": [
            "absolute corrected synthetic margin",
            "raw-versus-corrected synthetic margin disagreement",
            "offline synthetic error estimate",
            "causal RefPrice volatility",
        ],
        "inference_feature": False,
        "blocks": rows,
        "oof_markets": ledger.height,
        "oof_brier": float(
            np.mean(
                (ledger["reliability_probability"].to_numpy() - ledger["_synthetic_correct"].to_numpy()) ** 2
            )
        ),
    }


def _apply_history_arm(
    frame: pl.DataFrame,
    arm: str,
    reliability: ReliabilityModel,
    config: TournamentConfig,
) -> pl.DataFrame:
    if arm not in HISTORY_ARMS:
        raise ValueError(arm)
    source = frame
    authentic = pl.col("label_source").str.starts_with("authentic")
    chainlink = pl.col("label_source") == "chainlink_reconstructed_twap60"
    synthetic = pl.col("label_source") == "binance_synthetic_twap60"
    if arm == "authentic_only":
        return source.filter(authentic).with_columns(pl.lit(1.0).alias("history_weight"))
    if arm == "chainlink_reconstructed":
        return source.filter(authentic | chainlink).with_columns(
            pl.when(authentic).then(1.0).otherwise(pl.col("base_label_weight")).alias("history_weight")
        )
    probability = reliability.predict(source)
    near_boundary = float(config.raw["reliability"]["near_boundary_bps"])
    minimum_probability = float(config.raw["reliability"]["minimum_admitted_probability"])
    high_probability = float(config.raw["reliability"]["high_reliability_probability"])
    source = source.with_columns(pl.Series("_synthetic_reliability", probability))
    tier = (
        pl.when(pl.col("binance_corrected_margin_bps").abs() < near_boundary)
        .then(0.0)
        .when(pl.col("_synthetic_reliability") < minimum_probability)
        .then(0.0)
        .when(pl.col("_synthetic_reliability") < 0.70)
        .then(0.25)
        .when(pl.col("_synthetic_reliability") < high_probability)
        .then(0.50)
        .otherwise(0.75)
    )
    synthetic_weight = tier * pl.col("_synthetic_reliability")
    if arm == "uncertainty_weighted_hybrid":
        synthetic_weight = synthetic_weight * (
            1.0 - pl.col("estimated_synthetic_label_error").fill_null(0.5).clip(0.0, 1.0)
        )
    return source.filter(authentic | chainlink | synthetic).with_columns(
        pl.when(authentic)
        .then(1.0)
        .when(chainlink)
        .then(pl.col("base_label_weight"))
        .otherwise(synthetic_weight)
        .alias("history_weight")
    ).filter(pl.col("history_weight") > 0)


def _chronological_fit_calibration(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    markets = frame.select("market_id", "window_start").unique("market_id").sort("window_start")
    if markets.height < 300:
        raise RuntimeError("insufficient markets for chronological fit/calibration")
    boundary = markets["window_start"][max(1, int(markets.height * 0.80))]
    fit = frame.filter(pl.col("window_start") < boundary)
    calibration = frame.filter(pl.col("window_start") >= boundary)
    if fit.is_empty() or calibration.is_empty():
        raise RuntimeError("chronological fit/calibration split is empty")
    return fit, calibration


def _neutralize_all_missing(matrix: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    indices = tuple(int(index) for index in np.flatnonzero(np.isnan(matrix).all(axis=0)))
    if indices:
        matrix = matrix.copy()
        matrix[:, indices] = 0.0
    return matrix, indices


def _new_classifier(spec: TreeSpec, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=spec.learning_rate,
        max_iter=spec.max_iter,
        max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        max_bins=spec.max_bins,
        early_stopping=False,
        random_state=seed,
    )


def _new_quantile(spec: TreeSpec, quantile: float, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=spec.learning_rate,
        max_iter=spec.max_iter,
        max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        max_bins=spec.max_bins,
        early_stopping=False,
        random_state=seed,
    )


def _fit_tree_bundle(
    frame: pl.DataFrame,
    *,
    name: str,
    features: tuple[str, ...],
    spec: TreeSpec,
    seed: int,
    fit_end: datetime,
    history_arm: str,
    base_label_column: str = "label_up",
    base_margin_column: str = "target_margin_bps",
    calibration_label_column: str = "label_up",
    calibrator_uses_margin: bool = False,
) -> TreeBundle:
    source = frame.filter(
        pl.col(base_label_column).is_not_null()
        & pl.col(base_margin_column).is_not_null()
        & pl.col(base_margin_column).is_finite()
    )
    fit, calibration = _chronological_fit_calibration(source)
    fit_matrix, all_missing = _neutralize_all_missing(_matrix(fit, features))
    calibration_matrix = _matrix(calibration, features, all_missing)
    fit_weights = _market_band_weights(fit)
    fit_weight_audit = _training_weight_audit(fit, fit_weights)
    classifier = _new_classifier(spec, seed).fit(
        fit_matrix,
        fit[base_label_column].to_numpy(),
        sample_weight=fit_weights,
    )
    margins = {}
    for offset, (label, quantile) in enumerate((("lower", 0.05), ("median", 0.50), ("upper", 0.95))):
        margins[label] = _new_quantile(spec, quantile, seed + offset + 1).fit(
            fit_matrix,
            fit[base_margin_column].to_numpy(),
            sample_weight=fit_weights,
        )
    raw = np.clip(classifier.predict_proba(calibration_matrix)[:, 1], 1e-7, 1 - 1e-7)
    median = margins["median"].predict(calibration_matrix)
    width = np.maximum(
        margins["upper"].predict(calibration_matrix),
        margins["lower"].predict(calibration_matrix),
    ) - np.minimum(
        margins["upper"].predict(calibration_matrix),
        margins["lower"].predict(calibration_matrix),
    )
    calibrator_x = np.column_stack((logit(raw), median, width)) if calibrator_uses_margin else logit(raw).reshape(-1, 1)
    calibration_weights = _market_band_weights(calibration)
    calibration_weight_audit = _training_weight_audit(
        calibration, calibration_weights
    )
    calibrator = LogisticRegression(
        C=spec.calibration_c,
        max_iter=2000,
        random_state=seed + 4,
    ).fit(
        calibrator_x,
        calibration[calibration_label_column].to_numpy(),
        sample_weight=calibration_weights,
    )
    return TreeBundle(
        name,
        features,
        all_missing,
        classifier,
        margins["lower"],
        margins["median"],
        margins["upper"],
        calibrator,
        calibrator_uses_margin,
        spec,
        fit_end,
        history_arm,
        fit_weight_audit,
        calibration_weight_audit,
    )


def _fit_probability_bundle(
    frame: pl.DataFrame,
    *,
    features: tuple[str, ...],
    spec: TreeSpec,
    seed: int,
    fit_end: datetime,
    history_arm: str,
) -> ProbabilityTreeBundle:
    fit, calibration = _chronological_fit_calibration(frame)
    fit_matrix, all_missing = _neutralize_all_missing(_matrix(fit, features))
    fit_weights = _market_band_weights(fit)
    fit_weight_audit = _training_weight_audit(fit, fit_weights)
    classifier = _new_classifier(spec, seed).fit(
        fit_matrix, fit["label_up"].to_numpy(), sample_weight=fit_weights
    )
    raw = np.clip(
        classifier.predict_proba(_matrix(calibration, features, all_missing))[:, 1],
        1e-7,
        1 - 1e-7,
    )
    calibration_weights = _market_band_weights(calibration)
    calibration_weight_audit = _training_weight_audit(
        calibration, calibration_weights
    )
    calibrator = LogisticRegression(
        C=spec.calibration_c, max_iter=2000, random_state=seed + 1
    ).fit(
        logit(raw).reshape(-1, 1),
        calibration["label_up"].to_numpy(),
        sample_weight=calibration_weights,
    )
    return ProbabilityTreeBundle(
        features,
        all_missing,
        classifier,
        calibrator,
        spec,
        fit_end,
        history_arm,
        fit_weight_audit,
        calibration_weight_audit,
    )


def _tree_score_frame(frame: pl.DataFrame, bundle: TreeBundle, prefix: str) -> pl.DataFrame:
    values = bundle.score_arrays(frame)
    return frame.with_columns(
        pl.Series(f"{prefix}_probability_up", values["probability_up"]),
        pl.Series(f"{prefix}_margin_lower_bps", values["margin_lower_bps"]),
        pl.Series(f"{prefix}_margin_median_bps", values["margin_median_bps"]),
        pl.Series(f"{prefix}_margin_upper_bps", values["margin_upper_bps"]),
        pl.Series(f"{prefix}_uncertainty_bps", values["uncertainty_bps"]),
    )


def _class_balanced_history_weights(targets: np.ndarray, history_weights: np.ndarray) -> np.ndarray:
    labels = np.asarray(targets, dtype=float) >= 0.0
    history = np.asarray(history_weights, dtype=float)
    weights = np.zeros(len(labels), dtype=float)
    for value in (False, True):
        selected = labels == value
        total = float(history[selected].sum())
        if total > 0:
            weights[selected] = 0.5 * history[selected] / total
    total = float(weights.sum())
    if total <= 0:
        raise RuntimeError("latent fitting has no class-balanced support")
    return weights / total


def _latent_sequences(
    frame: pl.DataFrame,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
    finite = frame.filter(
        pl.all_horizontal(pl.col(name).is_not_null() & pl.col(name).is_finite() for name in LATENT_SENSORS)
    )
    complete = (
        finite.group_by("market_id")
        .agg(pl.len().alias("rows"), pl.col("seconds_elapsed").n_unique().alias("seconds"))
        .filter(
            (pl.col("rows") == len(ENTRY_SECONDS))
            & (pl.col("seconds") == len(ENTRY_SECONDS))
        )
        .select("market_id")
    )
    finite = finite.join(complete, on="market_id", how="inner").sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    market_ids: list[str] = []
    for market in finite.partition_by("market_id", maintain_order=True):
        sequences.append(market.select(LATENT_SENSORS).to_numpy())
        targets.append(float(market["target_margin_bps"][0]))
        weights.append(float(market["history_weight"][0]))
        market_ids.append(str(market["market_id"][0]))
    return sequences, np.asarray(targets), np.asarray(weights), market_ids


def _fit_regime_transition(
    velocity_sequences: list[np.ndarray], market_weights: np.ndarray, stickiness: float
) -> np.ndarray:
    absolute = np.concatenate([np.abs(values[1:]) for values in velocity_sequences])
    stable_threshold = float(np.quantile(absolute, 0.40)) if len(absolute) else 0.0
    counts = np.ones((3, 3), dtype=float) + np.eye(3) * stickiness
    for velocity, market_weight in zip(velocity_sequences, market_weights, strict=True):
        previous_sign = np.sign(velocity[:-1])
        current_sign = np.sign(velocity[1:])
        current = np.where(
            np.abs(velocity[1:]) <= stable_threshold,
            0,
            np.where(previous_sign * current_sign < 0, 2, 1),
        )
        previous = np.where(
            np.abs(velocity[:-1]) <= stable_threshold,
            0,
            np.where(
                np.r_[False, np.sign(velocity[:-2]) * previous_sign[1:] < 0],
                2,
                1,
            ),
        )
        for source, destination in zip(previous, current, strict=True):
            counts[int(source), int(destination)] += market_weight
    return counts / counts.sum(axis=1, keepdims=True)


def _measurement_updates(
    state: np.ndarray,
    covariance: np.ndarray,
    observations: np.ndarray,
    sensor_variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    updated_state = state.copy()
    updated_covariance = covariance.copy()
    log_likelihood_value = 0.0
    observation_vector = np.array([1.0, 0.0])
    for observation, sensor_variance in zip(observations, sensor_variances, strict=True):
        innovation = float(observation - observation_vector @ updated_state)
        innovation_variance = max(
            float(observation_vector @ updated_covariance @ observation_vector + sensor_variance),
            1e-9,
        )
        gain = updated_covariance @ observation_vector / innovation_variance
        updated_state = updated_state + gain * innovation
        updated_covariance = (
            np.eye(2) - np.outer(gain, observation_vector)
        ) @ updated_covariance
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
        log_likelihood_value += -0.5 * (
            math.log(2 * math.pi * innovation_variance)
            + innovation * innovation / innovation_variance
        )
    return updated_state, updated_covariance, float(math.exp(max(log_likelihood_value, -700.0)))


def _sequence_log_likelihood(sequence: np.ndarray, parameters: LatentParameters) -> float:
    calibrated = (
        np.asarray(parameters.sensor_intercepts)[None, :]
        + sequence * np.asarray(parameters.sensor_loadings)[None, :]
    )
    sensor_variances = np.asarray(parameters.sensor_variances)
    precision = 1.0 / sensor_variances
    state = np.array([float(np.sum(calibrated[0] * precision) / precision.sum()), 0.0])
    covariance = np.diag(
        [parameters.initial_margin_variance, parameters.initial_velocity_variance]
    )
    transition = np.array(
        [[1.0, parameters.transition_phi], [0.0, parameters.transition_phi]]
    )
    process = np.diag(
        [parameters.process_margin_variance, parameters.process_velocity_variance]
    )
    total = 0.0
    for index, row in enumerate(calibrated):
        if index:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
        state, covariance, likelihood = _measurement_updates(
            state, covariance, row, sensor_variances
        )
        total += math.log(max(likelihood, 1e-300))
    return total


def _fit_latent_parameters(
    sequences: list[np.ndarray],
    targets: np.ndarray,
    history_weights: np.ndarray,
    spec: LatentSpec,
) -> LatentParameters:
    if len(sequences) != len(targets) or not sequences:
        raise RuntimeError("latent state-space fit requires aligned sequences")
    market_weights = _class_balanced_history_weights(targets, history_weights)
    sensor_count = len(LATENT_SENSORS)
    reconstruction_variance = float((0.463 / 2.576) ** 2)
    flattened_targets = np.concatenate(
        [np.full(len(sequence), target) for sequence, target in zip(sequences, targets, strict=True)]
    )
    flattened_weights = np.concatenate(
        [
            np.full(len(sequence), weight / len(sequence))
            for sequence, weight in zip(sequences, market_weights, strict=True)
        ]
    )
    intercepts = np.zeros(sensor_count)
    loadings = np.ones(sensor_count)
    variances = np.ones(sensor_count)
    calibrated_sequences: list[np.ndarray] = []
    for sensor in range(sensor_count):
        values = np.concatenate([sequence[:, sensor] for sequence in sequences])
        design = np.column_stack((np.ones(len(values)), values))
        root_weight = np.sqrt(flattened_weights)
        coefficient, *_ = np.linalg.lstsq(
            design * root_weight[:, None], flattened_targets * root_weight, rcond=None
        )
        intercepts[sensor] = float(coefficient[0])
        loadings[sensor] = float(np.clip(coefficient[1], -4.0, 4.0))
        residual = flattened_targets - (intercepts[sensor] + loadings[sensor] * values)
        variances[sensor] = max(
            float(np.sum(flattened_weights * residual * residual)) + reconstruction_variance,
            1e-4,
        )
    for sequence in sequences:
        calibrated_sequences.append(intercepts + sequence * loadings)
    precision = 1.0 / variances
    consensus = [
        np.sum(sequence * precision[None, :], axis=1) / precision.sum()
        for sequence in calibrated_sequences
    ]
    velocities = [np.diff(values, prepend=values[0]) for values in consensus]
    numerator = sum(
        float(weight * np.dot(velocity[1:], velocity[:-1]))
        for velocity, weight in zip(velocities, market_weights, strict=True)
    )
    denominator = sum(
        float(weight * np.dot(velocity[:-1], velocity[:-1]))
        for velocity, weight in zip(velocities, market_weights, strict=True)
    )
    fitted_phi = numerator / max(denominator, 1e-12)
    transition_phi = float(np.clip(0.5 * fitted_phi + 0.5 * spec.velocity_decay, -0.95, 0.98))
    margin_residuals: list[np.ndarray] = []
    velocity_residuals: list[np.ndarray] = []
    residual_weights: list[np.ndarray] = []
    for values, velocity, weight in zip(consensus, velocities, market_weights, strict=True):
        margin_residuals.append(values[1:] - values[:-1] - transition_phi * velocity[:-1])
        velocity_residuals.append(velocity[1:] - transition_phi * velocity[:-1])
        residual_weights.append(np.full(len(values) - 1, weight / (len(values) - 1)))
    transition_weights = np.concatenate(residual_weights)
    transition_weights /= transition_weights.sum()
    process_margin_variance = max(
        float(np.sum(transition_weights * np.concatenate(margin_residuals) ** 2))
        * spec.process_margin_scale,
        1e-4,
    )
    process_velocity_variance = max(
        float(np.sum(transition_weights * np.concatenate(velocity_residuals) ** 2))
        * spec.process_velocity_scale,
        1e-4,
    )
    initial_errors = np.array(
        [values[0] - target for values, target in zip(consensus, targets, strict=True)]
    )
    initial_margin_variance = max(
        float(np.sum(market_weights * initial_errors**2)) * spec.initial_variance_scale,
        1e-3,
    )
    initial_velocity_variance = max(
        float(
            np.sum(
                market_weights
                * np.array([velocity[1] ** 2 for velocity in velocities])
            )
        )
        * spec.initial_variance_scale,
        1e-3,
    )
    transition = _fit_regime_transition(velocities, market_weights, spec.regime_stickiness)
    provisional = LatentParameters(
        tuple(intercepts),
        tuple(loadings),
        tuple(variances),
        transition_phi,
        process_margin_variance,
        process_velocity_variance,
        initial_margin_variance,
        initial_velocity_variance,
        tuple(tuple(float(value) for value in row) for row in transition),
        reconstruction_variance,
        len(sequences),
        sum(len(sequence) for sequence in sequences),
        0.0,
    )
    likelihood = float(sum(_sequence_log_likelihood(sequence, provisional) for sequence in sequences))
    provisional.chronological_log_likelihood = likelihood
    return provisional


def _latent_filter(sequence: np.ndarray, parameters: LatentParameters) -> dict[str, np.ndarray]:
    values = np.asarray(sequence, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(LATENT_SENSORS):
        raise ValueError("latent sequence is not a three-sensor matrix")
    available = np.isfinite(values)
    if np.any(~available.any(axis=1)):
        raise ValueError("every latent observation requires at least one causal sensor")
    calibrated = (
        np.asarray(parameters.sensor_intercepts)[None, :]
        + values * np.asarray(parameters.sensor_loadings)[None, :]
    )
    sensor_variances = np.asarray(parameters.sensor_variances)
    precision = 1.0 / sensor_variances
    initial_available = available[0]
    initial_margin = float(
        np.sum(calibrated[0, initial_available] * precision[initial_available])
        / precision[initial_available].sum()
    )
    states = np.tile(np.array([initial_margin, 0.0]), (3, 1))
    covariances = np.tile(
        np.diag([parameters.initial_margin_variance, parameters.initial_velocity_variance]),
        (3, 1, 1),
    )
    probabilities = np.array([0.60, 0.30, 0.10])
    regime_transition = np.asarray(parameters.regime_transition)
    phi = parameters.transition_phi
    phis = np.array([min(abs(phi), 0.35), max(abs(phi), 0.80), -max(abs(phi), 0.55)])
    process_scales = np.array([[0.50, 0.50], [0.90, 1.20], [1.80, 2.50]])
    means: list[float] = []
    variances: list[float] = []
    for index, row in enumerate(calibrated):
        observed = available[index]
        observed_row = row[observed]
        observed_variances = sensor_variances[observed]
        if index:
            prior = probabilities @ regime_transition
            mixed_state = np.sum(probabilities[:, None] * states, axis=0)
            centered = states - mixed_state
            mixed_covariance = np.sum(
                probabilities[:, None, None]
                * (covariances + centered[:, :, None] * centered[:, None, :]),
                axis=0,
            )
            likelihoods = np.zeros(3)
            for regime in range(3):
                transition = np.array([[1.0, phis[regime]], [0.0, phis[regime]]])
                process = np.diag(
                    [
                        parameters.process_margin_variance * process_scales[regime, 0],
                        parameters.process_velocity_variance * process_scales[regime, 1],
                    ]
                )
                states[regime] = transition @ mixed_state
                covariances[regime] = transition @ mixed_covariance @ transition.T + process
                states[regime], covariances[regime], likelihoods[regime] = _measurement_updates(
                    states[regime],
                    covariances[regime],
                    observed_row,
                    observed_variances,
                )
            probabilities = prior * np.maximum(likelihoods, 1e-300)
            probabilities /= max(float(probabilities.sum()), 1e-300)
        else:
            for regime in range(3):
                states[regime], covariances[regime], _ = _measurement_updates(
                    states[regime],
                    covariances[regime],
                    observed_row,
                    observed_variances,
                )
        mean_state = np.sum(probabilities[:, None] * states, axis=0)
        centered = states - mean_state
        covariance = np.sum(
            probabilities[:, None, None]
            * (covariances + centered[:, :, None] * centered[:, None, :]),
            axis=0,
        )
        means.append(float(mean_state[0]))
        variances.append(max(float(covariance[0, 0]), 1e-9))
    mean = np.asarray(means)
    sigma = np.sqrt(np.asarray(variances))
    return {
        "probability_up": norm.cdf(mean / sigma),
        "margin_lower_bps": mean + norm.ppf(0.05) * sigma,
        "margin_median_bps": mean,
        "margin_upper_bps": mean + norm.ppf(0.95) * sigma,
        "uncertainty_bps": sigma,
    }


def _fit_temperature(
    raw_probability: np.ndarray,
    labels: np.ndarray,
    market_ids: Iterable[str],
) -> TemperatureCalibrator:
    probability = np.clip(np.asarray(raw_probability), 1e-8, 1 - 1e-8)
    labels = np.asarray(labels, dtype=float)
    ids = np.asarray(tuple(market_ids), dtype=object)
    unique, inverse = np.unique(ids, return_inverse=True)
    counts = np.bincount(inverse)
    row_weights = 1.0 / counts[inverse]
    row_weights /= row_weights.sum()
    raw_logits = logit(probability)

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        slope = float(value[0])
        calibrated = np.clip(expit(slope * raw_logits), 1e-10, 1 - 1e-10)
        loss = -np.sum(
            row_weights
            * (labels * np.log(calibrated) + (1 - labels) * np.log(1 - calibrated))
        )
        gradient = np.sum(row_weights * (calibrated - labels) * raw_logits)
        return float(loss), np.array([gradient])

    result = minimize(
        objective,
        x0=np.array([1.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=((0.05, 8.0),),
    )
    if not result.success:
        raise RuntimeError("temperature calibration failed")
    return TemperatureCalibrator(float(result.x[0]), len(unique))


def _fit_latent_bundle(
    frame: pl.DataFrame,
    *,
    spec: LatentSpec,
    fit_end: datetime,
    history_arm: str,
) -> LatentBundle:
    fit, calibration = _chronological_fit_calibration(frame)
    sequences, targets, weights, _ = _latent_sequences(fit)
    parameters = _fit_latent_parameters(sequences, targets, weights, spec)
    calibration_sequences, calibration_targets, _, calibration_ids = _latent_sequences(calibration)
    raw = np.concatenate(
        [_latent_filter(sequence, parameters)["probability_up"] for sequence in calibration_sequences]
    )
    labels = np.concatenate(
        [np.full(len(sequence), target >= 0) for sequence, target in zip(calibration_sequences, calibration_targets, strict=True)]
    )
    ids = [market_id for market_id, sequence in zip(calibration_ids, calibration_sequences, strict=True) for _ in range(len(sequence))]
    calibrator = _fit_temperature(raw, labels, ids)
    return LatentBundle(parameters, calibrator, fit_end, history_arm)


def _latent_score_frame(frame: pl.DataFrame, bundle: LatentBundle) -> pl.DataFrame:
    values = bundle.score_arrays(frame)
    return frame.with_columns(
        pl.Series("latent_probability_up", values["probability_up"]),
        pl.Series("latent_margin_lower_bps", values["margin_lower_bps"]),
        pl.Series("latent_margin_median_bps", values["margin_median_bps"]),
        pl.Series("latent_margin_upper_bps", values["margin_upper_bps"]),
        pl.Series("latent_uncertainty_bps", values["uncertainty_bps"]),
    )


def _fit_constituents(
    frame: pl.DataFrame,
    fold: Fold,
    config: TournamentConfig,
    reliability: ReliabilityModel,
) -> dict[str, Any]:
    raw_train = frame.filter(pl.col("window_start") < fold.test_start)
    test = frame.filter(
        pl.col("window_start").is_between(fold.test_start, fold.test_end, closed="left")
    )
    primary_arm = config.raw["training"]["primary_history_arm"]
    train = _apply_history_arm(raw_train, primary_arm, reliability, config)
    bridge_train = raw_train.filter(
        pl.col("proxy_label_up").is_not_null()
        & pl.col("proxy_margin_bps").is_not_null()
        & pl.col("label_up").is_not_null()
    ).with_columns(pl.lit(1.0).alias("history_weight"))
    bridge = _fit_tree_bundle(
        bridge_train,
        name="frozen_refprice_settlement_bridge",
        features=BRIDGE_FEATURES,
        spec=config.bridge_spec,
        seed=config.random_seed + 1000,
        fit_end=fold.test_start,
        history_arm="frozen_refprice_bridge",
        base_label_column="proxy_label_up",
        base_margin_column="proxy_margin_bps",
        calibration_label_column="label_up",
        calibrator_uses_margin=True,
    )
    latent = _fit_latent_bundle(
        train,
        spec=config.latent_spec,
        fit_end=fold.test_start,
        history_arm=primary_arm,
    )
    causal = _fit_tree_bundle(
        train,
        name="frozen_refprice_causal_attribution",
        features=CAUSAL_FEATURES,
        spec=config.causal_spec,
        seed=config.random_seed + 3000,
        fit_end=fold.test_start,
        history_arm=primary_arm,
        calibrator_uses_margin=False,
    )
    ledger = _tree_score_frame(test, bridge, "bridge")
    ledger = _latent_score_frame(ledger, latent)
    ledger = _tree_score_frame(ledger, causal, "causal")
    ledger = ledger.with_columns(
        pl.lit(fold.name).alias("fold"),
        pl.lit(fold.official).alias("official_fold"),
        pl.lit(fold.test_start).alias("constituent_train_end"),
        pl.lit(fold.test_start).alias("prediction_block_start"),
        pl.lit(config.raw["constituents"]["bridge"]["tag"]).alias("bridge_model_tag"),
        pl.lit(config.raw["constituents"]["latent"]["tag"]).alias("latent_model_tag"),
        pl.lit(config.raw["constituents"]["causal"]["tag"]).alias("causal_model_tag"),
    )
    if ledger.filter(pl.col("constituent_train_end") > pl.col("prediction_block_start")).height:
        raise RuntimeError("constituent OOF prediction was not trained strictly earlier")
    if ledger.height != test.height:
        raise RuntimeError("constituent OOF ledger lost rows")
    return {"models": {"bridge": bridge, "latent": latent, "causal": causal}, "ledger": ledger}


def _constituent_arrays(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for family in ("bridge", "latent", "causal"):
        result[f"{family}_probability"] = np.clip(
            frame[f"{family}_probability_up"].to_numpy(), 1e-7, 1 - 1e-7
        )
        for statistic in ("lower", "median", "upper"):
            result[f"{family}_{statistic}"] = frame[
                f"{family}_margin_{statistic}_bps"
            ].to_numpy()
        result[f"{family}_uncertainty"] = frame[
            f"{family}_uncertainty_bps"
        ].to_numpy()
    return result


def _three_family_logits(values: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        tuple(logit(values[f"{family}_probability"]) for family in ("bridge", "latent", "causal"))
    )


def _directional_disagreement(values: dict[str, np.ndarray]) -> np.ndarray:
    votes = np.column_stack(
        tuple(values[f"{family}_probability"] >= 0.5 for family in ("bridge", "latent", "causal"))
    )
    return (votes.min(axis=1) != votes.max(axis=1))


def _margin_overlap(values: dict[str, np.ndarray]) -> np.ndarray:
    lower = np.maximum.reduce(
        tuple(values[f"{family}_lower"] for family in ("bridge", "latent", "causal"))
    )
    upper = np.minimum.reduce(
        tuple(values[f"{family}_upper"] for family in ("bridge", "latent", "causal"))
    )
    return np.maximum(upper - lower, 0.0)


def _uncertainty_stack_features(values: dict[str, np.ndarray]) -> np.ndarray:
    probabilities = np.column_stack(
        tuple(values[f"{family}_probability"] for family in ("bridge", "latent", "causal"))
    )
    medians = np.column_stack(
        tuple(values[f"{family}_median"] for family in ("bridge", "latent", "causal"))
    )
    widths = np.column_stack(
        tuple(
            values[f"{family}_upper"] - values[f"{family}_lower"]
            for family in ("bridge", "latent", "causal")
        )
    )
    return np.column_stack(
        (
            _three_family_logits(values),
            medians,
            widths,
            _directional_disagreement(values).astype(float),
            _margin_overlap(values),
            probabilities.std(axis=1),
        )
    )


def _uncertainty_margin_features(values: dict[str, np.ndarray]) -> np.ndarray:
    medians = np.column_stack(
        tuple(values[f"{family}_median"] for family in ("bridge", "latent", "causal"))
    )
    widths = np.column_stack(
        tuple(
            values[f"{family}_upper"] - values[f"{family}_lower"]
            for family in ("bridge", "latent", "causal")
        )
    )
    return np.column_stack((medians, widths, _margin_overlap(values)))


def _fit_nonnegative_logit(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> NonnegativeLogitModel:
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=float)
    w = np.asarray(weights, dtype=float)
    w /= w.sum()

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, coefficients = parameters[0], parameters[1:]
        probability = np.clip(expit(intercept + x @ coefficients), 1e-10, 1 - 1e-10)
        loss = -np.sum(w * (y * np.log(probability) + (1 - y) * np.log(1 - probability)))
        residual = w * (probability - y)
        gradient = np.r_[residual.sum(), x.T @ residual + 2 * l2 * coefficients]
        return float(loss + l2 * np.dot(coefficients, coefficients)), gradient

    result = minimize(
        objective,
        np.r_[logit(np.clip(np.average(y, weights=w), 1e-4, 1 - 1e-4)), np.full(x.shape[1], 0.20)],
        method="L-BFGS-B",
        jac=True,
        bounds=((None, None), *((0.0, None),) * x.shape[1]),
    )
    if not result.success:
        raise RuntimeError("nonnegative logit stack optimization failed")
    return NonnegativeLogitModel(float(result.x[0]), tuple(float(value) for value in result.x[1:]))


def _fit_nonnegative_margin(
    matrix: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> NonnegativeMarginModel:
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(targets, dtype=float)
    w = np.asarray(weights, dtype=float)
    w /= w.sum()

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, coefficients = parameters[0], parameters[1:]
        residual = intercept + x @ coefficients - y
        loss = np.sum(w * residual**2) + l2 * np.dot(coefficients, coefficients)
        gradient = np.r_[
            2 * np.sum(w * residual),
            2 * (x.T @ (w * residual) + l2 * coefficients),
        ]
        return float(loss), gradient

    result = minimize(
        objective,
        np.r_[np.average(y, weights=w), np.full(x.shape[1], 1 / x.shape[1])],
        method="L-BFGS-B",
        jac=True,
        bounds=((None, None), *((0.0, None),) * x.shape[1]),
    )
    if not result.success:
        raise RuntimeError("nonnegative margin stack optimization failed")
    return NonnegativeMarginModel(float(result.x[0]), tuple(float(value) for value in result.x[1:]))


def _meta_fit_calibration(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    return _chronological_fit_calibration(frame.with_columns(pl.lit(1.0).alias("history_weight")))


def _candidate_raw_probability(name: str, frame: pl.DataFrame, bundle: CandidateBundle) -> np.ndarray:
    values = _constituent_arrays(frame)
    if name == "frozen_bridge_control":
        return values["bridge_probability"]
    if name == "bridge_latent_equal_logit_pool":
        return expit(
            0.5 * logit(values["bridge_probability"])
            + 0.5 * logit(values["latent_probability"])
        )
    if name == "three_family_nonnegative_logit_stack":
        return bundle.probability_model.predict(_three_family_logits(values))
    if name == "bridge_latent_uncertainty_margin_stack":
        features = _uncertainty_stack_features(values)
        return bundle.probability_model.predict_proba(
            bundle.probability_scaler.transform(features)
        )[:, 1]
    raise ValueError(name)


def _fit_candidate_bundle(
    name: str,
    prior_oof: pl.DataFrame,
    fit_end: datetime,
    config: TournamentConfig,
) -> CandidateBundle:
    minimum = int(config.raw["ensemble"]["minimum_meta_markets"])
    if prior_oof["market_id"].n_unique() < minimum:
        raise RuntimeError(f"insufficient strictly earlier meta-OOF markets for {name}")
    fit, calibration = _meta_fit_calibration(prior_oof)
    fit_values = _constituent_arrays(fit)
    weights = _market_band_weights(fit)
    fit_weight_audit = _training_weight_audit(fit, weights)
    calibration_weights = _market_band_weights(calibration)
    calibration_weight_audit = _training_weight_audit(
        calibration, calibration_weights
    )
    labels = fit["label_up"].to_numpy()
    targets = fit["target_margin_bps"].to_numpy()
    probability_model: Any | None = None
    probability_scaler: StandardScaler | None = None
    margin_model: Any | None = None
    margin_scaler: StandardScaler | None = None
    residual_lower = 0.0
    residual_upper = 0.0
    if name == "three_family_nonnegative_logit_stack":
        l2 = float(config.raw["ensemble"]["strong_l2_regularization"])
        probability_model = _fit_nonnegative_logit(
            _three_family_logits(fit_values), labels, weights, l2
        )
        margin_matrix = np.column_stack(
            (
                fit_values["bridge_median"],
                fit_values["latent_median"],
                fit_values["causal_median"],
            )
        )
        margin_model = _fit_nonnegative_margin(margin_matrix, targets, weights, l2)
        calibration_values = _constituent_arrays(calibration)
        calibration_margin = margin_model.predict(
            np.column_stack(
                (
                    calibration_values["bridge_median"],
                    calibration_values["latent_median"],
                    calibration_values["causal_median"],
                )
            )
        )
        residual = calibration["target_margin_bps"].to_numpy() - calibration_margin
        residual_lower, residual_upper = tuple(float(np.quantile(residual, q)) for q in (0.05, 0.95))
    elif name == "bridge_latent_uncertainty_margin_stack":
        probability_scaler = StandardScaler().fit(_uncertainty_stack_features(fit_values))
        probability_model = LogisticRegression(
            C=float(config.raw["ensemble"]["calibration_c"]),
            max_iter=2000,
            random_state=config.random_seed + 5000,
        ).fit(
            probability_scaler.transform(_uncertainty_stack_features(fit_values)),
            labels,
            sample_weight=weights,
        )
        margin_features = _uncertainty_margin_features(fit_values)
        margin_scaler = StandardScaler().fit(margin_features)
        margin_model = Ridge(
            alpha=float(config.raw["ensemble"]["strong_l2_regularization"])
        ).fit(margin_scaler.transform(margin_features), targets, sample_weight=weights)
        calibration_values = _constituent_arrays(calibration)
        calibration_margin = margin_model.predict(
            margin_scaler.transform(_uncertainty_margin_features(calibration_values))
        )
        residual = calibration["target_margin_bps"].to_numpy() - calibration_margin
        residual_lower, residual_upper = tuple(float(np.quantile(residual, q)) for q in (0.05, 0.95))
    bundle = CandidateBundle(
        name,
        probability_model,
        probability_scaler,
        None,
        margin_model,
        margin_scaler,
        residual_lower,
        residual_upper,
        fit_end,
        fit_weight_audit,
        calibration_weight_audit,
    )
    raw = _candidate_raw_probability(name, calibration, bundle)
    calibrator = _fit_temperature(
        raw,
        calibration["label_up"].to_numpy(),
        calibration["market_id"].to_list(),
    )
    bundle.calibrator = calibrator
    return bundle


def _score_candidate(frame: pl.DataFrame, bundle: CandidateBundle) -> pl.DataFrame:
    values = bundle.score_arrays(frame)
    constituent = _constituent_arrays(frame)
    votes = np.column_stack(
        tuple(constituent[f"{family}_probability"] >= 0.5 for family in ("bridge", "latent", "causal"))
    )
    consensus = np.maximum(votes.mean(axis=1), 1 - votes.mean(axis=1))
    return frame.with_columns(
        pl.lit(bundle.name).alias("candidate"),
        pl.Series("probability_up", values["probability_up"]),
        pl.Series("predicted_margin_lower_bps", values["margin_lower_bps"]),
        pl.Series("predicted_margin_bps", values["margin_median_bps"]),
        pl.Series("predicted_margin_upper_bps", values["margin_upper_bps"]),
        pl.Series("prediction_uncertainty_bps", values["uncertainty_bps"]),
        pl.Series("consensus_strength", consensus),
        pl.Series("directional_disagreement", _directional_disagreement(constituent)),
        pl.Series("margin_interval_overlap_bps", _margin_overlap(constituent)),
        pl.Series(
            "prediction_dispersion",
            np.std(
                np.column_stack(
                    tuple(
                        constituent[f"{family}_probability"]
                        for family in ("bridge", "latent", "causal")
                    )
                ),
                axis=1,
            ),
        ),
    )


def _fit_and_score_candidates(
    constituent_oof: pl.DataFrame,
    config: TournamentConfig,
    checkpoints: CheckpointStore,
) -> tuple[pl.DataFrame, dict[str, dict[str, CandidateBundle]]]:
    scored: list[pl.DataFrame] = []
    models: dict[str, dict[str, CandidateBundle]] = {}
    for fold in (row for row in config.folds if row.official):
        prior = constituent_oof.filter(pl.col("window_start") < fold.test_start)
        test = constituent_oof.filter(pl.col("fold") == fold.name)
        models[fold.name] = {}
        for index, name in enumerate(CANDIDATE_NAMES):
            bundle = checkpoints.value(
                f"candidate-{fold.name}-{name}",
                {
                    "fold": asdict(fold),
                    "candidate": name,
                    "constituent_oof_hash": _frame_identity(prior),
                    "config": config.raw["ensemble"],
                },
                lambda name=name, prior=prior, fold=fold: _fit_candidate_bundle(
                    name, prior, fold.test_start, config
                ),
            )
            models[fold.name][name] = bundle
            prediction = _score_candidate(test, bundle).with_columns(
                pl.lit(fold.name).alias("candidate_fold"),
                pl.lit(fold.test_start).alias("candidate_train_end"),
            )
            if prediction.height != test.height:
                raise RuntimeError(f"candidate {name} lost OOF rows in {fold.name}")
            scored.append(prediction)
    result = pl.concat(scored, how="diagonal_relaxed", rechunk=True).sort(
        ["candidate", "window_start", "market_id", "seconds_elapsed"]
    )
    expected = sum(
        constituent_oof.filter(pl.col("official_fold")).height for _ in CANDIDATE_NAMES
    )
    if result.height != expected:
        raise RuntimeError("candidate OOF roster is incomplete")
    return result, models


def _frame_identity(frame: pl.DataFrame) -> str:
    keys = [name for name in (*KEY_COLUMNS, "fold", "label_up") if name in frame.columns]
    payload = frame.select(*keys).sort(keys[:4]).write_json()
    return hashlib.sha256(payload.encode()).hexdigest()


def _evaluation_weights(frame: pl.DataFrame) -> np.ndarray:
    counts = frame.group_by("market_id").len().rename({"len": "_market_rows"})
    joined = frame.select("market_id").join(counts, on="market_id", how="left")
    weights = 1.0 / joined["_market_rows"].to_numpy()
    return weights / weights.sum()


def _ece(labels: np.ndarray, probability: np.ndarray, weights: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if not np.any(selected):
            continue
        weight = weights[selected]
        total += float(weight.sum()) * abs(
            float(np.average(probability[selected], weights=weight))
            - float(np.average(labels[selected], weights=weight))
        )
    return total / max(float(weights.sum()), 1e-12)


def predictive_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    if frame.is_empty():
        return {
            "rows": 0,
            "markets": 0,
            "brier": None,
            "log_loss": None,
            "expected_calibration_error": None,
            "directional_accuracy": None,
            "terminal_margin_mae_bps": None,
            "quantile_interval_coverage": None,
        }
    labels = frame["label_up"].to_numpy().astype(float)
    probability = np.clip(frame["probability_up"].to_numpy(), 1e-9, 1 - 1e-9)
    margins = frame["target_margin_bps"].to_numpy()
    median = frame["predicted_margin_bps"].to_numpy()
    lower = frame["predicted_margin_lower_bps"].to_numpy()
    upper = frame["predicted_margin_upper_bps"].to_numpy()
    weights = _evaluation_weights(frame)
    return {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "brier": float(np.average((probability - labels) ** 2, weights=weights)),
        "log_loss": float(log_loss(labels, probability, sample_weight=weights, labels=[0, 1])),
        "expected_calibration_error": _ece(labels, probability, weights),
        "directional_accuracy": float(np.average((probability >= 0.5) == labels, weights=weights)),
        "terminal_margin_mae_bps": float(np.average(np.abs(median - margins), weights=weights)),
        "quantile_interval_coverage": float(
            np.average((margins >= lower) & (margins <= upper), weights=weights)
        ),
        "mean_uncertainty_bps": float(
            np.average(frame["prediction_uncertainty_bps"].to_numpy(), weights=weights)
        ),
    }


def _constituent_metric_frame(frame: pl.DataFrame, family: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.col(f"{family}_probability_up").alias("probability_up"),
        pl.col(f"{family}_margin_lower_bps").alias("predicted_margin_lower_bps"),
        pl.col(f"{family}_margin_median_bps").alias("predicted_margin_bps"),
        pl.col(f"{family}_margin_upper_bps").alias("predicted_margin_upper_bps"),
        pl.col(f"{family}_uncertainty_bps").alias("prediction_uncertainty_bps"),
    )


def _constituent_reproduction_audit(
    constituent_oof: pl.DataFrame,
    fold_models: dict[str, dict[str, Any]],
    config: TournamentConfig,
) -> dict[str, Any]:
    official = constituent_oof.filter(pl.col("official_fold"))
    if official.is_empty():
        raise RuntimeError("constituent reproduction gate has no official OOF rows")
    settings = config.raw["integrity_gates"]
    labels = official["label_up"].to_numpy().astype(float)
    evaluation_weights = _evaluation_weights(official)
    prior_probability = float(np.average(labels, weights=evaluation_weights))
    prior_brier = float(
        np.average((prior_probability - labels) ** 2, weights=evaluation_weights)
    )
    gates: list[dict[str, Any]] = []

    def maximum(name: str, actual: float, threshold: float) -> None:
        gates.append(
            {
                "name": name,
                "actual": actual,
                "operator": "<=",
                "threshold": threshold,
                "passed": actual <= threshold,
            }
        )

    def minimum(name: str, actual: float, threshold: float) -> None:
        gates.append(
            {
                "name": name,
                "actual": actual,
                "operator": ">=",
                "threshold": threshold,
                "passed": actual >= threshold,
            }
        )

    references: dict[str, Any] = {}
    bridge_row = config.raw["constituents"]["bridge"]
    bridge_reference_path = (
        config.repository_root
        / bridge_row["artifact_worktree"]
        / bridge_row["reference_metrics_path"]
    )
    bridge_reference = json.loads(bridge_reference_path.read_text())[
        "candidate_metrics"
    ][bridge_row["selected_specification"]]["brier"]
    references["bridge"] = {
        "path": str(bridge_reference_path),
        "sha256": file_sha256(bridge_reference_path),
        "brier": float(bridge_reference),
    }
    history_reference = json.loads(config.history_reference.read_text())
    primary_arm = config.raw["training"]["primary_history_arm"]
    causal_reference = history_reference["results"][primary_arm]["brier"]
    references["causal"] = {
        "path": str(config.history_reference),
        "sha256": file_sha256(config.history_reference),
        "history_arm": primary_arm,
        "brier": float(causal_reference),
    }

    families: dict[str, Any] = {}
    for family in ("bridge", "latent", "causal"):
        family_frame = _constituent_metric_frame(official, family)
        metrics = predictive_metrics(family_frame)
        probability = family_frame["probability_up"].to_numpy()
        fold_rows = []
        for fold in (row for row in config.folds if row.official):
            fold_frame = family_frame.filter(pl.col("fold") == fold.name)
            fold_metrics = predictive_metrics(fold_frame)
            fold_std = float(fold_frame["probability_up"].std(ddof=0))
            fold_rows.append(
                {
                    "fold": fold.name,
                    "rows": fold_frame.height,
                    "markets": fold_frame["market_id"].n_unique(),
                    "brier": fold_metrics["brier"],
                    "directional_accuracy": fold_metrics["directional_accuracy"],
                    "prediction_std": fold_std,
                }
            )
            minimum(
                f"{family}.{fold.name}.prediction_std",
                fold_std,
                float(settings["minimum_fold_prediction_std"]),
            )
            maximum(
                f"{family}.{fold.name}.brier",
                float(fold_metrics["brier"]),
                float(settings["maximum_fold_brier"]),
            )
        prediction_std = float(np.std(probability))
        families[family] = {
            "metrics": metrics,
            "prediction_std": prediction_std,
            "prediction_minimum": float(np.min(probability)),
            "prediction_maximum": float(np.max(probability)),
            "folds": fold_rows,
        }
        minimum(
            f"{family}.prediction_std",
            prediction_std,
            float(settings["minimum_prediction_std"]),
        )
        maximum(
            f"{family}.brier_vs_empirical_prior",
            float(metrics["brier"]),
            prior_brier - float(settings["minimum_brier_advantage_vs_prior"]),
        )
        minimum(
            f"{family}.directional_accuracy",
            float(metrics["directional_accuracy"]),
            float(settings["minimum_directional_accuracy"]),
        )

    maximum(
        "bridge.reference_brier_degradation",
        float(families["bridge"]["metrics"]["brier"]),
        float(bridge_reference)
        + float(settings["maximum_bridge_reference_brier_degradation"]),
    )
    maximum(
        "causal.reference_brier_degradation",
        float(families["causal"]["metrics"]["brier"]),
        float(causal_reference)
        + float(settings["maximum_causal_reference_brier_degradation"]),
    )
    causal_coefficients = [
        abs(float(fold_models[fold.name]["causal"].calibrator.coef_[0, 0]))
        for fold in config.folds
        if fold.official
    ]
    minimum(
        "causal.minimum_calibration_coefficient",
        min(causal_coefficients),
        float(settings["minimum_causal_calibration_coefficient"]),
    )
    failed = [row for row in gates if row["passed"] is not True]
    minimum_std = float(settings["minimum_prediction_std"])
    prior_limit = prior_brier - float(
        settings["minimum_brier_advantage_vs_prior"]
    )
    integrity_failures: list[dict[str, Any]] = []
    bridge_collapsed = (
        families["bridge"]["prediction_std"] < minimum_std
        and families["bridge"]["metrics"]["brier"] > prior_limit
        and families["bridge"]["metrics"]["brier"]
        > float(bridge_reference)
        + float(settings["maximum_bridge_reference_brier_degradation"])
    )
    if bridge_collapsed:
        integrity_failures.append(
            {
                "family": "bridge",
                "reason": "joint probability-collapse and immutable-reference reproduction failure",
            }
        )
    causal_collapsed = (
        families["causal"]["prediction_std"] < minimum_std
        and families["causal"]["metrics"]["brier"] > prior_limit
        and families["causal"]["metrics"]["brier"]
        > float(causal_reference)
        + float(settings["maximum_causal_reference_brier_degradation"])
        and min(causal_coefficients)
        < float(settings["minimum_causal_calibration_coefficient"])
    )
    if causal_collapsed:
        integrity_failures.append(
            {
                "family": "causal",
                "reason": "joint probability-collapse, calibration-collapse, and immutable-reference reproduction failure",
            }
        )
    latent_collapsed = (
        families["latent"]["prediction_std"] < minimum_std
        and families["latent"]["metrics"]["brier"] > prior_limit
        and families["latent"]["metrics"]["directional_accuracy"]
        < float(settings["minimum_directional_accuracy"])
    )
    if latent_collapsed:
        integrity_failures.append(
            {
                "family": "latent",
                "reason": "joint probability-collapse and chance-level reproduction failure",
            }
        )
    audit = {
        "passed": not integrity_failures,
        "executed_before_ensemble_fitting": True,
        "poor_performance_alone_is_not_an_integrity_failure": True,
        "empirical_prior_probability": prior_probability,
        "empirical_prior_brier": prior_brier,
        "references": references,
        "families": families,
        "causal_calibration_coefficients": causal_coefficients,
        "gates": gates,
        "failed_gates": failed,
        "integrity_failures": integrity_failures,
    }
    if integrity_failures:
        raise RuntimeError(
            "constituent reproduction integrity failure: "
            + ", ".join(row["family"] for row in integrity_failures)
        )
    return audit


def _paired_brier_evidence(
    control: pl.DataFrame,
    candidate: pl.DataFrame,
    config: TournamentConfig,
    *,
    comparison: str,
) -> dict[str, Any]:
    join_keys = (*KEY_COLUMNS, "fold")
    paired = (
        control.select(*join_keys, "label_up", "probability_up")
        .rename({"probability_up": "control_probability"})
        .join(
            candidate.select(*join_keys, "label_up", "probability_up").rename(
                {
                    "label_up": "candidate_label_up",
                    "probability_up": "candidate_probability",
                }
            ),
            on=list(join_keys),
            how="inner",
            validate="1:1",
        )
    )
    if paired.height != control.height or paired.height != candidate.height:
        raise RuntimeError(f"paired evidence lost rows for {comparison}")
    if paired.filter(pl.col("label_up") != pl.col("candidate_label_up")).height:
        raise RuntimeError(f"paired evidence labels disagree for {comparison}")
    per_market = (
        paired.with_columns(
            (
                (pl.col("candidate_probability") - pl.col("label_up")) ** 2
                - (pl.col("control_probability") - pl.col("label_up")) ** 2
            ).alias("candidate_minus_control_brier")
        )
        .group_by("fold", "market_id")
        .agg(
            pl.col("candidate_minus_control_brier").mean(),
            pl.len().alias("rows"),
        )
        .sort("fold", "market_id")
    )
    values = per_market["candidate_minus_control_brier"].to_numpy()
    if not len(values):
        raise RuntimeError(f"paired evidence has no markets for {comparison}")
    iterations = int(config.raw["integrity_gates"]["paired_bootstrap_iterations"])
    seed_offset = int(hashlib.sha256(comparison.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(config.random_seed + seed_offset)
    bootstrap = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected = rng.integers(0, len(values), size=len(values))
        bootstrap[index] = float(values[selected].mean())
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    fold_deltas = [
        {
            "fold": row["fold"],
            "markets": int(row["markets"]),
            "candidate_minus_control_brier": float(
                row["candidate_minus_control_brier"]
            ),
        }
        for row in (
            per_market.group_by("fold")
            .agg(
                pl.len().alias("markets"),
                pl.col("candidate_minus_control_brier").mean(),
            )
            .sort("fold")
        ).iter_rows(named=True)
    ]
    improving_folds = sum(
        row["candidate_minus_control_brier"] < 0 for row in fold_deltas
    )
    worsening_folds = sum(
        row["candidate_minus_control_brier"] > 0 for row in fold_deltas
    )
    minimum_effect = float(
        config.raw["integrity_gates"]["minimum_incremental_brier_effect"]
    )
    minimum_folds = int(config.raw["integrity_gates"]["minimum_improving_folds"])
    point = float(values.mean())
    if point <= -minimum_effect and upper < 0 and improving_folds >= minimum_folds:
        status = "positive"
    elif point >= minimum_effect and lower > 0 and worsening_folds >= minimum_folds:
        status = "negative"
    else:
        status = "not_demonstrated"
    return {
        "comparison": comparison,
        "unit": "market",
        "markets": per_market.height,
        "rows": paired.height,
        "candidate_minus_control_brier": point,
        "confidence_interval_95pct": {
            "lower": float(lower),
            "upper": float(upper),
            "iterations": iterations,
            "seed": config.random_seed + seed_offset,
        },
        "minimum_effect": minimum_effect,
        "minimum_consistent_folds": minimum_folds,
        "improving_folds": improving_folds,
        "worsening_folds": worsening_folds,
        "fold_deltas": fold_deltas,
        "status": status,
    }


def _history_ablation(
    frame: pl.DataFrame,
    config: TournamentConfig,
    checkpoints: CheckpointStore,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    ledgers: list[pl.DataFrame] = []
    ledger_by_arm: dict[str, pl.DataFrame] = {}
    report: dict[str, Any] = {}
    for arm_index, arm in enumerate(HISTORY_ARMS):
        arm_rows: list[pl.DataFrame] = []
        arm_weight_audits: list[dict[str, Any]] = []
        for fold_index, fold in enumerate(row for row in config.folds if row.official):
            train_raw = frame.filter(pl.col("window_start") < fold.test_start)
            reliability = _fit_reliability_model(frame, fold.test_start, config)
            train = _apply_history_arm(train_raw, arm, reliability, config)
            test = frame.filter(
                pl.col("window_start").is_between(fold.test_start, fold.test_end, closed="left")
            )
            model = checkpoints.value(
                f"history-{arm}-{fold.name}",
                {
                    "arm": arm,
                    "fold": asdict(fold),
                    "spec": asdict(config.causal_spec),
                    "train_identity": _frame_identity(train),
                },
                lambda train=train, fold=fold, arm=arm, arm_index=arm_index, fold_index=fold_index: _fit_probability_bundle(
                    train,
                    features=CAUSAL_FEATURES,
                    spec=config.causal_spec,
                    seed=config.random_seed + 8000 + arm_index * 100 + fold_index,
                    fit_end=fold.test_start,
                    history_arm=arm,
                ),
            )
            probability = model.predict(test)
            arm_weight_audits.append(
                {
                    "fold": fold.name,
                    "fit": model.fit_weight_audit,
                    "calibration": model.calibration_weight_audit,
                }
            )
            arm_rows.append(
                test.select(*KEY_COLUMNS, "label_up", "target_margin_bps", "label_source").with_columns(
                    pl.Series("probability_up", probability),
                    pl.lit(arm).alias("history_arm"),
                    pl.lit(fold.name).alias("fold"),
                    pl.lit(fold.test_start).alias("train_end"),
                )
            )
        ledger = pl.concat(arm_rows, how="vertical_relaxed")
        metric_frame = ledger.with_columns(
            pl.lit(0.0).alias("predicted_margin_bps"),
            pl.lit(-math.inf).alias("predicted_margin_lower_bps"),
            pl.lit(math.inf).alias("predicted_margin_upper_bps"),
            pl.lit(0.0).alias("prediction_uncertainty_bps"),
        )
        metric = predictive_metrics(metric_frame)
        report[arm] = {
            key: value
            for key, value in metric.items()
            if key in ("rows", "markets", "brier", "log_loss", "expected_calibration_error", "directional_accuracy")
        }
        report[arm]["training_weight_audits"] = arm_weight_audits
        ledger_by_arm[arm] = ledger
        ledgers.append(ledger)
    prior = json.loads(config.history_reference.read_text())
    report["immutable_prior_reference"] = {
        "sha256": file_sha256(config.history_reference),
        "source": str(config.history_reference),
        "metrics": prior,
        "used_as_prediction_input": False,
    }
    synthetic_evidence = _paired_brier_evidence(
        ledger_by_arm["chainlink_reconstructed"],
        ledger_by_arm["uncertainty_weighted_hybrid"],
        config,
        comparison="uncertainty_weighted_hybrid_minus_chainlink_reconstructed",
    )
    settlement_evidence = _paired_brier_evidence(
        ledger_by_arm["authentic_only"],
        ledger_by_arm["uncertainty_weighted_hybrid"],
        config,
        comparison="uncertainty_weighted_hybrid_minus_authentic_only",
    )
    report["synthetic_incremental_value"] = synthetic_evidence["status"]
    report["synthetic_incremental_evidence"] = synthetic_evidence
    report["settlement_hypothesis_evidence"] = settlement_evidence
    return pl.concat(ledgers, how="vertical_relaxed"), report


def _opportunity_panel(frame: pl.DataFrame, policy: Policy) -> pl.DataFrame:
    predicted_up = pl.col("probability_up") >= 0.5
    selected_probability = pl.when(predicted_up).then(pl.col("probability_up")).otherwise(
        1 - pl.col("probability_up")
    )
    selected_cost = pl.when(predicted_up).then(pl.col("up_ask_vwap_5")).otherwise(
        pl.col("down_ask_vwap_5")
    )
    fee = pl.col("fee_rate").fill_null(0.0) * selected_cost * (1 - selected_cost)
    uncertainty_reserve = (
        pl.col("prediction_uncertainty_bps") * policy.uncertainty_reserve_scale / 100.0
    ).clip(0.0, 0.08)
    margin_excludes_zero = pl.when(predicted_up).then(
        pl.col("predicted_margin_lower_bps") > 0
    ).otherwise(pl.col("predicted_margin_upper_bps") < 0)
    loss_amount = selected_cost + fee + policy.slippage_reserve + uncertainty_reserve
    win_amount = 1 - selected_cost - fee - policy.slippage_reserve - uncertainty_reserve
    recovery = loss_amount / win_amount.clip(1e-6, None)
    fresh = (
        ((pl.col("quality_flags").fill_null(63) & 63) == 0)
        & pl.col("up_provider_received_at").is_not_null()
        & pl.col("down_provider_received_at").is_not_null()
        & (pl.col("up_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("down_provider_received_at") <= pl.col("observed_at"))
        & (pl.col("up_provider_received_at") >= pl.col("observed_at") - pl.duration(seconds=10))
        & (pl.col("down_provider_received_at") >= pl.col("observed_at") - pl.duration(seconds=10))
    )
    panel = frame.with_columns(
        predicted_up.alias("predicted_up"),
        selected_probability.alias("selected_probability"),
        selected_cost.alias("contract_cost"),
        fee.alias("fee_per_share"),
        uncertainty_reserve.alias("uncertainty_reserve"),
        margin_excludes_zero.alias("margin_excludes_zero"),
        recovery.alias("loss_recovery_wins_at_entry"),
        fresh.alias("fresh_executable_evidence"),
    ).with_columns(
        (
            pl.col("selected_probability")
            - pl.col("contract_cost")
            - pl.col("fee_per_share")
            - policy.slippage_reserve
            - pl.col("uncertainty_reserve")
        ).alias("stressed_edge"),
        pl.when(pl.col("predicted_up"))
        .then(pl.col("label_up") == 1)
        .otherwise(pl.col("label_up") == 0)
        .alias("direction_correct"),
    )
    eligible = panel.with_columns(
        (
            pl.col("fresh_executable_evidence")
            & pl.col("contract_cost").is_not_null()
            & pl.col("contract_cost").is_finite()
            & (pl.col("stressed_edge") >= policy.minimum_stressed_edge)
            & (pl.col("contract_cost") <= policy.maximum_debit)
            & (pl.col("consensus_strength") >= policy.minimum_consensus)
            & (pl.col("loss_recovery_wins_at_entry") <= policy.maximum_loss_recovery_wins)
            & (
                pl.col("margin_excludes_zero")
                if policy.require_margin_excludes_zero
                else pl.lit(True)
            )
        ).alias("policy_eligible")
    )
    return eligible.with_columns(
        pl.lit(policy.name).alias("policy"),
        pl.lit(policy.slippage_reserve).alias("slippage_reserve"),
    )


def _select_one_trade_per_market(panel: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.filter(pl.col("policy_eligible"))
        .sort(
            ["window_start", "market_id", "seconds_elapsed", "stressed_edge"],
            descending=[False, False, False, True],
        )
        .group_by("market_id", maintain_order=True)
        .first()
        .sort(["window_start", "market_id"])
    )


def _maximum_losing_streak(values: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in values:
        if value < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _recovery_metrics(pnl: np.ndarray, timestamps: list[datetime]) -> dict[str, Any]:
    trade_counts: list[int] = []
    durations: list[float] = []
    unrecovered = 0
    for index, value in enumerate(pnl):
        if value >= 0:
            continue
        cumulative = 0.0
        recovered = False
        for later in range(index + 1, len(pnl)):
            cumulative += pnl[later]
            if cumulative >= -value:
                trade_counts.append(later - index)
                durations.append((timestamps[later] - timestamps[index]).total_seconds() / 3600.0)
                recovered = True
                break
        if not recovered:
            unrecovered += 1
    return {
        "recovered_losses": len(trade_counts),
        "unrecovered_losses": unrecovered,
        "mean_recovery_trades": float(np.mean(trade_counts)) if trade_counts else None,
        "maximum_recovery_trades": int(max(trade_counts)) if trade_counts else None,
        "mean_recovery_hours": float(np.mean(durations)) if durations else None,
        "maximum_recovery_hours": float(max(durations)) if durations else None,
    }


def _exact_accuracy_interval(wins: int, trades: int, alpha: float = 0.05) -> dict[str, float] | None:
    if trades == 0:
        return None
    lower = 0.0 if wins == 0 else float(beta.ppf(alpha / 2, wins, trades - wins + 1))
    upper = 1.0 if wins == trades else float(beta.ppf(1 - alpha / 2, wins + 1, trades - wins))
    return {"lower": lower, "upper": upper, "confidence": 1 - alpha}


def economic_metrics(trades: pl.DataFrame, scheduled_markets: int) -> dict[str, Any]:
    if trades.is_empty():
        return {
            "trades": 0,
            "trades_per_active_day": 0.0,
            "market_coverage": 0.0,
            "active_days": 0,
            "mean_contract_cost": None,
            "median_contract_cost": None,
            "maximum_contract_cost": None,
            "wins": 0,
            "losses": 0,
            "accuracy": None,
            "accuracy_exact_interval": None,
            "average_win": None,
            "average_loss": None,
            "base_pnl": 0.0,
            "stressed_pnl": 0.0,
            "base_expectancy": 0.0,
            "stressed_expectancy": 0.0,
            "profit_factor": None,
            "profit_factor_undefined_zero_losses": False,
            "loss_recovery_wins": None,
            "longest_losing_streak": 0,
            "maximum_drawdown": 0.0,
            "cvar_5pct": 0.0,
            "recovery": _recovery_metrics(np.array([]), []),
            "best_day_pnl_concentration": None,
            "pnl_excluding_best_day": 0.0,
            "maximum_hypothetical_next_loss_impact": None,
            "pnl_after_injected_stressed_loss": None,
        }
    quantity = 5.0
    correct = trades["direction_correct"].to_numpy().astype(float)
    cost = trades["contract_cost"].to_numpy()
    fee = trades["fee_per_share"].to_numpy()
    slippage = trades["slippage_reserve"].to_numpy()
    uncertainty = trades["uncertainty_reserve"].to_numpy()
    base = quantity * (correct - cost - fee)
    stressed = quantity * (correct - cost - fee - slippage - uncertainty)
    wins = int(np.count_nonzero(stressed > 0))
    losses = int(np.count_nonzero(stressed < 0))
    gross_wins = float(stressed[stressed > 0].sum())
    gross_losses = float(-stressed[stressed < 0].sum())
    average_win = gross_wins / wins if wins else None
    average_loss = -(gross_losses / losses) if losses else None
    cumulative = np.cumsum(stressed)
    peak = np.maximum.accumulate(np.r_[0.0, cumulative])
    drawdown = peak[1:] - cumulative
    tail_count = max(1, math.ceil(len(stressed) * 0.05))
    dates = trades["window_start"].dt.date()
    daily = pl.DataFrame({"date": dates, "pnl": stressed}).group_by("date").agg(
        pl.col("pnl").sum()
    ).sort("date")
    daily_values = daily["pnl"].to_numpy()
    best_day = float(daily_values.max())
    total_positive_days = float(daily_values[daily_values > 0].sum())
    hypothetical_loss = float(
        quantity
        * np.max(cost + fee + slippage + uncertainty)
    )
    timestamps = trades["window_start"].to_list()
    recovery = _recovery_metrics(stressed, timestamps)
    return {
        "trades": trades.height,
        "trades_per_active_day": trades.height / max(daily.height, 1),
        "market_coverage": trades["market_id"].n_unique() / max(scheduled_markets, 1),
        "active_days": daily.height,
        "mean_contract_cost": float(np.mean(cost)),
        "median_contract_cost": float(np.median(cost)),
        "maximum_contract_cost": float(np.max(cost)),
        "wins": wins,
        "losses": losses,
        "accuracy": float(np.mean(correct)),
        "accuracy_exact_interval": _exact_accuracy_interval(wins, trades.height),
        "average_win": average_win,
        "average_loss": average_loss,
        "base_pnl": float(base.sum()),
        "stressed_pnl": float(stressed.sum()),
        "base_expectancy": float(base.mean()),
        "stressed_expectancy": float(stressed.mean()),
        "profit_factor": gross_wins / gross_losses if gross_losses > 0 else None,
        "profit_factor_undefined_zero_losses": losses == 0,
        "loss_recovery_wins": (
            abs(average_loss) / average_win
            if average_loss is not None and average_win is not None and average_win > 0
            else None
        ),
        "longest_losing_streak": _maximum_losing_streak(stressed),
        "maximum_drawdown": float(drawdown.max(initial=0.0)),
        "cvar_5pct": float(np.mean(np.sort(stressed)[:tail_count])),
        "recovery": recovery,
        "best_day_pnl_concentration": (
            best_day / total_positive_days if total_positive_days > 0 else None
        ),
        "pnl_excluding_best_day": float(stressed.sum() - best_day),
        "maximum_hypothetical_next_loss_impact": -hypothetical_loss,
        "pnl_after_injected_stressed_loss": float(stressed.sum() - hypothetical_loss),
        "daily_pnl": daily.to_dicts(),
    }


def _classify_evidence(predictive: dict[str, Any], economic: dict[str, Any]) -> str:
    if economic["stressed_expectancy"] < 0 or economic["stressed_pnl"] < 0:
        if predictive["directional_accuracy"] is not None and predictive["directional_accuracy"] > 0.5:
            return "interesting_pattern"
        return "no_measured_edge"
    if economic["active_days"] >= 5 and economic["longest_losing_streak"] <= 2:
        return "confirmed_historical_edge"
    if economic["active_days"] >= 2:
        return "promising_sparse_edge"
    return "interesting_pattern"


def _policy_evaluation(
    candidate_oof: pl.DataFrame, config: TournamentConfig
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    official_market_count = candidate_oof.select("market_id").unique().height
    report: dict[str, Any] = {}
    panels: list[pl.DataFrame] = []
    ledgers: list[pl.DataFrame] = []
    for candidate in CANDIDATE_NAMES:
        candidate_rows = candidate_oof.filter(pl.col("candidate") == candidate)
        report[candidate] = {}
        for policy in config.policies:
            panel = _opportunity_panel(candidate_rows, policy)
            trades = _select_one_trade_per_market(panel)
            metrics = economic_metrics(trades, official_market_count)
            report[candidate][policy.name] = metrics
            panels.append(
                panel.select(
                    *KEY_COLUMNS,
                    "candidate",
                    "fold",
                    "policy",
                    "probability_up",
                    "contract_cost",
                    "fee_per_share",
                    "uncertainty_reserve",
                    "stressed_edge",
                    "loss_recovery_wins_at_entry",
                    "policy_eligible",
                    "direction_correct",
                )
            )
            if not trades.is_empty():
                ledgers.append(trades)
    panel_frame = pl.concat(panels, how="vertical_relaxed")
    trade_frame = pl.concat(ledgers, how="diagonal_relaxed") if ledgers else pl.DataFrame()
    return report, panel_frame, trade_frame


def _best_policy(metrics: dict[str, dict[str, Any]]) -> str:
    return max(
        metrics,
        key=lambda name: (
            metrics[name]["stressed_expectancy"],
            metrics[name]["stressed_pnl"],
            -(metrics[name]["loss_recovery_wins"] or math.inf),
            -metrics[name]["maximum_drawdown"],
            -metrics[name]["longest_losing_streak"],
            -metrics[name]["mean_contract_cost"] if metrics[name]["mean_contract_cost"] is not None else -math.inf,
            metrics[name]["trades"],
        ),
    )


def _constituent_diagnostics(frame: pl.DataFrame) -> dict[str, Any]:
    labels = frame["label_up"].to_numpy()
    target = frame["target_margin_bps"].to_numpy()
    probability_residual = {
        family: frame[f"{family}_probability_up"].to_numpy() - labels
        for family in ("bridge", "latent", "causal")
    }
    margin_residual = {
        family: frame[f"{family}_margin_median_bps"].to_numpy() - target
        for family in ("bridge", "latent", "causal")
    }
    pairs = (("bridge", "latent"), ("bridge", "causal"), ("latent", "causal"))
    correlations = {}
    for left, right in pairs:
        correlations[f"{left}_vs_{right}"] = {
            "probability_residual_correlation": float(
                np.corrcoef(probability_residual[left], probability_residual[right])[0, 1]
            ),
            "margin_residual_correlation": float(
                np.corrcoef(margin_residual[left], margin_residual[right])[0, 1]
            ),
        }
    votes = np.column_stack(
        tuple(frame[f"{family}_probability_up"].to_numpy() >= 0.5 for family in ("bridge", "latent", "causal"))
    )
    agreement = votes.min(axis=1) == votes.max(axis=1)
    majority = votes.sum(axis=1) >= 2
    return {
        "residual_correlations": correlations,
        "agreement": {
            "agreement_rows": int(agreement.sum()),
            "disagreement_rows": int((~agreement).sum()),
            "accuracy_when_all_agree": float(np.mean(majority[agreement] == labels[agreement])) if agreement.any() else None,
            "accuracy_when_disagree": float(np.mean(majority[~agreement] == labels[~agreement])) if (~agreement).any() else None,
        },
    }


def _slice_report(
    predictions: pl.DataFrame,
    trades: pl.DataFrame,
    best_policies: dict[str, str],
) -> dict[str, Any]:
    volatility = predictions["chainlink_ref_realized_volatility_60s_bps"].drop_nulls().to_numpy()
    q1, q2, q3 = tuple(float(np.quantile(volatility, value)) for value in (0.25, 0.50, 0.75))
    enriched = predictions.with_columns(
        _entry_band_expression().alias("entry_band"),
        pl.when(pl.col("label_up") == 1).then(pl.lit("up")).otherwise(pl.lit("down")).alias("direction"),
        pl.col("label_source").alias("settlement_regime"),
        pl.col("label_source").alias("training_history"),
        pl.when(pl.col("chainlink_ref_realized_volatility_60s_bps") <= q1)
        .then(pl.lit("q1"))
        .when(pl.col("chainlink_ref_realized_volatility_60s_bps") <= q2)
        .then(pl.lit("q2"))
        .when(pl.col("chainlink_ref_realized_volatility_60s_bps") <= q3)
        .then(pl.lit("q3"))
        .otherwise(pl.lit("q4"))
        .alias("volatility"),
        pl.when(pl.col("consensus_strength") >= 0.999)
        .then(pl.lit("unanimous"))
        .otherwise(pl.lit("two_of_three"))
        .alias("consensus"),
        pl.col("window_start").dt.date().cast(pl.String).alias("calendar_day"),
    )
    output: dict[str, Any] = {}
    for candidate in CANDIDATE_NAMES:
        output[candidate] = {}
        candidate_rows = enriched.filter(pl.col("candidate") == candidate)
        selected_trades = trades.filter(
            (pl.col("candidate") == candidate)
            & (pl.col("policy") == best_policies[candidate])
        ) if not trades.is_empty() else pl.DataFrame()
        if not selected_trades.is_empty():
            selected_trades = selected_trades.with_columns(
                pl.when(pl.col("contract_cost") < 0.35)
                .then(pl.lit("under_0_35"))
                .when(pl.col("contract_cost") < 0.50)
                .then(pl.lit("0_35_to_0_50"))
                .when(pl.col("contract_cost") < 0.65)
                .then(pl.lit("0_50_to_0_65"))
                .otherwise(pl.lit("0_65_plus"))
                .alias("contract_price_bucket")
            )
        for dimension in (
            "entry_band",
            "direction",
            "settlement_regime",
            "training_history",
            "volatility",
            "consensus",
            "calendar_day",
            "fold",
        ):
            slices = {}
            for key in candidate_rows[dimension].unique().sort().to_list():
                block = candidate_rows.filter(pl.col(dimension) == key)
                slices[str(key)] = predictive_metrics(block)
            output[candidate][dimension] = slices
        price_slices = {}
        if not selected_trades.is_empty():
            for key in selected_trades["contract_price_bucket"].unique().sort().to_list():
                block = selected_trades.filter(pl.col("contract_price_bucket") == key)
                price_slices[str(key)] = economic_metrics(
                    block, candidate_rows["market_id"].n_unique()
                )
        output[candidate]["contract_price_bucket"] = price_slices
    return output


def _fit_final_constituents(
    frame: pl.DataFrame,
    config: TournamentConfig,
    reliability: ReliabilityModel,
) -> dict[str, Any]:
    primary_arm = config.raw["training"]["primary_history_arm"]
    train = _apply_history_arm(frame, primary_arm, reliability, config)
    bridge_train = frame.filter(
        pl.col("proxy_label_up").is_not_null()
        & pl.col("proxy_margin_bps").is_not_null()
        & pl.col("label_up").is_not_null()
    ).with_columns(pl.lit(1.0).alias("history_weight"))
    return {
        "bridge": _fit_tree_bundle(
            bridge_train,
            name="frozen_refprice_settlement_bridge",
            features=BRIDGE_FEATURES,
            spec=config.bridge_spec,
            seed=config.random_seed + 11000,
            fit_end=config.candidate_freeze,
            history_arm="frozen_refprice_bridge",
            base_label_column="proxy_label_up",
            base_margin_column="proxy_margin_bps",
            calibration_label_column="label_up",
            calibrator_uses_margin=True,
        ),
        "latent": _fit_latent_bundle(
            train,
            spec=config.latent_spec,
            fit_end=config.candidate_freeze,
            history_arm=primary_arm,
        ),
        "causal": _fit_tree_bundle(
            train,
            name="frozen_refprice_causal_attribution",
            features=CAUSAL_FEATURES,
            spec=config.causal_spec,
            seed=config.random_seed + 13000,
            fit_end=config.candidate_freeze,
            history_arm=primary_arm,
        ),
    }


def _score_constituents(
    frame: pl.DataFrame,
    models: dict[str, Any],
    *,
    fold_name: str,
    train_end: datetime,
) -> pl.DataFrame:
    result = _tree_score_frame(frame, models["bridge"], "bridge")
    result = _latent_score_frame(result, models["latent"])
    result = _tree_score_frame(result, models["causal"], "causal")
    return result.with_columns(
        pl.lit(fold_name).alias("fold"),
        pl.lit(False).alias("official_fold"),
        pl.lit(train_end).alias("constituent_train_end"),
        pl.lit(train_end).alias("prediction_block_start"),
    )


def _score_all_candidates(
    constituent_frame: pl.DataFrame,
    models: dict[str, CandidateBundle],
) -> pl.DataFrame:
    return pl.concat(
        [_score_candidate(constituent_frame, models[name]) for name in CANDIDATE_NAMES],
        how="diagonal_relaxed",
        rechunk=True,
    )


def _parity_audit(
    artifact_path: Path,
    artifact: dict[str, Any],
    sample: pl.DataFrame,
) -> dict[str, Any]:
    loaded = joblib.load(artifact_path)
    if loaded["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError("serialized artifact schema changed")
    batch_constituents = _score_constituents(
        sample,
        loaded["constituents"],
        fold_name="parity",
        train_end=loaded["candidate_freeze"],
    )
    sequence_rows: list[pl.DataFrame] = []
    for market in sample.sort(["window_start", "market_id", "seconds_elapsed"]).partition_by(
        "market_id", maintain_order=True
    ):
        sequence_rows.append(
            _score_constituents(
                market,
                loaded["constituents"],
                fold_name="parity",
                train_end=loaded["candidate_freeze"],
            )
        )
    single_constituents = pl.concat(sequence_rows, how="vertical_relaxed").sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    batch_constituents = batch_constituents.sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    constituent_columns = [
        f"{family}_{name}"
        for family in ("bridge", "latent", "causal")
        for name in (
            "probability_up",
            "margin_lower_bps",
            "margin_median_bps",
            "margin_upper_bps",
            "uncertainty_bps",
        )
    ]
    constituent_error = max(
        float(
            np.max(
                np.abs(
                    batch_constituents[column].to_numpy()
                    - single_constituents[column].to_numpy()
                ),
                initial=0.0,
            )
        )
        for column in constituent_columns
    )
    batch_candidates = _score_all_candidates(
        batch_constituents, loaded["challengers"]
    ).sort(["candidate", "window_start", "market_id", "seconds_elapsed"])
    single_candidate_rows: list[pl.DataFrame] = []
    for row_index in range(batch_constituents.height):
        row = batch_constituents.slice(row_index, 1)
        single_candidate_rows.append(_score_all_candidates(row, loaded["challengers"]))
    single_candidates = pl.concat(single_candidate_rows, how="vertical_relaxed").sort(
        ["candidate", "window_start", "market_id", "seconds_elapsed"]
    )
    candidate_columns = (
        "probability_up",
        "predicted_margin_lower_bps",
        "predicted_margin_bps",
        "predicted_margin_upper_bps",
        "prediction_uncertainty_bps",
    )
    candidate_error = max(
        float(
            np.max(
                np.abs(
                    batch_candidates[column].to_numpy()
                    - single_candidates[column].to_numpy()
                ),
                initial=0.0,
            )
        )
        for column in candidate_columns
    )
    perturbed = sample.with_columns(
        (1 - pl.col("label_up")).alias("label_up"),
        (pl.col("target_margin_bps") + 10_000).alias("target_margin_bps"),
        pl.lit("perturbed-supervision").alias("label_source"),
        pl.lit(1.0).alias("estimated_synthetic_label_error"),
    )
    perturbed_constituents = _score_constituents(
        perturbed,
        loaded["constituents"],
        fold_name="parity",
        train_end=loaded["candidate_freeze"],
    )
    perturbed_candidates = _score_all_candidates(
        perturbed_constituents, loaded["challengers"]
    ).sort(["candidate", "window_start", "market_id", "seconds_elapsed"])
    supervision_error = max(
        float(
            np.max(
                np.abs(
                    batch_candidates[column].to_numpy()
                    - perturbed_candidates[column].to_numpy()
                ),
                initial=0.0,
            )
        )
        for column in candidate_columns
    )
    maximum = max(constituent_error, candidate_error, supervision_error)
    result = {
        "passed": maximum <= 1e-12,
        "tolerance": 1e-12,
        "constituent_batch_vs_market_sequence_max_error": constituent_error,
        "candidate_batch_vs_single_row_max_error": candidate_error,
        "supervision_perturbation_max_error": supervision_error,
        "artifact_sha256": file_sha256(artifact_path),
        "sample_rows": sample.height,
        "sample_markets": sample["market_id"].n_unique(),
    }
    if not result["passed"]:
        raise RuntimeError("batch/single-row, serialization, or supervision parity failure")
    return result


def _prospective_evaluation(
    config: TournamentConfig,
    source_module: Any,
    constituents: dict[str, Any],
    challengers: dict[str, CandidateBundle],
    best_policies: dict[str, str],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any], dict[str, Any]]:
    available_rows = (
        pl.scan_parquet(config.prospective_source_cache / "tournament-frame.parquet")
        .filter(
            pl.col("window_start").is_between(
                config.candidate_freeze, config.prospective_end, closed="left"
            )
        )
        .select(pl.len())
        .collect()
        .item()
    )
    if available_rows == 0:
        predictive_empty = predictive_metrics(pl.DataFrame())
        economic_empty = economic_metrics(pl.DataFrame(), 0)
        report = {
            candidate: {
                "policy": best_policies[candidate],
                "predictive": predictive_empty,
                "economic": economic_empty,
                "tuning_performed": False,
                "classification": "historically_consumed_holdout_unavailable",
                "reason": "no complete post-freeze causal path and executable snapshot exists",
            }
            for candidate in CANDIDATE_NAMES
        }
        empty = pl.DataFrame(
            schema={
                "market_id": pl.String,
                "window_start": pl.Datetime("us", "UTC"),
                "observed_at": pl.Datetime("us", "UTC"),
                "seconds_elapsed": pl.Int64,
                "candidate": pl.String,
            }
        )
        return empty, empty.clone(), report, {
            "status": "historically_consumed_holdout_unavailable",
            "source_panel": {
                "schema_version": SOURCE_PANEL_SCHEMA_VERSION,
                "rows": 0,
                "markets": 0,
                "excluded_incomplete_markets": 288,
                "observation_seconds": list(ENTRY_SECONDS),
            },
            "integrity": "not_applicable_no_complete_post_freeze_market",
            "tuning_performed": False,
            "loaded_after_models_and_policies_frozen": True,
        }
    prospective_frame, manifest = _build_exact_panel(
        config.prospective_source_cache,
        config.candidate_freeze,
        config.prospective_end,
        source_module,
    )
    prospective_integrity = {
        "exact_market_schedule": _market_schedule_audit(prospective_frame),
        "causal_availability": _availability_audit(prospective_frame),
        "official_label_agreement": _official_label_audit(prospective_frame, config),
    }
    failed = [name for name, row in prospective_integrity.items() if row["passed"] is not True]
    if failed:
        raise RuntimeError("post-freeze prospective integrity failure: " + ", ".join(failed))
    scored_constituents = _score_constituents(
        prospective_frame,
        constituents,
        fold_name="prospective_20260827",
        train_end=config.candidate_freeze,
    )
    predictions = _score_all_candidates(scored_constituents, challengers)
    report: dict[str, Any] = {}
    trades: list[pl.DataFrame] = []
    for candidate in CANDIDATE_NAMES:
        policy_name = best_policies[candidate]
        policy = next(row for row in config.policies if row.name == policy_name)
        rows = predictions.filter(pl.col("candidate") == candidate)
        panel = _opportunity_panel(rows, policy)
        selected = _select_one_trade_per_market(panel)
        metrics = economic_metrics(selected, prospective_frame["market_id"].n_unique())
        predictive = predictive_metrics(rows)
        report[candidate] = {
            "policy": policy_name,
            "predictive": predictive,
            "economic": metrics,
            "tuning_performed": False,
            "classification": (
                "historically_consumed_holdout_edge_replication"
                if metrics["trades"] > 0
                and metrics["stressed_expectancy"] > 0
                and metrics["stressed_pnl"] > 0
                else "historically_consumed_holdout_edge_not_confirmed"
            ),
        }
        if not selected.is_empty():
            trades.append(selected)
    trade_frame = pl.concat(trades, how="diagonal_relaxed") if trades else pl.DataFrame()
    return predictions, trade_frame, report, {
        "source_panel": manifest,
        "integrity": prospective_integrity,
        "tuning_performed": False,
        "loaded_after_models_and_policies_frozen": True,
    }


def _rank_candidates(
    predictive: dict[str, Any],
    economics: dict[str, Any],
    best_policies: dict[str, str],
) -> list[str]:
    def key(name: str) -> tuple[Any, ...]:
        economic = economics[name][best_policies[name]]
        return (
            economic["stressed_expectancy"] > 0,
            economic["stressed_expectancy"],
            economic["stressed_pnl"] > 0,
            economic["stressed_pnl"],
            -(economic["loss_recovery_wins"] or math.inf),
            -economic["maximum_drawdown"],
            -economic["longest_losing_streak"],
            economic["active_days"],
            -predictive[name]["expected_calibration_error"],
            -predictive[name]["brier"],
            -(economic["mean_contract_cost"] or math.inf),
            economic["trades"],
        )

    return sorted(CANDIDATE_NAMES, key=key, reverse=True)


def _report_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Early-Entry Settlement Consensus Tournament",
        "",
        f"Run: `{metrics['run_id']}`  ",
        f"Producing commit: `{metrics['producing_commit']}`  ",
        f"Candidate freeze: `{metrics['candidate_freeze']}`  ",
        "Deployment: unchanged; training-only artifact.",
        "",
        "## Outcome",
        "",
        f"Top historical candidate: `{metrics['ranking'][0]}`.  ",
        f"Synthetic TWAP incremental value: **{metrics['conclusions']['synthetic_twap_incremental_value']}**.  ",
        f"Constituent consensus incremental value: **{metrics['conclusions']['constituent_consensus_incremental_value']}**.",
        "This is a corrected historical rerun. The frozen August 27–28 holdout was already consumed by the prior run and is not new prospective evidence.",
        "",
        "## Constituent reproduction gate",
        "",
        "| Family | Brier | Accuracy | Probability std | Reference Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    reproduction = metrics["constituent_reproduction_audit"]
    for family in ("bridge", "latent", "causal"):
        row = reproduction["families"][family]
        reference = reproduction["references"].get(family, {}).get("brier")
        reference_value = "—" if reference is None else f"{reference:.6f}"
        lines.append(
            f"| {family} | {row['metrics']['brier']:.6f} | {row['metrics']['directional_accuracy']:.6f} | {row['prediction_std']:.6f} | {reference_value} |"
        )
    synthetic_evidence = metrics["history_ablation"][
        "synthetic_incremental_evidence"
    ]
    consensus_evidence = metrics["conclusions"]["constituent_consensus_evidence"]
    lines.extend(
        [
            "",
            f"Reproduction gate: **{'passed' if reproduction['passed'] else 'failed'}**. Causal calibration coefficient range: `{min(reproduction['causal_calibration_coefficients']):.6f}`–`{max(reproduction['causal_calibration_coefficients']):.6f}`.",
            "",
            "## Paired incremental evidence",
            "",
            f"- Synthetic history candidate-minus-control Brier: `{synthetic_evidence['candidate_minus_control_brier']:.6f}`; 95% market-bootstrap CI `{synthetic_evidence['confidence_interval_95pct']['lower']:.6f}` to `{synthetic_evidence['confidence_interval_95pct']['upper']:.6f}`; {synthetic_evidence['improving_folds']}/{len(synthetic_evidence['fold_deltas'])} improving folds; **{synthetic_evidence['status']}**.",
            f"- Best consensus candidate-minus-bridge Brier: `{consensus_evidence['candidate_minus_control_brier']:.6f}`; 95% market-bootstrap CI `{consensus_evidence['confidence_interval_95pct']['lower']:.6f}` to `{consensus_evidence['confidence_interval_95pct']['upper']:.6f}`; {consensus_evidence['improving_folds']}/{len(consensus_evidence['fold_deltas'])} improving folds; **{consensus_evidence['status']}**.",
        "",
        "## Candidate summary",
        "",
        "| Candidate | Policy | Brier | Log loss | ECE | Accuracy | Margin MAE bps | Coverage | Trades | Active days | Cost mean/max | W/L | Stressed PnL | Stressed expectancy | Profit factor | Recovery wins | Losing streak | Max DD | CVaR 5% | Classification |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in metrics["candidate_summary"]:
        def fmt(value: Any, digits: int = 6) -> str:
            return "—" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value)

        lines.append(
            "| {candidate} | {policy} | {brier} | {logloss} | {ece} | {accuracy} | {mae} | {coverage} | {trades} | {days} | {cost} | {wins}/{losses} | {pnl} | {expectancy} | {pf} | {recovery} | {streak} | {drawdown} | {cvar} | {classification} |".format(
                candidate=row["candidate"],
                policy=row["selected_policy"],
                brier=fmt(row["brier"]),
                logloss=fmt(row["log_loss"]),
                ece=fmt(row["expected_calibration_error"]),
                accuracy=fmt(row["directional_accuracy"]),
                mae=fmt(row["terminal_margin_mae_bps"], 3),
                coverage=fmt(row["market_coverage"], 4),
                trades=row["trades"],
                days=row["active_days"],
                cost=f"{fmt(row['mean_contract_cost'], 4)}/{fmt(row['maximum_contract_cost'], 4)}",
                wins=row["wins"],
                losses=row["losses"],
                pnl=fmt(row["stressed_pnl"], 4),
                expectancy=fmt(row["stressed_expectancy"], 5),
                pf=fmt(row["profit_factor"], 3),
                recovery=fmt(row["loss_recovery_wins"], 3),
                streak=row["longest_losing_streak"],
                drawdown=fmt(row["maximum_drawdown"], 4),
                cvar=fmt(row["cvar_5pct"], 4),
                classification=row["classification"],
            )
        )
    lines.extend(
        [
            "",
            "## History ablation",
            "",
            "| History | Brier | Log loss | ECE | Accuracy | Markets |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in HISTORY_ARMS:
        row = metrics["history_ablation"][arm]
        lines.append(
            f"| {arm} | {row['brier']:.6f} | {row['log_loss']:.6f} | {row['expected_calibration_error']:.6f} | {row['directional_accuracy']:.6f} | {row['markets']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen holdout replication (historically consumed)",
            "",
            "The August 27–28 batch was still loaded only after models and policy choices were frozen, with no subsequent tuning or retraining. Because the prior run already evaluated it, these results are a holdout replication and must not be represented as fresh prospective evidence.",
            "",
            "| Candidate | Policy | Brier | Trades | Stressed PnL | Expectancy | Classification |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for candidate in CANDIDATE_NAMES:
        row = metrics["prospective"][candidate]
        brier = "—" if row["predictive"]["brier"] is None else f"{row['predictive']['brier']:.6f}"
        lines.append(
            f"| {candidate} | {row['policy']} | {brier} | {row['economic']['trades']} | {row['economic']['stressed_pnl']:.4f} | {row['economic']['stressed_expectancy']:.5f} | {row['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity and limitations",
            "",
            f"- Exact schedule: 25 points at 60–180 seconds; `{metrics['source_panel']['excluded_incomplete_markets']}` incomplete eligible-source markets excluded explicitly.",
            "- All constituent predictions are chronological OOF; preprocessing, missing-column handling, calibration, reliability weighting, and stack fitting occur inside the applicable fold.",
            f"- Every regularized estimator fit passed `{metrics['training_weight_audits']['normalization']}`: finite nonnegative weights, total equal to rows, and mean equal to one. Full fold-level totals, ranges, effective sample sizes, label-source shares, and entry-band shares are in `metrics.json`.",
            "- Immutable-reference reproduction and collapse gates ran before any ensemble fitting.",
            "- Coverage and frequency are descriptive only; no coverage gate was applied.",
            "- Profit factor is reported as undefined when no observed loss exists, alongside an exact accuracy interval and an injected stressed-loss result.",
            "- Poor predictive or economic performance is a finding, not an integrity failure.",
            "- No database writes, migrations, tables, ingesters, runtime exports, deployments, or trading-process changes were made.",
        ]
    )
    return "\n".join(lines) + "\n"


def _bundle_manifest(root: Path) -> str:
    rows = []
    for path in sorted(path for path in root.rglob("*") if path.is_file() and path.name != "bundle.sha256"):
        rows.append(f"{file_sha256(path)}  {path.relative_to(root)}")
    content = "\n".join(rows) + "\n"
    (root / "bundle.sha256").write_text(content)
    return file_sha256(root / "bundle.sha256")


def _candidate_summary(
    predictive: dict[str, Any],
    economics: dict[str, Any],
    best_policies: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for candidate in CANDIDATE_NAMES:
        prediction = predictive[candidate]
        policy = best_policies[candidate]
        economic = economics[candidate][policy]
        rows.append(
            {
                "candidate": candidate,
                "selected_policy": policy,
                **prediction,
                **economic,
                "classification": _classify_evidence(prediction, economic),
            }
        )
    return rows


def run_tournament(config: TournamentConfig) -> tuple[Path, dict[str, Any]]:
    producing_commit = _git_revision(config.package_root)
    if _git_dirty(config.package_root):
        raise RuntimeError("training requires a clean committed producing worktree")
    config_sha = file_sha256(config.source_path)
    run_identity = _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "producing_commit": producing_commit,
            "config_sha256": config_sha,
            "historical_manifest_sha256": config.raw["source_identity"]["historical_manifest_sha256"],
            "candidate_freeze": config.candidate_freeze,
        }
    )
    run_root = config.runs / run_identity[:24]
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoints = CheckpointStore(run_root / "checkpoints", run_identity)

    source_preflight = checkpoints.value(
        "source-preflight",
        {"config_sha256": config_sha, "source_identity": config.raw["source_identity"]},
        lambda: _verify_frozen_inputs(config),
    )
    source_module = _load_external_source_module(config)
    frame, source_panel_manifest = _source_frame_checkpoint(
        config, run_root, source_module, source_preflight
    )
    split_manifest = {
        "schema_version": "btc-early-entry-settlement-consensus-split-v1",
        "market_level": True,
        "observation_seconds": list(ENTRY_SECONDS),
        "folds": [asdict(fold) for fold in config.folds],
    }
    split_manifest["sha256"] = _json_hash(split_manifest)
    integrity = checkpoints.value(
        "integrity-preflight",
        {
            "frame_identity": _frame_identity(frame),
            "split_sha256": split_manifest["sha256"],
        },
        lambda: _pre_training_integrity(frame, config),
    )
    reliability_preflight = checkpoints.value(
        "synthetic-reliability-preflight",
        {
            "frame_identity": _frame_identity(frame),
            "reliability_config": config.raw["reliability"],
        },
        lambda: _synthetic_reliability_preflight(frame, config),
    )

    constituent_rows: list[pl.DataFrame] = []
    fold_models: dict[str, dict[str, Any]] = {}
    for fold_index, fold in enumerate(config.folds):
        reliability = checkpoints.value(
            f"reliability-{fold.name}",
            {"fold": asdict(fold), "config": config.raw["reliability"]},
            lambda fold=fold: _fit_reliability_model(frame, fold.test_start, config),
        )
        result = checkpoints.value(
            f"constituents-{fold.name}",
            {
                "fold": asdict(fold),
                "bridge_spec": asdict(config.bridge_spec),
                "latent_spec": asdict(config.latent_spec),
                "causal_spec": asdict(config.causal_spec),
                "primary_history_arm": config.raw["training"]["primary_history_arm"],
                "frame_identity": _frame_identity(frame.filter(pl.col("window_start") < fold.test_end)),
            },
            lambda fold=fold, reliability=reliability: _fit_constituents(
                frame, fold, config, reliability
            ),
        )
        fold_models[fold.name] = result["models"]
        constituent_rows.append(result["ledger"])
    constituent_oof = pl.concat(constituent_rows, how="diagonal_relaxed", rechunk=True).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    if constituent_oof.filter(
        pl.col("constituent_train_end") > pl.col("prediction_block_start")
    ).height:
        raise RuntimeError("in-sample constituent predictions entered aligned OOF ledger")
    reproduction_audit = checkpoints.value(
        "constituent-reproduction-audit",
        {
            "constituent_oof_identity": _frame_identity(constituent_oof),
            "integrity_gates": config.raw["integrity_gates"],
            "references": {
                "bridge": config.raw["constituents"]["bridge"][
                    "reference_metrics_sha256"
                ],
                "causal": config.raw["source_identity"]["history_reference_sha256"],
            },
        },
        lambda: _constituent_reproduction_audit(
            constituent_oof, fold_models, config
        ),
    )
    candidate_oof, candidate_models = _fit_and_score_candidates(
        constituent_oof, config, checkpoints
    )
    history_ledger, history_report = _history_ablation(frame, config, checkpoints)

    official_predictions = candidate_oof.filter(pl.col("official_fold"))
    predictive = {
        candidate: predictive_metrics(
            official_predictions.filter(pl.col("candidate") == candidate)
        )
        for candidate in CANDIDATE_NAMES
    }
    economics, opportunity_panel, trade_ledger = _policy_evaluation(
        official_predictions, config
    )
    best_policies = {
        candidate: _best_policy(economics[candidate]) for candidate in CANDIDATE_NAMES
    }
    summary = _candidate_summary(predictive, economics, best_policies)
    ranking = _rank_candidates(predictive, economics, best_policies)
    diagnostics = _constituent_diagnostics(
        constituent_oof.filter(pl.col("official_fold"))
    )
    slices = _slice_report(
        official_predictions, trade_ledger, best_policies
    )

    final_reliability = checkpoints.value(
        "reliability-final",
        {"freeze": config.candidate_freeze, "config": config.raw["reliability"]},
        lambda: _fit_reliability_model(frame, config.candidate_freeze, config),
    )
    final_constituents = checkpoints.value(
        "constituents-final",
        {
            "freeze": config.candidate_freeze,
            "frame_identity": _frame_identity(frame),
            "specs": {
                "bridge": asdict(config.bridge_spec),
                "latent": asdict(config.latent_spec),
                "causal": asdict(config.causal_spec),
            },
        },
        lambda: _fit_final_constituents(frame, config, final_reliability),
    )
    final_challengers = {
        name: checkpoints.value(
            f"candidate-final-{name}",
            {
                "candidate": name,
                "freeze": config.candidate_freeze,
                "constituent_oof_identity": _frame_identity(constituent_oof),
            },
            lambda name=name: _fit_candidate_bundle(
                name, constituent_oof, config.candidate_freeze, config
            ),
        )
        for name in CANDIDATE_NAMES
    }
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_family": config.model_family,
        "producing_commit": producing_commit,
        "candidate_freeze": config.candidate_freeze,
        "observation_seconds": ENTRY_SECONDS,
        "constituent_tags": {
            name: row["tag"] for name, row in config.raw["constituents"].items()
        },
        "constituents": final_constituents,
        "challengers": final_challengers,
        "selected_policies": best_policies,
        "runtime_exported": False,
        "deployment_status": "not_deployed_training_only",
    }
    frozen_artifact_path = run_root / "frozen-tournament.joblib"
    if frozen_artifact_path.is_file():
        frozen_artifact = joblib.load(frozen_artifact_path)
        expected_identity = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_family": config.model_family,
            "producing_commit": producing_commit,
            "candidate_freeze": config.candidate_freeze,
            "selected_policies": best_policies,
        }
        actual_identity = {
            name: frozen_artifact.get(name) for name in expected_identity
        }
        if actual_identity != expected_identity:
            raise RuntimeError("frozen tournament artifact identity mismatch")
        artifact = frozen_artifact
    else:
        _write_joblib(frozen_artifact_path, artifact)
    sample_markets = frame.filter(
        pl.col("window_start") >= config.regimes["official_twap60_start"]
    )["market_id"].unique().head(8)
    parity_sample = frame.filter(pl.col("market_id").is_in(sample_markets)).sort(
        ["window_start", "market_id", "seconds_elapsed"]
    )
    parity = _parity_audit(frozen_artifact_path, artifact, parity_sample)

    prospective_partition_verification = checkpoints.value(
        "prospective-source-verification",
        {
            "manifest": config.raw["source_identity"]["prospective_manifest_sha256"],
            "binance_manifest": config.raw["source_identity"]["prospective_binance_manifest_sha256"],
            "models_frozen_sha256": parity["artifact_sha256"],
            "policies": best_policies,
        },
        lambda: {
            "source": _verify_manifest_partitions(
                config.prospective_source_cache, "source-manifest.json"
            ),
            "binance": _verify_manifest_partitions(
                config.prospective_source_cache, "binance-manifest.json"
            ),
            "passed": True,
        },
    )
    prospective_predictions, prospective_trades, prospective, prospective_audit = _prospective_evaluation(
        config,
        source_module,
        final_constituents,
        final_challengers,
        best_policies,
    )
    prospective_audit["partition_verification"] = prospective_partition_verification
    prospective_audit["historical_evidence_consumed"] = True
    prospective_audit["fresh_prospective_evidence"] = False

    bridge_economic = economics["frozen_bridge_control"][
        best_policies["frozen_bridge_control"]
    ]
    top_ensemble = next(
        name for name in ranking if name != "frozen_bridge_control"
    )
    top_economic = economics[top_ensemble][best_policies[top_ensemble]]
    consensus_evidence = _paired_brier_evidence(
        official_predictions.filter(
            pl.col("candidate") == "frozen_bridge_control"
        ),
        official_predictions.filter(pl.col("candidate") == top_ensemble),
        config,
        comparison=f"{top_ensemble}_minus_frozen_bridge_control",
    )
    consensus_value = (
        "positive"
        if consensus_evidence["status"] == "positive"
        and top_economic["stressed_expectancy"]
        > bridge_economic["stressed_expectancy"]
        else "not_demonstrated"
    )
    conclusions = {
        "synthetic_twap_incremental_value": history_report["synthetic_incremental_value"],
        "constituent_consensus_incremental_value": consensus_value,
        "constituent_consensus_evidence": consensus_evidence,
        "settlement_hypothesis": (
            "supported"
            if history_report["settlement_hypothesis_evidence"]["status"]
            == "positive"
            else "not_supported"
        ),
        "ensemble_hypothesis": "supported" if consensus_value == "positive" else "not_supported",
        "economic_hypothesis": (
            "supported"
            if any(
                economics[name][best_policies[name]]["stressed_expectancy"] > 0
                for name in CANDIDATE_NAMES
            )
            else "not_supported"
        ),
    }

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temporary = run_root / f"result-{run_id}.partial"
    final = config.committed_results / run_id
    if temporary.exists() or final.exists():
        raise FileExistsError(run_id)
    temporary.mkdir(parents=True)
    ledgers = temporary / "ledgers"
    ledgers.mkdir()
    artifact_path = temporary / "tournament.joblib"
    _copy_file_atomic(frozen_artifact_path, artifact_path)
    _write_parquet(
        ledgers / "aligned-constituent-oof.parquet",
        constituent_oof.select(
            *KEY_COLUMNS,
            "label_up",
            "target_margin_bps",
            "label_source",
            "fold",
            "official_fold",
            "constituent_train_end",
            "prediction_block_start",
            *[
                f"{family}_{name}"
                for family in ("bridge", "latent", "causal")
                for name in (
                    "probability_up",
                    "margin_lower_bps",
                    "margin_median_bps",
                    "margin_upper_bps",
                    "uncertainty_bps",
                )
            ],
        ),
    )
    _write_parquet(
        ledgers / "candidate-oof-predictions.parquet",
        official_predictions.select(
            *KEY_COLUMNS,
            "candidate",
            "fold",
            "label_up",
            "target_margin_bps",
            "label_source",
            "probability_up",
            "predicted_margin_lower_bps",
            "predicted_margin_bps",
            "predicted_margin_upper_bps",
            "prediction_uncertainty_bps",
            "consensus_strength",
            "directional_disagreement",
            "margin_interval_overlap_bps",
            "prediction_dispersion",
        ),
    )
    _write_parquet(ledgers / "opportunity-panel.parquet", opportunity_panel)
    _write_parquet(ledgers / "development-trades.parquet", trade_ledger)
    _write_parquet(ledgers / "history-ablation-oof.parquet", history_ledger)
    _write_parquet(ledgers / "prospective-predictions.parquet", prospective_predictions)
    _write_parquet(ledgers / "prospective-trades.parquet", prospective_trades)
    source_manifest = {
        "schema_version": "btc-early-entry-settlement-consensus-source-manifest-v1",
        "source_preflight": source_preflight,
        "source_panel": source_panel_manifest,
        "historical_panel_sha256": file_sha256(run_root / "historical-training-panel.parquet"),
        "prospective_audit": prospective_audit,
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "model_family": config.model_family,
        "producing_commit": producing_commit,
        "candidate_freeze": config.candidate_freeze.isoformat(),
        "data_watermark": config.data_watermark.isoformat(),
        "candidate_names": list(CANDIDATE_NAMES),
        "observation_seconds": list(ENTRY_SECONDS),
        "source_panel": source_panel_manifest,
        "split_manifest": split_manifest,
        "integrity": integrity,
        "constituent_reproduction_audit": reproduction_audit,
        "synthetic_reliability": reliability_preflight,
        "predictive": predictive,
        "history_ablation": history_report,
        "constituent_diagnostics": diagnostics,
        "economic_policy_evaluation": economics,
        "selected_policies": best_policies,
        "candidate_summary": summary,
        "ranking": ranking,
        "slices": slices,
        "prospective": prospective,
        "conclusions": conclusions,
        "parity": parity,
        "training_weight_audits": {
            "normalization": WEIGHT_NORMALIZATION,
            "constituent_oof": {
                fold_name: {
                    family: {
                        "fit": bundle.fit_weight_audit,
                        "calibration": bundle.calibration_weight_audit,
                    }
                    for family, bundle in models.items()
                    if isinstance(bundle, TreeBundle)
                }
                for fold_name, models in fold_models.items()
            },
            "candidate_oof": {
                fold_name: {
                    name: {
                        "fit": bundle.fit_weight_audit,
                        "calibration": bundle.calibration_weight_audit,
                    }
                    for name, bundle in models.items()
                }
                for fold_name, models in candidate_models.items()
            },
            "final_constituents": {
                family: {
                    "fit": bundle.fit_weight_audit,
                    "calibration": bundle.calibration_weight_audit,
                }
                for family, bundle in final_constituents.items()
                if isinstance(bundle, TreeBundle)
            },
            "final_candidates": {
                name: {
                    "fit": bundle.fit_weight_audit,
                    "calibration": bundle.calibration_weight_audit,
                }
                for name, bundle in final_challengers.items()
            },
        },
        "qualification_status": summary[next(index for index, row in enumerate(summary) if row["candidate"] == ranking[0])]["classification"],
        "deployment_status": "not_deployed_training_only",
        "runtime_exported": False,
        "historical_evidence_consumed": True,
        "corrected_historical_rerun": True,
        "database_mutations": False,
        "new_tables": False,
        "new_ingesters": False,
        "new_data_sources": False,
        "trading_processes_changed": False,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "polars_max_threads": os.environ.get("POLARS_MAX_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    model_provenance = {
        "schema_version": "btc-model-provenance-v1",
        "model_family": config.model_family,
        "model_artifact_sha256": file_sha256(artifact_path),
        "artifact_path": "tournament.joblib",
        "producing_commit": producing_commit,
        "training_run_id": run_id,
        "source_identity": source_manifest["historical_panel_sha256"],
        "source_manifest_sha256": _json_hash(source_manifest),
        "split_manifest_sha256": split_manifest["sha256"],
        "configuration_sha256": config_sha,
        "constituent_tags": artifact["constituent_tags"],
        "correction_identity": {
            "prior_defective_model_tag": config.raw["training"][
                "prior_defective_model_tag"
            ],
            "weight_normalization": WEIGHT_NORMALIZATION,
            "bridge_max_bins": config.bridge_spec.max_bins,
            "causal_max_bins": config.causal_spec.max_bins,
            "historical_evidence_consumed": True,
        },
        "qualification_status": metrics["qualification_status"],
        "historically_consumed_holdout_status": {
            name: prospective[name]["classification"] for name in CANDIDATE_NAMES
        },
        "deployment_status": "not_deployed_training_only",
        "runtime_exported": False,
    }
    _write_json(temporary / "integrity-preflight.json", integrity)
    _write_json(temporary / "source-manifest.json", source_manifest)
    _write_json(temporary / "split-manifest.json", split_manifest)
    _write_json(
        temporary / "constituent-oof-provenance.json",
        {
            "constituents": source_preflight,
            "aligned_oof_sha256": file_sha256(ledgers / "aligned-constituent-oof.parquet"),
            "strictly_earlier": True,
            "row_count": constituent_oof.height,
            "reproduction_audit": reproduction_audit,
        },
    )
    _write_json(
        temporary / "policy-grid.json",
        {
            "frozen_before_evaluation": True,
            "policies": [asdict(policy) for policy in config.policies],
            "sha256": _json_hash([asdict(policy) for policy in config.policies]),
        },
    )
    _write_json(temporary / "metrics.json", metrics)
    _write_json(temporary / "model-provenance.json", model_provenance)
    (temporary / "report.md").write_text(_report_markdown(metrics))
    _bundle_manifest(temporary)
    config.committed_results.mkdir(parents=True, exist_ok=True)
    temporary.replace(final)
    _write_json(
        run_root / "completion.json",
        {
            "run_identity": run_identity,
            "result": str(final.relative_to(config.package_root)),
            "artifact_sha256": model_provenance["model_artifact_sha256"],
            "bundle_sha256": file_sha256(final / "bundle.sha256"),
        },
    )
    return final, metrics
