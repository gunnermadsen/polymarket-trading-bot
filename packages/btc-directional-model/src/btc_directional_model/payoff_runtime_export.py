from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import polars as pl

from .core_extract import file_sha256
from .runtime_export import canonical_json_bytes, export_tree, write_immutable_directory

RUNTIME_SCHEMA = "capitonic-btc-payoff-aware-runtime-model-v1"
MANIFEST_SCHEMA = "capitonic-btc-directional-runtime-manifest-v1"
GOLDEN_SCHEMA = "capitonic-btc-payoff-aware-golden-vectors-v1"
Q5_FEATURE_SCHEMA = "btc-5m-payoff-aware-q5-features-v1"
MIDDLE_FEATURE_SCHEMA = "btc-5m-payoff-aware-middle-features-v1"
MIDDLE_OI_FEATURE_SCHEMA = "btc-5m-payoff-aware-middle-oi-features-v1"
OI_FEATURES = (
    "binance_oi_change_5m_bps", "binance_oi_change_15m_bps",
    "binance_oi_change_30m_bps", "binance_oi_change_60m_bps",
    "binance_oi_value_change_15m_bps", "binance_oi_value_change_60m_bps",
    "binance_oi_acceleration_5_vs_30_bps", "binance_oi_path_agreement_15m",
    "binance_oi_path_agreement_60m",
)


def _finite(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("runtime parameters must be finite")
    return value


def _logistic(estimator: Any, scaler: Any | None = None) -> dict[str, Any]:
    coefficients = np.asarray(estimator.coef_, dtype=float)
    if coefficients.shape[0] != 1:
        raise ValueError("only binary logistic models are supported")
    payload: dict[str, Any] = {
        "coefficients": [_finite(value) for value in coefficients[0]],
        "intercept": _finite(estimator.intercept_[0]),
    }
    if scaler is not None:
        payload["means"] = [_finite(value) for value in scaler.mean_]
        payload["scales"] = [_finite(value) for value in scaler.scale_]
    return payload


def _histogram(estimator: Any, local_names: Iterable[str], runtime_names: list[str], output: str) -> dict[str, Any]:
    local_names = list(local_names)
    if estimator.n_trees_per_iteration_ != 1:
        raise ValueError("only single-output histogram estimators are supported")
    return {
        "feature_indices": [runtime_names.index(name) for name in local_names],
        "baseline": _finite(estimator._baseline_prediction[0, 0]),
        "output": output,
        "trees": [export_tree(group[0], len(local_names)) for group in estimator._predictors],
    }


def _expert(source: dict[str, Any], runtime_names: list[str]) -> dict[str, Any]:
    names = list(source["feature_names"])
    calibrator = source["calibrator"]
    band = source.get("band", {"start_second": 15, "end_second_exclusive": 241})
    return {
        "start_second": int(band["start_second"]),
        "end_second_exclusive": int(band["end_second_exclusive"]),
        "outcome": _histogram(source["estimator"], names, runtime_names, "probability"),
        "calibrator": _logistic(calibrator),
    }


def _schema_sha256(version: str, names: list[str]) -> str:
    return hashlib.sha256(canonical_json_bytes({"schema_version": version, "feature_names": names})).hexdigest()


def _base_model(model_key: str, version: str, names: list[str], source_sha: str, payoff: dict[str, Any]) -> dict[str, Any]:
    schema_sha = _schema_sha256(version, names)
    return {
        "schema_version": RUNTIME_SCHEMA,
        "model_key": model_key,
        "features": {
            "schema_version": version,
            "schema_sha256": schema_sha,
            "names": names,
            "numeric_type": "float64",
            "non_finite_policy": "native_missing_branch",
            "imputation_medians": [0.0] * len(names),
        },
        "payoff_model": {key: value for key, value in payoff.items() if key != "prediction_policy"},
        "prediction_policy": payoff["prediction_policy"],
        "estimator": {
            "type": "hist_gradient_boosting_classifier", "class_order": [0, 1],
            "output": "raw_logit", "baseline_logit": 0.0,
            "tree_values_include_learning_rate": True, "split_comparison": "less_than_or_equal",
            "trees": [],
        },
        "calibration": {"type": "platt_logit", "slope": 1.0, "intercept": 0.0,
            "input_probability_clip": {"minimum": 1e-9, "maximum": 0.999999999},
            "output_logit_clip": {"minimum": -40.0, "maximum": 40.0}},
        "time_bands": None,
        "target": {"type": "outcome_up"},
        "asymmetric_value_calibration": None,
        "decision": {"probability_up_threshold": 0.5, "confidence_threshold": 0.5,
            "below_confidence_action": "no_trade", "up_action": "up", "down_action": "down"},
        "provenance": {"source_training_model_sha256": source_sha},
        "deployment": {"scope": "paper_only", "production_qualified": False, "live_capital_allowed": False},
    }


def q5_payload(source: dict[str, Any], model_key: str, source_sha: str) -> dict[str, Any]:
    names = list(source["feature_names"])
    for group in ("experts", "family_proxies"):
        for item in source[group].values():
            for name in item["feature_names"]:
                if name not in names:
                    names.append(name)
    if "fee_rate" not in names:
        names.append("fee_rate")
    admission = source["admission_model"]
    thresholds = source["thresholds"]
    payoff = {
        "kind": "q5",
        "prediction_policy": {
            "type": "first_confidence_crossing", "minimum_seconds_after_open": 15,
            "maximum_seconds_after_open": 240, "cadence_seconds": 5,
            "early_end_second": 89, "early_cadence_seconds": 1, "late_start_second": 180,
        },
        "execution_reserve_per_share": 0.005,
        "stress_slippage_per_share": 0.01,
        "price_bucket_edges": list(source["price_time_calibration"]["price_bucket_edges"]),
        "price_bucket_minimum_edges": [0.0, 0.005, 0.015, 0.04],
        "experts": {key: _expert(value, names) for key, value in source["experts"].items()},
        "family_proxies": {key: _expert(value, names) for key, value in source["family_proxies"].items()},
        "price_time_calibration": {
            "band_names": list(source["price_time_calibration"]["band_names"]),
            "model": _logistic(source["price_time_calibration"]["estimator"]),
        },
        "calibration_guard": source["calibration_guard"],
        "admission": {
            "feature_names": list(admission["feature_names"]),
            "classifier": _histogram(admission["profitable_classifier"], admission["feature_names"], list(admission["feature_names"]), "probability"),
            "regressor": _histogram(admission["stress_edge_regressor"], admission["feature_names"], list(admission["feature_names"]), "regression"),
        },
        "thresholds": {
            name: {
                "enabled": bool(value.get("enabled", True)),
                "confidence": _finite(value.get("confidence", 1.0)),
                "edge": _finite(value.get("edge", 0.0)),
                "admission": _finite(value.get("admission", 1.0)),
                "payoff_lower_bound": _finite(value.get("payoff_lower_bound", value.get("payoff_edge", 0.0))),
            }
            for name, value in thresholds.items()
        },
    }
    return _base_model(model_key, Q5_FEATURE_SCHEMA, names, source_sha, payoff)


def middle_payload(candidate: Any, model_key: str, source_sha: str) -> dict[str, Any]:
    names = list(candidate.feature_names)
    for name in candidate.eligibility_features:
        if name not in names:
            names.append(name)
    if "fee_rate" not in names:
        names.append("fee_rate")
    version = MIDDLE_OI_FEATURE_SCHEMA if any(name in OI_FEATURES for name in names) else MIDDLE_FEATURE_SCHEMA
    outcome_names = list(candidate.outcome.feature_names)
    correctness = candidate.correctness
    local = {
        key: {"model": _logistic(value.estimator), "weight": _finite(value.weight), "penalty": _finite(value.penalty)}
        for key, value in correctness.locals.items()
    }
    payoff: dict[str, Any] = {
        "kind": "middle",
        "prediction_policy": {"type": "first_confidence_crossing", "minimum_seconds_after_open": 90,
            "maximum_seconds_after_open": 179, "cadence_seconds": 5},
        "execution_reserve_per_share": 0.005,
        "stress_slippage_per_share": 0.01,
        "price_bucket_edges": [0.0, 0.65, 0.75, 0.85, 1.01],
        "outcome": _histogram(candidate.outcome.estimator, outcome_names, names, "probability"),
        "outcome_calibrator": _logistic(candidate.outcome.calibrator),
        "correctness": {
            "model": _logistic(correctness.estimator, correctness.scaler),
            "regime_feature_indices": [names.index(name) for name in correctness.regime_features],
            "global_penalty": _finite(correctness.global_penalty),
            "locals": local,
        },
        "policy": {
            "confidence": _finite(candidate.submitted_policy["confidence"]),
            "stress_edge": _finite(candidate.submitted_policy["stress_edge"]),
            "loss_severity": None if math.isinf(candidate.submitted_policy["loss_severity"]) else _finite(candidate.submitted_policy["loss_severity"]),
        },
        "eligibility_feature_indices": [names.index(name) for name in candidate.eligibility_features],
    }
    if candidate.probability_modifier is not None:
        payoff["probability_modifier"] = {
            "feature_indices": [names.index(name) for name in candidate.probability_modifier.feature_names],
            "model": _logistic(candidate.probability_modifier.estimator, candidate.probability_modifier.scaler),
        }
    if candidate.loss_model is not None:
        payoff["loss_model"] = {
            "feature_names": list(candidate.loss_feature_names),
            "model": _histogram(candidate.loss_model, candidate.loss_feature_names, list(candidate.loss_feature_names), "regression"),
        }
    return _base_model(model_key, version, names, source_sha, payoff)


def write_bundle(model: dict[str, Any], output_root: Path, source_sha: str, vectors: list[dict[str, Any]] | None = None) -> Path:
    model_bytes = canonical_json_bytes(model)
    golden = {
        "schema_version": GOLDEN_SCHEMA,
        "model_key": model["model_key"],
        "feature_schema_sha256": model["features"]["schema_sha256"],
        "vectors": vectors or [],
    }
    golden_bytes = canonical_json_bytes(golden)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "model_key": model["model_key"],
        "model_file": "model.json",
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "golden_vectors_file": "golden-vectors.json",
        "golden_vectors_sha256": hashlib.sha256(golden_bytes).hexdigest(),
        "feature_schema_version": model["features"]["schema_version"],
        "feature_schema_sha256": model["features"]["schema_sha256"],
        "source_freeze_manifest_sha256": source_sha,
        "source_training_model_sha256": source_sha,
        "deployment_scope": "paper_only",
        "production_qualified": False,
        "live_capital_allowed": False,
    }
    destination = output_root / model["model_key"]
    write_immutable_directory(destination, {
        "model.json": model_bytes,
        "manifest.json": canonical_json_bytes(manifest),
        "golden-vectors.json": golden_bytes,
    })
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q5-model", type=Path, required=True)
    parser.add_argument("--tournament", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path)
    parser.add_argument("--q5-golden-vector", type=Path)
    args = parser.parse_args()
    q5_sha = file_sha256(args.q5_model)
    tournament_sha = file_sha256(args.tournament)
    q5 = joblib.load(args.q5_model)
    tournament = joblib.load(args.tournament)
    q5_vectors = [json.loads(args.q5_golden_vector.read_text())] if args.q5_golden_vector else []
    write_bundle(q5_payload(q5, "btc-5m-payoff-aware-q5-paper-20260820", q5_sha), args.output_root, q5_sha, q5_vectors)
    keys = {
        "chainlink_stratified_payoff": "btc-5m-chainlink-stratified-payoff-paper-20260820",
        "chainlink_full_combined": "btc-5m-chainlink-full-combined-paper-20260820",
        "chainlink_regime_calibrated": "btc-5m-chainlink-regime-calibrated-paper-20260820",
    }
    for candidate, key in keys.items():
        model = middle_payload(tournament["candidates"][candidate], key, tournament_sha)
        vectors: list[dict[str, Any]] = []
        if args.ledger_root is not None:
            ledger = pl.read_parquet(args.ledger_root / f"{candidate}.parquet").head(3)
            for index, row in enumerate(ledger.iter_rows(named=True)):
                probability_up = float(row["probability_up"])
                vectors.append({
                    "id": f"development_selected_{index}",
                    "source": {"market_id": row["market_id"], "observed_at": row["observed_at"].isoformat()},
                    "seconds_elapsed": int(row["seconds_elapsed"]),
                    "feature_values": [None if row.get(name) is None or not math.isfinite(float(row[name])) else float(row[name]) for name in model["features"]["names"]],
                    "expected": {
                        "raw_logit": math.log(probability_up / (1.0 - probability_up)),
                        "probability_up": probability_up,
                        "confidence": float(row["lower_correctness_probability"]),
                        "action": "up" if bool(row["predicted_up"]) else "down",
                    },
                })
        write_bundle(model, args.output_root, tournament_sha, vectors)


if __name__ == "__main__":
    main()
