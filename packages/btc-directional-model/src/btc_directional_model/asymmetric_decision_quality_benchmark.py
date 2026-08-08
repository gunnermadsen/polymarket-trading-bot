"""Seal-first benchmark for early lower-price decision quality."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import polars as pl

from .asymmetric_decision_quality import (
    DECISION_QUALITY_SCHEMA_VERSION,
    FORBIDDEN_SELECTION_COLUMNS,
    MATCHED_CORE_CONTROL_CANDIDATE_ID,
    POST_SELECTION_ATTRIBUTION_PAIRS,
    decision_quality_oof_key_digest,
    fit_decision_quality_walk_forward,
    fit_final_decision_quality_model,
    fit_final_matched_core_model,
    fit_final_selected_attribution_pair,
    fit_selected_attribution_pair_walk_forward,
    fit_selected_matched_core_walk_forward,
    paired_probability_delta,
)
from .asymmetric_incumbent_replay import replay_frozen_asymmetric_incumbent
from .asymmetric_training_readiness import prepare_asymmetric_training_readiness
from .asymmetric_value_config import AsymmetricValueConfig
from .asymmetric_value_data import price_manifest_identity_sha256
from .asymmetric_value_evaluation import (
    IMMEDIATE_FIRST_CROSSING,
    accuracy_price_by_five_second_interval,
    accuracy_price_by_second,
    bootstrap_ledger_metrics,
    evidence_gate_checks,
    frequency_floor_check,
    ledger_metrics,
    policy_gate_checks,
    policy_ledger,
    rejection_funnel,
    score_two_sided_value,
    selected_win_rate_advantage_gate_checks,
    side_time_price_strata_economics,
    temporal_confirmation_ablation,
    vwap10_capacity_policy_ledger,
)
from .asymmetric_value_training import (
    CORE_CANDLES_PRICE,
    CORE_L2_PRICE,
    CORE_ORACLE_L2_PRICE,
    CORE_ORACLE_PRICE,
    asymmetric_probability_frame,
)
from .core_extract import file_sha256, write_json_atomic

DECISION_QUALITY_BENCHMARK_SCHEMA_VERSION = "btc-asymmetric-decision-quality-benchmark-v1"
DECISION_SELECTION_SEAL_SCHEMA_VERSION = "btc-asymmetric-decision-selection-seal-v1"
DEVELOPMENT_ECONOMIC_REVEAL_SCHEMA_VERSION = "btc-asymmetric-development-economic-reveal-v1"
POST_SELECTION_ATTRIBUTION_RUN_SCHEMA_VERSION = (
    "btc-asymmetric-post-selection-attribution-run-v1"
)
MATCHED_CORE_CONTROL_ID = MATCHED_CORE_CONTROL_CANDIDATE_ID
DEVELOPMENT_OOF_GATE_SCOPE = {
    "name": "consumed_oof_development",
    "policy_window_gates": True,
    "minimum_trades": 100,
    "minimum_trade_utc_days": 5,
    "minimum_strict_markets": 1_000,
    "minimum_strict_grid_coverage": 0.40,
    "minimum_candidate_grid_coverage": 0.40,
    "fresh_forward_contract_is_separate": True,
}

_SELECTION_FORBIDDEN_KEYS = frozenset(
    {
        *FORBIDDEN_SELECTION_COLUMNS,
        "realized_net",
        "entry_debit",
        "expectancy",
        "net_expectancy_per_trade",
        "stress_1c_net_expectancy_per_trade",
        "maximum_drawdown",
        "maximum_losing_streak",
        "average_loss",
        "maximum_loss",
        "pnl_rank",
    }
)


def write_decision_selection_seal(
    run_dir: Path,
    payload: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Write an immutable probability-only seal and reload it from disk."""

    _reject_economic_selection_keys(payload)
    sealed = {
        **payload,
        "schema_version": DECISION_SELECTION_SEAL_SCHEMA_VERSION,
        "economics_opened": False,
        "created_at": datetime.now(UTC).isoformat(),
    }
    sealed["selection_identity_sha256"] = _selection_identity(sealed)
    path = run_dir / "decision-selection-seal.json"
    write_json_atomic(path, sealed)
    return path, load_verified_decision_selection_seal(path)


def load_verified_decision_selection_seal(path: Path) -> dict[str, Any]:
    """Verify the disk boundary required before any economic reveal."""

    payload = json.loads(path.read_text())
    if payload.get("schema_version") != DECISION_SELECTION_SEAL_SCHEMA_VERSION:
        raise RuntimeError("decision-selection seal schema changed")
    if payload.get("economics_opened") is not False:
        raise RuntimeError("decision-selection seal opened economics before verification")
    if payload.get("selection_identity_sha256") != _selection_identity(payload):
        raise RuntimeError("decision-selection seal identity mismatch")
    _reject_economic_selection_keys(payload)
    run_dir = path.parent
    for relative_path, expected_sha256 in payload.get("artifact_sha256", {}).items():
        artifact = run_dir / relative_path
        if not artifact.is_file() or file_sha256(artifact) != expected_sha256:
            raise RuntimeError(f"decision-selection sealed artifact changed: {relative_path}")
    selection_artifact = run_dir / str(payload.get("selection_artifact"))
    if not selection_artifact.is_file():
        raise RuntimeError("decision-selection evidence artifact is missing")
    selection_evidence = json.loads(selection_artifact.read_text())
    if selection_evidence.get("selection") != payload.get("selection"):
        raise RuntimeError("decision-selection seal does not match its selection evidence")
    oof_artifact = run_dir / str(payload.get("oof_artifact"))
    if not oof_artifact.is_file():
        raise RuntimeError("decision-selection OOF artifact is missing")
    oof = pl.read_parquet(oof_artifact)
    if (
        file_sha256(oof_artifact) != payload["oof"]["content_sha256"]
        or decision_quality_oof_key_digest(oof) != payload["oof"]["key_sha256"]
    ):
        raise RuntimeError("decision-selection OOF evidence changed")
    if payload["selection"].get("status") == "selected":
        matched_name = payload.get("matched_core_oof_artifact")
        if not isinstance(matched_name, str) or matched_name not in payload["artifact_sha256"]:
            raise RuntimeError("decision-selection matched Core OOF artifact is missing")
        if payload.get("matched_core_probability") is None:
            raise RuntimeError("decision-selection matched Core probability evidence is missing")
    return payload


def _selection_identity(payload: dict[str, Any]) -> str:
    canonical = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "run_id", "selection_identity_sha256"}
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _reject_economic_selection_keys(value: Any, *, path: str = "selection") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _SELECTION_FORBIDDEN_KEYS:
                raise RuntimeError(f"economic field entered decision selection at {path}.{key}")
            _reject_economic_selection_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_economic_selection_keys(child, path=f"{path}[{index}]")


def _decision_contract_evidence(config: AsymmetricValueConfig) -> dict[str, Any]:
    contract = config.decision_quality
    if contract is None:
        raise ValueError("decision-quality benchmark requires its frozen contract")

    def window(value: Any) -> dict[str, str]:
        return {"start": value.start.isoformat(), "end": value.end.isoformat()}

    return {
        "schema_version": DECISION_QUALITY_SCHEMA_VERSION,
        "folds": [
            {
                "name": fold.name,
                "fit": window(fold.fit),
                "calibration": window(fold.calibration),
                "validation": window(fold.validation),
            }
            for fold in contract.folds
        ],
        "final_fit": window(contract.final_fit),
        "final_calibration": window(contract.final_calibration),
        "candidates": [asdict(candidate) for candidate in contract.candidates],
        "calibration_variants": [asdict(variant) for variant in contract.calibration_variants],
        "quality_gates": asdict(contract.gates),
        "price_strata": [list(value) for value in contract.price_strata],
        "time_strata": [list(value) for value in contract.time_strata],
    }


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**32)


def _market_id_digest(frame: pl.DataFrame) -> str:
    if "market_id" not in frame.columns or frame["market_id"].null_count():
        raise ValueError("market cohort requires non-null market_id evidence")
    market_ids = sorted(str(value) for value in frame["market_id"].unique().to_list())
    digest = hashlib.sha256(b"btc-asymmetric-market-cohort-v1\n")
    for market_id in market_ids:
        encoded = market_id.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _join_oof_probability_to_execution(
    probability_frame: pl.DataFrame,
    source_frame: pl.DataFrame,
    *,
    model: str,
    config: AsymmetricValueConfig,
) -> pl.DataFrame:
    contract = config.decision_quality
    if contract is None:
        raise ValueError("OOF execution join requires the decision-quality contract")
    keys = ["market_id", "window_start", "observed_at", "seconds_elapsed"]
    if probability_frame.filter(
        pl.col("window_start").is_between(
            contract.final_calibration.start,
            contract.final_calibration.end,
            closed="left",
        )
    ).height:
        raise RuntimeError("final calibration rows entered the economic reveal")
    if probability_frame.filter(
        ~pl.col("window_start").is_between(
            contract.folds[0].validation.start,
            contract.folds[-1].validation.end,
            closed="left",
        )
    ).height:
        raise RuntimeError("economic reveal contains rows outside the OOF validation union")
    if probability_frame.select(*keys).is_duplicated().any():
        raise RuntimeError("selected OOF probability keys are duplicated")
    source = source_frame.filter(
        pl.col("window_start").is_between(
            contract.folds[0].validation.start,
            contract.folds[-1].validation.end,
            closed="left",
        )
    )
    if source.select(*keys).is_duplicated().any():
        raise RuntimeError("OOF execution source keys are duplicated")
    probability = probability_frame.select(
        *keys,
        "fold",
        pl.col("label_up").alias("_oof_label_up"),
        "probability_yes",
    )
    joined = source.join(
        probability,
        on=keys,
        how="inner",
        validate="1:1",
    ).sort("window_start", "market_id", "seconds_elapsed", "observed_at")
    if joined.height != probability_frame.height:
        raise RuntimeError("OOF execution join did not preserve every selected key")
    if joined.filter(pl.col("label_up") != pl.col("_oof_label_up")).height:
        raise RuntimeError("OOF execution labels changed after selection")
    forbidden_final_calibration = joined.filter(
        pl.col("window_start").is_between(
            contract.final_calibration.start,
            contract.final_calibration.end,
            closed="left",
        )
    )
    if forbidden_final_calibration.height:
        raise RuntimeError("final calibration rows entered the economic reveal")
    predictions = asymmetric_probability_frame(
        joined,
        joined["probability_yes"].to_numpy(),
        model=model,
    )
    return predictions.with_columns(joined["fold"])


def _economic_model_evidence(
    predictions: pl.DataFrame,
    config: AsymmetricValueConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    primary = next(policy for policy in config.policies if policy.selection_eligible)
    scored = score_two_sided_value(predictions)
    ledger = policy_ledger(
        scored,
        primary,
        quantity=config.quantity,
        maximum_depth_participation=config.maximum_depth_participation,
    )
    metrics = ledger_metrics(ledger)
    metrics["utc_day_block_bootstrap"] = bootstrap_ledger_metrics(
        ledger,
        resamples=config.bootstrap_resamples,
        seed=_stable_seed(config.random_seed, str(predictions["model"][0]), "economics"),
    )
    return scored, ledger, metrics


def _matched_probability_checks(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for metric in ("brier_delta", "log_loss_delta"):
        values = evidence[metric]
        checks.extend(
            (
                {
                    "name": f"matched_core_{metric}_noninferiority",
                    "observed": values["upper_95"],
                    "threshold": 0.01,
                    "operator": "<=",
                    "passed": bool(values["upper_95"] <= 0.01),
                },
                {
                    "name": f"matched_core_{metric}_point_improvement",
                    "observed": values["point"],
                    "threshold": 0.0,
                    "operator": "<=",
                    "passed": bool(values["point"] <= 0.0),
                },
            )
        )
    return checks


def _matched_economic_checks(
    candidate_metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    paired_day_net: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_expectancy = candidate_metrics.get("net_expectancy_per_trade")
    control_expectancy = control_metrics.get("net_expectancy_per_trade")
    expectancy_delta = (
        float(candidate_expectancy) - float(control_expectancy)
        if candidate_expectancy is not None and control_expectancy is not None
        else None
    )
    candidate_yield = candidate_metrics.get("net_profit_per_resolved_market")
    control_yield = control_metrics.get("net_profit_per_resolved_market")
    yield_delta = (
        float(candidate_yield) - float(control_yield)
        if candidate_yield is not None and control_yield is not None
        else None
    )
    return [
        {
            "name": "matched_core_point_expectancy_noninferiority",
            "observed": expectancy_delta,
            "threshold": 0.0,
            "operator": ">=",
            "passed": bool(expectancy_delta is not None and expectancy_delta >= 0.0),
        },
        {
            "name": "matched_core_opportunity_yield_noninferiority",
            "observed": yield_delta,
            "threshold": 0.0,
            "operator": ">=",
            "passed": bool(yield_delta is not None and yield_delta >= 0.0),
        },
        {
            "name": "matched_core_paired_day_net_lower_95_noninferiority",
            "observed": paired_day_net["lower_95"],
            "threshold": 0.0,
            "operator": ">=",
            "passed": bool(paired_day_net["lower_95"] >= 0.0),
        },
    ]


def _fit_post_selection_attribution(
    *,
    run_dir: Path,
    development_model_frames: dict[str, pl.DataFrame],
    selection_seal: dict[str, Any],
    config: AsymmetricValueConfig,
    core_config: Any,
) -> Path:
    """Fit every predeclared source attribution pair after selection is sealed."""

    attribution_dir = run_dir / "post-selection-attribution"
    attribution_dir.mkdir()
    artifact_sha256: dict[str, str] = {}
    pairs: dict[str, Any] = {}
    for feature_set_name in sorted(POST_SELECTION_ATTRIBUTION_PAIRS):
        source_frame = development_model_frames[feature_set_name]
        oof_by_arm, walk_forward = fit_selected_attribution_pair_walk_forward(
            source_frame,
            selection_seal,
            config,
            core_config,
            feature_set_name=feature_set_name,
        )
        final_models, final_training = fit_final_selected_attribution_pair(
            source_frame,
            selection_seal,
            config,
            core_config,
            feature_set_name=feature_set_name,
        )
        pair_dir = attribution_dir / feature_set_name
        pair_dir.mkdir()
        oof_artifacts: dict[str, str] = {}
        model_artifacts: dict[str, str] = {}
        for arm_name in sorted(oof_by_arm):
            oof_path = pair_dir / f"{arm_name}-oof.parquet"
            oof_by_arm[arm_name].write_parquet(
                oof_path,
                compression="zstd",
                statistics=True,
            )
            relative = str(oof_path.relative_to(run_dir))
            oof_artifacts[arm_name] = relative
            artifact_sha256[relative] = file_sha256(oof_path)
        for arm_name in sorted(final_models):
            model_path = pair_dir / f"{arm_name}.joblib"
            joblib.dump(final_models[arm_name], model_path, compress=3)
            relative = str(model_path.relative_to(run_dir))
            model_artifacts[arm_name] = relative
            artifact_sha256[relative] = file_sha256(model_path)
        evidence_path = pair_dir / "training-evidence.json"
        write_json_atomic(
            evidence_path,
            {
                "walk_forward": walk_forward,
                "final": final_training,
                "oof_artifacts": oof_artifacts,
                "model_artifacts": model_artifacts,
            },
        )
        evidence_relative = str(evidence_path.relative_to(run_dir))
        artifact_sha256[evidence_relative] = file_sha256(evidence_path)
        pairs[feature_set_name] = {
            "candidate_feature_set": feature_set_name,
            "matched_control_feature_set": POST_SELECTION_ATTRIBUTION_PAIRS[
                feature_set_name
            ][0],
            "evidence_artifact": evidence_relative,
            "oof_artifacts": oof_artifacts,
            "model_artifacts": model_artifacts,
            "probability_comparison": walk_forward[
                "candidate_minus_matched_control_probability"
            ],
            "exact_within_pair_grid": walk_forward["exact_within_pair_grid"],
        }
    payload = {
        "schema_version": POST_SELECTION_ATTRIBUTION_RUN_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "selection_identity_sha256": selection_seal["selection_identity_sha256"],
        "selection_uses_economics": False,
        "economics_opened": False,
        "pairs": pairs,
        "artifact_sha256": artifact_sha256,
    }
    payload["attribution_identity_sha256"] = _post_selection_attribution_identity(payload)
    manifest_path = run_dir / "post-selection-attribution.json"
    write_json_atomic(manifest_path, payload)
    _load_verified_post_selection_attribution(
        manifest_path,
        selection_identity_sha256=selection_seal["selection_identity_sha256"],
    )
    return manifest_path


def _post_selection_attribution_identity(payload: dict[str, Any]) -> str:
    canonical = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "attribution_identity_sha256"}
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _load_verified_post_selection_attribution(
    path: Path,
    *,
    selection_identity_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != POST_SELECTION_ATTRIBUTION_RUN_SCHEMA_VERSION:
        raise RuntimeError("post-selection attribution schema changed")
    if payload.get("selection_identity_sha256") != selection_identity_sha256:
        raise RuntimeError("post-selection attribution winner changed")
    if payload.get("selection_uses_economics") is not False:
        raise RuntimeError("economics entered post-selection probability attribution")
    if payload.get("economics_opened") is not False:
        raise RuntimeError("post-selection attribution opened economics before verification")
    if payload.get("attribution_identity_sha256") != _post_selection_attribution_identity(payload):
        raise RuntimeError("post-selection attribution identity mismatch")
    for relative, expected_sha256 in payload.get("artifact_sha256", {}).items():
        artifact = path.parent / relative
        if not artifact.is_file() or file_sha256(artifact) != expected_sha256:
            raise RuntimeError(f"post-selection attribution artifact changed: {relative}")
    if set(payload.get("pairs", {})) != set(POST_SELECTION_ATTRIBUTION_PAIRS):
        raise RuntimeError("post-selection attribution pair matrix changed")
    return payload


def _post_selection_attribution_economics(
    *,
    manifest_path: Path,
    selection_identity_sha256: str,
    development_model_frames: dict[str, pl.DataFrame],
    config: AsymmetricValueConfig,
) -> dict[str, Any]:
    """Reveal fixed-policy economics for attribution pairs without selecting on them."""

    payload = _load_verified_post_selection_attribution(
        manifest_path,
        selection_identity_sha256=selection_identity_sha256,
    )
    from .asymmetric_value_benchmark import _paired_day_net_difference_bootstrap

    contract = config.decision_quality
    assert contract is not None
    result: dict[str, Any] = {}
    for feature_set_name, pair in sorted(payload["pairs"].items()):
        source_frame = development_model_frames[feature_set_name]
        source_oof = source_frame.filter(
            pl.col("window_start").is_between(
                contract.folds[0].validation.start,
                contract.folds[-1].validation.end,
                closed="left",
            )
        )
        source_resolved_markets = source_oof["market_id"].n_unique()
        if source_resolved_markets <= 0:
            raise RuntimeError(f"{feature_set_name} attribution OOF source cohort is empty")
        metrics_by_arm: dict[str, Any] = {}
        ledgers: dict[str, pl.DataFrame] = {}
        for arm_name, relative in sorted(pair["oof_artifacts"].items()):
            oof = pl.read_parquet(manifest_path.parent / relative)
            predictions = _join_oof_probability_to_execution(
                oof,
                source_frame,
                model=arm_name,
                config=config,
            )
            _, ledger, metrics = _economic_model_evidence(predictions, config)
            target_opportunity_markets = oof["market_id"].n_unique()
            metrics["eligible_resolved_markets"] = source_resolved_markets
            metrics["target_opportunity_markets"] = target_opportunity_markets
            metrics["net_profit_per_resolved_market"] = (
                float(metrics["net_profit"]) / source_resolved_markets
            )
            ledgers[arm_name] = ledger
            metrics_by_arm[arm_name] = metrics
        candidate_name = feature_set_name
        control_name = pair["matched_control_feature_set"]
        paired_day_net = _paired_day_net_difference_bootstrap(
            ledgers[candidate_name],
            ledgers[control_name],
            config,
            seed=_stable_seed(
                config.random_seed,
                feature_set_name,
                "post_selection_attribution_net",
            ),
            window_start=contract.folds[0].validation.start,
            window_end=contract.folds[-1].validation.end,
        )
        result[feature_set_name] = {
            "selection_eligible": False,
            "probability_only_selection": True,
            "candidate": candidate_name,
            "matched_control": control_name,
            "source_oof_resolved_markets": source_resolved_markets,
            "probability_comparison": pair["probability_comparison"],
            "candidate_metrics": metrics_by_arm[candidate_name],
            "matched_control_metrics": metrics_by_arm[control_name],
            "paired_day_net_difference": paired_day_net,
        }
    return result


def reveal_decision_quality_economics(
    *,
    seal_path: Path,
    attribution_manifest_path: Path,
    development_model_frames: dict[str, pl.DataFrame],
    l2_frame: pl.DataFrame,
    oracle_frame: pl.DataFrame,
    oof_core: pl.DataFrame,
    oof_grid_coverage: dict[str, Any],
    config: AsymmetricValueConfig,
) -> dict[str, Any]:
    """Reveal economics for the sealed winner and no other candidate."""

    seal = load_verified_decision_selection_seal(seal_path)
    selection = seal["selection"]
    if selection.get("status") != "selected":
        raise RuntimeError("economic reveal requires a quality-selected configuration")
    selected_id = str(selection["selected_candidate_id"])
    sealed_oof = pl.read_parquet(seal_path.parent / seal["oof_artifact"])
    selected_oof = sealed_oof.filter(pl.col("candidate_id") == selected_id)
    matched_core_oof = pl.read_parquet(
        seal_path.parent / seal["matched_core_oof_artifact"]
    )
    observed_ids = selected_oof["candidate_id"].unique().to_list()
    if observed_ids != [selected_id]:
        raise RuntimeError("economic reveal candidate does not match the sealed winner")
    matched_probability = paired_probability_delta(
        selected_oof,
        matched_core_oof,
        resamples=config.bootstrap_resamples,
        seed=_stable_seed(config.random_seed, selected_id, "matched_core_probability"),
    )
    if matched_probability != seal["matched_core_probability"]:
        raise RuntimeError("sealed matched Core probability evidence changed")
    candidate_predictions = _join_oof_probability_to_execution(
        selected_oof,
        l2_frame,
        model=selected_id,
        config=config,
    )
    control_predictions = _join_oof_probability_to_execution(
        matched_core_oof,
        l2_frame,
        model=MATCHED_CORE_CONTROL_ID,
        config=config,
    )
    candidate_scored, candidate_ledger, candidate_metrics = _economic_model_evidence(
        candidate_predictions,
        config,
    )
    control_scored, control_ledger, control_metrics = _economic_model_evidence(
        control_predictions,
        config,
    )
    resolved_markets = oof_core["market_id"].n_unique()
    for metrics in (candidate_metrics, control_metrics):
        metrics["eligible_resolved_markets"] = resolved_markets
        metrics["net_profit_per_resolved_market"] = float(metrics["net_profit"]) / resolved_markets

    from .asymmetric_value_benchmark import (
        _candidate_grid_summary,
        _paired_day_net_difference_bootstrap,
    )

    contract = config.decision_quality
    assert contract is not None
    oof_source = l2_frame.filter(
        pl.col("window_start").is_between(
            contract.folds[0].validation.start,
            contract.folds[-1].validation.end,
            closed="left",
        )
    )
    candidate_grid = _candidate_grid_summary(oof_source, oof_core, config)
    # Consumed OOF development evidence intentionally uses the frozen 100-trade,
    # five-day, 1,000-market and 40%-grid gates. Fresh forward evidence has the
    # separately sealed 21-day/200-trade/2,000-market/70%-grid contract.
    development_policy_gates = bool(DEVELOPMENT_OOF_GATE_SCOPE["policy_window_gates"])
    checks = policy_gate_checks(
        candidate_metrics,
        config,
        policy_window=development_policy_gates,
    )
    checks.extend(selected_win_rate_advantage_gate_checks(candidate_metrics))
    checks.extend(
        evidence_gate_checks(
            oof_source,
            config,
            policy_window=development_policy_gates,
            source_grid_coverage=float(oof_grid_coverage["retained_coverage"]),
            strict_grid_coverage=float(oof_grid_coverage["strict_coverage"]),
            candidate_grid_coverage=float(candidate_grid["prediction_grid_coverage"]),
        )
    )
    checks.extend(_matched_probability_checks(matched_probability))
    paired_day_net = _paired_day_net_difference_bootstrap(
        candidate_ledger,
        control_ledger,
        config,
        seed=_stable_seed(config.random_seed, selected_id, "matched_core_net"),
        window_start=contract.folds[0].validation.start,
        window_end=contract.folds[-1].validation.end,
    )
    checks.extend(_matched_economic_checks(candidate_metrics, control_metrics, paired_day_net))

    incumbent_frame = oracle_frame.filter(
        pl.col("window_start").is_between(
            contract.folds[0].validation.start,
            contract.folds[-1].validation.end,
            closed="left",
        )
    )
    incumbent = replay_frozen_asymmetric_incumbent(incumbent_frame)
    incumbent_markets = (
        incumbent_frame.filter(pl.col("seconds_elapsed") <= 55).select("market_id").unique()
    )
    if incumbent_markets.height != incumbent.eligible_resolved_markets:
        raise RuntimeError("frozen incumbent common frequency denominator changed")
    common_scored = candidate_scored.join(
        incumbent_markets,
        on="market_id",
        how="inner",
        validate="m:1",
    )
    primary = next(policy for policy in config.policies if policy.selection_eligible)
    common_ledger = policy_ledger(
        common_scored,
        primary,
        quantity=config.quantity,
        maximum_depth_participation=config.maximum_depth_participation,
    )
    frequency_check = frequency_floor_check(
        candidate_trades=common_ledger.height,
        eligible_resolved_markets=incumbent.eligible_resolved_markets,
        incumbent_trades_per_eligible_resolved_market=(
            incumbent.trades_per_eligible_resolved_market
        ),
    )
    checks.append(frequency_check)

    temporal_ledgers, temporal_metrics = temporal_confirmation_ablation(
        candidate_scored,
        primary,
        quantity=config.quantity,
        maximum_depth_participation=config.maximum_depth_participation,
    )
    vwap10_ledger, vwap10_evidence = vwap10_capacity_policy_ledger(
        candidate_ledger,
        primary,
        execution_reserve_per_share=config.execution_reserve_per_share,
        quantity=10.0,
        maximum_depth_participation=config.maximum_depth_participation,
    )
    qualified = all(check["passed"] for check in checks)
    attribution_economics = _post_selection_attribution_economics(
        manifest_path=attribution_manifest_path,
        selection_identity_sha256=seal["selection_identity_sha256"],
        development_model_frames=development_model_frames,
        config=config,
    )
    return {
        "schema_version": DEVELOPMENT_ECONOMIC_REVEAL_SCHEMA_VERSION,
        "selection_seal_sha256": file_sha256(seal_path),
        "selection_identity_sha256": seal["selection_identity_sha256"],
        "selected_candidate_id": selected_id,
        "selected_scored": candidate_scored,
        "selected_ledger": candidate_ledger,
        "matched_core_scored": control_scored,
        "matched_core_ledger": control_ledger,
        "temporal_ledgers": temporal_ledgers,
        "vwap10_ledger": vwap10_ledger,
        "incumbent_ledger": incumbent.selected_trades,
        "common_frequency_ledger": common_ledger,
        "evidence": {
            "status": "economically_qualified" if qualified else "blocked",
            "qualified": qualified,
            "selected_metrics": candidate_metrics,
            "matched_core_metrics": control_metrics,
            "matched_probability": matched_probability,
            "matched_paired_day_net": paired_day_net,
            "evidence_scope": dict(DEVELOPMENT_OOF_GATE_SCOPE),
            "incumbent": {
                "model_key": incumbent.model_key,
                "eligible_resolved_markets": incumbent.eligible_resolved_markets,
                "trades": incumbent.selected_trades.height,
                "trades_per_eligible_resolved_market": (
                    incumbent.trades_per_eligible_resolved_market
                ),
                "metrics": ledger_metrics(incumbent.selected_trades),
                "audit_hashes": incumbent.audit_hashes,
            },
            "common_frequency": frequency_check,
            "common_frequency_market_count": incumbent_markets.height,
            "common_frequency_market_sha256": _market_id_digest(incumbent_markets),
            "checks": checks,
            "candidate_grid": candidate_grid,
            "rejection_funnel": rejection_funnel(
                candidate_scored,
                primary,
                quantity=config.quantity,
                maximum_depth_participation=config.maximum_depth_participation,
                confirmation_rule=IMMEDIATE_FIRST_CROSSING,
            ),
            "temporal_confirmation": temporal_metrics,
            "vwap10_capacity": vwap10_evidence,
            "post_selection_source_attribution": attribution_economics,
            "accuracy_price_by_second": accuracy_price_by_second(candidate_scored),
        },
    }


def run_decision_quality_benchmark(
    *,
    config: AsymmetricValueConfig,
    core_config: Any,
    development_model_frames: dict[str, pl.DataFrame],
    oof_core: pl.DataFrame,
    oof_grid_coverage: dict[str, Any],
    development_coverage: dict[str, Any],
    development_price_manifest: dict[str, Any],
    implementation_sha256: str,
    dependency_versions: dict[str, str],
    current_process: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Run the probability-selected benchmark and reveal only its sealed winner."""

    if config.decision_quality is None:
        raise ValueError("decision-quality runner requires its frozen contract")
    required_frames = {
        CORE_L2_PRICE,
        CORE_ORACLE_PRICE,
        CORE_CANDLES_PRICE,
        CORE_ORACLE_L2_PRICE,
    }
    missing_frames = sorted(required_frames - set(development_model_frames))
    if missing_frames:
        raise ValueError(
            "decision-quality runner is missing model frames: " + ", ".join(missing_frames)
        )
    print(
        "asymmetric decision quality: sealing fail-closed readiness before fitting",
        flush=True,
    )
    readiness_path, readiness_payload = prepare_asymmetric_training_readiness(
        config,
        output_dir=config.feature_cache / "training-readiness",
    )
    if readiness_payload.get("ready") is not True:
        raise RuntimeError("decision-quality source readiness did not pass")

    print(
        "asymmetric decision quality: fitting five probability-only walk-forward folds",
        flush=True,
    )
    l2_frame = development_model_frames[CORE_L2_PRICE]
    oof, training = fit_decision_quality_walk_forward(
        l2_frame,
        config,
        core_config,
    )
    selection = training["selection"]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = config.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    oof_path = run_dir / "decision-quality-oof-predictions.parquet"
    oof.write_parquet(oof_path, compression="zstd", statistics=True)
    selection_path = run_dir / "decision-quality-selection.json"
    write_json_atomic(selection_path, training)
    quality_candidate_path = run_dir / "decision-quality-candidates.csv"
    pl.DataFrame(_quality_candidate_rows(selection)).write_csv(quality_candidate_path)

    common_seal = {
        "run_id": run_id,
        "training_contract": config.training_contract,
        "config_sha256": file_sha256(config.source_path),
        "implementation_sha256": implementation_sha256,
        "dependency_versions": dependency_versions,
        "decision_contract": _decision_contract_evidence(config),
        "readiness": {
            "manifest_sha256": file_sha256(readiness_path),
            "readiness_identity_sha256": readiness_payload["readiness_identity_sha256"],
            "payload_sha256": readiness_payload["payload_sha256"],
            "range_start": readiness_payload["range_start"],
            "range_end": readiness_payload["range_end"],
        },
        "source_lineage": {
            "polymarket_price_manifest_identity_sha256": (
                price_manifest_identity_sha256(development_price_manifest)
            ),
            "proxy_prices_used": False,
            "external_ssd_required": readiness_payload["external_ssd_required"],
        },
        "current_champion_reference": current_process,
        "selection": selection,
        "selection_artifact": selection_path.name,
        "quality_candidate_artifact": quality_candidate_path.name,
        "oof_artifact": oof_path.name,
        "oof": {
            "rows": oof.height,
            "markets": oof["market_id"].n_unique(),
            "key_sha256": decision_quality_oof_key_digest(oof),
            "content_sha256": file_sha256(oof_path),
        },
        "selection_uses_economics": False,
        "forbidden_selection_keys": sorted(_SELECTION_FORBIDDEN_KEYS),
    }
    if selection.get("status") != "selected":
        seal_path, verified_seal = write_decision_selection_seal(
            run_dir,
            {
                **common_seal,
                "artifact_sha256": {
                    oof_path.name: file_sha256(oof_path),
                    selection_path.name: file_sha256(selection_path),
                    quality_candidate_path.name: file_sha256(quality_candidate_path),
                },
                "final_model": None,
                "matched_core_probability": None,
            },
        )
        result = _benchmark_result(
            config=config,
            run_id=run_id,
            selection=selection,
            selection_seal=verified_seal,
            selection_seal_sha256=file_sha256(seal_path),
            development_coverage=development_coverage,
            economic_evidence=None,
            artifact_hashes={
                oof_path.name: file_sha256(oof_path),
                selection_path.name: file_sha256(selection_path),
                quality_candidate_path.name: file_sha256(quality_candidate_path),
                seal_path.name: file_sha256(seal_path),
            },
        )
        report_path = run_dir / "benchmark-report.md"
        report_path.write_text(_decision_quality_markdown_report(result))
        result["artifact_sha256"][report_path.name] = file_sha256(report_path)
        write_json_atomic(run_dir / "benchmark.json", result)
        return run_dir, result

    selected_id = str(selection["selected_candidate_id"])
    selected_oof = oof.filter(pl.col("candidate_id") == selected_id)
    print(
        "asymmetric decision quality: fitting sealed winner and exact-key matched Core control",
        flush=True,
    )
    matched_core_oof, matched_core_training = fit_selected_matched_core_walk_forward(
        l2_frame,
        selected_oof,
        selection,
        config,
        core_config,
    )
    final_model, final_training = fit_final_decision_quality_model(
        l2_frame,
        selection,
        config,
        core_config,
    )
    final_core_model, final_core_training = fit_final_matched_core_model(
        l2_frame,
        selection,
        config,
        core_config,
    )
    matched_probability = paired_probability_delta(
        selected_oof,
        matched_core_oof,
        resamples=config.bootstrap_resamples,
        seed=_stable_seed(config.random_seed, selected_id, "matched_core_probability"),
    )
    matched_core_path = run_dir / "matched-core-oof-predictions.parquet"
    matched_core_oof.write_parquet(
        matched_core_path,
        compression="zstd",
        statistics=True,
    )
    matched_training_path = run_dir / "matched-core-training.json"
    write_json_atomic(
        matched_training_path,
        {
            "walk_forward": matched_core_training,
            "final": final_core_training,
            "probability_comparison": matched_probability,
        },
    )
    final_training_path = run_dir / "selected-final-training.json"
    write_json_atomic(final_training_path, final_training)
    model_dir = run_dir / "models"
    model_dir.mkdir()
    final_model_path = model_dir / "selected-core-l2.joblib"
    final_core_model_path = model_dir / "matched-core.joblib"
    joblib.dump(final_model, final_model_path, compress=3)
    joblib.dump(final_core_model, final_core_model_path, compress=3)
    artifact_sha256 = {
        oof_path.name: file_sha256(oof_path),
        selection_path.name: file_sha256(selection_path),
        quality_candidate_path.name: file_sha256(quality_candidate_path),
        matched_core_path.name: file_sha256(matched_core_path),
        matched_training_path.name: file_sha256(matched_training_path),
        final_training_path.name: file_sha256(final_training_path),
        str(final_model_path.relative_to(run_dir)): file_sha256(final_model_path),
        str(final_core_model_path.relative_to(run_dir)): file_sha256(final_core_model_path),
    }
    seal_path, verified_seal = write_decision_selection_seal(
        run_dir,
        {
            **common_seal,
            "artifact_sha256": artifact_sha256,
            "final_model": {
                "selected_candidate_id": selected_id,
                "selected_model_artifact": str(final_model_path.relative_to(run_dir)),
                "selected_model_sha256": file_sha256(final_model_path),
                "matched_core_model_artifact": str(final_core_model_path.relative_to(run_dir)),
                "matched_core_model_sha256": file_sha256(final_core_model_path),
                "training_evidence_sha256": file_sha256(final_training_path),
            },
            "matched_core_probability": matched_probability,
            "matched_core_oof_artifact": matched_core_path.name,
        },
    )
    print(
        "asymmetric decision quality: replaying sealed source-attribution controls",
        flush=True,
    )
    attribution_manifest_path = _fit_post_selection_attribution(
        run_dir=run_dir,
        development_model_frames=development_model_frames,
        selection_seal=verified_seal,
        config=config,
        core_config=core_config,
    )
    print(
        "asymmetric decision quality: probability seal verified; revealing fixed-policy OOF economics",
        flush=True,
    )
    reveal = reveal_decision_quality_economics(
        seal_path=seal_path,
        attribution_manifest_path=attribution_manifest_path,
        development_model_frames=development_model_frames,
        l2_frame=l2_frame,
        oracle_frame=development_model_frames[CORE_ORACLE_PRICE],
        oof_core=oof_core,
        oof_grid_coverage=oof_grid_coverage,
        config=config,
    )
    selected_scored = reveal.pop("selected_scored")
    selected_ledger = reveal.pop("selected_ledger")
    matched_core_scored = reveal.pop("matched_core_scored")
    matched_core_ledger = reveal.pop("matched_core_ledger")
    temporal_ledgers = reveal.pop("temporal_ledgers")
    vwap10_ledger = reveal.pop("vwap10_ledger")
    incumbent_ledger = reveal.pop("incumbent_ledger")
    common_frequency_ledger = reveal.pop("common_frequency_ledger")
    selected_scored_path = run_dir / "selected-oof-scored-opportunities.parquet"
    selected_ledger_path = run_dir / "selected-oof-policy-ledger.parquet"
    matched_scored_path = run_dir / "matched-core-oof-scored-opportunities.parquet"
    matched_ledger_path = run_dir / "matched-core-oof-policy-ledger.parquet"
    incumbent_ledger_path = run_dir / "frozen-incumbent-oof-policy-ledger.parquet"
    common_frequency_path = run_dir / "selected-common-frequency-ledger.parquet"
    selected_scored.write_parquet(selected_scored_path, compression="zstd")
    selected_ledger.write_parquet(selected_ledger_path, compression="zstd")
    matched_core_scored.write_parquet(matched_scored_path, compression="zstd")
    matched_core_ledger.write_parquet(matched_ledger_path, compression="zstd")
    incumbent_ledger.write_parquet(incumbent_ledger_path, compression="zstd")
    common_frequency_ledger.write_parquet(common_frequency_path, compression="zstd")
    temporal_path = run_dir / "temporal-confirmation-ledger.parquet"
    if temporal_ledgers:
        pl.concat(list(temporal_ledgers.values()), how="vertical_relaxed").write_parquet(
            temporal_path,
            compression="zstd",
        )
    vwap10_path = run_dir / "vwap10-capacity-ledger.parquet"
    vwap10_ledger.write_parquet(
        vwap10_path,
        compression="zstd",
    )
    per_second = reveal["evidence"]["accuracy_price_by_second"]
    per_second_path = run_dir / "accuracy-price-by-observation-second.csv"
    pl.DataFrame(per_second).write_csv(per_second_path)
    five_second = accuracy_price_by_five_second_interval(selected_scored)
    five_second_path = run_dir / "accuracy-price-by-five-second-interval.csv"
    pl.DataFrame(five_second).write_csv(five_second_path)
    strata = side_time_price_strata_economics(
        selected_ledger,
        time_strata=config.decision_quality.time_strata,
        price_strata=config.decision_quality.price_strata,
    )
    strata_path = run_dir / "selected-side-time-price-strata-economics.csv"
    pl.DataFrame(strata).write_csv(strata_path)
    reveal["evidence"]["accuracy_price_by_five_second_interval"] = five_second
    reveal["evidence"]["selected_side_time_price_strata"] = strata
    reveal["economic_evidence_identity_sha256"] = _economic_evidence_identity(reveal)
    reveal_path = run_dir / "development-economic-reveal.json"
    write_json_atomic(reveal_path, reveal)

    attribution_payload = _load_verified_post_selection_attribution(
        attribution_manifest_path,
        selection_identity_sha256=verified_seal["selection_identity_sha256"],
    )
    artifact_hashes = {
        **artifact_sha256,
        **attribution_payload["artifact_sha256"],
        attribution_manifest_path.name: file_sha256(attribution_manifest_path),
        seal_path.name: file_sha256(seal_path),
        selected_scored_path.name: file_sha256(selected_scored_path),
        selected_ledger_path.name: file_sha256(selected_ledger_path),
        matched_scored_path.name: file_sha256(matched_scored_path),
        matched_ledger_path.name: file_sha256(matched_ledger_path),
        incumbent_ledger_path.name: file_sha256(incumbent_ledger_path),
        common_frequency_path.name: file_sha256(common_frequency_path),
        vwap10_path.name: file_sha256(vwap10_path),
        per_second_path.name: file_sha256(per_second_path),
        five_second_path.name: file_sha256(five_second_path),
        strata_path.name: file_sha256(strata_path),
        reveal_path.name: file_sha256(reveal_path),
    }
    if temporal_path.is_file():
        artifact_hashes[temporal_path.name] = file_sha256(temporal_path)
    result = _benchmark_result(
        config=config,
        run_id=run_id,
        selection=selection,
        selection_seal=verified_seal,
        selection_seal_sha256=file_sha256(seal_path),
        development_coverage=development_coverage,
        economic_evidence=reveal["evidence"],
        artifact_hashes=artifact_hashes,
    )
    report_path = run_dir / "benchmark-report.md"
    report_path.write_text(_decision_quality_markdown_report(result))
    result["artifact_sha256"][report_path.name] = file_sha256(report_path)
    write_json_atomic(run_dir / "benchmark.json", result)
    return run_dir, result


def _economic_evidence_identity(payload: dict[str, Any]) -> str:
    canonical = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "created_at",
            "selection_seal_sha256",
            "economic_evidence_identity_sha256",
        }
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _quality_selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    records = selection.get("candidate_records") or []
    summaries = [
        {
            "candidate_id": record["candidate_id"],
            "base_candidate": record["base_candidate"],
            "qualified": bool(record["qualified"]),
            "overall": record["metrics"]["overall"],
            "failed_gates": [gate for gate in record["gates"] if not gate["passed"]],
        }
        for record in records
    ]
    summaries.sort(
        key=lambda record: (
            not record["qualified"],
            record["overall"]["log_loss"],
            record["overall"]["brier"],
            record["candidate_id"],
        )
    )
    selected_id = selection.get("selected_candidate_id")
    selected = next(
        (record for record in summaries if record["candidate_id"] == selected_id),
        None,
    )
    return {
        "selection_uses_economics": False,
        "eligible_configurations": len(summaries),
        "qualified_configurations": sum(record["qualified"] for record in summaries),
        "selected": selected,
        "rank_trace": selection.get("rank_trace") or [],
        "candidate_summaries": summaries,
    }


def _quality_candidate_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rank = {
        record["candidate_id"]: index + 1
        for index, record in enumerate(selection.get("rank_trace") or [])
    }
    rows: list[dict[str, Any]] = []
    for record in selection.get("candidate_records") or []:
        overall = record["metrics"]["overall"]
        failed = [gate["name"] for gate in record["gates"] if not gate["passed"]]
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "base_candidate": record["base_candidate"],
                "target_weight": record["target_weight"],
                "histogram_profile": record["histogram_profile"],
                "parent_source": record["parent_source"],
                "identity_l2": record["identity_l2"],
                "qualified": bool(record["qualified"]),
                "rank": rank.get(record["candidate_id"]),
                "rows": overall["rows"],
                "log_loss": overall["log_loss"],
                "brier": overall["brier"],
                "bias": overall["bias"],
                "ece": overall["ece"],
                "failed_gate_count": len(failed),
                "failed_gates": ";".join(failed),
            }
        )
    return sorted(rows, key=lambda row: row["candidate_id"])


def _benchmark_result(
    *,
    config: AsymmetricValueConfig,
    run_id: str,
    selection: dict[str, Any],
    selection_seal: dict[str, Any],
    selection_seal_sha256: str,
    development_coverage: dict[str, Any],
    economic_evidence: dict[str, Any] | None,
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    selected_id = selection.get("selected_candidate_id")
    economically_qualified = bool(
        economic_evidence is not None and economic_evidence.get("qualified") is True
    )
    if selected_id is None:
        evaluation_status = "blocked_no_quality_configuration"
    elif economically_qualified:
        evaluation_status = "awaiting_fresh_forward_evidence"
    else:
        evaluation_status = "blocked_development_economics"
    return {
        "schema_version": DECISION_QUALITY_BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "training_contract": config.training_contract,
        "selection": {
            "selected_key": selected_id,
            "selected_candidate_id": selected_id,
            "selected_base_candidate": selection.get("selected_base_candidate"),
            "status": selection["status"],
            "economics_used": False,
            "selection_identity_sha256": selection_seal["selection_identity_sha256"],
            "selection_seal_sha256": selection_seal_sha256,
        },
        "decision_quality": _quality_selection_summary(selection),
        "development": {
            "economic_reveal_status": (
                economic_evidence.get("status") if economic_evidence is not None else "not_opened"
            ),
            "economically_qualified": economically_qualified,
            "evidence": economic_evidence,
            "coverage": development_coverage,
        },
        "evaluation": {
            "status": evaluation_status,
            "independent_forward_complete": False,
            "fresh_forward_start_not_before": "2026-08-09T00:00:00+00:00",
        },
        "forward_requirements": {
            "minimum_complete_utc_days": 21,
            "minimum_strict_markets": 2_000,
            "minimum_selected_trades": 200,
            "minimum_trade_utc_days": 10,
            "minimum_yes_trades": 20,
            "minimum_no_trades": 20,
            "minimum_source_grid_coverage": 0.90,
            "minimum_strict_grid_coverage": 0.70,
            "minimum_candidate_grid_coverage": 0.70,
            "pnl_early_stopping_allowed": False,
            "all_development_gates_must_repeat": True,
        },
        "deployment": {
            "authorized": False,
            "runtime_exported": False,
            "paper_process_changed": False,
            "rust_changed": False,
            "trading_pipeline_changed": False,
        },
        "artifact_sha256": artifact_hashes,
    }


def _decision_quality_markdown_report(result: dict[str, Any]) -> str:
    selection = result["selection"]
    quality = result["decision_quality"]
    development = result["development"]
    lines = [
        "# Asymmetric decision-quality benchmark",
        "",
        "The model was selected on walk-forward probability quality before any PnL was opened.",
        "",
        f"- Decision-quality status: `{selection['status']}`",
        f"- Selected configuration: `{selection['selected_candidate_id']}`",
        f"- Qualified probability configurations: {quality['qualified_configurations']} / {quality['eligible_configurations']}",
        f"- Development economic reveal: `{development['economic_reveal_status']}`",
        f"- Fresh forward status: `{result['evaluation']['status']}`",
        "- Runtime, Rust bot, trading processes, and the trading pipeline: unchanged",
    ]
    quality_record = quality.get("selected")
    if quality_record is None and quality["candidate_summaries"]:
        quality_record = quality["candidate_summaries"][0]
    if quality_record is not None:
        overall = quality_record["overall"]
        lines.extend(
            (
                "",
                "## Walk-forward decision quality",
                "",
                f"- Configuration: `{quality_record['candidate_id']}`",
                f"- Market-equal log loss: {overall['log_loss']:.6f}",
                f"- Market-equal Brier score: {overall['brier']:.6f}",
                f"- Calibration bias: {overall['bias']:.2%}",
                f"- ECE: {overall['ece']:.2%}",
                f"- Probability qualification: {str(quality_record['qualified']).lower()}",
            )
        )
        failed = quality_record["failed_gates"]
        if failed:
            lines.append("- Failed probability gates: " + ", ".join(item["name"] for item in failed))
    evidence = development.get("evidence")
    if evidence is not None:
        metrics = evidence["selected_metrics"]
        wins = metrics["trades"] - metrics.get("losing_trades", 0)
        losses = metrics.get("losing_trades", 0)
        bootstrap = metrics.get("utc_day_block_bootstrap") or {}
        expectancy_interval = bootstrap.get("net_expectancy_per_trade") or {}
        failed_checks = [check["name"] for check in evidence["checks"] if not check["passed"]]
        lines.extend(
            (
                "",
                "## Sealed-winner development projection",
                "",
                f"- Trades: {metrics['trades']}",
                f"- Wins / losses: {wins} / {losses} ({(metrics.get('accuracy') or 0.0):.2%} accuracy)",
                f"- YES / NO trades: {metrics.get('yes_trades')} / {metrics.get('no_trades')}",
                f"- Net PnL: ${metrics['net_profit']:.2f}",
                f"- Net expectancy/trade: ${metrics.get('net_expectancy_per_trade') or 0.0:.4f}",
                f"- UTC-day bootstrap lower-95 expectancy/trade: ${expectancy_interval.get('lower_95') or 0.0:.4f}",
                f"- Profit factor: {metrics.get('profit_factor')}",
                f"- Mean raw share: ${metrics.get('mean_share_price') or 0.0:.4f}",
                f"- Mean entry second: {metrics.get('mean_entry_second')}",
                f"- Economically qualified: {str(evidence['qualified']).lower()}",
                f"- Failed economic/evidence gates: {', '.join(failed_checks) if failed_checks else 'none'}",
                "",
                "These are consumed out-of-fold development projections, not independent forward proof.",
            )
        )
        attribution = evidence.get("post_selection_source_attribution") or {}
        if attribution:
            lines.extend(("", "## Source attribution (diagnostic only)", ""))
            for feature_set_name, pair in sorted(attribution.items()):
                candidate = pair["candidate_metrics"]
                control = pair["matched_control_metrics"]
                lines.append(
                    f"- `{feature_set_name}`: EV/trade ${candidate.get('net_expectancy_per_trade') or 0.0:.4f} "
                    f"vs matched control ${control.get('net_expectancy_per_trade') or 0.0:.4f}; "
                    f"net ${candidate.get('net_profit') or 0.0:.2f} vs ${control.get('net_profit') or 0.0:.2f}."
                )
    return "\n".join(lines) + "\n"
