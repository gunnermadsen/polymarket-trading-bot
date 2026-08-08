"""Leakage-safe decision-quality selection for asymmetric low-price entries."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl

from .asymmetric_value_config import (
    AsymmetricValueConfig,
    DecisionQualityCalibrationVariant,
    DecisionQualityCandidate,
    DecisionQualityFold,
)
from .asymmetric_value_training import (
    CORE_L2_PRICE,
    EXPECTED_MODEL_FEATURE_COUNTS,
    AsymmetricValueModel,
    HybridObjectiveCandidate,
    asymmetric_value_feature_sets,
    fit_asymmetric_time_band_calibrators,
    fit_hybrid_histogram_model,
    fit_side_price_time_calibrators,
    hybrid_target_mask,
    target_calibration_evidence,
)
from .core_config import CoreTrainingConfig

DECISION_QUALITY_SCHEMA_VERSION = "btc-asymmetric-decision-quality-v1"
OOF_SELECTION_COLUMNS = (
    "candidate_id",
    "base_candidate",
    "fold",
    "parent_source",
    "identity_l2",
    "market_id",
    "window_start",
    "observed_at",
    "seconds_elapsed",
    "label_up",
    "probability_yes",
    "yes_target_eligible",
    "no_target_eligible",
    "target_time_band",
)
FORBIDDEN_SELECTION_COLUMNS = frozenset(
    {
        "fee_rate",
        "yes_cost_per_share",
        "no_cost_per_share",
        "yes_execution_cost_per_share",
        "no_execution_cost_per_share",
        "pnl",
        "net_profit",
        "profit_factor",
        "capital_efficiency",
        "modeled_edge",
    }
)


def calibration_variant_id(
    base_candidate: str,
    variant: DecisionQualityCalibrationVariant,
) -> str:
    return f"{base_candidate}__{variant.name}"


def control_calibration_variant() -> DecisionQualityCalibrationVariant:
    return DecisionQualityCalibrationVariant(
        parent_source="alltime",
        identity_l2=1.0,
    )


def fit_decision_quality_walk_forward(
    frame: pl.DataFrame,
    config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Fit the frozen matrix and emit probability-only out-of-fold evidence."""

    contract = config.decision_quality
    if contract is None:
        raise ValueError("decision-quality walk-forward requires its frozen config")
    features = asymmetric_value_feature_sets()[CORE_L2_PRICE]
    _validate_primary_feature_contract(features)
    predictions: list[pl.DataFrame] = []
    fold_profiles: dict[str, Any] = {}
    expected_candidate_names = {item.name for item in contract.candidates}
    for fold in contract.folds:
        fit_frame = _window(frame, fold.fit.start, fold.fit.end)
        calibration_frame = _window(
            frame,
            fold.calibration.start,
            fold.calibration.end,
        )
        validation_frame = _window(
            frame,
            fold.validation.start,
            fold.validation.end,
        )
        _validate_fold_frames(
            fold,
            fit_frame,
            calibration_frame,
            validation_frame,
        )
        validation_target = validation_frame.filter(
            pl.Series(hybrid_target_mask(validation_frame, config))
        )
        if validation_target.is_empty():
            raise RuntimeError(f"{fold.name} validation target cohort is empty")
        fold_profiles[fold.name] = {
            "fold": _fold_evidence(fold, fit_frame, calibration_frame, validation_target),
            "candidates": {},
        }
        for candidate_config in contract.candidates:
            candidate = _training_candidate(candidate_config)
            model, fit_evidence = fit_hybrid_histogram_model(
                fit_frame,
                features,
                candidate,
                config,
                core_config,
            )
            variants = (
                contract.calibration_variants
                if candidate.selection_eligible
                else (control_calibration_variant(),)
            )
            base_profiles: dict[str, Any] = {}
            for variant in variants:
                bundle, calibration_profile = _fit_calibrated_bundle(
                    model,
                    calibration_frame,
                    config,
                    core_config,
                    base_candidate=candidate.name,
                    variant=variant,
                )
                candidate_id = calibration_variant_id(candidate.name, variant)
                probability = bundle.probability(validation_target)
                predictions.append(
                    _selection_prediction_frame(
                        validation_target,
                        probability,
                        candidate_id=candidate_id,
                        base_candidate=candidate.name,
                        fold=fold.name,
                        variant=variant,
                        config=config,
                    )
                )
                base_profiles[candidate_id] = calibration_profile
            fold_profiles[fold.name]["candidates"][candidate.name] = {
                "fit": fit_evidence,
                "calibrations": base_profiles,
            }
        if set(fold_profiles[fold.name]["candidates"]) != expected_candidate_names:
            raise RuntimeError(f"{fold.name} candidate matrix is incomplete")
    oof = pl.concat(predictions, how="vertical_relaxed").sort(
        "candidate_id",
        "fold",
        "window_start",
        "market_id",
        "seconds_elapsed",
        "observed_at",
    )
    _validate_oof_selection_frame(oof, config)
    selection = select_decision_quality_candidate(
        oof,
        fold_profiles,
        config,
    )
    return oof, {
        "schema_version": DECISION_QUALITY_SCHEMA_VERSION,
        "selection_uses_economics": False,
        "forbidden_selection_columns": sorted(FORBIDDEN_SELECTION_COLUMNS),
        "oof_rows": oof.height,
        "oof_markets": oof["market_id"].n_unique(),
        "oof_key_sha256": decision_quality_oof_key_digest(oof),
        "fold_profiles": fold_profiles,
        "selection": selection,
    }


def fit_final_decision_quality_model(
    frame: pl.DataFrame,
    selection: dict[str, Any],
    config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
) -> tuple[AsymmetricValueModel, dict[str, Any]]:
    """Refit only the quality-selected configuration on the frozen final windows."""

    contract = config.decision_quality
    if contract is None:
        raise ValueError("final decision-quality fit requires its frozen config")
    selected_id = selection.get("selected_candidate_id")
    if not selected_id or selection.get("status") != "selected":
        raise RuntimeError("no decision-quality configuration qualified for final fit")
    base_name = str(selection["selected_base_candidate"])
    candidate_config = next(
        (item for item in contract.candidates if item.name == base_name),
        None,
    )
    if candidate_config is None or not candidate_config.selection_eligible:
        raise RuntimeError("selected decision-quality base candidate is invalid")
    variant = next(
        (
            item
            for item in contract.calibration_variants
            if calibration_variant_id(base_name, item) == selected_id
        ),
        None,
    )
    if variant is None:
        raise RuntimeError("selected decision-quality calibration variant is invalid")
    fit_frame = _window(frame, contract.final_fit.start, contract.final_fit.end)
    calibration_frame = _window(
        frame,
        contract.final_calibration.start,
        contract.final_calibration.end,
    )
    if fit_frame.is_empty() or calibration_frame.is_empty():
        raise RuntimeError("final decision-quality fit/calibration frames are empty")
    features = asymmetric_value_feature_sets()[CORE_L2_PRICE]
    _validate_primary_feature_contract(features)
    model, fit_evidence = fit_hybrid_histogram_model(
        fit_frame,
        features,
        _training_candidate(candidate_config),
        config,
        core_config,
    )
    bundle, calibration_profile = _fit_calibrated_bundle(
        model,
        calibration_frame,
        config,
        core_config,
        base_candidate=base_name,
        variant=variant,
    )
    if not calibration_profile["target_calibration"]["qualified"]:
        raise RuntimeError("selected final calibration did not qualify")
    bundle.name = CORE_L2_PRICE
    return bundle, {
        "selected_candidate_id": selected_id,
        "selected_base_candidate": base_name,
        "final_fit_window": _window_evidence(contract.final_fit),
        "final_calibration_window": _window_evidence(contract.final_calibration),
        "fit": fit_evidence,
        "calibration": calibration_profile,
    }


def _fit_calibrated_bundle(
    model: Any,
    calibration_frame: pl.DataFrame,
    config: AsymmetricValueConfig,
    core_config: CoreTrainingConfig,
    *,
    base_candidate: str,
    variant: DecisionQualityCalibrationVariant,
) -> tuple[AsymmetricValueModel, dict[str, Any]]:
    calibrators = fit_asymmetric_time_band_calibrators(
        model,
        calibration_frame,
        config,
        core_config=core_config,
        parent_source=variant.parent_source,
    )
    _validate_parent_calibration_support(calibrators, config)
    cells = fit_side_price_time_calibrators(
        model,
        calibrators,
        calibration_frame,
        config,
        identity_l2_strength=variant.identity_l2,
        slope_bounds=(0.05, 3.0),
        intercept_bounds=(-2.0, 2.0),
    )
    target_evidence = target_calibration_evidence(cells, config)
    target_cells = [
        cell
        for cell in cells
        if 1 <= cell.start_second < 60
        and math.isclose(cell.minimum_price, 0.20)
        and math.isclose(cell.maximum_price, 0.30)
    ]
    positive_target_slopes = bool(
        len(target_cells) == 8 and all(cell.slope > 0.0 for cell in target_cells)
    )
    target_evidence["positive_target_slopes"] = positive_target_slopes
    target_evidence["qualified"] = bool(target_evidence["qualified"] and positive_target_slopes)
    bundle = AsymmetricValueModel(
        name=calibration_variant_id(base_candidate, variant),
        model=model,
        time_calibrators=calibrators,
        cells=cells,
        parent_calibration_source=variant.parent_source,
        identity_l2_strength=variant.identity_l2,
    )
    return bundle, {
        "parent_source": variant.parent_source,
        "identity_l2": variant.identity_l2,
        "parent_bands": [
            {
                "start_second": band.start_second,
                "end_second_exclusive": band.end_second_exclusive,
                "rows": band.rows,
                "markets": band.markets,
                **asdict(band.calibrator),
            }
            for band in calibrators
        ],
        "target_calibration": target_evidence,
    }


def _selection_prediction_frame(
    frame: pl.DataFrame,
    probability: np.ndarray,
    *,
    candidate_id: str,
    base_candidate: str,
    fold: str,
    variant: DecisionQualityCalibrationVariant,
    config: AsymmetricValueConfig,
) -> pl.DataFrame:
    policy = next(item for item in config.policies if item.selection_eligible)
    yes_target = frame["yes_ask_vwap_5"].is_between(
        policy.minimum_share_price,
        policy.maximum_share_price,
        closed="left",
    )
    no_target = frame["no_ask_vwap_5"].is_between(
        policy.minimum_share_price,
        policy.maximum_share_price,
        closed="left",
    )
    elapsed = frame["seconds_elapsed"].to_numpy()
    time_band = np.select(
        (
            (elapsed >= 1) & (elapsed < 15),
            (elapsed >= 15) & (elapsed < 30),
            (elapsed >= 30) & (elapsed < 45),
            (elapsed >= 45) & (elapsed < 56),
        ),
        ("1_15", "15_30", "30_45", "45_56"),
        default="outside",
    )
    result = frame.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
    ).with_columns(
        pl.lit(candidate_id).alias("candidate_id"),
        pl.lit(base_candidate).alias("base_candidate"),
        pl.lit(fold).alias("fold"),
        pl.lit(variant.parent_source).alias("parent_source"),
        pl.lit(variant.identity_l2).alias("identity_l2"),
        pl.Series("probability_yes", probability),
        yes_target.alias("yes_target_eligible"),
        no_target.alias("no_target_eligible"),
        pl.Series("target_time_band", time_band),
    )
    return result.select(*OOF_SELECTION_COLUMNS)


def select_decision_quality_candidate(
    oof: pl.DataFrame,
    fold_profiles: dict[str, Any],
    config: AsymmetricValueConfig,
) -> dict[str, Any]:
    """Select one candidate using proper scores and calibration only."""

    contract = config.decision_quality
    if contract is None:
        raise ValueError("decision-quality selection requires its frozen config")
    broad_id = calibration_variant_id(
        "broad_current",
        control_calibration_variant(),
    )
    target_id = calibration_variant_id(
        "target_only_current",
        control_calibration_variant(),
    )
    references = {"broad_current": broad_id, "target_only_current": target_id}
    metrics_by_candidate: dict[str, Any] = {}
    for candidate_id in sorted(oof["candidate_id"].unique().to_list()):
        candidate_frame = oof.filter(pl.col("candidate_id") == candidate_id)
        metrics_by_candidate[candidate_id] = decision_quality_metrics(candidate_frame)
    records: list[dict[str, Any]] = []
    for base in contract.candidates:
        if not base.selection_eligible:
            continue
        for variant in contract.calibration_variants:
            candidate_id = calibration_variant_id(base.name, variant)
            metrics = metrics_by_candidate[candidate_id]
            comparisons = {
                name: paired_probability_delta(
                    oof.filter(pl.col("candidate_id") == candidate_id),
                    oof.filter(pl.col("candidate_id") == reference_id),
                    resamples=config.bootstrap_resamples,
                    seed=_stable_seed(config.random_seed, candidate_id, reference_id),
                )
                for name, reference_id in references.items()
            }
            calibration_ok = all(
                fold_profiles[fold.name]["candidates"][base.name]["calibrations"][candidate_id][
                    "target_calibration"
                ]["qualified"]
                for fold in contract.folds
            )
            gates = _decision_quality_gate_evidence(
                metrics,
                comparisons,
                candidate_id=candidate_id,
                oof=oof,
                references=references,
                contract=contract,
                calibration_ok=calibration_ok,
            )
            records.append(
                {
                    "candidate_id": candidate_id,
                    "base_candidate": base.name,
                    "target_weight": base.target_weight,
                    "histogram_profile": base.histogram_profile,
                    "parent_source": variant.parent_source,
                    "identity_l2": variant.identity_l2,
                    "metrics": metrics,
                    "comparisons": comparisons,
                    "gates": gates,
                    "qualified": all(item["passed"] for item in gates),
                }
            )
    qualified = [record for record in records if record["qualified"]]
    rank_trace = _rank_decision_quality_candidates(
        qualified,
        oof,
        resamples=config.bootstrap_resamples,
        seed=config.random_seed,
    )
    if not rank_trace:
        return {
            "status": "no_qualified_configuration",
            "selected_candidate_id": None,
            "selected_base_candidate": None,
            "candidate_records": records,
            "rank_trace": [],
            "economics_used": False,
        }
    winner_id = rank_trace[0]["candidate_id"]
    winner = next(record for record in records if record["candidate_id"] == winner_id)
    return {
        "status": "selected",
        "selected_candidate_id": winner_id,
        "selected_base_candidate": winner["base_candidate"],
        "candidate_records": records,
        "rank_trace": rank_trace,
        "economics_used": False,
    }


def decision_quality_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    _validate_probability_frame(frame)
    probability = frame["probability_yes"].to_numpy()
    labels = frame["label_up"].to_numpy().astype(np.float64)
    market_ids = frame["market_id"].cast(pl.String).to_numpy()
    weights = _market_equal_weights(market_ids)
    overall = _weighted_probability_metrics(probability, labels, weights)
    result: dict[str, Any] = {"overall": overall, "sides": {}, "time_cells": {}}
    for side, eligibility_column in (
        ("YES", "yes_target_eligible"),
        ("NO", "no_target_eligible"),
    ):
        selected = frame[eligibility_column].to_numpy()
        side_probability = probability[selected] if side == "YES" else 1.0 - probability[selected]
        side_labels = labels[selected] if side == "YES" else 1.0 - labels[selected]
        result["sides"][side] = _weighted_probability_metrics(
            side_probability,
            side_labels,
            _market_equal_weights(market_ids[selected]),
        )
        for time_band in ("1_15", "15_30", "30_45", "45_56"):
            cell_selected = selected & (frame["target_time_band"].to_numpy() == time_band)
            cell_probability = (
                probability[cell_selected] if side == "YES" else 1.0 - probability[cell_selected]
            )
            cell_labels = labels[cell_selected] if side == "YES" else 1.0 - labels[cell_selected]
            result["time_cells"][f"{side}_{time_band}"] = _weighted_probability_metrics(
                cell_probability,
                cell_labels,
                _market_equal_weights(market_ids[cell_selected]),
            )
    return result


def paired_probability_delta(
    candidate: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    keys = ["fold", "market_id", "window_start", "observed_at", "seconds_elapsed"]
    candidate_keys = candidate.select(*keys).sort(*keys)
    reference_keys = reference.select(*keys).sort(*keys)
    if not candidate_keys.equals(reference_keys):
        raise RuntimeError("decision-quality comparison keys do not match")
    joined = (
        candidate.select(
            *keys,
            pl.col("label_up").alias("candidate_label_up"),
            pl.col("probability_yes").alias("candidate_probability"),
        )
        .join(
            reference.select(
                *keys,
                pl.col("label_up").alias("reference_label_up"),
                pl.col("probability_yes").alias("reference_probability"),
            ),
            on=keys,
            how="inner",
            validate="1:1",
        )
        .sort(*keys)
    )
    if not joined["candidate_label_up"].equals(joined["reference_label_up"]):
        raise RuntimeError("decision-quality comparison labels do not match")
    labels = joined["candidate_label_up"].to_numpy().astype(np.float64)
    candidate_probability = np.clip(
        joined["candidate_probability"].to_numpy(),
        1e-9,
        1.0 - 1e-9,
    )
    reference_probability = np.clip(
        joined["reference_probability"].to_numpy(),
        1e-9,
        1.0 - 1e-9,
    )
    row_deltas = {
        "brier_delta": (candidate_probability - labels) ** 2
        - (reference_probability - labels) ** 2,
        "log_loss_delta": _row_log_loss(labels, candidate_probability)
        - _row_log_loss(labels, reference_probability),
    }
    market_groups: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for index, (fold, market_id, window_start) in enumerate(
        zip(
            joined["fold"].to_list(),
            joined["market_id"].cast(pl.String).to_list(),
            joined["window_start"].to_list(),
            strict=True,
        )
    ):
        key = (window_start.date().isoformat(), str(fold), str(market_id))
        group = market_groups.setdefault(
            key,
            {"brier_delta": [], "log_loss_delta": []},
        )
        for metric, values in row_deltas.items():
            group[metric].append(float(values[index]))
    market_values = {
        metric: {
            key: math.fsum(group[metric]) / len(group[metric])
            for key, group in market_groups.items()
        }
        for metric in row_deltas
    }
    day_keys = sorted({key[0] for key in market_groups})
    daily_markets = np.asarray(
        [sum(key[0] == day for key in market_groups) for day in day_keys],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(day_keys), size=(resamples, len(day_keys)))
    result: dict[str, Any] = {"utc_days": len(day_keys), "rows": joined.height}
    for metric in ("brier_delta", "log_loss_delta"):
        values = market_values[metric]
        daily_sums = np.asarray(
            [math.fsum(value for key, value in values.items() if key[0] == day) for day in day_keys]
        )
        samples = daily_sums[indices].sum(axis=1) / daily_markets[indices].sum(axis=1)
        result[metric] = {
            "point": float(math.fsum(values.values()) / len(values)),
            "lower_95": float(np.quantile(samples, 0.025)),
            "upper_95": float(np.quantile(samples, 0.975)),
            "standard_error": float(samples.std(ddof=1)),
        }
    return result


def decision_quality_oof_key_digest(frame: pl.DataFrame) -> str:
    columns = [
        "candidate_id",
        "fold",
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
    ]
    digest = hashlib.sha256(b"btc-asymmetric-decision-quality-oof-key-v1\n")
    for row in frame.select(*columns).sort(*columns).iter_rows():
        for value in row:
            rendered = value.isoformat() if hasattr(value, "isoformat") else str(value)
            encoded = rendered.encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _decision_quality_gate_evidence(
    metrics: dict[str, Any],
    comparisons: dict[str, Any],
    *,
    candidate_id: str,
    oof: pl.DataFrame,
    references: dict[str, str],
    contract: Any,
    calibration_ok: bool,
) -> list[dict[str, Any]]:
    gates = contract.gates
    overall = metrics["overall"]
    evidence = [
        _check("calibration_support", calibration_ok, True, "=="),
        _check("overall_bias", abs(overall["bias"]), gates.maximum_overall_bias, "<="),
        _check("target_ece", overall["ece"], gates.maximum_ece, "<="),
    ]
    for side in ("YES", "NO"):
        evidence.append(
            _check(
                f"{side.lower()}_bias",
                abs(metrics["sides"][side]["bias"]),
                gates.maximum_side_bias,
                "<=",
            )
        )
    for name, cell in metrics["time_cells"].items():
        evidence.append(
            _check(
                f"cell_bias_{name.lower()}",
                abs(cell["bias"]),
                gates.maximum_cell_bias,
                "<=",
            )
        )
    for reference_name, comparison in comparisons.items():
        for metric in ("brier_delta", "log_loss_delta"):
            evidence.append(
                _check(
                    f"{metric}_noninferior_to_{reference_name}",
                    comparison[metric]["upper_95"],
                    gates.maximum_proper_score_noninferiority,
                    "<=",
                )
            )
    target_comparison = comparisons["target_only_current"]
    evidence.extend(
        (
            _check(
                "brier_point_improves_target_only",
                target_comparison["brier_delta"]["point"],
                0.0,
                "<",
            ),
            _check(
                "log_loss_point_improves_target_only",
                target_comparison["log_loss_delta"]["point"],
                0.0,
                "<",
            ),
            _check(
                "one_proper_score_significantly_improves_target_only",
                min(
                    target_comparison["brier_delta"]["upper_95"],
                    target_comparison["log_loss_delta"]["upper_95"],
                ),
                0.0,
                "<",
            ),
        )
    )
    candidate_frame = oof.filter(pl.col("candidate_id") == candidate_id)
    folds = sorted(candidate_frame["fold"].unique().to_list())
    candidate_fold_metrics = {
        fold: decision_quality_metrics(candidate_frame.filter(pl.col("fold") == fold))["overall"]
        for fold in folds
    }
    reference_fold_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for reference_name, reference_id in references.items():
        reference_frame = oof.filter(pl.col("candidate_id") == reference_id)
        reference_fold_metrics[reference_name] = {
            fold: decision_quality_metrics(reference_frame.filter(pl.col("fold") == fold))[
                "overall"
            ]
            for fold in folds
        }
        noninferior_folds = 0
        for fold in folds:
            candidate_fold = candidate_fold_metrics[fold]
            reference_fold = reference_fold_metrics[reference_name][fold]
            jointly_worse = (
                candidate_fold["brier"] > reference_fold["brier"]
                and candidate_fold["log_loss"] > reference_fold["log_loss"]
            )
            noninferior_folds += int(not jointly_worse)
        evidence.append(
            _check(
                f"noninferior_folds_to_{reference_name}",
                noninferior_folds,
                gates.minimum_noninferior_folds,
                ">=",
            )
        )
    jointly_noninferior_folds = sum(
        all(
            not (
                candidate_fold_metrics[fold]["brier"]
                > reference_fold_metrics[reference_name][fold]["brier"]
                and candidate_fold_metrics[fold]["log_loss"]
                > reference_fold_metrics[reference_name][fold]["log_loss"]
            )
            for reference_name in references
        )
        for fold in folds
    )
    evidence.append(
        _check(
            "noninferior_folds_to_all_controls",
            jointly_noninferior_folds,
            gates.minimum_noninferior_folds,
            ">=",
        )
    )
    return evidence


def _rank_decision_quality_candidates(
    records: list[dict[str, Any]],
    oof: pl.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not records:
        return []
    by_log_loss = sorted(
        records,
        key=lambda item: (item["metrics"]["overall"]["log_loss"], item["candidate_id"]),
    )
    minimum = by_log_loss[0]
    minimum_frame_se = _market_equal_log_loss_standard_error(
        oof.filter(pl.col("candidate_id") == minimum["candidate_id"]),
        resamples=resamples,
        seed=_stable_seed(seed, minimum["candidate_id"], "one_standard_error"),
    )
    threshold = minimum["metrics"]["overall"]["log_loss"] + minimum_frame_se
    one_se = [record for record in records if record["metrics"]["overall"]["log_loss"] <= threshold]
    regularization_rank = {"h1_regularized": 1, "h2_regularized": 2, "h3_regularized": 3}
    ranked = sorted(
        one_se,
        key=lambda item: (
            item["metrics"]["overall"]["brier"],
            abs(item["metrics"]["overall"]["bias"]),
            -regularization_rank[item["histogram_profile"]],
            -item["identity_l2"],
            item["target_weight"],
            item["candidate_id"],
        ),
    )
    return [
        {
            "rank": index + 1,
            "candidate_id": item["candidate_id"],
            "log_loss": item["metrics"]["overall"]["log_loss"],
            "brier": item["metrics"]["overall"]["brier"],
            "absolute_bias": abs(item["metrics"]["overall"]["bias"]),
            "one_standard_error_threshold": threshold,
        }
        for index, item in enumerate(ranked)
    ]


def _market_equal_log_loss_standard_error(
    frame: pl.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> float:
    _validate_probability_frame(frame)
    probability = np.clip(
        frame["probability_yes"].to_numpy(),
        1e-9,
        1.0 - 1e-9,
    )
    labels = frame["label_up"].to_numpy().astype(np.float64)
    ordered = frame.with_columns(
        pl.Series("_row_log_loss", _row_log_loss(labels, probability))
    ).sort("fold", "window_start", "market_id", "seconds_elapsed", "observed_at")
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in ordered.select("fold", "market_id", "window_start", "_row_log_loss").iter_rows():
        fold, market_id, window_start, row_loss = row
        key = (window_start.date().isoformat(), str(fold), str(market_id))
        groups.setdefault(key, []).append(float(row_loss))
    market_values = {key: math.fsum(values) / len(values) for key, values in groups.items()}
    day_keys = sorted({key[0] for key in market_values})
    if len(day_keys) < 2:
        raise RuntimeError("decision-quality one-SE ranking requires two UTC days")
    daily_markets = np.asarray(
        [sum(key[0] == day for key in market_values) for day in day_keys],
        dtype=np.int64,
    )
    daily_sums = np.asarray(
        [
            math.fsum(value for key, value in market_values.items() if key[0] == day)
            for day in day_keys
        ]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(day_keys), size=(resamples, len(day_keys)))
    samples = daily_sums[indices].sum(axis=1) / daily_markets[indices].sum(axis=1)
    return float(samples.std(ddof=1))


def _weighted_probability_metrics(
    probability: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    if probability.size == 0 or labels.size != probability.size:
        raise RuntimeError("probability metric cohort is empty or misaligned")
    clipped = np.clip(probability.astype(np.float64), 1e-9, 1.0 - 1e-9)
    labels = labels.astype(np.float64)
    weights = weights / weights.sum()
    brier = float(np.sum(weights * (clipped - labels) ** 2))
    log_loss = float(np.sum(weights * _row_log_loss(labels, clipped)))
    bias = float(np.sum(weights * (clipped - labels)))
    bin_index = np.minimum((clipped * 10).astype(np.int16), 9)
    ece = 0.0
    for index in range(10):
        selected = bin_index == index
        if not selected.any():
            continue
        bin_weight = float(weights[selected].sum())
        bin_probability = float(np.sum(weights[selected] * clipped[selected]) / bin_weight)
        bin_rate = float(np.sum(weights[selected] * labels[selected]) / bin_weight)
        ece += bin_weight * abs(bin_probability - bin_rate)
    return {
        "rows": probability.size,
        "brier": brier,
        "log_loss": log_loss,
        "bias": bias,
        "ece": ece,
        "mean_probability": float(np.sum(weights * clipped)),
        "actual_rate": float(np.sum(weights * labels)),
    }


def _market_equal_weights(market_ids: np.ndarray) -> np.ndarray:
    if market_ids.size == 0:
        raise RuntimeError("market-equal probability cohort is empty")
    _, inverse, counts = np.unique(market_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights / weights.sum()


def _row_log_loss(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    return -(labels * np.log(probability) + (1.0 - labels) * np.log(1.0 - probability))


def _training_candidate(candidate: DecisionQualityCandidate) -> HybridObjectiveCandidate:
    return HybridObjectiveCandidate(
        name=candidate.name,
        target_weight=candidate.target_weight,
        histogram_profile=candidate.histogram_profile,
        selection_eligible=candidate.selection_eligible,
    )


def _validate_primary_feature_contract(features: tuple[str, ...]) -> None:
    expected = EXPECTED_MODEL_FEATURE_COUNTS[CORE_L2_PRICE]
    if len(features) != expected or len(set(features)) != expected:
        raise RuntimeError(
            f"decision-quality primary model requires the exact {expected}-feature Core+L2 contract"
        )


def _validate_parent_calibration_support(
    calibrators: tuple[Any, ...],
    config: AsymmetricValueConfig,
) -> None:
    minimum = config.gates.minimum_calibration_markets_per_band
    unsupported = [
        f"{item.start_second}-{item.end_second_exclusive}:{item.markets}"
        for item in calibrators
        if item.markets < minimum
    ]
    if unsupported:
        raise RuntimeError(
            "parent calibration lacks the required market support: " + ", ".join(unsupported)
        )


def _window(frame: pl.DataFrame, start: Any, end: Any) -> pl.DataFrame:
    return frame.filter(pl.col("window_start").is_between(start, end, closed="left"))


def _validate_fold_frames(
    fold: DecisionQualityFold,
    fit_frame: pl.DataFrame,
    calibration_frame: pl.DataFrame,
    validation_frame: pl.DataFrame,
) -> None:
    if fit_frame.is_empty() or calibration_frame.is_empty() or validation_frame.is_empty():
        raise RuntimeError(f"{fold.name} contains an empty chronological cohort")
    if not (
        fit_frame["window_start"].max() < fold.fit.end
        and calibration_frame["window_start"].min() >= fold.calibration.start
        and validation_frame["window_start"].min() >= fold.validation.start
    ):
        raise RuntimeError(f"{fold.name} chronology leaked")


def _fold_evidence(
    fold: DecisionQualityFold,
    fit_frame: pl.DataFrame,
    calibration_frame: pl.DataFrame,
    validation_target: pl.DataFrame,
) -> dict[str, Any]:
    return {
        "name": fold.name,
        "fit": _window_evidence(fold.fit),
        "calibration": _window_evidence(fold.calibration),
        "validation": _window_evidence(fold.validation),
        "fit_rows": fit_frame.height,
        "fit_markets": fit_frame["market_id"].n_unique(),
        "calibration_rows": calibration_frame.height,
        "calibration_markets": calibration_frame["market_id"].n_unique(),
        "validation_target_rows": validation_target.height,
        "validation_target_markets": validation_target["market_id"].n_unique(),
    }


def _window_evidence(window: Any) -> dict[str, str]:
    return {
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
    }


def _validate_oof_selection_frame(frame: pl.DataFrame, config: AsymmetricValueConfig) -> None:
    if tuple(frame.columns) != OOF_SELECTION_COLUMNS:
        raise RuntimeError("decision-quality OOF selection schema changed")
    forbidden = sorted(FORBIDDEN_SELECTION_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise RuntimeError("economic fields entered decision selection: " + ", ".join(forbidden))
    contract = config.decision_quality
    if contract is None:
        raise ValueError("decision-quality OOF validation requires its config")
    expected_folds = {fold.name for fold in contract.folds}
    if set(frame["fold"].unique().to_list()) != expected_folds:
        raise RuntimeError("decision-quality OOF folds are incomplete")
    expected_candidates: dict[str, tuple[str, str, float]] = {}
    for base in contract.candidates:
        variants = (
            contract.calibration_variants
            if base.selection_eligible
            else (control_calibration_variant(),)
        )
        for variant in variants:
            candidate_id = calibration_variant_id(base.name, variant)
            expected_candidates[candidate_id] = (
                base.name,
                variant.parent_source,
                variant.identity_l2,
            )
    if set(frame["candidate_id"].unique().to_list()) != set(expected_candidates):
        raise RuntimeError("decision-quality OOF candidate matrix is incomplete")
    candidate_keys = [
        "candidate_id",
        "fold",
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
    ]
    if frame.select(*candidate_keys).is_duplicated().any():
        raise RuntimeError("decision-quality OOF keys are duplicated")
    for candidate_id, metadata in expected_candidates.items():
        candidate = frame.filter(pl.col("candidate_id") == candidate_id)
        observed = (
            candidate["base_candidate"].unique().to_list(),
            candidate["parent_source"].unique().to_list(),
            candidate["identity_l2"].unique().to_list(),
        )
        if observed != ([metadata[0]], [metadata[1]], [metadata[2]]):
            raise RuntimeError(f"decision-quality OOF metadata changed for {candidate_id}")
    comparison_keys = [
        "fold",
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
    ]
    invalid_grid = (
        frame.group_by(*comparison_keys)
        .agg(
            pl.len().alias("candidate_count"),
            pl.col("label_up").n_unique().alias("label_count"),
            pl.col("yes_target_eligible").n_unique().alias("yes_eligibility_count"),
            pl.col("no_target_eligible").n_unique().alias("no_eligibility_count"),
            pl.col("target_time_band").n_unique().alias("time_band_count"),
        )
        .filter(
            (pl.col("candidate_count") != len(expected_candidates))
            | (pl.col("label_count") != 1)
            | (pl.col("yes_eligibility_count") != 1)
            | (pl.col("no_eligibility_count") != 1)
            | (pl.col("time_band_count") != 1)
        )
    )
    if invalid_grid.height:
        raise RuntimeError("decision-quality OOF candidates do not share an exact invariant grid")


def _validate_probability_frame(frame: pl.DataFrame) -> None:
    required = {
        "market_id",
        "label_up",
        "probability_yes",
        "yes_target_eligible",
        "no_target_eligible",
        "target_time_band",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("probability quality frame is missing: " + ", ".join(missing))
    probability = frame["probability_yes"].to_numpy()
    if not np.isfinite(probability).all() or np.any((probability <= 0.0) | (probability >= 1.0)):
        raise RuntimeError("probability quality frame contains invalid probabilities")


def _check(name: str, observed: Any, threshold: Any, operator: str) -> dict[str, Any]:
    if operator == "<=":
        passed = observed <= threshold
    elif operator == "<":
        passed = observed < threshold
    elif operator == ">=":
        passed = observed >= threshold
    elif operator == "==":
        passed = observed == threshold
    else:
        raise ValueError(f"unsupported decision-quality gate operator: {operator}")
    return {
        "name": name,
        "observed": observed,
        "threshold": threshold,
        "operator": operator,
        "passed": bool(passed),
    }


def _stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**32)
