from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.asymmetric_book_admission import (
    BOOK_ADMISSION_SCHEMA_VERSION,
    DAY_MARKET_ROW_EQUAL,
    DYNAMIC_FEATURES,
    EXCLUDED_DAYS,
    MARKET_EQUAL,
    OOF_DAYS,
    PROBABILITY_COLUMN,
    SELECTED_SIDE_COLUMN,
    STATIC_FEATURES,
    attach_frozen_policy_selected_side,
    book_admission_probability_sha256,
    book_admission_support,
    book_admission_weights,
    fit_book_admission_candidate,
    load_book_admission_config,
    load_book_admission_model,
    score_book_admission_model,
    select_target_opportunity_rows,
    validate_book_admission_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PACKAGE_ROOT
    / "configs/btc-5m-asymmetric-core-oracle-book-admission-20260414-20260802.toml"
)


def test_config_pins_incumbent_chronology_candidates_and_deployment_boundary() -> None:
    config = load_book_admission_config(CONFIG_PATH)

    assert config.incumbent.process_id == "81f82de7-002b-4ac7-814b-236c6742d81c"
    assert config.incumbent.model_key == "btc-5m-asymmetric-core-oracle-paper-20260805-v1"
    assert config.incumbent.feature_count == 75
    assert config.runtime_deployable is False
    assert config.batch_forward_eligible is True
    assert config.process_change_allowed is False
    assert config.excluded_utc_days == EXCLUDED_DAYS
    assert config.oof_utc_days == OOF_DAYS
    assert len(config.folds) == 10
    for day, fold in zip(OOF_DAYS, config.folds, strict=True):
        validation_start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
        assert fold.validation.start == validation_start
        assert fold.validation.end == validation_start + timedelta(days=1)
        assert fold.calibration.start == validation_start - timedelta(days=14)
        assert fold.fit.end == fold.calibration.start
    assert config.final_fit.start == datetime(2026, 4, 14, tzinfo=UTC)
    assert config.final_fit.end == datetime(2026, 7, 16, tzinfo=UTC)
    assert config.final_calibration.start == datetime(2026, 7, 16, tzinfo=UTC)
    assert config.final_calibration.end == datetime(2026, 8, 2, tzinfo=UTC)

    candidates = config.model.candidates
    assert [candidate.name for candidate in candidates] == ["S0", "D1", "D2", "D3", "D4"]
    assert candidates[0].selection_eligible is False
    assert candidates[1].l2_regularization == 30.0
    assert candidates[2].l2_regularization == 10.0
    assert candidates[3].weighting == DAY_MARKET_ROW_EQUAL
    assert candidates[4].estimator == "histogram_gradient_boosting"
    assert (
        candidates[4].learning_rate,
        candidates[4].max_iter,
        candidates[4].max_leaf_nodes,
        candidates[4].min_samples_leaf,
        candidates[4].l2_regularization,
    ) == (0.03, 120, 5, 250, 20.0)
    assert len(STATIC_FEATURES) == 13
    assert len(DYNAMIC_FEATURES) == 40

    with pytest.raises(ValueError, match="deployment boundary"):
        validate_book_admission_config(replace(config, runtime_deployable=True))
    with pytest.raises(ValueError, match="economic gates weakened"):
        validate_book_admission_config(
            replace(
                config,
                economic_gates=replace(config.economic_gates, maximum_drawdown=999.0),
            )
        )


def test_side_orientation_uses_raw_price_eligibility_and_fee_sensitive_cost_edge() -> None:
    config = load_book_admission_config(CONFIG_PATH)
    instant = datetime(2026, 7, 21, tzinfo=UTC)
    row = _row("both-cheap", instant, 1, 1, yes_is_cheap=True)
    row.update(
        {
            "yes_ask_vwap_5": 0.25,
            "no_ask_vwap_5": 0.25,
            "yes_cost_per_share": 0.30,
            "no_cost_per_share": 0.26,
            "pm_yes_cost_per_share": 0.30,
            "pm_no_cost_per_share": 0.26,
            PROBABILITY_COLUMN: 0.50,
        }
    )
    frame = _frame([row]).drop(SELECTED_SIDE_COLUMN)

    oriented = attach_frozen_policy_selected_side(frame, config)

    assert oriented[SELECTED_SIDE_COLUMN].item() == "NO"
    assert oriented["selected_side_policy_eligible"].item() is True
    with pytest.raises(ValueError, match="frozen policy orientation"):
        attach_frozen_policy_selected_side(
            oriented.with_columns(pl.lit("YES").alias(SELECTED_SIDE_COLUMN)), config
        )


def test_target_selection_fails_closed_on_duplicate_noncausal_and_unsupported_rows() -> None:
    config = load_book_admission_config(CONFIG_PATH)
    start = datetime(2026, 7, 21, tzinfo=UTC)
    row = _row("market-a", start, 1, 1, yes_is_cheap=True)
    frame = _frame([row])

    selected = select_target_opportunity_rows(
        frame, config, require_label=True, feature_names=DYNAMIC_FEATURES
    )
    assert selected.height == 1
    with pytest.raises(ValueError, match="duplicate decision keys"):
        select_target_opportunity_rows(
            pl.concat([frame, frame]),
            config,
            require_label=True,
            feature_names=DYNAMIC_FEATURES,
        )
    with pytest.raises(ValueError, match="non-causal or misaligned"):
        select_target_opportunity_rows(
            frame.with_columns(
                (pl.col("observed_at") + pl.duration(milliseconds=1)).alias("observed_at")
            ),
            config,
            require_label=True,
            feature_names=DYNAMIC_FEATURES,
        )
    with pytest.raises(ValueError, match="labels must be binary"):
        select_target_opportunity_rows(
            frame.with_columns(pl.lit(2).alias("label_up")),
            config,
            require_label=True,
            feature_names=DYNAMIC_FEATURES,
        )
    with pytest.raises(ValueError, match="missing columns"):
        select_target_opportunity_rows(
            frame.drop(DYNAMIC_FEATURES[-1]),
            config,
            require_label=True,
            feature_names=DYNAMIC_FEATURES,
        )


def test_weight_contract_equalizes_markets_and_then_days() -> None:
    start = datetime(2026, 7, 21, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "b", "c", "c", "c"],
            "window_start": [
                start,
                start,
                start,
                start + timedelta(days=1),
                start + timedelta(days=1),
                start + timedelta(days=1),
            ],
        }
    ).with_columns(pl.col("window_start").cast(pl.Datetime("us", "UTC")))

    market = book_admission_weights(frame, MARKET_EQUAL)
    assert market.sum() == pytest.approx(frame.height)
    assert market[:2].sum() == pytest.approx(market[2:3].sum())
    assert market[:2].sum() == pytest.approx(market[3:].sum())

    day = book_admission_weights(frame, DAY_MARKET_ROW_EQUAL)
    assert day[:3].sum() == pytest.approx(day[3:].sum())
    assert day[:2].sum() == pytest.approx(day[2:3].sum())


def test_linear_fit_is_selected_side_coherent_deterministic_and_serializable(
    tmp_path: Path,
) -> None:
    config = load_book_admission_config(CONFIG_PATH)
    fit = _synthetic_frame(datetime(2026, 7, 1, tzinfo=UTC), 90)
    calibration = _synthetic_frame(datetime(2026, 7, 18, tzinfo=UTC), 50, offset=90)

    model = fit_book_admission_candidate(config, "D2", fit, calibration)
    score = score_book_admission_model(model, calibration, config)

    assert model.schema_version == BOOK_ADMISSION_SCHEMA_VERSION
    assert 0.0 <= model.gamma <= 1.0
    assert model.preprocessor.feature_names == DYNAMIC_FEATURES
    assert model.evidence.fit_rows == 90
    assert model.evidence.calibration_rows == 50
    target = select_target_opportunity_rows(
        calibration,
        config,
        require_label=False,
        feature_names=DYNAMIC_FEATURES,
    )
    residual = model.raw_residual(target)
    assert np.max(np.abs(residual)) <= 0.5
    incumbent_yes = target[PROBABILITY_COLUMN].to_numpy()
    candidate_yes = score.frame[PROBABILITY_COLUMN].to_numpy()
    selected_yes = target[SELECTED_SIDE_COLUMN].to_numpy() == "YES"
    incumbent_selected = np.where(selected_yes, incumbent_yes, 1.0 - incumbent_yes)
    candidate_selected = np.where(selected_yes, candidate_yes, 1.0 - candidate_yes)
    observed_delta = _logit(candidate_selected) - _logit(incumbent_selected)
    assert observed_delta == pytest.approx(model.gamma * residual)

    reversed_score = score_book_admission_model(model, calibration.reverse(), config)
    assert reversed_score.key_sha256 == score.key_sha256
    assert reversed_score.probability_sha256 == score.probability_sha256
    assert book_admission_probability_sha256(score.frame) == score.probability_sha256
    support = book_admission_support(calibration, config, "D2")
    assert support.rows == 50
    assert support.positive_selected_outcomes + support.negative_selected_outcomes == 50

    artifact = tmp_path / "d2.pkl"
    serialization = model.serialize(artifact)
    restored = load_book_admission_model(artifact)
    restored_score = score_book_admission_model(restored, calibration, config)
    assert serialization["model_semantic_sha256"] == model.semantic_sha256
    assert restored.semantic_sha256 == model.semantic_sha256
    assert restored_score.probability_sha256 == score.probability_sha256


def test_small_histogram_candidate_uses_frozen_hyperparameters(
    tmp_path: Path,
) -> None:
    config = load_book_admission_config(CONFIG_PATH)
    fit = _synthetic_frame(datetime(2026, 6, 20, tzinfo=UTC), 520)
    calibration = _synthetic_frame(datetime(2026, 7, 18, tzinfo=UTC), 80, offset=520)

    model = fit_book_admission_candidate(config, "D4", fit, calibration)
    score = score_book_admission_model(model, calibration, config)

    assert model.histogram_estimator is not None
    assert model.histogram_estimator.max_iter == 120
    assert model.histogram_estimator.max_leaf_nodes == 5
    assert model.histogram_estimator.min_samples_leaf == 250
    assert np.isfinite(score.frame[PROBABILITY_COLUMN].to_numpy()).all()

    artifact = tmp_path / "d4.pkl"
    semantic_sha256 = model.semantic_sha256
    serialization = model.serialize(artifact)
    assert serialization["model_semantic_sha256"] == semantic_sha256
    for _ in range(3):
        restored = load_book_admission_model(artifact)
        restored_score = score_book_admission_model(restored, calibration, config)
        assert restored.semantic_sha256 == semantic_sha256
        assert restored_score.probability_sha256 == score.probability_sha256


def _synthetic_frame(start: datetime, rows: int, *, offset: int = 0) -> pl.DataFrame:
    records = [
        _row(
            f"market-{offset + index:04d}",
            start + timedelta(minutes=5 * index),
            1 + index % 55,
            (offset + index) % 2,
            yes_is_cheap=(offset + index) % 3 != 0,
        )
        for index in range(rows)
    ]
    return _frame(records)


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("window_start").cast(pl.Datetime("us", "UTC")),
        pl.col("observed_at").cast(pl.Datetime("us", "UTC")),
        pl.col("yes_received_at").cast(pl.Datetime("us", "UTC")),
        pl.col("no_received_at").cast(pl.Datetime("us", "UTC")),
        *(pl.col(name).cast(pl.Boolean) for name in DYNAMIC_FEATURES if name.endswith("_mature")),
    )


def _row(
    market_id: str,
    window_start: datetime,
    second: int,
    label_up: int,
    *,
    yes_is_cheap: bool,
) -> dict[str, object]:
    observed_at = window_start + timedelta(seconds=second)
    yes_price = 0.24 + (second % 5) * 0.005 if yes_is_cheap else 0.74
    no_price = 0.74 if yes_is_cheap else 0.24 + (second % 5) * 0.005
    fee_rate = 0.02
    yes_cost = yes_price + fee_rate * yes_price * (1.0 - yes_price) + 0.01
    no_cost = no_price + fee_rate * no_price * (1.0 - no_price) + 0.01
    probability_yes = 0.58 if yes_is_cheap else 0.42
    values: dict[str, object] = {
        "market_id": market_id,
        "window_start": window_start,
        "observed_at": observed_at,
        "seconds_elapsed": second,
        "yes_received_at": observed_at - timedelta(milliseconds=200),
        "no_received_at": observed_at - timedelta(milliseconds=300),
        "yes_ask_vwap_5": yes_price,
        "no_ask_vwap_5": no_price,
        "yes_cost_per_share": yes_cost,
        "no_cost_per_share": no_cost,
        PROBABILITY_COLUMN: probability_yes,
        "label_up": label_up,
        SELECTED_SIDE_COLUMN: "YES" if yes_is_cheap else "NO",
        "pm_yes_cost_per_share": yes_cost,
        "pm_no_cost_per_share": no_cost,
        "pm_yes_cost_logit": _scalar_logit(yes_cost),
        "pm_no_cost_logit": _scalar_logit(no_cost),
        "pm_cost_overround": yes_cost + no_cost - 1.0,
        "pm_yes_minus_no_cost": yes_cost - no_cost,
        "pm_yes_vwap_slippage": yes_price / 100.0,
        "pm_no_vwap_slippage": no_price / 100.0,
        "pm_yes_depth_log": 3.0 + yes_price,
        "pm_no_depth_log": 3.0 + no_price,
        "pm_depth_imbalance": yes_price - no_price,
        "pm_yes_book_age_seconds": 0.2,
        "pm_no_book_age_seconds": 0.3,
    }
    for index, name in enumerate(DYNAMIC_FEATURES):
        if name in values:
            continue
        if name.endswith("_mature"):
            values[name] = second >= int(name.split("_horizon_")[1].split("s_")[0])
        else:
            values[name] = ((second + index) % 11 - 5) / 100.0
    return values


def _scalar_logit(value: float) -> float:
    return float(np.log(value) - np.log1p(-value))


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values) - np.log1p(-values)
