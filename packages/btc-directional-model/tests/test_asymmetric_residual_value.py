from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest

from btc_directional_model.asymmetric_residual_value import (
    CORE_ORACLE_ANCHOR,
    CROSS_SOURCE_CONFIRMATION,
    GROUP_PENALTIES,
    L2_CONFIRMATION,
    REQUIRED_FEATURE_COLUMNS,
    RESIDUAL_FEATURE_NAMES,
    RESIDUAL_FEATURE_SPECS,
    SIDE_CONDITIONED_RESIDUAL_MODEL,
    SideConditionedResidualModel,
    derive_side_conditioned_rows,
    feature_penalty_manifest,
    market_equal_side_weights,
    normalized_executable_cost_prior,
)
from btc_directional_model.asymmetric_value_benchmark import (
    _fit_development_residual,
    _project_candidate_source,
    _residual_matched_attribution,
)
from btc_directional_model.asymmetric_value_config import load_asymmetric_value_config
from btc_directional_model.asymmetric_value_evaluation import (
    candidate_policy_key,
    select_policy_candidate,
)
from btc_directional_model.asymmetric_value_training import (
    ASYMMETRIC_VALUE_CANDIDATES,
    CORE_ORACLE_L2_PRICE,
    MODEL_SELECTION_ELIGIBLE,
    THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
    asymmetric_value_feature_sets,
)


def residual_frame(
    *,
    markets: int = 12,
    seconds: tuple[int, ...] = (1, 5, 15, 30, 60),
    origin: datetime | None = None,
    market_prefix: str = "market",
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    origin = origin or datetime(2026, 6, 1, tzinfo=UTC)
    for market_index in range(markets):
        window_start = origin + timedelta(minutes=5 * market_index)
        label_up = market_index % 2
        direction = 1.0 if label_up else -1.0
        for elapsed in seconds:
            observed_at = window_start + timedelta(seconds=elapsed)
            yes_received_at = observed_at - timedelta(milliseconds=100)
            no_received_at = observed_at - timedelta(milliseconds=200)
            oracle_block_timestamp = observed_at - timedelta(seconds=2)
            oracle_source_timestamp = oracle_block_timestamp - timedelta(seconds=1)
            path = direction * (2.0 + 0.08 * elapsed) + (market_index % 3 - 1) * 0.2
            oracle_return = direction * (1.5 + 0.05 * elapsed)
            microprice = direction * (0.4 + 0.01 * elapsed)
            yes_cost = 0.44 - 0.04 * direction
            no_cost = 0.58 + 0.04 * direction
            rows.append(
                {
                    "market_id": f"{market_prefix}-{market_index}",
                    "window_start": window_start,
                    "observed_at": observed_at,
                    "seconds_elapsed": elapsed,
                    "label_up": label_up,
                    "yes_received_at": yes_received_at,
                    "no_received_at": no_received_at,
                    "oracle_source_timestamp": oracle_source_timestamp,
                    "oracle_block_timestamp": oracle_block_timestamp,
                    "early_oracle_eligible": True,
                    "pm_yes_cost_per_share": yes_cost,
                    "pm_no_cost_per_share": no_cost,
                    "btc_path_from_window_open_bps": path,
                    "btc_return_1s_bps": direction * 0.5,
                    "btc_return_5s_bps": direction * 1.0,
                    "btc_return_15s_bps": direction * 1.5,
                    "btc_return_30s_bps": direction * 2.0,
                    "btc_return_60s_bps": direction * 2.5,
                    "btc_signed_flow_5s": direction * 0.10,
                    "btc_signed_flow_30s": direction * 0.20,
                    "btc_signed_flow_60s": direction * 0.30,
                    "oracle_return_from_window_open_bps": oracle_return,
                    "binance_oracle_basis_bps": direction * 0.25,
                    "spot_l2_microprice_to_midpoint_bps": microprice,
                    "spot_l2_midpoint_change_1s_bps": direction * 0.1,
                    "spot_l2_midpoint_change_5s_bps": direction * 0.2,
                    "spot_l2_midpoint_change_15s_bps": direction * 0.3,
                    "spot_l2_midpoint_change_30s_bps": direction * 0.4,
                    "spot_l2_midpoint_change_60s_bps": direction * 0.5,
                    "spot_l2_imbalance_5": direction * 0.2,
                    "spot_l2_imbalance_20": direction * 0.3,
                    "pm_yes_depth_log": 3.0 + 0.1 * direction,
                    "pm_no_depth_log": 3.0 - 0.1 * direction,
                    "pm_yes_vwap_slippage": 0.008 - 0.001 * direction,
                    "pm_no_vwap_slippage": 0.008 + 0.001 * direction,
                    "pm_yes_book_age_seconds": 0.1,
                    "pm_no_book_age_seconds": 0.2,
                    "pm_cost_overround": yes_cost + no_cost - 1.0,
                    "oracle_age_seconds": 2.0,
                }
            )
    return pl.DataFrame(rows)


def zero_model() -> SideConditionedResidualModel:
    return SideConditionedResidualModel(
        coefficients=(0.0,) * len(RESIDUAL_FEATURE_NAMES),
        feature_scales=(1.0,) * len(RESIDUAL_FEATURE_NAMES),
        group_penalties=tuple(GROUP_PENALTIES.items()),
    )


def test_normalized_executable_cost_prior_uses_both_admission_costs() -> None:
    frame = pl.DataFrame(
        {
            "pm_yes_cost_per_share": [0.30, 0.08],
            "pm_no_cost_per_share": [0.75, 0.94],
        }
    )

    prior = normalized_executable_cost_prior(frame)

    np.testing.assert_allclose(prior[:, 0], [0.30 / 1.05, 0.08 / 1.02])
    np.testing.assert_allclose(prior.sum(axis=1), 1.0)


def test_every_decision_has_exactly_two_antisymmetric_coherent_side_rows() -> None:
    frame = residual_frame(markets=2, seconds=(1, 30))
    rows = derive_side_conditioned_rows(frame, require_labels=True)

    assert rows.sides == ("YES", "NO") * frame.height
    assert len(rows.market_ids) == frame.height * 2
    np.testing.assert_allclose(rows.features[0::2], -rows.features[1::2])
    np.testing.assert_allclose(rows.prior_probability[0::2] + rows.prior_probability[1::2], 1.0)
    np.testing.assert_allclose(rows.prior_logit[0::2], -rows.prior_logit[1::2])
    np.testing.assert_array_equal(rows.labels[0::2] + rows.labels[1::2], 1.0)

    model = SideConditionedResidualModel(
        coefficients=tuple(np.linspace(-0.03, 0.03, len(RESIDUAL_FEATURE_NAMES))),
        feature_scales=(1.0,) * len(RESIDUAL_FEATURE_NAMES),
        group_penalties=tuple(GROUP_PENALTIES.items()),
    )
    side_probability = model.predict_side_probability(frame.drop("label_up"))

    assert side_probability.shape == (frame.height, 2)
    np.testing.assert_allclose(side_probability.sum(axis=1), 1.0)
    assert np.isfinite(side_probability).all()


def test_zero_correction_is_exactly_the_normalized_executable_prior() -> None:
    frame = residual_frame(markets=2, seconds=(5,))

    expected = normalized_executable_cost_prior(frame)[:, 0]
    actual = zero_model().predict_yes_probability(frame.drop("label_up"))

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_feature_derivation_cannot_observe_labels_or_outcomes() -> None:
    frame = residual_frame(markets=3, seconds=(1, 15))
    flipped = frame.with_columns((1 - pl.col("label_up")).alias("label_up"))

    original = derive_side_conditioned_rows(frame, require_labels=False)
    changed = derive_side_conditioned_rows(flipped, require_labels=False)

    np.testing.assert_array_equal(original.features, changed.features)
    np.testing.assert_array_equal(original.prior_logit, changed.prior_logit)
    forbidden = ("label", "outcome", "pnl", "profit")
    assert not any(
        token in column
        for spec in RESIDUAL_FEATURE_SPECS
        for column in (spec.name, *spec.source_columns)
        for token in forbidden
    )


def test_market_equal_weights_total_one_even_with_unequal_decision_counts() -> None:
    ids = ("many", "many", "many", "many", "few", "few")

    weights = market_equal_side_weights(ids)

    assert weights[:4].sum() == pytest.approx(1.0)
    assert weights[4:].sum() == pytest.approx(1.0)


def test_explicit_maturity_flags_mask_unavailable_horizons_without_filling() -> None:
    frame = residual_frame(markets=1, seconds=(1, 5, 15, 30, 60))
    for horizon in (1, 5, 15, 30, 60):
        frame = frame.with_columns(
            pl.when(pl.col("seconds_elapsed") < horizon)
            .then(None)
            .otherwise(pl.col(f"btc_return_{horizon}s_bps"))
            .alias(f"btc_return_{horizon}s_bps"),
            pl.when(pl.col("seconds_elapsed") < horizon)
            .then(None)
            .otherwise(pl.col(f"spot_l2_midpoint_change_{horizon}s_bps"))
            .alias(f"spot_l2_midpoint_change_{horizon}s_bps"),
        )
    for horizon in (5, 30, 60):
        frame = frame.with_columns(
            pl.when(pl.col("seconds_elapsed") < horizon)
            .then(None)
            .otherwise(pl.col(f"btc_signed_flow_{horizon}s"))
            .alias(f"btc_signed_flow_{horizon}s")
        )
    rows = derive_side_conditioned_rows(frame, require_labels=False)
    yes = rows.features[0::2]

    for horizon in (1, 5, 15, 30, 60):
        flag_index = RESIDUAL_FEATURE_NAMES.index(f"side_horizon_mature_{horizon}s")
        return_index = RESIDUAL_FEATURE_NAMES.index(f"side_core_return_{horizon}s_bps")
        expected_flag = (np.asarray((1, 5, 15, 30, 60)) >= horizon).astype(float)
        np.testing.assert_array_equal(yes[:, flag_index], expected_flag)
        assert np.all(yes[expected_flag == 0.0, return_index] == 0.0)
        assert frame.filter(pl.col("seconds_elapsed") < horizon)[
            f"btc_return_{horizon}s_bps"
        ].null_count() == int((expected_flag == 0.0).sum())


def test_missing_mature_horizon_fails_closed() -> None:
    frame = residual_frame(markets=1, seconds=(60,)).with_columns(
        pl.lit(None).cast(pl.Float64).alias("btc_return_60s_bps")
    )

    with pytest.raises(ValueError, match="mature inputs.*btc_return_60s_bps"):
        derive_side_conditioned_rows(frame, require_labels=False)


def test_fit_is_deterministic_finite_and_joblib_compatible(tmp_path: Path) -> None:
    frame = residual_frame(markets=24)

    model, diagnostics = SideConditionedResidualModel.fit(frame)
    repeat, repeated_diagnostics = SideConditionedResidualModel.fit(frame)
    probability = model.predict_yes_probability(frame.drop("label_up"))

    assert model == repeat
    assert diagnostics == repeated_diagnostics
    assert diagnostics.converged
    assert diagnostics.market_weight_total_min == pytest.approx(1.0)
    assert diagnostics.market_weight_total_max == pytest.approx(1.0)
    assert np.isfinite(probability).all()
    assert np.all((probability > 0.0) & (probability < 1.0))

    artifact = tmp_path / "residual.joblib"
    joblib.dump(model, artifact)
    restored = joblib.load(artifact)
    assert restored == model
    np.testing.assert_array_equal(
        restored.predict_yes_probability(frame.drop("label_up")),
        probability,
    )


def test_penalty_manifest_keeps_l2_confirmation_weaker_than_anchor() -> None:
    manifest = feature_penalty_manifest()
    penalties = manifest["group_penalties"]

    assert penalties[L2_CONFIRMATION] >= penalties[CORE_ORACLE_ANCHOR]
    assert penalties[CROSS_SOURCE_CONFIRMATION] >= penalties[CORE_ORACLE_ANCHOR]
    assert manifest["selection_eligible"] is False
    assert manifest["runtime_exportable"] is False
    fitted_manifest = zero_model().manifest()
    assert fitted_manifest["no_policy_or_pnl_objective"] is True


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda frame: frame.drop("spot_l2_imbalance_20"), "missing columns"),
        (
            lambda frame: frame.with_columns(
                pl.when(pl.int_range(pl.len()) == 0)
                .then(None)
                .otherwise(pl.col("btc_return_1s_bps"))
                .alias("btc_return_1s_bps")
            ),
            "missing or non-finite",
        ),
        (
            lambda frame: frame.with_columns(
                (pl.col("observed_at") + pl.duration(seconds=1)).alias("yes_received_at")
            ),
            "missing or noncausal",
        ),
        (
            lambda frame: frame.with_columns(
                (pl.col("observed_at") + pl.duration(seconds=1)).alias(
                    "oracle_block_timestamp"
                )
            ),
            "missing or noncausal",
        ),
    ),
)
def test_missing_or_noncausal_inputs_fail_closed(mutator: object, message: str) -> None:
    frame = residual_frame(markets=2, seconds=(5, 30))

    with pytest.raises(ValueError, match=message):
        derive_side_conditioned_rows(mutator(frame), require_labels=False)  # type: ignore[operator]


def _development_config():
    return load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-calibrated-20260414-20260802.toml"
    )


def test_three_source_projection_retains_residual_audit_contract_without_filling() -> None:
    frame = residual_frame(markets=1, seconds=(5,))
    feature_sets = asymmetric_value_feature_sets()
    model_features = {
        *feature_sets[THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL],
        *feature_sets[CORE_ORACLE_L2_PRICE],
    }
    required_execution = {
        "fee_rate",
        "yes_best_ask",
        "yes_ask_vwap_5",
        "yes_ask_depth",
        "no_best_ask",
        "no_ask_vwap_5",
        "no_ask_depth",
        "yes_cost_per_share",
        "no_cost_per_share",
        "yes_execution_cost_per_share",
        "no_execution_cost_per_share",
    }
    frame = frame.with_columns(
        *(
            pl.lit(0.1).alias(name)
            for name in sorted(model_features | required_execution)
            if name not in frame.columns
        )
    )

    projected = _project_candidate_source(
        frame,
        THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
        CORE_ORACLE_L2_PRICE,
    )

    assert set(REQUIRED_FEATURE_COLUMNS).issubset(projected.columns)
    assert projected["yes_received_at"].item() == frame["yes_received_at"].item()
    assert projected["oracle_block_timestamp"].item() == (
        frame["oracle_block_timestamp"].item()
    )
    assert projected.null_count().select(pl.sum_horizontal(pl.all())).item() == 0


def test_development_wiring_fits_only_fit_rows_and_scores_exact_policy_rows(
    tmp_path: Path,
) -> None:
    config = _development_config()
    fit = residual_frame(
        markets=24,
        origin=config.fit.start,
        market_prefix="fit",
    )
    policy = residual_frame(
        markets=12,
        origin=config.policy.start,
        market_prefix="policy",
    )
    source = pl.concat((fit, policy), how="vertical_relaxed")

    model, profile, policy_frame = _fit_development_residual(source, config)
    original = model.predict_yes_probability(policy_frame)
    flipped = model.predict_yes_probability(
        policy_frame.with_columns((1 - pl.col("label_up")).alias("label_up"))
    )

    assert profile["fit_rows"] == fit.height
    assert profile["fit_markets"] == fit["market_id"].n_unique()
    assert profile["policy_rows"] == policy.height
    assert profile["fit_window"] == {
        "start": config.fit.start.isoformat(),
        "end": config.fit.end.isoformat(),
    }
    assert profile["policy_window"] == {
        "start": config.policy.start.isoformat(),
        "end": config.policy.end.isoformat(),
    }
    assert profile["calibration_labels_consumed"] is False
    assert profile["policy_labels_consumed_by_fit"] is False
    assert profile["pnl_consumed_by_fit"] is False
    assert profile["selection_eligible"] is False
    assert profile["runtime_exportable"] is False
    np.testing.assert_array_equal(original, flipped)

    artifact = tmp_path / f"{SIDE_CONDITIONED_RESIDUAL_MODEL}.joblib"
    joblib.dump(model, artifact, compress=3)
    restored = joblib.load(artifact)
    np.testing.assert_array_equal(
        restored.predict_yes_probability(policy_frame),
        original,
    )


def test_residual_is_matched_in_frontier_but_never_selection_eligible() -> None:
    config = _development_config()
    policy_name = next(
        policy.name for policy in config.policies if policy.selection_eligible
    )
    frame = residual_frame(
        markets=4,
        origin=config.policy.start,
        market_prefix="policy",
    )
    probability = {
        "rows": frame.height,
        "markets": frame["market_id"].n_unique(),
        "accuracy": 0.6,
        "brier_score": 0.2,
        "log_loss": 0.6,
        "mean_probability_yes": 0.5,
        "actual_yes_rate": 0.5,
    }
    training = {
        "profiles": {
            SIDE_CONDITIONED_RESIDUAL_MODEL: {
                "policy_probability_metrics": probability
            },
            THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL: {
                "policy_probability_metrics": {**probability, "accuracy": 0.55}
            },
            CORE_ORACLE_L2_PRICE: {
                "policy_probability_metrics": {**probability, "accuracy": 0.58}
            },
        }
    }
    economics = {
        "trades": 1,
        "accuracy": 1.0,
        "net_expectancy_per_trade": 0.1,
        "stress_1c_net_expectancy_per_trade": 0.05,
        "net_profit_per_resolved_market": 0.01,
        "capital_efficiency": 0.1,
        "profit_factor": 2.0,
        "selected_calibration_bias": 0.01,
        "trades_per_resolved_market": 0.01,
    }
    benchmark_models = (
        *ASYMMETRIC_VALUE_CANDIDATES,
        SIDE_CONDITIONED_RESIDUAL_MODEL,
    )
    metrics = {
        candidate_policy_key(model, policy_name): dict(economics)
        for model in benchmark_models
    }
    empty_ledger = pl.DataFrame(
        schema={
            "window_start": pl.Datetime("us", "UTC"),
            "realized_net": pl.Float64,
        }
    )
    ledgers = {
        candidate_policy_key(model, policy_name): empty_ledger
        for model in benchmark_models
    }
    frames = {
        THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL: frame,
        CORE_ORACLE_L2_PRICE: frame,
        SIDE_CONDITIONED_RESIDUAL_MODEL: frame,
    }

    attribution = _residual_matched_attribution(
        training,
        metrics,
        ledgers,
        frames,
        policy_name=policy_name,
        config=config,
    )
    mismatched_frames = dict(frames)
    mismatched_frames[CORE_ORACLE_L2_PRICE] = frame.with_columns(
        (pl.col("seconds_elapsed") + 1).alias("seconds_elapsed")
    )
    with pytest.raises(RuntimeError, match="does not share exact keys"):
        _residual_matched_attribution(
            training,
            metrics,
            ledgers,
            mismatched_frames,
            policy_name=policy_name,
            config=config,
        )
    selection = select_policy_candidate(
        metrics,
        config,
        eligible_models=set(MODEL_SELECTION_ELIGIBLE),
    )
    residual_frontier = next(
        row
        for row in selection["frontier"]
        if row["model"] == SIDE_CONDITIONED_RESIDUAL_MODEL
    )

    assert attribution["promotion_claim"] is False
    assert attribution["selection_eligible"] is False
    assert set(attribution["comparisons"]) == {
        THREE_SOURCE_MATCHED_CORE_ORACLE_PRICE_CONTROL,
        CORE_ORACLE_L2_PRICE,
    }
    assert all(
        comparison["identical_market_second_keys_verified"]
        for comparison in attribution["comparisons"].values()
    )
    assert residual_frontier["selection_eligible"] is False
    assert selection["selected_model"] in MODEL_SELECTION_ELIGIBLE
