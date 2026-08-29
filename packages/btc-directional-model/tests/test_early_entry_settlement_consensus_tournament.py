from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.early_entry_settlement_consensus_tournament import (
    BRIDGE_FEATURES,
    CANDIDATE_NAMES,
    CAUSAL_FEATURES,
    ENTRY_SECONDS,
    HISTORY_ARMS,
    LATENT_SENSORS,
    CandidateBundle,
    CheckpointStore,
    LatentSpec,
    Policy,
    _fit_latent_parameters,
    _fit_nonnegative_logit,
    _latent_filter,
    _market_schedule_audit,
    _opportunity_panel,
    _select_one_trade_per_market,
    economic_metrics,
    load_config,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PACKAGE_ROOT
    / "configs/btc-5m-early-entry-settlement-consensus-20260321-20260827.toml"
)


def test_frozen_config_has_exact_rosters_and_boundaries() -> None:
    config = load_config(CONFIG)
    assert tuple(config.raw["training"]["observation_seconds"]) == ENTRY_SECONDS
    assert tuple(config.raw["training"]["candidates"]) == CANDIDATE_NAMES
    assert tuple(config.raw["training"]["history_arms"]) == HISTORY_ARMS
    assert len(config.folds) == 8
    assert sum(fold.official for fold in config.folds) == 6
    assert config.candidate_freeze == datetime(2026, 8, 27, tzinfo=UTC)
    assert config.prospective_end == datetime(2026, 8, 28, tzinfo=UTC)
    assert config.raw["training"]["training_only"] is True
    assert config.raw["training"]["runtime_exported"] is False


def test_market_schedule_audit_requires_all_25_rows_together() -> None:
    rows = []
    for market_index in range(3):
        for second in ENTRY_SECONDS:
            rows.append(
                {
                    "market_id": f"market-{market_index}",
                    "seconds_elapsed": second,
                    "label_up": market_index % 2,
                    "target_margin_bps": float(market_index - 1),
                }
            )
    frame = pl.DataFrame(rows)
    assert _market_schedule_audit(frame)["passed"] is True
    broken = frame.filter(
        ~((pl.col("market_id") == "market-1") & (pl.col("seconds_elapsed") == 180))
    )
    audit = _market_schedule_audit(broken)
    assert audit["passed"] is False
    assert audit["invalid_markets"] == 1


def test_nonnegative_stack_cannot_cancel_constituents() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(500, 3))
    labels = (matrix[:, 0] + 0.5 * matrix[:, 1] + rng.normal(scale=0.3, size=500) > 0).astype(int)
    weights = np.full(500, 1 / 500)
    model = _fit_nonnegative_logit(matrix, labels, weights, l2=12.0)
    assert all(value >= 0 for value in model.coefficients)
    probability = model.predict(matrix)
    assert np.all((probability > 0) & (probability < 1))


def test_latent_state_filter_is_causal_and_finite() -> None:
    rng = np.random.default_rng(11)
    sequences = []
    targets = []
    for market in range(60):
        target = rng.normal(scale=8)
        path = np.linspace(0, target, len(ENTRY_SECONDS))
        sequence = np.column_stack(
            (
                path + rng.normal(scale=1.0, size=len(path)),
                path + rng.normal(scale=0.8, size=len(path)),
                path + rng.normal(scale=1.2, size=len(path)),
            )
        )
        sequences.append(sequence)
        targets.append(target)
    spec = LatentSpec(0.92, 2.0, 2.0, 1.5, 20.0)
    parameters = _fit_latent_parameters(
        sequences, np.asarray(targets), np.ones(len(targets)), spec
    )
    full = _latent_filter(sequences[0], parameters)
    prefix = _latent_filter(sequences[0][:-1], parameters)
    for key in (
        "probability_up",
        "margin_lower_bps",
        "margin_median_bps",
        "margin_upper_bps",
        "uncertainty_bps",
    ):
        assert np.all(np.isfinite(full[key]))
        np.testing.assert_allclose(full[key][:-1], prefix[key], atol=1e-12, rtol=0)


def _policy_frame() -> pl.DataFrame:
    start = datetime(2026, 8, 14, tzinfo=UTC)
    rows = []
    for market_index in range(2):
        for offset, second in enumerate((60, 65, 70)):
            observed = start + timedelta(minutes=5 * market_index, seconds=second)
            rows.append(
                {
                    "market_id": f"market-{market_index}",
                    "window_start": start + timedelta(minutes=5 * market_index),
                    "window_end": start + timedelta(minutes=5 * (market_index + 1)),
                    "observed_at": observed,
                    "seconds_elapsed": second,
                    "probability_up": 0.95 - offset * 0.01,
                    "predicted_margin_lower_bps": 2.0,
                    "predicted_margin_bps": 5.0,
                    "predicted_margin_upper_bps": 8.0,
                    "prediction_uncertainty_bps": 2.0,
                    "consensus_strength": 1.0,
                    "label_up": 1,
                    "fee_rate": 0.0,
                    "quality_flags": 0,
                    "up_provider_received_at": observed - timedelta(seconds=1),
                    "down_provider_received_at": observed - timedelta(seconds=1),
                    "up_best_ask": 0.30,
                    "down_best_ask": 0.72,
                    "up_ask_depth": 100.0,
                    "down_ask_depth": 100.0,
                    "up_ask_vwap_5": 0.30,
                    "down_ask_vwap_5": 0.72,
                    "candidate": "frozen_bridge_control",
                    "fold": "test",
                }
            )
    return pl.DataFrame(rows)


def test_policy_selects_at_most_one_trade_and_zero_loss_is_not_rejected() -> None:
    policy = Policy(
        "probability_edge_control",
        minimum_stressed_edge=0.05,
        maximum_debit=1.0,
        slippage_reserve=0.01,
        uncertainty_reserve_scale=0.0,
        require_margin_excludes_zero=False,
        minimum_consensus=0.0,
        maximum_loss_recovery_wins=100.0,
    )
    selected = _select_one_trade_per_market(_opportunity_panel(_policy_frame(), policy))
    assert selected.height == 2
    assert selected["market_id"].n_unique() == 2
    assert selected["seconds_elapsed"].to_list() == [60, 60]
    metrics = economic_metrics(selected, scheduled_markets=2)
    assert metrics["losses"] == 0
    assert metrics["profit_factor"] is None
    assert metrics["profit_factor_undefined_zero_losses"] is True
    assert metrics["accuracy_exact_interval"] is not None
    assert metrics["pnl_after_injected_stressed_loss"] is not None


def test_control_predictions_ignore_all_supervision_fields() -> None:
    rows = 5
    frame = pl.DataFrame(
        {
            "bridge_probability_up": np.linspace(0.2, 0.8, rows),
            "bridge_margin_lower_bps": np.arange(rows) - 2.0,
            "bridge_margin_median_bps": np.arange(rows, dtype=float),
            "bridge_margin_upper_bps": np.arange(rows) + 2.0,
            "bridge_uncertainty_bps": np.ones(rows),
            "latent_probability_up": np.full(rows, 0.5),
            "latent_margin_lower_bps": np.full(rows, -1.0),
            "latent_margin_median_bps": np.zeros(rows),
            "latent_margin_upper_bps": np.ones(rows),
            "latent_uncertainty_bps": np.ones(rows),
            "causal_probability_up": np.full(rows, 0.5),
            "causal_margin_lower_bps": np.full(rows, -1.0),
            "causal_margin_median_bps": np.zeros(rows),
            "causal_margin_upper_bps": np.ones(rows),
            "causal_uncertainty_bps": np.ones(rows),
            "label_up": np.zeros(rows, dtype=int),
            "target_margin_bps": np.zeros(rows),
            "label_source": ["authentic"] * rows,
        }
    )
    bundle = CandidateBundle(
        "frozen_bridge_control", None, None, None, None, None, 0.0, 0.0, datetime.now(UTC)
    )
    baseline = bundle.score_arrays(frame)
    perturbed = frame.with_columns(
        pl.lit(1).alias("label_up"),
        pl.lit(9999.0).alias("target_margin_bps"),
        pl.lit("synthetic-perturbed").alias("label_source"),
    )
    after = bundle.score_arrays(perturbed)
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], after[key])
    assert not ({*BRIDGE_FEATURES, *CAUSAL_FEATURES, *LATENT_SENSORS} & {"label_up", "label_source", "target_margin_bps"})


def test_checkpoint_resume_rejects_changed_identity(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "run-identity")
    assert store.value("stage", {"fold": 1}, lambda: {"value": 7}) == {"value": 7}
    assert store.value("stage", {"fold": 1}, lambda: pytest.fail("must resume")) == {"value": 7}
    with pytest.raises(RuntimeError, match="identity mismatch"):
        store.value("stage", {"fold": 2}, lambda: {"value": 9})
