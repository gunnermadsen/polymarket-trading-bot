from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from btc_directional_model.latent_twap_state_space import ProbabilityCalibrator
from btc_directional_model.latent_twap_tournament import (
    CHECKPOINT_SECONDS,
    VWAP_QUANTITIES,
    CheckpointStore,
    apply_shared_entry_controller,
    load_config,
    verify_split_separation,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PACKAGE_ROOT
    / "configs"
    / "btc-5m-latent-twap-settlement-risk-20260607-20260828.toml"
)


def test_frozen_config_enforces_periods_folds_and_training_only_scope() -> None:
    config = load_config(CONFIG)

    assert config.freeze_at == datetime(2026, 8, 28, tzinfo=UTC)
    assert config.historical_start == datetime(2026, 6, 7, tzinfo=UTC)
    assert config.calibration_start == datetime(2026, 8, 1, tzinfo=UTC)
    assert config.development_start == datetime(2026, 8, 14, tzinfo=UTC)
    assert len(config.folds) == 7
    assert CHECKPOINT_SECONDS == tuple(range(30, 151, 5))
    assert config.raw["training"]["strictly_training_only"] is True
    assert config.raw["training"]["live_capital_allowed"] is False


def _scored_rows() -> pl.DataFrame:
    start = datetime(2026, 8, 14, tzinfo=UTC)
    rows = []
    for market, probability, label, cost in (
        ("up-market", 0.96, 1, 0.60),
        ("down-market", 0.04, 0, 0.60),
        ("blocked-loss-ratio", 0.99, 1, 0.85),
    ):
        for second in (30, 35):
            row = {
                "market_id": market,
                "window_start": start,
                "window_end": start + timedelta(minutes=5),
                "observed_at": start + timedelta(seconds=second),
                "seconds_elapsed": second,
                "label_source": "authentic_official_twap60",
                "target_margin_bps": 2.0 if label else -2.0,
                "label_up": label,
                "book_valid": True,
                "fee_rate": 0.0,
                "probability_up": probability,
                "raw_probability_up": probability,
                "expected_margin_bps": 2.0 if probability > 0.5 else -2.0,
                "margin_p05_bps": 1.0,
                "margin_p50_bps": 2.0,
                "margin_p95_bps": 3.0,
                "reversal_probability": 0.01,
                "process_uncertainty_bps2": 1.0,
                "sensor_uncertainty_bps2": 1.0,
                "fold": "fold",
            }
            for quantity in VWAP_QUANTITIES:
                row[f"up_ask_vwap_{quantity}"] = cost
                row[f"down_ask_vwap_{quantity}"] = cost
            rows.append(row)
    return pl.DataFrame(rows)


def test_shared_controller_is_symmetric_earliest_and_enforces_three_win_recovery() -> None:
    config = load_config(CONFIG)
    calibrator = ProbabilityCalibrator(1.0, support_markets=1000, support_rows=25000)

    selected = apply_shared_entry_controller(_scored_rows(), calibrator, config)

    assert set(selected["market_id"]) == {"up-market", "down-market"}
    assert selected["seconds_elapsed"].to_list() == [30, 30]
    assert selected.filter(pl.col("market_id") == "up-market")["predicted_up"].item()
    assert not selected.filter(pl.col("market_id") == "down-market")["predicted_up"].item()
    assert selected["quoted_loss_recovery_ratio"].max() <= 3.0


def test_market_splits_are_pristine_and_fold_training_never_overlaps_testing() -> None:
    config = load_config(CONFIG)
    starts = (
        datetime(2026, 6, 7, tzinfo=UTC),
        datetime(2026, 8, 1, tzinfo=UTC),
        datetime(2026, 8, 14, tzinfo=UTC),
    )
    frame = pl.DataFrame(
        {
            "market_id": ["historical", "calibration", "development"],
            "window_start": starts,
        }
    )

    audit = verify_split_separation(frame, config)

    assert audit["passed"] is True
    assert all(value == 0 for value in audit["block_overlaps"].values())
    assert all(value == 0 for value in audit["fold_train_test_overlaps"].values())


def test_checkpoint_resumption_verifies_hash_and_skips_completed_work(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    calls = 0

    def compute() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"value": calls}

    first = store.get_or_compute("candidate-fold", "input-hash", compute)
    second = store.get_or_compute("candidate-fold", "input-hash", compute)

    assert first == second == {"value": 1}
    assert calls == 1
