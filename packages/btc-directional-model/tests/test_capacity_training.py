from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.capacity_training import (
    _attach_asymmetric_book_features,
    _metrics,
    _require_complete_capacity_coverage,
)
from btc_directional_model.capacity_training_config import (
    CapacityExecution,
    CapacityGates,
    CapacityTrainingConfig,
    CapacityWindows,
    load_capacity_training_config,
)


def test_capacity_training_config_pins_four_lineages_and_capacity_suite() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_capacity_training_config(
        root / "configs" / "btc-vwap-capacity-training-20260421-20260720.toml"
    )

    assert config.execution.quantities == (10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200)
    assert config.execution.maximum_depth_participation == 0.25
    assert len(config.lineages) == 4
    assert sum(item.hypothesis == "asymmetric_value" for item in config.lineages) == 1


def test_asymmetric_book_features_use_exact_target_vwap_and_causal_age() -> None:
    observed = datetime(2026, 7, 1, 0, 0, 10, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "fee_rate": [0.02],
            "up_ask_vwap_15": [0.27],
            "down_ask_vwap_15": [0.73],
            "up_best_ask": [0.25],
            "down_best_ask": [0.70],
            "up_ask_depth": [100.0],
            "down_ask_depth": [80.0],
            "observed_at": [observed],
            "up_provider_received_at": [observed - timedelta(seconds=2)],
            "down_provider_received_at": [observed - timedelta(seconds=3)],
        }
    )

    enriched = _attach_asymmetric_book_features(frame, 15, 0.005).row(0, named=True)

    assert abs(enriched["pm_yes_vwap_slippage"] - 0.02) < 1e-12
    assert abs(enriched["pm_no_vwap_slippage"] - 0.03) < 1e-12
    assert enriched["pm_yes_book_age_seconds"] == 2
    assert enriched["pm_no_book_age_seconds"] == 3
    assert enriched["pm_yes_cost_per_share"] > 0.275


def test_metrics_apply_quantity_fee_reserve_and_one_cent_stress() -> None:
    frame = pl.DataFrame(
        {
            "policy_selected": [True, True],
            "correct": [True, False],
            "selected_ask_vwap": [0.40, 0.30],
            "fee_rate": [0.0, 0.0],
        }
    )

    metrics = _metrics(frame, quantity=10, reserve=0.005)

    assert metrics["trades"] == 2
    assert abs(metrics["net_pnl"] - 2.9) < 1e-12
    assert abs(metrics["stress_plus_one_cent"]["net_pnl"] - 2.7) < 1e-12


def test_extraction_fails_closed_until_every_capacity_hour_is_complete(tmp_path: Path) -> None:
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, _query, _parameters):
            return None

        def fetchone(self):
            return 2, datetime(2026, 4, 21, tzinfo=UTC)

    class Connection:
        def cursor(self):
            return Cursor()

    start = datetime(2026, 4, 21, tzinfo=UTC)
    config = CapacityTrainingConfig(
        source_path=tmp_path / "config.toml",
        package_root=tmp_path,
        windows=CapacityWindows(
            development_start=start,
            calibration_start=start + timedelta(days=1),
            policy_start=start + timedelta(days=2),
            freeze_at=start + timedelta(days=3),
        ),
        execution=CapacityExecution((10, 15, 20), 10, 0.25, 0.005, 0.55),
        gates=CapacityGates(1, 1, 1.0, 0.0, 1),
        evidence=tmp_path / "evidence",
        runs=tmp_path / "runs",
        lineages=(),
    )

    with pytest.raises(RuntimeError, match="2 hourly partition"):
        _require_complete_capacity_coverage(Connection(), config)
