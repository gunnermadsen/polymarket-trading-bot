from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.asymmetric_book_admission import PROBABILITY_COLUMN
from btc_directional_model.asymmetric_d4_side_calibration import (
    D4_SIDE_CALIBRATION_SCHEMA_VERSION,
    PARENT_RUN_ID,
    d4_side_calibration_support,
    fit_d4_side_calibrator,
    load_d4_side_calibration_config,
    score_d4_side_calibrator,
    validate_d4_parent_run_artifacts,
    validate_d4_side_calibration_config,
)


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path() -> Path:
    return (
        package_root()
        / "configs"
        / "btc-5m-asymmetric-core-oracle-d4-side-calibration-20260414-20260802.toml"
    )


def calibration_frame(*, days: int = 14, no_wins_every: int = 4) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    start_day = datetime(2026, 7, 7, tzinfo=UTC)
    for day_offset in range(days):
        day = start_day + timedelta(days=day_offset)
        for market_offset in range(10):
            market_index = day_offset * 10 + market_offset
            window_start = day + timedelta(minutes=5 * market_offset)
            label_up = 0 if market_index % no_wins_every == 0 else 1
            for seconds in (2, 20, 35, 50):
                rows.append(
                    {
                        "market_id": f"market-{market_index}",
                        "window_start": window_start,
                        "observed_at": window_start + timedelta(seconds=seconds),
                        "seconds_elapsed": seconds,
                        "selected_side": "NO",
                        PROBABILITY_COLUMN: 0.65,
                        "label_up": label_up,
                    }
                )
            rows.append(
                {
                    "market_id": f"market-{market_index}",
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=10),
                    "seconds_elapsed": 10,
                    "selected_side": "YES",
                    PROBABILITY_COLUMN: 0.35,
                    "label_up": label_up,
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("window_start").cast(pl.Datetime("us", "UTC")),
        pl.col("observed_at").cast(pl.Datetime("us", "UTC")),
    )


def test_contract_freezes_d4_parent_policy_chronology_and_diagnostic_boundary() -> None:
    config = load_d4_side_calibration_config(config_path())

    assert config.evidence_scope == "consumed_architecture_feedback"
    assert config.projected_pnl_diagnostic is True
    assert config.qualification_eligible is False
    assert config.support_qualification_allowed is False
    assert config.runtime_deployable is False
    assert config.batch_forward_eligible is False
    assert config.process_change_allowed is False
    assert config.parent_run.run_id == PARENT_RUN_ID
    assert config.parent_run.d4_oof_predictions_sha256 == (
        "0faea82a211b968d5b182da56ab07a9fcfe0d0d909079d1f409946d7b7d0e6d3"
    )
    assert config.parent_d4.estimator == "histogram_gradient_boosting"
    assert config.parent_d4.learning_rate == 0.03
    assert config.parent_d4.max_iter == 120
    assert config.parent_d4.max_leaf_nodes == 5
    assert config.parent_d4.min_samples_leaf == 250
    assert config.parent_d4.l2_regularization == 20.0
    assert config.policy.minimum_share_price == 0.20
    assert config.policy.maximum_share_price == 0.30
    assert config.policy.minimum_entry_second == 1
    assert config.policy.maximum_entry_second == 55
    assert [arm.name for arm in config.calibration.arms] == ["D4-base", "N1", "N2"]
    assert [fold.name for fold in config.folds] == [
        "2026-07-21",
        "2026-07-22",
        "2026-07-23",
        "2026-07-24",
        "2026-07-25",
        "2026-07-28",
        "2026-07-29",
        "2026-07-30",
        "2026-07-31",
        "2026-08-01",
    ]
    assert config.folds[0].calibration.start == datetime(2026, 7, 7, tzinfo=UTC)
    assert config.folds[-1].validation.end == datetime(2026, 8, 2, tzinfo=UTC)

    with pytest.raises(ValueError, match="diagnostic boundary"):
        validate_d4_side_calibration_config(replace(config, qualification_eligible=True))


def test_parent_run_artifact_validation_fails_closed_when_run_is_unavailable(
    tmp_path: Path,
) -> None:
    config = load_d4_side_calibration_config(config_path())
    isolated = replace(config, paths=replace(config.paths, parent_run=tmp_path))

    with pytest.raises(FileNotFoundError, match="pinned D4 parent artifact"):
        validate_d4_parent_run_artifacts(isolated)


def test_support_counts_distinct_no_market_outcomes_and_all_time_cells() -> None:
    config = load_d4_side_calibration_config(config_path())
    frame = calibration_frame()

    support = d4_side_calibration_support(frame, config, "N2")

    assert support.rows == 700
    assert support.markets == 140
    assert support.utc_days == 14
    assert support.no_rows == 560
    assert support.no_markets == 140
    assert support.no_utc_days == 14
    assert support.no_winning_markets == 35
    assert support.no_losing_markets == 105
    assert [cell.name for cell in support.time_cells] == [
        "NO_1_15",
        "NO_15_30",
        "NO_30_45",
        "NO_45_56",
    ]
    assert all(cell.markets == 140 for cell in support.time_cells)
    assert all(cell.winning_markets == 35 for cell in support.time_cells)
    assert all(cell.losing_markets == 105 for cell in support.time_cells)


def test_d4_base_reproduces_probabilities_and_yes_is_unchanged_for_n1() -> None:
    config = load_d4_side_calibration_config(config_path())
    frame = calibration_frame()
    parent = frame[PROBABILITY_COLUMN].to_numpy()

    base = fit_d4_side_calibrator(config, "D4-base", frame)
    base_score = score_d4_side_calibrator(base, frame)
    assert base.schema_version == D4_SIDE_CALIBRATION_SCHEMA_VERSION
    assert base.no_logit_offsets == ()
    assert np.array_equal(base_score.frame[PROBABILITY_COLUMN].to_numpy(), parent)
    assert base_score.frame.columns == [
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "candidate_name",
        PROBABILITY_COLUMN,
    ]

    n1 = fit_d4_side_calibrator(config, "N1", frame)
    n1_probability = n1.probability_yes(frame)
    sides = frame["selected_side"].to_numpy()
    assert n1.fixed_slope == 1.0
    assert len(n1.no_logit_offsets) == 1
    assert -0.5 <= n1.no_logit_offsets[0][1] < 0.0
    assert np.array_equal(n1_probability[sides == "YES"], parent[sides == "YES"])
    assert np.all(n1_probability[sides == "NO"] > parent[sides == "NO"])
    assert n1.evidence.fitted_objective < n1.evidence.identity_objective


def test_n2_fits_four_bounded_conservative_offsets() -> None:
    config = load_d4_side_calibration_config(config_path())
    frame = calibration_frame()

    n2 = fit_d4_side_calibrator(config, "N2", frame)

    assert [name for name, _ in n2.no_logit_offsets] == [
        "NO_1_15",
        "NO_15_30",
        "NO_30_45",
        "NO_45_56",
    ]
    assert all(-0.5 <= value <= 0.0 for _, value in n2.no_logit_offsets)
    assert n2.evidence.fitted_objective < n2.evidence.identity_objective


def test_fit_rejects_non_frozen_chronology_and_insufficient_support() -> None:
    config = load_d4_side_calibration_config(config_path())

    with pytest.raises(ValueError, match="exactly one frozen 14-day"):
        fit_d4_side_calibrator(config, "N1", calibration_frame(days=13))

    diagnostic = fit_d4_side_calibrator(
        config,
        "N1",
        calibration_frame(days=9),
        fold_name="2026-07-21",
    )
    assert diagnostic.evidence.support_passed is False
    assert "NO markets" in diagnostic.evidence.support_failures
    assert "NO UTC days" in diagnostic.evidence.support_failures

    qualification_config = replace(config, qualification_eligible=True)
    with pytest.raises(RuntimeError, match="NO losing markets"):
        fit_d4_side_calibrator(
            qualification_config,
            "N1",
            calibration_frame(no_wins_every=1),
        )


def test_config_rejects_weakened_support_probability_and_economic_gates() -> None:
    config = load_d4_side_calibration_config(config_path())

    with pytest.raises(ValueError, match="support gates weakened"):
        validate_d4_side_calibration_config(
            replace(
                config,
                support_gates=replace(config.support_gates, minimum_calibration_no_markets=99),
            )
        )
    with pytest.raises(ValueError, match="probability gates weakened"):
        validate_d4_side_calibration_config(
            replace(
                config,
                probability_gates=replace(config.probability_gates, selection_uses_economics=True),
            )
        )
    with pytest.raises(ValueError, match="economic gates weakened"):
        validate_d4_side_calibration_config(
            replace(
                config,
                economic_gates=replace(config.economic_gates, minimum_profit_factor=1.0),
            )
        )
