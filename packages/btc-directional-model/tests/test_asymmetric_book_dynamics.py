from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from btc_directional_model.asymmetric_book_dynamics import (
    BOOK_DELTA_BASES,
    BOOK_DYNAMICS_FEATURES,
    BOOK_DYNAMICS_HORIZONS_SECONDS,
    EXPECTED_BOOK_DYNAMICS_FEATURE_COUNT,
    attach_causal_book_dynamics,
    book_dynamics_content_sha256,
    book_dynamics_evidence,
    book_dynamics_key_sha256,
    book_dynamics_schema_sha256,
)


def test_book_dynamics_use_exact_keys_and_neutralize_missing_horizons() -> None:
    frame = _frame(
        [
            _row("market-a", 1, selected_side="YES", yes_cost=0.20, no_cost=0.80),
            _row("market-a", 2, selected_side="YES", yes_cost=0.23, no_cost=0.77),
            _row("market-a", 4, selected_side="YES", yes_cost=0.30, no_cost=0.70),
            _row("market-a", 6, selected_side="YES", yes_cost=0.27, no_cost=0.73),
            _row("market-a", 16, selected_side="YES", yes_cost=0.35, no_cost=0.65),
        ]
    ).reverse()

    observed = attach_causal_book_dynamics(frame)
    second_2 = observed.filter(pl.col("seconds_elapsed") == 2).row(0, named=True)
    second_4 = observed.filter(pl.col("seconds_elapsed") == 4).row(0, named=True)
    second_6 = observed.filter(pl.col("seconds_elapsed") == 6).row(0, named=True)
    second_16 = observed.filter(pl.col("seconds_elapsed") == 16).row(0, named=True)

    assert second_2["pm_book_horizon_1s_mature"] is True
    assert second_2["pm_selected_cost_delta_1s"] == pytest.approx(0.03)
    assert second_4["pm_book_horizon_1s_mature"] is False
    assert second_4["pm_selected_cost_delta_1s"] == 0.0
    assert second_6["pm_book_horizon_5s_mature"] is True
    assert second_6["pm_selected_cost_delta_5s"] == pytest.approx(0.07)
    assert second_16["pm_book_horizon_15s_mature"] is True
    assert second_16["pm_selected_cost_delta_15s"] == pytest.approx(0.15)

    expected_order = frame["seconds_elapsed"].to_list()
    assert observed["seconds_elapsed"].to_list() == expected_order


def test_current_side_orients_both_current_and_prior_values() -> None:
    frame = _frame(
        [
            _row("market-a", 1, selected_side="YES", yes_cost=0.21, no_cost=0.80),
            _row("market-a", 6, selected_side="NO", yes_cost=0.38, no_cost=0.65),
        ]
    )

    second_6 = (
        attach_causal_book_dynamics(frame)
        .filter(pl.col("seconds_elapsed") == 6)
        .row(0, named=True)
    )

    assert second_6["pm_book_horizon_5s_mature"] is True
    assert second_6["pm_selected_cost_delta_5s"] == pytest.approx(0.65 - 0.80)
    assert second_6["pm_opposite_cost_delta_5s"] == pytest.approx(0.38 - 0.21)
    assert second_6["pm_selected_vwap_slippage_delta_5s"] == pytest.approx(
        0.65 / 20.0 - 0.80 / 20.0
    )
    assert second_6["pm_opposite_vwap_slippage_delta_5s"] == pytest.approx(
        0.38 / 20.0 - 0.21 / 20.0
    )
    assert second_6["pm_selected_depth_log_delta_5s"] == pytest.approx(3.65 - 3.80)
    assert second_6["pm_opposite_depth_log_delta_5s"] == pytest.approx(3.38 - 3.21)
    assert second_6["pm_cost_overround_delta_5s"] == pytest.approx(0.02)
    prior_imbalance = (3.21 - 3.80) / (3.21 + 3.80)
    current_imbalance = (3.38 - 3.65) / (3.38 + 3.65)
    assert second_6["pm_depth_imbalance_delta_5s"] == pytest.approx(
        current_imbalance - prior_imbalance
    )


def test_all_eight_deltas_are_emitted_for_each_horizon() -> None:
    observed = attach_causal_book_dynamics(
        _frame(
            [
                _row("market-a", 1, selected_side="YES", yes_cost=0.20, no_cost=0.80),
                _row("market-a", 16, selected_side="YES", yes_cost=0.30, no_cost=0.70),
            ]
        )
    )

    assert len(BOOK_DELTA_BASES) == 8
    assert len(BOOK_DYNAMICS_FEATURES) == EXPECTED_BOOK_DYNAMICS_FEATURE_COUNT == 40
    for horizon in BOOK_DYNAMICS_HORIZONS_SECONDS:
        assert f"pm_book_horizon_{horizon}s_mature" in observed.columns
        for base in BOOK_DELTA_BASES:
            assert f"pm_{base.name}_delta_{horizon}s" in observed.columns


def test_source_rejects_duplicate_or_misaligned_keys() -> None:
    row = _row("market-a", 1, selected_side="YES", yes_cost=0.20, no_cost=0.80)
    with pytest.raises(ValueError, match="duplicate decision keys"):
        attach_causal_book_dynamics(_frame([row, row]))

    misaligned = {**row, "observed_at": row["observed_at"] + timedelta(milliseconds=1)}
    with pytest.raises(ValueError, match="market-relative seconds"):
        attach_causal_book_dynamics(_frame([misaligned]))


def test_source_rejects_future_receipt_or_stale_book() -> None:
    row = _row("market-a", 1, selected_side="YES", yes_cost=0.20, no_cost=0.80)
    future = {
        **row,
        "yes_received_at": row["observed_at"] + timedelta(milliseconds=1),
    }
    with pytest.raises(ValueError, match="non-causal or stale|receipt timestamp"):
        attach_causal_book_dynamics(_frame([future]))

    stale = {**row, "pm_no_book_age_seconds": 2.001}
    with pytest.raises(ValueError, match="non-causal or stale"):
        attach_causal_book_dynamics(_frame([stale]))


def test_feature_digests_are_order_invariant_and_content_sensitive() -> None:
    enriched = attach_causal_book_dynamics(
        _frame(
            [
                _row("market-a", 1, selected_side="YES", yes_cost=0.20, no_cost=0.80),
                _row("market-a", 2, selected_side="YES", yes_cost=0.25, no_cost=0.75),
                _row("market-b", 1, selected_side="NO", yes_cost=0.78, no_cost=0.22),
            ]
        )
    )
    reversed_frame = enriched.reverse()
    changed = enriched.with_columns(
        pl.when(pl.col("market_id") == "market-a")
        .then(pl.col("pm_yes_cost_per_share") + 0.001)
        .otherwise(pl.col("pm_yes_cost_per_share"))
        .alias("pm_yes_cost_per_share")
    )

    assert len(book_dynamics_schema_sha256()) == 64
    assert book_dynamics_key_sha256(enriched) == book_dynamics_key_sha256(reversed_frame)
    assert book_dynamics_content_sha256(enriched) == book_dynamics_content_sha256(
        reversed_frame
    )
    assert book_dynamics_content_sha256(enriched) != book_dynamics_content_sha256(changed)
    evidence = book_dynamics_evidence(enriched)
    assert evidence["schema_sha256"] == book_dynamics_schema_sha256()
    assert evidence["feature_count"] == 40
    assert evidence["rows"] == 3
    assert evidence["markets"] == 2
    one_second = evidence["horizon_maturity"]["1s"]
    assert one_second["available_rows"] == 1
    assert one_second["raw_rate"] == pytest.approx(1 / 3)
    assert one_second["causally_mature_rows"] == 1
    assert one_second["conditional_available_rows"] == 1
    assert one_second["conditional_available_rate"] == 1.0
    assert evidence["horizon_maturity"]["5s"]["conditional_available_rate"] is None


def _frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("window_start").cast(pl.Datetime("us", "UTC")),
        pl.col("observed_at").cast(pl.Datetime("us", "UTC")),
        pl.col("yes_received_at").cast(pl.Datetime("us", "UTC")),
        pl.col("no_received_at").cast(pl.Datetime("us", "UTC")),
    )


def _row(
    market_id: str,
    second: int,
    *,
    selected_side: str,
    yes_cost: float,
    no_cost: float,
) -> dict[str, object]:
    window_start = datetime(2026, 7, 21, tzinfo=UTC)
    observed_at = window_start + timedelta(seconds=second)
    yes_age = 0.20
    no_age = 0.35
    yes_slippage = yes_cost / 20.0
    no_slippage = no_cost / 20.0
    yes_depth_log = 3.0 + yes_cost
    no_depth_log = 3.0 + no_cost
    return {
        "market_id": market_id,
        "window_start": window_start,
        "observed_at": observed_at,
        "seconds_elapsed": second,
        "selected_side": selected_side,
        "yes_received_at": observed_at - timedelta(seconds=yes_age),
        "no_received_at": observed_at - timedelta(seconds=no_age),
        "pm_yes_cost_per_share": yes_cost,
        "pm_no_cost_per_share": no_cost,
        "pm_yes_cost_logit": yes_cost - 0.5,
        "pm_no_cost_logit": no_cost - 0.5,
        "pm_cost_overround": yes_cost + no_cost - 1.0,
        "pm_yes_minus_no_cost": yes_cost - no_cost,
        "pm_yes_vwap_slippage": yes_slippage,
        "pm_no_vwap_slippage": no_slippage,
        "pm_yes_depth_log": yes_depth_log,
        "pm_no_depth_log": no_depth_log,
        "pm_depth_imbalance": (yes_depth_log - no_depth_log)
        / (yes_depth_log + no_depth_log),
        "pm_yes_book_age_seconds": yes_age,
        "pm_no_book_age_seconds": no_age,
    }
