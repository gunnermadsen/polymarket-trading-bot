from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from btc_directional_model.chainlink_oi_features import CHAINLINK_CANDLE_FEATURES
from btc_directional_model.core_execution import realized_pnl, taker_fee_per_share
from btc_directional_model.core_features import CORE_BOUNDARY_ENRICHED_FEATURES
from btc_directional_model.spot_l2_chainlink_benchmark import (
    POINT_KEYS,
    _assert_disjoint_markets,
    _assert_prediction_keys,
    _excluded_decision_intervals,
    _execution_coverage,
    _first_crossings,
    _is_advancement_authority,
    _market_equal_weights,
    _range,
)
from btc_directional_model.spot_l2_chainlink_config import (
    CANDLES,
    COMBINED,
    CONTROL,
    SPOT_L2,
    load_spot_l2_chainlink_config,
)
from btc_directional_model.spot_l2_chainlink_evaluation import (
    EconomicLedgerSpec,
    build_economic_ledger,
    economic_ledger_metrics,
    evaluate_advancement_gates,
    pair_economic_ledgers,
    paired_economic_metrics,
)
from btc_directional_model.spot_l2_chainlink_extract import (
    L2_SOURCE_SCHEMA,
    _l2_summary,
)
from btc_directional_model.spot_l2_chainlink_features import (
    L2_AUDIT_COLUMNS,
    L2_CAUSAL_AUDIT_COLUMNS,
    L2_FEATURES,
    L2_SOURCE_FEATURE_COLUMNS,
    L2Normalizer,
    join_closed_chainlink_candles,
    join_qualified_l2,
)

DECISION = datetime(2026, 7, 20, 0, 1, tzinfo=UTC)


def _core(*, observed_at: datetime = DECISION) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "market_id": ["m-1"],
            "window_start": [observed_at - timedelta(seconds=60)],
            "seconds_elapsed": [60],
            "observed_at": [observed_at],
            "btc_close": [100_000.0],
            "label_up": [1],
        }
    )


def _l2(
    *,
    available_at: datetime,
    source_event_timestamp: datetime | None = None,
) -> pl.DataFrame:
    second_start = available_at.replace(microsecond=0)
    row: dict[str, object] = {
        "symbol": "BTCUSDT",
        "second_start": second_start,
        "source_event_timestamp": source_event_timestamp or available_at,
        "provider_received_at": available_at,
        "available_at": available_at,
        "source_update_id": 101,
        "midpoint": 100_000.0,
        "microprice": 100_000.25,
        "spread_bps": 0.2,
        "bid_depth_5": 10.0,
        "ask_depth_5": 12.0,
        "imbalance_5": -0.09,
        "bid_depth_10": 20.0,
        "ask_depth_10": 22.0,
        "imbalance_10": -0.05,
        "bid_depth_20": 30.0,
        "ask_depth_20": 32.0,
        "imbalance_20": -0.03,
        "bid_depth_slope_20": 0.2,
        "ask_depth_slope_20": 0.3,
        "bid_depth_concentration_20": 0.45,
        "ask_depth_concentration_20": 0.55,
        "bid_quote_replenishment_1s": 2.0,
        "ask_quote_replenishment_1s": 3.0,
        "bid_quote_churn_1s": 1.0,
        "ask_quote_churn_1s": 2.0,
    }
    row.update(
        {name: (index + 1) / 100.0 for index, name in enumerate(L2_SOURCE_FEATURE_COLUMNS[20:])}
    )
    return pl.DataFrame([row])


def _candles(
    *,
    latest_close: datetime = DECISION,
    latest_available_at: datetime | None = None,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for minutes_ago in range(70, -1, -1):
        close_timestamp = latest_close - timedelta(minutes=minutes_ago)
        close_price = 100_000.0 * math.exp((70 - minutes_ago) * 0.00001)
        rows.append(
            {
                "open_timestamp": close_timestamp - timedelta(minutes=1),
                "close_timestamp": close_timestamp,
                "available_at": (
                    latest_available_at
                    if close_timestamp == latest_close and latest_available_at is not None
                    else close_timestamp
                ),
                "open_price": close_price - 0.25,
                "high_price": close_price + 1.0,
                "low_price": close_price - 1.0,
                "close_price": close_price,
            }
        )
    return pl.DataFrame(rows)


def _transformed_l2_rows(first: float, second: float) -> pl.DataFrame:
    return pl.DataFrame({name: [first, second] for name in L2_FEATURES})


def test_fixed_profile_schema_counts_and_feature_boundaries() -> None:
    root = Path(__file__).parents[1]
    config = load_spot_l2_chainlink_config(
        root / "configs/btc-5m-directional-spot-l2-chainlink-candles-20260414-20260802.toml"
    )

    assert len(CORE_BOUNDARY_ENRICHED_FEATURES) == 68
    assert len(L2_SOURCE_FEATURE_COLUMNS) == len(L2_FEATURES) == 40
    assert len(CHAINLINK_CANDLE_FEATURES) == 8
    assert {name: len(features) for name, features in config.feature_sets.items()} == {
        CONTROL: 68,
        SPOT_L2: 108,
        CANDLES: 76,
        COMBINED: 116,
    }
    assert config.feature_sets[CONTROL] == tuple(CORE_BOUNDARY_ENRICHED_FEATURES)
    assert len(set(config.feature_sets[COMBINED])) == 116
    assert not set(L2_AUDIT_COLUMNS).intersection(config.feature_sets[COMBINED])
    assert not set(L2_CAUSAL_AUDIT_COLUMNS).intersection(
        config.feature_sets[COMBINED]
    )


def test_execution_decision_grid_uses_unambiguous_integer_series() -> None:
    root = Path(__file__).parents[1]
    query = (root / "sql/btc-spot-l2-execution-stress-source.sql").read_text()

    assert "%(minimum_decision_second)s::integer" in query
    assert "%(maximum_decision_second)s::integer" in query
    assert "%(sample_interval_seconds)s::integer" in query


def test_l2_join_requires_strictly_prior_availability_with_two_second_cap() -> None:
    qualified = join_qualified_l2(
        _core(),
        _l2(available_at=DECISION - timedelta(seconds=2)),
    )

    assert qualified.height == 1
    assert set(L2_FEATURES).issubset(qualified.columns)
    assert not set(L2_AUDIT_COLUMNS).intersection(qualified.columns)
    assert set(L2_CAUSAL_AUDIT_COLUMNS).issubset(qualified.columns)
    assert qualified["spot_l2_source_event_timestamp"].item() == (
        DECISION - timedelta(seconds=2)
    )
    assert qualified["spot_l2_available_at"].item() == (
        DECISION - timedelta(seconds=2)
    )
    assert qualified["spot_l2_availability_age_seconds"].item() == pytest.approx(
        2.0
    )
    assert qualified["spot_l2_state_age_seconds"].item() == pytest.approx(2.0)
    assert qualified["spot_l2_midpoint_to_kline_close_bps"].item() == pytest.approx(0.0)
    assert qualified["spot_l2_microprice_to_midpoint_bps"].item() == pytest.approx(
        math.log(100_000.25 / 100_000.0) * 10_000.0
    )

    exact = join_qualified_l2(_core(), _l2(available_at=DECISION))
    too_old = join_qualified_l2(
        _core(),
        _l2(available_at=DECISION - timedelta(seconds=2, microseconds=1)),
    )
    assert exact.is_empty()
    assert too_old.is_empty()


def test_l2_partition_summary_reads_the_fixed_identity_columns(tmp_path: Path) -> None:
    source = _l2(available_at=DECISION - timedelta(seconds=1))
    path = tmp_path / "l2.parquet"
    pq.write_table(
        pa.Table.from_pylist(source.to_dicts(), schema=L2_SOURCE_SCHEMA),
        path,
    )

    summary = _l2_summary(
        path,
        DECISION.replace(hour=0, minute=0, second=0),
        DECISION.replace(hour=0, minute=0, second=0) + timedelta(days=1),
    )

    assert summary["rows"] == 1
    assert summary["qualified_seconds"] == 1


def test_l2_join_does_not_fill_a_missing_source_interval() -> None:
    source = pl.concat(
        [
            _l2(available_at=DECISION - timedelta(seconds=4)),
            _l2(available_at=DECISION + timedelta(seconds=1)),
        ]
    )

    assert join_qualified_l2(_core(), source).is_empty()


def test_l2_join_rejects_recent_availability_for_a_state_older_than_two_seconds() -> None:
    source = _l2(
        available_at=DECISION - timedelta(seconds=1),
        source_event_timestamp=DECISION - timedelta(seconds=2, microseconds=1),
    )

    assert join_qualified_l2(_core(), source).is_empty()


def test_l2_join_rejects_missing_source_schema() -> None:
    source = _l2(available_at=DECISION - timedelta(seconds=1)).drop("source_update_id")

    with pytest.raises(RuntimeError, match="missing columns: source_update_id"):
        join_qualified_l2(_core(), source)


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("symbol", "ETHUSDT"),
        ("source_update_id", -1),
        ("imbalance_20", 1.01),
        ("bid_depth_10", 5.0),
        ("bid_quote_churn_1s", -0.01),
    ),
)
def test_l2_join_rejects_invalid_or_unqualified_source_values(
    column: str,
    value: object,
) -> None:
    source = _l2(available_at=DECISION - timedelta(seconds=1)).with_columns(
        pl.lit(value).alias(column)
    )

    with pytest.raises(RuntimeError, match="stale, invalid, or unqualified"):
        join_qualified_l2(_core(), source)


def test_l2_join_rejects_duplicate_qualified_states_for_one_second() -> None:
    source = _l2(available_at=DECISION - timedelta(seconds=1))

    with pytest.raises(RuntimeError, match="more than one qualified state per second"):
        join_qualified_l2(_core(), pl.concat([source, source]))


def test_l2_maximum_age_is_not_configurable() -> None:
    with pytest.raises(ValueError, match="fixed at two seconds"):
        join_qualified_l2(
            _core(),
            _l2(available_at=DECISION - timedelta(seconds=1)),
            maximum_age_seconds=3,
        )


def test_l2_normalizer_uses_only_the_supplied_fit_rows() -> None:
    fit = _transformed_l2_rows(1.0, 3.0)
    normalizer = L2Normalizer.fit(fit)
    future = _transformed_l2_rows(1_000.0, 2_000.0)

    normalized_fit = normalizer.transform(fit)
    transformed_future = normalizer.transform(future)

    assert set(normalizer.means) == set(normalizer.scales) == set(L2_FEATURES)
    assert all(normalizer.means[name] == pytest.approx(2.0) for name in L2_FEATURES)
    assert all(normalizer.scales[name] == pytest.approx(1.0) for name in L2_FEATURES)
    assert normalized_fit.select(pl.col(L2_FEATURES[0]).mean()).item() == pytest.approx(0.0)
    assert transformed_future[L2_FEATURES[0]].to_list() == pytest.approx([998.0, 1_998.0])


def test_closed_chainlink_join_ignores_a_candle_closing_exactly_at_decision() -> None:
    source = _candles()
    changed_exact = source.with_columns(
        pl.when(pl.col("close_timestamp") == DECISION)
        .then(pl.col("close_price") * 2.0)
        .otherwise(pl.col("close_price"))
        .alias("close_price"),
        pl.when(pl.col("close_timestamp") == DECISION)
        .then(pl.col("high_price") * 2.0 + 1.0)
        .otherwise(pl.col("high_price"))
        .alias("high_price"),
    )

    original = join_closed_chainlink_candles(_core(), source)
    changed = join_closed_chainlink_candles(_core(), changed_exact)

    assert original.height == changed.height == 1
    assert original.select(CHAINLINK_CANDLE_FEATURES).equals(
        changed.select(CHAINLINK_CANDLE_FEATURES)
    )
    assert "close_timestamp" not in original.columns


def test_closed_chainlink_join_rejects_stale_and_invalid_candles() -> None:
    stale = _candles(latest_close=DECISION - timedelta(minutes=2))
    assert join_closed_chainlink_candles(_core(), stale).is_empty()

    invalid = _candles().with_columns(
        pl.when(pl.col("close_timestamp") == DECISION)
        .then(pl.col("open_timestamp") - timedelta(seconds=1))
        .otherwise(pl.col("open_timestamp"))
        .alias("open_timestamp")
    )
    with pytest.raises(RuntimeError, match="Chainlink candle source contains invalid rows"):
        join_closed_chainlink_candles(_core(), invalid)


def test_closed_chainlink_join_requires_availability_strictly_before_decision() -> None:
    source = _candles(
        latest_close=DECISION - timedelta(seconds=1),
        latest_available_at=DECISION,
    )

    assert join_closed_chainlink_candles(_core(), source).is_empty()


def test_chainlink_maximum_age_is_not_configurable() -> None:
    with pytest.raises(ValueError, match="fixed at sixty seconds"):
        join_closed_chainlink_candles(_core(), _candles(), maximum_age_seconds=61)


def test_market_equal_weights_give_each_market_equal_total_weight() -> None:
    frame = pl.DataFrame({"market_id": ["a", "a", "a", "b", "c", "c"]})
    weights = _market_equal_weights(frame)

    assert weights[:3].sum() == pytest.approx(weights[3])
    assert weights[3] == pytest.approx(weights[4:].sum())
    assert weights.sum() == pytest.approx(3.0)


def test_first_crossing_is_earliest_fixed_threshold_decision_per_market() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "b", "b"],
            "window_start": [DECISION - timedelta(seconds=60)] * 5,
            "seconds_elapsed": [60, 65, 70, 60, 65],
            "observed_at": [DECISION + timedelta(seconds=offset) for offset in (0, 5, 10, 0, 5)],
            "label_up": [0, 0, 0, 0, 0],
        }
    )

    rows = _first_crossings(frame, np.asarray([0.88, 0.10, 0.95, 0.11, 0.95]))

    assert rows.sort("market_id")["seconds_elapsed"].to_list() == [65, 60]
    assert rows.sort("market_id")["predicted_up"].to_list() == [0, 0]
    assert rows.sort("market_id")["confidence"].to_list() == pytest.approx([0.90, 0.89])
    assert rows["correct"].all()


def test_benchmark_splits_are_half_open_and_market_scoped() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["fit-last", "cal-first", "policy-first", "eval-first"],
            "window_start": [
                datetime(2026, 7, 5, 23, 55, tzinfo=UTC),
                datetime(2026, 7, 6, tzinfo=UTC),
                datetime(2026, 7, 13, tzinfo=UTC),
                datetime(2026, 7, 20, tzinfo=UTC),
            ],
        }
    )

    fit = _range(frame, "fit_start", "fit_end")
    calibration = _range(frame, "calibration_start", "calibration_end")
    policy = _range(frame, "policy_start", "policy_end")
    evaluation = _range(frame, "evaluation_start", "evaluation_end")

    assert fit["market_id"].to_list() == ["fit-last"]
    assert calibration["market_id"].to_list() == ["cal-first"]
    assert policy["market_id"].to_list() == ["policy-first"]
    assert evaluation["market_id"].to_list() == ["eval-first"]
    _assert_disjoint_markets(fit, calibration, policy, evaluation)


def test_split_validation_rejects_a_market_crossing_boundaries() -> None:
    left = pl.DataFrame({"market_id": ["same-market"]})
    right = pl.DataFrame({"market_id": ["same-market"]})

    with pytest.raises(RuntimeError, match="crosses benchmark splits"):
        _assert_disjoint_markets(left, right)


def test_prediction_key_validation_enforces_identical_cohorts() -> None:
    expected = pl.DataFrame(
        {
            "market_id": ["m-1"],
            "window_start": [DECISION - timedelta(seconds=60)],
            "seconds_elapsed": [60],
        }
    )
    predictions = expected.with_columns(pl.col("seconds_elapsed").alias("seconds_elapsed"))
    _assert_prediction_keys(predictions, expected.select(*POINT_KEYS), "control", "evaluation")

    mismatched = predictions.with_columns(pl.lit(65).alias("seconds_elapsed"))
    with pytest.raises(RuntimeError, match="prediction keys differ"):
        _assert_prediction_keys(mismatched, expected, "challenger", "evaluation")


def test_advancement_authority_prevents_duplicate_l2_selection() -> None:
    assert _is_advancement_authority("primary_l2", SPOT_L2)
    assert _is_advancement_authority("strict_combined", CANDLES)
    assert _is_advancement_authority("strict_combined", COMBINED)
    assert not _is_advancement_authority("strict_combined", SPOT_L2)
    assert not _is_advancement_authority("primary_l2", CANDLES)


def test_partial_exclusions_are_collapsed_on_the_fixed_decision_grid() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["m-1"] * 4,
            "window_start": [DECISION - timedelta(seconds=60)] * 4,
            "seconds_elapsed": [60, 65, 70, 80],
            "l2": [False, False, True, False],
            "candles": [True] * 4,
            "strict": [False, False, True, False],
        }
    )

    intervals = _excluded_decision_intervals(frame).filter(pl.col("excluded_from") == "primary_l2")

    assert intervals["start_second"].to_list() == [60, 80]
    assert intervals["end_second_inclusive"].to_list() == [65, 80]
    assert intervals["decision_rows"].to_list() == [2, 1]
    assert intervals["end_at_exclusive"].to_list() == [
        DECISION + timedelta(seconds=10),
        DECISION + timedelta(seconds=25),
    ]


def test_execution_coverage_reports_observed_days_instead_of_a_constant() -> None:
    scenarios = (
        "arrival_150ms_depth_80pct",
        "latency_300ms_depth_65pct",
        "latency_600ms_depth_50pct",
    )
    execution = pl.DataFrame(
        {
            "market_id": ["m-1"] * 3,
            "window_start": [DECISION - timedelta(seconds=60)] * 3,
            "seconds_elapsed": [60] * 3,
            "scenario_key": scenarios,
            "snapshot_at": [DECISION + timedelta(milliseconds=250), None, None],
            "strict_both_side_eligible_10": [True, False, False],
            "price_stress_method": ["vwap10_proxy", "vwap10_proxy", "vwap10_exact"],
            "price_stress_exact": [False, False, True],
        }
    )

    coverage = _execution_coverage(execution)

    assert coverage["decision_keys"] == 1
    assert coverage["snapshot_utc_days"] == 1
    assert coverage["strict_10_share_eligible_utc_days"] == 1
    assert coverage["by_scenario"][scenarios[1]]["snapshot_utc_days"] == 0


def test_economics_distinguishes_no_signal_from_missing_execution_evidence() -> None:
    window_start = DECISION - timedelta(seconds=60)
    eligible = pl.DataFrame(
        {
            "market_id": ["missing-book", "no-signal"],
            "window_start": [window_start, window_start],
            "label_up": [1, 0],
        }
    )
    signal = pl.DataFrame(
        {
            "market_id": ["missing-book"],
            "window_start": [window_start],
            "seconds_elapsed": [60],
            "observed_at": [DECISION],
            "label_up": [1],
            "predicted_up": [1],
            "probability_up": [0.95],
            "confidence": [0.95],
            "correct": [True],
        }
    )
    execution = pl.DataFrame(
        {
            "market_id": ["missing-book"],
            "window_start": [window_start],
            "seconds_elapsed": [60],
            "scenario_key": ["proxy"],
            "fee_rate": [0.02],
            "up_ask_vwap_10": [0.60],
            "down_ask_vwap_10": [0.40],
            "strict_both_side_eligible_10": [False],
            "price_stress_method": ["vwap10_proxy"],
            "price_stress_exact": [False],
        }
    )
    spec = EconomicLedgerSpec(scenario_key="proxy")

    control = build_economic_ledger(
        signal,
        execution,
        eligible_markets=eligible,
        profile=CONTROL,
        spec=spec,
    ).sort("market_id")
    candidate = build_economic_ledger(
        signal.head(0),
        execution,
        eligible_markets=eligible,
        profile=SPOT_L2,
        spec=spec,
    ).sort("market_id")

    assert control["signal"].to_list() == [True, False]
    assert control["signal_missing_execution_evidence"].to_list() == [True, False]
    assert control["net_pnl"].to_list() == [None, 0.0]
    metrics = economic_ledger_metrics(control)
    assert metrics["source_eligible_markets"] == 2
    assert metrics["eligible_markets"] == 1
    assert metrics["missing_actual_signal_execution_evidence"] == 1

    paired = pair_economic_ledgers(candidate, control).sort("market_id")
    assert paired["paired_execution_evidence_eligible"].to_list() == [False, True]
    assert paired["net_pnl_delta"].to_list() == [None, 0.0]
    paired_metrics = paired_economic_metrics(paired)
    control_evidence = paired_metrics["actual_signal_execution_evidence"]["control"]
    assert control_evidence["signal_markets"] == 1
    assert control_evidence["paired_executable_signal_markets"] == 0


def test_proxy_execution_scenarios_are_hard_advancement_failures() -> None:
    classification = {
        "aggregate": {
            "metrics": {
                "accuracy": 0.70,
                "expected_calibration_error": 0.01,
            }
        }
    }
    paired = {
        "candidate": {
            "net_expectancy_per_eligible_market": 0.1,
            "worst_trade": -1.0,
            "worst_one_percent_tail": -1.0,
        },
        "control": {
            "net_expectancy_per_eligible_market": 0.1,
            "worst_trade": -1.0,
            "worst_one_percent_tail": -1.0,
        },
        "deltas": {
            "net_expectancy_per_eligible_market": 0.0,
            "profit_factor_improvement": 0.0,
            "trade_coverage_ratio": 1.0,
            "gross_loss_reduction": 0.10,
            "consistent_nonnegative_days": 3,
        },
        "paired_outcomes": {"challenger_win_retention_ratio": 1.0},
    }
    exactness = {
        "arrival_150ms_depth_80pct": False,
        "latency_300ms_depth_65pct": False,
        "latency_600ms_depth_50pct": True,
    }
    scenarios = {
        key: {
            "metrics": {
                "execution_scenario": {
                    "exactly_reproducible": exact,
                    "price_stress_exact": exact,
                    "price_stress_methods": ["vwap10_exact" if exact else "vwap10_proxy"],
                },
                "actual_signal_execution_evidence": {
                    role: {
                        "signal_markets": 1,
                        "paired_executable_signal_markets": 1,
                    }
                    for role in ("candidate", "control")
                },
                "candidate": {"net_expectancy_per_eligible_market": 0.1},
            }
        }
        for key, exact in exactness.items()
    }

    gates = evaluate_advancement_gates(
        candidate_classification=classification,
        control_classification=classification,
        paired_economics=paired,
        stress_scenarios=scenarios,
    )

    assert gates["stress_gate_hard_failed"]
    assert not gates["eligible_for_forward_paper_validation"]
    assert gates["hard_failures"] == [
        "arrival_150ms_depth_80pct is exactly reproducible",
        "latency_300ms_depth_65pct is exactly reproducible",
    ]

    exact_scenario = scenarios["latency_600ms_depth_50pct"]["metrics"]
    exact_scenario["actual_signal_execution_evidence"]["candidate"][
        "paired_executable_signal_markets"
    ] = 0
    missing_signal_evidence = evaluate_advancement_gates(
        candidate_classification=classification,
        control_classification=classification,
        paired_economics=paired,
        stress_scenarios=scenarios,
    )
    assert (
        "latency_600ms_depth_50pct has paired execution evidence for candidate signals"
        in missing_signal_evidence["hard_failures"]
    )


@pytest.mark.parametrize("correct", (False, True))
def test_canonical_realized_pnl_applies_price_shaped_fee_per_share(correct: bool) -> None:
    price = 0.60
    fee_rate = 0.02
    quantity = 5.0
    expected_fee = fee_rate * price * (1.0 - price)

    assert taker_fee_per_share(fee_rate, price) == pytest.approx(expected_fee)
    assert realized_pnl(
        correct=correct,
        execution_price=price,
        fee_rate=fee_rate,
        quantity=quantity,
    ) == pytest.approx(quantity * (float(correct) - price - expected_fee))
