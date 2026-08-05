from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from btc_directional_model.asymmetric_value_config import load_asymmetric_value_config
from btc_directional_model.asymmetric_value_data import (
    EARLY_CAUSAL_ORACLE_FEATURES,
    POLYMARKET_VALUE_FEATURES,
    _execution_evidence_configs,
    _load_retained_side_execution_rows,
    attach_asymmetric_value_features,
    attach_early_causal_oracle_features,
    execution_grid_coverage,
    select_asymmetric_prediction_grid,
)


def _config():
    return load_asymmetric_value_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-directional-asymmetric-value-one-second-20260414-20260802.toml"
    )


def _frames(*, price_label: int = 1, age_seconds: float = 0.5):
    observed = datetime(2026, 7, 16, 0, 0, 5, tzinfo=UTC)
    start = observed - timedelta(seconds=5)
    features = pl.DataFrame(
        {
            "market_id": ["m"],
            "window_start": [start],
            "window_end": [start + timedelta(minutes=5)],
            "observed_at": [observed],
            "seconds_elapsed": [5],
            "label_up": [1],
            "seconds_elapsed_scaled": [5 / 300],
        }
    )
    received = observed - timedelta(seconds=age_seconds)
    prices = pl.DataFrame(
        {
            "market_id": ["m"],
            "window_start": [start],
            "window_end": [start + timedelta(minutes=5)],
            "observed_at": [observed],
            "seconds_elapsed": [5],
            "label_up": [price_label],
            "fee_rate": [0.02],
            "yes_received_at": [received],
            "yes_best_ask": [0.20],
            "yes_ask_vwap_5": [0.21],
            "yes_ask_depth": [25.0],
            "no_received_at": [received],
            "no_best_ask": [0.79],
            "no_ask_vwap_5": [0.80],
            "no_ask_depth": [20.0],
            "source_artifact_id": ["artifact"],
            "source_schema_version": ["btc5m-book-250ms-v1"],
            "quality_flags": [0],
            "source_book_regime": ["high"],
        }
    )
    return features, prices


def _raw_oracle_market(
    start: datetime,
    *,
    market_id: str,
    seconds: list[int],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "market_id": [market_id] * len(seconds),
            "window_start": [start] * len(seconds),
            "window_end": [start + timedelta(minutes=5)] * len(seconds),
            "official_outcome": ["Up"] * len(seconds),
            "label_up": [1] * len(seconds),
            "opening_boundary": [100_000.0] * len(seconds),
            "final_price": [100_100.0] * len(seconds),
            "observed_at": [start + timedelta(seconds=value) for value in seconds],
            "seconds_elapsed": seconds,
            "btc_open": [100_000.0 + value for value in seconds],
            "btc_high": [100_001.0 + value for value in seconds],
            "btc_low": [99_999.0 + value for value in seconds],
            "btc_close": [100_000.5 + value for value in seconds],
            "btc_base_volume": [1.0] * len(seconds),
            "btc_quote_volume": [100_000.0] * len(seconds),
            "trade_count": [10] * len(seconds),
            "btc_taker_buy_base_volume": [0.6] * len(seconds),
            "btc_taker_buy_quote_volume": [60_000.0] * len(seconds),
        }
    )


def test_value_features_use_fee_and_reserve_adjusted_costs() -> None:
    config = _config()
    features, prices = _frames()

    result = attach_asymmetric_value_features(features, prices, config)

    expected_fee = 0.02 * 0.21 * (1.0 - 0.21)
    assert result["yes_execution_cost_per_share"].item() == pytest.approx(
        0.21 + expected_fee
    )
    assert result["yes_cost_per_share"].item() == pytest.approx(
        0.21 + expected_fee + 0.01
    )
    assert set(POLYMARKET_VALUE_FEATURES).issubset(result.columns)


def test_value_features_require_depth_for_maximum_participation() -> None:
    config = _config()
    features, prices = _frames()
    prices = prices.with_columns(pl.lit(19.0).alias("yes_ask_depth"))

    with pytest.raises(RuntimeError, match="no strict executable books"):
        attach_asymmetric_value_features(features, prices, config)


def test_value_feature_join_asserts_label_identity() -> None:
    config = _config()
    features, prices = _frames(price_label=0)

    with pytest.raises(RuntimeError, match="labels disagree"):
        attach_asymmetric_value_features(features, prices, config)


def test_value_features_fail_closed_on_stale_books() -> None:
    config = _config()
    features, prices = _frames(age_seconds=2.1)

    with pytest.raises(RuntimeError, match="no strict executable books"):
        attach_asymmetric_value_features(features, prices, config)


def test_execution_evidence_uses_isolated_exact_cadences() -> None:
    config = _config()
    early, later = _execution_evidence_configs(config, scope="development")

    assert early.context_seconds == tuple(range(1, 60))
    assert later.context_seconds == tuple(range(60, 241, 5))
    assert early.range_end == config.evaluation.start
    assert "development" in early.output_dir.parts
    assert early.snapshot_schema_versions == ("btc5m-book-250ms-v1",)


def test_execution_query_is_bounded_to_canonical_pmxt_artifacts() -> None:
    query = (Path(__file__).parents[1] / "sql/btc-execution-evidence.sql").read_text()

    assert "polymarket.btc_market_execution_snapshots" in query
    assert "snapshot.artifact_id = %(artifact_id)s" in query
    assert "strict_both_side_eligible" in query
    assert "up_provider_received_at <= candidate.observed_at" in query


def test_execution_grid_coverage_counts_only_exact_core_market_keys(
    tmp_path: Path,
) -> None:
    config = replace(_config(), price_cache=tmp_path)
    early, later = _execution_evidence_configs(config, scope="development")
    early.output_dir.mkdir(parents=True)
    later.output_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "market_id": ["core-a", "unrelated"],
            "seconds_elapsed": [1, 1],
            "strict_both_side_eligible": [True, True],
            "yes_ask_depth": [20.0, 20.0],
            "no_ask_depth": [20.0, 20.0],
        }
    ).write_parquet(early.output_dir / "rows.parquet")
    pl.DataFrame(
        {
            "market_id": ["core-a"],
            "seconds_elapsed": [60],
            "strict_both_side_eligible": [True],
            "yes_ask_depth": [19.0],
            "no_ask_depth": [20.0],
        }
    ).write_parquet(later.output_dir / "rows.parquet")
    core = pl.DataFrame({"market_id": ["core-a", "core-b"]})

    coverage = execution_grid_coverage(
        config,
        scope="development",
        core=core,
    )

    assert coverage["expected_rows"] == 2 * len(config.price_seconds)
    assert coverage["retained_rows"] == 2
    assert coverage["strict_rows"] == 1


def test_prediction_grid_selects_exact_hybrid_cadence_without_filling_nulls() -> None:
    config = _config()
    start = datetime(2026, 7, 16, tzinfo=UTC)
    seconds = list(range(1, 241))
    core = pl.DataFrame(
        {
            "market_id": ["m"] * len(seconds),
            "window_start": [start] * len(seconds),
            "observed_at": [start + timedelta(seconds=value) for value in seconds],
            "seconds_elapsed": seconds,
            "causal_long_horizon_feature": [
                None if value < 60 else float(value) for value in seconds
            ],
        }
    )

    selected = select_asymmetric_prediction_grid(core, config)

    assert selected["seconds_elapsed"].to_list() == list(config.prediction_seconds)
    assert selected.height == 96
    assert selected.filter(pl.col("seconds_elapsed") < 60)[
        "causal_long_horizon_feature"
    ].null_count() == 59
    assert 61 not in selected["seconds_elapsed"]
    assert 65 in selected["seconds_elapsed"]


def test_prediction_grid_rejects_an_incomplete_core_market() -> None:
    config = _config()
    start = datetime(2026, 7, 16, tzinfo=UTC)
    seconds = [
        value for value in range(1, 241) if value != 55
    ]
    core = pl.DataFrame(
        {
            "market_id": ["m"] * len(seconds),
            "window_start": [start] * len(seconds),
            "observed_at": [start + timedelta(seconds=value) for value in seconds],
            "seconds_elapsed": seconds,
        }
    )

    with pytest.raises(RuntimeError, match="complete 96-point"):
        select_asymmetric_prediction_grid(core, config)


def test_prediction_grid_rejects_subsecond_timestamp_misalignment() -> None:
    config = _config()
    start = datetime(2026, 7, 16, tzinfo=UTC)
    seconds = list(range(1, 241))
    observed = [start + timedelta(seconds=value) for value in seconds]
    observed[0] += timedelta(milliseconds=500)
    core = pl.DataFrame(
        {
            "market_id": ["m"] * len(seconds),
            "window_start": [start] * len(seconds),
            "observed_at": observed,
            "seconds_elapsed": seconds,
        }
    )

    with pytest.raises(RuntimeError, match="timestamps are not aligned"):
        select_asymmetric_prediction_grid(core, config)


def test_early_oracle_features_are_derived_on_raw_one_second_rows(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 7, 16, tzinfo=UTC)
    seconds = list(range(300))
    raw = _raw_oracle_market(start, market_id="m", seconds=seconds)
    oracle_seconds = [-10, 0, 4, *range(10, 300, 10)]
    oracle = pl.DataFrame(
        {
            "oracle_price": [100_000.0 + value for value in oracle_seconds],
            "oracle_source_timestamp": [
                start + timedelta(seconds=value) for value in oracle_seconds
            ],
            "oracle_block_timestamp": [
                start + timedelta(seconds=value) for value in oracle_seconds
            ],
            "oracle_phase_id": [1] * len(oracle_seconds),
            "oracle_round_id": list(range(len(oracle_seconds))),
            "oracle_block_number": list(range(1_000, 1_000 + len(oracle_seconds))),
            "oracle_log_index": [0] * len(oracle_seconds),
        }
    )
    raw.reverse().write_parquet(tmp_path / "2026-07-16.parquet")
    oracle.write_parquet(tmp_path / "oracle-2026-07-16.parquet")
    config = _config()
    core = raw.filter(
        pl.col("seconds_elapsed").is_in(list(config.prediction_seconds))
    ).select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
    )

    result = attach_early_causal_oracle_features(core, tmp_path)
    at_five = result.filter(pl.col("seconds_elapsed") == 5)

    assert result.height == 96
    assert result["seconds_elapsed"].to_list() == list(config.prediction_seconds)
    assert at_five.select(
        pl.all_horizontal(
            pl.col(feature).is_finite() for feature in EARLY_CAUSAL_ORACLE_FEATURES
        )
    ).item()
    assert result.filter(pl.col("early_oracle_eligible"))[
        "oracle_age_seconds"
    ].min() >= 2
    assert result["oracle_age_seconds"].max() <= 11
    assert at_five["oracle_block_timestamp"].item() == start


def test_early_oracle_features_reject_any_gapped_market(tmp_path: Path) -> None:
    start = datetime(2026, 7, 16, tzinfo=UTC)
    complete = _raw_oracle_market(
        start,
        market_id="complete",
        seconds=list(range(300)),
    )
    gapped_start = start + timedelta(minutes=5)
    gapped = _raw_oracle_market(
        gapped_start,
        market_id="gapped",
        seconds=[value for value in range(300) if value != 42],
    )
    pl.concat((complete, gapped), how="vertical_relaxed").write_parquet(
        tmp_path / "2026-07-16.parquet"
    )
    oracle = pl.DataFrame(
        {
            "oracle_price": [100_000.0],
            "oracle_source_timestamp": [start - timedelta(seconds=1)],
            "oracle_block_timestamp": [start - timedelta(seconds=1)],
            "oracle_phase_id": [1],
            "oracle_round_id": [1],
            "oracle_block_number": [1_000],
            "oracle_log_index": [0],
        }
    )
    oracle.write_parquet(tmp_path / "oracle-2026-07-16.parquet")
    core = pl.concat(
        (
            complete.filter(pl.col("seconds_elapsed") == 5),
            gapped.filter(pl.col("seconds_elapsed") == 5),
        ),
        how="vertical_relaxed",
    ).select(
        "market_id", "window_start", "observed_at", "seconds_elapsed", "label_up"
    )

    with pytest.raises(RuntimeError, match="complete 0-299 one-second grid"):
        attach_early_causal_oracle_features(core, tmp_path)


def test_retained_reference_keeps_each_fresh_side_independently(
    tmp_path: Path,
) -> None:
    timestamp = datetime(2026, 7, 20, tzinfo=UTC)
    pl.DataFrame(
        {
            "market_id": ["m"],
            "window_start": [timestamp],
            "observed_at": [timestamp + timedelta(seconds=60)],
            "seconds_elapsed": [60],
            "fee_rate": [0.02],
            "up_side_fresh": [True],
            "down_side_fresh": [False],
            "up_ask_vwap_5": [0.20],
            "up_ask_depth": [25.0],
            "down_ask_vwap_5": [0.80],
            "down_ask_depth": [25.0],
        }
    ).write_parquet(tmp_path / "2026-07-20.parquet")

    result = _load_retained_side_execution_rows(tmp_path)

    assert result["yes_execution_cost_per_share"].item() == pytest.approx(
        0.2032
    )
    assert result["yes_ask_vwap_5"].item() == pytest.approx(0.20)
    assert result["no_execution_cost_per_share"].item() is None
    assert result["no_ask_vwap_5"].item() is None
