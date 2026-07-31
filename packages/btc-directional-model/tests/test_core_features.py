from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from btc_directional_model.core_features import (
    CORE_BOUNDARY_ENRICHED_FEATURES,
    CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES,
    CORE_BOUNDARY_REVERSAL_FEATURE_SCHEMA_VERSION,
    CORE_BOUNDARY_REVERSAL_FEATURES,
    CORE_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION,
    CORE_MATURE_REVERSAL_FEATURES,
    CORE_MATURE_REVERSAL_ORACLE_FEATURES,
    CORE_MODEL_FEATURES,
    CORE_ORACLE_FEATURES,
    CORE_ORACLE_MODEL_FEATURES,
    CORE_REGIME_REVERSAL_ENRICHED_FEATURES,
    CORE_REGIME_REVERSAL_FEATURE_SCHEMA_VERSION,
    CORE_REGIME_REVERSAL_FEATURES,
    attach_causal_oracle_rounds,
    derive_core_point_in_time_features,
    derive_oracle_point_in_time_features,
    model_feature_groups,
    validate_feature_allowlists,
)


def core_source_frame() -> pl.DataFrame:
    start = datetime(2026, 5, 1, tzinfo=UTC)
    rows = []
    for market_index, market_id in enumerate(("a", "b")):
        boundary = 100_000.0 + market_index * 1_000
        for second in range(300):
            close = boundary + ((second % 40) - 20) * 0.5
            observed_at = start + timedelta(minutes=market_index * 5, seconds=second)
            rows.append(
                {
                    "market_id": market_id,
                    "window_start": start + timedelta(minutes=market_index * 5),
                    "window_end": start + timedelta(minutes=(market_index + 1) * 5),
                    "official_outcome": "up" if market_index else "down",
                    "label_up": market_index,
                    "opening_boundary": boundary,
                    "final_price": boundary + (1 if market_index else -1),
                    "observed_at": observed_at,
                    "seconds_elapsed": second,
                    "btc_open": close - 0.1,
                    "btc_high": close + 0.5,
                    "btc_low": close - 0.5,
                    "btc_close": close,
                    "btc_base_volume": 1.0,
                    "btc_quote_volume": 10_000.0 + second,
                    "trade_count": 10,
                    "btc_taker_buy_base_volume": 0.5,
                    "btc_taker_buy_quote_volume": 5_000.0 + second / 2,
                }
            )
    return pl.DataFrame(rows).sort(["market_id", "seconds_elapsed"])


def mirrored_path_source_frame() -> pl.DataFrame:
    start = datetime(2026, 5, 2, tzinfo=UTC)
    rows = []
    boundary_anchor = 100_000.0
    log_move_per_second = 0.00001
    for market_index, (market_id, direction) in enumerate(
        (("path-up", 1.0), ("path-down", -1.0))
    ):
        opening_boundary = boundary_anchor * math.exp(
            direction * log_move_per_second * 10
        )
        for second in range(300):
            close = boundary_anchor * math.exp(
                direction * log_move_per_second * second
            )
            quote_volume = 10_000.0
            taker_buy_share = 0.75 if direction > 0 else 0.25
            observed_at = start + timedelta(
                minutes=market_index * 5, seconds=second
            )
            rows.append(
                {
                    "market_id": market_id,
                    "window_start": start + timedelta(minutes=market_index * 5),
                    "window_end": start
                    + timedelta(minutes=(market_index + 1) * 5),
                    "official_outcome": "up" if direction > 0 else "down",
                    "label_up": int(direction > 0),
                    "opening_boundary": opening_boundary,
                    "final_price": opening_boundary
                    * (1.001 if direction > 0 else 0.999),
                    "observed_at": observed_at,
                    "seconds_elapsed": second,
                    "btc_open": close,
                    "btc_high": close * math.exp(0.000005),
                    "btc_low": close * math.exp(-0.000005),
                    "btc_close": close,
                    "btc_base_volume": 1.0,
                    "btc_quote_volume": quote_volume,
                    "trade_count": 10,
                    "btc_taker_buy_base_volume": taker_buy_share,
                    "btc_taker_buy_quote_volume": (
                        quote_volume * taker_buy_share
                    ),
                }
            )
    return pl.DataFrame(rows).sort(["market_id", "seconds_elapsed"])


def oracle_rounds_for_core(
    frame: pl.DataFrame,
    *,
    first_offset_seconds: int = -1,
) -> pl.DataFrame:
    minimum = frame["observed_at"].min()
    maximum = frame["observed_at"].max()
    assert minimum is not None
    assert maximum is not None
    first = minimum + timedelta(seconds=first_offset_seconds)
    rows = []
    current = first
    index = 1
    while current <= maximum + timedelta(seconds=1):
        rows.append(
            {
                "oracle_price": 99_900.0 * math.exp(index * 0.000001),
                "oracle_source_timestamp": current,
                "oracle_block_timestamp": current,
                "oracle_phase_id": 3,
                "oracle_round_id": index,
                "oracle_block_number": 50_000_000 + index,
                "oracle_log_index": 0,
            }
        )
        current += timedelta(seconds=1)
        index += 1
    return pl.DataFrame(rows)


def test_enriched_core_features_need_no_book_columns() -> None:
    features = derive_core_point_in_time_features(core_source_frame())

    for allowlist in CORE_MODEL_FEATURES.values():
        assert set(allowlist).issubset(features.columns)
    assert features.filter(pl.col("seconds_elapsed") == 240).height == 2
    assert (
        features.filter(pl.col("seconds_elapsed") == 240)[
            "btc_path_terminal_volatility_z"
        ].null_count()
        == 0
    )


def test_core_features_are_invariant_to_future_price_mutation() -> None:
    original = core_source_frame()
    altered = original.with_columns(
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 120))
        .then(pl.col("btc_close") * 1.5)
        .otherwise(pl.col("btc_close"))
        .alias("btc_close")
    )
    original_features = derive_core_point_in_time_features(original)
    altered_features = derive_core_point_in_time_features(altered)
    allowlist = CORE_MODEL_FEATURES["histogram_enriched"]

    before = original_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 120)
    ).select(allowlist)
    after = altered_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 120)
    ).select(allowlist)

    assert before.equals(after, null_equal=True)


def test_core_lags_and_cross_counts_do_not_cross_markets() -> None:
    features = derive_core_point_in_time_features(core_source_frame())
    first_b = features.filter(
        (pl.col("market_id") == "b") & (pl.col("seconds_elapsed") == 0)
    )

    assert first_b["btc_return_1s_bps"][0] is None
    assert first_b["btc_path_cross_count"][0] == 0


def test_model_features_ignore_cross_venue_opening_basis() -> None:
    original = core_source_frame()
    shifted_boundary = original.with_columns(
        (pl.col("opening_boundary") * 1.02).alias("opening_boundary")
    )
    original_features = derive_core_point_in_time_features(original)
    shifted_features = derive_core_point_in_time_features(shifted_boundary)

    assert original_features.select(CORE_ENRICHED_FEATURES).equals(
        shifted_features.select(CORE_ENRICHED_FEATURES),
        null_equal=True,
    )
    assert not original_features.select(CORE_BOUNDARY_ENRICHED_FEATURES).equals(
        shifted_features.select(CORE_BOUNDARY_ENRICHED_FEATURES),
        null_equal=True,
    )
    assert original_features["binance_sign_up"].equals(
        shifted_features["binance_sign_up"]
    )


def test_boundary_features_are_causal_and_market_local() -> None:
    original = core_source_frame()
    altered = original.with_columns(
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 120))
        .then(pl.col("btc_close") * 1.5)
        .otherwise(pl.col("btc_close"))
        .alias("btc_close")
    )
    original_features = derive_core_point_in_time_features(original)
    altered_features = derive_core_point_in_time_features(altered)

    before = original_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 120)
    ).select(CORE_BOUNDARY_ENRICHED_FEATURES)
    after = altered_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 120)
    ).select(CORE_BOUNDARY_ENRICHED_FEATURES)
    first_b = original_features.filter(
        (pl.col("market_id") == "b") & (pl.col("seconds_elapsed") == 0)
    )

    assert before.equals(after, null_equal=True)
    assert first_b["btc_boundary_cross_count"][0] == 0
    assert first_b["btc_boundary_distance_velocity_5s_bps"][0] is None


def test_boundary_reversal_schema_is_additive_and_allowlisted() -> None:
    features = derive_core_point_in_time_features(core_source_frame())

    assert len(CORE_ENRICHED_FEATURES) == 58
    assert len(CORE_BOUNDARY_ENRICHED_FEATURES) == 68
    assert (
        CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES[
            : len(CORE_BOUNDARY_ENRICHED_FEATURES)
        ]
        == CORE_BOUNDARY_ENRICHED_FEATURES
    )
    assert (
        CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES[
            len(CORE_BOUNDARY_ENRICHED_FEATURES) :
        ]
        == CORE_BOUNDARY_REVERSAL_FEATURES
    )
    assert (
        CORE_MODEL_FEATURES["histogram_boundary_reversal"]
        == CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES
    )
    assert (
        CORE_BOUNDARY_REVERSAL_FEATURE_SCHEMA_VERSION
        == "btc-5m-directional-boundary-reversal-features-v1"
    )
    assert not {
        "label_up",
        "official_outcome",
        "final_price",
        "window_end",
        "opening_boundary",
    }.intersection(CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES)
    validate_feature_allowlists(features)


def test_mature_reversal_schema_is_narrow_boundary_independent_and_allowlisted() -> None:
    original = core_source_frame()
    shifted_boundary = original.with_columns(
        (pl.col("opening_boundary") * 1.02).alias("opening_boundary")
    )
    original_features = derive_core_point_in_time_features(original)
    shifted_features = derive_core_point_in_time_features(shifted_boundary)
    mature = original_features.filter(pl.col("seconds_elapsed") == 60)

    assert len(CORE_MATURE_REVERSAL_ENRICHED_FEATURES) == 71
    assert (
        CORE_MATURE_REVERSAL_ENRICHED_FEATURES[: len(CORE_ENRICHED_FEATURES)]
        == CORE_ENRICHED_FEATURES
    )
    assert (
        CORE_MATURE_REVERSAL_ENRICHED_FEATURES[len(CORE_ENRICHED_FEATURES) :]
        == CORE_MATURE_REVERSAL_FEATURES
    )
    assert CORE_MATURE_REVERSAL_FEATURES == [
        "btc_path_max_favorable_excursion_bps",
        "btc_path_max_adverse_excursion_bps",
        "btc_path_pullback_from_favorable_extreme_bps",
        "btc_path_recovery_from_adverse_extreme_bps",
        "btc_seconds_since_path_high_scaled",
        "btc_seconds_since_path_low_scaled",
        "btc_path_sign_normalized_return_5s_bps",
        "btc_path_sign_normalized_return_15s_bps",
        "btc_path_sign_normalized_return_30s_bps",
        "btc_path_sign_normalized_return_60s_bps",
        "btc_path_sign_normalized_flow_5s",
        "btc_path_sign_normalized_flow_30s",
        "btc_path_sign_normalized_flow_60s",
    ]
    assert (
        CORE_MODEL_FEATURES["histogram_mature_reversal"]
        == CORE_MATURE_REVERSAL_ENRICHED_FEATURES
    )
    assert (
        CORE_MATURE_REVERSAL_FEATURE_SCHEMA_VERSION
        == "btc-5m-directional-mature-reversal-features-v1"
    )
    assert not {
        "label_up",
        "official_outcome",
        "final_price",
        "window_end",
        "opening_boundary",
        "btc_cross_venue_boundary_gap_bps",
        "btc_path_sign_normalized_boundary_gap_bps",
    }.intersection(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
    assert original_features.select(CORE_MATURE_REVERSAL_ENRICHED_FEATURES).equals(
        shifted_features.select(CORE_MATURE_REVERSAL_ENRICHED_FEATURES),
        null_equal=True,
    )
    mature_values = mature.select(CORE_MATURE_REVERSAL_FEATURES)
    assert mature_values.null_count().sum_horizontal()[0] == 0
    assert all(
        math.isfinite(float(value))
        for value in mature_values.row(0)
    )
    validate_feature_allowlists(original_features)


def test_regime_reversal_schema_is_immutable_additive_and_allowlisted() -> None:
    features = derive_core_point_in_time_features(core_source_frame())

    assert len(CORE_REGIME_REVERSAL_ENRICHED_FEATURES) == 77
    assert (
        CORE_REGIME_REVERSAL_ENRICHED_FEATURES[
            : len(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
        ]
        == CORE_MATURE_REVERSAL_ENRICHED_FEATURES
    )
    assert (
        CORE_REGIME_REVERSAL_ENRICHED_FEATURES[
            len(CORE_MATURE_REVERSAL_ENRICHED_FEATURES) :
        ]
        == CORE_REGIME_REVERSAL_FEATURES
    )
    assert CORE_REGIME_REVERSAL_FEATURES == [
        "btc_path_sign_normalized_return_90s_bps",
        "btc_path_sign_normalized_return_120s_bps",
        "btc_path_sign_normalized_flow_90s",
        "btc_path_sign_normalized_flow_120s",
        "btc_realized_volatility_90s_bps",
        "btc_realized_volatility_120s_bps",
    ]
    assert (
        CORE_MODEL_FEATURES["histogram_regime_reversal"]
        == CORE_REGIME_REVERSAL_ENRICHED_FEATURES
    )
    assert (
        CORE_REGIME_REVERSAL_FEATURE_SCHEMA_VERSION
        == "btc-5m-directional-regime-reversal-features-v1"
    )
    assert not {
        "label_up",
        "official_outcome",
        "final_price",
        "window_end",
        "opening_boundary",
        "btc_cross_venue_boundary_gap_bps",
        "btc_path_sign_normalized_boundary_gap_bps",
    }.intersection(CORE_REGIME_REVERSAL_ENRICHED_FEATURES)
    validate_feature_allowlists(features)


def test_regime_reversal_features_are_causal_and_boundary_independent() -> None:
    original = core_source_frame()
    future_mutated = original.with_columns(
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 120))
        .then(pl.col("btc_close") * 1.5)
        .otherwise(pl.col("btc_close"))
        .alias("btc_close"),
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 120))
        .then(pl.col("btc_quote_volume") * 100)
        .otherwise(pl.col("btc_quote_volume"))
        .alias("btc_quote_volume"),
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 120))
        .then(0.0)
        .otherwise(pl.col("btc_taker_buy_quote_volume"))
        .alias("btc_taker_buy_quote_volume"),
    )
    shifted_boundary = original.with_columns(
        (pl.col("opening_boundary") * 1.02).alias("opening_boundary")
    )
    original_features = derive_core_point_in_time_features(original)
    future_mutated_features = derive_core_point_in_time_features(future_mutated)
    shifted_boundary_features = derive_core_point_in_time_features(shifted_boundary)

    original_at_120 = original_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 120)
    ).select(CORE_REGIME_REVERSAL_ENRICHED_FEATURES)
    mutated_at_120 = future_mutated_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 120)
    ).select(CORE_REGIME_REVERSAL_ENRICHED_FEATURES)

    assert original_at_120.equals(mutated_at_120, null_equal=True)
    assert original_features.select(CORE_REGIME_REVERSAL_ENRICHED_FEATURES).equals(
        shifted_boundary_features.select(CORE_REGIME_REVERSAL_ENRICHED_FEATURES),
        null_equal=True,
    )


def test_regime_reversal_features_require_complete_long_horizon_history() -> None:
    features = derive_core_point_in_time_features(core_source_frame()).filter(
        pl.col("market_id") == "a"
    )
    early = features.filter(pl.col("seconds_elapsed") == 60).select(
        CORE_REGIME_REVERSAL_FEATURES
    )
    mature = features.filter(pl.col("seconds_elapsed") == 120).select(
        CORE_REGIME_REVERSAL_FEATURES
    )

    assert early.null_count().sum_horizontal()[0] == len(
        CORE_REGIME_REVERSAL_FEATURES
    )
    assert mature.null_count().sum_horizontal()[0] == 0
    assert all(math.isfinite(float(value)) for value in mature.row(0))


def test_boundary_reversal_features_are_causal_with_early_history_nulls() -> None:
    original = core_source_frame()
    altered = original.with_columns(
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 120))
        .then(pl.col("btc_close") * 1.5)
        .otherwise(pl.col("btc_close"))
        .alias("btc_close"),
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 120))
        .then(pl.col("btc_high") * 1.5)
        .otherwise(pl.col("btc_high"))
        .alias("btc_high"),
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 120))
        .then(pl.col("btc_low") * 0.5)
        .otherwise(pl.col("btc_low"))
        .alias("btc_low"),
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 120))
        .then(pl.col("btc_quote_volume") * 100)
        .otherwise(pl.col("btc_quote_volume"))
        .alias("btc_quote_volume"),
        pl.when((pl.col("market_id") == "a") & (pl.col("seconds_elapsed") > 120))
        .then(0.0)
        .otherwise(pl.col("btc_taker_buy_quote_volume"))
        .alias("btc_taker_buy_quote_volume"),
    )
    original_features = derive_core_point_in_time_features(original)
    altered_features = derive_core_point_in_time_features(altered)

    before = original_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 120)
    ).select(CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES)
    after = altered_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 120)
    ).select(CORE_BOUNDARY_REVERSAL_ENRICHED_FEATURES)
    early = original_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 60)
    )
    mature = original_features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 180)
    )

    assert before.equals(after, null_equal=True)
    assert early["btc_return_90s_bps"][0] is None
    assert early["btc_realized_volatility_90s_bps"][0] is None
    assert early["btc_signed_flow_90s"][0] is None
    assert early["btc_price_acceleration_30_vs_90"][0] is None
    assert mature.select(CORE_BOUNDARY_REVERSAL_FEATURES).null_count().sum_horizontal()[
        0
    ] == 0


def test_path_sign_normalization_is_symmetric_for_mirrored_paths() -> None:
    features = derive_core_point_in_time_features(
        mirrored_path_source_frame()
    ).filter(pl.col("seconds_elapsed") == 180)
    path_up = features.filter(pl.col("market_id") == "path-up")
    path_down = features.filter(pl.col("market_id") == "path-down")

    assert path_up["btc_return_90s_bps"][0] == pytest.approx(
        -path_down["btc_return_90s_bps"][0]
    )
    assert path_up["btc_signed_flow_120s"][0] == pytest.approx(
        -path_down["btc_signed_flow_120s"][0]
    )
    for feature in (
        "btc_path_sign_normalized_return_30s_bps",
        "btc_path_sign_normalized_return_90s_bps",
        "btc_path_sign_normalized_return_180s_bps",
        "btc_path_sign_normalized_flow_30s",
        "btc_path_sign_normalized_flow_120s",
        "btc_path_sign_normalized_boundary_gap_bps",
        "btc_path_sign_normalized_last_boundary_cross_direction",
    ):
        assert path_up[feature][0] == pytest.approx(
            path_down[feature][0], abs=1e-9
        )


def test_oracle_asof_join_never_attaches_a_future_round() -> None:
    source = core_source_frame().filter(pl.col("market_id") == "a")
    start = source["window_start"][0]
    rounds = pl.DataFrame(
        [
            {
                "oracle_price": 100_000.0,
                "oracle_source_timestamp": start - timedelta(seconds=1),
                "oracle_block_timestamp": start - timedelta(seconds=1),
                "oracle_phase_id": 3,
                "oracle_round_id": 1,
                "oracle_block_number": 1,
                "oracle_log_index": 0,
            },
            {
                "oracle_price": 200_000.0,
                "oracle_source_timestamp": start + timedelta(seconds=121),
                "oracle_block_timestamp": start + timedelta(seconds=121),
                "oracle_phase_id": 3,
                "oracle_round_id": 2,
                "oracle_block_number": 2,
                "oracle_log_index": 0,
            },
        ]
    )

    joined = attach_causal_oracle_rounds(source, rounds)

    assert joined.filter(pl.col("seconds_elapsed") == 120)[
        "oracle_price"
    ][0] == 100_000.0
    assert joined.filter(pl.col("seconds_elapsed") == 121)[
        "oracle_price"
    ][0] == 200_000.0


def test_oracle_features_are_causal_market_local_and_formula_exact() -> None:
    source = core_source_frame()
    rounds = oracle_rounds_for_core(source)
    features = derive_oracle_point_in_time_features(
        derive_core_point_in_time_features(
            attach_causal_oracle_rounds(source, rounds)
        )
    )
    row = features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 120)
    )
    lagged = features.filter(
        (pl.col("market_id") == "a") & (pl.col("seconds_elapsed") == 90)
    )
    first_b = features.filter(
        (pl.col("market_id") == "b") & (pl.col("seconds_elapsed") == 0)
    )
    expected_return = math.log(
        row["oracle_price"][0] / lagged["oracle_price"][0]
    ) * 10_000

    assert row["oracle_return_30s_bps"][0] == pytest.approx(
        expected_return,
        abs=1e-10,
    )
    assert row["oracle_model_eligible"][0] is True
    assert first_b["oracle_return_30s_bps"][0] is None
    assert first_b["oracle_update_count_since_open_scaled"][0] == 0.0


def test_oracle_features_at_decision_are_invariant_to_future_rounds() -> None:
    source = core_source_frame().filter(pl.col("market_id") == "a")
    original_rounds = oracle_rounds_for_core(source)
    future_mutated = original_rounds.with_columns(
        pl.when(
            pl.col("oracle_block_timestamp")
            > source["window_start"][0] + timedelta(seconds=120)
        )
        .then(pl.col("oracle_price") * 2.0)
        .otherwise(pl.col("oracle_price"))
        .alias("oracle_price")
    )
    before = derive_oracle_point_in_time_features(
        derive_core_point_in_time_features(
            attach_causal_oracle_rounds(source, original_rounds)
        )
    ).filter(pl.col("seconds_elapsed") == 120)
    after = derive_oracle_point_in_time_features(
        derive_core_point_in_time_features(
            attach_causal_oracle_rounds(source, future_mutated)
        )
    ).filter(pl.col("seconds_elapsed") == 120)

    assert before.select(CORE_ORACLE_FEATURES).equals(
        after.select(CORE_ORACLE_FEATURES),
        null_equal=True,
    )


def test_missing_opening_oracle_anchor_fails_eligibility_closed() -> None:
    source = core_source_frame().filter(pl.col("market_id") == "a")
    rounds = oracle_rounds_for_core(source, first_offset_seconds=1)
    features = derive_oracle_point_in_time_features(
        derive_core_point_in_time_features(
            attach_causal_oracle_rounds(source, rounds)
        )
    )
    decision = features.filter(pl.col("seconds_elapsed") == 120)

    assert decision["oracle_window_open_price"][0] is None
    assert decision["oracle_model_eligible"][0] is False


def test_oracle_allowlist_is_additive_and_excludes_routing_provenance() -> None:
    source = core_source_frame()
    features = derive_oracle_point_in_time_features(
        derive_core_point_in_time_features(
            attach_causal_oracle_rounds(
                source,
                oracle_rounds_for_core(source),
            )
        )
    )

    assert (
        CORE_MATURE_REVERSAL_ORACLE_FEATURES[
            : len(CORE_MATURE_REVERSAL_ENRICHED_FEATURES)
        ]
        == CORE_MATURE_REVERSAL_ENRICHED_FEATURES
    )
    assert (
        CORE_MATURE_REVERSAL_ORACLE_FEATURES[
            len(CORE_MATURE_REVERSAL_ENRICHED_FEATURES) :
        ]
        == CORE_ORACLE_FEATURES
    )
    assert model_feature_groups(include_oracle=False) == CORE_MODEL_FEATURES
    assert (
        model_feature_groups(include_oracle=True)[
            "histogram_mature_reversal_oracle"
        ]
        == CORE_MATURE_REVERSAL_ORACLE_FEATURES
    )
    assert not {
        "oracle_price",
        "oracle_source_timestamp",
        "oracle_block_timestamp",
        "oracle_phase_id",
        "oracle_round_id",
        "oracle_block_number",
        "oracle_log_index",
        "oracle_model_eligible",
    }.intersection(
        CORE_ORACLE_MODEL_FEATURES["histogram_mature_reversal_oracle"]
    )
    validate_feature_allowlists(features, include_oracle=True)
