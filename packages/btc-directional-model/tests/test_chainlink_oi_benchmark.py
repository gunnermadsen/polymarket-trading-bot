from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from btc_directional_model.chainlink_oi_benchmark import (
    _attach_economics,
    _candidate_feature_sets,
    _loss_rejection_metrics,
    _native_champion_market_pairing,
    _validate_common_short_cohort,
)
from btc_directional_model.chainlink_oi_config import (
    CHAINLINK_FULL_CANDIDATE,
    CHAINLINK_FULL_OI_CANDIDATE,
    LONG_HISTORY_CANDLE_CANDIDATE,
)
from btc_directional_model.chainlink_oi_features import (
    BINANCE_OI_FEATURES,
    CHAINLINK_CANDLE_FEATURES,
    CHAINLINK_EXTERNAL_FEATURES,
)
from btc_directional_model.core_features import (
    CORE_BOUNDARY_ENRICHED_FEATURES,
    CORE_ORACLE_FEATURES,
)

START = datetime(2026, 7, 16, tzinfo=UTC)


def _proposal_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "market_id": f"market-{index}",
                "window_start": START + timedelta(minutes=5 * index),
                "seconds_elapsed": 120,
                "label_up": label,
                "predicted_up": predicted,
                "probability_up": probability,
                "confidence": max(probability, 1.0 - probability),
                "policy_selected": True,
                "fee_rate": 0.02,
                "up_ask_vwap_5": 0.9,
                "up_ask_vwap_10": 0.91,
                "down_ask_vwap_5": 0.1,
                "down_ask_vwap_10": 0.11,
            }
            for index, (label, predicted, probability) in enumerate(
                (
                    (1, 1, 0.9),
                    (0, 0, 0.1),
                    (0, 1, 0.9),
                    (1, 0, 0.1),
                )
            )
        ]
    )


def test_feature_ablation_sets_preserve_the_core_and_source_contract() -> None:
    features = _candidate_feature_sets()
    base = {*CORE_BOUNDARY_ENRICHED_FEATURES, *CORE_ORACLE_FEATURES}

    assert set(features[CHAINLINK_FULL_CANDIDATE]) == base | set(CHAINLINK_EXTERNAL_FEATURES)
    assert set(features[CHAINLINK_FULL_OI_CANDIDATE]) == (
        base | set(CHAINLINK_EXTERNAL_FEATURES) | set(BINANCE_OI_FEATURES)
    )
    assert set(features[LONG_HISTORY_CANDLE_CANDIDATE]) == base | set(CHAINLINK_CANDLE_FEATURES)


def test_loss_rejection_distinguishes_saved_losses_from_forgone_wins() -> None:
    control = _attach_economics(_proposal_frame(), "predicted_up", 5.0)
    candidate = control.with_columns(
        pl.Series("policy_selected", [True, False, False, True]),
        pl.Series("correct", [True, True, False, False]),
    )

    result = _loss_rejection_metrics(control, candidate)

    assert result["champion_wins"] == 2
    assert result["champion_losses"] == 2
    assert result["rejected_champion_wins"] == 1
    assert result["rejected_champion_losses"] == 1
    assert result["win_retention_rate"] == 0.5
    assert result["loss_rejection_rate"] == 0.5
    assert result["saved_loss_dollars_from_rejection"] > 0
    assert result["forgone_win_dollars_from_rejection"] > 0


def test_common_short_cohort_rejects_key_drift() -> None:
    left = _proposal_frame().select(
        "market_id",
        "window_start",
        "seconds_elapsed",
    )
    _validate_common_short_cohort(left, left.clone())

    shifted = left.with_columns((pl.col("seconds_elapsed") + 5).alias("seconds_elapsed"))
    try:
        _validate_common_short_cohort(left, shifted)
    except RuntimeError as error:
        assert "rows diverged" in str(error)
    else:
        raise AssertionError("key drift must fail the common-row contract")


def test_native_pairing_counts_no_trade_and_corrected_losses_separately() -> None:
    source = _proposal_frame()
    champion = _attach_economics(source, "predicted_up", 5.0)
    candidate = (
        champion.select("market_id", "correct")
        .filter(pl.col("market_id") != "market-2")
        .with_columns(
            pl.when(pl.col("market_id") == "market-3")
            .then(pl.lit(True))
            .otherwise(pl.col("correct"))
            .alias("correct")
        )
    )
    execution = source.select(
        "market_id",
        "window_start",
        "seconds_elapsed",
        "fee_rate",
        "up_ask_vwap_5",
        "up_ask_vwap_10",
        "down_ask_vwap_5",
        "down_ask_vwap_10",
    )

    result = _native_champion_market_pairing(champion, candidate, execution, 5.0)

    assert result["champion_losses"] == 2
    assert result["champion_losses_rejected_as_no_trade"] == 1
    assert result["champion_losses_corrected_by_direction"] == 1
    assert result["champion_losses_repeated"] == 0
    assert result["champion_loss_no_trade_rate"] == 0.5
    assert result["champion_loss_avoidance_rate"] == 1.0
