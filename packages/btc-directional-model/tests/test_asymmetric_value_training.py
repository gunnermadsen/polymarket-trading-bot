from __future__ import annotations

from btc_directional_model.asymmetric_value_data import (
    EARLY_CAUSAL_ORACLE_FEATURES,
    POLYMARKET_VALUE_FEATURES,
)
from btc_directional_model.asymmetric_value_training import (
    ASYMMETRIC_VALUE_CANDIDATES,
    CORE_CONTROL,
    CORE_ORACLE_PRICE,
    CORE_PRICE,
    MODEL_SELECTION_ELIGIBLE,
    ORACLE_MATCHED_CORE_PRICE_CONTROL,
    asymmetric_value_feature_sets,
)


def test_price_features_are_challengers_not_core_control() -> None:
    feature_sets = asymmetric_value_feature_sets()

    assert set(POLYMARKET_VALUE_FEATURES).isdisjoint(feature_sets[CORE_CONTROL])
    assert set(POLYMARKET_VALUE_FEATURES).issubset(feature_sets[CORE_PRICE])


def test_early_oracle_contract_excludes_unproven_boundary_features() -> None:
    feature_sets = asymmetric_value_feature_sets()

    assert "oracle_gap_to_opening_boundary_bps" not in EARLY_CAUSAL_ORACLE_FEATURES
    assert "oracle_boundary_binance_path_agreement" not in (
        EARLY_CAUSAL_ORACLE_FEATURES
    )
    assert set(EARLY_CAUSAL_ORACLE_FEATURES).issubset(
        feature_sets[CORE_ORACLE_PRICE]
    )


def test_oracle_ablation_has_a_same_cohort_core_price_control() -> None:
    feature_sets = asymmetric_value_feature_sets()

    assert ORACLE_MATCHED_CORE_PRICE_CONTROL in ASYMMETRIC_VALUE_CANDIDATES
    assert ORACLE_MATCHED_CORE_PRICE_CONTROL not in MODEL_SELECTION_ELIGIBLE
    assert feature_sets[ORACLE_MATCHED_CORE_PRICE_CONTROL] == feature_sets[CORE_PRICE]
    assert set(EARLY_CAUSAL_ORACLE_FEATURES).isdisjoint(
        feature_sets[ORACLE_MATCHED_CORE_PRICE_CONTROL]
    )
