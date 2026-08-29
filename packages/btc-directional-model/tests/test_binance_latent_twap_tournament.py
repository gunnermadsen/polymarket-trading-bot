from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from btc_directional_model.binance_latent_twap_tournament import CHALLENGERS, load_config

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PACKAGE_ROOT
    / "configs"
    / "btc-5m-binance-latent-twap-context-20260607-20260828.toml"
)


def test_frozen_tournament_roster_and_data_boundaries() -> None:
    config = load_config(CONFIG)

    assert config.freeze_at == datetime(2026, 8, 28, tzinfo=UTC)
    assert config.historical_start == datetime(2026, 6, 7, tzinfo=UTC)
    assert config.calibration_start == datetime(2026, 8, 1, tzinfo=UTC)
    assert config.development_start == datetime(2026, 8, 14, tzinfo=UTC)
    assert config.prospective_start == config.freeze_at
    assert len(config.folds) == 7
    assert tuple(candidate.name for candidate in CHALLENGERS) == (
        "twap_regime_kline_residual",
        "twap_regime_open_interest_residual",
        "twap_regime_kline_open_interest_residual",
        "twap_regime_l2_residual_research",
    )
    assert tuple(candidate.base_candidate for candidate in CHALLENGERS) == (
        "twap_regime_switching",
    ) * 4
    assert [candidate.name for candidate in CHALLENGERS if candidate.research_only] == [
        "twap_regime_l2_residual_research"
    ]
    assert config.raw["training"]["strictly_training_only"] is True
    assert config.raw["training"]["live_capital_allowed"] is False
    assert config.raw["binance"]["open_interest_max_age_seconds"] == 300
    assert config.raw["binance"]["l2_max_age_seconds"] == 2
