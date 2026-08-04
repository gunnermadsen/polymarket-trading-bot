from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from btc_directional_model.early_value_config import (
    EarlyValueConfig,
    EvidenceWindow,
    validate_early_value_config,
)


def _config(tmp_path: Path) -> EarlyValueConfig:
    required = [tmp_path / name for name in ("core.toml", "l2", "candles.parquet", "books.sql", "model.json")]
    required[1].mkdir()
    for path in required:
        if path != required[1]:
            path.touch()
    dates = [datetime(2026, 4, 14, tzinfo=UTC), datetime(2026, 7, 6, tzinfo=UTC), datetime(2026, 7, 13, tzinfo=UTC), datetime(2026, 7, 20, tzinfo=UTC), datetime(2026, 8, 2, tzinfo=UTC)]
    return EarlyValueConfig(
        source_path=tmp_path / "benchmark.toml", package_root=tmp_path,
        core_config=required[0], l2_source=required[1], candle_source=required[2],
        refprice_source=tmp_path / "refprice.parquet", price_source_sql=required[3],
        price_cache=tmp_path / "prices", runs=tmp_path / "runs", champion_model=required[4],
        fit=EvidenceWindow(dates[0], dates[1]), calibration=EvidenceWindow(dates[1], dates[2]),
        policy=EvidenceWindow(dates[2], dates[3]), evaluation=EvidenceWindow(dates[3], dates[4]),
        prediction_seconds=tuple(range(5, 241, 5)),
        price_seconds=(*range(1, 60), *range(60, 241, 5)),
        calibration_bands=((5, 15), (15, 30), (30, 45), (45, 60), (60, 90), (90, 120), (120, 180), (180, 241)),
        quantity=5.0, book_freshness_seconds=2, minimum_edge_per_share=0.015,
        cheap_price_min=0.20, cheap_price_max=0.30, confidence_reference=0.89,
        bootstrap_resamples=1000, random_seed=20260804,
    )


def test_early_value_contract_accepts_required_timing(tmp_path: Path) -> None:
    validate_early_value_config(_config(tmp_path))


def test_early_value_contract_rejects_missing_pre60_predictions(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="5 through 240"):
        validate_early_value_config(EarlyValueConfig(**{**config.__dict__, "prediction_seconds": tuple(range(60, 241, 5))}))
