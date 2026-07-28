from __future__ import annotations

from pathlib import Path

import pytest

from kraken_ml.config import load_config, parse_utc


def test_loads_frozen_benchmark_config_and_stable_fingerprint(
    config_path: Path, tmp_path: Path
) -> None:
    config = load_config(config_path)
    copied_path = tmp_path / "benchmark.toml"
    copied_path.write_bytes(config_path.read_bytes())
    copied = load_config(copied_path)

    assert config.dataset.source == "parquet_lake"
    assert config.dataset.symbol == "PF_XBTUSD"
    assert config.dataset.interval_seconds == 900
    assert config.dataset.horizon_bars == 4
    assert config.validation.purge_bars >= config.dataset.horizon_bars + 1
    assert config.compute.reserve_cores == 2
    assert 1 <= config.compute.available_cores <= config.compute.max_parallel_fits
    assert len(config.validation.folds) == 6
    assert config.fingerprint == copied.fingerprint
    assert len(config.fingerprint) == 64
    assert "source_path" not in config.canonical_payload()


def test_parse_utc_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone"):
        parse_utc("2026-01-01T00:00:00")


def test_cpu_budget_reserves_host_cores_and_honors_fit_cap(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(config_path)

    monkeypatch.setattr("kraken_ml.config.os.cpu_count", lambda: 12)
    assert config.compute.available_cores == 10
    monkeypatch.setattr("kraken_ml.config.os.cpu_count", lambda: 4)
    assert config.compute.available_cores == 2


def test_config_rejects_purge_shorter_than_next_open_dependency(
    config_path: Path, tmp_path: Path
) -> None:
    text = config_path.read_text(encoding="utf-8").replace("purge_bars = 5", "purge_bars = 4")
    invalid_path = tmp_path / "invalid-purge.toml"
    invalid_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="purge_bars must be at least 5"):
        load_config(invalid_path)
