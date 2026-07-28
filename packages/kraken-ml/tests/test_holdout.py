from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from kraken_ml.config import load_config
from kraken_ml.training import (
    _claim_holdout,
    _holdout_identity,
    _verify_frozen_environment,
)


def test_holdout_seal_is_global_to_market_and_time_window(
    config_path: Path, tmp_path: Path
) -> None:
    config = load_config(config_path)
    isolated = replace(
        config,
        artifacts=replace(config.artifacts, root=tmp_path / "artifacts"),
    )
    first_run = tmp_path / "runs" / "first"
    second_run = tmp_path / "runs" / "second"

    run_marker, global_marker = _claim_holdout(isolated, first_run, "first")

    assert run_marker.exists()
    assert global_marker.name == f"{_holdout_identity(isolated)}.json"
    with pytest.raises(RuntimeError, match="already opened"):
        _claim_holdout(isolated, second_run, "second")
    assert not (second_run / "holdout-access.json").exists()


def test_frozen_environment_rejects_runtime_dependency_drift(
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(config_path)
    lock = {
        "code_lineage": {
            "package_source_sha256": "source",
            "requirements_lock_sha256": "requirements",
        },
        "runtime_versions": {
            "python": "3.13.7",
            "scipy": "1.18.0",
        },
    }
    monkeypatch.setattr(
        "kraken_ml.training._package_lineage",
        lambda: {
            "package_source_sha256": "source",
            "requirements_lock_sha256": "requirements",
        },
    )
    monkeypatch.setattr(
        "kraken_ml.training._runtime_metadata",
        lambda _config: {"python": "3.13.7", "scipy": "1.18.0"},
    )
    _verify_frozen_environment(config, lock)

    monkeypatch.setattr(
        "kraken_ml.training._runtime_metadata",
        lambda _config: {"python": "3.13.7", "scipy": "1.19.0"},
    )
    with pytest.raises(RuntimeError, match="runtime version changed for scipy"):
        _verify_frozen_environment(config, lock)
