from __future__ import annotations

from pathlib import Path

import pytest

from btc_directional_model.config import load_config

VALID_CONFIG = """
[data]
range_start = "2026-04-21T00:00:00Z"
range_end = "2026-05-21T00:00:00Z"
sample_interval_seconds = 5
min_seconds_after_open = 60
min_seconds_before_close = 60
strict_final_price_audit = false

[split]
train_fraction = 0.60
calibration_fraction = 0.20
test_fraction = 0.20

[model]
c_candidates = [0.01, 0.1]
confidence_min = 0.50
confidence_max = 0.90
confidence_step = 0.01
target_accuracy = 0.65
target_wilson_lower = 0.60
minimum_calibration_markets = 300
minimum_test_markets = 500
maximum_train_test_accuracy_gap = 0.05
random_seed = 20260726

[evaluation]
holdout_is_independent = true
holdout_note = "fresh chronological holdout"

[paths]
source_data = "data/source"
feature_data = "data/features.parquet"
runs = "runs"
artifacts = "artifacts"
"""


def write_config(tmp_path: Path, contents: str = VALID_CONFIG) -> Path:
    path = tmp_path / "package" / "configs" / "model.toml"
    path.parent.mkdir(parents=True)
    path.write_text(contents)
    return path


def test_load_config_resolves_package_local_paths(tmp_path: Path) -> None:
    path = write_config(tmp_path)
    config = load_config(path)

    assert config.package_root == tmp_path / "package"
    assert config.paths.source_data == tmp_path / "package" / "data" / "source"
    assert config.model.c_candidates == (0.01, 0.1)
    assert config.data.range_start.isoformat() == "2026-04-21T00:00:00+00:00"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "sample_interval_seconds = 5",
            "sample_interval_seconds = 7",
            "sample cadence",
        ),
        (
            "confidence_min = 0.50",
            "confidence_min = 0.40",
            "confidence bounds",
        ),
        (
            "c_candidates = [0.01, 0.1]",
            "c_candidates = []",
            "c_candidates",
        ),
        (
            'range_end = "2026-05-21T00:00:00Z"',
            'range_end = "2026-05-21T00:00:01Z"',
            "UTC days",
        ),
    ],
)
def test_invalid_contract_is_rejected(tmp_path: Path, old: str, new: str, message: str) -> None:
    path = write_config(tmp_path, VALID_CONFIG.replace(old, new))

    with pytest.raises(ValueError, match=message):
        load_config(path)
