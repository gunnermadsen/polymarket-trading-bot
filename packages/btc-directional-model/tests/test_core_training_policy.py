from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from btc_directional_model import core_training
from btc_directional_model.core_config import (
    EQUAL_TOTAL_PER_MARKET_NORMALIZATION,
    RowWeightScheduleConfig,
    load_core_config,
)
from btc_directional_model.core_features import CORE_ENRICHED_FEATURES
from btc_directional_model.core_training import (
    CANDIDATES,
    EARLY_ENTRY_TRAINING_WEIGHT_MULTIPLIER,
    candidate_rank,
    candidate_spec,
    candidate_training_weights,
    combined_scored_probability_rows,
    configured_candidate_specs,
    development_gate_passed,
    evaluate_core_holdout,
    market_equal_weights,
    qualification_checks,
    row_weight_schedule_payload,
    scored_fold_probability_rows,
)


def repository_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-20260421-20260620.toml"
    )


def early_entry_profile_config() -> Path:
    return (
        Path(__file__).parent.parent
        / "configs"
        / "btc-5m-directional-core-early-entry-calibration-20260421-20260720.toml"
    )


def qualifying_metrics() -> dict[str, float | int]:
    return {
        "accuracy": 0.75,
        "wilson_lower_95": 0.70,
        "balanced_accuracy": 0.75,
        "up_recall": 0.74,
        "down_recall": 0.76,
        "coverage": 0.60,
        "markets": 1_200,
        "expected_calibration_error": 0.03,
    }


def test_zero_same_time_path_uplift_is_noninferior() -> None:
    config = load_core_config(repository_config())
    checks = qualification_checks(
        config,
        qualifying_metrics(),
        {"accuracy_uplift": 0.0},
        {"lower_95": 0.0},
        walk_forward_accuracy=0.75,
    )

    assert all(check["passed"] for check in checks)


def test_negative_same_time_path_uplift_blocks_qualification() -> None:
    config = load_core_config(repository_config())
    checks = qualification_checks(
        config,
        qualifying_metrics(),
        {"accuracy_uplift": -0.001},
        {"lower_95": -0.001},
        walk_forward_accuracy=0.75,
    )

    failed = {check["name"] for check in checks if not check["passed"]}
    assert failed == {
        "same_cohort_accuracy_uplift",
        "hourly_bootstrap_lower_95",
    }


def test_development_gate_enforces_wilson_calibration_and_bootstrap() -> None:
    config = load_core_config(early_entry_profile_config())
    metrics = {
        "accuracy": 0.90,
        "wilson_lower_95": 0.88,
        "balanced_accuracy": 0.90,
        "up_recall": 0.90,
        "down_recall": 0.90,
        "coverage": 0.60,
        "expected_calibration_error": 0.04,
    }
    paired = {"accuracy_uplift": 0.01}
    bootstrap = {"lower_95": 0.0}

    assert development_gate_passed(config, metrics, paired, bootstrap, 5)

    below_wilson = {**metrics, "wilson_lower_95": 0.864999}
    above_ece = {**metrics, "expected_calibration_error": 0.050001}
    assert not development_gate_passed(
        config,
        below_wilson,
        paired,
        bootstrap,
        5,
    )
    assert not development_gate_passed(
        config,
        above_ece,
        paired,
        bootstrap,
        5,
    )
    assert not development_gate_passed(
        config,
        metrics,
        paired,
        {"lower_95": -0.000001},
        5,
    )


def test_holdout_evaluation_rejects_zero_length_contract_before_access(
    tmp_path: Path,
) -> None:
    source = (
        early_entry_profile_config()
        .read_text()
        .replace(
            'independent_holdout_start = "2026-07-21T00:00:00Z"\n',
            "",
        )
        .replace(
            'independent_holdout_end = "2026-08-04T00:00:00Z"\n',
            "",
        )
    )
    path = tmp_path / "package" / "configs" / "core.toml"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    config = load_core_config(path)

    with pytest.raises(RuntimeError, match="positive holdout range"):
        evaluate_core_holdout(config, tmp_path / "missing-freeze")


def test_promotion_candidates_are_four_deploy_compatible_histograms() -> None:
    candidates = configured_candidate_specs(
        load_core_config(early_entry_profile_config())
    )

    assert [candidate.name for candidate in candidates] == [
        "histogram_enriched",
        "histogram_early_weighted",
        "histogram_early_weighted_moderate",
        "histogram_early_90_120",
    ]

    assert all(candidate.family == "histogram" for candidate in candidates)
    assert all(
        candidate.feature_names == tuple(CORE_ENRICHED_FEATURES)
        for candidate in candidates
    )
    assert len(CORE_ENRICHED_FEATURES) == 58


def test_historical_configs_retain_the_original_candidate_matrix() -> None:
    config = load_core_config(repository_config())

    assert config.model.candidate_names == tuple(
        candidate.name for candidate in CANDIDATES
    )
    assert [candidate.name for candidate in configured_candidate_specs(config)] == [
        "logistic_baseline",
        "logistic_enriched",
        "histogram_enriched",
        "histogram_early_weighted",
    ]


def test_candidate_schedules_are_explicit_and_frozen() -> None:
    candidates = configured_candidate_specs(
        load_core_config(early_entry_profile_config())
    )
    schedules = {
        candidate.name: row_weight_schedule_payload(candidate)
        for candidate in candidates
    }

    assert schedules == {
        "histogram_enriched": {
            "start_second": None,
            "end_second_inclusive": None,
            "multiplier": 1.0,
            "normalization": EQUAL_TOTAL_PER_MARKET_NORMALIZATION,
        },
        "histogram_early_weighted": {
            "start_second": 60,
            "end_second_inclusive": 120,
            "multiplier": 3.0,
            "normalization": EQUAL_TOTAL_PER_MARKET_NORMALIZATION,
        },
        "histogram_early_weighted_moderate": {
            "start_second": 60,
            "end_second_inclusive": 120,
            "multiplier": 1.5,
            "normalization": EQUAL_TOTAL_PER_MARKET_NORMALIZATION,
        },
        "histogram_early_90_120": {
            "start_second": 90,
            "end_second_inclusive": 120,
            "multiplier": 2.0,
            "normalization": EQUAL_TOTAL_PER_MARKET_NORMALIZATION,
        },
    }


def test_candidate_weights_follow_the_loaded_profile_schedule(
    tmp_path: Path,
) -> None:
    source = early_entry_profile_config().read_text().replace(
        'candidate = "histogram_early_weighted_moderate"\n'
        "start_second = 60\n"
        "end_second_inclusive = 120\n"
        "multiplier = 1.5",
        'candidate = "histogram_early_weighted_moderate"\n'
        "start_second = 60\n"
        "end_second_inclusive = 120\n"
        "multiplier = 1.75",
    )
    path = tmp_path / "package" / "configs" / "core.toml"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    config = load_core_config(path)
    candidate = next(
        spec
        for spec in configured_candidate_specs(config)
        if spec.name == "histogram_early_weighted_moderate"
    )
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a"],
            "seconds_elapsed": [60, 120, 180],
        }
    )

    weights = candidate_training_weights(frame, candidate)

    assert weights[0] / weights[2] == pytest.approx(1.75)
    assert weights[1] / weights[2] == pytest.approx(1.75)


def test_early_weights_emphasize_timing_with_equal_total_per_market() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "a", "b", "b"],
            "seconds_elapsed": [60, 90, 120, 180, 60, 180],
        }
    )
    candidate = candidate_spec("histogram_early_weighted")

    weights = candidate_training_weights(frame, candidate)

    assert weights[0] / weights[3] == pytest.approx(
        EARLY_ENTRY_TRAINING_WEIGHT_MULTIPLIER
    )
    assert weights[1] / weights[3] == pytest.approx(
        EARLY_ENTRY_TRAINING_WEIGHT_MULTIPLIER
    )
    assert weights[2] / weights[3] == pytest.approx(
        EARLY_ENTRY_TRAINING_WEIGHT_MULTIPLIER
    )
    assert weights[4] / weights[5] == pytest.approx(
        EARLY_ENTRY_TRAINING_WEIGHT_MULTIPLIER
    )
    assert weights[:4].sum() == pytest.approx(weights[4:].sum())


@pytest.mark.parametrize(
    ("candidate_name", "expected_ratios"),
    [
        (
            "histogram_early_weighted_moderate",
            (1.5, 1.5, 1.5, 1.0),
        ),
        (
            "histogram_early_90_120",
            (1.0, 2.0, 2.0, 1.0),
        ),
    ],
)
def test_configured_weight_schedules_preserve_equal_market_totals(
    candidate_name: str,
    expected_ratios: tuple[float, float, float, float],
) -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "a", "b", "b"],
            "seconds_elapsed": [60, 90, 120, 180, 60, 180],
        }
    )

    weights = candidate_training_weights(frame, candidate_spec(candidate_name))
    reference = weights[3]

    assert tuple(weights[index] / reference for index in range(4)) == pytest.approx(
        expected_ratios
    )
    assert weights[:4].sum() == pytest.approx(weights[4:].sum())
    assert weights.mean() == pytest.approx(1.0)


def test_row_weight_schedule_rejects_partial_or_nonpositive_configuration() -> None:
    with pytest.raises(ValueError, match="both be configured"):
        RowWeightScheduleConfig(
            start_second=60,
            end_second_inclusive=None,
            multiplier=2.0,
        )
    with pytest.raises(ValueError, match="positive"):
        RowWeightScheduleConfig(
            start_second=60,
            end_second_inclusive=120,
            multiplier=0.0,
        )


@pytest.mark.parametrize(
    ("source_replacement", "expected_error"),
    [
        (
            ('"histogram_early_90_120"', '"unknown_candidate"'),
            "unknown candidates",
        ),
        (
            ('"histogram_early_90_120",', '"histogram_enriched",'),
            "candidate_names must be unique",
        ),
        (
            ('"histogram_early_90_120"', '"logistic_enriched"'),
            "histogram-only",
        ),
        (
            ("start_second = 90", "start_second = 55"),
            "configured scoring window",
        ),
    ],
)
def test_candidate_profile_rejects_invalid_names_families_and_schedules(
    tmp_path: Path,
    source_replacement: tuple[str, str],
    expected_error: str,
) -> None:
    source = early_entry_profile_config().read_text()
    old, new = source_replacement
    source = source.replace(old, new)
    path = tmp_path / "package" / "configs" / "core.toml"
    path.parent.mkdir(parents=True)
    path.write_text(source)

    with pytest.raises(ValueError, match=expected_error):
        load_core_config(path)


def test_existing_candidates_retain_market_equal_weights() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "b", "b"],
            "seconds_elapsed": [60, 90, 180, 60, 180],
        }
    )
    expected = market_equal_weights(frame)

    for name in (
        "logistic_baseline",
        "logistic_enriched",
        "histogram_enriched",
    ):
        assert np.array_equal(
            candidate_training_weights(frame, candidate_spec(name)),
            expected,
        )


def test_scored_fold_probabilities_preserve_every_validation_checkpoint() -> None:
    validation_start = datetime(2026, 6, 2, tzinfo=UTC)
    rows = []
    for market_index in range(2):
        window_start = validation_start + timedelta(minutes=market_index * 5)
        for second in range(60, 241, 5):
            rows.append(
                {
                    "market_id": f"market-{market_index}",
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "label_up": market_index,
                    "binance_sign_up": market_index,
                }
            )
    validation = pl.DataFrame(rows)
    probabilities = np.linspace(0.1, 0.9, validation.height)
    candidate = candidate_spec("histogram_early_90_120")

    scored = scored_fold_probability_rows(
        validation,
        probabilities,
        candidate,
        fold_index=3,
    )
    combined = combined_scored_probability_rows(
        [{"scored_probability_rows": scored.to_dicts()}]
    )

    assert scored.height == 2 * 37
    assert scored.group_by("market_id").len()["len"].to_list() == [37, 37]
    assert set(scored["seconds_elapsed"]) == set(range(60, 241, 5))
    assert scored["window_start"].min() >= validation_start
    assert scored["window_start"].max() < validation_start + timedelta(minutes=10)
    assert set(scored["candidate"]) == {"histogram_early_90_120"}
    assert set(scored["fold_index"]) == {3}
    assert combined.height == scored.height
    assert combined["probability_up"].sort().to_numpy() == pytest.approx(probabilities)


def test_candidate_rank_never_weakens_accuracy_for_earlier_timing() -> None:
    later_more_accurate = ranking_result(
        passed=True,
        accuracy=0.90,
        early_coverage=0.10,
        median_seconds=180.0,
    )
    earlier_less_accurate = ranking_result(
        passed=True,
        accuracy=0.89,
        early_coverage=0.50,
        median_seconds=90.0,
    )
    earlier_blocked = ranking_result(
        passed=False,
        accuracy=0.99,
        early_coverage=0.90,
        median_seconds=60.0,
    )

    assert candidate_rank(later_more_accurate) > candidate_rank(
        earlier_less_accurate
    )
    assert candidate_rank(later_more_accurate) > candidate_rank(earlier_blocked)


def test_candidate_task_loads_feature_cache_once_for_all_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        compute=SimpleNamespace(threads_per_fit=1),
        split=SimpleNamespace(validation_windows=(("a", "b"), ("b", "c"), ("c", "d"))),
    )
    frame = pl.DataFrame({"market_id": ["market"]})
    feature_loads = 0

    monkeypatch.setattr(core_training, "load_core_config", lambda _: config)
    monkeypatch.setattr(core_training, "configure_native_thread_limits", lambda _: None)

    def load_frame(_config: object, cohort: str) -> pl.DataFrame:
        nonlocal feature_loads
        feature_loads += 1
        assert cohort == "pre_holdout"
        return frame

    monkeypatch.setattr(core_training, "load_core_feature_frame", load_frame)
    monkeypatch.setattr(
        core_training,
        "evaluate_fold",
        lambda loaded, spec, fold_index, _config, **_: {
            "candidate": spec.name,
            "fold_index": fold_index,
            "same_frame": loaded is frame,
        },
    )

    results = core_training.evaluate_candidate_task(
        Path("unused.toml"),
        "histogram_early_weighted",
    )

    assert feature_loads == 1
    assert [result["fold_index"] for result in results] == [0, 1, 2]
    assert all(result["same_frame"] for result in results)


def ranking_result(
    *,
    passed: bool,
    accuracy: float,
    early_coverage: float,
    median_seconds: float,
) -> dict[str, object]:
    return {
        "passed_development": passed,
        "bootstrap": {"lower_95": 0.01},
        "paired": {"accuracy_uplift": 0.01},
        "out_of_fold": {
            "wilson_lower_95": accuracy - 0.02,
            "balanced_accuracy": accuracy,
            "accuracy": accuracy,
            "coverage": 0.60,
        },
        "timing": {
            "early_entry_coverage": early_coverage,
            "median_first_crossing_seconds": median_seconds,
            "p90_first_crossing_seconds": median_seconds + 30,
        },
        "feature_count": len(CORE_ENRICHED_FEATURES),
    }
