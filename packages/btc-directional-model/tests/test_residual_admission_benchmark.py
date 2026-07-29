from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from btc_directional_model.residual_admission_benchmark import (
    RESIDUAL_FEATURES,
    balanced_direction_market_weights,
    compose_control_priority_policy,
    head_opportunity_rows,
    paired_residual_attribution,
    prepare_residual_opportunities,
    residual_advancement_checks,
    select_residual_threshold,
)
from btc_directional_model.residual_admission_config import (
    COMBINED_RESIDUAL_CANDIDATE,
    EARLY_RESIDUAL_CANDIDATE,
    RESCUE_RESIDUAL_CANDIDATE,
    RESIDUAL_CONTROL_CANDIDATE,
    RESIDUAL_PROPOSAL_CANDIDATE,
    RESIDUAL_THRESHOLD_GRID,
    ResidualAdmissionBenchmarkConfig,
    ResidualAdmissionGates,
    ResidualHeadConfig,
    ResidualSelectorConfig,
)


def test_control_crossing_excludes_current_and_later_residual_rows() -> None:
    control, proposal, core = _source_rows(markets=2, seconds=(60, 65, 70))
    control = control.with_columns(
        pl.when(
            (pl.col("market_id") == "market-000")
            & (pl.col("seconds_elapsed") >= 65)
        )
        .then(0.91)
        .otherwise(0.80)
        .alias("confidence")
    )

    prepared = prepare_residual_opportunities(
        control,
        proposal,
        core,
        base_confidence_threshold=0.89,
        agreement_cadences=2,
    )

    crossed = prepared.filter(pl.col("market_id") == "market-000")
    open_market = prepared.filter(pl.col("market_id") == "market-001")
    assert crossed["residual_opportunity"].to_list() == [False, False, False]
    assert open_market["residual_opportunity"].to_list() == [False, True, True]


def test_future_mutation_cannot_change_earlier_opportunity_or_features() -> None:
    control, proposal, core = _source_rows(
        markets=2,
        seconds=(60, 65, 70, 75),
    )
    original = prepare_residual_opportunities(
        control,
        proposal,
        core,
        base_confidence_threshold=0.89,
        agreement_cadences=2,
    )
    mutated = proposal.with_columns(
        pl.when(pl.col("seconds_elapsed") >= 75)
        .then(0.99)
        .otherwise(pl.col("probability_up"))
        .alias("probability_up"),
        pl.when(pl.col("seconds_elapsed") >= 75)
        .then(0.99)
        .otherwise(pl.col("confidence"))
        .alias("confidence"),
    )
    observed = prepare_residual_opportunities(
        control,
        mutated,
        core,
        base_confidence_threshold=0.89,
        agreement_cadences=2,
    )
    columns = [
        "market_id",
        "seconds_elapsed",
        "residual_opportunity",
        *RESIDUAL_FEATURES,
    ]
    assert (
        original.filter(pl.col("seconds_elapsed") <= 70)
        .select(columns)
        .to_dicts()
        == observed.filter(pl.col("seconds_elapsed") <= 70)
        .select(columns)
        .to_dicts()
    )


def test_head_boundary_is_exactly_before_and_at_second_120() -> None:
    control, proposal, core = _source_rows(
        markets=2,
        seconds=(110, 115, 120, 125),
    )
    prepared = prepare_residual_opportunities(
        control,
        proposal,
        core,
        base_confidence_threshold=0.89,
        agreement_cadences=2,
    )
    early = head_opportunity_rows(
        prepared,
        ResidualHeadConfig(
            "early",
            EARLY_RESIDUAL_CANDIDATE,
            60,
            120,
            RESIDUAL_THRESHOLD_GRID,
        ),
    )
    rescue = head_opportunity_rows(
        prepared,
        ResidualHeadConfig(
            "rescue",
            RESCUE_RESIDUAL_CANDIDATE,
            120,
            241,
            RESIDUAL_THRESHOLD_GRID,
        ),
    )
    assert early["seconds_elapsed"].max() == 115
    assert rescue["seconds_elapsed"].min() == 120


def test_same_timestamp_control_has_priority_over_residual() -> None:
    control, proposal, core = _source_rows(markets=1, seconds=(60, 65))
    control = control.with_columns(
        pl.when(pl.col("seconds_elapsed") == 65)
        .then(0.91)
        .otherwise(0.80)
        .alias("confidence")
    )
    prepared = prepare_residual_opportunities(
        control,
        proposal,
        core,
        base_confidence_threshold=0.89,
        agreement_cadences=2,
    )
    scored = prepared.filter(pl.col("seconds_elapsed") == 65).with_columns(
        pl.lit(0.99).alias("residual_q")
    )
    selected = compose_control_priority_policy(
        prepared,
        candidate_name=EARLY_RESIDUAL_CANDIDATE,
        base_confidence_threshold=0.89,
        early=(scored, 0.93),
    )
    assert selected.height == 1
    assert selected["seconds_elapsed"][0] == 65
    assert selected["decision_source"][0] == "control"


def test_market_weights_are_equalized_then_direction_balanced() -> None:
    frame = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "b", "c", "c"],
            "proposal_predicted_up": [0, 0, 0, 1, 1, 1],
        }
    )
    weights = balanced_direction_market_weights(frame)
    assert weights.sum() == pytest.approx(frame.height)
    assert weights[np.array([0, 1, 2])].sum() == pytest.approx(
        weights[np.array([3, 4, 5])].sum()
    )
    assert weights[np.array([0, 1, 2])].sum() / 3 == pytest.approx(weights[0])


def test_attribution_is_disjoint_and_reports_advances_rescues_and_no_loss() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    control = _selected_rows(
        [
            ("a", 120, 1, True, "control"),
            ("b", 180, 0, True, "control"),
        ],
        start,
    )
    candidate = _selected_rows(
        [
            ("a", 120, 1, True, "control"),
            ("b", 90, 0, True, "early"),
            ("c", 150, 1, False, "rescue"),
        ],
        start,
    )
    attribution = paired_residual_attribution(
        control,
        candidate,
        ["a", "b", "c", "d"],
    )
    categories = attribution["categories"]
    assert categories["preserved_control"]["markets"] == 1
    assert categories["earlier_same_direction"]["markets"] == 1
    assert categories["rescued_control_no_trade"]["markets"] == 1
    assert categories["still_no_trade"]["markets"] == 1
    assert attribution["advanced_markets"] == 1
    assert attribution["rescued_markets"] == 1
    assert attribution["lost_control_markets"] == 0
    assert sum(
        category["added_wrong_trades"]
        for category in categories.values()
    ) == 1


def test_threshold_selection_fails_closed_without_a_quality_candidate(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    full_rows = []
    opportunity_rows = []
    for index in range(200):
        market_id = f"market-{index:03d}"
        window_start = start + timedelta(minutes=5 * index)
        label = index % 2
        predicted = 1 - label
        full_rows.append(
            {
                "market_id": market_id,
                "window_start": window_start,
                "observed_at": window_start + timedelta(seconds=90),
                "seconds_elapsed": 90,
                "label_up": label,
                "fold_index": 1,
                "control_probability_up": 0.55,
                "control_confidence": 0.55,
                "control_predicted_up": predicted,
                "control_correct": False,
            }
        )
        opportunity_rows.append(
            {
                **full_rows[-1],
                "proposal_predicted_up": predicted,
                "proposal_correct": False,
                "residual_q": 0.99,
            }
        )
    selection = select_residual_threshold(
        pl.DataFrame(full_rows),
        pl.DataFrame(opportunity_rows),
        head=ResidualHeadConfig(
            "early",
            EARLY_RESIDUAL_CANDIDATE,
            60,
            120,
            RESIDUAL_THRESHOLD_GRID,
        ),
        config=_config(tmp_path),
    )
    assert selection.qualified is False
    assert selection.threshold is None
    assert selection.objective["fallback"] == "control"


def test_early_candidate_requires_residual_accuracy_floor(
    tmp_path: Path,
) -> None:
    gates = replace(_config(tmp_path).gates, require_every_fold=False)
    result = residual_advancement_checks(
        candidate_name=EARLY_RESIDUAL_CANDIDATE,
        control_candidate=RESIDUAL_CONTROL_CANDIDATE,
        metrics=_advancement_metrics(median=120.0),
        control_metrics=_advancement_metrics(median=140.0),
        checkpoints={"120": 0.52},
        control_checkpoints={"120": 0.49},
        attribution=_advancement_attribution(),
        residual=_advancement_residual(accuracy=0.86),
        folds=[],
        control_folds=[],
        gates=gates,
        quantity=5.0,
    )
    check = next(
        item for item in result["checks"]
        if item["name"] == "minimum residual accuracy"
    )
    assert check["passed"] is False
    assert result["benchmark_passed"] is False


def test_candidate_requires_residual_executable_sample_floor(
    tmp_path: Path,
) -> None:
    gates = replace(_config(tmp_path).gates, require_every_fold=False)
    result = residual_advancement_checks(
        candidate_name=EARLY_RESIDUAL_CANDIDATE,
        control_candidate=RESIDUAL_CONTROL_CANDIDATE,
        metrics=_advancement_metrics(median=120.0),
        control_metrics=_advancement_metrics(median=140.0),
        checkpoints={"120": 0.52},
        control_checkpoints={"120": 0.49},
        attribution=_advancement_attribution(),
        residual=_advancement_residual(economic_markets=499),
        folds=[],
        control_folds=[],
        gates=gates,
        quantity=5.0,
    )
    check = next(
        item for item in result["checks"]
        if item["name"] == "minimum residual executable markets"
    )
    assert check["passed"] is False
    assert result["benchmark_passed"] is False


def test_candidate_requires_threshold_qualification_in_every_fold(
    tmp_path: Path,
) -> None:
    gates = _config(tmp_path).gates
    control_fold = _advancement_fold(
        fold_index=2,
        selection_qualified=True,
        median=140.0,
        control=True,
    )
    candidate_fold = _advancement_fold(
        fold_index=2,
        selection_qualified=False,
        median=120.0,
        control=False,
    )
    result = residual_advancement_checks(
        candidate_name=EARLY_RESIDUAL_CANDIDATE,
        control_candidate=RESIDUAL_CONTROL_CANDIDATE,
        metrics=_advancement_metrics(median=120.0),
        control_metrics=_advancement_metrics(median=140.0),
        checkpoints={"120": 0.52},
        control_checkpoints={"120": 0.49},
        attribution=_advancement_attribution(),
        residual=_advancement_residual(),
        folds=[
            {**candidate_fold, "fold_index": fold_index}
            for fold_index in range(2, 7)
        ],
        control_folds=[
            {**control_fold, "fold_index": fold_index}
            for fold_index in range(2, 7)
        ],
        gates=gates,
        quantity=5.0,
    )
    check = next(
        item for item in result["checks"]
        if item["name"] == "all advancement gates pass every evaluation fold"
    )
    assert check["observed"] == 0
    assert check["passed"] is False
    assert result["benchmark_passed"] is False


def _source_rows(
    *,
    markets: int,
    seconds: tuple[int, ...],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    probability_rows = []
    core_rows = []
    for market_index in range(markets):
        market_id = f"market-{market_index:03d}"
        window_start = start + timedelta(minutes=5 * market_index)
        label = market_index % 2
        predicted = label
        probability = 0.80 if predicted else 0.20
        for position, second in enumerate(seconds):
            probability_rows.append(
                {
                    "market_id": market_id,
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "label_up": label,
                    "fold_index": 0,
                    "probability_up": probability,
                    "confidence": 0.80,
                    "predicted_up": predicted,
                    "correct": True,
                }
            )
            sign = 1.0 if predicted else -1.0
            core_rows.append(
                {
                    "market_id": market_id,
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "btc_cross_venue_boundary_gap_bps": sign * (2 + position),
                    "btc_window_open_cross_venue_basis_bps": sign,
                    "btc_boundary_terminal_volatility_z": sign * 0.5,
                    "btc_boundary_cross_count": 1.0,
                    "btc_seconds_since_boundary_cross": float(second - seconds[0]),
                    "btc_fraction_time_boundary_positive": (
                        0.75 if predicted else 0.25
                    ),
                    "btc_fraction_time_boundary_negative": (
                        0.25 if predicted else 0.75
                    ),
                    "btc_boundary_distance_velocity_5s_bps": sign,
                    "btc_boundary_momentum_alignment_5s": sign,
                }
            )
    control = pl.DataFrame(probability_rows).with_columns(
        pl.lit("histogram_enriched").alias("candidate")
    )
    proposal = pl.DataFrame(probability_rows).with_columns(
        pl.lit("histogram_boundary_reversal").alias("candidate")
    )
    return control, proposal, pl.DataFrame(core_rows)


def _selected_rows(
    rows: list[tuple[str, int, int, bool, str]],
    start: datetime,
) -> pl.DataFrame:
    payload = []
    for index, (market_id, second, predicted, correct, source) in enumerate(rows):
        window_start = start + timedelta(minutes=5 * index)
        payload.append(
            {
                "candidate": "candidate",
                "market_id": market_id,
                "window_start": window_start,
                "observed_at": window_start + timedelta(seconds=second),
                "seconds_elapsed": second,
                "label_up": predicted if correct else 1 - predicted,
                "fold_index": 2,
                "probability_up": 0.9 if predicted else 0.1,
                "confidence": 0.9,
                "predicted_up": predicted,
                "correct": correct,
                "decision_source": source,
                "residual_q": 0.9 if source != "control" else None,
                "model_eligible": True,
                "policy_selected": True,
            }
        )
    return pl.DataFrame(payload)


def _advancement_metrics(*, median: float) -> dict[str, object]:
    return {
        "markets": 700,
        "eligible_markets": 1_000,
        "coverage": 0.7,
        "accuracy": 0.88,
        "balanced_accuracy": 0.88,
        "up_recall": 0.88,
        "down_recall": 0.88,
        "wilson_lower_95": 0.87,
        "expected_calibration_error": 0.03,
        "no_trade_rate": 0.3,
        "median_seconds_elapsed": median,
        "execution": _advancement_execution(600),
    }


def _advancement_execution(economic_markets: int) -> dict[str, object]:
    return {
        "economic_markets": economic_markets,
        "mean_direct_edge_per_share": 0.02,
        "realized_net_expectancy_per_trade": 0.10,
    }


def _advancement_residual(
    *,
    accuracy: float = 0.88,
    economic_markets: int = 600,
) -> dict[str, object]:
    return {
        "markets": 600,
        "accuracy": accuracy,
        "expected_calibration_error": 0.03,
        "execution": _advancement_execution(economic_markets),
        "hourly_net_bootstrap": {"lower_95": 0.01},
    }


def _advancement_attribution() -> dict[str, object]:
    return {
        "advanced_markets": 550,
        "median_advancement_seconds": 20.0,
        "lost_control_markets": 0,
        "rescued_markets": 500,
    }


def _advancement_fold(
    *,
    fold_index: int,
    selection_qualified: bool,
    median: float,
    control: bool,
) -> dict[str, object]:
    return {
        "fold_index": fold_index,
        "metrics": _advancement_metrics(median=median),
        "checkpoints": {"120": 0.49 if control else 0.52},
        "residual_cohort": _advancement_residual(),
        "paired_attribution": (
            None if control else _advancement_attribution()
        ),
        "selection_qualified": selection_qualified,
    }


def _config(tmp_path: Path) -> ResidualAdmissionBenchmarkConfig:
    return ResidualAdmissionBenchmarkConfig(
        source_path=tmp_path / "config.toml",
        package_root=tmp_path,
        profile="residual_admission",
        probability_manifest=tmp_path / "manifest.json",
        core_config=tmp_path / "core.toml",
        control_candidate=RESIDUAL_CONTROL_CANDIDATE,
        proposal_candidate=RESIDUAL_PROPOSAL_CANDIDATE,
        early_head=ResidualHeadConfig(
            "early",
            EARLY_RESIDUAL_CANDIDATE,
            60,
            120,
            RESIDUAL_THRESHOLD_GRID,
        ),
        rescue_head=ResidualHeadConfig(
            "rescue",
            RESCUE_RESIDUAL_CANDIDATE,
            120,
            241,
            RESIDUAL_THRESHOLD_GRID,
        ),
        combined_candidate=COMBINED_RESIDUAL_CANDIDATE,
        evaluation_folds=(2, 3, 4, 5, 6),
        evaluation_note="Consumed development evidence.",
        evaluation_is_independent=False,
        quantity=5.0,
        selector=ResidualSelectorConfig(
            regularization_c=1.0,
            minimum_calibration_rows_per_direction=500,
            minimum_calibration_markets_per_direction=100,
            agreement_cadences=2,
            base_confidence_threshold=0.89,
            random_seed=20260728,
        ),
        gates=ResidualAdmissionGates(
            minimum_accuracy=0.874,
            minimum_balanced_accuracy=0.874,
            minimum_direction_recall=0.874,
            minimum_wilson_lower_95=0.865,
            maximum_expected_calibration_error=0.05,
            minimum_selected_markets=500,
            minimum_executable_markets=500,
            maximum_accuracy_regression=0.0,
            maximum_balanced_accuracy_regression=0.0,
            maximum_direction_recall_regression=0.0,
            maximum_median_entry_second=125.0,
            minimum_median_entry_improvement_seconds=5.0,
            minimum_decisions_by_120_uplift=0.02,
            minimum_early_residual_markets=500,
            minimum_median_advancement_seconds=10.0,
            minimum_no_trade_reduction=0.02,
            minimum_rescued_markets=500,
            minimum_residual_accuracy=0.874,
            minimum_mean_direct_edge_per_share=0.0,
            minimum_realized_net_per_share=0.0,
            require_every_fold=True,
            required_evaluation_folds=5,
        ),
        execution_evidence=tmp_path / "execution",
        runs=tmp_path / "runs",
    )
