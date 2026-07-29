from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from btc_directional_model.book_residual import (
    CALIBRATION_BANDS,
    RAW_DIRECTIONS,
    CalibrationCell,
    DirectionTimeCalibrator,
    ResidualBookModel,
)
from btc_directional_model.book_residual_benchmark import (
    _join_oof_strict_rows,
    _score_candidate_pair,
    _validate_oof_training_provenance,
)


class FixedCoreBundle:
    def __init__(self, probabilities: tuple[float, ...]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float64)

    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        assert frame.height == len(self.probabilities)
        return self.probabilities.copy()


def calibrator(*, slope: float, intercept: float) -> DirectionTimeCalibrator:
    return DirectionTimeCalibrator(
        tuple(
            CalibrationCell(
                band=band.name,
                raw_direction=direction,
                slope=slope,
                intercept=intercept,
                converged=True,
                optimizer_status=0,
                optimizer_message="test",
                iterations=1,
                function_evaluations=1,
                rows=10,
                markets=10,
                positives=5,
                identity_l2_strength=1.0,
                objective=0.5,
                weighted_log_loss=0.5,
                penalty=0.0,
            )
            for band in CALIBRATION_BANDS
            for direction in RAW_DIRECTIONS
        )
    )


def core_frame() -> pl.DataFrame:
    start = datetime(2026, 7, 18, tzinfo=UTC)
    return pl.DataFrame(
        {
            "market_id": [
                "strict-market-one",
                "fallback-market",
                "strict-market-two",
            ],
            "window_start": [start, start, start],
            "observed_at": [start, start, start],
            "seconds_elapsed": [60, 60, 60],
            "label_up": [1, 0, 1],
            "binance_sign_up": [1, 0, 1],
        }
    )


def out_of_order_strict_rows() -> pl.DataFrame:
    rows = pl.concat(
        [core_frame().tail(1), core_frame().head(1)],
        how="vertical",
    )
    return rows.with_columns(
        pl.lit(True).alias("model_eligible"),
        pl.lit(0.80).alias("book_up_mid"),
        pl.lit(0.20).alias("book_down_mid"),
        pl.lit(0.82).alias("book_up_ask_vwap_10"),
        pl.lit(0.22).alias("book_down_ask_vwap_10"),
        pl.lit(0.30).alias("book_up_imbalance"),
        pl.lit(-0.20).alias("book_down_imbalance"),
        pl.lit(2.1).alias("book_up_log_bid_depth"),
        pl.lit(1.7).alias("book_down_log_bid_depth"),
        pl.lit(1.8).alias("book_up_log_ask_depth"),
        pl.lit(2.0).alias("book_down_log_ask_depth"),
        pl.lit(0.02).alias("book_up_spread"),
        pl.lit(0.03).alias("book_down_spread"),
        pl.lit(0.01).alias("book_up_mid_delta_5s"),
        pl.lit(-0.01).alias("book_down_mid_delta_5s"),
        pl.lit(0.02).alias("book_up_imbalance_delta_5s"),
        pl.lit(-0.01).alias("book_down_imbalance_delta_5s"),
    )


def test_candidate_pair_preserves_calibrated_core_on_fallback_rows() -> None:
    config = SimpleNamespace(
        benchmark=SimpleNamespace(
            control_candidate="core",
            strict_book_candidate="residual",
        )
    )
    model = ResidualBookModel(
        gamma=1.0,
        beta=(0.0,) * 7,
        feature_scales=(1.0,) * 7,
        l2_strength=1.0,
    )

    scored, route = _score_candidate_pair(
        core_frame(),
        out_of_order_strict_rows(),
        FixedCoreBundle((0.60, 0.40, 0.55)),
        model,
        calibrator(slope=1.0, intercept=0.0),
        calibrator(slope=0.5, intercept=0.2),
        config,
    )

    core_probability = scored["core"]["probability_up"].to_numpy()
    residual_probability = scored["residual"]["probability_up"].to_numpy()
    assert residual_probability[0] != core_probability[0]
    assert residual_probability[1] == core_probability[1]
    assert residual_probability[2] != core_probability[2]
    assert route["strict_mask"].tolist() == [True, False, True]
    assert scored["residual"]["prediction_route"].to_list() == [
        "book_residual",
        "universal_btc_core_fallback",
        "book_residual",
    ]


def test_oof_join_rejects_label_disagreement() -> None:
    strict = core_frame().head(1)
    oof = strict.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
    ).with_columns(pl.lit(0).alias("label_up"))

    with pytest.raises(RuntimeError, match="labels disagree"):
        _join_oof_strict_rows(strict, oof)


def test_oof_join_requires_exact_unique_keys_and_overlap() -> None:
    strict = core_frame().head(1)
    oof = strict.select(
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "label_up",
    ).with_columns(
        pl.lit(0.61).alias("core_probability_up"),
        pl.lit(3).alias("fold_index"),
    )

    joined = _join_oof_strict_rows(strict, oof)
    assert joined.height == 1
    assert joined["core_probability_up"].item() == 0.61
    assert joined["fold_index"].item() == 3

    with pytest.raises(pl.exceptions.ComputeError):
        _join_oof_strict_rows(strict, pl.concat([oof, oof]))

    with pytest.raises(RuntimeError, match="do not overlap"):
        _join_oof_strict_rows(
            strict,
            oof.with_columns(pl.lit("different-market").alias("market_id")),
        )


def test_oof_provenance_requires_past_only_reported_fold_ranges(
    tmp_path: Path,
) -> None:
    def fold(
        index: int,
        *,
        fit_end: str,
        calibration_start: str,
        calibration_end: str,
        policy_start: str,
        policy_end: str,
        validation_start: str,
        validation_end: str,
    ) -> dict:
        return {
            "fold_index": index,
            "candidate": "core",
            "fit_range_start": "2026-01-01T00:00:00+00:00",
            "fit_range_end": fit_end,
            "calibration_range_start": calibration_start,
            "calibration_range_end": calibration_end,
            "policy_range_start": policy_start,
            "policy_range_end": policy_end,
            "validation_range_start": validation_start,
            "validation_range_end": validation_end,
            "eligible_markets": 1,
        }

    folds = [
        fold(
            0,
            fit_end="2026-01-02T00:00:00+00:00",
            calibration_start="2026-01-03T00:00:00+00:00",
            calibration_end="2026-01-04T00:00:00+00:00",
            policy_start="2026-01-05T00:00:00+00:00",
            policy_end="2026-01-06T00:00:00+00:00",
            validation_start="2026-01-07T00:00:00+00:00",
            validation_end="2026-01-08T00:00:00+00:00",
        ),
        fold(
            1,
            fit_end="2026-01-03T00:00:00+00:00",
            calibration_start="2026-01-04T00:00:00+00:00",
            calibration_end="2026-01-05T00:00:00+00:00",
            policy_start="2026-01-06T00:00:00+00:00",
            policy_end="2026-01-07T00:00:00+00:00",
            validation_start="2026-01-08T00:00:00+00:00",
            validation_end="2026-01-09T00:00:00+00:00",
        ),
    ]
    benchmark_path = tmp_path / "benchmark.json"
    record = {
        "run_id": "run-1",
        "training_evidence": {
            "core_candidates": {
                "core": {
                    "candidate": "core",
                    "total_folds": 2,
                    "folds": folds,
                }
            }
        },
    }
    benchmark_path.write_text(json.dumps(record))
    config = SimpleNamespace(
        residual_model=SimpleNamespace(
            oof_benchmark_path=benchmark_path,
            oof_run_id="run-1",
            oof_candidate="core",
        )
    )
    frame = pl.DataFrame(
        {
            "market_id": ["market-0", "market-1"],
            "window_start": [
                datetime(2026, 1, 7, tzinfo=UTC),
                datetime(2026, 1, 8, tzinfo=UTC),
            ],
            "observed_at": [
                datetime(2026, 1, 7, 0, 1, tzinfo=UTC),
                datetime(2026, 1, 8, 0, 1, tzinfo=UTC),
            ],
            "seconds_elapsed": [60, 60],
            "fold_index": [0, 1],
        }
    )

    _validate_oof_training_provenance(config, frame)

    record["training_evidence"]["core_candidates"]["core"]["folds"][1][
        "policy_range_end"
    ] = "2026-01-08T00:00:00+00:00"
    benchmark_path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="not chronologically trained"):
        _validate_oof_training_provenance(config, frame)
