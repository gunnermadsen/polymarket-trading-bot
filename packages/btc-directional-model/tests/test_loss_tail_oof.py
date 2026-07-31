from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from btc_directional_model import loss_tail_oof
from btc_directional_model.core_features import (
    CORE_BOUNDARY_ENRICHED_FEATURES,
    CORE_ENRICHED_FEATURES,
    CORE_MATURE_REVERSAL_ENRICHED_FEATURES,
)
from btc_directional_model.core_training import ProbabilityCalibrator
from btc_directional_model.loss_tail_oof import (
    BOUNDARY_CORRECTNESS_FEATURES,
    LOSS_TAIL_OOF_BLOCKS,
    LOSS_TAIL_OOF_CALIBRATION_WEIGHTING,
    LOSS_TAIL_OOF_HEADS,
    LOSS_TAIL_OOF_HGB_PARAMETERS,
    LOSS_TAIL_OOF_HISTORY_START,
    OOF_SOURCE_SIGNAL_FEATURES,
    ORACLE_BOOK_CONTRADICTION_FEATURES,
    build_causal_oof_signals,
    load_loss_tail_oof_signals,
)


def _core_config() -> object:
    return SimpleNamespace(
        model=SimpleNamespace(random_seed=20260731),
        compute=SimpleNamespace(threads_per_fit=1),
        source_path=None,
    )


def _benchmark_config() -> object:
    blocks = tuple(
        SimpleNamespace(name=block.name, start=block.start, end=block.end)
        for block in LOSS_TAIL_OOF_BLOCKS
    )
    return SimpleNamespace(
        walk_forward=SimpleNamespace(
            history_start=LOSS_TAIL_OOF_HISTORY_START,
            blocks=blocks,
        )
    )


def _market_rows(
    market_id: str,
    window_start: datetime,
    label_up: int,
) -> list[dict[str, object]]:
    feature_names = {feature for head in LOSS_TAIL_OOF_HEADS for feature in head.feature_names}
    rows: list[dict[str, object]] = []
    for seconds_elapsed in range(60, 241, 5):
        row: dict[str, object] = {
            "market_id": market_id,
            "window_start": window_start,
            "observed_at": window_start + timedelta(seconds=seconds_elapsed),
            "seconds_elapsed": seconds_elapsed,
            "label_up": label_up,
        }
        for feature_index, feature in enumerate(sorted(feature_names)):
            row[feature] = label_up * 0.25 + seconds_elapsed / 1_000.0 + feature_index / 10_000.0
        rows.append(row)
    return rows


def _universal_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    # The oldest history has 24 markets. Its chronological 80/20 split leaves
    # both labels in fit and calibration, as do all later expanding histories.
    history_start = LOSS_TAIL_OOF_HISTORY_START
    for market_index in range(24):
        rows.extend(
            _market_rows(
                f"history-{market_index:02d}",
                history_start + timedelta(hours=18 * market_index),
                market_index % 2,
            )
        )
    for block_index, block in enumerate(LOSS_TAIL_OOF_BLOCKS):
        for market_index in range(4):
            rows.extend(
                _market_rows(
                    f"{block.name}-{market_index}",
                    block.start + timedelta(minutes=5 * market_index),
                    market_index % 2,
                )
            )
    return pl.DataFrame(rows)


def _strict_frame(universal: pl.DataFrame) -> pl.DataFrame:
    strict = universal.filter(
        (pl.col("window_start") >= LOSS_TAIL_OOF_BLOCKS[0].start)
        & (pl.col("window_start") < LOSS_TAIL_OOF_BLOCKS[-1].end)
    )
    expressions: list[pl.Expr] = []
    for index, feature in enumerate(ORACLE_BOOK_CONTRADICTION_FEATURES[:20]):
        if feature not in strict.columns:
            expressions.append(pl.lit(0.01 + index / 100.0).alias(feature))
    return strict.with_columns(
        *expressions,
        pl.lit(0.72).alias("up_entry_debit_per_share"),
        pl.lit(0.34).alias("down_entry_debit_per_share"),
    )


def test_correctness_feature_contract_is_fixed_and_causal() -> None:
    assert tuple(len(head.feature_names) for head in LOSS_TAIL_OOF_HEADS) == (58, 68, 71)
    assert tuple(CORE_ENRICHED_FEATURES) == LOSS_TAIL_OOF_HEADS[0].feature_names
    assert tuple(CORE_BOUNDARY_ENRICHED_FEATURES) == LOSS_TAIL_OOF_HEADS[1].feature_names
    assert tuple(CORE_MATURE_REVERSAL_ENRICHED_FEATURES) == LOSS_TAIL_OOF_HEADS[2].feature_names
    assert BOUNDARY_CORRECTNESS_FEATURES == (
        *OOF_SOURCE_SIGNAL_FEATURES,
        *ORACLE_BOOK_CONTRADICTION_FEATURES,
    )
    assert len(set(BOUNDARY_CORRECTNESS_FEATURES)) == len(BOUNDARY_CORRECTNESS_FEATURES)
    assert {
        "oracle_gap_to_opening_boundary_bps",
        "binance_oracle_basis_bps",
        "book_mid_difference",
        "book_up_imbalance",
        "boundary_selected_debit_per_share",
        "boundary_selected_debit_severity",
    }.issubset(BOUNDARY_CORRECTNESS_FEATURES)
    forbidden = {
        "label_up",
        "official_outcome",
        "final_price",
        "boundary_direction_correct",
        "realized_selected_net_per_share",
    }
    assert forbidden.isdisjoint(BOUNDARY_CORRECTNESS_FEATURES)


def test_source_signal_deltas_require_an_exact_previous_five_second_point() -> None:
    head = LOSS_TAIL_OOF_HEADS[0]
    start = datetime(2026, 6, 9, tzinfo=UTC)
    score = pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "b", "b"],
            "window_start": [start] * 3 + [start + timedelta(minutes=5)] * 2,
            "observed_at": [
                start + timedelta(seconds=60),
                start + timedelta(seconds=65),
                start + timedelta(seconds=75),
                start + timedelta(minutes=5, seconds=60),
                start + timedelta(minutes=5, seconds=65),
            ],
            "seconds_elapsed": [60, 65, 75, 60, 65],
            "label_up": [1, 1, 1, 0, 0],
        }
    )
    signaled = loss_tail_oof._source_signal_frame(
        score,
        np.array([0.60, 0.75, 0.80, 0.40, 0.30]),
        head,
        "evaluation_jun09",
    ).sort(["market_id", "seconds_elapsed"])

    assert signaled[f"{head.prefix}_probability_delta_5s"].to_list() == pytest.approx(
        [None, 0.15, None, None, -0.10]
    )
    assert signaled[f"{head.prefix}_confidence_delta_5s"].to_list() == pytest.approx(
        [None, 0.15, None, None, 0.10]
    )
    assert signaled[f"{head.prefix}_no_trade"].to_list() == [1, 1, 1, 1, 1]


def test_head_fit_uses_fixed_parameters_recency_and_four_boundary_bands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_calls: list[tuple[object, dict[str, object]]] = []
    calibration_calls: list[tuple[str, int]] = []

    class _Model:
        def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
            return np.linspace(-1.0, 1.0, frame.height)

    def fake_fit_model(
        frame: pl.DataFrame,
        spec: object,
        parameters: dict[str, object],
        config: object,
    ) -> _Model:
        del frame, config
        fit_calls.append((spec, parameters))
        return _Model()

    def fake_fit_calibrator(
        model: _Model,
        frame: pl.DataFrame,
        config: object,
        spec: object,
    ) -> ProbabilityCalibrator:
        del model, config
        calibration_calls.append((spec.name, frame.height))
        return ProbabilityCalibrator(1.0, 0.0, True, 3)

    monkeypatch.setattr(loss_tail_oof, "fit_model", fake_fit_model)
    monkeypatch.setattr(
        loss_tail_oof,
        "fit_probability_calibrator",
        fake_fit_calibrator,
    )
    elapsed = [60, 65, 90, 95, 120, 125, 180, 185]
    frame = pl.DataFrame(
        {
            "market_id": [f"market-{index}" for index in range(len(elapsed))],
            "seconds_elapsed": elapsed,
            "label_up": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    boundary_probability, boundary_record = loss_tail_oof._fit_head_probability(
        frame,
        frame,
        frame,
        LOSS_TAIL_OOF_HEADS[1],
        _core_config(),  # type: ignore[arg-type]
    )
    assert len(boundary_probability) == frame.height
    assert len(boundary_record["calibration"]["bands"]) == 4
    assert calibration_calls == [("loss_tail_oof_boundary_68", 2)] * 4
    boundary_spec, parameters = fit_calls[-1]
    assert boundary_spec.feature_names == tuple(CORE_BOUNDARY_ENRICHED_FEATURES)
    assert boundary_spec.recency_half_life_days is None
    assert parameters == LOSS_TAIL_OOF_HGB_PARAMETERS

    calibration_calls.clear()
    _, mature_record = loss_tail_oof._fit_head_probability(
        frame,
        frame,
        frame,
        LOSS_TAIL_OOF_HEADS[2],
        _core_config(),  # type: ignore[arg-type]
    )
    assert calibration_calls == [("loss_tail_oof_mature_reversal_71", frame.height)]
    mature_spec, _ = fit_calls[-1]
    assert mature_spec.recency_half_life_days == pytest.approx(28.0)
    assert mature_record["calibration"]["weighting"] == (LOSS_TAIL_OOF_CALIBRATION_WEIGHTING)


def test_build_and_load_oof_cache_preserves_strict_keys_and_causality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    universal = _universal_frame()
    strict = _strict_frame(universal)
    core_path = tmp_path / "core.parquet"
    strict_path = tmp_path / "strict.parquet"
    output_path = tmp_path / "oof.parquet"
    universal.write_parquet(core_path)
    strict.write_parquet(strict_path)
    calls: list[tuple[str, datetime, datetime, datetime]] = []

    def fake_fit_head_probability(
        fit: pl.DataFrame,
        calibration: pl.DataFrame,
        score: pl.DataFrame,
        head: object,
        core_config: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del core_config
        calls.append(
            (
                head.name,
                fit["window_start"].max(),
                calibration["window_start"].max(),
                score["window_start"].min(),
            )
        )
        base = {
            "enriched_58": 0.62,
            "boundary_68": 0.82,
            "mature_reversal_71": 0.38,
        }[head.name]
        probability = np.full(score.height, base, dtype=np.float64)
        probability += (score["seconds_elapsed"].to_numpy() - 60.0) / 10_000.0
        return probability, {
            "head": head.name,
            "feature_count": len(head.feature_names),
            "calibration": {
                "kind": head.calibration_kind,
                "weighting": LOSS_TAIL_OOF_CALIBRATION_WEIGHTING,
            },
            "recency_half_life_days": head.recency_half_life_days,
            "deployed_artifact_used": False,
        }

    monkeypatch.setattr(
        loss_tail_oof,
        "_fit_head_probability",
        fake_fit_head_probability,
    )
    metadata = build_causal_oof_signals(
        core_path=core_path,
        strict_path=strict_path,
        output_path=output_path,
        config=_benchmark_config(),
        core_config=_core_config(),  # type: ignore[arg-type]
    )
    loaded = load_loss_tail_oof_signals(output_path)

    assert len(calls) == len(LOSS_TAIL_OOF_HEADS) * len(LOSS_TAIL_OOF_BLOCKS)
    assert all(
        fit_end < calibration_end < score_start
        for _, fit_end, calibration_end, score_start in calls
    )
    assert metadata["rows"] == strict.height == loaded.height
    assert metadata["markets"] == strict["market_id"].n_unique()
    assert metadata["deployment"] == {
        "authorized": False,
        "runtime_artifact_loaded": False,
        "runtime_artifact_exported": False,
    }
    assert set(metadata["blocks"]) == {block.name for block in LOSS_TAIL_OOF_BLOCKS}
    assert loaded["walk_forward_block"].equals(loaded["oof_block"])
    assert loaded.select("market_id", "window_start", "observed_at", "seconds_elapsed").equals(
        strict.select("market_id", "window_start", "observed_at", "seconds_elapsed").sort(
            ["window_start", "market_id", "seconds_elapsed"]
        )
    )
    assert (
        loaded["boundary_direction_correct"].to_list()
        == loaded.select(
            (pl.col("oof_boundary_predicted_up") == pl.col("label_up").cast(pl.Int8)).cast(pl.Int8)
        )
        .to_series()
        .to_list()
    )
    assert loaded["boundary_selected_debit_per_share"].unique().to_list() == [0.72]
    assert loaded["boundary_selected_debit_severity"].unique().item() == pytest.approx(0.72 / 0.28)

    def fail_if_refit(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("an immutable matching OOF cache must be reused")

    monkeypatch.setattr(loss_tail_oof, "_fit_head_probability", fail_if_refit)
    cached = build_causal_oof_signals(
        core_path=core_path,
        strict_path=strict_path,
        output_path=output_path,
        config=_benchmark_config(),
        core_config=_core_config(),  # type: ignore[arg-type]
    )
    assert cached["output_sha256"] == metadata["output_sha256"]
