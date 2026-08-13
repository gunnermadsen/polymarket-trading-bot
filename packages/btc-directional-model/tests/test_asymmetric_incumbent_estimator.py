from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

import btc_directional_model.asymmetric_incumbent_estimator as estimator_fallback
from btc_directional_model.asymmetric_incumbent_calibration import (
    load_incumbent_calibration_config,
)
from btc_directional_model.asymmetric_incumbent_estimator import (
    BOUNDARY_WEIGHT_POLICY,
    E1_HYBRID50_H3,
    E2_HYBRID50_H3_BOUNDARY,
    ESTIMATOR_FALLBACK_CANDIDATES,
    EXPECTED_HISTOGRAM_PARAMETERS,
    EXPECTED_TARGET_PRICE_BAND,
    EXPECTED_TARGET_SIDES,
    EXPECTED_TARGET_TIME_BANDS,
    FittedEstimatorFallback,
    _bundle_semantic_sha256,
    _fit_histogram_estimator,
    fallback_training_weights,
    fit_target_side_price_time_calibrators,
    incumbent_admission_boundary_mask,
    refit_selected_core_oracle_estimator_fallback,
    score_core_oracle_estimator_fallback,
)
from btc_directional_model.asymmetric_value_config import load_asymmetric_value_config
from btc_directional_model.asymmetric_value_training import AsymmetricValueModel
from btc_directional_model.core_config import load_core_config
from btc_directional_model.core_training import ProbabilityCalibrator
from btc_directional_model.early_value_training import TimeBandCalibrator


def _asymmetric_config():
    # The config's large external source paths are intentionally ignored in a
    # fresh feature worktree.  These unit tests exercise the frozen contract,
    # not source-materialization readiness.
    with patch.object(Path, "exists", return_value=True):
        return load_asymmetric_value_config(
            Path(__file__).parents[1]
            / "configs/btc-5m-directional-asymmetric-value-calibrated-20260414-20260802.toml"
        )


def _estimator_contract():
    histogram = SimpleNamespace(**EXPECTED_HISTOGRAM_PARAMETERS)
    candidates = (
        SimpleNamespace(
            name=E1_HYBRID50_H3,
            target_weight=0.5,
            boundary_weighted=False,
            boundary_minimum_edge=None,
            boundary_maximum_edge=None,
            boundary_closed=None,
            boundary_multiplier=None,
            weighting=None,
        ),
        SimpleNamespace(
            name=E2_HYBRID50_H3_BOUNDARY,
            target_weight=0.5,
            boundary_weighted=True,
            boundary_minimum_edge=0.02,
            boundary_maximum_edge=0.04,
            boundary_closed="both",
            boundary_multiplier=2.0,
            weighting="market_equal_renormalized",
        ),
    )
    return SimpleNamespace(
        enabled=True,
        trigger="runner_owned",
        maximum_total_challengers=5,
        forbidden_triggers=(),
        histogram=histogram,
        candidates=candidates,
    )


def _incumbent_config():
    return load_incumbent_calibration_config(
        Path(__file__).parents[1]
        / "configs/btc-5m-asymmetric-core-oracle-gen2-calibration-20260716-20260802.toml"
    )


def _weight_frame() -> pl.DataFrame:
    start = datetime(2026, 5, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "market_id": ["a", "a", "a", "b", "b", "b"],
            "window_start": [start, start, start, start, start, start],
            "observed_at": [start + timedelta(seconds=value) for value in (1, 2, 60, 1, 2, 60)],
            "seconds_elapsed": [1, 2, 60, 1, 2, 60],
            "label_up": [1, 1, 1, 0, 0, 0],
            "yes_ask_vwap_5": [0.25, 0.25, 0.50, 0.75, 0.75, 0.50],
            "no_ask_vwap_5": [0.75, 0.75, 0.50, 0.25, 0.25, 0.50],
            "yes_cost_per_share": [0.26, 0.26, 0.51, 0.76, 0.76, 0.51],
            "no_cost_per_share": [0.76, 0.76, 0.51, 0.26, 0.26, 0.51],
        }
    )


def test_sealed_candidate_matrix_and_h3_contract_are_exact() -> None:
    contract = _estimator_contract()

    assert tuple(item.name for item in contract.candidates) == ESTIMATOR_FALLBACK_CANDIDATES
    assert vars(contract.histogram) == EXPECTED_HISTOGRAM_PARAMETERS
    assert contract.candidates[0].target_weight == 0.5
    assert contract.candidates[0].boundary_weighted is False
    assert vars(contract.candidates[1]) == {
        "name": E2_HYBRID50_H3_BOUNDARY,
        "target_weight": 0.5,
        "boundary_weighted": True,
        "boundary_minimum_edge": 0.02,
        "boundary_maximum_edge": 0.04,
        "boundary_closed": "both",
        "boundary_multiplier": 2.0,
        "weighting": "market_equal_renormalized",
    }


def test_incumbent_boundary_is_closed_and_only_uses_target_eligible_side() -> None:
    start = datetime(2026, 5, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "seconds_elapsed": [1, 1, 1, 1, 56],
            "yes_ask_vwap_5": [0.25, 0.25, 0.25, 0.40, 0.25],
            "no_ask_vwap_5": [0.75, 0.75, 0.75, 0.60, 0.75],
            "yes_cost_per_share": [0.28, 0.28, 0.28, 0.40, 0.28],
            "no_cost_per_share": [0.78, 0.78, 0.78, 0.60, 0.78],
            "window_start": [start] * 5,
        }
    )
    probabilities = np.asarray((0.30, 0.32, 0.299, 0.43, 0.31))

    selected = incumbent_admission_boundary_mask(
        frame,
        probabilities,
        target_mask=np.ones(5, dtype=bool),
    )

    assert selected.tolist() == [True, True, False, False, False]


def test_e2_doubles_boundary_influence_and_restores_each_market_total() -> None:
    frame = _weight_frame()
    config = _asymmetric_config()
    contract = _estimator_contract()
    # First row of each market has 3c incumbent edge; the second does not.
    incumbent_probability = np.asarray((0.29, 0.60, 0.50, 0.71, 0.40, 0.50))

    e1, e1_evidence = fallback_training_weights(
        frame,
        config,
        contract.candidates[0],
        incumbent_probabilities=incumbent_probability,
    )
    e2, e2_evidence = fallback_training_weights(
        frame,
        config,
        contract.candidates[1],
        incumbent_probabilities=incumbent_probability,
    )

    for market in ("a", "b"):
        selected = frame["market_id"].to_numpy() == market
        assert np.isclose(e1[selected].sum(), e2[selected].sum())
        early = np.flatnonzero(selected)[:2]
        assert np.isclose(
            (e2[early[0]] / e2[early[1]]) / (e1[early[0]] / e1[early[1]]),
            2.0,
        )
    assert e1_evidence["boundary_rows"] == 0
    assert e2_evidence["boundary_rows"] == 2
    assert e2_evidence["boundary_markets"] == 2
    assert e2_evidence["maximum_market_weight_total_delta"] < 1e-12


class _SyntheticLogitModel:
    candidate_name = "synthetic_core_oracle_h3"

    def raw_logit(self, frame: pl.DataFrame) -> np.ndarray:
        return frame["raw_logit"].to_numpy()


def _calibration_frame() -> pl.DataFrame:
    start = datetime(2026, 7, 16, tzinfo=UTC)
    seconds = (1, 15, 30, 45, 60, 90, 120, 180)
    rows: list[dict[str, object]] = []
    for market_index in range(120):
        side = "YES" if market_index % 2 == 0 else "NO"
        label = (market_index // 2) % 2
        window_start = start + timedelta(
            days=market_index % 7,
            minutes=5 * (market_index // 7),
        )
        for second in seconds:
            early = second < 60
            rows.append(
                {
                    "market_id": f"m-{market_index}",
                    "window_start": window_start,
                    "observed_at": window_start + timedelta(seconds=second),
                    "seconds_elapsed": second,
                    "label_up": label,
                    "yes_ask_vwap_5": 0.25 if early and side == "YES" else 0.75,
                    "no_ask_vwap_5": 0.25 if early and side == "NO" else 0.75,
                    "raw_logit": ((market_index % 7) - 3) / 10,
                }
            )
    return pl.DataFrame(rows)


def _identity_time_calibrators(frame: pl.DataFrame) -> tuple[TimeBandCalibrator, ...]:
    config = _asymmetric_config()
    return tuple(
        TimeBandCalibrator(
            start_second=start,
            end_second_exclusive=end,
            calibrator=ProbabilityCalibrator(
                slope=1.0,
                intercept=0.0,
                converged=True,
                iterations=1,
            ),
            rows=frame.filter(
                pl.col("seconds_elapsed").is_between(start, end, closed="left")
            ).height,
            markets=120,
        )
        for start, end in config.calibration_bands
    )


def test_target_calibration_fits_only_eight_cells_without_target_fallback() -> None:
    frame = _calibration_frame()
    config = _asymmetric_config()

    cells, evidence = fit_target_side_price_time_calibrators(
        _SyntheticLogitModel(),  # type: ignore[arg-type]
        _identity_time_calibrators(frame),
        frame,
        config,
        target_price_band=EXPECTED_TARGET_PRICE_BAND,
        target_time_bands=EXPECTED_TARGET_TIME_BANDS,
        target_sides=EXPECTED_TARGET_SIDES,
        minimum_markets_per_cell=50,
        minimum_days_per_cell=7,
        identity_l2=1.0,
        slope_bounds=(0.05, 3.0),
        intercept_bounds=(-2.0, 2.0),
    )

    fitted = [cell for cell in cells if cell.fitted]
    identity = [cell for cell in cells if not cell.fitted]
    assert len(cells) == 160
    assert len(fitted) == 8
    assert len(identity) == 152
    assert all(cell.fallback is None and cell.converged for cell in fitted)
    assert all(cell.positives > 0 and cell.negatives > 0 for cell in fitted)
    assert all(cell.slope > 0.0 for cell in fitted)
    assert all(
        cell.fallback == "target_contract_identity" and cell.slope == 1.0 and cell.intercept == 0.0
        for cell in identity
    )
    assert evidence["fitted_cells"] == 8
    assert evidence["identity_non_target_cells"] == 152
    assert evidence["target_contract"]["qualified"] is True


def test_h3_fit_keeps_exact_parameters_and_boundary_provenance() -> None:
    rows = 640
    start = datetime(2026, 5, 1, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": [f"m-{index // 2}" for index in range(rows)],
            "label_up": np.arange(rows) % 2,
            "feature_a": np.sin(np.arange(rows) / 13),
            "feature_b": np.cos(np.arange(rows) / 17),
            "window_start": [start] * rows,
        }
    )
    contract = _estimator_contract()
    core_config = load_core_config(_asymmetric_config().core_config)

    model = _fit_histogram_estimator(
        frame,
        ("feature_a", "feature_b"),
        np.ones(rows),
        candidate=contract.candidates[1],
        estimator_contract=contract,
        random_seed=20260809,
        threads=core_config.compute.threads_per_fit,
    )

    assert model.estimator.get_params()["learning_rate"] == 0.02
    assert model.estimator.get_params()["max_iter"] == 240
    assert model.estimator.get_params()["max_leaf_nodes"] == 7
    assert model.estimator.get_params()["min_samples_leaf"] == 300
    assert model.estimator.get_params()["l2_regularization"] == 10.0
    assert model.row_weight_policy == BOUNDARY_WEIGHT_POLICY
    assert model.hyperparameters["target_weight"] == 0.5
    assert model.hyperparameters["boundary_contract"]["minimum_edge"] == 0.02


class _ProbabilityBundle:
    def probability(self, frame: pl.DataFrame) -> np.ndarray:
        return 0.2 + frame["seconds_elapsed"].to_numpy() / 1_000


def test_probability_artifact_is_order_stable_and_excludes_outcomes_and_economics() -> None:
    start = datetime(2026, 7, 23, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "market_id": ["b", "a"],
            "window_start": [start + timedelta(minutes=5), start],
            "observed_at": [start + timedelta(minutes=5, seconds=5), start + timedelta(seconds=1)],
            "seconds_elapsed": [5, 1],
            "label_up": [0, 1],
            "yes_cost_per_share": [0.25, 0.25],
        }
    )
    fitted = FittedEstimatorFallback(
        candidate_id=E1_HYBRID50_H3,
        bundle=_ProbabilityBundle(),  # type: ignore[arg-type]
        evidence={},
        semantic_sha256="a" * 64,
    )

    first = score_core_oracle_estimator_fallback(fitted, frame)
    second = score_core_oracle_estimator_fallback(fitted, frame.reverse())

    assert first.predictions.columns == [
        "candidate_id",
        "market_id",
        "window_start",
        "observed_at",
        "seconds_elapsed",
        "probability_yes",
    ]
    assert "label_up" not in first.predictions.columns
    assert "yes_cost_per_share" not in first.predictions.columns
    assert first.key_sha256 == second.key_sha256
    assert first.probability_sha256 == second.probability_sha256
    assert first.artifact_sha256 == second.artifact_sha256


def test_selected_final_refit_uses_final_calibration_window_without_new_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asymmetric_config = _asymmetric_config()
    contract = _estimator_contract()
    final_window = SimpleNamespace(
        start=asymmetric_config.calibration.start,
        end=asymmetric_config.policy.end,
    )
    incumbent_config = SimpleNamespace(
        asymmetric_value_config=asymmetric_config.source_path,
        calibration_fit=asymmetric_config.calibration,
        matched_comparison=asymmetric_config.policy,
        final_refit=final_window,
        target_price_band=EXPECTED_TARGET_PRICE_BAND,
        target_time_bands=EXPECTED_TARGET_TIME_BANDS,
        sides=EXPECTED_TARGET_SIDES,
        minimum_markets_per_cell=50,
        minimum_days_per_cell=7,
        identity_l2=1.0,
        slope_bounds=(0.05, 3.0),
        intercept_bounds=(-2.0, 2.0),
        random_seed=20260809,
        conditional_estimator=contract,
    )
    expected = FittedEstimatorFallback(
        candidate_id=E2_HYBRID50_H3_BOUNDARY,
        bundle=_ProbabilityBundle(),  # type: ignore[arg-type]
        evidence={},
        semantic_sha256="b" * 64,
    )
    observed: dict[str, object] = {}

    def fake_fit(*args, **kwargs):
        observed.update(kwargs)
        return {E2_HYBRID50_H3_BOUNDARY: expected}

    monkeypatch.setattr(estimator_fallback, "fit_core_oracle_estimator_fallbacks", fake_fit)

    result = refit_selected_core_oracle_estimator_fallback(
        pl.DataFrame(),
        incumbent_config,
        asymmetric_config,
        load_core_config(asymmetric_config.core_config),
        candidate_id=E2_HYBRID50_H3_BOUNDARY,
        incumbent_probabilities=np.asarray([], dtype=np.float64),
    )

    assert result is expected
    assert observed["fit_window"] == asymmetric_config.fit
    assert observed["calibration_window"] is final_window
    assert observed["candidate_ids"] == (E2_HYBRID50_H3_BOUNDARY,)
    assert observed["asymmetric_config"].random_seed == 20260809


def test_selected_final_refit_reuses_exact_estimator_and_only_changes_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asymmetric_config = _asymmetric_config()
    incumbent_config = _incumbent_config()
    core_config = load_core_config(asymmetric_config.core_config)
    rows = 640
    training_frame = pl.DataFrame(
        {
            "market_id": [f"fit-{index // 2}" for index in range(rows)],
            "label_up": np.arange(rows) % 2,
            "feature": np.sin(np.arange(rows) / 13),
        }
    )
    candidate = _estimator_contract().candidates[0]
    model = _fit_histogram_estimator(
        training_frame,
        ("feature",),
        np.ones(rows),
        candidate=candidate,
        estimator_contract=_estimator_contract(),
        random_seed=20260809,
        threads=core_config.compute.threads_per_fit,
    )
    source_frame = _calibration_frame().with_columns(
        pl.col("raw_logit").alias("feature"),
        (pl.col("yes_ask_vwap_5") + 0.01).alias("yes_cost_per_share"),
        (pl.col("no_ask_vwap_5") + 0.01).alias("no_cost_per_share"),
    )
    development_bundle = AsymmetricValueModel(
        name=E1_HYBRID50_H3,
        model=model,
        time_calibrators=(),
        cells=(),
        parent_calibration_source="alltime",
        identity_l2_strength=1.0,
    )
    development_semantic = _bundle_semantic_sha256(development_bundle)
    selected_fit = FittedEstimatorFallback(
        candidate_id=E1_HYBRID50_H3,
        bundle=development_bundle,
        evidence={"semantic_sha256": development_semantic},
        semantic_sha256=development_semantic,
    )
    refit_calibrators = _identity_time_calibrators(source_frame)
    refit_cells, refit_evidence = fit_target_side_price_time_calibrators(
        _SyntheticLogitModel(),  # type: ignore[arg-type]
        refit_calibrators,
        source_frame,
        asymmetric_config,
        target_price_band=EXPECTED_TARGET_PRICE_BAND,
        target_time_bands=EXPECTED_TARGET_TIME_BANDS,
        target_sides=EXPECTED_TARGET_SIDES,
        minimum_markets_per_cell=50,
        minimum_days_per_cell=7,
        identity_l2=1.0,
        slope_bounds=(0.05, 3.0),
        intercept_bounds=(-2.0, 2.0),
    )
    estimator_object = model.estimator
    raw_logit_before = model.raw_logit(source_frame)
    monkeypatch.setattr(
        estimator_fallback,
        "_validate_selected_development_fit",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        estimator_fallback,
        "_core_oracle_feature_contract",
        lambda: ("feature",),
    )
    monkeypatch.setattr(
        estimator_fallback,
        "fit_asymmetric_time_band_calibrators",
        lambda *args, **kwargs: refit_calibrators,
    )
    monkeypatch.setattr(
        estimator_fallback,
        "fit_target_side_price_time_calibrators",
        lambda *args, **kwargs: (refit_cells, refit_evidence),
    )

    result = refit_selected_core_oracle_estimator_fallback(
        source_frame,
        incumbent_config,
        asymmetric_config,
        core_config,
        candidate_id=E1_HYBRID50_H3,
        incumbent_probabilities=np.asarray([], dtype=np.float64),
        selected_fit=selected_fit,
    )

    assert result.bundle.model is model
    assert result.bundle.model.estimator is estimator_object
    np.testing.assert_array_equal(result.bundle.model.raw_logit(source_frame), raw_logit_before)
    assert result.bundle.time_calibrators
    assert len([cell for cell in result.bundle.cells if cell.fitted]) == 8
    assert result.semantic_sha256 != development_semantic
    assert selected_fit.semantic_sha256 == development_semantic
    assert selected_fit.evidence == {"semantic_sha256": development_semantic}
    reuse = result.evidence["selected_estimator_reuse"]
    assert reuse["estimator_refit_performed"] is False
    assert reuse["training_weights_recomputed"] is False
    assert reuse["estimator_object_identity_preserved"] is True
    assert reuse["estimator_semantic_sha256_before"] == reuse[
        "estimator_semantic_sha256_after"
    ]
    assert reuse["estimator_bytes_sha256_before"] == reuse[
        "estimator_bytes_sha256_after"
    ]
    assert reuse["raw_logit_sha256_before"] == reuse["raw_logit_sha256_after"]
    assert reuse["only_calibration_window_expanded"] is True
    assert result.evidence["fit_window"] == {
        "start": asymmetric_config.fit.start.isoformat(),
        "end_exclusive": asymmetric_config.fit.end.isoformat(),
    }
    assert result.evidence["calibration_window"] == {
        "start": incumbent_config.final_refit.start.isoformat(),
        "end_exclusive": incumbent_config.final_refit.end.isoformat(),
    }


def test_selected_final_refit_rejects_candidate_mismatch() -> None:
    asymmetric_config = _asymmetric_config()
    incumbent_config = _incumbent_config()
    selected_fit = FittedEstimatorFallback(
        candidate_id=E2_HYBRID50_H3_BOUNDARY,
        bundle=_ProbabilityBundle(),  # type: ignore[arg-type]
        evidence={},
        semantic_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="candidate mismatch"):
        refit_selected_core_oracle_estimator_fallback(
            pl.DataFrame(),
            incumbent_config,
            asymmetric_config,
            load_core_config(asymmetric_config.core_config),
            candidate_id=E1_HYBRID50_H3,
            incumbent_probabilities=np.asarray([], dtype=np.float64),
            selected_fit=selected_fit,
        )


def test_bundle_semantic_hash_changes_with_estimator_candidate_contract() -> None:
    rows = 640
    frame = pl.DataFrame(
        {
            "market_id": [f"m-{index // 2}" for index in range(rows)],
            "label_up": np.arange(rows) % 2,
            "feature": np.sin(np.arange(rows) / 13),
        }
    )
    contract = _estimator_contract()
    core_config = load_core_config(_asymmetric_config().core_config)
    model = _fit_histogram_estimator(
        frame,
        ("feature",),
        np.ones(rows),
        candidate=contract.candidates[0],
        estimator_contract=contract,
        random_seed=20260809,
        threads=core_config.compute.threads_per_fit,
    )
    calibration_frame = _calibration_frame()
    calibrators = _identity_time_calibrators(calibration_frame)
    cells, _ = fit_target_side_price_time_calibrators(
        _SyntheticLogitModel(),  # type: ignore[arg-type]
        calibrators,
        calibration_frame,
        _asymmetric_config(),
        target_price_band=EXPECTED_TARGET_PRICE_BAND,
        target_time_bands=EXPECTED_TARGET_TIME_BANDS,
        target_sides=EXPECTED_TARGET_SIDES,
        minimum_markets_per_cell=50,
        minimum_days_per_cell=7,
        identity_l2=1.0,
        slope_bounds=(0.05, 3.0),
        intercept_bounds=(-2.0, 2.0),
    )
    # Hash only needs a runtime-exportable model plus calibrated surface; scoring
    # these synthetic calibration rows through the one-feature HGB is not required.
    from btc_directional_model.asymmetric_value_training import AsymmetricValueModel

    bundle = AsymmetricValueModel(
        name=E1_HYBRID50_H3,
        model=model,
        time_calibrators=calibrators,
        cells=cells,
    )
    first = _bundle_semantic_sha256(bundle)
    model.hyperparameters["target_weight"] = 0.25
    second = _bundle_semantic_sha256(bundle)

    assert len(first) == 64
    assert first != second
