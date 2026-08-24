from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from .core_extract import file_sha256
from .payoff_runtime_export import _base_model, _finite, _histogram, write_bundle

MODEL_KEY = "btc-5m-specialist-distilled-fair-value-paper-20260823-v1"
CANDIDATE = "specialist_distilled_fair_value"
SOURCE_FOLD = "fold_5"
FAIR_VALUE_FEATURE_SCHEMA = "btc-5m-payoff-aware-fair-value-features-v1"
ENTRY_CELLS = (
    ("early_15_89", 15, 90),
    ("middle_90_119", 90, 120),
    ("middle_120_149", 120, 150),
    ("middle_150_179", 150, 180),
    ("late_180_240", 180, 241),
)


def fair_value_payload(
    source: dict[str, Any],
    policy: dict[str, Any],
    model_key: str,
    source_sha: str,
) -> dict[str, Any]:
    """Export the specialist through the existing payoff-aware bundle contract."""

    estimator_names = list(source["feature_names"])
    names = [*estimator_names, "pm_yes_cost_per_share", "pm_no_cost_per_share"]
    cells = policy.get("cells", {})
    if policy.get("type") != "cell_first_crossing" or not cells:
        raise ValueError("fair-value runtime policy must define entry cells")
    payoff = {
        "kind": "fair_value",
        "stress_slippage_per_share": 0.01,
        "outcome": _histogram(source["distilled"], estimator_names, names, "regression"),
        "yes_cost_feature_index": names.index("pm_yes_cost_per_share"),
        "no_cost_feature_index": names.index("pm_no_cost_per_share"),
        "policy": {
            "cells": {
                name: {
                    "enabled": bool(value.get("enabled", False)),
                    "confidence": _finite(value.get("confidence", 1.0)),
                    "stress_edge": _finite(value.get("edge", 0.0)),
                }
                for name, value in cells.items()
            }
        },
        "prediction_policy": {
            "type": "first_confidence_crossing",
            "minimum_seconds_after_open": 15,
            "maximum_seconds_after_open": 240,
            "cadence_seconds": 5,
            "early_end_second": 89,
            "early_cadence_seconds": 1,
        },
    }
    return _base_model(
        model_key,
        FAIR_VALUE_FEATURE_SCHEMA,
        names,
        source_sha,
        payoff,
    )


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _cell(seconds_elapsed: int) -> str:
    for name, start, end in ENTRY_CELLS:
        if start <= seconds_elapsed < end:
            return name
    raise ValueError(f"elapsed second {seconds_elapsed} is outside the fair-value policy")


def _score(
    estimator: Any,
    estimator_names: list[str],
    policy: dict[str, Any],
    feature_names: list[str],
    feature_values: list[float | None],
    seconds_elapsed: int,
) -> dict[str, Any]:
    values = dict(zip(feature_names, feature_values, strict=True))
    matrix = np.asarray(
        [[np.nan if values[name] is None else values[name] for name in estimator_names]],
        dtype=float,
    )
    probability_up = float(np.clip(estimator.predict(matrix)[0], 1e-6, 1.0 - 1e-6))
    predicted_up = probability_up >= 0.5
    confidence = probability_up if predicted_up else 1.0 - probability_up
    cost_name = "pm_yes_cost_per_share" if predicted_up else "pm_no_cost_per_share"
    cost = values[cost_name]
    if cost is None or not math.isfinite(cost):
        raise ValueError("selected executable cost is unavailable")
    threshold = policy["cells"][_cell(seconds_elapsed)]
    accepted = bool(
        threshold.get("enabled", False)
        and confidence >= float(threshold["confidence"])
        and confidence - cost - 0.01 >= float(threshold["edge"])
    )
    return {
        "raw_logit": math.log(probability_up / (1.0 - probability_up)),
        "probability_up": probability_up,
        "confidence": confidence,
        "action": "up" if accepted and predicted_up else "down" if accepted else "no_trade",
    }


def _vector(
    *,
    identifier: str,
    row: dict[str, Any],
    estimator: Any,
    estimator_names: list[str],
    policy: dict[str, Any],
    feature_names: list[str],
    cost_override: float | None = None,
    missing_feature: str | None = None,
) -> dict[str, Any]:
    values = []
    for name in feature_names:
        if name == "pm_yes_cost_per_share":
            value = row["up_cost_5"]
        elif name == "pm_no_cost_per_share":
            value = row["down_cost_5"]
        else:
            value = row.get(name)
        values.append(_finite_or_none(value))
    if cost_override is not None:
        values[feature_names.index("pm_yes_cost_per_share")] = cost_override
        values[feature_names.index("pm_no_cost_per_share")] = cost_override
    if missing_feature is not None:
        values[feature_names.index(missing_feature)] = None
    seconds_elapsed = int(row["seconds_elapsed"])
    return {
        "id": identifier,
        "source": {
            "market_id": str(row["market_id"]),
            "observed_at": row["observed_at"].isoformat(),
        },
        "seconds_elapsed": seconds_elapsed,
        "feature_values": values,
        "expected": _score(
            estimator,
            estimator_names,
            policy,
            feature_names,
            values,
            seconds_elapsed,
        ),
    }


def golden_vectors(
    source: dict[str, Any], policy: dict[str, Any], model: dict[str, Any], ledger: pl.DataFrame
) -> list[dict[str, Any]]:
    fold = ledger.filter(pl.col("outer_fold") == SOURCE_FOLD)
    estimator = source["distilled"]
    estimator_names = list(source["feature_names"])
    feature_names = list(model["features"]["names"])
    vectors: list[dict[str, Any]] = []
    for cell, _, _ in ENTRY_CELLS:
        for predicted_up in (False, True):
            selected = fold.filter(
                (pl.col("entry_cell") == cell) & (pl.col("predicted_up") == predicted_up)
            ).head(1)
            if selected.is_empty():
                raise RuntimeError(f"{SOURCE_FOLD} lacks a {cell}/{predicted_up} parity row")
            row = selected.row(0, named=True)
            vectors.append(
                _vector(
                    identifier=f"{cell}_{'up' if predicted_up else 'down'}",
                    row=row,
                    estimator=estimator,
                    estimator_names=estimator_names,
                    policy=policy,
                    feature_names=feature_names,
                )
            )
    reference = fold.head(1).row(0, named=True)
    vectors.append(
        _vector(
            identifier="executable_edge_rejected",
            row=reference,
            estimator=estimator,
            estimator_names=estimator_names,
            policy=policy,
            feature_names=feature_names,
            cost_override=0.999,
        )
    )
    vectors.append(
        _vector(
            identifier="native_missing_chainlink_candle",
            row=reference,
            estimator=estimator,
            estimator_names=estimator_names,
            policy=policy,
            feature_names=feature_names,
            missing_feature="chainlink_candle_return_5m_bps",
        )
    )
    actions = {vector["expected"]["action"] for vector in vectors}
    if actions != {"up", "down", "no_trade"}:
        raise RuntimeError(f"golden vectors do not cover all actions: {sorted(actions)}")
    return vectors


def export(
    *,
    tournament_path: Path,
    metrics_path: Path,
    ledger_path: Path,
    output_root: Path,
    model_key: str = MODEL_KEY,
) -> Path:
    tournament_sha = file_sha256(tournament_path)
    metrics = json.loads(metrics_path.read_text())
    if metrics["model_artifact"]["sha256"] != tournament_sha:
        raise ValueError("tournament artifact does not match its frozen metrics")
    tournament = joblib.load(tournament_path)
    if tournament.get("diagnostic_leader") != CANDIDATE:
        raise ValueError("source tournament does not identify the specialist diagnostic leader")
    if tournament.get("runtime_exported") is not False:
        raise ValueError("source tournament runtime status is invalid")
    source = tournament["fold_models"][SOURCE_FOLD]["fair_value"]
    policy = metrics["fold_results"][CANDIDATE][SOURCE_FOLD]["policy"]
    model = fair_value_payload(source, policy, model_key, tournament_sha)
    model["provenance"].update(
        {
            "candidate": CANDIDATE,
            "source_fold": SOURCE_FOLD,
            "source_commit": tournament["source_commit"],
            "source_metrics_run_id": metrics["run_id"],
        }
    )
    vectors = golden_vectors(source, policy, model, pl.read_parquet(ledger_path))
    return write_bundle(model, output_root, tournament_sha, vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-key", default=MODEL_KEY)
    args = parser.parse_args()
    destination = export(
        tournament_path=args.tournament,
        metrics_path=args.metrics,
        ledger_path=args.ledger,
        output_root=args.output_root,
        model_key=args.model_key,
    )
    print(destination)


if __name__ == "__main__":
    main()
