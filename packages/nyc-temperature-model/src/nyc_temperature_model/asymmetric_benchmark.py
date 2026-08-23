from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

from . import PROCESS_ID
from .config import Settings
from .contracts import canonical_market_rows
from .database import connection
from .fees import taker_fee_per_share
from .modeling import (
    ROUNDED_TEMPERATURE_SUPPORT_MAX_F,
    ROUNDED_TEMPERATURE_SUPPORT_MIN_F,
    build_feature_rows,
    load_model,
    normalized_bucket_probabilities,
    point_prediction,
    raw_point_prediction,
)
from .sources import file_sha256

DECISION_MODES = ("midnight_only", "noon_only", "midnight_then_noon")
PROBABILITY_BOOTSTRAP_SEED = 24_051_985
ECONOMIC_BOOTSTRAP_CONFIDENCE = 0.90
ECONOMIC_BOOTSTRAP_ITERATIONS = 5000
ECONOMIC_BOOTSTRAP_BLOCK_DAYS = 7
ECONOMIC_BOOTSTRAP_SEED = 41_819_850
PRICE_CELL_MINIMUM_CLUSTER_DAYS = 14
POLICY_SPECIFICATIONS = (
    ("deep_asymmetry", 0.16, 0.03, 0.35),
    ("balanced_asymmetry", 0.25, 0.04, 0.35),
    ("strong_edge", 0.25, 0.06, 0.50),
)
PRICE_BANDS = (
    (0.00, 0.04, "00-04c"),
    (0.04, 0.08, "04-08c"),
    (0.08, 0.12, "08-12c"),
    (0.12, 0.16, "12-16c"),
    (0.16, 0.25, "16-25c"),
    (0.25, 0.50, "25-50c"),
    (0.50, 0.75, "50-75c"),
    (0.75, 0.90, "75-90c"),
    (0.90, 1.000001, "90-100c"),
)


@dataclass(frozen=True)
class AsymmetricPolicy:
    name: str
    decision_mode: str
    minimum_all_in_cost: float
    maximum_all_in_cost: float
    minimum_robust_edge: float
    minimum_robust_roi: float
    minimum_probability: float = 0.0
    maximum_probability: float = 1.0
    rank_by: str = "robust_expected_roi"

    def __post_init__(self) -> None:
        if self.decision_mode not in DECISION_MODES:
            raise ValueError(f"unsupported decision mode: {self.decision_mode}")
        if not 0 <= self.minimum_all_in_cost <= self.maximum_all_in_cost <= 1:
            raise ValueError("policy all-in cost bounds are invalid")
        if not 0 <= self.minimum_probability <= self.maximum_probability <= 1:
            raise ValueError("policy probability bounds are invalid")
        if self.rank_by not in {"robust_expected_roi", "probability"}:
            raise ValueError(f"unsupported policy ranking: {self.rank_by}")


def policy_frontier() -> list[AsymmetricPolicy]:
    return [
        AsymmetricPolicy(
            name=f"{name}:{decision_mode}",
            decision_mode=decision_mode,
            minimum_all_in_cost=0.04,
            maximum_all_in_cost=maximum_cost,
            minimum_robust_edge=minimum_edge,
            minimum_robust_roi=minimum_roi,
        )
        for name, maximum_cost, minimum_edge, minimum_roi in POLICY_SPECIFICATIONS
        for decision_mode in DECISION_MODES
    ]


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _circular_block_indices(
    length: int,
    *,
    iterations: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    if length <= 0 or iterations <= 0 or block_size <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    rng = np.random.default_rng(seed)
    blocks = math.ceil(length / block_size)
    starts = rng.integers(0, length, size=(iterations, blocks, 1))
    offsets = np.arange(block_size).reshape(1, 1, block_size)
    return ((starts + offsets) % length).reshape(iterations, -1)[:, :length]


def _bootstrap_lower_mean(
    values: list[float],
    *,
    confidence: float = ECONOMIC_BOOTSTRAP_CONFIDENCE,
    iterations: int = ECONOMIC_BOOTSTRAP_ITERATIONS,
    block_size: int = ECONOMIC_BOOTSTRAP_BLOCK_DAYS,
    seed: int = ECONOMIC_BOOTSTRAP_SEED,
) -> float:
    if not values:
        return float("nan")
    array = np.asarray(values, dtype=np.float64)
    indices = _circular_block_indices(
        array.size,
        iterations=iterations,
        block_size=min(block_size, array.size),
        seed=seed,
    )
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 1.0 - confidence))


def _probability_estimates(
    point: float,
    residuals: np.ndarray,
    buckets: list[tuple[int | None, int | None]],
    bootstrap_indices: np.ndarray,
) -> list[dict[str, float]]:
    point_probabilities = normalized_bucket_probabilities(point, residuals, buckets)
    simulated = np.floor(point + residuals[bootstrap_indices] + 0.5).astype(np.int64)
    denominator = residuals.size + 0.5 * len(buckets)
    output = []
    for point_probability, (lower, upper) in zip(
        point_probabilities, buckets, strict=True
    ):
        selected = np.ones(simulated.shape, dtype=bool)
        if lower is not None:
            selected &= simulated >= lower
        if upper is not None:
            selected &= simulated <= upper
        sampled_yes = (selected.sum(axis=1) + 0.5) / denominator
        yes_lower = min(point_probability, float(np.quantile(sampled_yes, 0.10)))
        no_probability = 1.0 - point_probability
        no_lower = min(no_probability, float(np.quantile(1.0 - sampled_yes, 0.10)))
        output.append(
            {
                "yes": point_probability,
                "yes_lower": max(0.0, yes_lower),
                "no": no_probability,
                "no_lower": max(0.0, no_lower),
            }
        )
    return output


def _market_rows(database_url: str, start: date, end: date) -> list[dict[str, Any]]:
    with connection(database_url) as conn:
        rows = list(
            conn.execute(
                """
                SELECT market_id,event_id,event_slug,event_date,bucket_lower_f,bucket_upper_f,
                       resolved_yes,fees_enabled,fee_rate::double precision,
                       fee_exponent::double precision,fee_taker_only
                FROM weather.temperature_markets
                WHERE event_date >= %s AND event_date <= %s AND resolved_yes IS NOT NULL
                ORDER BY event_date,event_id,market_id
                """,
                (start, end),
            ).fetchall()
        )
    return canonical_market_rows(rows)


def _model_metadata(database_url: str, model_run_id: str) -> dict[str, Any]:
    with connection(database_url) as conn:
        row = conn.execute(
            """
            SELECT model_run_id::text,process_id::text,candidate,decision_hour_local,
                   training_start,training_end,calibration_start,calibration_end,
                   model_uri,model_sha256,metrics
            FROM weather.model_runs
            WHERE process_id=%s AND model_run_id=%s
            """,
            (PROCESS_ID, model_run_id),
        ).fetchone()
    if not row:
        raise ValueError(f"unknown process-owned model_run_id: {model_run_id}")
    return dict(row)


def _validate_process_contract(
    database_url: str,
    *,
    discovery_start: date,
    discovery_end: date,
    evaluation_start: date,
    evaluation_end: date,
    quantity: float,
    modeled_slippage_per_share: float,
    probability_bootstrap_iterations: int,
) -> dict[str, Any]:
    with connection(database_url) as conn:
        row = conn.execute(
            """
            SELECT enabled,config
            FROM polymarket.trading_processes
            WHERE process_id=%s
            """,
            (PROCESS_ID,),
        ).fetchone()
    if not row:
        raise ValueError("canonical NYC temperature trading process is missing")
    config = dict(row["config"])
    if row["enabled"] or config.get("order_submission_enabled") is not False:
        raise ValueError("asymmetric benchmark requires a disabled offline-only process")
    if config.get("sides") != ["buy_yes", "buy_no"]:
        raise ValueError("process side contract does not allow bidirectional evaluation")
    if config.get("objective") != (
        "positive_net_expectancy_at_low_executable_cost_without_accuracy_gate"
    ):
        raise ValueError("process objective does not match the asymmetric benchmark")

    expected_forecast = {
        "eligible_ml_candidates": ["linear_bias", "histogram_residual"],
        "diagnostic_candidates": ["raw_hrrr"],
        "metric": "calibration_loo_rounded_log_loss",
        "secondary_metric": "calibration_loo_ranked_probability_score",
        "point_diagnostic": "calibration_rmse_f",
        "rounded_temperature_support_min_f": ROUNDED_TEMPERATURE_SUPPORT_MIN_F,
        "rounded_temperature_support_max_f": ROUNDED_TEMPERATURE_SUPPORT_MAX_F,
        "training_start": "2019-01-01",
        "training_end": "2024-12-31",
        "calibration_start": "2025-01-01",
        "calibration_end": "2025-12-31",
        "holdout_influence": False,
    }
    if config.get("forecast_selection") != expected_forecast:
        raise ValueError("process forecast-selection contract does not match implementation")

    economic = config.get("economic_evaluation") or {}
    expected_economic = {
        "discovery_start": discovery_start.isoformat(),
        "discovery_end": discovery_end.isoformat(),
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end": evaluation_end.isoformat(),
        "quantity": quantity,
        "modeled_slippage_per_share": modeled_slippage_per_share,
        "slippage_stress_per_share": [0, 0.005, 0.01, 0.02],
        "maximum_positions_per_event_day": 1,
        "price_cell_minimum_cluster_days": PRICE_CELL_MINIMUM_CLUSTER_DAYS,
    }
    for key, expected in expected_economic.items():
        if economic.get(key) != expected:
            raise ValueError(
                f"process economic contract mismatch for {key}: "
                f"expected {expected!r}, found {economic.get(key)!r}"
            )

    configured_frontier = config.get("policy_frontier") or {}
    expected_specifications = [
        {
            "name": name,
            "maximum_all_in_cost": maximum_cost,
            "minimum_robust_edge": minimum_edge,
            "minimum_robust_roi": minimum_roi,
        }
        for name, maximum_cost, minimum_edge, minimum_roi in POLICY_SPECIFICATIONS
    ]
    expected_frontier = {
        "minimum_all_in_cost": 0.04,
        "rank_by": "robust_expected_roi",
        "probability_lower_quantile": 0.10,
        "residual_bootstrap_block_days": 7,
        "probability_bootstrap_iterations": probability_bootstrap_iterations,
        "probability_bootstrap_seed": PROBABILITY_BOOTSTRAP_SEED,
        "probability_bootstrap_common_random_numbers": True,
        "economic_bootstrap_confidence": ECONOMIC_BOOTSTRAP_CONFIDENCE,
        "economic_bootstrap_iterations": ECONOMIC_BOOTSTRAP_ITERATIONS,
        "economic_bootstrap_block_days": ECONOMIC_BOOTSTRAP_BLOCK_DAYS,
        "economic_bootstrap_seed": ECONOMIC_BOOTSTRAP_SEED,
        "decision_modes": list(DECISION_MODES),
        "specifications": expected_specifications,
    }
    if configured_frontier != expected_frontier:
        raise ValueError("process policy frontier does not match the executable implementation")
    return config


def _calibration_selection_leaderboard(
    database_url: str,
    *,
    selected_model_run_id: str,
    decision_hour: int,
    forecast_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    with connection(database_url) as conn:
        rows = conn.execute(
            """
            SELECT model_run_id::text,candidate,model_sha256,metrics
            FROM weather.model_runs
            WHERE process_id=%s AND decision_hour_local=%s
              AND training_start=%s AND training_end=%s
              AND calibration_start=%s AND calibration_end=%s
            ORDER BY candidate,model_run_id
            """,
            (
                PROCESS_ID,
                decision_hour,
                forecast_contract["training_start"],
                forecast_contract["training_end"],
                forecast_contract["calibration_start"],
                forecast_contract["calibration_end"],
            ),
        ).fetchall()
    metric_name = forecast_contract["metric"]
    secondary_metric_name = forecast_contract["secondary_metric"]
    eligible_candidates = set(forecast_contract["eligible_ml_candidates"])
    diagnostic_candidates = set(forecast_contract["diagnostic_candidates"])
    required_candidates = eligible_candidates | diagnostic_candidates
    leaderboard = []
    observed_candidates = set()
    for row in rows:
        if row["candidate"] not in required_candidates:
            continue
        observed_candidates.add(row["candidate"])
        metric = float(row["metrics"].get(metric_name, math.nan))
        secondary_metric = float(
            row["metrics"].get(secondary_metric_name, math.nan)
        )
        if not math.isfinite(metric) or not math.isfinite(secondary_metric):
            raise ValueError(
                f"model {row['model_run_id']} lacks finite calibration selection metrics"
            )
        leaderboard.append(
            {
                "model_run_id": row["model_run_id"],
                "candidate": row["candidate"],
                "eligible_for_ml_selection": row["candidate"] in eligible_candidates,
                "model_sha256": row["model_sha256"],
                "metric": metric_name,
                "value": metric,
                "secondary_metric": secondary_metric_name,
                "secondary_value": secondary_metric,
            }
        )
    missing = required_candidates - observed_candidates
    if missing:
        raise ValueError(f"required calibration candidates were not trained: {sorted(missing)}")
    leaderboard.sort(
        key=lambda row: (
            row["value"],
            row["secondary_value"],
            row["candidate"],
            row["model_sha256"],
            row["model_run_id"],
        )
    )
    selected = next(
        (row for row in leaderboard if row["model_run_id"] == selected_model_run_id),
        None,
    )
    if selected is None:
        raise ValueError("selected model is absent from the calibration leaderboard")
    best = next(row for row in leaderboard if row["eligible_for_ml_selection"])
    selected_matches_best_artifact = (
        selected["candidate"] == best["candidate"]
        and selected["model_sha256"] == best["model_sha256"]
        and math.isclose(selected["value"], best["value"], abs_tol=1e-12)
        and math.isclose(
            selected["secondary_value"], best["secondary_value"], abs_tol=1e-12
        )
    )
    if not selected_matches_best_artifact:
        raise ValueError(
            f"selected model {selected_model_run_id} is not the best calibrated artifact"
        )
    return leaderboard


def _generate_probabilities(
    settings: Settings,
    *,
    model_run_id: str,
    start: date,
    end: date,
    bootstrap_iterations: int,
    probability_source: str = "selected_ml",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if probability_source not in {"selected_ml", "raw_hrrr"}:
        raise ValueError(f"unsupported probability source: {probability_source}")
    metadata = _model_metadata(settings.database_url, model_run_id)
    if start <= metadata["calibration_end"]:
        raise ValueError(
            f"evaluation start {start} must follow calibration end {metadata['calibration_end']}"
        )
    model_path = Path(metadata["model_uri"])
    model_digest, _ = file_sha256(model_path)
    if model_digest != metadata["model_sha256"]:
        raise ValueError(f"model artifact hash mismatch: {model_run_id}")
    bundle = load_model(model_path)
    decision_hour = int(metadata["decision_hour_local"])
    feature_rows = build_feature_rows(
        settings.database_url, start, end + timedelta(days=1), decision_hour
    )
    markets = _market_rows(settings.database_url, start, end)
    markets_by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for market in markets:
        markets_by_date[market["event_date"]].append(market)

    residual_key = "residuals" if probability_source == "selected_ml" else "raw_residuals"
    residuals = np.asarray(bundle[residual_key], dtype=np.float64)
    if residuals.size < 90:
        raise ValueError("probability bounds require at least 90 calibration residuals")
    # Common random indices make ML-versus-raw differences attributable to the
    # residual distribution rather than Monte Carlo stream noise.
    seed = PROBABILITY_BOOTSTRAP_SEED ^ decision_hour
    bootstrap_indices = _circular_block_indices(
        residuals.size,
        iterations=bootstrap_iterations,
        block_size=min(7, residuals.size),
        seed=seed,
    )
    if probability_source == "selected_ml":
        points = point_prediction(bundle, feature_rows) if feature_rows else np.asarray([])
    else:
        points = raw_point_prediction(feature_rows) if feature_rows else np.asarray([])
    probability_rows: list[dict[str, Any]] = []
    log_losses = []
    brier_scores = []
    point_errors = []
    exact = []
    within_one = []
    for feature, point in zip(feature_rows, points, strict=True):
        day_markets = markets_by_date.get(feature.event_date, [])
        if not day_markets:
            continue
        buckets = [
            (market["bucket_lower_f"], market["bucket_upper_f"]) for market in day_markets
        ]
        estimates = _probability_estimates(
            float(point), residuals, buckets, bootstrap_indices
        )
        labels = np.asarray([int(market["resolved_yes"]) for market in day_markets])
        if int(labels.sum()) != 1:
            raise ValueError(
                f"event date {feature.event_date} does not have exactly one resolved winner"
            )
        probabilities = np.asarray([estimate["yes"] for estimate in estimates])
        winner_probability = float(probabilities[labels == 1][0])
        log_losses.append(-math.log(max(winner_probability, 1e-12)))
        brier_scores.append(float(np.sum((probabilities - labels) ** 2)))
        point_errors.append(float(point) - feature.target_daily_max_f)
        rounded_point = math.floor(float(point) + 0.5)
        exact.append(rounded_point == feature.target_rounded_max_f)
        within_one.append(abs(float(point) - feature.target_daily_max_f) <= 1.0)
        for market, estimate in zip(day_markets, estimates, strict=True):
            probability_rows.append(
                {
                    **market,
                    "model_run_id": model_run_id,
                    "decision_time": feature.decision_time,
                    "decision_hour_local": decision_hour,
                    "point_prediction_f": float(point),
                    "probability_yes": estimate["yes"],
                    "probability_yes_lower": estimate["yes_lower"],
                    "probability_no": estimate["no"],
                    "probability_no_lower": estimate["no_lower"],
                }
            )

    errors = np.asarray(point_errors, dtype=np.float64)
    metrics = {
        "model_run_id": model_run_id,
        "probability_source": probability_source,
        "candidate": metadata["candidate"] if probability_source == "selected_ml" else "raw_hrrr",
        "decision_hour_local": decision_hour,
        "resolved_event_days": len(log_losses),
        "multiclass_log_loss": float(np.mean(log_losses)) if log_losses else float("nan"),
        "multiclass_brier_score": (
            float(np.mean(brier_scores)) if brier_scores else float("nan")
        ),
        "temperature_mae_f": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
        "temperature_rmse_f": (
            float(np.mean(errors**2) ** 0.5) if errors.size else float("nan")
        ),
        "exact_degree_rate": float(np.mean(exact)) if exact else float("nan"),
        "within_one_f_rate": float(np.mean(within_one)) if within_one else float("nan"),
    }
    return probability_rows, metrics


def _persist_predictions(database_url: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with connection(database_url) as conn, conn.transaction(), conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO weather.predictions (
              process_id,model_run_id,market_id,decision_time,probability_yes
            ) VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (process_id,model_run_id,market_id,decision_time) DO UPDATE SET
              probability_yes=EXCLUDED.probability_yes,created_at=now()
            """,
            [
                (
                    PROCESS_ID,
                    row["model_run_id"],
                    row["market_id"],
                    row["decision_time"],
                    row["probability_yes"],
                )
                for row in rows
            ],
        )


def _execution_snapshots(
    database_url: str, start: date, end: date, quantity: float
) -> dict[tuple[str, datetime], dict[str, Any]]:
    with connection(database_url) as conn:
        rows = conn.execute(
            """
            SELECT e.market_id,e.decision_time,e.yes_ask_vwap::double precision,
                   e.no_ask_vwap::double precision,e.yes_best_ask::double precision,
                   e.no_best_ask::double precision,e.source_timestamp,e.quality_flags
            FROM weather.execution_snapshots e
            JOIN weather.temperature_markets m ON m.market_id=e.market_id
            WHERE m.event_date >= %s AND m.event_date <= %s AND e.quantity=%s
            ORDER BY e.decision_time,e.market_id
            """,
            (start, end, quantity),
        ).fetchall()
    return {(row["market_id"], row["decision_time"]): dict(row) for row in rows}


def _side_quality_reasons(
    side: str, snapshot: dict[str, Any] | None, decision_time: datetime
) -> list[str]:
    if snapshot is None:
        return ["missing_execution_snapshot"]
    flags = set(snapshot.get("quality_flags") or [])
    reasons = sorted(flag for flag in flags if str(flag).startswith("pmxt_archive_"))
    normalized = side.lower()
    if f"crossed_{normalized}_book" in flags:
        reasons.append(f"crossed_{normalized}_book")
    reasons.extend(
        sorted(
            flag
            for flag in flags
            if str(flag).startswith((f"missing_{normalized}", f"insufficient_{normalized}"))
        )
    )
    if snapshot.get(f"{normalized}_ask_vwap") is None:
        reasons.append(f"missing_{normalized}_ask_vwap")
    source_timestamp = snapshot.get("source_timestamp")
    if source_timestamp is not None and source_timestamp > decision_time:
        reasons.append("future_source_timestamp")
    return sorted(set(reasons))


def _build_candidates(
    database_url: str,
    *,
    probability_rows: list[dict[str, Any]],
    start: date,
    end: date,
    quantity: float,
    modeled_slippage_per_share: float,
) -> list[dict[str, Any]]:
    snapshots = _execution_snapshots(database_url, start, end, quantity)
    candidates = []
    for row in probability_rows:
        snapshot = snapshots.get((row["market_id"], row["decision_time"]))
        yes_best = snapshot.get("yes_best_ask") if snapshot else None
        no_best = snapshot.get("no_best_ask") if snapshot else None
        yes_mid = None
        if yes_best is not None and no_best is not None:
            yes_mid = min(1.0, max(0.0, (float(yes_best) + 1.0 - float(no_best)) / 2.0))
        for side in ("YES", "NO"):
            normalized = side.lower()
            reasons = _side_quality_reasons(side, snapshot, row["decision_time"])
            ask = snapshot.get(f"{normalized}_ask_vwap") if snapshot else None
            probability = row[f"probability_{normalized}"]
            probability_lower = row[f"probability_{normalized}_lower"]
            resolved_side = (
                bool(row["resolved_yes"])
                if side == "YES"
                else not bool(row["resolved_yes"])
            )
            economics = {
                "fee_per_share": None,
                "modeled_slippage_per_share": None,
                "all_in_cost_per_share": None,
                "break_even_probability": None,
                "model_edge_per_share": None,
                "robust_edge_per_share": None,
                "expected_roi": None,
                "robust_expected_roi": None,
                "realized_net_per_share": None,
            }
            executable = not reasons and ask is not None
            if executable:
                modeled_execution_price = float(ask) + modeled_slippage_per_share
                if modeled_execution_price <= 0 or modeled_execution_price > 1:
                    reasons.append("all_in_cost_outside_contract_payout")
                    executable = False
                else:
                    fee = taker_fee_per_share(
                        modeled_execution_price,
                        quantity=quantity,
                        enabled=bool(row["fees_enabled"]),
                        rate=float(row["fee_rate"]),
                        exponent=float(row["fee_exponent"]),
                    )
                    all_in = modeled_execution_price + fee
                    if all_in > 1:
                        reasons.append("all_in_cost_outside_contract_payout")
                        executable = False
                    else:
                        point_edge = probability - all_in
                        robust_edge = probability_lower - all_in
                        economics = {
                            "fee_per_share": fee,
                            "modeled_slippage_per_share": modeled_slippage_per_share,
                            "all_in_cost_per_share": all_in,
                            "break_even_probability": all_in,
                            "model_edge_per_share": point_edge,
                            "robust_edge_per_share": robust_edge,
                            "expected_roi": point_edge / all_in,
                            "robust_expected_roi": robust_edge / all_in,
                            "realized_net_per_share": (
                                1.0 if resolved_side else 0.0
                            ) - all_in,
                        }
            source_timestamp = snapshot.get("source_timestamp") if snapshot else None
            quote_age_seconds = None
            if source_timestamp is not None and source_timestamp <= row["decision_time"]:
                quote_age_seconds = (
                    row["decision_time"] - source_timestamp
                ).total_seconds()
            candidates.append(
                {
                    "process_id": PROCESS_ID,
                    "model_run_id": row["model_run_id"],
                    "market_id": row["market_id"],
                    "event_date": row["event_date"],
                    "decision_time": row["decision_time"],
                    "decision_hour_local": row["decision_hour_local"],
                    "side": side,
                    "quantity": quantity,
                    "probability": probability,
                    "probability_lower": probability_lower,
                    "market_probability_proxy": yes_mid if side == "YES" else (
                        1.0 - yes_mid if yes_mid is not None else None
                    ),
                    "ask_vwap": float(ask) if ask is not None else None,
                    "fees_enabled": bool(row["fees_enabled"]),
                    "fee_rate": float(row["fee_rate"]),
                    "fee_exponent": float(row["fee_exponent"]),
                    "fee_taker_only": bool(row["fee_taker_only"]),
                    **economics,
                    "resolved_side": resolved_side,
                    "executable": executable,
                    "eligible": False,
                    "selected": False,
                    "source_timestamp": source_timestamp,
                    "quote_age_seconds": quote_age_seconds,
                    "quality_flags": list(snapshot.get("quality_flags") or []) if snapshot else [],
                    "rejection_reasons": reasons,
                }
            )
    return candidates


def _policy_rejection_reasons(
    candidate: dict[str, Any], policy: AsymmetricPolicy
) -> list[str]:
    reasons = list(candidate["rejection_reasons"])
    if not candidate["executable"]:
        return sorted(set(reasons))
    if not policy.minimum_probability <= candidate["probability"] <= policy.maximum_probability:
        reasons.append("probability_outside_policy")
    if not (
        policy.minimum_all_in_cost
        <= candidate["all_in_cost_per_share"]
        <= policy.maximum_all_in_cost
    ):
        reasons.append("all_in_cost_outside_policy")
    if candidate["robust_edge_per_share"] < policy.minimum_robust_edge:
        reasons.append("robust_edge_below_policy")
    if candidate["robust_expected_roi"] < policy.minimum_robust_roi:
        reasons.append("robust_roi_below_policy")
    return sorted(set(reasons))


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, str, datetime, str]:
    return (
        candidate["model_run_id"],
        candidate["market_id"],
        candidate["decision_time"],
        candidate["side"],
    )


def _select_policy_trades(
    candidates: list[dict[str, Any]], policy: AsymmetricPolicy
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, datetime, str], list[str]]]:
    by_date_hour: dict[date, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    reasons_by_key = {}
    for candidate in candidates:
        reasons = _policy_rejection_reasons(candidate, policy)
        reasons_by_key[_candidate_key(candidate)] = reasons
        if not reasons:
            by_date_hour[candidate["event_date"]][candidate["decision_hour_local"]].append(
                candidate
            )

    selected = []
    for event_date in sorted(by_date_hour):
        if policy.decision_mode == "midnight_only":
            pool = by_date_hour[event_date].get(0, [])
        elif policy.decision_mode == "noon_only":
            pool = by_date_hour[event_date].get(12, [])
        else:
            pool = by_date_hour[event_date].get(0, [])
            if not pool:
                pool = by_date_hour[event_date].get(12, [])
        if not pool:
            continue
        selected.append(
            max(
                pool,
                key=lambda row: (
                    row[policy.rank_by],
                    row["robust_edge_per_share"],
                    -row["all_in_cost_per_share"],
                    row["market_id"],
                    row["side"],
                ),
            )
        )
    return selected, reasons_by_key


def _max_drawdown(values: list[float]) -> tuple[float, int, list[dict[str, float]]]:
    cumulative = 0.0
    peak = 0.0
    maximum = 0.0
    current_duration = 0
    maximum_duration = 0
    curve = []
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        if drawdown > 0:
            current_duration += 1
        else:
            current_duration = 0
        maximum_duration = max(maximum_duration, current_duration)
        maximum = max(maximum, drawdown)
        curve.append({"cumulative_net": cumulative, "drawdown": drawdown})
    return maximum, maximum_duration, curve


def _chronological_fold_metrics(
    event_dates: list[date], daily_pnl: list[float], selected_dates: set[date]
) -> list[dict[str, Any]]:
    folds = np.array_split(np.arange(len(event_dates)), 3)
    output = []
    for index, fold in enumerate(folds, start=1):
        values = [daily_pnl[int(position)] for position in fold]
        dates = [event_dates[int(position)] for position in fold]
        output.append(
            {
                "fold": index,
                "start": dates[0] if dates else None,
                "end": dates[-1] if dates else None,
                "trades": sum(day in selected_dates for day in dates),
                "total_net": float(sum(values)),
                "mean_daily_net": float(np.mean(values)) if values else float("nan"),
            }
        )
    return output


def _policy_metrics(
    candidates: list[dict[str, Any]], event_dates: list[date], policy: AsymmetricPolicy
) -> tuple[dict[str, Any], list[dict[str, Any]], dict]:
    selected, reasons_by_key = _select_policy_trades(candidates, policy)
    selected_by_date = {row["event_date"]: row for row in selected}
    daily_pnl = [
        selected_by_date[day]["realized_net_per_share"]
        * selected_by_date[day]["quantity"]
        if day in selected_by_date
        else 0.0
        for day in event_dates
    ]
    daily_return = [
        selected_by_date[day]["realized_net_per_share"]
        / selected_by_date[day]["all_in_cost_per_share"]
        if day in selected_by_date
        else 0.0
        for day in event_dates
    ]
    maximum_drawdown, drawdown_duration, curve_values = _max_drawdown(daily_pnl)
    equity_curve = [
        {
            "event_date": day,
            "daily_net": pnl,
            **curve,
        }
        for day, pnl, curve in zip(event_dates, daily_pnl, curve_values, strict=True)
    ]
    trade_pnl = [row["realized_net_per_share"] * row["quantity"] for row in selected]
    wins = [value for value in trade_pnl if value > 0]
    losses = [value for value in trade_pnl if value <= 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    sorted_profit = sorted(trade_pnl, reverse=True)
    running_loss_streak = 0
    maximum_loss_streak = 0
    for value in trade_pnl:
        running_loss_streak = running_loss_streak + 1 if value <= 0 else 0
        maximum_loss_streak = max(maximum_loss_streak, running_loss_streak)
    worst_trade = min(trade_pnl) if trade_pnl else float("nan")
    median_win = float(np.median(wins)) if wins else float("nan")
    selected_dates = set(selected_by_date)
    folds = _chronological_fold_metrics(event_dates, daily_pnl, selected_dates)
    metrics = {
        "event_days": len(event_dates),
        "trades": len(selected),
        "coverage": len(selected) / len(event_dates) if event_dates else float("nan"),
        "wins": len(wins),
        "losses": len(losses),
        "hit_rate": len(wins) / len(selected) if selected else float("nan"),
        "mean_model_probability": (
            float(np.mean([row["probability"] for row in selected]))
            if selected
            else float("nan")
        ),
        "mean_probability_lower": (
            float(np.mean([row["probability_lower"] for row in selected]))
            if selected
            else float("nan")
        ),
        "mean_break_even_probability": (
            float(np.mean([row["break_even_probability"] for row in selected]))
            if selected
            else float("nan")
        ),
        "mean_ask_vwap": (
            float(np.mean([row["ask_vwap"] for row in selected]))
            if selected
            else float("nan")
        ),
        "maximum_ask_vwap": (
            max((row["ask_vwap"] for row in selected), default=float("nan"))
        ),
        "total_entry_debit": sum(
            row["all_in_cost_per_share"] * row["quantity"] for row in selected
        ),
        "total_net": float(sum(trade_pnl)),
        "mean_net_per_trade": float(np.mean(trade_pnl)) if trade_pnl else float("nan"),
        "mean_net_per_event_day": (
            float(np.mean(daily_pnl)) if daily_pnl else float("nan")
        ),
        "return_on_deployed_capital": (
            sum(trade_pnl)
            / sum(row["all_in_cost_per_share"] * row["quantity"] for row in selected)
            if selected
            else float("nan")
        ),
        "mean_daily_return_on_debit": (
            float(np.mean(daily_return)) if daily_return else float("nan")
        ),
        "lower_90pct_block_bootstrap_mean_daily_return": _bootstrap_lower_mean(
            daily_return
        ),
        "lower_90pct_block_bootstrap_mean_daily_net": _bootstrap_lower_mean(daily_pnl),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "worst_trade": worst_trade,
        "median_winning_trade": median_win,
        "median_wins_erased_by_worst_loss": (
            abs(worst_trade) / median_win
            if wins and math.isfinite(worst_trade) and median_win > 0
            else float("nan")
        ),
        "maximum_consecutive_losses": maximum_loss_streak,
        "maximum_drawdown": maximum_drawdown,
        "maximum_drawdown_duration_days": drawdown_duration,
        "net_without_best_trade": (
            sum(sorted_profit[1:]) if len(sorted_profit) > 1 else float("nan")
        ),
        "net_without_best_three_trades": (
            sum(sorted_profit[3:]) if len(sorted_profit) > 3 else float("nan")
        ),
        "chronological_folds": folds,
        "equity_curve": equity_curve,
    }
    return metrics, selected, reasons_by_key


def _price_band(price: float) -> str:
    for lower, upper, label in PRICE_BANDS:
        if lower <= price < upper:
            return label
    raise ValueError(f"price outside supported bands: {price}")


def _date_clustered_hit_interval(
    rows: list[dict[str, Any]],
    *,
    iterations: int = 5000,
) -> tuple[float | None, float | None]:
    by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_date[row["event_date"]].append(row)
    if len(by_date) < PRICE_CELL_MINIMUM_CLUSTER_DAYS:
        return None, None
    ordered = [by_date[event_date] for event_date in sorted(by_date)]
    wins = np.asarray(
        [sum(int(row["resolved_side"]) for row in day_rows) for day_rows in ordered],
        dtype=np.float64,
    )
    counts = np.asarray([len(day_rows) for day_rows in ordered], dtype=np.float64)
    indices = _circular_block_indices(
        len(ordered),
        iterations=iterations,
        block_size=min(7, len(ordered)),
        seed=ECONOMIC_BOOTSTRAP_SEED,
    )
    sampled_rates = wins[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return (
        float(np.quantile(sampled_rates, 0.10)),
        float(np.quantile(sampled_rates, 0.90)),
    )


def _price_cells(
    candidates: list[dict[str, Any]], selected_keys: set[tuple[str, str, datetime, str]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if not candidate["executable"]:
            continue
        groups[
            (
                candidate["decision_hour_local"],
                candidate["side"],
                _price_band(candidate["ask_vwap"]),
            )
        ].append(candidate)
    output = []
    for (hour, side, band), rows in sorted(groups.items()):
        outcomes = [int(row["resolved_side"]) for row in rows]
        wins = sum(outcomes)
        clustered_p10, clustered_p90 = _date_clustered_hit_interval(rows)
        selected = [row for row in rows if _candidate_key(row) in selected_keys]
        output.append(
            {
                "decision_hour_local": hour,
                "side": side,
                "ask_price_band": band,
                "quotes": len(rows),
                "event_days": len({row["event_date"] for row in rows}),
                "mean_ask_vwap": float(np.mean([row["ask_vwap"] for row in rows])),
                "mean_all_in_cost": float(
                    np.mean([row["all_in_cost_per_share"] for row in rows])
                ),
                "mean_model_probability": float(
                    np.mean([row["probability"] for row in rows])
                ),
                "mean_probability_lower": float(
                    np.mean([row["probability_lower"] for row in rows])
                ),
                "observed_hit_rate": wins / len(rows),
                "hit_rate_inference_sufficient": (
                    len({row["event_date"] for row in rows})
                    >= PRICE_CELL_MINIMUM_CLUSTER_DAYS
                ),
                "hit_rate_minimum_cluster_days": PRICE_CELL_MINIMUM_CLUSTER_DAYS,
                "hit_rate_date_clustered_block_bootstrap_p10": clustered_p10,
                "hit_rate_date_clustered_block_bootstrap_p90": clustered_p90,
                "mean_net_per_share_if_all_bought": float(
                    np.mean([row["realized_net_per_share"] for row in rows])
                ),
                "selected_trades": len(selected),
                "selected_total_net": sum(
                    row["realized_net_per_share"] * row["quantity"] for row in selected
                ),
            }
        )
    return output


def _candidate_coverage(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[(candidate["decision_hour_local"], candidate["side"])].append(candidate)
    output = []
    for (hour, side), rows in sorted(groups.items()):
        rejection_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            for reason in row["rejection_reasons"]:
                rejection_counts[reason] += 1
        output.append(
            {
                "decision_hour_local": hour,
                "side": side,
                "candidate_contracts": len(rows),
                "event_days": len({row["event_date"] for row in rows}),
                "executable_quotes": sum(row["executable"] for row in rows),
                "rejection_counts": dict(sorted(rejection_counts.items())),
            }
        )
    return output


def _compact_policy_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "equity_curve"}


def _selection_key(result: dict[str, Any]) -> tuple:
    metrics = result["metrics"]
    folds = metrics["chronological_folds"]
    sample_sufficient = metrics["trades"] >= 18 and all(fold["trades"] >= 4 for fold in folds)
    lower_bound = metrics["lower_90pct_block_bootstrap_mean_daily_return"]
    mean_return = metrics["mean_daily_return_on_debit"]
    maximum_drawdown = metrics["maximum_drawdown"]
    return (
        sample_sufficient,
        lower_bound if math.isfinite(lower_bound) else -math.inf,
        mean_return if math.isfinite(mean_return) else -math.inf,
        -maximum_drawdown if math.isfinite(maximum_drawdown) else -math.inf,
        -result["policy"]["maximum_all_in_cost"],
        result["policy"]["name"],
    )


def _label_metrics(database_url: str, start: date, end: date) -> dict[str, Any]:
    with connection(database_url) as conn:
        row = conn.execute(
            """
            SELECT count(*) FILTER (WHERE winner_matches_station IS NOT NULL)::int AS compared,
                   count(*) FILTER (WHERE winner_matches_station)::int AS matched,
                   (
                     SELECT count(DISTINCT event_date)::int
                     FROM weather.temperature_markets
                     WHERE event_date >= %s AND event_date <= %s
                       AND resolved_yes IS NOT NULL
                   ) AS resolved_event_days
            FROM weather.label_reconciliation
            WHERE process_id=%s AND event_date >= %s AND event_date <= %s
            """,
            (start, end, PROCESS_ID, start, end),
        ).fetchone()
    compared = int(row["compared"] or 0)
    matched = int(row["matched"] or 0)
    resolved_event_days = int(row["resolved_event_days"] or 0)
    return {
        "resolved_event_days": resolved_event_days,
        "days_compared": compared,
        "days_matched": matched,
        "coverage": compared / resolved_event_days if resolved_event_days else float("nan"),
        "agreement": matched / compared if compared else float("nan"),
    }


def _persist_policy_run(
    settings: Settings,
    *,
    policy_run_id: str,
    decision_models: dict[str, str],
    policy: AsymmetricPolicy,
    discovery_start: date,
    discovery_end: date,
    evaluation_start: date,
    evaluation_end: date,
    quantity: float,
    discovery_metrics: dict[str, Any],
    evaluation_metrics: dict[str, Any],
    qualified: bool,
    report_path: Path,
    candidates: list[dict[str, Any]],
    split_by_key: dict[tuple[str, str, datetime, str], str],
    eligible_keys: set[tuple[str, str, datetime, str]],
    selected_keys: set[tuple[str, str, datetime, str]],
    rejection_reasons: dict[tuple[str, str, datetime, str], list[str]],
) -> None:
    with connection(settings.database_url) as conn, conn.transaction():
        conn.execute(
            """
            INSERT INTO weather.asymmetric_policy_runs (
              policy_run_id,process_id,discovery_start,discovery_end,evaluation_start,
              evaluation_end,quantity,decision_models,policy,discovery_metrics,
              evaluation_metrics,qualified,report_uri
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                policy_run_id,
                PROCESS_ID,
                discovery_start,
                discovery_end,
                evaluation_start,
                evaluation_end,
                quantity,
                psycopg.types.json.Jsonb(_json_safe(decision_models)),
                psycopg.types.json.Jsonb(_json_safe(asdict(policy))),
                psycopg.types.json.Jsonb(_json_safe(discovery_metrics)),
                psycopg.types.json.Jsonb(_json_safe(evaluation_metrics)),
                qualified,
                str(report_path),
            ),
        )
        values = []
        for candidate in candidates:
            key = _candidate_key(candidate)
            values.append(
                (
                    PROCESS_ID,
                    policy_run_id,
                    candidate["model_run_id"],
                    candidate["market_id"],
                    split_by_key[key],
                    candidate["event_date"],
                    candidate["decision_time"],
                    candidate["decision_hour_local"],
                    candidate["side"],
                    candidate["quantity"],
                    candidate["probability"],
                    candidate["probability_lower"],
                    candidate["market_probability_proxy"],
                    candidate["ask_vwap"],
                    candidate["fees_enabled"],
                    candidate["fee_rate"],
                    candidate["fee_exponent"],
                    candidate["fee_taker_only"],
                    candidate["fee_per_share"],
                    candidate["modeled_slippage_per_share"],
                    candidate["all_in_cost_per_share"],
                    candidate["break_even_probability"],
                    candidate["model_edge_per_share"],
                    candidate["robust_edge_per_share"],
                    candidate["expected_roi"],
                    candidate["robust_expected_roi"],
                    candidate["resolved_side"],
                    candidate["executable"],
                    key in eligible_keys,
                    key in selected_keys,
                    candidate["realized_net_per_share"],
                    candidate["source_timestamp"],
                    candidate["quote_age_seconds"],
                    psycopg.types.json.Jsonb(candidate["quality_flags"]),
                    psycopg.types.json.Jsonb(rejection_reasons.get(key, [])),
                )
            )
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO weather.asymmetric_candidate_ledger (
                  process_id,policy_run_id,model_run_id,market_id,split,event_date,
                  decision_time,decision_hour_local,side,quantity,probability,
                  probability_lower,market_probability_proxy,ask_vwap,fees_enabled,
                  fee_rate,fee_exponent,fee_taker_only,fee_per_share,
                  modeled_slippage_per_share,all_in_cost_per_share,break_even_probability,
                  model_edge_per_share,robust_edge_per_share,expected_roi,
                  robust_expected_roi,resolved_side,executable,eligible,selected,
                  realized_net_per_share,source_timestamp,quote_age_seconds,
                  quality_flags,rejection_reasons
                ) VALUES (
                  %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                  %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s
                )
                """,
                values,
            )


def run_asymmetric_benchmark(
    settings: Settings,
    *,
    midnight_model_run_id: str,
    noon_model_run_id: str,
    discovery_start: date,
    discovery_end: date,
    evaluation_start: date,
    evaluation_end: date,
    quantity: float = 5.0,
    modeled_slippage_per_share: float = 0.01,
    probability_bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    if not discovery_start <= discovery_end < evaluation_start <= evaluation_end:
        raise ValueError("discovery and evaluation ranges must be disjoint and chronological")
    if quantity not in (1.0, 5.0, 10.0):
        raise ValueError("quantity must be 1, 5, or 10 shares")
    if not 0 <= modeled_slippage_per_share <= 0.10:
        raise ValueError("modeled slippage must be between zero and ten cents")
    process_config = _validate_process_contract(
        settings.database_url,
        discovery_start=discovery_start,
        discovery_end=discovery_end,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        quantity=quantity,
        modeled_slippage_per_share=modeled_slippage_per_share,
        probability_bootstrap_iterations=probability_bootstrap_iterations,
    )
    decision_models = {"0": midnight_model_run_id, "12": noon_model_run_id}
    probability_rows = []
    raw_probability_rows = []
    forecast_metrics = {}
    raw_forecast_metrics = {}
    decision_model_metadata = {}
    calibration_selection = {}
    for expected_hour, model_run_id in ((0, midnight_model_run_id), (12, noon_model_run_id)):
        metadata = _model_metadata(settings.database_url, model_run_id)
        if int(metadata["decision_hour_local"]) != expected_hour:
            raise ValueError(f"model {model_run_id} does not belong to hour {expected_hour}")
        forecast_contract = process_config["forecast_selection"]
        if metadata["candidate"] not in forecast_contract["eligible_ml_candidates"]:
            raise ValueError(f"model {model_run_id} is not an eligible ML candidate")
        for range_name in (
            "training_start",
            "training_end",
            "calibration_start",
            "calibration_end",
        ):
            if metadata[range_name].isoformat() != forecast_contract[range_name]:
                raise ValueError(
                    f"model {model_run_id} does not match process {range_name}"
                )
        calibration_selection[str(expected_hour)] = _calibration_selection_leaderboard(
            settings.database_url,
            selected_model_run_id=model_run_id,
            decision_hour=expected_hour,
            forecast_contract=forecast_contract,
        )
        decision_model_metadata[str(expected_hour)] = {
            "model_run_id": metadata["model_run_id"],
            "candidate": metadata["candidate"],
            "training_start": metadata["training_start"],
            "training_end": metadata["training_end"],
            "calibration_start": metadata["calibration_start"],
            "calibration_end": metadata["calibration_end"],
            "model_sha256": metadata["model_sha256"],
            "calibration_metrics": metadata["metrics"],
        }
        discovery_rows, discovery_forecast_metrics = _generate_probabilities(
            settings,
            model_run_id=model_run_id,
            start=discovery_start,
            end=discovery_end,
            bootstrap_iterations=probability_bootstrap_iterations,
        )
        evaluation_rows, evaluation_forecast_metrics = _generate_probabilities(
            settings,
            model_run_id=model_run_id,
            start=evaluation_start,
            end=evaluation_end,
            bootstrap_iterations=probability_bootstrap_iterations,
        )
        probability_rows.extend(discovery_rows)
        probability_rows.extend(evaluation_rows)
        forecast_metrics[str(expected_hour)] = {
            "discovery": discovery_forecast_metrics,
            "evaluation": evaluation_forecast_metrics,
        }
        raw_discovery_rows, raw_discovery_metrics = _generate_probabilities(
            settings,
            model_run_id=model_run_id,
            start=discovery_start,
            end=discovery_end,
            bootstrap_iterations=probability_bootstrap_iterations,
            probability_source="raw_hrrr",
        )
        raw_evaluation_rows, raw_evaluation_metrics = _generate_probabilities(
            settings,
            model_run_id=model_run_id,
            start=evaluation_start,
            end=evaluation_end,
            bootstrap_iterations=probability_bootstrap_iterations,
            probability_source="raw_hrrr",
        )
        raw_probability_rows.extend(raw_discovery_rows)
        raw_probability_rows.extend(raw_evaluation_rows)
        raw_forecast_metrics[str(expected_hour)] = {
            "discovery": raw_discovery_metrics,
            "evaluation": raw_evaluation_metrics,
        }
    _persist_predictions(settings.database_url, probability_rows)

    discovery_probability_rows = [
        row for row in probability_rows if discovery_start <= row["event_date"] <= discovery_end
    ]
    evaluation_probability_rows = [
        row for row in probability_rows if evaluation_start <= row["event_date"] <= evaluation_end
    ]
    raw_discovery_probability_rows = [
        row
        for row in raw_probability_rows
        if discovery_start <= row["event_date"] <= discovery_end
    ]
    raw_evaluation_probability_rows = [
        row
        for row in raw_probability_rows
        if evaluation_start <= row["event_date"] <= evaluation_end
    ]
    discovery_candidates = _build_candidates(
        settings.database_url,
        probability_rows=discovery_probability_rows,
        start=discovery_start,
        end=discovery_end,
        quantity=quantity,
        modeled_slippage_per_share=modeled_slippage_per_share,
    )
    evaluation_candidates = _build_candidates(
        settings.database_url,
        probability_rows=evaluation_probability_rows,
        start=evaluation_start,
        end=evaluation_end,
        quantity=quantity,
        modeled_slippage_per_share=modeled_slippage_per_share,
    )
    raw_discovery_candidates = _build_candidates(
        settings.database_url,
        probability_rows=raw_discovery_probability_rows,
        start=discovery_start,
        end=discovery_end,
        quantity=quantity,
        modeled_slippage_per_share=modeled_slippage_per_share,
    )
    raw_evaluation_candidates = _build_candidates(
        settings.database_url,
        probability_rows=raw_evaluation_probability_rows,
        start=evaluation_start,
        end=evaluation_end,
        quantity=quantity,
        modeled_slippage_per_share=modeled_slippage_per_share,
    )
    discovery_dates = _date_range(discovery_start, discovery_end)
    evaluation_dates = _date_range(evaluation_start, evaluation_end)

    frontier = []
    for policy in policy_frontier():
        metrics, _, _ = _policy_metrics(discovery_candidates, discovery_dates, policy)
        frontier.append({"policy": asdict(policy), "metrics": _compact_policy_metrics(metrics)})
    selected_frontier = max(frontier, key=_selection_key)
    frozen_policy = AsymmetricPolicy(**selected_frontier["policy"])
    discovery_metrics, discovery_trades, discovery_reasons = _policy_metrics(
        discovery_candidates, discovery_dates, frozen_policy
    )
    evaluation_metrics, evaluation_trades, evaluation_reasons = _policy_metrics(
        evaluation_candidates, evaluation_dates, frozen_policy
    )
    selected_keys = {
        _candidate_key(row) for row in discovery_trades + evaluation_trades
    }
    eligible_keys = {
        key
        for key, reasons in {**discovery_reasons, **evaluation_reasons}.items()
        if not reasons
    }

    uncapped_policy = replace(
        frozen_policy,
        name=f"{frozen_policy.name}:uncapped",
        maximum_all_in_cost=0.999999,
    )
    legacy_policy = AsymmetricPolicy(
        name=f"legacy_high_accuracy:{frozen_policy.decision_mode}",
        decision_mode=frozen_policy.decision_mode,
        minimum_all_in_cost=0.75,
        maximum_all_in_cost=0.999999,
        minimum_robust_edge=-1.0,
        minimum_robust_roi=-1.0,
        minimum_probability=0.90,
        rank_by="probability",
    )
    raw_discovery_policy_metrics, _, _ = _policy_metrics(
        raw_discovery_candidates, discovery_dates, frozen_policy
    )
    raw_evaluation_policy_metrics, _, _ = _policy_metrics(
        raw_evaluation_candidates, evaluation_dates, frozen_policy
    )
    comparator_metrics = {
        "raw_hrrr_frozen_policy": {
            "policy": asdict(frozen_policy),
            "policy_reoptimized": False,
            "forecast_metrics": raw_forecast_metrics,
            "discovery": _compact_policy_metrics(raw_discovery_policy_metrics),
            "evaluation": _compact_policy_metrics(raw_evaluation_policy_metrics),
            "evaluation_ml_net_lift": (
                evaluation_metrics["total_net"]
                - raw_evaluation_policy_metrics["total_net"]
            ),
            "evaluation_ml_return_on_capital_lift": (
                evaluation_metrics["return_on_deployed_capital"]
                - raw_evaluation_policy_metrics["return_on_deployed_capital"]
            ),
        }
    }
    for name, policy in (("uncapped_value", uncapped_policy), ("high_accuracy", legacy_policy)):
        discovery_comparator, _, _ = _policy_metrics(
            discovery_candidates, discovery_dates, policy
        )
        evaluation_comparator, _, _ = _policy_metrics(
            evaluation_candidates, evaluation_dates, policy
        )
        comparator_metrics[name] = {
            "policy": asdict(policy),
            "discovery": _compact_policy_metrics(discovery_comparator),
            "evaluation": _compact_policy_metrics(evaluation_comparator),
        }

    cost_stress = {}
    for stress_slippage in sorted(
        {0.0, 0.005, 0.01, 0.02, modeled_slippage_per_share}
    ):
        stress_discovery_candidates = _build_candidates(
            settings.database_url,
            probability_rows=discovery_probability_rows,
            start=discovery_start,
            end=discovery_end,
            quantity=quantity,
            modeled_slippage_per_share=stress_slippage,
        )
        stress_evaluation_candidates = _build_candidates(
            settings.database_url,
            probability_rows=evaluation_probability_rows,
            start=evaluation_start,
            end=evaluation_end,
            quantity=quantity,
            modeled_slippage_per_share=stress_slippage,
        )
        stress_discovery, _, _ = _policy_metrics(
            stress_discovery_candidates, discovery_dates, frozen_policy
        )
        stress_evaluation, _, _ = _policy_metrics(
            stress_evaluation_candidates, evaluation_dates, frozen_policy
        )
        cost_stress[f"{stress_slippage:.3f}"] = {
            "modeled_slippage_per_share": stress_slippage,
            "policy_reoptimized": False,
            "discovery": _compact_policy_metrics(stress_discovery),
            "evaluation": _compact_policy_metrics(stress_evaluation),
        }

    labels = _label_metrics(settings.database_url, discovery_start, evaluation_end)
    discovery_folds = discovery_metrics["chronological_folds"]
    discovery_sample_sufficient = discovery_metrics["trades"] >= 18 and all(
        fold["trades"] >= 4 for fold in discovery_folds
    )
    checks = {
        "label_coverage_at_least_95pct_and_100_days": (
            labels["days_compared"] >= 100 and labels["coverage"] >= 0.95
        ),
        "label_agreement_at_least_99pct": labels["agreement"] >= 0.99,
        "discovery_sample_sufficient": discovery_sample_sufficient,
        "evaluation_positive_total_net": evaluation_metrics["total_net"] > 0,
        "evaluation_positive_return_on_capital": (
            evaluation_metrics["return_on_deployed_capital"] > 0
        ),
        "evaluation_positive_lower_90pct_daily_net": (
            evaluation_metrics["lower_90pct_block_bootstrap_mean_daily_net"] > 0
        ),
        "evaluation_positive_without_best_trade": (
            evaluation_metrics["net_without_best_trade"] > 0
        ),
        "evaluation_ml_net_exceeds_raw_hrrr": (
            evaluation_metrics["total_net"]
            > raw_evaluation_policy_metrics["total_net"]
        ),
        "low_price_cap_at_most_25c": frozen_policy.maximum_all_in_cost <= 0.25,
        "at_least_50_confirming_trades": evaluation_metrics["trades"] >= 50,
        "at_least_10_wins_and_losses": (
            evaluation_metrics["wins"] >= 10 and evaluation_metrics["losses"] >= 10
        ),
    }
    pilot_edge_supported = all(
        checks[key]
        for key in (
            "label_coverage_at_least_95pct_and_100_days",
            "label_agreement_at_least_99pct",
            "evaluation_positive_total_net",
            "evaluation_positive_return_on_capital",
            "evaluation_ml_net_exceeds_raw_hrrr",
            "low_price_cap_at_most_25c",
        )
    )
    production_qualified = all(checks.values())
    discovery_selected_keys = {_candidate_key(row) for row in discovery_trades}
    evaluation_selected_keys = {_candidate_key(row) for row in evaluation_trades}
    report = {
        "schema_version": "nyc-temperature-asymmetric-expectancy-v1",
        "objective": "positive_net_expectancy_at_low_executable_cost_without_accuracy_gate",
        "process_id": PROCESS_ID,
        "frozen_process_config": process_config,
        "decision_models": decision_models,
        "decision_model_metadata": decision_model_metadata,
        "calibration_model_selection": calibration_selection,
        "ranges": {
            "discovery_start": discovery_start,
            "discovery_end": discovery_end,
            "evaluation_start": evaluation_start,
            "evaluation_end": evaluation_end,
        },
        "execution": {
            "quantity": quantity,
            "modeled_slippage_per_share": modeled_slippage_per_share,
            "slippage_treatment": "adverse_execution_price_move_before_dynamic_fee",
            "fee_method": "captured_schedule_at_slipped_order_vwap_rounded_to_5dp",
            "level_weighted_match_fees_available": False,
            "vwap_fee_curve_bias_before_rounding": "conservative_by_concavity",
            "maker_fills_assumed": False,
            "maximum_positions_per_event_day": 1,
        },
        "forecast_metrics": forecast_metrics,
        "labels": labels,
        "policy_frontier": frontier,
        "policy_selection_contract": {
            "frontier_size": len(frontier),
            "selection_sample": "discovery_only",
            "primary_objective": "lower_90pct_block_bootstrap_mean_daily_return",
            "economic_bootstrap_confidence": ECONOMIC_BOOTSTRAP_CONFIDENCE,
            "economic_bootstrap_iterations": ECONOMIC_BOOTSTRAP_ITERATIONS,
            "economic_bootstrap_block_days": ECONOMIC_BOOTSTRAP_BLOCK_DAYS,
            "economic_bootstrap_seed": ECONOMIC_BOOTSTRAP_SEED,
            "minimum_preferred_discovery_trades": 18,
            "minimum_preferred_trades_per_chronological_third": 4,
            "evaluation_reoptimization_allowed": False,
        },
        "frozen_policy": asdict(frozen_policy),
        "discovery": {
            "metrics": discovery_metrics,
            "candidate_coverage": _candidate_coverage(discovery_candidates),
            "price_cells": _price_cells(discovery_candidates, discovery_selected_keys),
        },
        "evaluation": {
            "metrics": evaluation_metrics,
            "candidate_coverage": _candidate_coverage(evaluation_candidates),
            "price_cells": _price_cells(evaluation_candidates, evaluation_selected_keys),
        },
        "comparators": comparator_metrics,
        "frozen_policy_cost_stress": cost_stress,
        "qualification_checks": checks,
        "pilot_edge_supported": pilot_edge_supported,
        "production_qualified": production_qualified,
    }
    policy_run_id = str(uuid.uuid4())
    settings.report_directory.mkdir(parents=True, exist_ok=True)
    report_path = settings.report_directory / f"asymmetric-benchmark-{policy_run_id}.json"
    report["policy_run_id"] = policy_run_id
    report["report_uri"] = str(report_path)
    report = _json_safe(report)

    split_by_key = {
        **{_candidate_key(row): "discovery" for row in discovery_candidates},
        **{_candidate_key(row): "evaluation" for row in evaluation_candidates},
    }
    all_reasons = {**discovery_reasons, **evaluation_reasons}
    _persist_policy_run(
        settings,
        policy_run_id=policy_run_id,
        decision_models=decision_models,
        policy=frozen_policy,
        discovery_start=discovery_start,
        discovery_end=discovery_end,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        quantity=quantity,
        discovery_metrics=_compact_policy_metrics(discovery_metrics),
        evaluation_metrics=_compact_policy_metrics(evaluation_metrics),
        qualified=production_qualified,
        report_path=report_path,
        candidates=discovery_candidates + evaluation_candidates,
        split_by_key=split_by_key,
        eligible_keys=eligible_keys,
        selected_keys=selected_keys,
        rejection_reasons=all_reasons,
    )
    temporary = report_path.with_suffix(".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n")
    temporary.replace(report_path)
    return report
