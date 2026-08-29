from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss

from . import PROCESS_ID
from .asymmetric_benchmark import _build_candidates
from .config import Settings
from .database import connection
from .fees import taker_fee_per_share
from .sources import file_sha256
from .tail_calibration_tournament import _market_probability_rows

SCHEMA_VERSION = "nyc-temperature-pmxt-full-market-tournament-v1"
FEATURE_SCHEMA_VERSION = "nyc-temperature-pmxt-market-state-v1"
SOURCE_WEATHER_CANDIDATE = "tail_calibrated_ensemble"
WINDOW_MINUTES = (5, 15, 30, 60)
REGULARIZATION_GRID = (0.1, 1.0, 10.0, 100.0)
EDGE_THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10)
BOOTSTRAP_ITERATIONS = 200
BOOTSTRAP_SEED = 82_608_2026
QUANTITY = 5.0
MODELED_SLIPPAGE = 0.01
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
MODEL_FEATURE_SETS = {
    "weather_market_offset": (
        "weather_residual_midnight",
        "weather_residual_noon",
    ),
    "market_state_without_pmxt": (
        "weather_residual_midnight",
        "weather_residual_noon",
        "market_uncertainty",
        "market_extremity",
        "ask_overround",
        "yes_vwap_depth_cost",
        "no_vwap_depth_cost",
        "quote_age_minutes",
    ),
}
PMXT_FEATURE_NAMES = tuple(
    feature
    for minutes in WINDOW_MINUTES
    for feature in (
        f"pmxt_log_trade_count_{minutes}m",
        f"pmxt_log_trade_size_{minutes}m",
        f"pmxt_outcome_size_imbalance_{minutes}m",
        f"pmxt_signed_flow_imbalance_{minutes}m",
        f"pmxt_confirmation_return_{minutes}m",
        f"pmxt_price_range_{minutes}m",
    )
) + (
    "pmxt_yes_last_market_divergence",
    "pmxt_no_last_market_divergence",
    "pmxt_yes_last_age_minutes",
    "pmxt_no_last_age_minutes",
    "pmxt_yes_trade_missing",
    "pmxt_no_trade_missing",
)
MODEL_FEATURE_SETS["pmxt_market_state"] = (
    *MODEL_FEATURE_SETS["market_state_without_pmxt"],
    *PMXT_FEATURE_NAMES,
)


@dataclass(frozen=True)
class OffsetFit:
    feature_names: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    coefficients: np.ndarray
    regularization: float
    converged: bool


@dataclass(frozen=True)
class FoldContract:
    name: str
    training_end: date
    evaluation_start: date
    evaluation_end: date


OUTER_FOLDS = (
    FoldContract("june", date(2026, 5, 31), date(2026, 6, 1), date(2026, 6, 30)),
    FoldContract("early_july", date(2026, 6, 30), date(2026, 7, 1), date(2026, 7, 15)),
    FoldContract("late_july_august", date(2026, 7, 15), date(2026, 7, 16), date(2026, 8, 4)),
)


def _clip_probability(value: float) -> float:
    return min(0.995, max(0.005, float(value)))


def _logit(value: float) -> float:
    clipped = _clip_probability(value)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_digest(rows: list[dict[str, Any]]) -> str:
    payload = []
    for row in sorted(rows, key=lambda item: (item["decision_time"], item["market_id"])):
        payload.append(
            {
                "event_date": row["event_date"].isoformat(),
                "decision_time": row["decision_time"].isoformat(),
                "market_id": row["market_id"],
                "resolved_yes": row["resolved_yes"],
                "weather_probability_yes": row["weather_probability_yes"],
                "market_probability_yes": row["market_probability_yes"],
                "yes_cost": row["candidates"]["YES"].get("all_in_cost_per_share"),
                "no_cost": row["candidates"]["NO"].get("all_in_cost_per_share"),
                "features": row["features"],
                "coverage_manifest": row["coverage_manifest"],
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_weather_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed_sha256, _ = file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"source weather artifact SHA-256 mismatch: {observed_sha256} != {expected_sha256}"
        )
    artifact = joblib.load(path)
    if artifact.get("schema_version") != "nyc-temperature-tail-calibration-tournament-v1":
        raise ValueError("source weather artifact has an incompatible schema")
    if artifact.get("selected_champion") != SOURCE_WEATHER_CANDIDATE:
        raise ValueError("source weather artifact does not freeze the required champion")
    if set(artifact.get("hours", {})) != {0, 12}:
        raise ValueError("source weather artifact must contain midnight and noon models")
    return artifact


def _coverage_and_trades(
    database_url: str, start: date, end: date
) -> tuple[dict[tuple[str, datetime], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    with connection(database_url) as conn:
        coverage_rows = conn.execute(
            """
            SELECT c.market_id,c.decision_time,c.yes_token_id,c.no_token_id,
                   c.archive_manifest_sha256,c.expected_archive_hours,
                   c.available_archive_hours,c.quality_flags
            FROM weather.pmxt_trade_window_coverage c
            JOIN weather.temperature_markets m ON m.market_id=c.market_id
            WHERE c.process_id=%s AND m.event_date >= %s AND m.event_date <= %s
            ORDER BY c.decision_time,c.market_id
            """,
            (PROCESS_ID, start, end),
        ).fetchall()
        trade_rows = conn.execute(
            """
            SELECT t.market_id,t.token_id,t.outcome,t.source_timestamp,
                   t.provider_received_at,t.price::double precision,
                   t.size::double precision,t.trade_side
            FROM weather.pmxt_last_trade_prices t
            JOIN weather.temperature_markets m ON m.market_id=t.market_id
            WHERE t.process_id=%s AND m.event_date >= %s AND m.event_date <= %s
            ORDER BY t.market_id,t.provider_received_at,t.source_timestamp,t.event_id
            """,
            (PROCESS_ID, start, end),
        ).fetchall()
    coverage = {
        (row["market_id"], row["decision_time"]): dict(row) for row in coverage_rows
    }
    trades: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trade_rows:
        trades[row["market_id"]].append(dict(row))
    return coverage, trades


def _best_asks(
    database_url: str, start: date, end: date
) -> dict[tuple[str, datetime], dict[str, float | None]]:
    with connection(database_url) as conn:
        rows = conn.execute(
            """
            SELECT e.market_id,e.decision_time,
                   e.yes_best_ask::double precision,e.no_best_ask::double precision
            FROM weather.execution_snapshots e
            JOIN weather.temperature_markets m ON m.market_id=e.market_id
            WHERE m.event_date >= %s AND m.event_date <= %s AND e.quantity=%s
            ORDER BY e.decision_time,e.market_id
            """,
            (start, end, QUANTITY),
        ).fetchall()
    return {
        (row["market_id"], row["decision_time"]): {
            "yes_best_ask": row["yes_best_ask"],
            "no_best_ask": row["no_best_ask"],
        }
        for row in rows
    }


def _side_window(rows: list[dict[str, Any]], outcome: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["outcome"]).upper() == outcome]


def _return(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    return float(rows[-1]["price"] - rows[0]["price"])


def _range(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    prices = [float(row["price"]) for row in rows]
    return max(prices) - min(prices)


def _trade_features(
    all_trades: list[dict[str, Any]], decision_time: datetime, market_probability_yes: float
) -> dict[str, float]:
    causal = [
        row
        for row in all_trades
        if row["source_timestamp"] <= decision_time
        and row["provider_received_at"] <= decision_time
        and row["provider_received_at"] > decision_time - timedelta(minutes=60)
    ]
    features: dict[str, float] = {}
    for minutes in WINDOW_MINUTES:
        rows = [
            row
            for row in causal
            if row["provider_received_at"] > decision_time - timedelta(minutes=minutes)
        ]
        yes = _side_window(rows, "YES")
        no = _side_window(rows, "NO")
        total_size = sum(float(row["size"]) for row in rows)
        yes_size = sum(float(row["size"]) for row in yes)
        no_size = sum(float(row["size"]) for row in no)
        signed_size = sum(
            float(row["size"])
            * (1.0 if str(row["trade_side"]).upper() == "BUY" else -1.0)
            for row in rows
            if str(row["trade_side"]).upper() in {"BUY", "SELL"}
        )
        features[f"pmxt_log_trade_count_{minutes}m"] = math.log1p(len(rows))
        features[f"pmxt_log_trade_size_{minutes}m"] = math.log1p(total_size)
        features[f"pmxt_outcome_size_imbalance_{minutes}m"] = (
            (yes_size - no_size) / total_size if total_size else 0.0
        )
        features[f"pmxt_signed_flow_imbalance_{minutes}m"] = (
            signed_size / total_size if total_size else 0.0
        )
        features[f"pmxt_confirmation_return_{minutes}m"] = _return(yes) - _return(no)
        features[f"pmxt_price_range_{minutes}m"] = max(_range(yes), _range(no))
    yes_all = _side_window(causal, "YES")
    no_all = _side_window(causal, "NO")
    features["pmxt_yes_last_market_divergence"] = (
        float(yes_all[-1]["price"]) - market_probability_yes if yes_all else 0.0
    )
    features["pmxt_no_last_market_divergence"] = (
        float(no_all[-1]["price"]) - (1.0 - market_probability_yes) if no_all else 0.0
    )
    features["pmxt_yes_last_age_minutes"] = (
        min(60.0, (decision_time - yes_all[-1]["provider_received_at"]).total_seconds() / 60)
        if yes_all
        else 60.0
    )
    features["pmxt_no_last_age_minutes"] = (
        min(60.0, (decision_time - no_all[-1]["provider_received_at"]).total_seconds() / 60)
        if no_all
        else 60.0
    )
    features["pmxt_yes_trade_missing"] = float(not yes_all)
    features["pmxt_no_trade_missing"] = float(not no_all)
    return features


def build_training_rows(
    settings: Settings,
    *,
    source_weather_artifact_path: Path,
    source_weather_artifact_sha256: str,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    source = _source_weather_artifact(
        source_weather_artifact_path, source_weather_artifact_sha256
    )
    probability_rows = _market_probability_rows(
        settings,
        source["hours"],
        start=start,
        end=end,
        tournament_id=f"pmxt-source-{source['tournament_id']}",
    )[SOURCE_WEATHER_CANDIDATE]
    candidates = _build_candidates(
        settings.database_url,
        probability_rows=probability_rows,
        start=start,
        end=end,
        quantity=QUANTITY,
        modeled_slippage_per_share=MODELED_SLIPPAGE,
    )
    by_contract: dict[tuple[str, datetime], dict[str, dict[str, Any]]] = defaultdict(dict)
    for candidate in candidates:
        by_contract[(candidate["market_id"], candidate["decision_time"])][
            candidate["side"]
        ] = candidate
    coverage, trades = _coverage_and_trades(settings.database_url, start, end)
    best_asks = _best_asks(settings.database_url, start, end)
    rows = []
    for (market_id, decision_time), sides in sorted(
        by_contract.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        yes = sides.get("YES")
        no = sides.get("NO")
        coverage_row = coverage.get((market_id, decision_time))
        if yes is None or no is None or coverage_row is None:
            continue
        if not yes["executable"] and not no["executable"]:
            continue
        if yes.get("market_probability_proxy") is None:
            continue
        flags = list(coverage_row.get("quality_flags") or [])
        if coverage_row["available_archive_hours"] != coverage_row["expected_archive_hours"]:
            continue
        if any(str(flag).startswith(("pmxt_archive_missing_", "pmxt_archive_corrupt_")) for flag in flags):
            continue
        weather_probability = float(yes["probability"])
        market_probability = float(yes["market_probability_proxy"])
        delta = _logit(weather_probability) - _logit(market_probability)
        hour = int(yes["decision_hour_local"])
        snapshot = best_asks.get((market_id, decision_time), {})
        yes_best = snapshot.get("yes_best_ask")
        no_best = snapshot.get("no_best_ask")
        yes_ask = yes.get("ask_vwap")
        no_ask = no.get("ask_vwap")
        features = {
            "weather_residual_midnight": delta if hour == 0 else 0.0,
            "weather_residual_noon": delta if hour == 12 else 0.0,
            "market_uncertainty": market_probability * (1.0 - market_probability),
            "market_extremity": abs(_logit(market_probability)),
            "ask_overround": (
                float(yes_best) + float(no_best) - 1.0
                if yes_best is not None and no_best is not None
                else 0.0
            ),
            "yes_vwap_depth_cost": max(
                0.0,
                float(yes_ask) - float(yes_best),
            )
            if yes_ask is not None and yes_best is not None
            else 0.0,
            "no_vwap_depth_cost": max(
                0.0,
                float(no_ask) - float(no_best),
            )
            if no_ask is not None and no_best is not None
            else 0.0,
            "quote_age_minutes": min(60.0, float(yes["quote_age_seconds"] or 0.0) / 60.0),
        }
        features.update(_trade_features(trades.get(market_id, []), decision_time, market_probability))
        rows.append(
            {
                "event_date": yes["event_date"],
                "decision_time": decision_time,
                "decision_hour_local": hour,
                "market_id": market_id,
                "resolved_yes": bool(yes["resolved_side"]),
                "weather_probability_yes": weather_probability,
                "market_probability_yes": market_probability,
                "features": features,
                "coverage_manifest": coverage_row["archive_manifest_sha256"],
                "candidates": {"YES": yes, "NO": no},
            }
        )
    if not rows:
        raise ValueError("no complete PMXT, executable, weather, and outcome rows were available")
    return rows


def _event_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    keys = [
        (row["event_date"], row.get("_bootstrap_instance", 0))
        for row in rows
    ]
    counts = Counter(keys)
    return np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)


def _matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[float(row["features"][name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )


def fit_offset_model(
    rows: list[dict[str, Any]], feature_names: tuple[str, ...], regularization: float
) -> OffsetFit:
    event_instances = {
        (row["event_date"], row.get("_bootstrap_instance", 0))
        for row in rows
    }
    if len(event_instances) < 20:
        raise ValueError("offset model requires at least 20 event dates")
    raw = _matrix(rows, feature_names)
    means = np.mean(raw, axis=0)
    scales = np.std(raw, axis=0)
    scales[scales < 1e-8] = 1.0
    matrix = (raw - means) / scales
    matrix = np.column_stack((np.ones(len(rows)), matrix))
    offsets = np.asarray([_logit(row["market_probability_yes"]) for row in rows])
    outcomes = np.asarray([float(row["resolved_yes"]) for row in rows])
    weights = _event_weights(rows)
    weight_total = float(weights.sum())

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        probabilities = _sigmoid(offsets + matrix @ coefficients)
        loss = -float(
            np.sum(
                weights
                * (
                    outcomes * np.log(np.clip(probabilities, 1e-12, 1.0))
                    + (1.0 - outcomes)
                    * np.log(np.clip(1.0 - probabilities, 1e-12, 1.0))
                )
            )
            / weight_total
        )
        penalty_weights = np.ones_like(coefficients)
        penalty_weights[0] = 0.1
        loss += 0.5 * regularization * float(
            np.sum(penalty_weights * coefficients**2)
        )
        gradient = matrix.T @ (weights * (probabilities - outcomes)) / weight_total
        gradient += regularization * penalty_weights * coefficients
        return loss, gradient

    result = minimize(
        lambda values: objective(values)[0],
        np.zeros(matrix.shape[1]),
        jac=lambda values: objective(values)[1],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return OffsetFit(
        feature_names=feature_names,
        means=means,
        scales=scales,
        coefficients=np.asarray(result.x, dtype=np.float64),
        regularization=regularization,
        converged=bool(result.success),
    )


def predict_offset_model(fit: OffsetFit, rows: list[dict[str, Any]]) -> np.ndarray:
    raw = _matrix(rows, fit.feature_names)
    matrix = np.column_stack((np.ones(len(rows)), (raw - fit.means) / fit.scales))
    offsets = np.asarray([_logit(row["market_probability_yes"]) for row in rows])
    return _sigmoid(offsets + matrix @ fit.coefficients)


def _bootstrap_predictions(
    training_rows: list[dict[str, Any]],
    scoring_rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    regularization: float,
    iterations: int,
    seed: int,
) -> np.ndarray:
    dates = sorted({row["event_date"] for row in training_rows})
    by_date = defaultdict(list)
    for row in training_rows:
        by_date[row["event_date"]].append(row)
    rng = np.random.default_rng(seed)
    predictions = []
    block = min(7, len(dates))
    for _ in range(iterations):
        sampled_dates = []
        while len(sampled_dates) < len(dates):
            start = int(rng.integers(0, max(1, len(dates) - block + 1)))
            sampled_dates.extend(dates[start : start + block])
        sampled = [
            {**row, "_bootstrap_instance": instance}
            for instance, day in enumerate(sampled_dates[: len(dates)])
            for row in by_date[day]
        ]
        fit = fit_offset_model(sampled, feature_names, regularization)
        if fit.converged:
            predictions.append(predict_offset_model(fit, scoring_rows))
    if len(predictions) < max(20, iterations // 2):
        raise ValueError("insufficient converged event-block bootstrap fits")
    return np.asarray(predictions, dtype=np.float64)


def _weighted_probability_metrics(
    rows: list[dict[str, Any]], probabilities: np.ndarray
) -> dict[str, float]:
    outcomes = np.asarray([float(row["resolved_yes"]) for row in rows])
    weights = _event_weights(rows)
    return {
        "event_days": len({row["event_date"] for row in rows}),
        "rows": len(rows),
        "binary_log_loss": float(log_loss(outcomes, probabilities, sample_weight=weights)),
        "binary_brier_score": float(
            brier_score_loss(outcomes, probabilities, sample_weight=weights)
        ),
    }


def _price_band(price: float) -> str:
    for lower, upper, label in PRICE_BANDS:
        if lower <= price < upper:
            return label
    raise ValueError(f"price outside contract range: {price}")


def _candidate_sides(
    row: dict[str, Any], point_yes: float, lower_yes: float, upper_yes: float
) -> list[dict[str, Any]]:
    output = []
    for side, probability, lower in (
        ("YES", point_yes, lower_yes),
        ("NO", 1.0 - point_yes, 1.0 - upper_yes),
    ):
        candidate = row["candidates"][side]
        if not candidate.get("executable") or candidate.get("all_in_cost_per_share") is None:
            continue
        cost = float(candidate["all_in_cost_per_share"])
        output.append(
            {
                **candidate,
                "predicted_probability": probability,
                "probability_lower": lower,
                "robust_edge_per_share": lower - cost,
                "point_edge_per_share": probability - cost,
            }
        )
    return output


def select_trades(
    rows: list[dict[str, Any]],
    probabilities: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    edge_threshold: float,
    maximum_cost: float,
) -> list[dict[str, Any]]:
    by_date_hour: dict[date, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row, probability, low, high in zip(rows, probabilities, lower, upper, strict=True):
        for candidate in _candidate_sides(row, float(probability), float(low), float(high)):
            cost = float(candidate["all_in_cost_per_share"])
            if 0.0 < cost <= maximum_cost and candidate["robust_edge_per_share"] >= edge_threshold:
                by_date_hour[row["event_date"]][row["decision_hour_local"]].append(candidate)
    selected = []
    for event_date in sorted(by_date_hour):
        midnight = by_date_hour[event_date].get(0, [])
        noon = by_date_hour[event_date].get(12, [])
        pool = midnight if midnight else noon
        if pool:
            selected.append(max(pool, key=lambda row: (row["robust_edge_per_share"], row["point_edge_per_share"])))
    return selected


def _stress_trade_net(trade: dict[str, Any], slippage: float) -> float:
    execution_price = float(trade["ask_vwap"]) + slippage
    if execution_price <= 0 or execution_price > 1:
        return -float(trade["all_in_cost_per_share"]) * QUANTITY
    fee = taker_fee_per_share(
        execution_price,
        quantity=QUANTITY,
        enabled=bool(trade["fees_enabled"]),
        rate=float(trade["fee_rate"]),
        exponent=float(trade["fee_exponent"]),
    )
    cost = execution_price + fee
    return ((1.0 if trade["resolved_side"] else 0.0) - cost) * QUANTITY


def _daily_bootstrap_lower(daily: list[float], seed: int) -> float:
    if not daily:
        return 0.0
    rng = np.random.default_rng(seed)
    values = np.asarray(daily, dtype=np.float64)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(5000)]
    return float(np.quantile(means, 0.10))


def economic_metrics(trades: list[dict[str, Any]], event_dates: list[date]) -> dict[str, Any]:
    nets = [float(trade["realized_net_per_share"]) * QUANTITY for trade in trades]
    by_date = {trade["event_date"]: net for trade, net in zip(trades, nets, strict=True)}
    daily = [by_date.get(day, 0.0) for day in event_dates]
    cumulative = np.cumsum(daily) if daily else np.asarray([], dtype=np.float64)
    running_max = np.maximum.accumulate(np.maximum(cumulative, 0.0)) if daily else cumulative
    drawdowns = running_max - cumulative if daily else cumulative
    profits = [net for net in nets if net > 0]
    losses = [-net for net in nets if net < 0]
    debit = sum(float(trade["all_in_cost_per_share"]) * QUANTITY for trade in trades)
    ordered = sorted(nets, reverse=True)
    price_bands = []
    for _, _, label in PRICE_BANDS:
        band_nets = [
            net
            for trade, net in zip(trades, nets, strict=True)
            if _price_band(float(trade["ask_vwap"])) == label
        ]
        price_bands.append(
            {
                "price_band": label,
                "trades": len(band_nets),
                "wins": sum(net > 0 for net in band_nets),
                "total_net": float(sum(band_nets)),
            }
        )
    return {
        "event_days": len(event_dates),
        "trades": len(trades),
        "wins": len(profits),
        "losses": len(losses),
        "hit_rate": len(profits) / len(trades) if trades else None,
        "mean_break_even_probability": (
            float(np.mean([trade["all_in_cost_per_share"] for trade in trades]))
            if trades
            else None
        ),
        "mean_ask_vwap": float(np.mean([trade["ask_vwap"] for trade in trades])) if trades else None,
        "maximum_ask_vwap": max((float(trade["ask_vwap"]) for trade in trades), default=None),
        "total_entry_debit": float(debit),
        "total_net": float(sum(nets)),
        "return_on_deployed_capital": float(sum(nets) / debit) if debit else None,
        "profit_factor": float(sum(profits) / sum(losses)) if losses else (math.inf if profits else None),
        "maximum_drawdown": float(max(drawdowns, default=0.0)),
        "worst_trade": min(nets, default=None),
        "net_without_best_trade": float(sum(ordered[1:])) if ordered else None,
        "net_without_best_three_trades": float(sum(ordered[3:])) if len(ordered) >= 3 else None,
        "lower_90pct_bootstrap_mean_daily_net": _daily_bootstrap_lower(daily, BOOTSTRAP_SEED),
        "price_bands": price_bands,
        "slippage_stress": {
            f"{slippage:.3f}": {
                "total_net": float(sum(_stress_trade_net(trade, slippage) for trade in trades)),
                "selected_trade_set_frozen": True,
            }
            for slippage in (0.0, 0.01, 0.02)
        },
    }


def _inner_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dates = sorted({row["event_date"] for row in rows})
    split = max(20, int(len(dates) * 0.75))
    if split >= len(dates):
        split = len(dates) - 1
    training_dates = set(dates[:split])
    return (
        [row for row in rows if row["event_date"] in training_dates],
        [row for row in rows if row["event_date"] not in training_dates],
    )


def _select_fit_contract(
    rows: list[dict[str, Any]], model_name: str, bootstrap_iterations: int
) -> tuple[float, float, dict[str, Any]]:
    feature_names = MODEL_FEATURE_SETS[model_name]
    inner_train, inner_validation = _inner_split(rows)
    candidates = []
    for regularization in REGULARIZATION_GRID:
        fit = fit_offset_model(inner_train, feature_names, regularization)
        probabilities = predict_offset_model(fit, inner_validation)
        metrics = _weighted_probability_metrics(inner_validation, probabilities)
        candidates.append((metrics["binary_log_loss"], regularization, fit.converged))
    _, regularization, converged = min(candidates)
    if not converged:
        raise ValueError(f"selected {model_name} inner fit did not converge")
    inner_fit = fit_offset_model(inner_train, feature_names, regularization)
    point = predict_offset_model(inner_fit, inner_validation)
    sampled = _bootstrap_predictions(
        inner_train,
        inner_validation,
        feature_names,
        regularization,
        max(40, bootstrap_iterations // 2),
        BOOTSTRAP_SEED ^ len(rows) ^ len(feature_names),
    )
    lower = np.quantile(sampled, 0.10, axis=0)
    upper = np.quantile(sampled, 0.90, axis=0)
    validation_dates = sorted({row["event_date"] for row in inner_validation})
    threshold_metrics = []
    for threshold in EDGE_THRESHOLDS:
        trades = select_trades(
            inner_validation,
            point,
            lower,
            upper,
            edge_threshold=threshold,
            maximum_cost=0.999999,
        )
        metrics = economic_metrics(trades, validation_dates)
        threshold_metrics.append((threshold, metrics))
    eligible = [item for item in threshold_metrics if item[1]["trades"] >= 3]
    pool = eligible or threshold_metrics
    threshold, selected_metrics = max(
        pool,
        key=lambda item: (
            item[1]["total_net"],
            item[1]["net_without_best_trade"] or -math.inf,
            item[1]["trades"],
            -item[0],
        ),
    )
    return regularization, threshold, {
        "regularization_candidates": [
            {"regularization": item[1], "binary_log_loss": item[0], "converged": item[2]}
            for item in candidates
        ],
        "selected_threshold_validation_metrics": selected_metrics,
    }


def _fit_and_score(
    training_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    model_name: str,
    bootstrap_iterations: int,
) -> tuple[dict[str, Any], OffsetFit, np.ndarray, np.ndarray, np.ndarray]:
    regularization, threshold, selection = _select_fit_contract(
        training_rows, model_name, bootstrap_iterations
    )
    fit = fit_offset_model(training_rows, MODEL_FEATURE_SETS[model_name], regularization)
    point = predict_offset_model(fit, evaluation_rows)
    sampled = _bootstrap_predictions(
        training_rows,
        evaluation_rows,
        fit.feature_names,
        regularization,
        bootstrap_iterations,
        BOOTSTRAP_SEED ^ len(training_rows) ^ len(evaluation_rows) ^ len(fit.feature_names),
    )
    lower = np.quantile(sampled, 0.10, axis=0)
    upper = np.quantile(sampled, 0.90, axis=0)
    contract = {
        "selected_regularization": regularization,
        "selected_edge_threshold": threshold,
        "feature_names": fit.feature_names,
        "inner_selection": selection,
        "converged": fit.converged,
    }
    return contract, fit, point, lower, upper


def _score_period(
    rows: list[dict[str, Any]],
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    dates = sorted({row["event_date"] for row in rows})
    policies = {}
    for name, maximum_cost in (
        ("capped_25c", 0.25),
        ("capped_50c", 0.50),
        ("full_market", 0.999999),
    ):
        trades = select_trades(
            rows,
            point,
            lower,
            upper,
            edge_threshold=threshold,
            maximum_cost=maximum_cost,
        )
        policies[name] = economic_metrics(trades, dates)
    return {
        "probability_metrics": _weighted_probability_metrics(rows, point),
        "policies": policies,
    }


def _market_only_score(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = np.asarray([row["market_probability_yes"] for row in rows], dtype=np.float64)
    return point, point, point


def _qualification(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "positive_total_net": metrics["total_net"] > 0,
        "profit_factor_above_one": metrics["profit_factor"] is not None
        and metrics["profit_factor"] > 1,
        "positive_return_on_deployed_capital": metrics["return_on_deployed_capital"] is not None
        and metrics["return_on_deployed_capital"] > 0,
        "hit_rate_exceeds_mean_break_even": metrics["hit_rate"] is not None
        and metrics["hit_rate"] > metrics["mean_break_even_probability"],
        "positive_without_best_trade": metrics["net_without_best_trade"] is not None
        and metrics["net_without_best_trade"] > 0,
        "positive_without_best_three": metrics["net_without_best_three_trades"] is not None
        and metrics["net_without_best_three_trades"] > 0,
        "positive_lower_90pct_daily_net": metrics["lower_90pct_bootstrap_mean_daily_net"] > 0,
        "positive_at_two_cent_slippage": metrics["slippage_stress"]["0.020"]["total_net"] > 0,
        "minimum_30_trades": metrics["trades"] >= 30,
    }


def run_pmxt_full_market_tournament(
    settings: Settings,
    *,
    source_weather_artifact_path: Path,
    source_weather_artifact_sha256: str,
    development_start: date,
    development_end: date,
    sealed_start: date,
    sealed_end: date,
    git_revision: str,
    runner_image_id: str,
    output_directory: Path | None = None,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    if len(git_revision) != 40 or any(value not in "0123456789abcdef" for value in git_revision):
        raise ValueError("git_revision must be an exact lowercase Git SHA-1")
    if not runner_image_id.startswith("sha256:") or len(runner_image_id) != 71:
        raise ValueError("runner_image_id must be sha256:<64 lowercase hexadecimal characters>")
    if not development_start <= development_end < sealed_start <= sealed_end:
        raise ValueError("development and sealed periods must be chronological and disjoint")
    rows = build_training_rows(
        settings,
        source_weather_artifact_path=source_weather_artifact_path,
        source_weather_artifact_sha256=source_weather_artifact_sha256,
        start=development_start,
        end=sealed_end,
    )
    development_rows = [row for row in rows if row["event_date"] <= development_end]
    sealed_rows = [row for row in rows if sealed_start <= row["event_date"] <= sealed_end]
    if not development_rows or not sealed_rows:
        raise ValueError("development and sealed periods both require complete rows")
    input_sha256 = _canonical_digest(rows)
    fold_reports = []
    oof_by_model: dict[str, list[tuple[dict[str, Any], float, float, float]]] = defaultdict(list)
    for fold in OUTER_FOLDS:
        train = [row for row in development_rows if row["event_date"] <= fold.training_end]
        evaluate = [
            row
            for row in development_rows
            if fold.evaluation_start <= row["event_date"] <= fold.evaluation_end
        ]
        if not train or not evaluate:
            raise ValueError(f"fold {fold.name} has no complete training or evaluation rows")
        fold_entry = {"contract": asdict(fold), "models": {}}
        market_point, market_lower, market_upper = _market_only_score(evaluate)
        fold_entry["models"]["market_only"] = _score_period(
            evaluate, market_point, market_lower, market_upper, 0.0
        )
        for row, point, low, high in zip(
            evaluate, market_point, market_lower, market_upper, strict=True
        ):
            oof_by_model["market_only"].append((row, float(point), float(low), float(high)))
        for model_name in MODEL_FEATURE_SETS:
            contract, _, point, lower, upper = _fit_and_score(
                train, evaluate, model_name, bootstrap_iterations
            )
            fold_entry["models"][model_name] = {
                "fit_contract": contract,
                **_score_period(evaluate, point, lower, upper, contract["selected_edge_threshold"]),
            }
            for row, probability, low, high in zip(evaluate, point, lower, upper, strict=True):
                oof_by_model[model_name].append(
                    (row, float(probability), float(low), float(high))
                )
        fold_reports.append(fold_entry)

    oof_reports = {}
    for model_name, scored in oof_by_model.items():
        scored.sort(key=lambda item: (item[0]["decision_time"], item[0]["market_id"]))
        scored_rows = [item[0] for item in scored]
        point = np.asarray([item[1] for item in scored])
        lower = np.asarray([item[2] for item in scored])
        upper = np.asarray([item[3] for item in scored])
        threshold = 0.0
        if model_name != "market_only":
            threshold = float(
                np.median(
                    [
                        fold["models"][model_name]["fit_contract"]["selected_edge_threshold"]
                        for fold in fold_reports
                    ]
                )
            )
        oof_reports[model_name] = {
            "frozen_oof_edge_threshold": threshold,
            **_score_period(scored_rows, point, lower, upper, threshold),
        }

    final_models = {}
    sealed_reports = {}
    market_point, market_lower, market_upper = _market_only_score(sealed_rows)
    sealed_reports["market_only"] = _score_period(
        sealed_rows, market_point, market_lower, market_upper, 0.0
    )
    for model_name in MODEL_FEATURE_SETS:
        contract, fit, point, lower, upper = _fit_and_score(
            development_rows, sealed_rows, model_name, bootstrap_iterations
        )
        final_models[model_name] = {
            "feature_names": fit.feature_names,
            "means": fit.means,
            "scales": fit.scales,
            "coefficients": fit.coefficients,
            "regularization": fit.regularization,
            "edge_threshold": contract["selected_edge_threshold"],
            "converged": fit.converged,
        }
        sealed_reports[model_name] = {
            "fit_contract": contract,
            **_score_period(sealed_rows, point, lower, upper, contract["selected_edge_threshold"]),
        }

    primary_oof = oof_reports["pmxt_market_state"]["policies"]["full_market"]
    primary_sealed = sealed_reports["pmxt_market_state"]["policies"]["full_market"]
    oof_checks = _qualification(primary_oof)
    all_folds_positive = all(
        fold["models"]["pmxt_market_state"]["policies"]["full_market"]["total_net"] > 0
        for fold in fold_reports
    )
    oof_checks["positive_all_outer_folds"] = all_folds_positive
    qualification_supported = all(oof_checks.values())
    tournament_id = str(uuid.uuid4())
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "tournament_id": tournament_id,
        "process_id": PROCESS_ID,
        "producing_git_revision": git_revision,
        "source_input_sha256": input_sha256,
        "source_weather_artifact_sha256": source_weather_artifact_sha256,
        "source_weather_candidate": SOURCE_WEATHER_CANDIDATE,
        "final_models": final_models,
        "quantity": QUANTITY,
        "modeled_slippage": MODELED_SLIPPAGE,
    }
    output = output_directory or settings.model_directory
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "tournament.joblib"
    partial_artifact = output / "tournament.joblib.partial"
    joblib.dump(artifact, partial_artifact, compress=3)
    partial_artifact.replace(artifact_path)
    artifact_sha256, artifact_bytes = file_sha256(artifact_path)
    (output / "tournament.sha256").write_text(f"{artifact_sha256}\n")
    report = {
        "schema_version": SCHEMA_VERSION,
        "tournament_id": tournament_id,
        "objective": "pmxt_market_state_full_price_curve_positive_net_expectancy",
        "provenance": {
            "producing_git_revision": git_revision,
            "runner_existing_image_id": runner_image_id,
            "container_image_rebuilt": False,
            "generated_at": datetime.now(UTC),
            "artifact_uri": str(artifact_path),
            "artifact_sha256": artifact_sha256,
            "artifact_bytes": artifact_bytes,
            "source_weather_artifact_uri": str(source_weather_artifact_path),
            "source_weather_artifact_sha256": source_weather_artifact_sha256,
            "source_input_sha256": input_sha256,
        },
        "data_contract": {
            "development_start": development_start,
            "development_end": development_end,
            "sealed_start": sealed_start,
            "sealed_end": sealed_end,
            "complete_rows": len(rows),
            "development_rows": len(development_rows),
            "development_event_days": len({row["event_date"] for row in development_rows}),
            "sealed_rows": len(sealed_rows),
            "sealed_event_days": len({row["event_date"] for row in sealed_rows}),
            "source_input_sha256": input_sha256,
        },
        "model_contract": {
            "prediction_target": "canonical_yes_contract_resolution_probability",
            "decision_target": "five_share_net_value_after_captured_fee_and_slippage",
            "model_feature_sets": MODEL_FEATURE_SETS,
            "regularization_grid": REGULARIZATION_GRID,
            "edge_threshold_grid": EDGE_THRESHOLDS,
            "pmxt_windows_minutes": WINDOW_MINUTES,
            "canonical_yes_training_no_complement_duplicate": True,
            "event_date_grouped_chronological_validation": True,
        },
        "folds": fold_reports,
        "out_of_fold": oof_reports,
        "sealed": sealed_reports,
        "primary_challenger": "pmxt_market_state:full_market",
        "primary_oof_qualification_checks": oof_checks,
        "research_qualification_supported": qualification_supported,
        "sealed_diagnostic_positive": primary_sealed["total_net"] > 0,
        "production_qualified": False,
    }
    safe_report = _json_safe(report)
    report_path = output / "metrics.json"
    partial_report = output / "metrics.json.partial"
    partial_report.write_text(json.dumps(safe_report, indent=2, sort_keys=True) + "\n")
    partial_report.replace(report_path)
    provenance = {
        "artifact_sha256": artifact_sha256,
        "artifact_path": str(artifact_path),
        "producing_git_revision": git_revision,
        "training_run_identity": tournament_id,
        "source_input_sha256": input_sha256,
        "qualification_status": (
            "research_qualified_not_production" if qualification_supported else "no_economic_champion"
        ),
        "deployment_status": "not_deployed_training_only",
    }
    (output / "model-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return safe_report
