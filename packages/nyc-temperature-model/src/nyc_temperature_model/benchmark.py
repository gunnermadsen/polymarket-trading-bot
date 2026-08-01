from __future__ import annotations

import json
import math
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import psycopg

from . import PROCESS_ID
from .config import Settings
from .database import connection
from .modeling import (
    build_feature_rows,
    load_model,
    normalized_bucket_probabilities,
    point_prediction,
)


def _ece(probabilities: list[float], labels: list[int], bins: int = 10) -> float:
    if not probabilities:
        return float("nan")
    p = np.asarray(probabilities)
    y = np.asarray(labels)
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (p >= low) & (p < high if index < bins - 1 else p <= high)
        if selected.any():
            result += selected.mean() * abs(float(p[selected].mean() - y[selected].mean()))
    return float(result)


def _bootstrap_lower(values: np.ndarray, confidence: float = 0.90, iterations: int = 5000) -> float:
    if values.size == 0:
        return float("nan")
    rng = np.random.default_rng(24051985)
    samples = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    return float(np.quantile(samples, 1 - confidence))


def _fold_means(trades: list[dict], folds: int = 3) -> list[float]:
    ordered = sorted(trades, key=lambda row: row["event_date"])
    chunks = np.array_split(np.arange(len(ordered)), folds)
    return [
        float(np.mean([ordered[int(index)]["net_per_share"] for index in chunk]))
        if len(chunk)
        else float("nan")
        for chunk in chunks
    ]


def _noise_sensitivity(trades: list[dict]) -> dict[str, Any]:
    if not trades:
        return {}
    rng = np.random.default_rng(418)
    prices = np.asarray([row["price"] for row in trades])
    outcomes = np.asarray([row["resolved_yes"] for row in trades], dtype=bool)
    fee_and_buffer = np.asarray([row["fee_and_buffer"] for row in trades])
    output = {}
    for rate in (0.01, 0.02, 0.05):
        means = []
        for _ in range(1000):
            flips = rng.random(outcomes.size) < rate
            stressed = np.logical_xor(outcomes, flips)
            pnl = np.where(stressed, 1 - prices, -prices) - fee_and_buffer
            means.append(float(pnl.mean()))
        output[f"label_flip_{int(rate * 100)}pct"] = {
            "mean_net_per_share": float(np.mean(means)),
            "p10_net_per_share": float(np.quantile(means, 0.10)),
        }
    return output


def _market_rows(database_url: str, start: date, end: date) -> list[dict]:
    with connection(database_url) as conn:
        return list(
            conn.execute(
                """
                SELECT market_id,event_date,bucket_lower_f,bucket_upper_f,resolved_yes,
                       yes_token_id,COALESCE(fee_rate_bps,0) AS fee_rate_bps
                FROM weather.temperature_markets
                WHERE event_date >= %s AND event_date <= %s AND resolved_yes IS NOT NULL
                ORDER BY event_date,market_id
                """,
                (start, end),
            ).fetchall()
        )


def _prices(database_url: str, model_run_id: str, quantity: float, evidence_tier: str) -> list[dict]:
    with connection(database_url) as conn:
        if evidence_tier == "executable_taker":
            return list(
                conn.execute(
                    """
                    SELECT p.market_id,p.decision_time,m.event_date,m.resolved_yes,
                           m.fee_rate_bps,p.probability_yes::double precision,
                           e.yes_ask_vwap::double precision AS price,e.quality_flags
                    FROM weather.predictions p
                    JOIN weather.temperature_markets m ON m.market_id=p.market_id
                    JOIN weather.execution_snapshots e
                      ON e.market_id=p.market_id AND e.decision_time=p.decision_time
                     AND e.quantity=%s
                    WHERE p.process_id=%s AND p.model_run_id=%s
                      AND e.yes_ask_vwap IS NOT NULL
                    ORDER BY m.event_date,p.market_id
                    """,
                    (quantity, PROCESS_ID, model_run_id),
                ).fetchall()
            )
        if evidence_tier != "indicative":
            raise ValueError("evidence_tier must be indicative or executable_taker")
        return list(
            conn.execute(
                """
                SELECT p.market_id,p.decision_time,m.event_date,m.resolved_yes,
                       m.fee_rate_bps,p.probability_yes::double precision,
                       ph.price::double precision AS price,'[]'::jsonb AS quality_flags
                FROM weather.predictions p
                JOIN weather.temperature_markets m ON m.market_id=p.market_id
                JOIN LATERAL (
                  SELECT price
                  FROM weather.price_history ph
                  WHERE ph.token_id=m.yes_token_id
                    AND ph.observed_at <= p.decision_time
                    AND ph.observed_at >= p.decision_time - interval '30 minutes'
                  ORDER BY ph.observed_at DESC
                  LIMIT 1
                ) ph ON true
                WHERE p.process_id=%s AND p.model_run_id=%s
                ORDER BY m.event_date,p.market_id
                """,
                (PROCESS_ID, model_run_id),
            ).fetchall()
        )


def run_benchmark(
    settings: Settings,
    *,
    model_run_id: str,
    model_path: Path,
    evaluation_start: date,
    evaluation_end: date,
    quantity: float,
    evidence_tier: str,
    safety_buffer: float = 0.02,
) -> dict[str, Any]:
    if quantity not in (1.0, 5.0, 10.0):
        raise ValueError("quantity must be 1, 5, or 10 shares")
    if not 0 <= safety_buffer <= 0.25:
        raise ValueError("safety_buffer must be between 0 and 0.25")
    bundle = load_model(model_path)
    decision_hour = int(bundle["decision_hour_local"])
    feature_rows = build_feature_rows(
        settings.database_url,
        evaluation_start,
        evaluation_end + timedelta(days=1),
        decision_hour,
    )
    markets = _market_rows(settings.database_url, evaluation_start, evaluation_end)
    markets_by_date: dict[date, list[dict]] = {}
    for market in markets:
        markets_by_date.setdefault(market["event_date"], []).append(market)
    candidate_points = point_prediction(bundle, feature_rows) if feature_rows else np.array([])
    predictions = []
    skill_candidate = []
    skill_raw = []
    binary_probabilities = []
    binary_labels = []
    residuals = np.asarray(bundle["residuals"], dtype=np.float64)
    raw_residuals = np.asarray(bundle["raw_residuals"], dtype=np.float64)
    for feature, point in zip(feature_rows, candidate_points, strict=True):
        day_markets = markets_by_date.get(feature.event_date, [])
        if not day_markets:
            continue
        raw_point = max(
            feature.forecast_max_remaining_f,
            feature.observed_max_so_far_f
            if feature.observed_max_so_far_f is not None
            else -np.inf,
        )
        buckets = [
            (market["bucket_lower_f"], market["bucket_upper_f"]) for market in day_markets
        ]
        candidate_probabilities = normalized_bucket_probabilities(
            float(point), residuals, buckets
        )
        raw_probabilities = normalized_bucket_probabilities(
            float(raw_point), raw_residuals, buckets
        )
        winner_candidate_probability = None
        winner_raw_probability = None
        for market, probability, raw_probability in zip(
            day_markets, candidate_probabilities, raw_probabilities, strict=True
        ):
            predictions.append(
                (
                    PROCESS_ID,
                    model_run_id,
                    market["market_id"],
                    feature.decision_time,
                    probability,
                )
            )
            label = int(market["resolved_yes"])
            binary_probabilities.append(probability)
            binary_labels.append(label)
            if label:
                winner_candidate_probability = probability
                winner_raw_probability = raw_probability
        if winner_candidate_probability is not None:
            skill_candidate.append(-math.log(max(winner_candidate_probability, 1e-12)))
            skill_raw.append(-math.log(max(winner_raw_probability, 1e-12)))
    with connection(settings.database_url) as conn, conn.transaction():
        conn.execute(
            "DELETE FROM weather.predictions WHERE process_id=%s AND model_run_id=%s",
            (PROCESS_ID, model_run_id),
        )
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO weather.predictions (
                  process_id,model_run_id,market_id,decision_time,probability_yes
                ) VALUES (%s,%s,%s,%s,%s)
                """,
                predictions,
            )
        reconciliation = conn.execute(
            """
            SELECT count(*) FILTER (WHERE winner_matches_station IS NOT NULL)::int AS compared,
                   count(*) FILTER (WHERE winner_matches_station)::int AS matched
            FROM weather.label_reconciliation
            WHERE process_id=%s AND event_date >= %s AND event_date <= %s
            """,
            (PROCESS_ID, evaluation_start, evaluation_end),
        ).fetchone()
    candidate_log_loss = float(np.mean(skill_candidate)) if skill_candidate else float("nan")
    raw_log_loss = float(np.mean(skill_raw)) if skill_raw else float("nan")
    ece = _ece(binary_probabilities, binary_labels)
    price_rows = _prices(settings.database_url, model_run_id, quantity, evidence_tier)
    eligible = []
    for row in price_rows:
        flags = set(row["quality_flags"] or [])
        if "crossed_yes_book" in flags or any(
            flag.startswith(("missing_yes", "insufficient_yes")) for flag in flags
        ):
            continue
        fee_per_share = float(row["price"]) * int(row["fee_rate_bps"] or 0) / 10_000
        edge = float(row["probability_yes"]) - float(row["price"]) - fee_per_share - safety_buffer
        if edge <= 0:
            continue
        eligible.append({**row, "edge": edge, "fee_per_share": fee_per_share})
    selected = {}
    for row in eligible:
        previous = selected.get(row["event_date"])
        if previous is None or row["edge"] > previous["edge"]:
            selected[row["event_date"]] = row
    trades = []
    for row in selected.values():
        fee_and_buffer = row["fee_per_share"] + safety_buffer
        net_per_share = (
            (1 - float(row["price"])) if row["resolved_yes"] else -float(row["price"])
        ) - fee_and_buffer
        trades.append(
            {
                "event_date": row["event_date"],
                "market_id": row["market_id"],
                "price": float(row["price"]),
                "probability_yes": float(row["probability_yes"]),
                "resolved_yes": bool(row["resolved_yes"]),
                "edge": row["edge"],
                "fee_and_buffer": fee_and_buffer,
                "net_per_share": net_per_share,
                "net": net_per_share * quantity,
            }
        )
    values = np.asarray([row["net_per_share"] for row in trades], dtype=np.float64)
    fold_means = _fold_means(trades)
    without_top_five = sorted(values, reverse=True)[5:] if values.size > 5 else []
    label_compared = int(reconciliation["compared"] or 0)
    label_matched = int(reconciliation["matched"] or 0)
    label_agreement = label_matched / label_compared if label_compared else float("nan")
    metrics = {
        "forecast": {
            "resolved_event_days": len(skill_candidate),
            "candidate_log_loss": candidate_log_loss,
            "raw_hrrr_log_loss": raw_log_loss,
            "log_loss_uplift": raw_log_loss - candidate_log_loss,
            "expected_calibration_error": ece,
        },
        "labels": {
            "days_compared": label_compared,
            "days_matched": label_matched,
            "agreement": label_agreement,
        },
        "economics": {
            "evidence_tier": evidence_tier,
            "quantity": quantity,
            "candidate_quotes": len(price_rows),
            "positive_edge_quotes": len(eligible),
            "independent_event_days": len(trades),
            "mean_net_per_share": float(values.mean()) if values.size else float("nan"),
            "total_net": float(values.sum() * quantity) if values.size else 0.0,
            "lower_90pct_bootstrap_mean_net_per_share": _bootstrap_lower(values),
            "chronological_fold_mean_net_per_share": fold_means,
            "mean_without_top_five_days": (
                float(np.mean(without_top_five)) if len(without_top_five) else float("nan")
            ),
            "label_noise_sensitivity": _noise_sensitivity(trades),
        },
    }
    checks = {
        "label_agreement_at_least_99pct": label_compared > 0 and label_agreement >= 0.99,
        "beats_raw_hrrr_log_loss": bool(skill_candidate) and candidate_log_loss < raw_log_loss,
        "ece_at_most_5pct": math.isfinite(ece) and ece <= 0.05,
        "at_least_75_independent_event_days": len(trades) >= 75,
        "positive_mean_net_per_share": values.size > 0 and float(values.mean()) > 0,
        "positive_lower_90pct_bound": values.size > 0 and _bootstrap_lower(values) > 0,
        "positive_all_chronological_folds": all(math.isfinite(value) and value > 0 for value in fold_means),
        "positive_without_top_five_days": bool(without_top_five) and float(np.mean(without_top_five)) > 0,
        "five_share_economics": quantity == 5.0,
        "executable_taker_evidence": evidence_tier == "executable_taker",
    }
    qualified = all(checks.values())
    report = {
        "schema_version": "nyc-temperature-expectancy-benchmark-v1",
        "model_run_id": model_run_id,
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end": evaluation_end.isoformat(),
        "policy": {
            "side": "buy_yes",
            "maximum_positions_per_event_day": 1,
            "minimum_edge_after_costs": 0.0,
            "safety_buffer_per_share": safety_buffer,
            "maker_fills_assumed": False,
        },
        "metrics": metrics,
        "qualification_checks": checks,
        "qualified": qualified,
    }
    benchmark_run_id = str(uuid.uuid4())
    with connection(settings.database_url) as conn, conn.transaction():
        conn.execute(
            """
            INSERT INTO weather.benchmark_runs (
              benchmark_run_id,process_id,model_run_id,evaluation_start,evaluation_end,
              quantity,evidence_tier,policy,metrics,qualified
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                benchmark_run_id,
                PROCESS_ID,
                model_run_id,
                evaluation_start,
                evaluation_end,
                quantity,
                evidence_tier,
                psycopg.types.json.Jsonb(report["policy"]),
                psycopg.types.json.Jsonb({**metrics, "qualification_checks": checks}),
                qualified,
            ),
        )
    settings.report_directory.mkdir(parents=True, exist_ok=True)
    report_path = settings.report_directory / f"benchmark-{benchmark_run_id}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    report["benchmark_run_id"] = benchmark_run_id
    report["report_uri"] = str(report_path)
    return report
