from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import psycopg
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import PROCESS_ID, STATION_ID
from .config import Settings
from .database import connection
from .sources import file_sha256

NYC = ZoneInfo("America/New_York")
FEATURE_SCHEMA_VERSION = "nyc-temperature-features-v1"
FEATURE_NAMES = (
    "forecast_max_remaining_f",
    "forecast_mean_remaining_f",
    "forecast_min_remaining_f",
    "observed_max_so_far_f",
    "latest_observation_f",
    "day_of_year_sin",
    "day_of_year_cos",
    "decision_hour_local",
)
CANDIDATES = ("raw_hrrr", "linear_bias", "histogram_residual")


@dataclass(frozen=True)
class FeatureRow:
    event_date: date
    decision_time: datetime
    decision_hour_local: int
    forecast_max_remaining_f: float
    forecast_mean_remaining_f: float
    forecast_min_remaining_f: float
    observed_max_so_far_f: float | None
    latest_observation_f: float | None
    day_of_year_sin: float
    day_of_year_cos: float
    target_daily_max_f: float
    target_rounded_max_f: int
    observation_count: int

    def vector(self) -> list[float]:
        return [
            float(value) if (value := getattr(self, name)) is not None else np.nan
            for name in FEATURE_NAMES
        ]


def round_temperature(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _local_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    return (
        datetime.combine(start, datetime.min.time(), NYC).astimezone(UTC),
        datetime.combine(end, datetime.min.time(), NYC).astimezone(UTC),
    )


def build_feature_rows(database_url: str, start: date, end: date, decision_hour: int) -> list[FeatureRow]:
    if decision_hour not in (0, 12):
        raise ValueError("decision_hour must be 0 or 12")
    start_utc, end_utc = _local_bounds(start, end)
    with connection(database_url) as conn:
        forecasts = conn.execute(
            """
            SELECT decision_time, valid_at, temperature_f::double precision AS temperature_f
            FROM weather.hrrr_point_forecasts
            WHERE station_id = %s AND decision_time >= %s AND decision_time < %s
            ORDER BY decision_time, valid_at
            """,
            (STATION_ID, start_utc, end_utc),
        ).fetchall()
        observations = conn.execute(
            """
            SELECT observed_at, temperature_f::double precision AS temperature_f
            FROM weather.station_observations
            WHERE station_id = %s AND provider = 'iem_asos_metar'
              AND observed_at >= %s - interval '6 hours' AND observed_at < %s
              AND temperature_f IS NOT NULL
            ORDER BY observed_at
            """,
            (STATION_ID, start_utc, end_utc),
        ).fetchall()
    forecast_groups: dict[datetime, list[float]] = {}
    for row in forecasts:
        local = row["decision_time"].astimezone(NYC)
        if local.hour == decision_hour:
            forecast_groups.setdefault(row["decision_time"], []).append(row["temperature_f"])
    output = []
    for decision_time, forecast_values in sorted(forecast_groups.items()):
        local_decision = decision_time.astimezone(NYC)
        event_date = local_decision.date()
        local_start = datetime.combine(event_date, datetime.min.time(), NYC).astimezone(UTC)
        local_end = datetime.combine(
            event_date + timedelta(days=1), datetime.min.time(), NYC
        ).astimezone(UTC)
        day_observations = [
            row
            for row in observations
            if local_start <= row["observed_at"] < local_end
        ]
        if len(day_observations) < 20:
            continue
        first_local_hour = day_observations[0]["observed_at"].astimezone(NYC).hour
        last_local_hour = day_observations[-1]["observed_at"].astimezone(NYC).hour
        if first_local_hour > 1 or last_local_hour < 22:
            continue
        past = [row for row in day_observations if row["observed_at"] < decision_time]
        prior = [row for row in observations if row["observed_at"] < decision_time]
        target = max(row["temperature_f"] for row in day_observations)
        day_of_year = event_date.timetuple().tm_yday
        output.append(
            FeatureRow(
                event_date=event_date,
                decision_time=decision_time,
                decision_hour_local=decision_hour,
                forecast_max_remaining_f=max(forecast_values),
                forecast_mean_remaining_f=float(np.mean(forecast_values)),
                forecast_min_remaining_f=min(forecast_values),
                observed_max_so_far_f=max((row["temperature_f"] for row in past), default=None),
                latest_observation_f=prior[-1]["temperature_f"] if prior else None,
                day_of_year_sin=math.sin(2 * math.pi * day_of_year / 365.25),
                day_of_year_cos=math.cos(2 * math.pi * day_of_year / 365.25),
                target_daily_max_f=target,
                target_rounded_max_f=round_temperature(target),
                observation_count=len(day_observations),
            )
        )
    return output


def _matrix(rows: list[FeatureRow]) -> np.ndarray:
    return np.asarray([row.vector() for row in rows], dtype=np.float64)


def _impute(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    result = matrix.copy()
    missing = ~np.isfinite(result)
    result[missing] = np.take(medians, np.where(missing)[1])
    return result


def _raw_point(rows: list[FeatureRow]) -> np.ndarray:
    return np.asarray(
        [
            max(
                row.forecast_max_remaining_f,
                row.observed_max_so_far_f
                if row.observed_max_so_far_f is not None
                else -np.inf,
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def point_prediction(bundle: dict[str, Any], rows: list[FeatureRow]) -> np.ndarray:
    raw = _raw_point(rows)
    if bundle["candidate"] == "raw_hrrr":
        return raw
    matrix = _impute(_matrix(rows), np.asarray(bundle["imputation_medians"]))
    return np.asarray(bundle["estimator"].predict(matrix), dtype=np.float64)


def bucket_probability(
    point: float,
    residuals: np.ndarray,
    lower: int | None,
    upper: int | None,
) -> float:
    rounded = np.floor(point + residuals + 0.5).astype(np.int64)
    selected = np.ones(rounded.shape, dtype=bool)
    if lower is not None:
        selected &= rounded >= lower
    if upper is not None:
        selected &= rounded <= upper
    # Jeffreys smoothing prevents impossible zero/one probabilities in finite samples.
    return float((selected.sum() + 0.5) / (selected.size + 1.0))


def normalized_bucket_probabilities(
    point: float,
    residuals: np.ndarray,
    buckets: list[tuple[int | None, int | None]],
) -> list[float]:
    values = [bucket_probability(point, residuals, lower, upper) for lower, upper in buckets]
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("temperature bucket probabilities cannot be normalized")
    return [value / total for value in values]


def train_model(
    settings: Settings,
    *,
    candidate: str,
    decision_hour: int,
    training_start: date,
    training_end: date,
    calibration_start: date,
    calibration_end: date,
) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError(f"candidate must be one of {', '.join(CANDIDATES)}")
    if not training_start <= training_end < calibration_start <= calibration_end:
        raise ValueError("training and calibration ranges must be disjoint and chronological")
    train_rows = build_feature_rows(
        settings.database_url, training_start, training_end + timedelta(days=1), decision_hour
    )
    calibration_rows = build_feature_rows(
        settings.database_url,
        calibration_start,
        calibration_end + timedelta(days=1),
        decision_hour,
    )
    if len(train_rows) < 365 or len(calibration_rows) < 90:
        raise ValueError(
            f"insufficient complete days: training={len(train_rows)}, calibration={len(calibration_rows)}"
        )
    train_matrix = _matrix(train_rows)
    medians = np.nanmedian(train_matrix, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("one or more features have no finite training values")
    train_matrix = _impute(train_matrix, medians)
    train_target = np.asarray([row.target_daily_max_f for row in train_rows])
    estimator = None
    if candidate == "linear_bias":
        estimator = Ridge(alpha=1.0).fit(train_matrix, train_target)
    elif candidate == "histogram_residual":
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=17,
        ).fit(train_matrix, train_target)
    bundle: dict[str, Any] = {
        "schema_version": "nyc-temperature-model-v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "candidate": candidate,
        "decision_hour_local": decision_hour,
        "imputation_medians": medians,
        "estimator": estimator,
        "ranges": {
            "training_start": training_start.isoformat(),
            "training_end": training_end.isoformat(),
            "calibration_start": calibration_start.isoformat(),
            "calibration_end": calibration_end.isoformat(),
        },
    }
    calibration_target = np.asarray([row.target_daily_max_f for row in calibration_rows])
    candidate_point = point_prediction(bundle, calibration_rows)
    raw_point = _raw_point(calibration_rows)
    bundle["residuals"] = calibration_target - candidate_point
    bundle["raw_residuals"] = calibration_target - raw_point
    metrics = {
        "training_days": len(train_rows),
        "calibration_days": len(calibration_rows),
        "calibration_mae_f": float(mean_absolute_error(calibration_target, candidate_point)),
        "calibration_rmse_f": float(mean_squared_error(calibration_target, candidate_point) ** 0.5),
        "raw_calibration_mae_f": float(mean_absolute_error(calibration_target, raw_point)),
        "raw_calibration_rmse_f": float(mean_squared_error(calibration_target, raw_point) ** 0.5),
        "residual_p05_f": float(np.quantile(bundle["residuals"], 0.05)),
        "residual_p95_f": float(np.quantile(bundle["residuals"], 0.95)),
    }
    model_run_id = str(uuid.uuid4())
    settings.model_directory.mkdir(parents=True, exist_ok=True)
    model_path = settings.model_directory / (
        f"{candidate}-h{decision_hour:02d}-{training_start}-{calibration_end}-{model_run_id}.joblib"
    )
    temporary = model_path.with_suffix(".partial")
    joblib.dump(bundle, temporary, compress=3)
    temporary.replace(model_path)
    digest, _ = file_sha256(model_path)
    with connection(settings.database_url) as conn, conn.transaction():
        conn.execute(
            """
            INSERT INTO weather.model_runs (
              model_run_id,process_id,candidate,decision_hour_local,
              training_start,training_end,calibration_start,calibration_end,
              feature_schema_version,model_uri,model_sha256,metrics
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                model_run_id,
                PROCESS_ID,
                candidate,
                decision_hour,
                training_start,
                training_end,
                calibration_start,
                calibration_end,
                FEATURE_SCHEMA_VERSION,
                str(model_path),
                digest,
                psycopg.types.json.Jsonb(metrics),
            ),
        )
    return {
        "model_run_id": model_run_id,
        "model_uri": str(model_path),
        "model_sha256": digest,
        "metrics": metrics,
    }


def load_model(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if bundle.get("schema_version") != "nyc-temperature-model-v1":
        raise ValueError("unsupported model schema")
    if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("model feature contract does not match this package")
    return bundle


def reconcile_labels(settings: Settings, start: date, end: date) -> dict[str, Any]:
    start_utc, end_utc = _local_bounds(start, end)
    with connection(settings.database_url) as conn:
        observations = conn.execute(
            """
            SELECT observed_at, temperature_f::double precision AS temperature_f
            FROM weather.station_observations
            WHERE station_id=%s AND provider='iem_asos_metar'
              AND temperature_f IS NOT NULL
              AND observed_at >= %s AND observed_at < %s
            ORDER BY observed_at
            """,
            (STATION_ID, start_utc, end_utc),
        ).fetchall()
        markets = conn.execute(
            """
            SELECT market_id,event_date,bucket_lower_f,bucket_upper_f,resolved_yes
            FROM weather.temperature_markets
            WHERE event_date >= %s AND event_date < %s AND resolved_yes IS NOT NULL
            ORDER BY event_date,market_id
            """,
            (start, end),
        ).fetchall()
    by_date_observations: dict[date, list[dict]] = {}
    for row in observations:
        by_date_observations.setdefault(row["observed_at"].astimezone(NYC).date(), []).append(row)
    by_date_markets: dict[date, list[dict]] = {}
    for market in markets:
        by_date_markets.setdefault(market["event_date"], []).append(market)
    matched = 0
    compared = 0
    with connection(settings.database_url) as conn, conn.transaction():
        current = start
        while current < end:
            obs = by_date_observations.get(current, [])
            day_markets = by_date_markets.get(current, [])
            winners = [market for market in day_markets if market["resolved_yes"]]
            flags = []
            daily_max = max((row["temperature_f"] for row in obs), default=None)
            rounded = round_temperature(daily_max) if daily_max is not None else None
            winner = winners[0] if len(winners) == 1 else None
            complete_day = False
            if obs:
                first_local_hour = obs[0]["observed_at"].astimezone(NYC).hour
                last_local_hour = obs[-1]["observed_at"].astimezone(NYC).hour
                complete_day = len(obs) >= 20 and first_local_hour <= 1 and last_local_hour >= 22
            if not complete_day:
                flags.append("incomplete_station_day")
            if len(winners) != 1:
                flags.append("ambiguous_polymarket_winner")
            winner_matches = None
            if rounded is not None and winner is not None and complete_day:
                lower = winner["bucket_lower_f"]
                upper = winner["bucket_upper_f"]
                winner_matches = (lower is None or rounded >= lower) and (
                    upper is None or rounded <= upper
                )
                compared += 1
                matched += int(winner_matches)
            conn.execute(
                """
                INSERT INTO weather.label_reconciliation (
                  process_id,event_date,station_id,station_daily_max_f,station_rounded_max_f,
                  winning_market_id,winner_matches_station,quality_flags
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (process_id,event_date) DO UPDATE SET
                  station_daily_max_f=EXCLUDED.station_daily_max_f,
                  station_rounded_max_f=EXCLUDED.station_rounded_max_f,
                  winning_market_id=EXCLUDED.winning_market_id,
                  winner_matches_station=EXCLUDED.winner_matches_station,
                  quality_flags=EXCLUDED.quality_flags,calculated_at=now()
                """,
                (
                    PROCESS_ID,
                    current,
                    STATION_ID,
                    daily_max,
                    rounded,
                    winner["market_id"] if winner else None,
                    winner_matches,
                    psycopg.types.json.Jsonb(flags),
                ),
            )
            current += timedelta(days=1)
    return {
        "days_compared": compared,
        "days_matched": matched,
        "agreement": matched / compared if compared else None,
    }
