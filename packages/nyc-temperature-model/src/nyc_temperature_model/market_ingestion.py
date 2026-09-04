from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo

import psycopg

from .config import Settings
from .database import connection, insert_artifact
from .fees import fee_schedule_from_market
from .backfill_contract import Job, update_progress
from .sources import atomic_write, source_client

NYC = ZoneInfo("America/New_York")
NYC_DAILY_WEATHER_SERIES_ID = 10005
TITLE_PATTERN = re.compile(r"highest temperature in nyc", re.IGNORECASE)
RANGE_PATTERN = re.compile(r"(-?\d+)\s*(?:-|–|to)\s*(-?\d+)\s*°?\s*f", re.IGNORECASE)
LOWER_PATTERN = re.compile(
    r"(-?\d+)\s*°?\s*f\s*(?:or|and)?\s*(?:below|lower|less)", re.IGNORECASE
)
UPPER_PATTERN = re.compile(
    r"(-?\d+)\s*°?\s*f\s*(?:or|and)?\s*(?:above|higher|more)", re.IGNORECASE
)
EXACT_PATTERN = re.compile(r"(?:be|is)\s*(-?\d+)\s*°?\s*f", re.IGNORECASE)
DATE_PATTERN = re.compile(r"\bon\s+([A-Za-z]+\s+\d{1,2}(?:,?\s+\d{4})?)", re.IGNORECASE)


def _json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    raise ValueError("expected a JSON array")


def parse_bucket(question: str) -> tuple[int | None, int | None]:
    if match := RANGE_PATTERN.search(question):
        lower, upper = int(match.group(1)), int(match.group(2))
        if upper < lower:
            raise ValueError(f"temperature bucket is reversed: {question}")
        return lower, upper
    if match := LOWER_PATTERN.search(question):
        return None, int(match.group(1))
    if match := UPPER_PATTERN.search(question):
        return int(match.group(1)), None
    if match := EXACT_PATTERN.search(question):
        value = int(match.group(1))
        return value, value
    raise ValueError(f"could not parse temperature bucket: {question}")


def parse_event_date(event: dict[str, Any]) -> datetime.date:
    title = str(event.get("title") or "")
    if match := DATE_PATTERN.search(title):
        value = match.group(1).replace(",", "")
        for pattern in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(value, pattern).replace(tzinfo=NYC).date()
            except ValueError:
                pass
        slug_year = re.search(r"-(20\d{2})(?:$|-)", str(event.get("slug") or ""))
        if slug_year:
            value = f"{value} {slug_year.group(1)}"
            for pattern in ("%B %d %Y", "%b %d %Y"):
                try:
                    return datetime.strptime(value, pattern).replace(tzinfo=NYC).date()
                except ValueError:
                    pass
    end_date = event.get("endDate") or event.get("end_date")
    if not end_date:
        raise ValueError(f"event has no usable date: {event.get('slug')}")
    parsed = datetime.fromisoformat(str(end_date))
    return parsed.astimezone(NYC).date()


def resolved_yes(market: dict[str, Any]) -> bool | None:
    if not market.get("closed"):
        return None
    prices = [Decimal(str(value)) for value in _json_array(market.get("outcomePrices", []))]
    outcomes = [str(value).strip().lower() for value in _json_array(market.get("outcomes", []))]
    if len(prices) != 2 or len(outcomes) != 2 or set(outcomes) != {"yes", "no"}:
        return None
    winners = [index for index, price in enumerate(prices) if price >= Decimal("0.999")]
    losers = [index for index, price in enumerate(prices) if price <= Decimal("0.001")]
    if len(winners) != 1 or len(losers) != 1:
        return None
    return outcomes[winners[0]] == "yes"


def _market_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    if not TITLE_PATTERN.search(str(event.get("title") or "")):
        return []
    markets = event.get("markets") or []
    if not markets or any(not _eligible_resolution_source(event, market) for market in markets):
        return []
    event_date = parse_event_date(event)
    rows = []
    for market in markets:
        fee_schedule = fee_schedule_from_market(market)
        question = str(market.get("question") or "")
        lower, upper = parse_bucket(question)
        tokens = [str(value) for value in _json_array(market.get("clobTokenIds", []))]
        outcomes = [str(value).strip().lower() for value in _json_array(market.get("outcomes", []))]
        if len(tokens) != 2 or len(outcomes) != 2 or set(outcomes) != {"yes", "no"}:
            raise ValueError(f"NYC temperature market has an invalid outcome contract: {question}")
        by_outcome = dict(zip(outcomes, tokens, strict=True))
        resolution = resolved_yes(market)
        resolved_at = None
        if resolution is not None:
            raw_time = market.get("closedTime") or event.get("closedTime") or event.get("endDate")
            resolved_at = datetime.fromisoformat(str(raw_time))
        rows.append(
            {
                "market_id": str(market["id"]),
                "event_id": str(event["id"]),
                "event_slug": str(event["slug"]),
                "market_slug": str(market["slug"]),
                "event_date": event_date,
                "question": question,
                "condition_id": str(market["conditionId"]),
                "yes_token_id": by_outcome["yes"],
                "no_token_id": by_outcome["no"],
                "bucket_lower_f": lower,
                "bucket_upper_f": upper,
                "active": bool(market.get("active")),
                "closed": bool(market.get("closed")),
                "accepting_orders": bool(market.get("acceptingOrders")),
                "volume_usd": market.get("volumeNum") or market.get("volume"),
                "liquidity_usd": market.get("liquidityNum") or market.get("liquidity"),
                "resolved_yes": resolution,
                "resolved_at": resolved_at,
                "resolution_source": "gamma_terminal_prices" if resolution is not None else None,
                "fee_rate_bps": round(fee_schedule.rate * 10_000),
                "fees_enabled": fee_schedule.enabled,
                "fee_rate": fee_schedule.rate,
                "fee_exponent": fee_schedule.exponent,
                "fee_taker_only": fee_schedule.taker_only,
                "raw_payload": market,
            }
        )
    validate_bucket_partition(rows)
    return rows


def _eligible_resolution_source(event: dict[str, Any], market: dict[str, Any]) -> bool:
    resolution_url = str(market.get("resolutionSource") or event.get("resolutionSource") or "")
    normalized = resolution_url.lower()
    return "wunderground.com/history/daily" in normalized and "klga" in normalized


def validate_bucket_partition(rows: list[dict[str, Any]]) -> None:
    if len(rows) < 3:
        raise ValueError("NYC temperature event must expose at least three buckets")
    if sum(row["bucket_lower_f"] is None for row in rows) != 1:
        raise ValueError("NYC temperature buckets must have exactly one open lower tail")
    if sum(row["bucket_upper_f"] is None for row in rows) != 1:
        raise ValueError("NYC temperature buckets must have exactly one open upper tail")
    ordered = sorted(
        rows,
        key=lambda row: row["bucket_lower_f"] if row["bucket_lower_f"] is not None else -10_000,
    )
    for previous, current in pairwise(ordered):
        previous_upper = previous["bucket_upper_f"]
        current_lower = current["bucket_lower_f"]
        if previous_upper is None or current_lower is None or previous_upper + 1 != current_lower:
            raise ValueError("NYC temperature buckets must be non-overlapping and contiguous")


def ingest_markets(settings: Settings, job: Job) -> dict[str, Any]:
    records = 0
    pages = 0
    ineligible_events = 0
    seen_markets: set[str] = set()
    cache = settings.cache_directory / "gamma"
    with source_client() as client:
        for closed in (True, False):
            offset = 0
            while True:
                params = {
                    "closed": str(closed).lower(),
                    "limit": 100,
                    "offset": offset,
                    "series_id": NYC_DAILY_WEATHER_SERIES_ID,
                    "end_date_min": job.range_start.astimezone(UTC).isoformat(),
                    "end_date_max": job.range_end.astimezone(UTC).isoformat(),
                }
                response = client.get(f"{settings.gamma_base_url}/events", params=params)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list):
                    raise TypeError("Gamma events response was not a list")
                encoded = response.content
                logical_key = f"gamma:events:{job.job_id}:{int(closed)}:{offset}"
                path = cache / f"{job.job_id}-{int(closed)}-{offset}.json"
                digest, size = atomic_write(path, encoded)
                rows = []
                for event in payload:
                    markets = event.get("markets") or []
                    if TITLE_PATTERN.search(str(event.get("title") or "")) and (
                        not markets
                        or any(not _eligible_resolution_source(event, market) for market in markets)
                    ):
                        ineligible_events += 1
                    for row in _market_rows(event):
                        if row["market_id"] not in seen_markets:
                            seen_markets.add(row["market_id"])
                            rows.append(row)
                with connection(settings.database_url) as conn, conn.transaction():
                    artifact_id = insert_artifact(
                        conn,
                        provider="polymarket_gamma",
                        logical_key=logical_key,
                        source_uri=str(response.request.url),
                        sha256=digest,
                        compressed_bytes=size,
                        record_count=len(rows),
                        metadata={"closed": closed, "offset": offset},
                        source_start=job.range_start,
                        source_end=job.range_end,
                    )
                    for row in rows:
                        conn.execute(
                            """
                            INSERT INTO weather.temperature_markets (
                              market_id,event_id,event_slug,market_slug,event_date,question,
                              condition_id,yes_token_id,no_token_id,bucket_lower_f,bucket_upper_f,
                              active,closed,accepting_orders,volume_usd,liquidity_usd,resolved_yes,
                              resolved_at,resolution_source,fee_rate_bps,fees_enabled,fee_rate,
                              fee_exponent,fee_taker_only,source_artifact_id,raw_payload
                            ) VALUES (
                              %(market_id)s,%(event_id)s,%(event_slug)s,%(market_slug)s,%(event_date)s,
                              %(question)s,%(condition_id)s,%(yes_token_id)s,%(no_token_id)s,
                              %(bucket_lower_f)s,%(bucket_upper_f)s,%(active)s,%(closed)s,
                              %(accepting_orders)s,%(volume_usd)s,%(liquidity_usd)s,%(resolved_yes)s,
                              %(resolved_at)s,%(resolution_source)s,%(fee_rate_bps)s,
                              %(fees_enabled)s,%(fee_rate)s,%(fee_exponent)s,
                              %(fee_taker_only)s,%(artifact_id)s,
                              %(raw_payload)s
                            )
                            ON CONFLICT (market_id) DO UPDATE SET
                              active=EXCLUDED.active, closed=EXCLUDED.closed,
                              accepting_orders=EXCLUDED.accepting_orders,
                              volume_usd=EXCLUDED.volume_usd, liquidity_usd=EXCLUDED.liquidity_usd,
                              resolved_yes=EXCLUDED.resolved_yes, resolved_at=EXCLUDED.resolved_at,
                              resolution_source=EXCLUDED.resolution_source,
                              fee_rate_bps=EXCLUDED.fee_rate_bps,
                              fees_enabled=EXCLUDED.fees_enabled,
                              fee_rate=EXCLUDED.fee_rate,
                              fee_exponent=EXCLUDED.fee_exponent,
                              fee_taker_only=EXCLUDED.fee_taker_only,
                              source_artifact_id=EXCLUDED.source_artifact_id,
                              raw_payload=EXCLUDED.raw_payload, refreshed_at=now()
                            """,
                            {
                                **row,
                                "artifact_id": artifact_id,
                                "raw_payload": psycopg.types.json.Jsonb(row["raw_payload"]),
                            },
                        )
                records += len(rows)
                pages += 1
                update_progress(
                    settings,
                    job,
                    {
                        "pages": pages,
                        "markets": records,
                        "ineligible_events": ineligible_events,
                    },
                )
                if len(payload) < 100:
                    break
                offset += 100
    return {"pages": pages, "markets": records, "ineligible_events": ineligible_events}


def ingest_price_history(settings: Settings, job: Job) -> dict[str, Any]:
    with connection(settings.database_url) as conn:
        markets = conn.execute(
            """
            SELECT market_id, yes_token_id, no_token_id, event_date
            FROM weather.temperature_markets
            WHERE event_date >= %s::date AND event_date < %s::date
            ORDER BY event_date, market_id
            """,
            (job.range_start, job.range_end),
        ).fetchall()
    points = 0
    tokens = 0
    cache = settings.cache_directory / "clob-price-history"
    with source_client() as client:
        for market in markets:
            for outcome, token_id in (("yes", market["yes_token_id"]), ("no", market["no_token_id"])):
                start = datetime.combine(
                    market["event_date"] - timedelta(days=1), datetime.min.time(), NYC
                ).astimezone(UTC)
                end = datetime.combine(
                    market["event_date"] + timedelta(days=1), datetime.min.time(), NYC
                ).astimezone(UTC)
                response = client.get(
                    f"{settings.clob_base_url}/prices-history",
                    params={
                        "market": token_id,
                        "startTs": int(start.timestamp()),
                        "endTs": int(end.timestamp()),
                        "interval": "all",
                        "fidelity": 1,
                    },
                )
                response.raise_for_status()
                history = response.json().get("history", [])
                logical_key = f"clob:prices-history:{token_id}:{start.date()}:{end.date()}"
                path = cache / f"{token_id}-{start.date()}-{end.date()}.json"
                digest, size = atomic_write(path, response.content)
                with connection(settings.database_url) as conn, conn.transaction():
                    artifact_id = insert_artifact(
                        conn,
                        provider="polymarket_clob_price_history",
                        logical_key=logical_key,
                        source_uri=str(response.request.url),
                        sha256=digest,
                        compressed_bytes=size,
                        record_count=len(history),
                        metadata={"market_id": market["market_id"], "outcome": outcome},
                        source_start=start,
                        source_end=end,
                    )
                    with conn.cursor() as cursor:
                        cursor.executemany(
                            """
                            INSERT INTO weather.price_history (
                              token_id, observed_at, price, source_artifact_id
                            ) VALUES (%s,to_timestamp(%s),%s,%s)
                            ON CONFLICT (token_id, observed_at) DO UPDATE SET
                              price=EXCLUDED.price, source_artifact_id=EXCLUDED.source_artifact_id
                            """,
                            [
                                (token_id, int(item["t"]), Decimal(str(item["p"])), artifact_id)
                                for item in history
                            ],
                        )
                points += len(history)
                tokens += 1
                update_progress(settings, job, {"tokens": tokens, "price_points": points})
    return {"tokens": tokens, "price_points": points}
