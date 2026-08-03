from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.dataset as ds

from .config import Settings
from .database import connection, insert_artifact
from .jobs import Job, update_progress
from .sources import download_atomic, file_sha256, source_client

NYC = ZoneInfo("America/New_York")
PMXT_COVERAGE_START = datetime(2026, 4, 13, 19, tzinfo=UTC)
MAXIMUM_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024


@dataclass
class Book:
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)
    seeded: bool = False
    source_timestamp: datetime | None = None


def _levels(value: Any) -> dict[Decimal, Decimal]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    output = {}
    for level in value:
        if isinstance(level, dict):
            price_value, size_value = level["price"], level["size"]
        else:
            price_value, size_value = level
        price = Decimal(str(price_value))
        size = Decimal(str(size_value))
        if size > 0:
            output[price] = size
    return output


def _apply_event(book: Book, event: dict[str, Any]) -> None:
    event_type = str(event["event_type"])
    timestamp = event["timestamp"]
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    book.source_timestamp = max(book.source_timestamp or timestamp, timestamp)
    if event_type == "book":
        book.bids = _levels(event.get("bids"))
        book.asks = _levels(event.get("asks"))
        book.seeded = True
        return
    if event_type != "price_change" or not book.seeded:
        return
    side = str(event.get("side") or "").upper()
    levels = book.bids if side == "BUY" else book.asks if side == "SELL" else None
    if levels is None or event.get("price") is None or event.get("size") is None:
        return
    price = Decimal(str(event["price"]))
    size = Decimal(str(event["size"]))
    if size <= 0:
        levels.pop(price, None)
    else:
        levels[price] = size


def _vwap(asks: dict[Decimal, Decimal], quantity: Decimal) -> Decimal | None:
    remaining = quantity
    notional = Decimal(0)
    for price, size in sorted(asks.items()):
        take = min(remaining, size)
        notional += price * take
        remaining -= take
        if remaining <= 0:
            return notional / quantity
    return None


def _archive_spec(base_url: str, hour: datetime) -> tuple[str, str]:
    stamp = hour.astimezone(UTC).strftime("%Y-%m-%dT%H")
    name = f"polymarket_orderbook_{stamp}.parquet"
    return f"{base_url}/{name}", name


def _ensure_archive(settings: Settings, hour: datetime) -> tuple[Path, str, int, str]:
    uri, name = _archive_spec(settings.pmxt_base_url, hour)
    path = settings.cache_directory / "pmxt" / name
    if path.exists():
        digest, size = file_sha256(path)
    else:
        with source_client() as client:
            digest, size = download_atomic(
                client,
                uri,
                path,
                MAXIMUM_ARCHIVE_BYTES,
                attempts=settings.pmxt_download_attempts,
                retry_base_seconds=settings.pmxt_retry_base_seconds,
            )
    return path, digest, size, uri


def _filtered_events(paths: list[Path], condition_ids: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    byte_ids = [value.encode("ascii") for value in condition_ids]
    columns = [
        "timestamp_received",
        "timestamp",
        "market",
        "event_type",
        "asset_id",
        "bids",
        "asks",
        "price",
        "size",
        "side",
    ]
    for path in paths:
        dataset = ds.dataset(path, format="parquet")
        market_type = dataset.schema.field("market").type
        values = pa.array(byte_ids, type=market_type)
        table = dataset.to_table(filter=ds.field("market").isin(values), columns=columns)
        events.extend(table.to_pylist())
    events.sort(key=lambda row: (row["timestamp_received"], row["timestamp"]))
    return events


def _decision_groups(settings: Settings, job: Job):
    with connection(settings.database_url) as conn:
        rows = conn.execute(
            """
            SELECT market_id, event_date, condition_id, yes_token_id, no_token_id
            FROM weather.temperature_markets
            WHERE event_date >= (%s AT TIME ZONE 'America/New_York')::date
              AND event_date <= ((%s - interval '1 microsecond')
                                  AT TIME ZONE 'America/New_York')::date
            ORDER BY event_date, market_id
            """,
            (job.range_start, job.range_end),
        ).fetchall()
    grouped: dict[datetime, list[dict]] = {}
    for market in rows:
        for hour in (0, 12):
            decision = datetime.combine(market["event_date"], datetime.min.time(), NYC).replace(
                hour=hour
            ).astimezone(UTC)
            if job.range_start <= decision < job.range_end and decision >= PMXT_COVERAGE_START:
                grouped.setdefault(decision, []).append(market)
    return sorted(grouped.items())


def ingest_pmxt_execution(settings: Settings, job: Job) -> dict:
    groups = _decision_groups(settings, job)
    snapshots = 0
    archives_seen: set[str] = set()
    for index, (decision_time, markets) in enumerate(groups, start=1):
        hour = decision_time.replace(minute=0, second=0, microsecond=0)
        archive_hours = [hour - timedelta(hours=1), hour]
        paths = []
        artifact_ids = []
        for archive_hour in archive_hours:
            if archive_hour < PMXT_COVERAGE_START.replace(minute=0):
                continue
            path, digest, size, uri = _ensure_archive(settings, archive_hour)
            with connection(settings.database_url) as conn, conn.transaction():
                artifact_id = insert_artifact(
                    conn,
                    provider="pmxt_v2",
                    logical_key=f"pmxt:v2:polymarket_orderbook:{archive_hour:%Y-%m-%dT%H}",
                    source_uri=uri,
                    sha256=digest,
                    compressed_bytes=size,
                    record_count=0,
                    metadata={"archive_hour": archive_hour.isoformat()},
                    source_start=archive_hour,
                    source_end=archive_hour + timedelta(hours=1),
                )
            paths.append(path)
            artifact_ids.append(artifact_id)
            archives_seen.add(str(path))
        events = _filtered_events(paths, [market["condition_id"] for market in markets])
        books: dict[str, Book] = {}
        for event in events:
            received = event["timestamp_received"]
            if received.tzinfo is None:
                received = received.replace(tzinfo=UTC)
            if received > decision_time:
                break
            asset = str(event["asset_id"])
            _apply_event(books.setdefault(asset, Book()), event)
        with connection(settings.database_url) as conn, conn.transaction():
            for market in markets:
                yes = books.get(market["yes_token_id"], Book())
                no = books.get(market["no_token_id"], Book())
                for quantity in (Decimal(1), Decimal(5), Decimal(10)):
                    flags = []
                    if not yes.seeded:
                        flags.append("missing_yes_book_seed")
                    if not no.seeded:
                        flags.append("missing_no_book_seed")
                    yes_vwap = _vwap(yes.asks, quantity) if yes.seeded else None
                    no_vwap = _vwap(no.asks, quantity) if no.seeded else None
                    if yes_vwap is None:
                        flags.append(f"insufficient_yes_ask_depth_{int(quantity)}")
                    if no_vwap is None:
                        flags.append(f"insufficient_no_ask_depth_{int(quantity)}")
                    if yes.bids and yes.asks and max(yes.bids) >= min(yes.asks):
                        flags.append("crossed_yes_book")
                    if no.bids and no.asks and max(no.bids) >= min(no.asks):
                        flags.append("crossed_no_book")
                    source_timestamp = max(
                        [value for value in (yes.source_timestamp, no.source_timestamp) if value],
                        default=None,
                    )
                    conn.execute(
                        """
                        INSERT INTO weather.execution_snapshots (
                          market_id,decision_time,quantity,yes_ask_vwap,no_ask_vwap,
                          yes_best_ask,no_best_ask,source_timestamp,quality_flags,source_artifact_id
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (market_id,decision_time,quantity) DO UPDATE SET
                          yes_ask_vwap=EXCLUDED.yes_ask_vwap,
                          no_ask_vwap=EXCLUDED.no_ask_vwap,
                          yes_best_ask=EXCLUDED.yes_best_ask,
                          no_best_ask=EXCLUDED.no_best_ask,
                          source_timestamp=EXCLUDED.source_timestamp,
                          quality_flags=EXCLUDED.quality_flags,
                          source_artifact_id=EXCLUDED.source_artifact_id
                        """,
                        (
                            market["market_id"],
                            decision_time,
                            quantity,
                            yes_vwap,
                            no_vwap,
                            min(yes.asks) if yes.asks else None,
                            min(no.asks) if no.asks else None,
                            source_timestamp,
                            __import__("psycopg").types.json.Jsonb(flags),
                            artifact_ids[-1] if artifact_ids else None,
                        ),
                    )
                    snapshots += 1
        for path in paths:
            path.unlink(missing_ok=True)
        update_progress(
            settings,
            job,
            {
                "decision_groups_completed": index,
                "decision_groups_total": len(groups),
                "snapshots": snapshots,
                "archives": len(archives_seen),
                "last_decision_time": decision_time.isoformat(),
            },
        )
    return {
        "decision_groups": len(groups),
        "snapshots": snapshots,
        "archives": len(archives_seen),
    }
