from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row


@contextmanager
def connection(database_url: str, *, autocommit: bool = False) -> Iterator[Connection]:
    conn = psycopg.connect(
        database_url,
        autocommit=autocommit,
        row_factory=dict_row,
    )
    try:
        conn.execute("SET statement_timeout = 300000")
        conn.execute("SET lock_timeout = 5000")
        yield conn
    finally:
        conn.close()


def insert_artifact(
    conn: Connection,
    *,
    provider: str,
    logical_key: str,
    source_uri: str,
    sha256: str | None,
    compressed_bytes: int | None,
    record_count: int,
    metadata: dict,
    source_start=None,
    source_end=None,
) -> str:
    row = conn.execute(
        """
        INSERT INTO weather.source_artifacts (
          provider, logical_key, source_uri, source_start, source_end,
          sha256, compressed_bytes, record_count, metadata
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (provider, logical_key) DO UPDATE SET
          source_uri = EXCLUDED.source_uri,
          source_start = EXCLUDED.source_start,
          source_end = EXCLUDED.source_end,
          sha256 = EXCLUDED.sha256,
          compressed_bytes = EXCLUDED.compressed_bytes,
          record_count = EXCLUDED.record_count,
          metadata = EXCLUDED.metadata,
          ingested_at = now()
        RETURNING artifact_id::text
        """,
        (
            provider,
            logical_key,
            source_uri,
            source_start,
            source_end,
            sha256,
            compressed_bytes,
            record_count,
            psycopg.types.json.Jsonb(metadata),
        ),
    ).fetchone()
    return row["artifact_id"]


def insert_immutable_artifact(
    conn: Connection,
    *,
    provider: str,
    logical_key: str,
    source_uri: str,
    sha256: str,
    compressed_bytes: int,
    record_count: int,
    metadata: dict,
    source_start=None,
    source_end=None,
) -> str:
    """Insert an immutable source ledger row or verify an identical prior insert."""
    row = conn.execute(
        """
        INSERT INTO weather.source_artifacts (
          provider, logical_key, source_uri, source_start, source_end,
          sha256, compressed_bytes, record_count, metadata
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (provider, logical_key) DO NOTHING
        RETURNING artifact_id::text
        """,
        (
            provider,
            logical_key,
            source_uri,
            source_start,
            source_end,
            sha256,
            compressed_bytes,
            record_count,
            psycopg.types.json.Jsonb(metadata),
        ),
    ).fetchone()
    if row:
        return row["artifact_id"]
    existing = conn.execute(
        """
        SELECT artifact_id::text, source_uri, sha256, compressed_bytes
        FROM weather.source_artifacts
        WHERE provider=%s AND logical_key=%s
        """,
        (provider, logical_key),
    ).fetchone()
    if not existing or (
        existing["source_uri"] != source_uri
        or existing["sha256"] != sha256
        or existing["compressed_bytes"] != compressed_bytes
    ):
        raise RuntimeError(f"immutable artifact conflict for {provider}:{logical_key}")
    return existing["artifact_id"]


def insert_unified_backfill_artifact(
    conn: Connection,
    *,
    job_id: str,
    strategy_key: str,
    provider: str,
    logical_key: str,
    source_uri: str,
    sha256: str,
    compressed_bytes: int,
    record_count: int,
    metadata: dict,
    source_start=None,
    source_end=None,
) -> str:
    """Record immutable source lineage in the single ingester artifact ledger."""
    row = conn.execute(
        """
        INSERT INTO ingester.backfill_artifacts (
          job_id,strategy_key,logical_key,provider,source_uri,checksum,byte_size,
          record_count,minimum_source_timestamp,maximum_source_timestamp,
          status,metadata,completed_at
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'completed',%s,now())
        ON CONFLICT (strategy_key,logical_key) DO NOTHING
        RETURNING artifact_id::text
        """,
        (
            job_id,
            strategy_key,
            logical_key,
            provider,
            source_uri,
            sha256,
            compressed_bytes,
            record_count,
            source_start,
            source_end,
            psycopg.types.json.Jsonb(metadata),
        ),
    ).fetchone()
    if row:
        return row["artifact_id"]
    existing = conn.execute(
        """
        SELECT artifact_id::text,source_uri,checksum,byte_size
        FROM ingester.backfill_artifacts
        WHERE strategy_key=%s AND logical_key=%s
        """,
        (strategy_key, logical_key),
    ).fetchone()
    if not existing or (
        existing["source_uri"] != source_uri
        or existing["checksum"] != sha256
        or existing["byte_size"] != compressed_bytes
    ):
        raise RuntimeError(f"immutable artifact conflict for {provider}:{logical_key}")
    return existing["artifact_id"]
