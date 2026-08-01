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
        options="-c statement_timeout=300000 -c lock_timeout=5000",
    )
    try:
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
