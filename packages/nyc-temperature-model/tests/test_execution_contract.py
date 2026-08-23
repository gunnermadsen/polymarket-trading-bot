from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import httpx

from nyc_temperature_model import execution_ingestion
from nyc_temperature_model.execution_ingestion import (
    Book,
    _apply_event,
    _decode_archive_with_redownload,
    _last_trade_rows,
    _levels,
    _load_archive_events,
    _manifest_digest,
    _retryable_archive_gap,
    _vwap,
)
from nyc_temperature_model.sources import download_atomic


def test_book_reconstruction_and_taker_vwap():
    book = Book()
    _apply_event(
        book,
        {
            "event_type": "book",
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").UTC),
            "bids": '[{"price":"0.40","size":"5"}]',
            "asks": '[{"price":"0.50","size":"2"},{"price":"0.60","size":"4"}]',
        },
    )
    assert _vwap(book.asks, Decimal(5)) == Decimal("0.56")
    assert _vwap(book.asks, Decimal(10)) is None


def test_price_change_never_invents_an_unseeded_book():
    book = Book()
    _apply_event(
        book,
        {
            "event_type": "price_change",
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").UTC),
            "side": "SELL",
            "price": "0.50",
            "size": "10",
        },
    )
    assert not book.seeded
    assert book.asks == {}


def test_pmxt_book_levels_accept_archive_pair_encoding():
    assert _levels('[["0.40", "2.5"], ["0.41", "0"]]') == {
        Decimal("0.40"): Decimal("2.5")
    }


def test_pmxt_archive_download_resumes_a_legacy_partial_file(tmp_path):
    payload = b"archive-data"
    destination = tmp_path / "archive.parquet"
    legacy_partial = tmp_path / ".archive.parquet.123.partial"
    legacy_partial.write_bytes(payload[:7])
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["range"] == "bytes=7-"
        return httpx.Response(
            206,
            headers={"content-range": f"bytes 7-{len(payload) - 1}/{len(payload)}"},
            content=payload[7:],
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        digest, size = download_atomic(client, "https://example.test/archive", destination, 1024)

    assert len(requests) == 1
    assert destination.read_bytes() == payload
    assert size == len(payload)
    assert digest == __import__("hashlib").sha256(payload).hexdigest()


def test_pmxt_archive_download_retries_a_transient_source_failure(tmp_path):
    payload = b"archive-data"
    destination = tmp_path / "archive.parquet"
    calls = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, headers={"content-length": str(len(payload))}, content=payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        digest, size = download_atomic(
            client,
            "https://example.test/archive",
            destination,
            1024,
            attempts=2,
            retry_base_seconds=0,
            sleep=lambda _seconds: None,
        )

    assert len(calls) == 2
    assert destination.read_bytes() == payload
    assert size == len(payload)
    assert digest == __import__("hashlib").sha256(payload).hexdigest()


def test_pmxt_corrupt_archive_is_evicted_and_redownloaded_once(tmp_path):
    corrupt = tmp_path / "archive.parquet"
    replacement = tmp_path / "replacement.parquet"
    corrupt.write_bytes(b"corrupt")
    replacement.write_bytes(b"valid")
    calls = []

    def decode(paths, _condition_ids):
        calls.append(paths[0])
        if len(calls) == 1:
            raise OSError("ZSTD decompression failed")
        return [{"event_type": "book"}]

    events, selected, recovered = _decode_archive_with_redownload(
        corrupt,
        ["condition"],
        lambda: replacement,
        decoder=decode,
    )

    assert events == [{"event_type": "book"}]
    assert selected == replacement
    assert recovered
    assert not corrupt.exists()
    assert calls == [corrupt, replacement]


def test_pmxt_twice_corrupt_archive_is_evicted_and_reported(tmp_path):
    corrupt = tmp_path / "archive.parquet"
    replacement = tmp_path / "replacement.parquet"
    corrupt.write_bytes(b"corrupt")
    replacement.write_bytes(b"also-corrupt")

    def decode(_paths, _condition_ids):
        raise OSError("ZSTD decompression failed")

    with __import__("pytest").raises(OSError, match="ZSTD"):
        _decode_archive_with_redownload(
            corrupt,
            ["condition"],
            lambda: replacement,
            decoder=decode,
        )

    assert not corrupt.exists()
    assert not replacement.exists()


def test_pmxt_missing_archive_is_recorded_as_a_quality_gap(monkeypatch):
    archive_hour = datetime(2026, 6, 11, 4, tzinfo=UTC)
    recorded = {}

    def missing_archive(_settings, _archive_hour):
        raise FileNotFoundError("missing archive")

    def record_archive(_settings, _archive_hour, uri, **metadata):
        recorded.update(uri=uri, **metadata)
        return "artifact-id"

    monkeypatch.setattr(execution_ingestion, "_ensure_archive", missing_archive)
    monkeypatch.setattr(execution_ingestion, "_record_archive", record_archive)

    events, artifact_id, quality_flag = _load_archive_events(
        SimpleNamespace(pmxt_base_url="https://example.test"),
        archive_hour,
        ["condition"],
    )

    assert events == []
    assert artifact_id == "artifact-id"
    assert quality_flag == "pmxt_archive_missing_2026-06-11T04Z"
    assert recorded["availability"] == "missing"
    assert recorded["uri"].endswith("polymarket_orderbook_2026-06-11T04.parquet")


def test_archive_gap_snapshots_are_retryable():
    assert _retryable_archive_gap(["pmxt_archive_missing_2026-08-13T04Z"])
    assert _retryable_archive_gap('["pmxt_archive_corrupt_2026-08-13T04Z"]')
    assert not _retryable_archive_gap(["insufficient_yes_ask_depth_5"])


def test_manifest_digest_covers_both_archive_hours_and_token_identity():
    decision = datetime(2026, 8, 13, 4, tzinfo=UTC)
    market = {
        "market_id": "market-1",
        "condition_id": "condition-1",
        "yes_token_id": "yes-1",
        "no_token_id": "no-1",
    }

    first, archive_count = _manifest_digest([(decision, [market])])
    second, _ = _manifest_digest([(decision, [{**market, "yes_token_id": "yes-2"}])])

    assert archive_count == 2
    assert first != second


def test_last_trade_prices_are_causal_filtered_and_content_addressed():
    decision = datetime(2026, 8, 13, 16, tzinfo=UTC)
    base = {
        "event_type": "last_trade_price",
        "asset_id": "yes-token",
        "timestamp": datetime(2026, 8, 13, 15, 59, 57, tzinfo=UTC),
        "timestamp_received": datetime(2026, 8, 13, 15, 59, 58, tzinfo=UTC),
        "price": "0.12",
        "size": "5",
        "side": "BUY",
        "_source_artifact_id": "artifact-1",
    }
    future = {
        **base,
        "timestamp_received": datetime(2026, 8, 13, 16, 0, 1, tzinfo=UTC),
    }
    rows = _last_trade_rows(
        [base, future],
        [
            {
                "market_id": "market-1",
                "yes_token_id": "yes-token",
                "no_token_id": "no-token",
            }
        ],
        decision,
    )

    assert len(rows) == 1
    assert rows[0]["outcome"] == "YES"
    assert rows[0]["price"] == Decimal("0.12")
    assert len(rows[0]["event_id"]) == 64
    assert rows[0]["provider_received_at"] <= decision
