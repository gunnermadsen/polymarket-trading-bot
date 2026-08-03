from decimal import Decimal

import httpx

from nyc_temperature_model.execution_ingestion import Book, _apply_event, _levels, _vwap
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
