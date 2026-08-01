from decimal import Decimal

from nyc_temperature_model.execution_ingestion import Book, _apply_event, _levels, _vwap


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
