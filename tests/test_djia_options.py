from datetime import date, datetime, timezone

from collect_djia_options import ET, normalize_contract, rate_limited, within_market_window


CAPTURED = datetime(2026, 11, 3, 20, 0, tzinfo=timezone.utc)
EXPIRY = date(2026, 11, 6)
BASE = {
    "contractSymbol": "AAPL261106C00300000",
    "strike": 300.0,
    "bid": 9.0,
    "ask": 10.0,
    "openInterest": 150,
    "volume": 5,
    "lastPrice": 9.5,
    "impliedVolatility": 0.3,
    "inTheMoney": False,
}


def quote(**changes):
    record = dict(BASE)
    record.update(changes)
    return normalize_contract(record, "AAPL", EXPIRY, "C", CAPTURED, 305.0)


def test_accepted_quote_retains_identifiers_and_market_fields():
    row = quote()
    assert row is not None
    assert row[3] == BASE["contractSymbol"]
    assert row[6:8] == (9.0, 10.0)
    assert row[10] == 150


def test_liquidity_boundaries():
    assert quote(openInterest=99) is None
    assert quote(openInterest=100) is not None
    assert quote(bid=0) is None
    assert quote(bid=10, ask=9) is None
    assert quote(bid=7, ask=10) is None
    assert quote(bid=8, ask=10) is not None


def test_bad_contract_or_nan_is_rejected():
    assert quote(contractSymbol="MSFT261106C00300000") is None
    assert quote(contractSymbol="AAPL261113C00300000") is None
    assert quote(ask=float("nan")) is None
    assert quote(openInterest=None) is None


def test_rate_limit_detection():
    assert rate_limited(RuntimeError("HTTP 429 Too Many Requests"))
    assert not rate_limited(ValueError("empty option chain"))


def test_market_window_uses_nyse_holidays_and_early_closes():
    assert not within_market_window(datetime(2026, 11, 26, 10, 0, tzinfo=ET))
    assert within_market_window(datetime(2026, 11, 27, 12, 0, tzinfo=ET))
    assert within_market_window(datetime(2026, 11, 27, 13, 0, tzinfo=ET))
    assert not within_market_window(datetime(2026, 11, 27, 14, 0, tzinfo=ET))
    assert within_market_window(datetime(2026, 12, 4, 16, 0, tzinfo=ET))
    assert not within_market_window(datetime(2026, 12, 4, 16, 6, tzinfo=ET))
