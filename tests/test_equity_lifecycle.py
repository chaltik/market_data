import pandas as pd

import data_retrieval as retrieval


def test_lifecycle_cutoff_removes_only_post_acquisition_rows(monkeypatch, tmp_path):
    registry = tmp_path / "events.yaml"
    registry.write_text("equities:\n  GONE:\n    last_trading_date: '2024-01-02'\n")
    monkeypatch.setattr(retrieval, "EQUITY_LIFECYCLE_FILE", registry)
    retrieval.equity_lifecycle_events.cache_clear()
    frame = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-02 21:00Z", "2024-01-03 21:00Z"]),
        "close": [10.0, 10.0], "symbol": ["GONE", "GONE"],
    })
    result = retrieval.apply_equity_lifecycle_cutoff(frame, "GONE")
    assert result["ts"].tolist() == [pd.Timestamp("2024-01-02 21:00Z")]
    retrieval.equity_lifecycle_events.cache_clear()


def test_registered_lifecycle_skips_a_later_requested_range(monkeypatch, tmp_path):
    registry = tmp_path / "events.yaml"
    registry.write_text("equities:\n  GONE:\n    last_trading_date: '2024-01-02'\n")
    monkeypatch.setattr(retrieval, "EQUITY_LIFECYCLE_FILE", registry)
    retrieval.equity_lifecycle_events.cache_clear()
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("provider must not be queried after lifecycle cutoff")

    monkeypatch.setattr(retrieval, "get_latest_ts_for_symbol", fail_if_called)
    result = retrieval.fetch_equity_prices("GONE", start_date="2024-01-03")
    assert result.empty
    assert not called
    retrieval.equity_lifecycle_events.cache_clear()


def test_lifecycle_registry_preserves_successor_identity(monkeypatch, tmp_path):
    registry = tmp_path / "events.yaml"
    registry.write_text(
        "equities:\n"
        "  OLD:\n"
        "    last_trading_date: '2024-01-02'\n"
        "    successor_symbol: CANONICAL\n"
        "    successor_ticker_at_effective_date: OLDNEW\n"
        "    conversion_ratio: 0.5\n"
    )
    monkeypatch.setattr(retrieval, "EQUITY_LIFECYCLE_FILE", registry)
    retrieval.equity_lifecycle_events.cache_clear()
    event = retrieval.equity_lifecycle_events()["OLD"]
    assert event["last_trading_date"] == pd.Timestamp("2024-01-02").date()
    assert event["successor_symbol"] == "CANONICAL"
    assert event["successor_ticker_at_effective_date"] == "OLDNEW"
    assert event["conversion_ratio"] == 0.5
    retrieval.equity_lifecycle_events.cache_clear()
