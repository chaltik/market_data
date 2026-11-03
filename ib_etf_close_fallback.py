"""Fill missing completed ETF closes from read-only IB historical daily bars.

The scheduled repair first tries Tiingo. This script runs only if that repair
fails, and writes only exact-session rows still missing raw or adjusted closes.
"""

import logging
import math
import os
import sys
from datetime import date

import pandas as pd

from data_retrieval import (latest_completed_etf_close,
                            load_etf_close_policy, save_equity_prices_bulk,
                            stored_completed_etf_bar)


LOG = logging.getLogger(__name__)


def exact_bar(bars, session_date: date):
    """Require exactly one daily bar for the completed NYSE session."""
    matches = [bar for bar in bars if pd.Timestamp(bar.date).date() == session_date]
    if len(matches) != 1:
        raise ValueError(f"IB returned {len(matches)} bars for {session_date}")
    value = float(matches[0].close)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"IB returned an invalid {session_date} close")
    return matches[0]


def fetch_missing_close(ib, symbol, close_ts):
    from ib_async import Stock
    contracts = ib.qualifyContracts(Stock(symbol, "SMART", "USD"))
    if len(contracts) != 1:
        raise ValueError(f"IB did not uniquely qualify {symbol}")
    contract = contracts[0]
    bars = {}
    for kind in ("TRADES", "ADJUSTED_LAST"):
        bars[kind] = ib.reqHistoricalData(
            contract, endDateTime="", durationStr="5 D",
            barSizeSetting="1 day", whatToShow=kind, useRTH=True,
            formatDate=1, timeout=30)
    session_date = close_ts.tz_convert("America/New_York").date()
    raw = exact_bar(bars["TRADES"], session_date)
    adjusted = exact_bar(bars["ADJUSTED_LAST"], session_date)
    return pd.DataFrame([{
        "symbol": symbol, "ts": close_ts, "open": float(raw.open),
        "high": float(raw.high), "low": float(raw.low),
        "close": float(raw.close),
        "volume": float(raw.volume) if raw.volume is not None and raw.volume >= 0 else None,
        "adj_close": float(adjusted.close),
    }])


def main():
    from ib_async import IB
    logging.getLogger("ib_async").setLevel(logging.WARNING)
    logging.getLogger("ib_async.wrapper").setLevel(logging.WARNING)
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} REQUIRED_ETF_ASSETS.yaml")
    policy = load_etf_close_policy(sys.argv[1])
    close_ts = latest_completed_etf_close()
    missing = [s for s in policy["symbols"] if not stored_completed_etf_bar(s, close_ts)]
    LOG.warning("IB fallback for %s: %s missing required ETF closes: %s",
                close_ts, len(missing), missing)
    if not missing:
        return
    port = int(os.environ.get("IBKR_PORT", "4002"))
    if port in (4001, 7496):
        raise RuntimeError("IB fallback refuses a live trading port")
    ib = IB()
    ib.RequestTimeout = 35
    ib.TimezoneTWS = os.environ.get("IBKR_TWS_TIMEZONE", "UTC")
    try:
        ib.connect(os.environ.get("IBKR_HOST", "127.0.0.1"), port,
                   clientId=int(os.environ.get("IBKR_CLIENT_ID", "98")),
                   timeout=15, readonly=True)
        for symbol in missing:
            try:
                frame = fetch_missing_close(ib, symbol, close_ts)
                save_equity_prices_bulk(frame)
                if not stored_completed_etf_bar(symbol, close_ts):
                    raise RuntimeError("IB close did not persist")
                row = frame.iloc[0]
                LOG.info("Verified %s %s via IB historical bars: raw close=%s adjusted close=%s",
                         symbol, close_ts.date(), row["close"], row["adj_close"])
            except Exception:
                LOG.exception("IB fallback failed for %s", symbol)
    finally:
        ib.disconnect()
    remaining = [s for s in policy["symbols"] if not stored_completed_etf_bar(s, close_ts)]
    if remaining:
        raise RuntimeError(f"Required ETF bars still missing after IB fallback: {remaining}")
    LOG.info("ETF audit passed for %s: %s required closes present (Tiingo/IB)",
             close_ts, len(policy["symbols"]))


if __name__ == "__main__":
    main()
