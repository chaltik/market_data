# =============================================
# build_ndx_history.py
# =============================================
"""
Usage:
  python build_ndx_history.py [--seed-source nasdaq|qqq] [--start-year 2008]

Notes:
- Requires: requests, pandas, beautifulsoup4, lxml, python-dateutil, tenacity
  pip install requests pandas beautifulsoup4 lxml python-dateutil tenacity
- Expects:
  from db_utils import get_connection        # yields (conn, cur)
  from db_config import config as conn_params  # returns connection params dict
"""

import argparse
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tenacity import retry, stop_after_attempt, wait_exponential
from io import StringIO

from db_utils import get_connection
from db_config import config as conn_params

WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
NASDAQ_COMPANIES_URL = "https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies"
QQQ_HOLDINGS_CSV = (
    "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0?action=download&audienceType=Investor&ticker=QQQ"
)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "ndx-history-builder/1.0 (+https://example.org; contact: data@local)",
})

@dataclass
class Source:
    source_type: str
    title: str
    url: str
    published_date: Optional[date]
    raw_text: Optional[str]

@dataclass
# dataclass (add sector at the end with a default)
@dataclass
class ChangeEvent:
    action: str
    company_name: Optional[str]
    symbol: Optional[str]
    effective_date: date
    effective_session: str
    announcement_date: Optional[date]
    source_url: str
    source_title: Optional[str]
    source_type: str
    notes: Optional[str] = None
    sector: Optional[str] = None     # NEW

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _get(url: str) -> requests.Response:
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    return r

def _hash_event(index_code: str, ev: ChangeEvent) -> str:
    s = f"{index_code}|{ev.action}|{ev.symbol or ''}|{ev.company_name or ''}|{ev.effective_date.isoformat()}|{ev.source_url}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def upsert_source(cur, src: Source) -> int:
    """Insert or update a source; return source_id."""
    checksum = hashlib.sha256((src.raw_text or "").encode("utf-8")).hexdigest() if src.raw_text else None
    cur.execute(
        """
        INSERT INTO market_indices.sources (source_type, title, url, published_date, raw_text, checksum)
        VALUES (%(source_type)s, %(title)s, %(url)s, %(published_date)s, %(raw_text)s, %(checksum)s)
        ON CONFLICT (url) DO UPDATE SET
            source_type    = EXCLUDED.source_type,
            title          = COALESCE(EXCLUDED.title, market_indices.sources.title),
            published_date = COALESCE(EXCLUDED.published_date, market_indices.sources.published_date),
            raw_text       = COALESCE(EXCLUDED.raw_text, market_indices.sources.raw_text),
            checksum       = COALESCE(EXCLUDED.checksum, market_indices.sources.checksum)
        RETURNING source_id
        """,
        {
            "source_type": src.source_type,
            "title": src.title,
            "url": src.url,
            "published_date": src.published_date,
            "raw_text": src.raw_text,
            "checksum": checksum,
        },
    )
    return cur.fetchone()["source_id"]

def insert_change_event(cur, index_id: int, index_code: str, ev: ChangeEvent, source_id: int) -> Optional[int]:
    """Insert a change event if new (dedup via event_hash); return change_id or None."""
    event_hash = _hash_event(index_code, ev)
    cur.execute(
        """
        INSERT INTO market_indices.index_changes_raw (
            index_id, action, company_name, symbol, sector, effective_date, effective_session,
            announcement_date, source_id, notes, event_hash
        ) VALUES (
            %(index_id)s, %(action)s, %(company_name)s, %(symbol)s, %(sector)s, %(effective_date)s, %(effective_session)s,
            %(announcement_date)s, %(source_id)s, %(notes)s, %(event_hash)s
        )
        ON CONFLICT (event_hash) DO NOTHING
        RETURNING change_id
        """,
        {
            "index_id": index_id,
            "action": ev.action,
            "company_name": ev.company_name,
            "symbol": (ev.symbol or None),
            "sector": (ev.sector or None),
            "effective_date": ev.effective_date,
            "effective_session": ev.effective_session,
            "announcement_date": ev.announcement_date,
            "source_id": source_id,
            "notes": ev.notes,
            "event_hash": event_hash,
        },
    )
    r = cur.fetchone()
    return r["change_id"] if r else None

# -----------------------------
# Parsing helpers
# -----------------------------
TICKER_RE = re.compile(r"\((?:Nasdaq|NASDAQ|NasdaqGM|NasdaqGS|NMS|NYSE)?:\s*([A-Z]{1,6})\)")
DATE_IN_TEXT_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}"
)

def parse_pr_body_for_changes(url: str, html: str) -> Tuple[List[str], List[str], Optional[date], str]:
    """Return (added_tickers, removed_tickers, effective_date, effective_session) from a PR page."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    # Effective timing
    eff_session = "UNKNOWN"
    if re.search(r"prior to (the )?market open", text, re.I):
        eff_session = "PRIOR_TO_OPEN"
    elif re.search(r"after (the )?market close|after the close", text, re.I):
        eff_session = "AFTER_CLOSE"

    # Effective date (first Month DD, YYYY we see in body)
    eff_date = None
    m = DATE_IN_TEXT_RE.search(text)
    if m:
        try:
            eff_date = dateparser.parse(m.group(0)).date()
        except Exception:
            eff_date = None

    # Added & removed lists
    added, removed = [], []

    # Bullet lists first
    for ul in soup.find_all(["ul", "ol"]):
        header_el = ul.find_previous(["h2", "h3", "p"]) or None
        header = header_el.get_text(strip=True).lower() if header_el else ""
        items = [li.get_text(" ", strip=True) for li in ul.find_all("li")]
        tickers = []
        for it in items:
            tickers.extend(TICKER_RE.findall(it))
        if tickers:
            if "added" in header or "will be added" in header:
                added.extend(tickers)
            elif "removed" in header or "will be removed" in header:
                removed.extend(tickers)

    # Fallback: sentence scan
    if not added or not removed:
        lines = [x.strip() for x in text.split(".")]
        for i, line in enumerate(lines):
            if re.search(r"will be added|to be added|added to the index", line, re.I):
                added.extend(TICKER_RE.findall(line))
                if not added and i + 1 < len(lines):
                    added.extend(TICKER_RE.findall(lines[i + 1]))
            if re.search(r"will be removed|to be removed|removed from the index", line, re.I):
                removed.extend(TICKER_RE.findall(line))
                if not removed and i + 1 < len(lines):
                    removed.extend(TICKER_RE.findall(lines[i + 1]))

    # Deduplicate preserving order
    def _dedupe(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return _dedupe(added), _dedupe(removed), eff_date, eff_session

def fetch_wikipedia_yearly_changes(start_year: int = 2008) -> List[ChangeEvent]:
    r = _get(WIKI_URL)
    tables = pd.read_html(StringIO(r.text))  # use pandas to parse all tables

    events: List[ChangeEvent] = []
    for df in tables:
        cols = [str(c).lower() for c in df.columns.tolist()]
        if (any("added" in c for c in cols)
                and any("removed" in c for c in cols)
                and any("date" in c for c in cols)):
            c_date = next(c for c in df.columns if "date" in str(c).lower())
            c_added = next(c for c in df.columns if "added" in str(c).lower())
            c_removed = next(c for c in df.columns if "removed" in str(c).lower())
            for _, row in df.iterrows():
                try:
                    d = dateparser.parse(str(row[c_date])).date()
                except Exception:
                    continue
                if d.year < start_year:
                    continue
                added = str(row[c_added]) if pd.notna(row[c_added]) else ""
                removed = str(row[c_removed]) if pd.notna(row[c_removed]) else ""
                for m in re.findall(r"([A-Z]{1,6})", added):
                    events.append(ChangeEvent(
                        action="ADD", company_name=None, symbol=m, effective_date=d,
                        effective_session="UNKNOWN", announcement_date=None,
                        source_url=WIKI_URL, source_title="Wikipedia Yearly changes", source_type="WIKIPEDIA"
                    ))
                for m in re.findall(r"([A-Z]{1,6})", removed):
                    events.append(ChangeEvent(
                        action="REMOVE", company_name=None, symbol=m, effective_date=d,
                        effective_session="UNKNOWN", announcement_date=None,
                        source_url=WIKI_URL, source_title="Wikipedia Yearly changes", source_type="WIKIPEDIA"
                    ))
    return events

def extract_pr_urls_from_wikipedia() -> List[str]:
    r = _get(WIKI_URL)
    soup = BeautifulSoup(r.text, "lxml")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = (a.get_text(strip=True) or "").lower()
        if ("annual changes" in text and "nasdaq-100" in text) or ("update: annual changes" in text):
            if href.startswith("/"):
                href = f"https://en.wikipedia.org{href}"
            urls.append(href)
    # Dedup
    out, seen = [], set()
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def fetch_current_ndx_constituents(seed_source: str = "nasdaq"):
    """
    Returns: (pairs, src_title, src_url, src_type)
    pairs is List[Tuple[symbol, company_name]]
    - seed_source='nasdaq' -> try Nasdaq, then Wikipedia, then QQQ
    - seed_source='qqq'    -> QQQ only
    """
    def _from_nasdaq():
        r = _get(NASDAQ_COMPANIES_URL)
        soup = BeautifulSoup(r.text, "lxml")
        out = []
        for t in soup.find_all("table"):
            headers = [th.get_text(strip=True).lower() for th in t.find_all("th")]
            if any("ticker" in h for h in headers) and any("company" in h for h in headers):
                for tr in t.find_all("tr"):
                    tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
                    if len(tds) >= 2:
                        sym = tds[0].strip().upper()
                        name = tds[1].strip()
                        if re.fullmatch(r"[A-Z\.]{1,6}", sym):
                            out.append((sym, name))
        logging.info(f'fetched {len(out)} data pairs from nasdaq url')
        return out

    def _from_wikipedia():
        r = _get(WIKI_URL)
        tables = pd.read_html(StringIO(r.text))  # avoids FutureWarning
        for df in tables:
            cols = [str(c).lower() for c in df.columns]
            if any("ticker" in c for c in cols) and any("company" in c for c in cols):
                c_sym  = next(c for c in df.columns if "ticker"  in str(c).lower())
                c_name = next(c for c in df.columns if "company" in str(c).lower())
                c_sector = next(c for c in df.columns if "sector" in str(c).lower())
                out = []
                for _, row in df.iterrows():
                    sym = str(row[c_sym]).strip().upper()
                    name = str(row[c_name]).strip()
                    sector = str(row[c_sector]).strip()
                    if re.fullmatch(r"[A-Z\.]{1,6}", sym):
                        out.append((sym, name, sector))
                if out:
                    logging.info(f'fetched {len(out)} data tuples from {WIKI_URL}')
                    return out
        logging.info(f'didnt find any table at {WIKI_URL}')
        return []

    def _from_qqq():
        df = pd.read_csv(QQQ_HOLDINGS_CSV)
        c_sym  = next((c for c in df.columns if str(c).lower() in ("holding ticker","ticker","ticker symbol","symbol")), None)
        c_name = next((c for c in df.columns if "name" in str(c).lower()), None)
        c_sector = next((c for c in df.columns if "sector" in str(c).lower()), None )
        out = []
        for _, row in df.iterrows():
            sym  = str(row[c_sym]).strip().upper() if c_sym else None
            name = str(row[c_name]).strip() if c_name else None
            sector = str(row[c_sector]).strip() if c_sector else None
            if sym and re.fullmatch(r"[A-Z\.]{1,6}", sym):
                out.append((sym, name, sector))
        logging.info(f'fetched {len(out)} data tuples from {QQQ_HOLDINGS_CSV}')
        return out

    try:
        data = _from_qqq()
        if data and len(data) > 0:
            return data, "Invesco QQQ holdings", QQQ_HOLDINGS_CSV, "ETF"
        logging.warning("QQQ csv file parsing didn't result in any holdings, trying Wiki next")
    except Exception as e:
        logging.warning("QQQ csv file failed (%s); falling back to Wikipedia", e)
    try:
        data = _from_wikipedia()
        if data and len(data) > 0:
            return data, "Wikipedia current NDX table", WIKI_URL, "WIKIPEDIA"
        logging.warning("Wikipedia current table empty; falling back to QQQ holdings")
    except Exception as e:
        logging.warning("Wikipedia current table failed (%s); giving up, e")
        raise



    # # default: nasdaq cascade
    # try:
    #     pairs = _from_nasdaq()
    #     if pairs:
    #         return pairs, "Nasdaq-100 Companies", NASDAQ_COMPANIES_URL, "MANUAL"
    #     logging.warning("Nasdaq companies page empty; trying Wikipedia current table")
    # except Exception as e:
    #     logging.warning("Nasdaq companies page failed (%s); trying Wikipedia", e)



from datetime import date
from typing import List, Dict, Optional, Tuple

def compute_intervals_from_events(
    seed_symbols: List[str],
    events: List[Tuple[int, str, str, date]],
    seed_start_date: Optional[date] = None,   # optional anchor (see note below)
):
    """
    events: [(change_id, action, symbol, effective_date)] sorted ASC by date.
    Returns: [(symbol, start_date, end_date, added_change_id, removed_change_id)]
    """
    # active: symbol -> (start_date, add_change_id or None if from seed)
    active: Dict[str, Tuple[Optional[date], Optional[int]]] = {}

    # Prefill from seed; anchor start at seed_start_date if provided (prevents 1900 sentinel later)
    for s in seed_symbols:
        active[s] = (seed_start_date, None)

    intervals: List[Tuple[str, Optional[date], Optional[date], Optional[int], Optional[int]]] = []

    for change_id, action, symbol, d in events:
        if not symbol:
            continue

        if action == 'ADD':
            prev = active.get(symbol)
            if prev is None:
                # normal add: start new active interval
                active[symbol] = (d, change_id)
            else:
                prev_start, prev_add_id = prev
                # If the "active" came from seed (no prior ADD) or this ADD back-dates earlier than prev_start,
                # just move/mark the start to this ADD and record add_id; DO NOT emit a closing interval.
                if prev_add_id is None or (prev_start is not None and d <= prev_start) or prev_start is None:
                    active[symbol] = (d, change_id)
                else:
                    # Duplicate ADD while already active (rare) -> ignore
                    pass

        elif action == 'REMOVE':
            prev = active.pop(symbol, None)
            if prev is not None:
                prev_start, prev_add_id = prev
                intervals.append((symbol, prev_start, d, prev_add_id, change_id))
            else:
                # Remove without a known add: keep a truncated interval with unknown start
                intervals.append((symbol, None, d, None, change_id))

    # Any still-active positions become open-ended
    for s, (start, add_id) in active.items():
        intervals.append((s, start, None, add_id, None))

    # Normalize & sort per symbol
    cleaned: Dict[str, List[Tuple[Optional[date], Optional[date], Optional[int], Optional[int]]]] = {}
    for sym, st, en, add_id, rem_id in intervals:
        cleaned.setdefault(sym, []).append((st, en, add_id, rem_id))

    out: List[Tuple[str, Optional[date], Optional[date], Optional[int], Optional[int]]] = []
    for sym, lst in cleaned.items():
        lst.sort(key=lambda x: (x[0] or date(1900,1,1), x[1] or date(9999,1,1)))
        for st, en, add_id, rem_id in lst:
            out.append((sym, st, en, add_id, rem_id))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2008)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    # Get index_id for NDX
    index_code = "NDX"
    params = conn_params()
    with get_connection(params) as (conn, cur):
        cur.execute(
            "SELECT index_id FROM market_indices.indices WHERE code = %(code)s",
            {"code": index_code},
        )
        row = cur.fetchone()
        if row is None:
            raise LookupError(f"Index code {index_code} not found")
        index_id = row["index_id"]      

        data, src_title, src_url, src_type = fetch_current_ndx_constituents()


    as_of = date.today()
    with get_connection(params) as (conn, cur):
        src_id = upsert_source(cur, Source(src_type, src_title, src_url, as_of, None))
        for sym, name, sector in data:
            cur.execute(
                """
                INSERT INTO market_indices.seed_snapshots (index_id, as_of_date, symbol, company_name, sector, source_id)
                VALUES (%(idx)s, %(as_of)s, %(sym)s, %(name)s, %(sector)s, %(src)s)
                ON CONFLICT DO NOTHING
                """,
                {"idx": index_id, "as_of": as_of, "sym": sym, "name": name, "sector": sector, "src": src_id},
            )
        # If your context manager does not auto-commit, uncomment:
        # conn.commit()

    seed_symbols = sorted({sym for sym, _, _ in data})
    logging.info("Seed snapshot size: %d symbols", len(seed_symbols))
    with get_connection(params) as (conn, cur):
        cur.execute("SELECT COUNT(*) AS n FROM market_indices.seed_snapshots")
        logging.info("seed_snapshots rows now: %s", cur.fetchone()["n"])
    # Wikipedia baseline events
    wiki_events = fetch_wikipedia_yearly_changes(start_year=args.start_year)
    logging.info("Wikipedia parsed events: %d", len(wiki_events))
    # --- INSERT WIKIPEDIA EVENTS FIRST ---
    with get_connection(params) as (conn, cur):
        wiki_src_id = upsert_source(cur, Source("WIKIPEDIA", "Nasdaq-100 Wikipedia", WIKI_URL, None, None))
        inserted = 0
        for ev in wiki_events:
            cid = insert_change_event(cur, index_id, "NDX", ev, wiki_src_id)
            if cid:
                inserted += 1
        logging.info("Inserted %d Wikipedia change events", inserted)

    # quick count so we know we actually persisted
    with get_connection(params) as (conn, cur):
        cur.execute("SELECT COUNT(*) AS n FROM market_indices.index_changes_raw WHERE index_id = %(i)s", {"i": index_id})
        logging.info("index_changes_raw rows now: %s", cur.fetchone()["n"])

    # Build intervals
    with get_connection(params) as (conn, cur):
        cur.execute(
            """
            SELECT change_id, action::text, COALESCE(symbol,'') AS symbol, effective_date
            FROM market_indices.index_changes_raw
            WHERE index_id = %(idx)s
            ORDER BY effective_date ASC, change_id ASC
            """,
            {"idx": index_id},
        )
        rows = cur.fetchall()

    ev_list = []
    for r in rows:
        if isinstance(r, dict):  # RealDictRow
            if r["symbol"]:
                ev_list.append((r["change_id"], r["action"], r["symbol"], r["effective_date"]))
        else:  # fallback: regular tuple cursor
            # order must match SELECT
            change_id, action, symbol, eff_date = r
            if symbol:
                ev_list.append((change_id, action, symbol, eff_date))
        intervals = compute_intervals_from_events(seed_symbols, ev_list)

    # Store intervals
    # build a lookup from the seed we just loaded
    seed_info = {sym: (name, sect) for sym, name, sect in data}  # symbol -> (company_name, sector)
    with get_connection(params) as (conn, cur):
        cur.execute("DELETE FROM market_indices.index_constituents WHERE index_id = %(idx)s", {"idx": index_id})
        for sym, start, end, add_id, rem_id in intervals:
            name, sect = seed_info.get(sym, (None, None))
            cur.execute(
            """
            INSERT INTO market_indices.index_constituents
                (index_id, symbol, company_name, sector, start_date, end_date, added_change_id, removed_change_id)
            VALUES (%(idx)s, %(sym)s, %(name)s, %(sect)s, %(start)s, %(end)s, %(add_id)s, %(rem_id)s)
            """,
            {"idx": index_id, "sym": sym, "name": name, "sect": sect,
             "start": start or date(1900,1,1), "end": end, "add_id": add_id, "rem_id": rem_id},
        )
        logging.info("Done. Constituents rows: %d", len(intervals))
        # conn.commit()  # if needed

if __name__ == "__main__":
    main()