# =============================================
# update_ndx_history.py
# =============================================
"""
Usage:
  python update_ndx_history.py [--code NDX] [--asof YYYY-MM-DD]

Behavior:
- Pulls QQQ holdings via simple pandas.read_csv(URL).
- Inserts today's seed snapshot (ON CONFLICT DO NOTHING).
- Diffs vs the most recent prior snapshot:
    * removed  -> close open interval at prev_asof + 1 day
    * added    -> open  interval at prev_asof + 1 day (or today if no prev)
    * continued-> update name/sector if missing; bump last_verified_asof
Assumes:
- get_connection(params) yields (conn, cur) and commits on success.
- Cursor is RealDictCursor (access rows by name).
- UNIQUE (index_id, as_of_date, symbol) on seed_snapshots.
- UNIQUE (index_id, symbol, start_date, end_date) on index_constituents.
"""

import argparse
import logging
import re
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from db_utils import get_connection
from db_config import config as conn_params
from build_ndx_history import Source, upsert_source

# -------------------------------
# Config
# -------------------------------
QQQ_HOLDINGS_CSV = (
    "https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0"
    "?action=download&audienceType=Investor&ticker=QQQ"
)
MIN_QQQ_ROWS = 80  # sanity floor


def _choose_symbol_col(df: pd.DataFrame) -> str:
    """
    Prefer 'Holding Ticker' (or any col that includes both 'holding' and 'ticker').
    Avoid 'Fund Ticker'. Fall back to 'Ticker', 'Symbol', or the first column.
    """
    cols = [str(c) for c in df.columns]
    lower = {str(c).lower(): c for c in df.columns}

    # Exact/best
    for exact in ("Holding Ticker", "holding ticker", "holding_ticker"):
        if exact in cols:
            return exact
        if exact in lower:
            return lower[exact]

    # Contains both 'holding' and 'ticker'
    for k, orig in lower.items():
        if "holding" in k and "ticker" in k:
            return orig

    # Plain 'ticker' but avoid fund ticker
    for k, orig in lower.items():
        if "ticker" in k and "fund" not in k:
            return orig

    # 'symbol'
    for k, orig in lower.items():
        if "symbol" in k:
            return orig

    # fallback
    return df.columns[0]


def _pick_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    # exact
    for c in candidates:
        if c in lower:
            return lower[c]
    # substring
    for c in candidates:
        for k, orig in lower.items():
            if c in k:
                return orig
    return None


def fetch_qqq_triples() -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Simple fetch like your REPL:
    df = pd.read_csv(URL) -> normalize to [(symbol, name, sector)].
    """
    df = pd.read_csv(QQQ_HOLDINGS_CSV, dtype=str)

    c_sym = _choose_symbol_col(df)
    c_name = _pick_col(df, ("name", "holding name", "security name", "company", "company name")) or "Name"
    c_sect = _pick_col(df, ("sector", "gics sector", "gics"))

    # Guard: if symbol col turned out to be Fund Ticker (QQQ), try to recover
    head_syms = df[c_sym].astype(str).str.strip().str.upper().head(10).tolist()
    if head_syms and all(x == "QQQ" for x in head_syms if x):
        for col in df.columns:
            if str(col).lower() == "holding ticker":
                c_sym = col
                break

    triples: List[Tuple[str, Optional[str], Optional[str]]] = []
    for _, r in df.iterrows():
        sym = str(r.get(c_sym, "")).strip().upper()
        name = str(r.get(c_name, "")).strip() if c_name in df.columns else None
        sect = str(r.get(c_sect, "")).strip() if c_sect in df.columns else None
        if re.fullmatch(r"[A-Z\.]{1,6}", sym) and sym != "QQQ":
            triples.append((sym, name or None, sect or None))

    # de-dupe
    seen, out = set(), []
    for s, n, c in triples:
        if s not in seen:
            seen.add(s)
            out.append((s, n, c))

    logging.info("Fetched %d holdings from QQQ CSV", len(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="NDX", help="Index code (default: NDX)")
    ap.add_argument("--asof", help="As-of date YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    as_of = date.fromisoformat(args.asof) if args.asof else date.today()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    params = conn_params()

    # Resolve index_id
    with get_connection(params) as (conn, cur):
        cur.execute("SELECT index_id FROM market_indices.indices WHERE code = %(code)s", {"code": args.code})
        row = cur.fetchone()
        if not row:
            raise LookupError(f"Index code {args.code} not found")
        index_id = row["index_id"]

    # Fetch snapshot BEFORE touching DB
    try:
        triples = fetch_qqq_triples()  # [(sym, name, sector)]
    except Exception as e:
        logging.error("Aborting update: QQQ fetch/parse failed: %s", e)
        return

    if len(triples) < MIN_QQQ_ROWS:
        logging.error("Aborting update: too few QQQ rows (%d < %d).", len(triples), MIN_QQQ_ROWS)
        return

    sym_to_info: Dict[str, Tuple[Optional[str], Optional[str]]] = {s: (n, c) for s, n, c in triples}
    curr_syms = {s for s, _, _ in triples}

    # Previous snapshot date
    with get_connection(params) as (conn, cur):
        cur.execute(
            """
            SELECT MAX(as_of_date) AS prev_asof
            FROM market_indices.seed_snapshots
            WHERE index_id = %(idx)s AND as_of_date < %(asof)s
            """,
            {"idx": index_id, "asof": as_of},
        )
        prev_row = cur.fetchone()
        prev_asof = prev_row["prev_asof"] if prev_row else None

    # Insert today's seed snapshot
    with get_connection(params) as (conn, cur):
        src_id = upsert_source(cur, Source("ETF", "Invesco QQQ holdings", QQQ_HOLDINGS_CSV, as_of, None))
        inserted = 0
        for sym, name, sect in triples:
            cur.execute(
                """
                INSERT INTO market_indices.seed_snapshots
                    (index_id, as_of_date, symbol, company_name, sector, source_id)
                VALUES (%(idx)s, %(asof)s, %(sym)s, %(name)s, %(sect)s, %(src)s)
                ON CONFLICT DO NOTHING
                """,
                {"idx": index_id, "asof": as_of, "sym": sym, "name": name, "sect": sect, "src": src_id},
            )
            inserted += cur.rowcount
        logging.info("Seed snapshot upserts today: %d", inserted)

    # First snapshot on record: open rows and exit
    if not prev_asof:
        with get_connection(params) as (conn, cur):
            opened = 0
            for sym in sorted(curr_syms):
                name, sect = sym_to_info.get(sym, (None, None))
                # update if exists, else insert
                cur.execute(
                    "SELECT 1 FROM market_indices.index_constituents WHERE index_id = %(idx)s AND symbol = %(sym)s AND end_date IS NULL",
                    {"idx": index_id, "sym": sym},
                )
                exists = cur.fetchone() is not None
                if exists:
                    cur.execute(
                        """
                        UPDATE market_indices.index_constituents
                        SET company_name = COALESCE(company_name, %(name)s),
                            sector = COALESCE(sector, %(sect)s),
                            last_verified_asof = GREATEST(COALESCE(last_verified_asof, %(asof)s), %(asof)s)
                        WHERE index_id = %(idx)s AND symbol = %(sym)s AND end_date IS NULL
                        """,
                        {"idx": index_id, "sym": sym, "name": name, "sect": sect, "asof": as_of},
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO market_indices.index_constituents
                            (index_id, symbol, company_name, sector, start_date, end_date)
                        VALUES (%(idx)s, %(sym)s, %(name)s, %(sect)s, %(start)s, NULL)
                        ON CONFLICT ON CONSTRAINT uq_constituent_period DO NOTHING
                        """,
                        {"idx": index_id, "sym": sym, "name": name, "sect": sect, "start": as_of},
                    )
                    opened += cur.rowcount
            logging.info("Opened intervals (first snapshot): %d", opened)

        with get_connection(params) as (conn, cur):
            cur.execute("SELECT COUNT(*) AS n FROM market_indices.index_constituents WHERE index_id = %(i)s", {"i": index_id})
            logging.info("index_constituents rows now: %s", cur.fetchone()["n"])
        return

    # Diff vs previous snapshot
    with get_connection(params) as (conn, cur):
        cur.execute(
            "SELECT symbol FROM market_indices.seed_snapshots WHERE index_id = %(idx)s AND as_of_date = %(d)s",
            {"idx": index_id, "d": prev_asof},
        )
        prev_syms = {r["symbol"] for r in cur.fetchall()}

    # basic safety
    if prev_syms and len(curr_syms) < int(0.6 * len(prev_syms)):
        logging.error("Aborting update: size dropped >40%% (%d -> %d).", len(prev_syms), len(curr_syms))
        return

    added_syms = curr_syms - prev_syms
    removed_syms = prev_syms - curr_syms
    continued = curr_syms & prev_syms

    boundary_d = prev_asof + timedelta(days=1)
    start_d = boundary_d if boundary_d <= as_of else as_of

    logging.info(
        "Diff — added:%d removed:%d continued:%d (prev_asof=%s, boundary=%s)",
        len(added_syms), len(removed_syms), len(continued), prev_asof, boundary_d,
    )

    # Close removals
    if removed_syms:
        with get_connection(params) as (conn, cur):
            closed = 0
            for sym in sorted(removed_syms):
                cur.execute(
                    """
                    UPDATE market_indices.index_constituents
                    SET end_date = %(end)s
                    WHERE index_id = %(idx)s AND symbol = %(sym)s AND end_date IS NULL
                    """,
                    {"idx": index_id, "sym": sym, "end": boundary_d},
                )
                closed += cur.rowcount
            logging.info("Closed intervals: %d", closed)

    # Open additions
    if added_syms:
        with get_connection(params) as (conn, cur):
            opened = 0
            for sym in sorted(added_syms):
                name, sect = sym_to_info.get(sym, (None, None))
                cur.execute(
                    "SELECT 1 FROM market_indices.index_constituents WHERE index_id = %(idx)s AND symbol = %(sym)s AND end_date IS NULL",
                    {"idx": index_id, "sym": sym},
                )
                exists = cur.fetchone() is not None
                if not exists:
                    cur.execute(
                        """
                        INSERT INTO market_indices.index_constituents
                            (index_id, symbol, company_name, sector, start_date, end_date)
                        VALUES (%(idx)s, %(sym)s, %(name)s, %(sect)s, %(start)s, NULL)
                        ON CONFLICT ON CONSTRAINT uq_constituent_period DO NOTHING
                        """,
                        {"idx": index_id, "sym": sym, "name": name, "sect": sect, "start": start_d},
                    )
                    opened += cur.rowcount
            logging.info("Opened intervals: %d", opened)

    # Update continued metadata
    if continued:
        with get_connection(params) as (conn, cur):
            bumped = 0
            for sym in sorted(continued):
                name, sect = sym_to_info.get(sym, (None, None))
                cur.execute(
                    """
                    UPDATE market_indices.index_constituents
                    SET company_name = COALESCE(company_name, %(name)s),
                        sector = COALESCE(sector, %(sect)s),
                        last_verified_asof = GREATEST(COALESCE(last_verified_asof, %(asof)s), %(asof)s)
                    WHERE index_id = %(idx)s AND symbol = %(sym)s AND end_date IS NULL
                    """,
                    {"idx": index_id, "sym": sym, "name": name, "sect": sect, "asof": as_of},
                )
                bumped += cur.rowcount
            logging.info("Updated metadata for continued: %d", bumped)

    # Final count
    with get_connection(params) as (conn, cur):
        cur.execute("SELECT COUNT(*) AS n FROM market_indices.index_constituents WHERE index_id = %(i)s", {"i": index_id})
        logging.info("index_constituents rows now: %s", cur.fetchone()["n"])


if __name__ == "__main__":
    main()