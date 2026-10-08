import os
import re
import requests
import pandas as pd
from datetime import datetime as dtm
import click
from dotenv import load_dotenv

# Load environment from .env (mirrors data_retrieval.py behavior)
load_dotenv()

# Constants
_FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"


def _fetch_alfred_observations(
    fred_series_id: str,
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> pd.DataFrame:
    """
    Pull all vintages for a FRED series via the ALFRED mechanism and return a
    DataFrame with columns: realtime_start, realtime_end, date, value

    This mirrors the behaviour in the main project but is self-contained so it
    can be used to compute true release dates.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not set in environment")

    params = {
        "series_id": fred_series_id,
        "api_key": api_key,
        "file_type": "json",
        "realtime_start": "1776-07-04",
        "realtime_end": "9999-12-31",
        "limit": 100000,
        "offset": 0,
        "sort_order": "asc",
    }
    if observation_start:
        params["observation_start"] = observation_start
    if observation_end:
        params["observation_end"] = observation_end

    obs = []
    while True:
        r = requests.get(_FRED_OBS_URL, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        batch = payload.get("observations", []) or []
        if not batch:
            break
        obs.extend(batch)

        count = int(payload.get("count", len(obs)))
        params["offset"] = int(params["offset"]) + int(params["limit"])
        if len(obs) >= count:
            break

    if not obs:
        return pd.DataFrame(columns=["realtime_start", "realtime_end", "date", "value"])

    df = pd.DataFrame(obs)
    df["realtime_start"] = pd.to_datetime(df["realtime_start"], errors="coerce").dt.date
    df["realtime_end"] = pd.to_datetime(df["realtime_end"], errors="coerce").dt.date
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    # Some values are "." (missing) in FRED JSON
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def fetch_eco_from_fred(
    fred_series_id: str,
    observation_start: str | None = None,
    observation_end: str | None = None,
    series_name: str | None = None,
    release_name: str | None = None,
) -> pd.DataFrame:
    """
    Fetch an economic series from FRED with correct first-release dates computed
    from ALFRED vintages.

    Returns a DataFrame with columns:
      fred_series_id, series_name, release_name, release_id, release_date,
      reference_date, value
    """
    series_name = series_name or fred_series_id
    release_name = release_name or series_name

    vint = _fetch_alfred_observations(
        fred_series_id=fred_series_id,
        observation_start=observation_start,
        observation_end=observation_end,
    )
    if vint.empty:
        return pd.DataFrame(
            columns=[
                "fred_series_id",
                "series_name",
                "release_name",
                "release_id",
                "release_date",
                "reference_date",
                "value",
            ]
        )

    # first release is minimum realtime_start for each observation date
    first_release = (
        vint.groupby("date", as_index=False)["realtime_start"]
        .min()
        .rename(columns={"date": "reference_date", "realtime_start": "release_date"})
    )

    # Latest value per reference_date: prefer realtime_end == 9999-12-31
    latest_end_sentinel = dtm.strptime("9999-12-31", "%Y-%m-%d").date()
    is_latest = (vint["realtime_end"] == latest_end_sentinel)
    if is_latest.any():
        latest = vint[is_latest].copy()
        latest = latest.sort_values(["date", "realtime_start"]).drop_duplicates("date", keep="last")
    else:
        latest = vint.sort_values(["date", "realtime_end", "realtime_start"]).drop_duplicates("date", keep="last")

    latest = latest[["date", "value"]].rename(columns={"date": "reference_date"})

    out = first_release.merge(latest, on="reference_date", how="left")
    out["fred_series_id"] = fred_series_id
    out["series_name"] = series_name
    out["release_name"] = release_name

    out["release_date"] = pd.to_datetime(out["release_date"])
    out["release_id"] = out["release_date"].dt.strftime("%Y%m%d").astype(int)

    out["reference_date"] = pd.to_datetime(out["reference_date"])
    out = out.dropna(subset=["release_date", "reference_date"])

    return out[
        [
            "fred_series_id",
            "series_name",
            "release_name",
            "release_id",
            "release_date",
            "reference_date",
            "value",
        ]
    ].sort_values(["reference_date"])


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^0-9A-Za-z._-]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def save_df_parquet(df: pd.DataFrame, dest: str, filename: str) -> str:
    """
    Save DataFrame to parquet. 'dest' may be a local directory or an s3 prefix
    (e.g. s3://bucket/path). Returns the full path written.
    """
    if not filename.endswith('.parquet'):
        filename = filename + '.parquet'

    # support s3:// and local paths
    if dest.startswith('s3://'):
        # let pandas/fsspec handle the filesystem; pyarrow + s3fs are required
        path = dest.rstrip('/') + '/' + filename
        df.to_parquet(path, engine='pyarrow')
    else:
        os.makedirs(dest, exist_ok=True)
        path = os.path.join(dest, filename)
        df.to_parquet(path, engine='pyarrow')
    return path


def _load_eco_series(series_file: str) -> list[tuple[str, str, str]]:
    """Load and deduplicate series definitions from a macro_series YAML file."""
    import yaml
    with open(series_file) as f:
        cfg = yaml.safe_load(f)
    seen: set[tuple[str, str]] = set()
    out = []
    for group in cfg.values():
        for s in group:
            key = (s["fred_series_id"], s["series_name"])
            if key not in seen:
                seen.add(key)
                out.append((s["fred_series_id"], s["series_name"], s["release_name"]))
    return out


@click.command()
@click.option('--dest', required=True, help='Destination folder (local path or s3://bucket/path)')
@click.option('--start', default=None, help='Observation start date (YYYY-MM-DD)')
@click.option('--end', default=None, help='Observation end date (YYYY-MM-DD)')
@click.option(
    '--series-file',
    default=os.path.join(os.path.dirname(__file__), "macro_series.yaml"),
    help='Path to macro_series YAML file.',
    show_default=True,
)
def main(dest, start, end, series_file):
    """Fetch economic series from FRED/ALFRED defined in SERIES_FILE and write parquet files to DEST."""
    eco_series = _load_eco_series(series_file)
    click.echo(f"Loaded {len(eco_series)} series from {series_file}")

    for fred_id, series_name, release_name in eco_series:
        click.echo(f"Fetching {series_name} ({fred_id}) via ALFRED...")
        df = fetch_eco_from_fred(fred_id, observation_start=start, observation_end=end, series_name=series_name, release_name=release_name)
        if df.empty:
            click.echo(f"No data for {fred_id}; skipping")
            continue
        safe = f"{fred_id}_{_safe_filename(series_name)}.parquet"
        out_path = save_df_parquet(df, dest, safe)
        click.echo(f"Wrote {len(df)} rows to {out_path}")


if __name__ == '__main__':
    main()
