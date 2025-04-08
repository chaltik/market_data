import os
import click
import pandas as pd
import numpy as np
from db_utils import get_connection
from db_config import config
from data_retrieval import save_treasury_cmt

@click.command()
@click.option('--ust_file', required=True, help="Path to the historical UST CSV file.")
@click.option('--vix_file', required=True, help="Path to the historical VIX CSV file.")
def seed_data(ust_file, vix_file):
    """Seeds historical data for Treasury CMT and VIX into the database."""

    print("📈 Loading UST CMT data...")
    ust_df = pd.read_csv(ust_file)
    ust_df = ust_df.rename(columns={
        'Unnamed: 0': 'Date',
        'UST 1m': '1 Mo', 'UST 2m': '2 Mo', 'UST 3m': '3 Mo', 'UST 4m': '4 Mo',
        'UST 6m': '6 Mo', 'UST 1y': '1 Yr', 'UST 2y': '2 Yr', 'UST 3y': '3 Yr',
        'UST 5y': '5 Yr', 'UST 7y': '7 Yr', 'UST 10y': '10 Yr',
        'UST 20y': '20 Yr', 'UST 30y': '30 Yr'
    })
    ust_df["Date"] = pd.to_datetime(ust_df["Date"], errors="coerce")
    ust_df = ust_df.dropna(subset=["Date"])  # Drop rows with no valid date
    rate_columns = [col for col in ust_df.columns if col != "Date"]
    ust_df = ust_df.dropna(subset=rate_columns, how='all')
    print(f"✅ Loaded {len(ust_df)} valid UST rows")

    print("🌪️ Loading VIX data...")
    vix_df = pd.read_csv(vix_file)
    vix_df = vix_df.rename(columns={"observation_date": "date", "VIXCLS": "value"})
    vix_df["date"] = pd.to_datetime(vix_df["date"], errors="coerce")
    vix_df = vix_df.dropna(subset=["date", "value"])
    print(f"✅ Loaded {len(vix_df)} valid VIX rows")

    print("📤 Saving to database...")
    with get_connection(config()) as (conn, cur):
        for _, row in vix_df.iterrows():
            cur.execute("""
                INSERT INTO market_indices.vix (date, value)
                VALUES (%s, %s)
                ON CONFLICT (date) DO NOTHING;
            """, (row["date"], row["value"]))

        conn.commit()

    save_treasury_cmt(ust_df)
    print("🎉 Done seeding interest rate and VIX data.")


if __name__ == "__main__":
    seed_data()
