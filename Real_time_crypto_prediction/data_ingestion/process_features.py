import sqlite3
import pandas as pd
from typing import Optional

DB_FILE = "trades.db"
FEATURE_DB_FILE = "features.db"


def setup_feature_db() -> None:
    """
    Sets up the feature database by creating the trade_features table if it doesn't exist.
    """
    conn = sqlite3.connect(FEATURE_DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            timestamp REAL,
            price REAL,
            volume REAL,
            price_change REAL,
            sma_5 REAL,
            sma_10 REAL,
            volume_sum_5 REAL
        )
    """
    )
    conn.commit()
    conn.close()


def fetch_latest_trades(limit: Optional[int] = None) -> pd.DataFrame:
    """
    Fetches the latest trades from the database.

    Args:
        limit (Optional[int]): The maximum number of trades to fetch (if None, fetches all).

    Returns:
        pd.DataFrame: A DataFrame containing the fetched trades.
    """
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT pair, timestamp, price, volume FROM trades ORDER BY timestamp DESC"

    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql_query(query, conn)
    conn.close()

    df = df[::-1]
    df["price"] = df["price"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["timestamp"] = df["timestamp"].astype(float)

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes trade features such as price change, SMAs, and volume sum.

    Args:
        df (pd.DataFrame): A DataFrame containing trade data.

    Returns:
        pd.DataFrame: A DataFrame with the computed features.
    """
    initial_count = len(df)
    df["price_change"] = df["price"].pct_change() * 100
    df["sma_5"] = df["price"].rolling(window=5, min_periods=5).mean()
    df["sma_10"] = df["price"].rolling(window=10, min_periods=10).mean()
    df["volume_sum_5"] = df["volume"].rolling(window=5, min_periods=5).sum()

    df_dropped = df.dropna()
    dropped_count = initial_count - len(df_dropped)

    print(f"🚨 Dropped {dropped_count} rows due to NaN values.")

    return df_dropped.reset_index(drop=True)


def store_features(df: pd.DataFrame) -> None:
    """
    Stores the processed features in the feature database.

    Args:
        df (pd.DataFrame): A DataFrame containing the processed features.
    """
    conn = sqlite3.connect(FEATURE_DB_FILE)
    cursor = conn.cursor()

    for index, row in df.iterrows():
        try:
            cursor.execute(
                """
                INSERT INTO trade_features (pair, timestamp, price, volume, price_change, sma_5, sma_10, volume_sum_5)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    row["pair"],
                    row["timestamp"],
                    row["price"],
                    row["volume"],
                    row["price_change"],
                    row["sma_5"],
                    row["sma_10"],
                    row["volume_sum_5"],
                ),
            )

        except sqlite3.Error as e:
            print(f"Error inserting row {row['timestamp']}: {e}")

    conn.commit()
    conn.close()


def process_trade_features() -> None:
    """
    Fetches the latest trades, computes their features, and stores them in the database.
    """
    df = fetch_latest_trades()
    if not df.empty:
        df = compute_features(df)
        store_features(df)
        print(f"✅ {len(df)} Features processed and stored!")


if __name__ == "__main__":
    setup_feature_db()
    process_trade_features()
