import asyncio
import websockets
import json
import sqlite3
import pandas as pd
import joblib
import argparse
from typing import Optional
import numpy as np

KRAKEN_WS_URL = "wss://ws.kraken.com"
TRADES_DB_FILE = "trades.db"
FEATURES_DB_FILE = "features.db"
MODEL_FILE = "model_training/model_registry/models.pkl"


def create_trades_table(cursor: sqlite3.Cursor) -> None:
    """
    Creates the trades table if it doesn't exist.
    """
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            price REAL,
            volume REAL,
            timestamp REAL,
            side TEXT,
            order_type TEXT
        )
                   """
    )


def store_trade(
    pair: str, price: float, volume: float, timestamp: float, side: str, order_type: str
) -> None:
    """
    Stores a new trade in the database.
    """
    conn = sqlite3.connect(TRADES_DB_FILE)
    cursor = conn.cursor()
    create_trades_table(cursor)
    cursor.execute(
        "INSERT INTO trades (pair, price, volume, timestamp, side, order_type) VALUES (?, ?, ?, ?, ?, ?)",
        (pair, price, volume, timestamp, side, order_type),
    )
    conn.commit()
    conn.close()


def predict_price() -> Optional[float]:
    """
    Predicts the next price based on recent trade features using multiple models.

    Returns:
        float or None: The predicted price if prediction is successful, otherwise None.
    """
    try:
        conn = sqlite3.connect(FEATURES_DB_FILE)
        df = pd.read_sql_query(
            "SELECT * FROM trade_features ORDER BY timestamp DESC LIMIT 10", conn
        )
        conn.close()

        if len(df) < 5:
            return None

        feature_columns = [
            "price",
            "volume",
            "price_change",
            "sma_5",
            "sma_10",
            "volume_sum_5",
        ]
        X = df[feature_columns].iloc[::-1]
        X_scaled = scaler.transform(X)

        predictions = []
        for model_name, model in models.items():
            prediction = model.predict([X_scaled[0]])[0]
            predictions.append(prediction)

        average_prediction = np.mean(predictions)
        return average_prediction

    except Exception as e:
        print(f"⚠️ Prediction Error: {e}")
        return None


async def subscribe_trades(pair: str, infer: bool) -> None:
    """
    Subscribes to Kraken WebSocket for real-time trades and stores them in the database.
    Optionally performs price prediction with multiple models.

    Args:
        pair (str): The trading pair (e.g., BTC/USD).
        infer (bool): Whether to perform inference and predict prices.
    """
    async with websockets.connect(KRAKEN_WS_URL) as ws:
        subscribe_msg = {
            "event": "subscribe",
            "pair": [pair],
            "subscription": {"name": "trade"},
        }
        await ws.send(json.dumps(subscribe_msg))
        print(f"✅ Subscribed to {pair} trades")

        while True:
            response = await ws.recv()
            data = json.loads(response)

            if isinstance(data, list) and len(data) >= 4:
                trade_info = data[1]
                if isinstance(trade_info, list):
                    for trade in trade_info:
                        if isinstance(trade, list) and len(trade) >= 5:
                            price = float(trade[0])
                            volume = float(trade[1])
                            timestamp = float(trade[2])
                            side = trade[3]
                            order_type = trade[4]

                            store_trade(
                                pair, price, volume, timestamp, side, order_type
                            )

                            if infer:
                                prediction = predict_price()
                                if prediction:
                                    print(
                                        f"💰 Trade - Price: {price}, Predicted Next Price: {prediction:.5f}"
                                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch real-time trades and optionally predict prices."
    )
    parser.add_argument(
        "--infer", action="store_true", help="Enable real-time price prediction."
    )
    args = parser.parse_args()

    if args.infer:
        print("🔮 Running in **INFERENCE MODE** (Fetching trades & predicting prices)")
        model_data = joblib.load(MODEL_FILE)
        models = model_data["models"]
        scaler = model_data["scaler"]
    else:
        print("📥 Running in **TRAINING MODE** (Only fetching & storing trades)")

    pair = "BTC/USD"
    asyncio.run(subscribe_trades(pair, args.infer))
