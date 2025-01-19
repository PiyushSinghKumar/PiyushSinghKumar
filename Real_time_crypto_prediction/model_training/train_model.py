import sqlite3
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

DB_FILE = "features.db"
MODEL_FILE = "model_training/model_registry/models.pkl"


def load_data() -> pd.DataFrame:
    """
    Loads processed features from the SQLite database.

    Returns:
        pd.DataFrame: A DataFrame containing the features from the 'trade_features' table.
    """
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM trade_features ORDER BY timestamp ASC", conn)
    conn.close()

    df = df.drop(columns=["id", "timestamp"], errors="ignore")
    return df


def train_model() -> None:
    """
    Loads data, trains multiple models, evaluates their performance, computes a weighted average of their predictions,
    and saves the models and scaler.
    Currently using one model, just because.
    """
    df = load_data()

    feature_columns = [
        "price",
        "volume",
        "price_change",
        "sma_5",
        "sma_10",
        "volume_sum_5",
    ]
    df["target"] = df["price"].shift(-1)
    df = df.dropna()

    X = df[feature_columns]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {"linear_regression": LinearRegression()}

    trained_models = {}
    mae_scores = []

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        mae_scores.append(mae)
        trained_models[model_name] = model
        print(f"📉 {model_name} MAE: {mae:.6f}")

    mae_inverse = [1 / mae for mae in mae_scores]
    total_inverse = sum(mae_inverse)
    weights = [score / total_inverse for score in mae_inverse]

    predictions = []
    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test_scaled)
        predictions.append(y_pred)

    weighted_predictions = np.average(predictions, axis=0, weights=weights)

    mae_weighted = mean_absolute_error(y_test, weighted_predictions)
    print(f"📉 Weighted Average Model MAE: {mae_weighted:.6f}")

    joblib.dump({"models": trained_models, "scaler": scaler}, MODEL_FILE)
    print(f"✅ Models saved to {MODEL_FILE}")


if __name__ == "__main__":
    train_model()
