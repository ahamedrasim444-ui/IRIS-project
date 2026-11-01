"""
database.py
Persistent SQLite Database Engine for the Iris EDA Dashboard.
Logs live predictions, query vectors, and decision metrics.
Safe for Pyodide/WebAssembly and standard Python environments.
"""

try:
    import sqlite3
except (ImportError, ModuleNotFoundError):
    sqlite3 = None

import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

DB_PATH = os.path.join("data", "predictions.db")


def get_connection():
    """Returns a SQLite connection or None."""
    if sqlite3 is None:
        return None
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    """Initializes the database schema if not present."""
    if sqlite3 is None:
        return
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    sepal_length REAL NOT NULL,
                    sepal_width REAL NOT NULL,
                    petal_length REAL NOT NULL,
                    petal_width REAL NOT NULL,
                    predicted_species TEXT NOT NULL,
                    confidence_pct REAL NOT NULL,
                    distance_cm REAL NOT NULL
                )
            """)
            conn.commit()
    except Exception:
        pass


def log_prediction(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    predicted_species: str,
    confidence_pct: float,
    distance_cm: float,
) -> int:
    """Inserts a new prediction record into the database."""
    if sqlite3 is None:
        return 1
    try:
        init_db()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO prediction_logs 
                (timestamp, sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence_pct, distance_cm)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (now_str, sepal_length, sepal_width, petal_length, petal_width, predicted_species, confidence_pct, distance_cm))
            conn.commit()
            return cursor.lastrowid
    except Exception:
        return 1


def get_recent_predictions(limit: int = 20) -> pd.DataFrame:
    """Retrieves recent logged predictions as a pandas DataFrame."""
    if sqlite3 is None or not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        init_db()
        with get_connection() as conn:
            query = f"""
                SELECT id, timestamp, sepal_length, sepal_width, petal_length, petal_width, 
                       predicted_species, confidence_pct, distance_cm
                FROM prediction_logs
                ORDER BY id DESC
                LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
        return df
    except Exception:
        return pd.DataFrame()


def get_prediction_stats() -> Dict[str, any]:
    """Computes summary statistics of logged predictions."""
    df = get_recent_predictions(limit=1000)
    if df.empty:
        return {
            "total_records": 0,
            "total_predictions": 0,
            "species_counts": {},
            "avg_confidence": 0.0,
            "avg_distance": 0.0,
        }
    return {
        "total_records": int(len(df)),
        "total_predictions": int(len(df)),
        "species_counts": df["predicted_species"].value_counts().to_dict(),
        "avg_confidence": round(float(df["confidence_pct"].mean()), 2),
        "avg_distance": round(float(df["distance_cm"].mean()), 4),
    }


def clear_logs() -> bool:
    """Deletes all records from the prediction_logs table."""
    if sqlite3 is None or not os.path.exists(DB_PATH):
        return True
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM prediction_logs")
            conn.commit()
        return True
    except Exception:
        return False

# Backward compatibility alias
clear_prediction_history = clear_logs

