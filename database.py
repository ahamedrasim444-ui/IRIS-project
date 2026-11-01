"""
database.py
Persistent SQLite Database Engine for the Iris EDA Backend.
Logs live predictions, query vectors, and decision metrics.
Serverless-ready with automated writable storage fallback (/tmp for Vercel/Lambda).
"""

import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


def get_db_path() -> str:
    """Returns a writable path for SQLite database, supporting serverless /tmp fallback."""
    if "DB_PATH" in os.environ:
        return os.environ["DB_PATH"]
    
    # In Vercel serverless environment, local filesystem is read-only except /tmp
    if os.environ.get("VERCEL"):
        return os.path.join("/tmp", "predictions.db")
    
    # Local environment: check if data/ is writable
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        test_path = os.path.join(data_dir, ".write_check")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        return os.path.join(data_dir, "predictions.db")
    except Exception:
        # Read-only filesystem fallback
        return os.path.join("/tmp", "predictions.db")


def get_connection():
    """Returns a SQLite connection with parent directory ensured."""
    db_file = get_db_path()
    parent = os.path.dirname(db_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return sqlite3.connect(db_file)


def init_db():
    """Initializes the database schema if not present."""
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


def get_recent_predictions(limit: int = 20) -> pd.DataFrame:
    """Retrieves recent logged predictions as a pandas DataFrame."""
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


def get_prediction_stats() -> Dict[str, any]:
    """Returns total count and distribution of saved predictions."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM prediction_logs")
        total = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT predicted_species, COUNT(*) 
            FROM prediction_logs 
            GROUP BY predicted_species
        """)
        distribution = dict(cursor.fetchall())
        
    return {
        "total_records": total,
        "species_distribution": distribution,
    }


def clear_database():
    """Wipes all records from the prediction_logs table."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_logs")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='prediction_logs'")
        conn.commit()
