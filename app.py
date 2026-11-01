# -*- coding: utf-8 -*-
"""
app.py
Production-Ready Flask Backend for Iris Botanical Analytics Lab.
Engineered for serverless execution on Vercel (@vercel/python) and local WSGI environments.
Exposes top-level 'app', 'application', and 'handler' variables.
"""

import os
import sys
from typing import Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

from data_engine import (
    FEATURE_COLS,
    FEATURE_LABELS,
    SPECIES_COLORS,
    load_dataset,
    compute_descriptive_stats,
    compute_species_centroids,
    compute_covariance_matrix_manual,
    compute_correlation_matrix_manual,
    compute_eigen_decomposition,
    predict_nearest_centroid,
)
import database

# ------------------------------------------------------------------------------
# 1. FLASK APPLICATION INITIALIZATION & CONFIGURATION
# ------------------------------------------------------------------------------
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing for global web accessibility
CORS(app, resources={r"/*": {"origins": "*"}})

# Serverless WSGI Aliases: Vercel's Python runtime inspects for app, application, or handler
application = app
handler = app

# Initialize database schema if not present
try:
    database.init_db()
except Exception as _init_err:
    print(f"Database initialization notice: {_init_err}")


# ------------------------------------------------------------------------------
# 2. SYSTEM STATUS & DISCOVERY ROOT ROUTE (DUAL BROWSER UI & REST API)
# ------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """
    Root endpoint:
    - If accessed from a web browser (Accept: text/html), serves the real-time Streamlit WebAssembly UI.
    - If accessed programmatically / via API client, returns the JSON status payload.
    """
    accept = request.headers.get("Accept", "")
    force_json = request.args.get("json", "").lower() in ("1", "true")

    if "text/html" in accept and not force_json:
        base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
        html_file = os.path.join(base_dir, "index.html")
        if os.path.exists(html_file):
            with open(html_file, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}

    return jsonify({
        "message": "Welcome to the Iris Botanical Analytics Lab API",
        "status": "online",
        "api_status": "healthy",
        "service": "Iris Botanical Analytics Lab API",
        "runtime": "Vercel Serverless Python (Flask WSGI)",
        "framework": "Flask 3.0.2",
        "scientific_stack": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "endpoints": {
            "root": "/",
            "interactive_ui": "/streamlit",
            "health": "/api/health",
            "dataset": "/api/dataset",
            "predict": "/api/predict"
        },
        "description": "Production Flask backend providing dataset analytics, pure NumPy nearest-centroid species classification, and embedded Streamlit real-time dashboard."
    }), 200


@app.route("/streamlit", methods=["GET"])
@app.route("/dashboard", methods=["GET"])
@app.route("/ui", methods=["GET"])
def streamlit_portal():
    """Direct URL endpoint guaranteed to return the real-time Streamlit Web UI."""
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    html_file = os.path.join(base_dir, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
    return "Streamlit Dashboard initializing...", 200


@app.route("/streamlit_app.py", methods=["GET"])
def serve_streamlit_app_file():
    """Serves the Python Streamlit script for static CDN and Stlite runtime fetch."""
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    py_file = os.path.join(base_dir, "streamlit_app.py")
    if os.path.exists(py_file):
        with open(py_file, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/plain; charset=utf-8"}
    return "File not found", 404


@app.route("/data/iris.csv", methods=["GET"])
def serve_iris_csv_file():
    """Serves the Iris dataset CSV for static CDN and Stlite runtime fetch."""
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    csv_file = os.path.join(base_dir, "data", "iris.csv")
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/csv; charset=utf-8"}
    return "File not found", 404


# ------------------------------------------------------------------------------
# 3. SYSTEM HEALTH & DIAGNOSTIC ROUTE
# ------------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    """
    Verifies NumPy/Pandas computation engine availability and SQLite database connectivity.
    """
    engine_status = "healthy"
    database_status = "connected"
    details = {}

    # 1. Test Data Engine & Centroids
    try:
        df = load_dataset()
        centroids = compute_species_centroids(df, FEATURE_COLS)
        details["dataset_rows"] = int(len(df))
        details["species_classes"] = list(centroids.keys())
        details["features"] = FEATURE_COLS
    except Exception as e:
        engine_status = f"unhealthy: {str(e)}"

    # 2. Test Database Connectivity
    try:
        db_stats = database.get_prediction_stats()
        details["database_stats"] = db_stats
        details["database_path"] = database.get_db_path()
    except Exception as e:
        database_status = f"degraded: {str(e)}"

    overall_status = "healthy" if engine_status == "healthy" else "unhealthy"
    status_code = 200 if overall_status == "healthy" else 500

    return jsonify({
        "status": overall_status,
        "engine": engine_status,
        "database": database_status,
        "diagnostics": details
    }), status_code


# ------------------------------------------------------------------------------
# 4. DATASET & DESCRIPTIVE STATISTICS ROUTE
# ------------------------------------------------------------------------------
@app.route("/api/dataset", methods=["GET"])
def dataset():
    """
    Serves dataset records and descriptive summary statistics from data/iris.csv.
    Optional query parameters:
      - species: Filter by species name ('setosa', 'versicolor', 'virginica')
      - limit: Maximum number of rows to return (default: 150)
    """
    try:
        df = load_dataset()

        # Optional species filter
        species_filter = request.args.get("species")
        if species_filter:
            filtered_df = df[df["species"].str.lower() == species_filter.strip().lower()].copy()
        else:
            filtered_df = df.copy()

        # Optional row limit
        limit_arg = request.args.get("limit", default=150, type=int)
        limit = max(1, min(limit_arg, 1000)) if limit_arg else 150

        # Compute summary statistics
        stats_df = compute_descriptive_stats(df, FEATURE_COLS)
        summary_stats = stats_df.to_dict(orient="index")

        # Species distribution
        species_counts = df["species"].value_counts().to_dict()

        # Format records for JSON response
        records = filtered_df.head(limit).to_dict(orient="records")

        return jsonify({
            "status": "success",
            "total_records": int(len(df)),
            "filtered_records": int(len(filtered_df)),
            "returned_records": int(len(records)),
            "species_distribution": species_counts,
            "summary_statistics": summary_stats,
            "data": records
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Failed to fetch dataset",
            "details": str(e)
        }), 500


# ------------------------------------------------------------------------------
# 5. NEAREST-CENTROID CLASSIFIER & PREDICTION ROUTE
# ------------------------------------------------------------------------------
@app.route("/api/predict", methods=["GET", "POST"])
def predict():
    """
    Accepts morphological JSON coordinates:
      {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    Computes L2 Euclidean centroid distances using pure NumPy, logs the transaction
    to SQLite, and returns the predicted species, confidence scores, and metrics.
    """
    # Informative GET response for browser exploration & API contract documentation
    if request.method == "GET":
        return jsonify({
            "message": "Send a POST request with JSON coordinates to classify an Iris flower specimen.",
            "required_fields": {
                "sepal_length": "float (cm)",
                "sepal_width": "float (cm)",
                "petal_length": "float (cm)",
                "petal_width": "float (cm)"
            },
            "example_payload": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            "specimen_archetypes": {
                "Typical Setosa": {"sepal_length": 5.0, "sepal_width": 3.5, "petal_length": 1.5, "petal_width": 0.2},
                "Typical Versicolor": {"sepal_length": 5.9, "sepal_width": 2.8, "petal_length": 4.3, "petal_width": 1.3},
                "Typical Virginica": {"sepal_length": 6.6, "sepal_width": 3.0, "petal_length": 5.6, "petal_width": 2.0}
            }
        }), 200

    try:
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({
                "status": "error",
                "message": "Invalid request body: Expected application/json object"
            }), 400

        # Validate required coordinate keys
        missing_keys = [k for k in FEATURE_COLS if k not in data]
        if missing_keys:
            return jsonify({
                "status": "error",
                "message": f"Missing required coordinates: {', '.join(missing_keys)}"
            }), 400

        # Parse & validate floating-point numbers
        try:
            sl = float(data["sepal_length"])
            sw = float(data["sepal_width"])
            pl = float(data["petal_length"])
            pw = float(data["petal_width"])
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "All coordinates (sepal_length, sepal_width, petal_length, petal_width) must be valid numbers"
            }), 400

        # Biological plausibility validation
        if not (0.0 < sl <= 20.0 and 0.0 < sw <= 20.0 and 0.0 < pl <= 20.0 and 0.0 < pw <= 20.0):
            return jsonify({
                "status": "error",
                "message": "Coordinates out of plausible botanical range (values must be positive and <= 20.0 cm)"
            }), 400

        # Load dataset & compute centroids via NumPy
        df = load_dataset()
        centroids = compute_species_centroids(df, FEATURE_COLS)
        query_vector = np.array([sl, sw, pl, pw], dtype=float)

        # Execute pure NumPy L2 nearest-centroid classification
        res = predict_nearest_centroid(query_vector, centroids)
        predicted_species = res["predicted_species"]
        confidence_pct = round(res["proximity_pct"][predicted_species], 2)
        min_distance = round(res["min_distance"], 4)

        # Log transaction to SQLite (with serverless safe fallback)
        tx_id = None
        try:
            tx_id = database.log_prediction(
                sepal_length=sl,
                sepal_width=sw,
                petal_length=pl,
                petal_width=pw,
                predicted_species=predicted_species,
                confidence_pct=confidence_pct,
                distance_cm=min_distance
            )
        except Exception as db_err:
            print(f"Serverless database logging warning: {db_err}")

        return jsonify({
            "status": "success",
            "predicted_species": predicted_species,
            "confidence_percentage": confidence_pct,
            "softmax_confidence_percentage": confidence_pct,
            "distance_cm": min_distance,
            "all_distances": {k: round(v, 4) for k, v in res["distances"].items()},
            "class_probabilities": {k: round(v, 2) for k, v in res["proximity_pct"].items()},
            "input_coordinates": {
                "sepal_length": sl,
                "sepal_width": sw,
                "petal_length": pl,
                "petal_width": pw
            },
            "transaction_id": tx_id
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Prediction failed",
            "details": str(e)
        }), 500


# ------------------------------------------------------------------------------
# 6. LOCAL DEVELOPMENT ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
