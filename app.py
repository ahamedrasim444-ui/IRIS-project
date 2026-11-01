# -*- coding: utf-8 -*-
"""
app.py
Production WSGI Backend for Iris Botanical Analytics Lab.
Engineered for Vercel (@vercel/python) and local WSGI environments.
Serves the real-time Streamlit WebAssembly application at / and REST endpoints at /api/*.
"""

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

import data_engine
import database

# Streamlit CLI execution interceptor (`streamlit run app.py`)
try:
    import streamlit as st
    if hasattr(st, "runtime") and st.runtime.exists():
        base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
        target_script = os.path.join(base_dir, "streamlit_app.py")
        with open(target_script, "r", encoding="utf-8") as _st_f:
            _st_code = compile(_st_f.read(), target_script, "exec")
            exec(_st_code, globals())
except Exception as _st_err:
    pass

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Serverless WSGI variables
application = app
handler = app

# Database init
try:
    database.init_db()
except Exception as e:
    print(f"Database init notice: {e}")

@app.route("/", methods=["GET"])
def index():
    """
    Root endpoint:
    - Default: Serves the full-featured Iris Botanical Streamlit Web Dashboard (via Stlite).
    - If ?json=true or Accept: application/json (without text/html): Returns the JSON API status.
    """
    accept = request.headers.get("Accept", "")
    force_json = request.args.get("json", "").lower() in ("1", "true")

    if force_json or (accept == "application/json" and "text/html" not in accept):
        return jsonify({
            "status": "online",
            "message": "Welcome to the Iris Botanical Analytics Lab API",
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
                "dashboard": "/streamlit",
                "health": "/api/health",
                "dataset": "/api/dataset",
                "predict": "/api/predict"
            }
        }), 200

    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    html_file = os.path.join(base_dir, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
    return "Iris Botanical Analytics Lab initializing...", 200

@app.route("/streamlit", methods=["GET"])
@app.route("/dashboard", methods=["GET"])
def streamlit_ui():
    """Dedicated route to access the Streamlit Web UI."""
    base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    html_file = os.path.join(base_dir, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
    return "Iris Botanical Analytics Lab initializing...", 200

@app.route("/api/health", methods=["GET"])
def health():
    try:
        df = data_engine.load_dataset()
        centroids = data_engine.compute_species_centroids(df, data_engine.FEATURE_COLS)
        return jsonify({
            "status": "healthy",
            "engine": "healthy",
            "database": "connected",
            "diagnostics": {
                "dataset_rows": int(len(df)),
                "species_classes": list(centroids.keys()),
                "features": data_engine.FEATURE_COLS
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/api/dataset", methods=["GET"])
def dataset():
    try:
        df = data_engine.load_dataset()
        species_filter = request.args.get("species")
        if species_filter:
            df = df[df["species"].str.lower() == species_filter.strip().lower()]
        limit = min(request.args.get("limit", default=150, type=int), 1000)
        records = df.head(limit).to_dict(orient="records")
        return jsonify({
            "status": "success",
            "total_records": int(len(df)),
            "returned_records": len(records),
            "data": records
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({
            "message": "Send POST request with flower coordinates to classify species.",
            "example": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        }), 200
    try:
        data = request.get_json(force=True, silent=True) or {}
        sl = float(data.get("sepal_length", 0))
        sw = float(data.get("sepal_width", 0))
        pl = float(data.get("petal_length", 0))
        pw = float(data.get("petal_width", 0))
        df = data_engine.load_dataset()
        centroids = data_engine.compute_species_centroids(df, data_engine.FEATURE_COLS)
        query = np.array([sl, sw, pl, pw], dtype=float)
        res = data_engine.predict_nearest_centroid(query, centroids)
        pred_sp = res["predicted_species"]
        conf_pct = round(res["proximity_pct"][pred_sp], 2)
        dist_cm = round(res["min_distance"], 4)
        tx_id = database.log_prediction(sl, sw, pl, pw, pred_sp, conf_pct, dist_cm)
        return jsonify({
            "status": "success",
            "predicted_species": pred_sp,
            "confidence_percentage": conf_pct,
            "distance_cm": dist_cm,
            "transaction_id": tx_id
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400
