"""
test_engine.py
Automated test suite validating data processing, pure NumPy linear algebra,
predictive engine, Matplotlib chart generation, and Flask REST API endpoints.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_engine import (
    FEATURE_COLS,
    load_dataset,
    filter_dataset,
    compute_descriptive_stats,
    compute_species_aggregations,
    compute_covariance_matrix_manual,
    compute_correlation_matrix_manual,
    compute_species_centroids,
    compute_eigen_decomposition,
    predict_nearest_centroid,
)
from visualizations import (
    plot_covariance_heatmap,
    plot_correlation_heatmap,
    plot_bivariate_scatter,
    plot_distributions_grid,
    plot_prediction_diagnostics,
    plot_query_point_projection,
)
from styles import get_custom_css, render_hero_banner, render_easter_egg_card
from app import app


def test_data_loading():
    print("Testing data loading...")
    df = load_dataset("data/iris.csv")
    assert not df.empty, "Dataset should not be empty"
    assert len(df) == 150, f"Expected 150 rows, got {len(df)}"
    assert "species" in df.columns, "Missing 'species' column"
    assert "petal_ratio" in df.columns, "Missing engineered 'petal_ratio'"
    assert "sepal_ratio" in df.columns, "Missing engineered 'sepal_ratio'"
    assert set(df["species"].unique()) == {"Setosa", "Versicolor", "Virginica"}, "Species classes mismatch"
    print("  [OK] Data loading and feature engineering passed.")
    return df


def test_filtering(df):
    print("Testing dynamic filtering...")
    filtered = filter_dataset(
        df,
        selected_species=["Setosa", "Versicolor"],
        ranges={
            "sepal_length": (4.5, 6.0),
            "sepal_width": (2.5, 4.0),
            "petal_length": (1.0, 5.0),
            "petal_width": (0.1, 1.8),
        },
    )
    assert not filtered.empty, "Filtered df should not be empty"
    assert set(filtered["species"].unique()).issubset({"Setosa", "Versicolor"}), "Filter failed on species"
    print("  [OK] Dynamic filtering passed.")


def test_numpy_math_engine(df):
    print("Testing pure NumPy linear algebra engine...")
    manual_cov, np_cov = compute_covariance_matrix_manual(df, FEATURE_COLS)
    assert manual_cov.shape == (4, 4), f"Expected 4x4 cov matrix, got {manual_cov.shape}"
    cov_diff = np.max(np.abs(manual_cov - np_cov))
    assert cov_diff < 1e-12, f"Covariance diff too large: {cov_diff}"
    print(f"  [OK] Manual Covariance vs np.cov diff: {cov_diff:.2e} (Passed)")

    manual_corr, np_corr = compute_correlation_matrix_manual(df, FEATURE_COLS)
    assert manual_corr.shape == (4, 4), f"Expected 4x4 corr matrix, got {manual_corr.shape}"
    corr_diff = np.max(np.abs(manual_corr - np_corr))
    assert corr_diff < 1e-12, f"Correlation diff too large: {corr_diff}"
    print(f"  [OK] Manual Correlation vs np.corrcoef diff: {corr_diff:.2e} (Passed)")

    centroids = compute_species_centroids(df, FEATURE_COLS)
    assert len(centroids) == 3, "Expected 3 centroids"
    for sp in ["Setosa", "Versicolor", "Virginica"]:
        assert sp in centroids, f"Missing centroid for {sp}"
        assert centroids[sp].shape == (4,), f"Centroid shape invalid for {sp}"
    print("  [OK] Species centroids passed.")

    eigen_res = compute_eigen_decomposition(manual_cov)
    assert len(eigen_res["eigenvalues"]) == 4, "Expected 4 eigenvalues"
    assert np.isclose(np.sum(eigen_res["variance_explained_pct"]), 100.0), "Variance pct sum should be 100%"
    print(f"  [OK] Eigen-decomposition passed (PC1 explains {eigen_res['variance_explained_pct'][0]:.1f}% var).")


def test_nearest_centroid_classifier(df):
    print("Testing Nearest-Centroid Classifier...")
    centroids = compute_species_centroids(df, FEATURE_COLS)

    # Test Setosa query
    setosa_query = np.array([5.0, 3.5, 1.4, 0.2])
    res_setosa = predict_nearest_centroid(setosa_query, centroids)
    assert res_setosa["predicted_species"] == "Setosa", f"Expected Setosa, got {res_setosa['predicted_species']}"

    # Test Versicolor query
    versicolor_query = np.array([5.9, 2.8, 4.3, 1.3])
    res_versicolor = predict_nearest_centroid(versicolor_query, centroids)
    assert res_versicolor["predicted_species"] == "Versicolor", f"Expected Versicolor, got {res_versicolor['predicted_species']}"

    # Test Virginica query
    virginica_query = np.array([6.6, 3.0, 5.6, 2.0])
    res_virginica = predict_nearest_centroid(virginica_query, centroids)
    assert res_virginica["predicted_species"] == "Virginica", f"Expected Virginica, got {res_virginica['predicted_species']}"

    print("  [OK] Nearest-Centroid live predictions passed.")


def test_visualizations(df):
    print("Testing Matplotlib visualizations...")
    manual_cov, _ = compute_covariance_matrix_manual(df, FEATURE_COLS)
    fig1 = plot_covariance_heatmap(manual_cov, FEATURE_COLS)
    assert fig1 is not None
    plt.close(fig1)

    manual_corr, _ = compute_correlation_matrix_manual(df, FEATURE_COLS)
    fig2 = plot_correlation_heatmap(manual_corr, FEATURE_COLS)
    assert fig2 is not None
    plt.close(fig2)

    centroids = compute_species_centroids(df, FEATURE_COLS)
    fig3 = plot_bivariate_scatter(df, "sepal_length", "petal_length", centroids)
    assert fig3 is not None
    plt.close(fig3)

    fig4 = plot_distributions_grid(df)
    assert fig4 is not None
    plt.close(fig4)

    print("  [OK] Matplotlib figures generated successfully without errors.")


def test_database():
    print("Testing SQLite database operations...")
    import database
    database.init_db()
    row_id = database.log_prediction(5.5, 3.2, 1.3, 0.2, "Setosa", 97.4, 0.22)
    assert row_id > 0, "Failed to insert prediction row into database"
    rec_df = database.get_recent_predictions(limit=5)
    assert not rec_df.empty, "Recent predictions should not be empty"
    assert "predicted_species" in rec_df.columns
    stats = database.get_prediction_stats()
    assert stats["total_records"] >= 1
    print("  [OK] Database logging and statistics verified.")


def test_flask_api():
    print("Testing Flask REST API endpoints...")
    client = app.test_client()

    # 1. Test Root Route (/)
    res_root = client.get("/")
    assert res_root.status_code == 200, f"Expected 200 on /, got {res_root.status_code}"
    data_root = res_root.get_json()
    assert data_root["status"] == "online"
    assert "endpoints" in data_root

    # Test Browser UI Endpoint (/streamlit and Accept: text/html)
    res_ui = client.get("/streamlit")
    assert res_ui.status_code == 200
    assert "Iris Botanical Analytics Lab" in res_ui.get_data(as_text=True)

    res_html = client.get("/", headers={"Accept": "text/html"})
    assert res_html.status_code == 200
    assert "Iris Botanical Analytics Lab" in res_html.get_data(as_text=True)

    # 2. Test Health Route (/api/health)
    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Expected 200 on /api/health, got {res_health.status_code}"
    data_health = res_health.get_json()
    assert data_health["status"] == "healthy"
    assert "database" in data_health

    # 3. Test Dataset Route (/api/dataset)
    res_data = client.get("/api/dataset")
    assert res_data.status_code == 200, f"Expected 200 on /api/dataset, got {res_data.status_code}"
    data_ds = res_data.get_json()
    assert data_ds["total_records"] == 150
    assert len(data_ds["data"]) == 150
    assert "summary_statistics" in data_ds

    # Test dataset filtering
    res_data_filtered = client.get("/api/dataset?species=setosa&limit=10")
    assert res_data_filtered.status_code == 200
    data_filtered = res_data_filtered.get_json()
    assert data_filtered["returned_records"] == 10
    assert all(row["species"] == "Setosa" for row in data_filtered["data"])

    # 4. Test Predict Route GET (/api/predict)
    res_pred_get = client.get("/api/predict")
    assert res_pred_get.status_code == 200
    assert "specimen_archetypes" in res_pred_get.get_json()

    # 5. Test Predict Route POST Setosa (/api/predict)
    setosa_payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    res_pred_setosa = client.post("/api/predict", json=setosa_payload)
    assert res_pred_setosa.status_code == 200, f"Expected 200 on POST /api/predict, got {res_pred_setosa.status_code}"
    pred_data = res_pred_setosa.get_json()
    assert pred_data["predicted_species"] == "Setosa"
    assert pred_data["confidence_percentage"] > 70.0
    assert "transaction_id" in pred_data

    # 6. Test Predict Route POST Virginica (/api/predict)
    virginica_payload = {"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 5.6, "petal_width": 2.4}
    res_pred_virginica = client.post("/api/predict", json=virginica_payload)
    assert res_pred_virginica.status_code == 200
    assert res_pred_virginica.get_json()["predicted_species"] == "Virginica"

    # 7. Test Predict Error Handling (missing fields)
    bad_payload = {"sepal_length": 5.1}
    res_bad = client.post("/api/predict", json=bad_payload)
    assert res_bad.status_code == 400
    assert "Missing required coordinates" in res_bad.get_json()["message"]

    print("  [OK] Flask REST API endpoints verified successfully.")


if __name__ == "__main__":
    print("========================================")
    print("STARTING TEST SUITE FOR IRIS FLASK BACKEND")
    print("========================================")
    df = test_data_loading()
    test_filtering(df)
    test_numpy_math_engine(df)
    test_nearest_centroid_classifier(df)
    test_visualizations(df)
    test_database()
    test_flask_api()
    print("========================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("========================================")
