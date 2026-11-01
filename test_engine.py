"""
test_engine.py
Automated test suite validating data processing, pure NumPy linear algebra,
predictive engine, and Matplotlib chart generation.
"""

import os
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

    # Test Virginica query
    virginica_query = np.array([6.7, 3.1, 5.6, 2.1])
    res_virg = predict_nearest_centroid(virginica_query, centroids)
    assert res_virg["predicted_species"] == "Virginica", f"Expected Virginica, got {res_virg['predicted_species']}"

    print("  [OK] Nearest-Centroid live predictions passed.")


def test_visualizations(df):
    print("Testing Matplotlib visualizations...")
    cov, _ = compute_covariance_matrix_manual(df, FEATURE_COLS)
    corr, _ = compute_correlation_matrix_manual(df, FEATURE_COLS)
    centroids = compute_species_centroids(df, FEATURE_COLS)

    fig1 = plot_covariance_heatmap(cov, FEATURE_COLS)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    fig2 = plot_correlation_heatmap(corr, FEATURE_COLS)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)

    fig3 = plot_bivariate_scatter(df, "petal_length", "petal_width", centroids)
    assert isinstance(fig3, plt.Figure)
    plt.close(fig3)

    fig4 = plot_distributions_grid(df)
    assert isinstance(fig4, plt.Figure)
    plt.close(fig4)

    test_q = np.array([5.8, 3.0, 4.2, 1.3])
    pred_res = predict_nearest_centroid(test_q, centroids)
    
    fig5 = plot_prediction_diagnostics(pred_res["distances"], pred_res["proximity_pct"], pred_res["predicted_species"])
    assert isinstance(fig5, plt.Figure)
    plt.close(fig5)

    fig6 = plot_query_point_projection(df, test_q, centroids, pred_res["predicted_species"])
    assert isinstance(fig6, plt.Figure)
    plt.close(fig6)

    from visualizations import plot_boxplot_dispersion, plot_interactive_iris_flower
    fig7 = plot_boxplot_dispersion(df, "petal_length")
    assert isinstance(fig7, plt.Figure)
    plt.close(fig7)

    fig8 = plot_interactive_iris_flower(5.8, 3.0, 4.2, 1.3, "Versicolor")
    assert isinstance(fig8, plt.Figure)
    assert len(fig8.axes) == 2
    plt.close(fig8)

    print("  [OK] All 8 Matplotlib figures generated successfully without errors.")


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


def test_styles():
    print("Testing styles and Easter egg assets...")
    css_default = get_custom_css(False)
    assert "font-family" in css_default
    css_active = get_custom_css(True)
    assert "floatEffect" in css_active
    
    banner = render_hero_banner()
    assert "Iris Flower Data Analysis" in banner
    
    easter_egg = render_easter_egg_card()
    assert "import antigravity" in easter_egg
    print("  [OK] Styles and banners verified.")


if __name__ == "__main__":
    print("========================================")
    print("STARTING TEST SUITE FOR IRIS EDA DASHBOARD")
    print("========================================")
    df = test_data_loading()
    test_filtering(df)
    test_numpy_math_engine(df)
    test_nearest_centroid_classifier(df)
    test_visualizations(df)
    test_database()
    test_styles()
    print("========================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("========================================")
