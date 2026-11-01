"""
data_engine.py
Data processing, feature engineering, and pure NumPy linear algebra engine
for the Multidimensional Iris EDA Dashboard.
"""

from typing import Dict, List, Tuple
import os
import numpy as np
import pandas as pd


FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURE_LABELS = {
    "sepal_length": "Sepal Length (cm)",
    "sepal_width": "Sepal Width (cm)",
    "petal_length": "Petal Length (cm)",
    "petal_width": "Petal Width (cm)",
}
SPECIES_COLORS = {
    "Setosa":     "#38BDF8",  # Electric Cyan
    "Versicolor": "#F59E0B",  # Warm Amber
    "Virginica":  "#A78BFA",  # Soft Lavender
}


def load_dataset(csv_path: str = "data/iris.csv") -> pd.DataFrame:
    """Loads Iris dataset, standardizes columns, and adds engineered features."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback to local file in current dir if any
        if os.path.exists("iris.csv"):
            df = pd.read_csv("iris.csv")
        else:
            raise FileNotFoundError(f"Dataset not found at {csv_path}")

    # Standardize column naming
    df.columns = [col.strip().lower().replace(" ", "_").replace("_(cm)", "") for col in df.columns]
    
    # Capitalize species names consistently
    df["species"] = df["species"].astype(str).str.capitalize()
    
    # Feature Engineering
    df["petal_ratio"] = df["petal_length"] / df["petal_width"].replace(0, np.nan)
    df["sepal_ratio"] = df["sepal_length"] / df["sepal_width"].replace(0, np.nan)
    df["petal_area_proxy"] = df["petal_length"] * df["petal_width"]
    df["sepal_area_proxy"] = df["sepal_length"] * df["sepal_width"]
    
    return df


def filter_dataset(
    df: pd.DataFrame,
    selected_species: List[str],
    ranges: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """Applies multi-criteria slicing on species and 4 dimensional ranges."""
    filtered_df = df[df["species"].isin(selected_species)].copy()
    
    for feat, (low, high) in ranges.items():
        if feat in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df[feat] >= low) & (filtered_df[feat] <= high)]
            
    return filtered_df


def compute_descriptive_stats(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Computes comprehensive summary statistics using Pandas."""
    if df.empty:
        return pd.DataFrame()
    stats = df[features].describe().T
    stats["median"] = df[features].median()
    stats["skewness"] = df[features].skew()
    stats["kurtosis"] = df[features].kurtosis()
    stats["iqr"] = df[features].quantile(0.75) - df[features].quantile(0.25)
    return stats[["count", "mean", "std", "min", "25%", "median", "75%", "max", "iqr", "skewness"]]


def compute_species_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Performs Pandas GroupBy aggregations per species."""
    if df.empty:
        return pd.DataFrame()
    agg_cols = FEATURE_COLS + ["petal_ratio", "sepal_ratio", "petal_area_proxy", "sepal_area_proxy"]
    valid_cols = [c for c in agg_cols if c in df.columns]
    
    grouped = df.groupby("species")[valid_cols].agg(["mean", "std", "median"])
    return grouped


# ==============================================================================
# NumPy Mathematical Engine (Linear Algebra, Distance, Covariance)
# ==============================================================================

def compute_covariance_matrix_manual(
    df: pd.DataFrame, features: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes sample covariance matrix using pure NumPy matrix multiplication:
    Sigma = (1 / (N - 1)) * (X - mu)^T * (X - mu)
    
    Returns:
        (manual_cov, numpy_builtin_cov)
    """
    X = df[features].to_numpy(dtype=float)
    n, d = X.shape
    if n < 2:
        return np.zeros((d, d)), np.zeros((d, d))
    
    # 1. Mean Vector mu = (1 / N) * sum(X)
    mu = np.mean(X, axis=0)
    
    # 2. Mean-centered matrix (X - mu)
    X_centered = X - mu
    
    # 3. Outer product summation via matrix transpose: (X_centered.T @ X_centered) / (n - 1)
    manual_cov = (X_centered.T @ X_centered) / (n - 1)
    
    # 4. Standard NumPy implementation for academic verification
    np_cov = np.cov(X, rowvar=False)
    
    return manual_cov, np_cov


def compute_correlation_matrix_manual(
    df: pd.DataFrame, features: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Pearson correlation matrix using pure NumPy:
    R = D^(-1/2) * Sigma * D^(-1/2), where D is diag(Sigma)
    
    Returns:
        (manual_corr, numpy_builtin_corr)
    """
    X = df[features].to_numpy(dtype=float)
    n, d = X.shape
    if n < 2:
        return np.eye(d), np.eye(d)
        
    cov, _ = compute_covariance_matrix_manual(df, features)
    variances = np.diag(cov)
    # Standard deviations vector
    std_devs = np.sqrt(variances)
    std_devs[std_devs == 0] = 1e-9  # Avoid divide by zero
    
    # R_ij = Cov_ij / (std_i * std_j)
    outer_std = np.outer(std_devs, std_devs)
    manual_corr = cov / outer_std
    # Ensure exact 1.0 on diagonal
    np.fill_diagonal(manual_corr, 1.0)
    
    np_corr = np.corrcoef(X, rowvar=False)
    return manual_corr, np_corr


def compute_species_centroids(
    df: pd.DataFrame, features: List[str] = FEATURE_COLS
) -> Dict[str, np.ndarray]:
    """
    Computes class mean centroids:
    mu_k = (1 / N_k) * sum_{x in C_k} x
    using pure NumPy.
    """
    centroids = {}
    for species_name in ["Setosa", "Versicolor", "Virginica"]:
        sub_df = df[df["species"] == species_name]
        if not sub_df.empty:
            centroids[species_name] = np.mean(sub_df[features].to_numpy(), axis=0)
        else:
            centroids[species_name] = np.zeros(len(features))
    return centroids


def compute_eigen_decomposition(cov_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Performs eigen-decomposition on the covariance matrix using np.linalg.eigh:
    Sigma * v_i = lambda_i * v_i
    
    Returns eigenvalues sorted descending and variance explained ratio.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues & eigenvectors descending
    idx = np.argsort(eigenvalues)[::-1]
    sorted_evals = eigenvalues[idx]
    sorted_evecs = eigenvectors[:, idx]
    
    total_var = np.sum(sorted_evals)
    if total_var > 0:
        variance_explained_pct = (sorted_evals / total_var) * 100.0
    else:
        variance_explained_pct = np.zeros_like(sorted_evals)
        
    cum_var_pct = np.cumsum(variance_explained_pct)
    
    return {
        "eigenvalues": sorted_evals,
        "eigenvectors": sorted_evecs,
        "variance_explained_pct": variance_explained_pct,
        "cumulative_variance_pct": cum_var_pct,
    }


def predict_nearest_centroid(
    query_vector: np.ndarray,
    centroids: Dict[str, np.ndarray],
) -> Dict[str, any]:
    """
    Predicts species for an input vector using NumPy Euclidean Distance:
    d(x, mu_k) = ||x - mu_k||_2 = sqrt(sum_j (x_j - mu_{k,j})^2)
    
    Also computes a normalized inverse distance proximity score.
    """
    distances: Dict[str, float] = {}
    
    for species, centroid in centroids.items():
        # Pure NumPy L2 Norm
        dist = float(np.linalg.norm(query_vector - centroid))
        distances[species] = dist
        
    # Best match: minimum distance
    predicted_species = min(distances, key=distances.get)
    min_dist = distances[predicted_species]
    
    # Proximity / Confidence formulation using softmax over negative distances:
    # w_k = exp(-d_k / T) / sum(exp(-d_j / T))
    # Softmax temperature tuned for feature space scale (~1.0)
    T = 0.85
    exp_neg_dist = {sp: np.exp(-d / T) for sp, d in distances.items()}
    sum_exp = sum(exp_neg_dist.values()) + 1e-12
    proximity_pct = {sp: float((val / sum_exp) * 100.0) for sp, val in exp_neg_dist.items()}
    
    return {
        "predicted_species": predicted_species,
        "min_distance": min_dist,
        "distances": distances,
        "proximity_pct": proximity_pct,
    }
