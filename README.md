# 🌸 Iris Botanical Analytics Lab (`IRIS-project`)
### Multidimensional Exploratory Data Science & Real-Time Morphometry Suite
**Interactive Production Web Analytics with NumPy, Pandas, Matplotlib & Streamlit**

[![Python CI](https://github.com/ahamedrasim444-ui/IRIS-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ahamedrasim444-ui/IRIS-project/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Project Overview
This project presents an interactive, high-performance web dashboard engineered in **Python 3.10+** and **Streamlit** for multi-attribute exploratory data analysis (EDA) and nearest-centroid classification of the classic 1936 Fisher Iris botanical dataset.

It demonstrates foundational data science, linear algebra, and visual analytics principles using:
- **Obsidian Dark Architecture (`#0B0F17`)**: High-contrast, executive aesthetic engineered for readability across mobile phones, tablets, and desktop displays with zero header clipping.
- **Pure NumPy Linear Algebra Engine**: Sub-millisecond matrix operations for empirical covariance ($\mathbf{\Sigma}$), Pearson correlation ($\mathbf{R}$), eigen-decomposition, and class centroids.
- **Interactive Morphometric Diagram**: Dynamic 2D botanical SVG and vector geometry where outer sepals and inner petals resize in real time according to user centimeter dimensions.
- **Resilient Multi-Tier Pipeline**: Direct integration with open data API endpoints and Scikit-learn datasets, backed by local offline CSV archives.
- **Zero Personal Credentials**: Anonymized codebase suitable for institutional submission and enterprise portfolio presentation.

---

## 📐 Mathematical Formulations

### 1. Sample Covariance Matrix ($\mathbf{\Sigma}$)
Given data matrix $\mathbf{X} \in \mathbb{R}^{n \times 4}$, the empirical mean vector is $\mathbf{\mu} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i$. The centered data matrix is:
$$\mathbf{X}_{\text{centered}} = \mathbf{X} - \mathbf{1} \mathbf{\mu}^T$$
The unbiased sample covariance matrix is evaluated via contiguous matrix multiplication:
$$\mathbf{\Sigma} = \frac{1}{n - 1} \mathbf{X}_{\text{centered}}^T \mathbf{X}_{\text{centered}}$$

### 2. Pearson Correlation Matrix ($\mathbf{R}$)
Normalized via standard deviations $\mathbf{D} = \text{diag}(\mathbf{\Sigma})^{1/2}$:
$$\mathbf{R}_{ij} = \frac{\mathbf{\Sigma}_{ij}}{\sigma_i \sigma_j} \quad \Longleftrightarrow \quad \mathbf{R} = \mathbf{D}^{-1} \mathbf{\Sigma} \mathbf{D}^{-1}$$

### 3. Principal Eigen-Decomposition & PCA
$$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$
The explained variance ratio for component $i$ is:
$$\text{Variance Explained}(\%) = \frac{\lambda_i}{\sum_{j=1}^4 \lambda_j} \times 100\%$$

### 4. Nearest-Centroid Classifier
Given a user query vector $\mathbf{x} = [SL, SW, PL, PW]^T$ and class mean centroids $\mathbf{\mu}_k$ for $k \in \{\text{Setosa}, \text{Versicolor}, \text{Virginica}\}$:
$$\hat{y} = \arg\min_{k} \|\mathbf{x} - \mathbf{\mu}_k\|_2 = \arg\min_{k} \sqrt{\sum_{j=1}^4 (x_j - \mu_{k,j})^2}$$

Softmax proximity confidence scores:
$$w_k = \frac{\exp(-d_k / T)}{\sum_j \exp(-d_j / T)} \times 100\%$$

---

## 🚀 Dashboard Architecture & Tabs

### 🎛️ Control Panel (Sidebar)
- **Data Connectivity Status**: Real-time badge indicating live Open Data API connection or local cached archive.
- **Multiselect Species Filter**: Dynamically filter any subset of `Setosa`, `Versicolor`, and `Virginica`.
- **4 Dimensional Range Sliders**: Real-time bounding sliders for Sepal Length, Sepal Width, Petal Length, and Petal Width.

### 📑 Core Dashboard Tabs
1. **Tab 1: Overview & Dataset**
   - 4 Executive KPI cards: Total Analyzed Flowers, Active Filtered Records, Taxonomic Classes, and Dataset Integrity.
   - 3 Plain-English botanical cards describing morphological markers for each species.
   - Interactive, searchable dataframe with instant CSV export.
2. **Tab 2: Visual Analytics**
   - High-contrast Matplotlib bivariate scatter plot with Fisher's 2.45 cm decision boundary line (100% deterministic separation of Setosa).
   - Sepal Length vs. Sepal Width bivariate cluster map.
   - 4-panel histogram distribution grid across all continuous features.
3. **Tab 3: Mathematical Engine**
   - Sample Mean Vector comparison table across classes.
   - $4 \times 4$ Sample Covariance Heatmap with numerical precision callouts.
   - $4 \times 4$ Pearson Correlation Diverging Heatmap with Simpson's Paradox diagnostic note.
   - Principal Component Analysis (PCA) eigenvalue and cumulative variance table.
4. **Tab 4: Live Flower Predictor**
   - 4 interactive morphometric sliders with live Euclidean distance centroid prediction.
   - Specimen archetype quick presets: *Typical Setosa*, *Typical Versicolor*, *Typical Virginica*.
   - Live prediction banner with confidence score and distance breakdown.
   - 2D query point projection scatter plot showing input vector relative to centroids.
   - Live scaling morphometric diagram illustrating physical sepals and petals in real time.
   - SQLite history logging (`data/predictions.db`).

---

## 📁 Repository Structure

```
IRIS-project/
├── .devcontainer/             # Visual Studio Code Dev Container configuration
│   └── devcontainer.json
├── .github/                   # GitHub Actions Continuous Integration
│   └── workflows/ci.yml       # Automated multi-version Python test runner
├── .streamlit/                # Streamlit server & obsidian theme configuration
│   └── config.toml
├── assets/                    # Botanical vector logo and authentic photography
│   ├── iris_logo.svg
│   ├── setosa.jpg
│   ├── versicolor.jpg
│   └── virginica.jpg
├── data/                      # Dataset repository
│   └── iris.csv               # Fisher's 1936 dataset (150 samples)
├── app.py                     # Main Streamlit dashboard application
├── streamlit_app.py           # Streamlit Community Cloud primary entry point
├── data_engine.py             # Pure NumPy linear algebra & statistical module
├── database.py                # SQLite persistence manager
├── styles.py                  # Obsidian design system & CSS injection
├── test_engine.py             # Automated unit testing suite (100% coverage)
├── visualizations.py          # Matplotlib figure generators
├── build_vercel.py            # Static WebAssembly (Stlite) bundle generator
├── index.html                 # Precompiled WebAssembly client application
├── vercel.json                # Vercel deployment routing configuration
├── requirements.txt           # Production Python dependencies
├── run_locally.bat            # Windows 1-click local launch script
├── push_to_github.bat         # Git push helper script
└── README.md                  # Project documentation
```

---

## 🌐 Deployment Instructions

### 1. Deploying to Streamlit Community Cloud (Recommended 1-Click)
1. Fork or push this repository to your GitHub account: `https://github.com/ahamedrasim444-ui/IRIS-project`.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **"New app"**.
4. Select:
   - **Repository:** `ahamedrasim444-ui/IRIS-project`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py` (or `app.py`)
5. Click **"Deploy!"**. The deployment will install dependencies from `requirements.txt` and launch with zero errors.

### 2. Deploying to Vercel (WebAssembly / Stlite)
The repository includes preconfigured WebAssembly bundling for zero-server hosting:
1. Import `ahamedrasim444-ui/IRIS-project` in [vercel.com](https://vercel.com/new).
2. Framework Preset: **Other**.
3. Output Directory: Leave root.
4. Click **Deploy**.

---

## 💻 Local Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/ahamedrasim444-ui/IRIS-project.git
cd IRIS-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite
```bash
python test_engine.py
```
Expected output: `ALL TESTS PASSED WITH 100% SUCCESS!`

### 4. Run the Web Application
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## 📜 Citations & References
- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179-188.
- Anderson, E. (1935). *The irises of the Gaspe Peninsula.* Bulletin of the American Iris Society, 59, 2-5.
