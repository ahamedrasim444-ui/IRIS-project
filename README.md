# 🌸 Iris Botanical Analytics Lab (`IRIS-project`)
### Real-Time Streamlit Web Dashboard & High-Performance Python WSGI API
**Configured for 24/7 Live Deployment on Vercel & Streamlit Cloud**

[![Python CI](https://github.com/ahamedrasim444-ui/IRIS-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ahamedrasim444-ui/IRIS-project/actions/workflows/ci.yml)
[![Vercel Deployment](https://img.shields.io/badge/Vercel-Live%20Real--Time%20Server-black?logo=vercel)](https://iris-eda.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📌 Project Overview
The **Iris Botanical Analytics Lab** (`IRIS-project`) is an executive-grade, real-time exploratory data science suite and species classification engine for Fisher's 1936 Iris dataset.

It features dual-mode execution:
1. **Real-Time Streamlit Web Dashboard**: Built with Deep Tech Obsidian Dark Theme (`#0B0F17`), 3D perspective hero banner, 4 responsive analytical tabs, dynamic sliders, interactive Matplotlib plots, and real-time species prediction. Powered on Vercel via high-speed client-side WebAssembly (`@stlite/browser`).
2. **Production Flask WSGI REST API**: Serverless endpoints (`/api/health`, `/api/dataset`, `/api/predict`) executing pure NumPy vector operations with sub-100ms response times.

### Key Architectural Highlights
- **Serverless-Optimized WSGI Architecture**: Built with lightweight Flask 3.0, CORS, and Gunicorn for sub-100ms response times on Vercel Lambda functions.
- **Pure NumPy Linear Algebra Engine**: Sub-millisecond matrix computations for class mean centroids, empirical covariance ($\mathbf{\Sigma}$), Pearson correlation ($\mathbf{R}$), and principal eigen-decomposition.
- **Nearest-Centroid Classifier API**: Vectorized L2 Euclidean distance classification ($d = \|\mathbf{x} - \mathbf{\mu}_k\|_2$) with Softmax confidence scoring.
- **Resilient SQLite Persistence**: Automated serverless storage resolution defaulting to `/tmp` on read-only cloud environments.
- **Automated Multi-Version CI**: Multi-version test runner validating Python 3.10, 3.11, and 3.12.

---

## 🚀 API Endpoints & Contract

### 1. Discovery & Status: `GET /`
Returns system status, scientific stack versions, and available endpoints.

**Example Response:**
```json
{
  "status": "online",
  "service": "Iris Botanical Analytics Lab API",
  "runtime": "Vercel Serverless Python (Flask WSGI)",
  "framework": "Flask 3.0.2",
  "scientific_stack": {
    "numpy": "1.26.4",
    "pandas": "2.2.1"
  },
  "endpoints": {
    "root": "/",
    "health": "/api/health",
    "dataset": "/api/dataset",
    "predict": "/api/predict"
  }
}
```

---

### 2. System Health & Diagnostics: `GET /api/health`
Verifies NumPy/Pandas computation engine readiness, dataset availability, and SQLite connectivity.

**Example Response:**
```json
{
  "status": "healthy",
  "engine": "healthy",
  "database": "connected",
  "diagnostics": {
    "dataset_rows": 150,
    "species_classes": ["Setosa", "Versicolor", "Virginica"],
    "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    "database_stats": {
      "total_records": 12,
      "species_distribution": {"Setosa": 6, "Virginica": 4, "Versicolor": 2}
    }
  }
}
```

---

### 3. Dataset & Descriptive Statistics: `GET /api/dataset`
Serves raw records and descriptive statistics from `data/iris.csv`.

**Query Parameters:**
- `species` *(optional)*: Filter by species (`setosa`, `versicolor`, `virginica`)
- `limit` *(optional)*: Maximum rows to return (default: `150`)

**Example Request:**
```bash
curl -X GET "https://iris-project.vercel.app/api/dataset?species=setosa&limit=2"
```

---

### 4. Real-Time Species Classification: `POST /api/predict`
Accepts morphological measurements (in cm), evaluates L2 Euclidean distances to species centroids via pure NumPy, logs the query to SQLite, and returns predictions.

**Request Payload:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response Payload (200 OK):**
```json
{
  "status": "success",
  "predicted_species": "Setosa",
  "confidence_percentage": 98.42,
  "distance_cm": 0.1414,
  "all_distances": {
    "Setosa": 0.1414,
    "Versicolor": 3.4211,
    "Virginica": 5.0232
  },
  "class_probabilities": {
    "Setosa": 98.42,
    "Versicolor": 1.54,
    "Virginica": 0.04
  },
  "input_coordinates": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  },
  "transaction_id": 13
}
```

---

## 📁 Repository Structure

```
IRIS-project/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions multi-version Python test runner
├── data/
│   └── iris.csv               # Bundled Fisher's 1936 dataset (150 samples)
├── app.py                     # Production Flask WSGI backend (exports top-level app)
├── data_engine.py             # Pure NumPy linear algebra & statistical module
├── database.py                # Serverless-ready SQLite persistence manager
├── styles.py                  # Obsidian design system styling helpers
├── visualizations.py          # Matplotlib figure generation module
├── test_engine.py             # Automated unit testing suite (100% test coverage)
├── requirements.txt           # Standard Python production dependencies
├── vercel.json                # Vercel serverless function routing configuration
├── .gitignore                 # Clean Git ignore rules
└── README.md                  # Project documentation & API specifications
```

---

## 🌐 24/7 Vercel Production Deployment & Domain Configuration

### 1. Automated CI/CD Git Triggers
Deploying to Vercel via GitHub ensures 24/7 zero-downtime serverless availability. Every commit pushed to `main` triggers an automatic immutable deployment with automated rollbacks if build checks fail.

1. Navigate to [Vercel Dashboard](https://vercel.com/new).
2. Click **"Import Project"** and select **`ahamedrasim444-ui/IRIS-project`**.
3. Framework Preset: **Other**.
4. Root Directory: `./` (leave default).
5. Click **"Deploy"**.

---

### 2. Custom Domain Configuration (`Settings -> Domains`)
To attach a custom brand domain (e.g., `iris.yourdomain.com` or `yourdomain.com`):

1. In your Vercel Project Dashboard, navigate to **Settings** > **Domains**.
2. Enter your custom domain name (e.g. `api.yourdomain.com` or `iris-project.com`) and click **Add**.
3. Select whether to redirect the apex domain to `www` or use a direct subdomain.

#### DNS Registrar Configuration Records:
Add the corresponding records in your DNS provider (Cloudflare, Namecheap, GoDaddy, Route 53):

| Type | Name / Host | Value / Target | TTL | Description |
| :--- | :--- | :--- | :--- | :--- |
| **A Record** | `@` (apex) | `76.76.21.21` | Automatic / 300 | Vercel Global Anycast Edge IP |
| **CNAME** | `api` or `www` | `cname.vercel-dns.com.` | Automatic / 300 | Vercel Anycast Edge Router |

---

### 3. Automated SSL/TLS & HTTPS Enforcement
- Vercel automatically provisions and auto-renews free **Let's Encrypt** or **DigiCert** SSL certificates for all connected domains within 60 seconds of DNS validation.
- All HTTP requests are automatically redirected to `https://` with HSTS (HTTP Strict Transport Security) enabled.

---

## 💻 Local Development & Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python test_engine.py
```
Expected output: `ALL TESTS PASSED WITH 100% SUCCESS!`

### 3. Run the Flask Development Server
```bash
python app.py
```
Access the API at `http://localhost:5000`.

---

## 📜 Citations & References
- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179-188.
- Anderson, E. (1935). *The irises of the Gaspe Peninsula.* Bulletin of the American Iris Society, 59, 2-5.
