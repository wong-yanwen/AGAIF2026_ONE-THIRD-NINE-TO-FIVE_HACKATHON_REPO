# AGAIF2026_ONE-THIRD-NINE-TO-FIVE_HACKATHON_REPO
Hackathon repository for team "One-third-Nine-to-five"

## 🛠️ Environment Setup & Dependencies

This project runs a modular Python geospatial pipeline. To ensure spatial calculation accuracy and prevent cross-platform dependency conflicts, please set up your environment using the following steps:

### 1. Core Prerequisites
* **Python Version:** 3.10 to 3.12 recommended.
* **Google Earth Engine Account:** Ensure your Google account is registered for an [Earth Engine Developer License](https://code.earthengine.google.com/).

### 2. Virtual Environment Installation

We highly recommend using a fresh virtual environment. Open your terminal in the root project directory:

```bash
# Create the environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Upgrade package manager
pip install --upgrade pip

```

### 3. Install Package Requirements
Our stack relies heavily on compiled C-libraries for fast spatial queries. Install the pinned dependencies:

```bash
pip install pandas geopandas duckdb shapely earthengine-api lightgbm shap scikit-learn pyarrow  mapclassify
```
Note: pyarrow is strictly required to read and write the pipeline's high-performance .parquet feature matrices.

### 4. Google Earth Engine Authentication
Before running `main.py` or the `dashboard_prototyping.ipynb` notebook for the first time, you must authenticate your machine with Google Cloud. 
```bash
earthengine authenticate
```
Follow the browser prompts to log in with your authorized Google Account.

### 5. Required Local Raw Data Files
Ensure the following baseline files are placed inside the `data/` directory (these are excluded from version control due to file size):
<br>[link to gdrive](https://drive.google.com/drive/folders/10lmJf6Ac1BC10_Ry8uGMd_v30uAPdZY_?usp=sharing)

* data/Asean.geojson 
* data/cell_towers.csv.gz
  