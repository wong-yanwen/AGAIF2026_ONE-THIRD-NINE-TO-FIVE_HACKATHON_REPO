# AGAIF2026_ONE-THIRD-NINE-TO-FIVE_HACKATHON_REPO
Hackathon repository for team "One-third-Nine-to-five"

## 🚦 For Judges: Quick Start & Dashboard Access

Welcome to our project! Our solution is built with a modular architecture, separating the heavy geospatial data pipeline from the interactive user interface.

To evaluate our final deliverable and interactive map, please refer to our **Frontend User Interface**:

* 🖥️ **Live Streamlit Dashboard:** `https://hackathonmcmc.streamlit.app/`
* 📁 **Dashboard Source Code (Separate Repo):** `https://github.com/Hush840/Hackathon_v0.9`

**This repository (the one you are currently viewing) contains our Backend Data Engineering & GeoAI Pipeline.** It is responsible for ingesting Ookla/OpenCelliD data, querying Google Earth Engine, scoring sites, and generating the final `.parquet` matrices used by the dashboard.

If you wish to review our algorithm, methodology, or run the data pipeline yourself, please proceed with the technical setup below.

---

## 🛠️ Backend Environment Setup & Dependencies

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
pip install pandas geopandas duckdb shapely earthengine-api lightgbm shap scikit-learn pyarrow mapclassify folium streamlit

```

*Note: `pyarrow` is strictly required to read and write the pipeline's high-performance `.parquet` feature matrices.*

### 4. Google Earth Engine Authentication

Before running `main.py` or the `dashboard_prototyping.ipynb` notebook for the first time, you must authenticate your machine with Google Cloud.

```bash
earthengine authenticate

```

Follow the browser prompts to log in with your authorized Google Account.

### 5. Required Local Data Files (Data Engineering vs. UI Workflow)

Depending on your role in the team, you need different files in your local `data/` directory (these are excluded from version control due to file size).




[Link to Team GDrive](https://drive.google.com/drive/folders/10lmJf6Ac1BC10_Ry8uGMd_v30uAPdZY_?usp=sharing)

**For Data Engineers / Judges (Running the full ETL Pipeline):**
Download the raw baseline files into `/data`:

* `Asean.geojson`
* `cell_towers.csv.gz`

**For UI/UX & Algorithm Leads (Building Dashboards):**
You do NOT need to run `main.py`. Simply download the pre-computed national feature matrices into `/data`:

* `jendela_phase2_esg_matrix_malaysia.parquet`
* `jendela_phase2_esg_matrix_indonesia.parquet`
* *(etc...)*

---

## ⚠️ Development Notes

* **Jupyter Notebooks:** If a `.ipynb` notebook is run during testing, please ensure all cell outputs are cleared before pushing to GitHub to prevent massive diffs and repository bloating.

---

## 🤖 AI Transparency Declaration

Team One-third-Nine-to-five acknowledges the use of Gemini 1.5 Pro and Claude 3.5 Sonnet to assist with project brainstorming, code generation, debugging, and presentation design.