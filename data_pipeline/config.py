import os
# ---- setting file path ----
# 1. Dynamically find the data_pipeline/ directory
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Step one level up to find the main repository root
PROJECT_ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, ".."))

# 3. Define the main data directory
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 4. Define all specific file paths using the DATA_DIR
GEOJSON_PATH = os.path.join(DATA_DIR, "Asean.geojson")
OPENCELLID_PATH = os.path.join(DATA_DIR, "cell_towers.csv.gz")
OUTPUT_FILE_PATH = os.path.join(DATA_DIR, "jendela_phase2_esg_matrix.parquet")

GEE_DRIVE_FOLDER = 'ESG_Hackathon'

# Tuple of all ASEAN Mobile Country Codes
ASEAN_MCCS = (414, 452, 456, 457, 502, 510, 515, 520, 525, 528)

ASEAN_BOUNDS = {
    "Malaysia": [99.6, 0.8, 119.3, 7.5],
    "Indonesia": [95.0, -11.0, 141.0, 6.0],
    "Vietnam": [102.1, 8.5, 109.5, 23.4],
    "Philippines": [116.9, 4.6, 126.6, 19.5],
    "Thailand": [97.3, 5.6, 105.6, 20.5],
    "Singapore": [103.6, 1.15, 104.1, 1.48],
    "Cambodia": [102.3, 10.4, 107.6, 14.7],
    "Myanmar": [92.2, 9.6, 101.2, 28.5],
    "Laos": [100.1, 13.9, 107.7, 22.5],
    "Brunei": [114.2, 4.0, 115.4, 4.6] 
}

OOKLA_DATA_URL = 'https://ookla-open-data.s3.us-west-2.amazonaws.com/parquet/performance/type=mobile/year=2023/quarter=4/2023-10-01_performance_mobile_tiles.parquet'

BUFFER_RADIUS_M = 2750  # ~5.5km diameter
GEE_SCALE_M = 500       # middle point between accuracy and extraction speed
CHUNK_SIZE = 8000        # For GEE batch extraction (Keeps the HTTP payload well under the 10MB limit)