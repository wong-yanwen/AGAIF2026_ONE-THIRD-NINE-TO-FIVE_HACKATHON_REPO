import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_008.8

def process_vector_proximity(df_underserved, site_nodes_final, region):
    print("⏳ Finding nearest candidate tower using Haversine BallTree...")

    tiles = df_underserved.reset_index(drop=True).copy()
    sites = site_nodes_final.reset_index(drop=True).copy()

    # Convert coordinates to radians for the spherical tree
    tile_coords = np.radians(np.column_stack([tiles.geometry.y, tiles.geometry.x]))
    site_coords = np.radians(np.column_stack([sites.geometry.y, sites.geometry.x]))

    # Build spatial index
    tree = BallTree(site_coords, metric="haversine")

    # Find the nearest 1 tower for every tile
    distances, indices = tree.query(tile_coords, k=1)
    nearest_idx = indices[:, 0]

    # Convert radians back to meters
    tiles["distance_to_nearest_tower"] = distances[:, 0] * EARTH_RADIUS_M

    # Pull the matching tower attributes 
    nearest_sites = sites.iloc[nearest_idx].reset_index(drop=True)

    transfer_cols = [
        "distance_to_power_m", "distance_to_road_m", "distance_to_amenity_m", 
        "antenna_count", "antennas_4G", 
        "antennas_3G", "antennas_2G", "antennas_5G"
    ]

    for col in transfer_cols:
        if col in nearest_sites.columns:
            tiles[col] = nearest_sites[col].to_numpy()

    return gpd.GeoDataFrame(tiles, geometry="geometry", crs=df_underserved.crs)