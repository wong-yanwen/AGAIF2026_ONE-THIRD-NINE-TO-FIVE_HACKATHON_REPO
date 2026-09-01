import pandas as pd
from data_pipeline.config import OPENCELLID_PATH, ASEAN_BOUNDS, ASEAN_MCC_BY_REGION
from data_pipeline.infrastructure_pipeline import process_candidate_site_clusters

def run_fast_cluster_test():
    print("📥 Loading raw OpenCelliD data...")
    df_raw = pd.read_csv(OPENCELLID_PATH, compression='gzip')
    df_raw = df_raw.rename(columns={'lon': 'longitude', 'lat': 'latitude'})

    region = "Malaysia"
    bounds = ASEAN_BOUNDS[region]
    mcc = ASEAN_MCC_BY_REGION[region]

    # Apply your exact new regional filter
    df_my = df_raw[
        (df_raw['mcc'] == mcc) &
        (df_raw['longitude'] >= bounds[0]) & (df_raw['longitude'] <= bounds[2]) &
        (df_raw['latitude'] >= bounds[1]) & (df_raw['latitude'] <= bounds[3])
    ].copy()

    print(f"✅ Extracted {len(df_my):,} raw antennas for {region}.")

    test_radii = [25, 50, 75, 100, 500]

    for eps in test_radii:
        print(f"\n" + "="*40)
        print(f"🧪 TESTING RADIUS: {eps} METERS")
        print("="*40)
        
        # Run your existing clustering function
        clustered = process_candidate_site_clusters(df_my, eps_meters=eps)
        
        total_clusters = len(clustered)
        singletons = (clustered['antenna_count'] == 1).sum()
        pct_singletons = (singletons / total_clusters) * 100
        max_antennas = clustered['antenna_count'].max()
        p99_antennas = clustered['antenna_count'].quantile(0.99)
        
        print(f"Total candidate sites: {total_clusters:,}")
        print(f"Singleton sites (1 antenna): {singletons:,} ({pct_singletons:.1f}%)")
        print(f"95th Percentile antennas per site: {clustered['antenna_count'].quantile(0.95):.0f}")
        print(f"99th Percentile antennas per site: {p99_antennas:.0f}")
        print(f"MAX antennas on one site: {max_antennas}")

if __name__ == "__main__":
    run_fast_cluster_test()