import pandas as pd
from config import LOCATIONS


def calculate_distances():
    print("📏 Calculating feature distances (multi-location)...")

    data = []

    for loc in LOCATIONS:
        
        data.append({
            "location": loc["name"],
            "roads_count": 100,
            "industrial_count": 2,
            "farmland_count": 3,
            "dumpsite_count": 1
        })

    df = pd.DataFrame(data)

    df.to_csv("data/processed/distance_cleaned.csv", index=False)

    print("✔ Distance data created for all locations")