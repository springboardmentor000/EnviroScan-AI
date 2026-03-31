# scripts/feature_engineering.py

import pandas as pd
from geopy.distance import geodesic
from config import *

def feature_engineering():

    weather = pd.read_csv("data/raw/weather_data.csv")
    air = pd.read_csv("data/raw/air_quality_data.csv")
    counts = pd.read_csv("data/processed/location_counts.csv")

    df = pd.merge(air, weather, on=["latitude","longitude"])

    # Distance from Vizag center
    distances = []

    for _, row in df.iterrows():
        dist = geodesic(
            (LATITUDE, LONGITUDE),
            (row["latitude"], row["longitude"])
        ).meters
        distances.append(round(dist,2))

    df["distance"] = distances

    # Time features (fake simulation for variation)
    df["hour"] = pd.date_range(
        start="2026-03-01 10:00",
        periods=len(df),
        freq="h"
    ).strftime("%I %p")

    df["day_of_week"] = "Monday"
    df["month"] = 3
    df["season"] = "Dry"

    # OSM counts (same for region)
    df["roads_count"] = counts["roads_count"][0]
    df["industrial_count"] = counts["industrial_count"][0]
    df["farmland_count"] = counts["farmland_count"][0]
    df["dumpsite_count"] = counts["dumpsite_count"][0]

    df.to_csv("data/processed/engineered_data.csv", index=False)

    print("✔ Feature Engineering Done (20 varied rows)")