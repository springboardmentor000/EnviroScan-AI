import pandas as pd


def feature_engineering():
    print("🔧 Feature Engineering...")

    air = pd.read_csv("data/raw/air_quality_data.csv")
    weather = pd.read_csv("data/raw/weather_data.csv")

    air["timestamp"] = pd.to_datetime(air["timestamp"])
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    merged = pd.merge_asof(
        air.sort_values("timestamp"),
        weather.sort_values("timestamp"),
        on="timestamp",
        by="location",
        direction="nearest"
    )

    merged["hour"] = merged["timestamp"].dt.hour
    merged["day"] = merged["timestamp"].dt.day

    merged.columns = [
        col.strip().lower().replace(".", "_").replace(" ", "_")
        for col in merged.columns
    ]

    merged.to_csv("data/processed/final_dataset.csv", index=False)

    print("✔ Final dataset created → final_dataset.csv")