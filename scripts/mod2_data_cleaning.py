import pandas as pd

def clean_weather_data():
    print(" Cleaning Weather Data...")

    df = pd.read_csv("data/raw/weather_data.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df.ffill(inplace=True)

    df.to_csv("data/processed/weather_cleaned.csv", index=False)

    print("✔ Weather cleaned")

def clean_air_quality_data():
    print(" Cleaning Air Quality Data...")

    df = pd.read_csv("data/raw/air_quality_data.csv")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df.ffill(inplace=True)

    df.to_csv("data/processed/air_quality_cleaned.csv", index=False)

    print("✔ Air quality cleaned")

def clean_distance_data():
    print(" Cleaning Distance Data...")

    df = pd.read_csv("data/processed/distance_cleaned.csv")

    df.fillna(0, inplace=True)

    df.to_csv("data/processed/distance_cleaned.csv", index=False)

    print(" Distance cleaned")