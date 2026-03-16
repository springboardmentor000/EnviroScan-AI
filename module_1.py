import requests
import pandas as pd
from datetime import datetime
import osmnx as ox
import os

# ==============================
# API KEY (OpenWeatherMap)
# ==============================
API_KEY = "bb12516604fd2a9fb10e5b4a62905356"

print("Starting Module 1: Data Collection\n")

# ==============================
# USER INPUT
# ==============================
CITY = input("Enter City Name: ")
LAT = float(input("Enter Latitude: "))
LON = float(input("Enter Longitude: "))

timestamp = datetime.utcnow().isoformat()

# ==============================
# WEATHER DATA
# ==============================
print("Fetching Weather Data...")

try:
    weather_url = "https://api.openweathermap.org/data/2.5/weather"
    weather_params = {"lat": LAT, "lon": LON, "appid": API_KEY, "units": "metric"}
    weather_response = requests.get(weather_url, params=weather_params)
    weather_data = weather_response.json()

    weather_info = {
        "source": "OpenWeatherMap",
        "city": CITY,
        "latitude": LAT,
        "longitude": LON,
        "temperature_C": weather_data["main"]["temp"],
        "humidity_%": weather_data["main"]["humidity"],
        "wind_speed_mps": weather_data["wind"]["speed"],
        "wind_direction_deg": weather_data["wind"].get("deg"),
        "timestamp": timestamp
    }

    pd.DataFrame([weather_info]).to_csv(
        "weather_data.csv",
        mode="a",
        header=not os.path.exists("weather_data.csv"),
        index=False
    )

    print("✔ Weather data saved")

except Exception as e:
    print("Weather API Error:", e)

# ==============================
# AIR QUALITY (OpenWeatherMap)
# ==============================
print("Fetching Air Quality Data (OpenWeatherMap)...")

try:
    air_url = "https://api.openweathermap.org/data/2.5/air_pollution"
    air_params = {"lat": LAT, "lon": LON, "appid": API_KEY}
    air_response = requests.get(air_url, params=air_params)
    air_data = air_response.json()

    components = air_data["list"][0]["components"]

    air_info = {
        "source": "OpenWeatherMap Air Pollution",
        "city": CITY,
        "latitude": LAT,
        "longitude": LON,
        "PM2_5": components.get("pm2_5"),
        "PM10": components.get("pm10"),
        "NO2": components.get("no2"),
        "CO": components.get("co"),
        "SO2": components.get("so2"),
        "O3": components.get("o3"),
        "timestamp": timestamp
    }

    pd.DataFrame([air_info]).to_csv(
        "air_quality_data.csv",
        mode="a",
        header=not os.path.exists("air_quality_data.csv"),
        index=False
    )

    print("✔ OpenWeatherMap air quality saved")

except Exception as e:
    print("Air Pollution API Error:", e)

# ==============================
# AIR QUALITY (OpenAQ)
# ==============================
print("Fetching Air Quality Data (OpenAQ)...")

try:
    openaq_url = "https://api.openaq.org/v2/latest"
    openaq_params = {"coordinates": f"{LAT},{LON}", "radius": 10000, "limit": 1}
    openaq_response = requests.get(openaq_url, params=openaq_params)
    openaq_data = openaq_response.json()

    if "results" in openaq_data and len(openaq_data["results"]) > 0:
        measurements = openaq_data["results"][0].get("measurements", [])
        openaq_dict = {
            "source": "OpenAQ",
            "city": CITY,
            "latitude": LAT,
            "longitude": LON,
            "PM2_5": None,
            "PM10": None,
            "NO2": None,
            "CO": None,
            "SO2": None,
            "O3": None,
            "timestamp": timestamp
        }
        for m in measurements:
            param = m.get("parameter", "").upper()
            if param in openaq_dict:
                openaq_dict[param] = m.get("value")

        pd.DataFrame([openaq_dict]).to_csv(
            "openaq_air_quality.csv",
            mode="a",
            header=not os.path.exists("openaq_air_quality.csv"),
            index=False
        )
        print("✔ OpenAQ data saved")
    else:
        print("⚠ OpenAQ data not available for this location")

except Exception as e:
    print("OpenAQ API Error:", e)

# ==============================
# OSM PHYSICAL FEATURES
# ==============================
print("Extracting Physical Features (industrial, farmland, roads, waste disposal)...")

try:
    tags = {
        "landuse": ["industrial", "farmland"],
        "amenity": ["waste_disposal"],
        "highway": True
    }
    features = ox.features_from_place(f"{CITY}, India", tags)
    features_df = features.drop(columns="geometry", errors="ignore")
    features_df.to_csv(
        "physical_features.csv",
        mode="a",
        header=not os.path.exists("physical_features.csv"),
        index=False
    )
    print("✔ Physical features saved")

except Exception as e:
    print("OSM Error:", e)

# ==============================
# FINAL MERGED DATASET
# ==============================
print("Creating Final Merged Dataset...")

try:
    weather = pd.read_csv("weather_data.csv").tail(1)
    air = pd.read_csv("air_quality_data.csv").tail(1)
    final_df = pd.concat([weather.reset_index(drop=True), air.reset_index(drop=True)], axis=1)
    final_df.to_csv(
        "final_dataset.csv",
        mode="a",
        header=not os.path.exists("final_dataset.csv"),
        index=False
    )
    print("✅ Final dataset saved → final_dataset.csv")
    print("Module 1 Completed Successfully!")

except Exception as e:
    print("Final Dataset Error:", e)