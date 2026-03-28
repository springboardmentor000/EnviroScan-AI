import requests
import pandas as pd
import time
from datetime import datetime
import os

API_KEY = ""   # <-- paste your working key

cities = [
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
]

# Repeat multiple times to increase dataset
for cycle in range(10):   # increase for more data
    print(f"\n--- Cycle {cycle+1} ---")

    all_data = []

    for city in cities:
        print(f"Fetching data for {city['name']}...")

        lat = city["lat"]
        lon = city["lon"]

        # ---------------- AIR POLLUTION ----------------
        air_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        air_response = requests.get(air_url)

        if air_response.status_code != 200:
            print("Air API Error:", air_response.text)
            components = {
                "pm2_5": None,
                "pm10": None,
                "no2": None,
                "co": None,
                "so2": None,
                "o3": None
            }
        else:
            air_data = air_response.json()
            components = air_data["list"][0]["components"]

        # ---------------- WEATHER ----------------
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        weather_response = requests.get(weather_url)

        if weather_response.status_code != 200:
            print("Weather API Error:", weather_response.text)
            weather_data = {
                "main": {"temp": None, "humidity": None},
                "wind": {"speed": None, "deg": None}
            }
        else:
            weather_data = weather_response.json()

        # ---------------- RECORD ----------------
        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "City": city["name"],
            "Latitude": lat,
            "Longitude": lon,
            "PM2.5": components.get("pm2_5"),
            "PM10": components.get("pm10"),
            "NO2": components.get("no2"),
            "CO": components.get("co"),
            "SO2": components.get("so2"),
            "O3": components.get("o3"),
            "Temperature (°C)": weather_data["main"]["temp"],
            "Humidity (%)": weather_data["main"]["humidity"],
            "Wind Speed (m/s)": weather_data["wind"]["speed"],
            "Wind Direction (°)": weather_data["wind"].get("deg"),
        }

        all_data.append(record)
        time.sleep(1)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # ---------------- SAVE (APPEND FIX) ----------------
    file_exists = os.path.isfile("enviro_data_india.csv")

    df.to_csv(
        "enviro_data_india.csv",
        mode='a',
        header=not file_exists,
        index=False
    )

    print("Cycle completed and data saved.")

    time.sleep(5)  # delay between cycles

print("\nAll data collection completed!")
