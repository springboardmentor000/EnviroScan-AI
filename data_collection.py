import requests
import pandas as pd
from datetime import datetime
from geopy.distance import geodesic
import osmnx as ox
import os

# =========================================
#  PUT YOUR API KEY HERE
# =========================================
API_KEY = "0905378d18d6b90dcd9e1054ed0c2586"

# =========================================
#  CITIES
# =========================================
cities = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Bangalore": (12.9716, 77.5946)
}

# =========================================
#  POLLUTION DATA
# =========================================
def get_pollution(lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        res = requests.get(url).json()

        if 'list' not in res:
            return {"pm2_5": 0, "pm10": 0, "no2": 0, "co": 0, "so2": 0, "o3": 0}

        return res['list'][0]['components']

    except:
        return {"pm2_5": 0, "pm10": 0, "no2": 0, "co": 0, "so2": 0, "o3": 0}


# =========================================
# WEATHER DATA
# =========================================
def get_weather(lat, lon):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        res = requests.get(url).json()

        if 'main' not in res:
            return {"temp": 0, "humidity": 0, "wind": 0, "wind_dir": 0}

        return {
            "temp": res['main']['temp'],
            "humidity": res['main']['humidity'],
            "wind": res['wind']['speed'],
            "wind_dir": res['wind'].get('deg', 0)
        }

    except:
        return {"temp": 0, "humidity": 0, "wind": 0, "wind_dir": 0}


# =========================================
#  DISTANCE FUNCTION
# =========================================
def get_distance(lat, lon, tags, name):
    try:
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=3000)

        if gdf.empty:
            return 3000

        coords = []
        for geom in gdf.geometry:
            if geom is not None:
                centroid = geom.centroid
                coords.append((centroid.y, centroid.x))

        if not coords:
            return 3000

        return round(min([geodesic((lat, lon), c).meters for c in coords]), 2)

    except:
        return 3000


# =========================================
#  DATA COLLECTION
# =========================================
data = []

for city, (lat, lon) in cities.items():
    print(f" Processing {city}...")

    pollution = get_pollution(lat, lon)
    weather = get_weather(lat, lon)

    row = {
        "city": city,
        "lat": lat,
        "lon": lon,
        "timestamp": datetime.now().replace(microsecond=0),  

        # Pollution
        "pm2_5": pollution.get("pm2_5", 0),
        "pm10": pollution.get("pm10", 0),
        "no2": pollution.get("no2", 0),
        "co": pollution.get("co", 0),
        "so2": pollution.get("so2", 0),
        "o3": pollution.get("o3", 0),

        # Weather
        "temp": weather.get("temp", 0),
        "humidity": weather.get("humidity", 0),
        "wind": weather.get("wind", 0),
        "wind_dir": weather.get("wind_dir", 0),

        # Spatial
        "dist_road": get_distance(lat, lon, {"highway": True}, "road"),
        "dist_industry": get_distance(lat, lon, {"landuse": "industrial"}, "industry"),
        "dist_dump": get_distance(lat, lon, {"amenity": ["waste_disposal", "landfill"]}, "dump"),
        "dist_farm": get_distance(lat, lon, {"landuse": ["farmland", "farm"]}, "farm")
    }

    data.append(row)

new_df = pd.DataFrame(data)

# =========================================
#  SAVE DATA 
# =========================================
file_path = "data/raw_data.csv"

if os.path.exists(file_path):
    old_df = pd.read_csv(file_path)
    df = pd.concat([old_df, new_df], ignore_index=True)
else:
    df = new_df

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Sort
df = df.sort_values(by="timestamp")

# Save
df.to_csv(file_path, index=False)

print(f" Data saved successfully! Total rows: {len(df)}")