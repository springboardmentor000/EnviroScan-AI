
# ==========================================
# ENVIRONMENTAL DATASET GENERATOR (API + 50 ROWS + RANDOM GPS)
# ==========================================

import requests
import pandas as pd
import random
from datetime import datetime, timedelta

# ==============================
# API KEYS
# ==============================

OWM_API_KEY = "5ec26652e0c8495ca989e870aa1c8207"
OPENAQ_API_KEY = "e7d57256e3371e283ed2546983feeef049d830f6127b2ed923db4dfd952bfe49"

# ==============================
# LOCATION
# ==============================

CITY = "Visakhapatnam"
LAT = 17.6868
LON = 83.2185

# ==============================
# API URLS
# ==============================

weather_url = "https://api.openweathermap.org/data/2.5/weather"
air_url = "https://api.openaq.org/v2/latest"

headers = {"X-API-Key": OPENAQ_API_KEY}

required_pollutants = ["pm25","pm10","no2","co","so2","o3"]

# ==============================
# DATA STORAGE
# ==============================

records = []

start_time = datetime(2024,1,1,0,0,0)

ROWS = 200

# ==============================
# FETCH BASE API DATA
# ==============================

print("Fetching environmental data from APIs...")

# AIR QUALITY
air_params = {
    "coordinates": f"{LAT},{LON}",
    "radius": 10000,
    "limit": 100
}

air_response = requests.get(air_url, headers=headers, params=air_params)

air_base = {p:0 for p in required_pollutants}

if air_response.status_code == 200:

    data = air_response.json()

    for result in data.get("results", []):
        for m in result.get("measurements", []):

            param = m.get("parameter")
            value = m.get("value")

            if param in air_base and air_base[param] == 0:
                air_base[param] = value

# WEATHER
weather_params = {
    "lat": LAT,
    "lon": LON,
    "appid": OWM_API_KEY,
    "units": "metric"
}

weather_response = requests.get(weather_url, params=weather_params)

temperature_base = 30
humidity_base = 60
wind_base = 2

if weather_response.status_code == 200:

    w = weather_response.json()

    temperature_base = w["main"]["temp"]
    humidity_base = w["main"]["humidity"]
    wind_base = w["wind"]["speed"]

# ==============================
# DATA GENERATION
# ==============================

for i in range(ROWS):

    timestamp = start_time + timedelta(hours=i)

    # Random GPS around Visakhapatnam
    latitude = LAT + random.uniform(-0.01,0.01)
    longitude = LON + random.uniform(-0.01,0.01)

    # Air pollution variation
    pm25 = air_base["pm25"] + random.uniform(-5,5)
    pm10 = air_base["pm10"] + random.uniform(-10,10)
    no2 = air_base["no2"] + random.uniform(-3,3)
    co = air_base["co"] + random.uniform(-0.5,0.5)
    so2 = air_base["so2"] + random.uniform(-2,2)
    o3 = air_base["o3"] + random.uniform(-5,5)

    nh3 = random.uniform(5,40)
    benzene = random.uniform(1,20)
    toluene = random.uniform(1,25)

    # Weather variation
    temperature = temperature_base + random.uniform(-2,2)
    humidity = humidity_base + random.uniform(-5,5)
    wind_speed = wind_base + random.uniform(-1,1)

    wind_direction = random.randint(0,360)

    pressure = random.uniform(995,1030)
    visibility = random.uniform(4,10)

    # AQI calculation
    aqi = (pm25*0.5 + pm10*0.3 + no2*0.1 + co*10)/4

    row = {

        "city": CITY,
        "latitude": round(latitude,6),
        "longitude": round(longitude,6),
        "timestamp": timestamp,

        "hour": timestamp.hour,
        "day": timestamp.day,
        "month": timestamp.month,

        "pm25": round(pm25,2),
        "pm10": round(pm10,2),
        "no2": round(no2,2),
        "co": round(co,2),
        "so2": round(so2,2),
        "o3": round(o3,2),
        "nh3": round(nh3,2),
        "benzene": round(benzene,2),
        "toluene": round(toluene,2),

        "temperature": round(temperature,2),
        "humidity": round(humidity,2),
        "wind_speed": round(wind_speed,2),
        "wind_direction": wind_direction,
        "pressure": round(pressure,2),
        "visibility": round(visibility,2),

        "aqi": round(aqi,2)
    }

    records.append(row)

# ==============================
# CREATE DATASET
# ==============================

df = pd.DataFrame(records)

df.to_csv("dataset.csv", index=False)

print("\nDataset Created Successfully!")
print("Rows:", len(df))
print(df.head())
