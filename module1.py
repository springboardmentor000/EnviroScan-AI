# ==========================================
# MODULE 1: INDIA DATASET (OPENWEATHER + OPENAQ)
# ==========================================

import requests
import pandas as pd
import random
import time
from datetime import datetime, timedelta

# ==============================
# API KEYS
# ==============================
OWM_API_KEY = "5ec26652e0c8495ca989e870aa1c8207"
OPENAQ_API_KEY = "e7d57256e3371e283ed2546983feeef049d830f6127b2ed923db4dfd952bfe49"


# ==============================
# INDIA STATES + MAJOR CITIES
# ==============================

states_data = {
    "Andhra Pradesh": [("Visakhapatnam", 17.6868, 83.2185)],
    "Telangana": [("Hyderabad", 17.3850, 78.4867)],
    "Maharashtra": [("Mumbai", 19.0760, 72.8777)],
    "Delhi": [("Delhi", 28.6139, 77.2090)],
    "Karnataka": [("Bangalore", 12.9716, 77.5946)],
    "Tamil Nadu": [("Chennai", 13.0827, 80.2707)],
    "West Bengal": [("Kolkata", 22.5726, 88.3639)],
    "Gujarat": [("Ahmedabad", 23.0225, 72.5714)],
    "Rajasthan": [("Jaipur", 26.9124, 75.7873)],
    "Uttar Pradesh": [("Lucknow", 26.8467, 80.9462)],
    "Punjab": [("Amritsar", 31.6340, 74.8723)],
    "Bihar": [("Patna", 25.5941, 85.1376)],
    "Madhya Pradesh": [("Bhopal", 23.2599, 77.4126)],
    "Kerala": [("Kochi", 9.9312, 76.2673)],
    "Odisha": [("Bhubaneswar", 20.2961, 85.8245)],
    "Assam": [("Guwahati", 26.1445, 91.7362)]
}

# ==============================
# API FUNCTIONS
# ==============================

def get_weather(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
        res = requests.get(url).json()

        return {
            "temperature": res["main"]["temp"],
            "humidity": res["main"]["humidity"],
            "pressure": res["main"]["pressure"],
            "wind_speed": res["wind"]["speed"]
        }
    except:
        return None


def get_pollution_owm(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OWM_API_KEY}"
        res = requests.get(url).json()

        comp = res["list"][0]["components"]

        return {
            "pm25": comp.get("pm2_5"),
            "pm10": comp.get("pm10"),
            "no2": comp.get("no2"),
            "co": comp.get("co") / 100,
            "so2": comp.get("so2"),
            "o3": comp.get("o3")
        }
    except:
        return None


def get_pollution_openaq(city):
    try:
        url = f"https://api.openaq.org/v2/latest?city={city}"
        headers = {"X-API-Key": OPENAQ_API_KEY}
        res = requests.get(url, headers=headers).json()

        measurements = res["results"][0]["measurements"]

        data = {}
        for m in measurements:
            data[m["parameter"]] = m["value"]

        return data
    except:
        return None


# ==============================
# CONFIG
# ==============================

ROWS = 150
start_time = datetime(2024, 1, 1)

records = []

print("🚀 Generating India Dataset (API Integrated)...")

# ==============================
# MAIN LOOP
# ==============================

for i in range(ROWS):

    timestamp = start_time + timedelta(hours=i)

    state = random.choice(list(states_data.keys()))
    city, base_lat, base_lon = random.choice(states_data[state])

    latitude = base_lat + random.uniform(-0.01, 0.01)
    longitude = base_lon + random.uniform(-0.01, 0.01)

    # ==============================
    # API CALLS
    # ==============================

    weather = get_weather(latitude, longitude)
    owm_pollution = get_pollution_owm(latitude, longitude)
    aq_pollution = get_pollution_openaq(city)

    # ==============================
    # MERGE DATA (PRIORITY: OWM > OpenAQ > Fallback)
    # ==============================

    def safe(val, fallback):
        return val if val is not None else fallback

    pm25 = safe(
        owm_pollution.get("pm25") if owm_pollution else None,
        aq_pollution.get("pm25") if aq_pollution else None
    ) or random.uniform(20,150)

    pm10 = safe(
        owm_pollution.get("pm10") if owm_pollution else None,
        aq_pollution.get("pm10") if aq_pollution else None
    ) or random.uniform(30,200)

    no2 = safe(
        owm_pollution.get("no2") if owm_pollution else None,
        aq_pollution.get("no2") if aq_pollution else None
    ) or random.uniform(10,100)

    co = safe(
        owm_pollution.get("co") if owm_pollution else None,
        aq_pollution.get("co") if aq_pollution else None
    ) or random.uniform(0.5,5)

    so2 = safe(
        owm_pollution.get("so2") if owm_pollution else None,
        aq_pollution.get("so2") if aq_pollution else None
    ) or random.uniform(5,50)

    o3 = safe(
        owm_pollution.get("o3") if owm_pollution else None,
        aq_pollution.get("o3") if aq_pollution else None
    ) or random.uniform(10,100)

    # Weather fallback
    if weather:
        temperature = weather["temperature"]
        humidity = weather["humidity"]
        pressure = weather["pressure"]
        wind_speed = weather["wind_speed"]
    else:
        temperature = random.uniform(20, 40)
        humidity = random.uniform(40, 80)
        pressure = random.uniform(995, 1030)
        wind_speed = random.uniform(1, 5)

    # ==============================
    # DISTANCES
    # ==============================

    dist_to_road = random.uniform(0.001, 0.05)
    dist_to_industry = random.uniform(0.001, 0.05)
    dist_to_dump = random.uniform(0.001, 0.05)

    # ==============================
    # SAVE ROW
    # ==============================

    records.append({
        "state": state,
        "city": city,
        "latitude": round(latitude,6),
        "longitude": round(longitude,6),
        "timestamp": timestamp,

        "pm25": round(pm25,2),
        "pm10": round(pm10,2),
        "no2": round(no2,2),
        "co": round(co,2),
        "so2": round(so2,2),
        "o3": round(o3,2),

        "temperature": round(temperature,2),
        "humidity": round(humidity,2),
        "pressure": round(pressure,2),
        "wind_speed": round(wind_speed,2),

        "dist_to_road": round(dist_to_road,4),
        "dist_to_industry": round(dist_to_industry,4),
        "dist_to_dump": round(dist_to_dump,4)
    })

    time.sleep(1)  # avoid API limit

# ==============================
# SAVE DATASET
# ==============================

df = pd.DataFrame(records)
df.to_csv("dataset.csv", index=False)

print("✅ Dataset Created Successfully!")
print("States Covered:", df["state"].unique())
print("Rows:", len(df))
print(df.head())