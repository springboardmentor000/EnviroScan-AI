import requests
import pandas as pd
from datetime import datetime, timezone

OWM_API_KEY = "c630fe86123ccf82de5644158b421b80"

STATIONS = [
    {"id": 344140, "name": "Somajiguda, Hyderabad - TSPCB", "lat": 17.417094, "lon": 78.457437},
    {"id": 407, "name": "Zoo Park, Hyderabad - TSPCB", "lat": 17.349694, "lon": 78.451437},
    {"id": 5647, "name": "Sanathnagar, Hyderabad - TSPCB", "lat": 17.4559458, "lon": 78.4332152},
    {"id": 5623, "name": "Central University, Hyderabad - TSPCB", "lat": 17.460103, "lon": 78.334361},
    {"id": 344104, "name": "Kompally Municipal Office, Hyderabad - TSPCB", "lat": 17.544899, "lon": 78.486949}
]
CITY = "Hyderabad"

def fetch_openmeteo(lat, lon, start="2026-02-01", end="2026-03-30"):
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "surface_pressure"
            ]),
            "timezone": "Asia/Kolkata",
            "wind_speed_unit": "ms"
        },
        timeout=30
    )
    if resp.status_code != 200:
        print(f"❌ Open-Meteo Error {resp.status_code}: {resp.text}")
        return []
    h = resp.json()["hourly"]
    return [{
        "timestamp": t,
        "temperature_c": h["temperature_2m"][i],
        "humidity": h["relative_humidity_2m"][i],
        "pressure_hpa": h["surface_pressure"][i],
        "wind_speed_ms": h["wind_speed_10m"][i],
        "wind_direction": h["wind_direction_10m"][i]
    } for i, t in enumerate(h["time"])]

def fetch_owm_current(lat, lon):
    if not OWM_API_KEY:
        print("⚠️ No OWM API key - skipping current weather")
        return []
    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "units": "metric", "appid": OWM_API_KEY},
        timeout=15
    )
    if resp.status_code != 200:
        print(f"⚠️ OWM Error {resp.status_code} — skipping current weather")
        return []

    d = resp.json()
    main = d.get("main", {})
    wind = d.get("wind", {})

    return [{
        "timestamp": datetime.fromtimestamp(d["dt"], tz=timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S"),
        "temperature_c": main.get("temp"),
        "humidity": main.get("humidity"),
        "pressure_hpa": main.get("pressure"),
        "wind_speed_ms": wind.get("speed"),
        "wind_direction": wind.get("deg")
    }]

all_records = []

for station in STATIONS:
    print(f"\n📍 Station: {station['name']} (ID: {station['id']})")

    print(f"   ⏬ Fetching Open-Meteo historical...")
    historical = fetch_openmeteo(station["lat"], station["lon"])
    print(f"   ✅ {len(historical)} historical records")

    print(f"   ⏬ Fetching OWM current weather...")
    current = fetch_owm_current(station["lat"], station["lon"])
    print(f"   ✅ {len(current)} current record(s)")

    combined = historical + current
    for row in combined:
        row["city"] = CITY
        row["location_id"] = station["id"]
        row["location"] = station["name"]
        row["lat"] = station["lat"]
        row["lon"] = station["lon"]

    all_records.extend(combined)

df = pd.DataFrame(all_records)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

df["date"] = df["timestamp"].dt.date
df["time"] = df["timestamp"].dt.strftime("%H:%M:%S")

df = df.drop_duplicates(subset=["location_id", "timestamp"], keep="first")
df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)

df = df[[
    "city", "timestamp", "date", "time", "location_id", "location", "lat", "lon",
    "temperature_c", "humidity", "pressure_hpa",
    "wind_speed_ms", "wind_direction"
]]

print(f"\n✅ Total records: {len(df)}")
print(f"   From: {df['timestamp'].min()}")
print(f"   To: {df['timestamp'].max()}")
print(f"\n   Rows per station:")
print(df.groupby("location")["timestamp"].count().to_string())
print(df.head(5).to_string())

filename = f"hyderabad_weather.csv"
df.to_csv(filename, index=False)
print(f"\n💾 Saved → {filename}")