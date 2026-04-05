# EnviroScan-AI
EnviroScan AI-Powered Pollution Source Identifier using Geospatial Analytics
import requests
import csv
import time
from datetime import datetime

# -------------------------------------------------
# PASTE YOUR API KEYS HERE
# -------------------------------------------------
OPENAQ_KEY = "2a8c8e8998b251815df77b5b17bcc6736153cd2800c793d6ad430e7e7c0a1995"
OPENWEATHER_KEY = "83ac533dfb63f99c25607c05ec6eeaf2"

# Hyderabad Coordinates
HYD_LAT = 17.3850
HYD_LON = 78.4867

# OSM search radius in metres
OSM_RADIUS_M = 3000

# -------------------------------------------------
# 1. Get nearest monitoring stations (OpenAQ v3)
# -------------------------------------------------
def get_stations():
    print(" Fetching monitoring stations near Hyderabad...")
    url = "https://api.openaq.org/v3/locations"
    params = {
        "coordinates": f"{HYD_LAT},{HYD_LON}",
        "radius": 25000,
        "limit": 50
    }
    headers = {"X-API-Key": OPENAQ_KEY}
    r = requests.get(url, params=params, headers=headers, timeout=15).json()
    stations = r.get("results", [])
    print(f"Found {len(stations)} stations.\n")
    return stations


# -------------------------------------------------
# 2. Get pollution data — FIXED using sensors endpoint
# -------------------------------------------------
def get_pollution(station_id):
    headers = {"X-API-Key": OPENAQ_KEY}

    measurements = {
        "pm25": None, "pm10": None, "no2": None,
        "so2": None,  "o3": None,   "co": None, "bc": None
    }

    # STEP 1: Get sensors for this station
    try:
        sensors_url = f"https://api.openaq.org/v3/locations/{station_id}/sensors"
        r = requests.get(sensors_url, headers=headers, timeout=15).json()
        sensors = r.get("results", [])
    except Exception as e:
        print(f"      Could not fetch sensors: {e}")
        return measurements

    if not sensors:
        print(f"      No sensors found for station {station_id}")
        return measurements

    # STEP 2: For each sensor, get its latest measurement
    for sensor in sensors:
        sensor_id = sensor.get("id")
        param_info = sensor.get("parameter", {})
        param_name = param_info.get("name", "").lower()

        if param_name not in measurements:
            continue

        try:
            meas_url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
            params = {"limit": 1, "sort": "desc"}
            meas_r = requests.get(meas_url, headers=headers, params=params, timeout=15).json()
            meas_data = meas_r.get("results", [])

            if meas_data:
                measurements[param_name] = meas_data[0].get("value")
                print(f"     {param_name}: {measurements[param_name]}")

        except Exception as e:
            print(f"      Could not fetch {param_name}: {e}")

        time.sleep(0.2)  # be polite to the API

    return measurements


# -------------------------------------------------
# 3. OpenWeather — get weather data
# -------------------------------------------------
def get_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat, "lon": lon,
        "appid": OPENWEATHER_KEY,
        "units": "metric"
    }
    try:
        r = requests.get(url, params=params, timeout=15).json()
        return {
            "temp":           r["main"]["temp"],
            "humidity":       r["main"]["humidity"],
            "pressure":       r["main"]["pressure"],
            "wind_speed":     r["wind"]["speed"],
            "wind_direction": r["wind"].get("deg", None)
        }
    except Exception as e:
        print(f"      Weather fetch failed: {e}")
        return {"temp": None, "humidity": None, "pressure": None,
                "wind_speed": None, "wind_direction": None}


# -------------------------------------------------
# 4. OSM geospatial features via Overpass API
# -------------------------------------------------
def get_osm_features(lat, lon, radius_m=OSM_RADIUS_M):
    overpass_url = "https://overpass-api.de/api/interpreter"
    deg_offset = radius_m / 111000
    south = lat - deg_offset
    north = lat + deg_offset
    west  = lon - deg_offset
    east  = lon + deg_offset
    bbox  = f"{south},{west},{north},{east}"

    queries = {
        "roads":      f'[out:json][timeout:25]; way["highway"~"motorway|trunk|primary|secondary"]({bbox}); out count;',
        "industrial": f'[out:json][timeout:25]; way["landuse"~"industrial|commercial"]({bbox}); out count;',
        "dumps":      f'[out:json][timeout:25]; ( way["landuse"="landfill"]({bbox}); node["amenity"~"waste_disposal|recycling"]({bbox}); ); out count;',
        "agri":       f'[out:json][timeout:25]; way["landuse"~"farmland|farm|orchard|meadow"]({bbox}); out count;',
    }

    counts = {"roads": 0, "industrial": 0, "dumps": 0, "agri": 0}

    for category, query in queries.items():
        for attempt in range(2):   # retry once on timeout
            try:
                resp = requests.post(overpass_url, data={"data": query}, timeout=30)
                resp.raise_for_status()
                elements = resp.json().get("elements", [])
                if elements:
                    counts[category] = int(elements[0].get("tags", {}).get("total", 0))
                break
            except Exception as e:
                if attempt == 0:
                    print(f"      OSM {category} timeout — retrying...")
                    time.sleep(5)
                else:
                    print(f"      OSM {category} failed: {e}")

    print(f"    OSM → roads:{counts['roads']} industrial:{counts['industrial']} dumps:{counts['dumps']} agri:{counts['agri']}")

    return {
        "road_count":             counts["roads"],
        "has_major_road":         1 if counts["roads"] > 0 else 0,
        "industrial_zone_count":  counts["industrial"],
        "near_industrial_zone":   1 if counts["industrial"] > 0 else 0,
        "dump_site_count":        counts["dumps"],
        "near_dump_site":         1 if counts["dumps"] > 0 else 0,
        "agricultural_count":     counts["agri"],
        "near_agricultural_area": 1 if counts["agri"] > 0 else 0,
    }


# -------------------------------------------------
# 5. Build dataset & save CSV
# -------------------------------------------------
def build_dataset():
    stations = get_stations()
    filename = "Hyderabad_pollution_latest.csv"

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "station_id", "station_name", "latitude", "longitude", "timestamp",
            "pm25", "pm10", "no2", "so2", "o3", "co", "bc",
            "temp", "humidity", "pressure", "wind_speed", "wind_direction",
            "road_count", "has_major_road",
            "industrial_zone_count", "near_industrial_zone",
            "dump_site_count", "near_dump_site",
            "agricultural_count", "near_agricultural_area"
        ])

        print("Fetching pollution + weather + OSM data...\n")

        for i, s in enumerate(stations, 1):
            sid  = s["id"]
            name = s["name"]
            lat  = s["coordinates"]["latitude"]
            lon  = s["coordinates"]["longitude"]

            print(f"→ [{i}/{len(stations)}] {name} (ID: {sid})")

            pollution = get_pollution(sid)
            weather   = get_weather(lat, lon)
            osm       = get_osm_features(lat, lon)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            writer.writerow([
                sid, name, lat, lon, timestamp,
                pollution["pm25"], pollution["pm10"], pollution["no2"],  
                pollution["so2"],  pollution["o3"],   pollution["co"], pollution["bc"],
                weather["temp"], weather["humidity"], weather["pressure"],
                weather["wind_speed"], weather["wind_direction"],
                osm["road_count"],            osm["has_major_road"],
                osm["industrial_zone_count"], osm["near_industrial_zone"],
                osm["dump_site_count"],       osm["near_dump_site"],
                osm["agricultural_count"],    osm["near_agricultural_area"],
            ])
            print(f" Done\n")

    print(f"DONE! File saved as {filename}")
# RUN
# -------------------------------------------------
build_dataset()
