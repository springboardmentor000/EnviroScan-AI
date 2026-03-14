import re
import requests
import pandas as pd
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY  = "8c92f0cb58b0377bd0c09e47ae9b4b7d1e8550d5bf42c4e93115dee063c80a94"
BASE_URL = "https://api.openaq.org/v3"
HEADERS  = {"X-API-Key": API_KEY}

STATIONS = [
    {"id": 3409492, "name": "HB Colony, Vijayawada - APPCB",  "lat": 16.5206, "lon": 80.6404},
    {"id": 3409392, "name": "Kanuru, Vijayawada - APPCB",     "lat": 16.4890, "lon": 80.6940}
]

# ✅ Full winter season: Nov 2025 - Mar 2026
date_from = "2026-01-01T00:00:00Z"  # Nov 1, 2025
date_to   = "2026-03-10T23:59:59Z"  # Mar 6, 2026 (today)


TARGET_PARAMS = {"pm25", "pm10", "no2", "o3", "so2", "co"}

# ──────────────────────────────────────
def parse_found(val):
    """Handles '>1000', '500', None safely"""
    if val is None:
        return 0
    digits = re.sub(r"[^\d]", "", str(val))
    return int(digits) if digits else 0

def safe_get(url, params=None):
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.json()
        print(f"   ❌ {r.status_code}: {r.text[:150]}")
        return {}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {}

def get_sensors(station):
    data = safe_get(f"{BASE_URL}/locations/{station['id']}/sensors")
    tasks = []
    for s in data.get("results", []):
        param = (s.get("parameter", {}).get("name") or "").lower().replace("_", "").replace(".", "")
        if param in TARGET_PARAMS:
            tasks.append({
                "sensor_id":   s["id"],
                "parameter":   param,
                "location_id": station["id"],
                "location":    station["name"],
                "lat":         station["lat"],
                "lon":         station["lon"],
            })
    return tasks

def fetch_hourly(task):
    all_data, page = [], 1
    while True:
        data = safe_get(f"{BASE_URL}/sensors/{task['sensor_id']}/hours", params={
            "datetime_from": date_from,
            "datetime_to":   date_to,
            "limit": 1000,
            "page":  page
        })
        results = data.get("results", [])
        if not results:
            break

        for r in results:
            all_data.append({
                "city":        "Vijayawada",
                "timestamp":   r.get("period", {}).get("datetimeTo", {}).get("local"),
                "location_id": task["location_id"],
                "location":    task["location"],
                "lat":         task["lat"],
                "lon":         task["lon"],
                "parameter":   task["parameter"],
                "value":       r.get("value"),
            })

        found = parse_found(data.get("meta", {}).get("found", 0))
        if found > 0 and len(all_data) >= found:
            break
        elif len(results) < 1000:
            break  # last page
        page += 1

    return all_data

# ──────────────────────────────────────
def main():
    print("🚀 Vijayawada Hourly Air Quality (HB Colony + Kanuru)")
    print("=" * 55)

    # Collect sensors
    all_sensor_tasks = []
    for station in STATIONS:
        tasks = get_sensors(station)
        print(f"📍 {station['name']} → {len(tasks)} sensors: {[t['parameter'] for t in tasks]}")
        all_sensor_tasks.extend(tasks)

    print(f"\n⚡ Fetching {len(all_sensor_tasks)} sensors in parallel...")

    # Parallel fetch
    all_records = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(fetch_hourly, t): t for t in all_sensor_tasks}
        for future in as_completed(futures):
            records = future.result()
            t = futures[future]
            print(f"  ✅ {t['location']} | {t['parameter']} → {len(records)} hourly records")
            all_records.extend(records)

    if not all_records:
        print("❌ No data found.")
        return

    # Pivot to wide format
    df = pd.DataFrame(all_records)

# Convert timestamp to Indian time
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)\
                        .dt.tz_convert("Asia/Kolkata")\
                        .dt.tz_localize(None)
    
# Align timestamps to exact hour (important for merging with weather)
    df["timestamp"] = df["timestamp"].dt.floor("h")

    df = df.groupby(["city", "timestamp", "location_id", "location", "lat", "lon", "parameter"])["value"].mean().reset_index()

    df_wide = df.pivot_table(
        index=["city", "timestamp", "location_id", "location", "lat", "lon"],
        columns="parameter",
        values="value",
        aggfunc="mean"
    ).reset_index()
    df_wide.columns.name = None

    if "pm25" in df_wide.columns:
        df_wide = df_wide.rename(columns={"pm25": "pm2_5"})

    for col in ["pm2_5", "pm10", "no2", "o3", "so2", "co"]:
        if col not in df_wide.columns:
            df_wide[col] = None

    df_wide = df_wide[["city", "timestamp", "location_id", "location", "lat", "lon",
                        "pm2_5", "pm10", "no2", "o3", "so2", "co"]]
    df_wide = df_wide.sort_values(["location", "timestamp"]).reset_index(drop=True)

    # Verify hourly spacing
    for loc in df_wide["location"].unique():
        subset = df_wide[df_wide["location"] == loc]["timestamp"].sort_values()
        diffs  = subset.diff().dropna().dt.total_seconds() / 3600
        print(f"\n⏱ {loc}")
        print(f"   Avg interval: {diffs.mean():.1f}h | Min: {diffs.min():.1f}h | Max: {diffs.max():.1f}h")

    df_wide.to_csv("vijayawada_airquality.csv", index=False)
    print(f"\n🎉 Saved {len(df_wide)} rows → 'vijayawada_airquality.csv'")
    print(f"   Date range: {df_wide['timestamp'].min()} → {df_wide['timestamp'].max()}")
    print(df_wide.head(5).to_string())

if __name__ == "__main__":
    main()
