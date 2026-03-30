import requests
import pandas as pd
import joblib
import osmnx as ox
import numpy as np

from datetime import datetime
from zoneinfo import ZoneInfo

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


OPENAQ_API_KEY = "86cc13fc2903a925f8b0785b10aa36d73439de75a5ea65461d06cd1dcdcb7113"
OPENWEATHER_API_KEY = "4418090bafbbeac22965bf2a0b65a75b"

headers = {"X-API-Key": OPENAQ_API_KEY}
IST = ZoneInfo("Asia/Kolkata")

locations = {
    "Delhi_Anand_Vihar": (28.6469,77.3153),
    "Bangalore_Hebbal": (13.0358,77.5970),
    "Mumbai_Worli": (19.0176,72.8562),
    "Chennai_Adyar": (13.0067,80.2575)
}

print("\n Starting EnviroScan Pipeline...\n")


# NOISE FUNCTION
def noise(val, scale):
    if val is None or pd.isna(val) or val == 0:
        return np.random.uniform(20, 120)
    return max(0, val + np.random.uniform(-scale, scale))

# WEATHER API
def fetch_weather(lat, lon):
    try:
        res = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat":lat,"lon":lon,"appid":OPENWEATHER_API_KEY,"units":"metric"},
            timeout=5
        ).json()

        return res["main"]["temp"], res["main"]["humidity"], res["wind"]["speed"]
    except:
        return None, None, None


# OPENAQ POLLUTION
def fetch_pollutants(lat, lon):
    pollutants = {"pm25":None,"pm10":None,"no2":None,"so2":None,"co":None,"o3":None}

    try:
        loc = requests.get(
            "https://api.openaq.org/v3/locations",
            headers=headers,
            params={"coordinates":f"{lat},{lon}","radius":25000,"limit":1},
            timeout=5
        ).json()

        if loc.get("results"):
            loc_id = loc["results"][0]["id"]

            sensors = requests.get(
                f"https://api.openaq.org/v3/locations/{loc_id}/sensors",
                headers=headers
            ).json().get("results", [])

            for s in sensors:
                param = s["parameter"]["name"].lower()

                if param in pollutants:
                    meas = requests.get(
                        f"https://api.openaq.org/v3/sensors/{s['id']}/measurements",
                        headers=headers,
                        params={"limit":1}
                    ).json()

                    if meas.get("results"):
                        pollutants[param] = meas["results"][0]["value"]
    except:
        pass

    return pollutants

# OSM FEATURES 
def get_osm_features(lat, lon):
    try:
        # Roads
        G = ox.graph_from_point((lat, lon), dist=1000, network_type="drive")
        road_count = len(G.nodes)

        # Land use
        tags = {"landuse": ["industrial", "farmland"]}
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=1000)

        if not gdf.empty and "landuse" in gdf.columns:
            industry_count = (gdf["landuse"] == "industrial").sum()
            farmland_count = (gdf["landuse"] == "farmland").sum()
        else:
            industry_count = 0
            farmland_count = 0

        dump_count = 0

        return road_count, int(industry_count), int(farmland_count), dump_count

    except Exception as e:
        print("OSM ERROR:", e)
        return 0, 0, 0, 0


# DATA COLLECTION
rows = []
NUM_RECORDS_PER_LOCATION = 150
for place, (lat, lon) in locations.items():

    print("Collecting:", place)

    temp, hum, wind = fetch_weather(lat, lon)
    pol = fetch_pollutants(lat, lon)
    road, ind, farm, dump = get_osm_features(lat, lon)
    for i in range(NUM_RECORDS_PER_LOCATION):
        timestamp = datetime.now(IST) - pd.Timedelta(minutes=5*i)

        rows.append({
            "place": place,
            "latitude": lat,
            "longitude": lon,
            "timestamp": timestamp,

            "pm25": noise(pol["pm25"],8),
            "pm10": noise(pol["pm10"],10),
            "no2": noise(pol["no2"],8),
            "so2": noise(pol["so2"],8),
            "co": noise(pol["co"],0.5),
            "o3": noise(pol["o3"],15),

            "temperature": noise(temp,2),
            "humidity": noise(hum,5),
            "wind_speed": noise(wind,1),

            "road_count": road,
            "industry_count": ind,
            "farmland_count": farm,
            "dump_count": dump
        })

df = pd.DataFrame(rows)


# CLEANING
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.dayofweek

# FEATURE ENGINEERING
df["pollution_index"] = df["pm25"] + df["pm10"] + df["no2"]
df["gas_ratio"] = df["no2"] / (df["co"] + 1)
df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1)

# LABELING
def identify_source(row):
    if row["pm25"] > 70:
        return "Burning"
    elif row["so2"] > 12:
        return "Industrial"
    elif row["farmland_count"] > 0 and row["pm25"] > 25:
        return "Agricultural"
    elif row["road_count"] > 5 and row["no2"] > 10:
        return "Vehicular"
    else:
        return "Natural"
df["pollution_source"] = df.apply(identify_source, axis=1)


# BALANCING
min_count = df["pollution_source"].value_counts().min()
df = df.groupby("pollution_source").sample(n=min_count, random_state=42)

# MODEL PREP
features = [
    "pm25","pm10","no2","so2","co","o3",
    "temperature","humidity","wind_speed",
    "road_count","industry_count","farmland_count","dump_count",
    "pollution_index","gas_ratio","pm_ratio","hour","day"
]
X = df[features]
y = df["pollution_source"]
le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.25,random_state=42,stratify=y
)

# MODELS
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced")
xgb = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric="mlogloss")
rf.fit(X_train,y_train)
xgb.fit(X_train,y_train)

# EVALUATION
rf_acc = accuracy_score(y_test, rf.predict(X_test))
xgb_acc = accuracy_score(y_test, xgb.predict(X_test))

print("\nRF Accuracy:", rf_acc)
print("XGB Accuracy:", xgb_acc)
final_model = xgb if xgb_acc > rf_acc else rf


# FINAL RESULTS
pred = final_model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test,pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test,pred))
print("\nReport:\n", classification_report(y_test,pred))

# SAVE FILES
joblib.dump(final_model,"rf_model.pkl")
joblib.dump(scaler,"scaler.pkl")
joblib.dump(le,"label_encoder.pkl")
df.to_csv("enviro_scan_dataset.csv", index=False)
print("\nDONE - Files saved successfully!")