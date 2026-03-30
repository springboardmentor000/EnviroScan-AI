# ==========================================
# MODULE 5: GEOSPATIAL MAP + HEATMAP (FINAL)
# ==========================================

import pandas as pd
import folium
from folium.plugins import HeatMap
import joblib

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("labeled_environment_dataset.csv")

# ==============================
# LOAD MODEL + ENCODERS
# ==============================

model = joblib.load("pollution_source_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
city_encoder = joblib.load("city_encoder.pkl")

# ==============================
# HANDLE CITY ENCODING (FIXED)
# ==============================

def safe_encode(city):
    if city in city_encoder.classes_:
        return city_encoder.transform([city])[0]
    else:
        return -1   # unknown city

df["city_encoded"] = df["city"].apply(safe_encode)

# ==============================
# FEATURES (MATCH MODEL)
# ==============================

features = [
    "pm25","pm10","no2","co","so2","o3",
    "temperature","humidity","pressure","wind_speed",
    "dist_to_road","dist_to_industry","dist_to_dump",
    "city_encoded"
]

X = df[features]

# ==============================
# PREDICTION
# ==============================

pred = model.predict(X)
df["predicted_source"] = label_encoder.inverse_transform(pred)

# ==============================
# CREATE MAP (INDIA CENTER)
# ==============================

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# ==============================
# HEATMAP
# ==============================

heat_data = [
    [row["latitude"], row["longitude"], row["pm25"]]
    for _, row in df.iterrows()
]

HeatMap(heat_data).add_to(m)

# ==============================
# ICON FUNCTION (SYMBOLS)
# ==============================

def get_icon(source):

    if source == "Industrial":
        return folium.Icon(color="red", icon="industry", prefix="fa")

    elif source == "Vehicular":
        return folium.Icon(color="blue", icon="car", prefix="fa")

    elif source == "Agricultural":
        return folium.Icon(color="orange", icon="leaf", prefix="fa")

    elif source == "Burning":
        return folium.Icon(color="darkred", icon="fire", prefix="fa")

    else:
        return folium.Icon(color="green", icon="tree", prefix="fa")

# ==============================
# ADD MARKERS
# ==============================

for _, row in df.iterrows():

    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        icon=get_icon(row["predicted_source"]),
        popup=f"""
        <b>State:</b> {row.get('state','N/A')}<br>
        <b>City:</b> {row['city']}<br>
        <b>Source:</b> {row['predicted_source']}<br>
        <b>PM2.5:</b> {row['pm25']}
        """
    ).add_to(m)

# ==============================
# HIGH POLLUTION ZONES
# ==============================

for _, row in df.iterrows():
    if row["pm25"] > 0.7:
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=500,
            color="darkred",
            fill=True,
            fill_opacity=0.3,
            popup="⚠ HIGH POLLUTION ZONE"
        ).add_to(m)

# ==============================
# SAVE MAP
# ==============================

m.save("pollution_map.html")

print("✅ FINAL MAP CREATED SUCCESSFULLY!")