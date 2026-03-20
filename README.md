# EnviroScan-AI
EnviroScan AI-Powered Pollution Source Identifier using Geospatial Analytics
owm_data.py
import requests
import pandas as pd
import time
from datetime import datetime
import os

API_KEY = ""   

cities = [
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
    {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
]

# Repeat data collection multiple times
for cycle in range(3):   # collect 3 times
    print(f"\n--- Cycle {cycle+1} ---")

    all_data = []

    for city in cities:
        print(f"Fetching data for {city['name']}...")

        lat = city["lat"]
        lon = city["lon"]

        # ---- Air Pollution ----
        air_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
        air_response = requests.get(air_url)

        if air_response.status_code != 200:
            print("Air API Error:", air_response.text)
            continue

        air_data = air_response.json()
        components = air_data["list"][0]["components"]

        # ---- Weather ----
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        weather_response = requests.get(weather_url)

        if weather_response.status_code != 200:
            print("Weather API Error:", weather_response.text)
            continue

        weather_data = weather_response.json()

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

    # Append to CSV (not overwrite)
    file_exists = os.path.isfile("enviro_data_india.csv")

    df.to_csv("enviro_data_india.csv", index=False)

    print("Cycle completed and data saved.")

    # wait before next cycle (for demo use 10 sec, later change to 1 hour)
    time.sleep(10)

print("\nAll data collection completed!")


distance_featuers.py
import pandas as pd
import osmnx as ox
from shapely.geometry import Point

# Load dataset
df = pd.read_csv("enviro_data_india.csv")

def get_distances(lat, lon):
    try:
        point = (lat, lon)

        tags = {
            "highway": True,
            "landuse": True
        }

        # Get nearby features
        gdf = ox.features_from_point(point, tags=tags, dist=2000)

        if gdf.empty:
            return (None, None, None)

        # Convert to projected CRS (meters)
        gdf = gdf.to_crs(epsg=3857)

        # Convert point to same CRS
        point_geom = Point(lon, lat)
        point_geom = ox.projection.project_geometry(point_geom, crs="EPSG:4326")[0]

        # Filter safely
        roads = gdf[gdf["highway"].notnull()] if "highway" in gdf.columns else gdf.iloc[0:0]
        industrial = gdf[gdf["landuse"] == "industrial"] if "landuse" in gdf.columns else gdf.iloc[0:0]
        farmland = gdf[gdf["landuse"] == "farmland"] if "landuse" in gdf.columns else gdf.iloc[0:0]

        def min_distance(geo_df):
            if geo_df.empty:
                return None
            return geo_df.distance(point_geom).min()

        return (
            min_distance(roads),
            min_distance(industrial),
            min_distance(farmland)
        )

    except Exception as e:
        print("Error:", e)
        return (None, None, None)


# -------- MAIN EXECUTION --------
print("Processing rows...")

distances = []

for index, row in df.iterrows():
    print(f"Processing row {index+1}/{len(df)}...")
    d = get_distances(row["Latitude"], row["Longitude"])
    distances.append(d)

# Add new columns
df[["Dist_Road", "Dist_Industrial", "Dist_Farmland"]] = pd.DataFrame(distances)

# Save file
df.to_csv("enviro_data_with_distance.csv", index=False)

print("\n✅ Distance features added successfully!")

model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("enviro_data_with_distance.csv")

# Drop missing values
df = df.dropna()

# -------------------------------
# ✅ SOURCE LABELING (IMPORTANT)
# -------------------------------
def label_source(row):
    # Vehicular: high NO2 OR very close to road
    if row["NO2"] > 10 or row["Dist_Road"] < 800:
        return "Vehicular"

    # Industrial: high SO2 OR close to industrial area
    elif row["SO2"] > 5 or row["Dist_Industrial"] < 1200:
        return "Industrial"

    # Agricultural: high PM2.5 + near farmland
    elif row["PM2.5"] > 25 or row["Dist_Farmland"] < 1200:
        return "Agricultural"

    else:
        return "Natural"

df["Source"] = df.apply(label_source, axis=1)

print("\nSample labeled data:")
print(df[["City", "Source"]])

# -------------------------------
# ✅ FEATURES (INPUT)
# -------------------------------
X = df[[
    "Dist_Road",
    "Dist_Industrial",
    "Dist_Farmland",
    "Temperature (°C)",
    "Humidity (%)",
    "Wind Speed (m/s)",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2"
]]

# -------------------------------
# ✅ TARGET (OUTPUT)
# -------------------------------
y = df["Source"]

# Encode labels (text → numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# -------------------------------
# ✅ MODEL (Decision Tree)
# -------------------------------
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# Accuracy (on same data)
accuracy = model.score(X, y_encoded)
print("\nModel Accuracy:", accuracy)

# -------------------------------
# ✅ TEST PREDICTION
# -------------------------------
sample = X.iloc[0:1]
prediction = model.predict(sample)

print("\nPredicted Source:", le.inverse_transform(prediction)[0])  


map.py

import pandas as pd
import folium
from folium.plugins import HeatMap

# Load dataset
df = pd.read_csv("enviro_data_with_distance.csv")

# Create base map (center of India)
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# -------------------------------
# Add city markers
# -------------------------------
for _, row in df.iterrows():
    popup_text = f"""
    City: {row['City']}<br>
    PM2.5: {row['PM2.5']}<br>
    Source: {row.get('Source', 'N/A')}
    """

    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=popup_text,
        icon=folium.Icon(color="blue")
    ).add_to(m)

# -------------------------------
# Heatmap (PM2.5 intensity)
# -------------------------------
heat_data = [
    [row["Latitude"], row["Longitude"], row["PM2.5"]]
    for _, row in df.iterrows()
]

HeatMap(heat_data).add_to(m)

# Save map
m.save("pollution_map.html")

print("Map created successfully!")

visualization.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("enviro_data_with_distance.csv")

# Create figure with 2 plots
plt.figure(figsize=(10,5))

# ---- Bar Graph ----
plt.subplot(1,2,1)
plt.bar(df["City"], df["PM2.5"])
plt.xticks(rotation=45)
plt.title("PM2.5 Levels")

# ---- Scatter Plot ----
plt.subplot(1,2,2)
plt.scatter(df["Dist_Road"], df["PM2.5"])
plt.title("Distance vs PM2.5")
plt.xlabel("Dist_Road")
plt.ylabel("PM2.5")

plt.tight_layout()
plt.show() 

visualization.py

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Title
st.title("🌍 Air Pollution Source Predictor")

# Load data
df = pd.read_csv("enviro_data_with_distance.csv")
df = df.dropna()

# Labeling function
def label_source(row):
    if row["NO2"] > 10 or row["Dist_Road"] < 800:
        return "Vehicular"
    elif row["SO2"] > 5 or row["Dist_Industrial"] < 1200:
        return "Industrial"
    elif row["PM2.5"] > 25 or row["Dist_Farmland"] < 1200:
        return "Agricultural"
    else:
        return "Natural"

df["Source"] = df.apply(label_source, axis=1)

# Features
X = df[[
    "Dist_Road",
    "Dist_Industrial",
    "Dist_Farmland",
    "Temperature (°C)",
    "Humidity (%)",
    "Wind Speed (m/s)",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2"
]]

y = df["Source"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# ------------------------
# USER INPUT
# ------------------------
st.sidebar.header("Enter Data")

dist_road = st.sidebar.number_input("Distance to Road", 0, 5000, 500)
dist_ind = st.sidebar.number_input("Distance to Industrial Area", 0, 5000, 1000)
dist_farm = st.sidebar.number_input("Distance to Farmland", 0, 5000, 1000)
temp = st.sidebar.number_input("Temperature", 0.0, 50.0, 30.0)
hum = st.sidebar.number_input("Humidity", 0, 100, 60)
wind = st.sidebar.number_input("Wind Speed", 0.0, 20.0, 3.0)
pm25 = st.sidebar.number_input("PM2.5", 0.0, 500.0, 50.0)
pm10 = st.sidebar.number_input("PM10", 0.0, 500.0, 80.0)
no2 = st.sidebar.number_input("NO2", 0.0, 200.0, 20.0)
so2 = st.sidebar.number_input("SO2", 0.0, 200.0, 10.0)

# Prediction
if st.button("Predict Source"):
    sample = [[
        dist_road, dist_ind, dist_farm,
        temp, hum, wind,
        pm25, pm10, no2, so2
    ]]

    prediction = model.predict(sample)
    result = le.inverse_transform(prediction)

    st.success(f"Predicted Pollution Source: {result[0]}")


    app.py

    import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Title
st.title("🌍 Air Pollution Source Predictor")

# Load data
df = pd.read_csv("enviro_data_with_distance.csv")
df = df.dropna()

# Labeling function
def label_source(row):
    if row["NO2"] > 10 or row["Dist_Road"] < 800:
        return "Vehicular"
    elif row["SO2"] > 5 or row["Dist_Industrial"] < 1200:
        return "Industrial"
    elif row["PM2.5"] > 25 or row["Dist_Farmland"] < 1200:
        return "Agricultural"
    else:
        return "Natural"

df["Source"] = df.apply(label_source, axis=1)

# Features
X = df[[
    "Dist_Road",
    "Dist_Industrial",
    "Dist_Farmland",
    "Temperature (°C)",
    "Humidity (%)",
    "Wind Speed (m/s)",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2"
]]

y = df["Source"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# ------------------------
# USER INPUT
# ------------------------
st.sidebar.header("Enter Data")

dist_road = st.sidebar.number_input("Distance to Road", 0, 5000, 500)
dist_ind = st.sidebar.number_input("Distance to Industrial Area", 0, 5000, 1000)
dist_farm = st.sidebar.number_input("Distance to Farmland", 0, 5000, 1000)
temp = st.sidebar.number_input("Temperature", 0.0, 50.0, 30.0)
hum = st.sidebar.number_input("Humidity", 0, 100, 60)
wind = st.sidebar.number_input("Wind Speed", 0.0, 20.0, 3.0)
pm25 = st.sidebar.number_input("PM2.5", 0.0, 500.0, 50.0)
pm10 = st.sidebar.number_input("PM10", 0.0, 500.0, 80.0)
no2 = st.sidebar.number_input("NO2", 0.0, 200.0, 20.0)
so2 = st.sidebar.number_input("SO2", 0.0, 200.0, 10.0)

# Prediction
if st.button("Predict Source"):
    sample = [[
        dist_road, dist_ind, dist_farm,
        temp, hum, wind,
        pm25, pm10, no2, so2
    ]]

    prediction = model.predict(sample)
    result = le.inverse_transform(prediction)

    st.success(f"Predicted Pollution Source: {result[0]}")
