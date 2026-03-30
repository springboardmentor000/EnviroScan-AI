# EnviroScan-AI
EnviroScan AI-Powered Pollution Source Identifier using Geospatial Analytics
#######  owm_data.py   ##########



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


##########   distance_featuers.py    ########

import pandas as pd
import osmnx as ox
from shapely.geometry import Point
import geopandas as gpd

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

        # ✅ FIX: Create proper GeoDataFrame for point
        point_geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

        # Safe filtering
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


##########    model.py   ########


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("enviro_data_with_distance.csv")

# ✅ ADD THIS
print(df["City"].unique())
print("Total cities:", df["City"].nunique())

# Drop missing values
df = df.fillna(0)

# -------------------------------
# ✅ SOURCE LABELING (IMPORTANT)
# -------------------------------
def label_source(row):
    if row["Dist_Industrial"] and row["Dist_Industrial"] < 1000 and row["SO2"] > 5:
        return "Industrial"
    elif row["Dist_Farmland"] and row["Dist_Farmland"] < 1000 and row["PM2.5"] > 20:
        return "Agricultural"
    elif row["Dist_Road"] and row["Dist_Road"] < 500 and row["NO2"] > 10:
        return "Vehicular"
    else:
        return "Natural"

df["Source"] = df.apply(label_source, axis=1)
# Save updated dataset with Source column
df.to_csv("enviro_data_with_distance.csv", index=False)

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


######### map.py  ########

import pandas as pd
import folium

# Load dataset
df = pd.read_csv("enviro_data_with_distance.csv")

# Create base map
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# Feature groups (checkbox layers)
vehicular_group = folium.FeatureGroup(name="Vehicular")
industrial_group = folium.FeatureGroup(name="Industrial")
agriculture_group = folium.FeatureGroup(name="Agricultural")
natural_group = folium.FeatureGroup(name="Natural")

# Function to assign group + color + icon
def get_properties(source):
    if source == "Vehicular":
        return vehicular_group, "red", "car"
    elif source == "Industrial":
        return industrial_group, "blue", "industry"
    elif source == "Agricultural":
        return agriculture_group, "green", "leaf"
    else:
        return natural_group, "gray", "cloud"

# Add markers with icons
for _, row in df.iterrows():
    group, color, icon = get_properties(row["Source"])

    folium.Marker(
        location=[row["Latitude"], row["Longitude"]],
        popup=f"""
        City: {row['City']}<br>
        Source: {row['Source']}<br>
        PM2.5: {row['PM2.5']}
        """,
        icon=folium.Icon(color=color, icon=icon, prefix="fa")  # FontAwesome icons
    ).add_to(group)

# Add groups to map
vehicular_group.add_to(m)
industrial_group.add_to(m)
agriculture_group.add_to(m)
natural_group.add_to(m)

# Add checkbox control
folium.LayerControl().add_to(m)

# Save map
m.save("pollution_map.html")

print("✅ Map with icons + filter created!")

#######   visualization.py  #####

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("enviro_data_with_distance.csv")

# ✅ NOW print
print(df["City"].unique())
print("Total unique cities:", df["City"].nunique())

# -------------------------------
# 1️⃣ BAR GRAPH (PM2.5 per City)
# -------------------------------
# Group by city and take average
city_pm25 = df.groupby("City")["PM2.5"].mean()

plt.figure()
plt.bar(city_pm25.index, city_pm25.values)
plt.title("Average PM2.5 by City")
plt.xlabel("City")
plt.ylabel("PM2.5")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 2️⃣ SCATTER PLOT (PM2.5 vs NO2)
# -------------------------------
city_data = df.groupby("City")[["PM2.5", "NO2"]].mean()

plt.figure()
plt.scatter(city_data["PM2.5"], city_data["NO2"])

for i, city in enumerate(city_data.index):
    plt.text(city_data["PM2.5"].iloc[i], city_data["NO2"].iloc[i], city)

plt.title("PM2.5 vs NO2 (City-wise)")
plt.xlabel("PM2.5")
plt.ylabel("NO2")
plt.show()

# -------------------------------
# 3️⃣ PIE CHART (Source Distribution)
# -------------------------------
# Check if Source column exists
if "Source" in df.columns:
    source_counts = df["Source"].value_counts()

    plt.figure()
    plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%')
    plt.title("Pollution Source Distribution")
    plt.show()
else:
    print("⚠️ Source column not found. Run model.py first.")
    

   ###### app.py #########

   

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("enviro_data_with_distance.csv")
df = df.dropna()

# -------------------------------
# SOURCE LABEL
# -------------------------------
def label_source(row):
    if row["Dist_Industrial"] < 1000 and row["SO2"] > 5:
        return "Industrial"
    elif row["Dist_Farmland"] < 1000 and row["PM2.5"] > 20:
        return "Agricultural"
    elif row["Dist_Road"] < 500 and row["NO2"] > 10:
        return "Vehicular"
    else:
        return "Natural"

df["Source"] = df.apply(label_source, axis=1)

# -------------------------------
# MODEL
# -------------------------------
X = df[[
    "Dist_Road", "Dist_Industrial", "Dist_Farmland",
    "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)",
    "PM2.5", "PM10", "NO2", "SO2"
]]

y = df["Source"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Controls")

cities = df["City"].unique()
selected_city = st.sidebar.selectbox("Select City", cities)

city_df = df[df["City"] == selected_city]

# -------------------------------
# TITLE
# -------------------------------
st.title("EnviroScan")

# -------------------------------
# BAR GRAPH (SMALL)
# -------------------------------
st.subheader("PM2.5 by City")

city_pm25 = df.groupby("City")["PM2.5"].mean()

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(city_pm25.index, city_pm25.values)
ax.set_xticklabels(city_pm25.index, rotation=45)
ax.set_ylabel("PM2.5")

st.pyplot(fig, use_container_width=False)

# -------------------------------
# SCATTER (SMALL)
# -------------------------------
st.subheader("PM2.5 vs NO2")

city_data = df.groupby("City")[["PM2.5", "NO2"]].mean()

fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(city_data["PM2.5"], city_data["NO2"])

for i, city in enumerate(city_data.index):
    ax.text(city_data["PM2.5"].iloc[i], city_data["NO2"].iloc[i], city, fontsize=8)

ax.set_xlabel("PM2.5")
ax.set_ylabel("NO2")

st.pyplot(fig, use_container_width=False)

# -------------------------------
# PIE CHART (SMALL)
# -------------------------------
st.subheader(f"Source Distribution - {selected_city}")

source_counts = city_df["Source"].value_counts()

fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%')

st.pyplot(fig, use_container_width=False)

# -------------------------------
# INPUT
# -------------------------------
st.subheader("Input")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.slider("PM2.5", 0.0, 500.0, 50.0)
    no2 = st.slider("NO2", 0.0, 300.0, 30.0)
    temp = st.slider("Temperature", 0.0, 50.0, 25.0)

with col2:
    pm10 = st.slider("PM10", 0.0, 1000.0, 100.0)
    so2 = st.slider("SO2", 0.0, 100.0, 10.0)
    humidity = st.slider("Humidity", 0, 100, 40)

# -------------------------------
# PREDICT
# -------------------------------
if st.button("Predict"):

    input_data = [[
        200, 500, 800,
        temp, humidity, 3,
        pm25, pm10, no2, so2
    ]]

    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)[0]

    st.write("Predicted Source:", result)
