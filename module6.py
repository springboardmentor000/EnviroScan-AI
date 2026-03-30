# ==========================================
# MODULE 6: FINAL PROFESSIONAL DASHBOARD
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="AI Pollution Dashboard", layout="wide")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("labeled_environment_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

@st.cache_resource
def load_model():
    return (
        joblib.load("pollution_source_model.pkl"),
        joblib.load("label_encoder.pkl"),
        joblib.load("city_encoder.pkl")
    )

model, label_encoder, city_encoder = load_model()

# ==============================
# TITLE
# ==============================
st.title("🌍 AI Pollution Monitoring Dashboard")

# ==============================
# SIDEBAR (FINAL)
# ==============================
st.sidebar.header("🌫 Pollution Controls")
st.sidebar.markdown("Select location to view pollution insights")
st.sidebar.divider()

state = st.sidebar.selectbox("Select State", sorted(df["state"].unique()))
cities = df[df["state"] == state]["city"].unique()
city = st.sidebar.selectbox("Select City", sorted(cities))

# ==============================
# STATIC INPUT (IMPORTANT)
# ==============================
base_row = df[df["city"] == city].iloc[0]

pm25 = base_row["pm25"]
no2 = base_row["no2"]
so2 = base_row["so2"]
dist_road = base_row["dist_to_road"]

# ==============================
# AUTO FEATURES (MODEL REQUIREMENT)
# ==============================
pm10 = df["pm10"].mean()
co = df["co"].mean()
o3 = df["o3"].mean()
temperature = df["temperature"].mean()
humidity = df["humidity"].mean()
pressure = df["pressure"].mean()
wind_speed = df["wind_speed"].mean()
dist_industry = df["dist_to_industry"].mean()
dist_dump = df["dist_to_dump"].mean()

# ==============================
# ENCODE CITY
# ==============================
city_encoded = city_encoder.transform([city])[0]

# ==============================
# MODEL INPUT
# ==============================
features = [
    "pm25","pm10","no2","co","so2","o3",
    "temperature","humidity","pressure","wind_speed",
    "dist_to_road","dist_to_industry","dist_to_dump",
    "city_encoded"
]

input_df = pd.DataFrame([[  
    pm25, pm10, no2, co, so2, o3,
    temperature, humidity, pressure, wind_speed,
    dist_road, dist_industry, dist_dump,
    city_encoded
]], columns=features)

# ==============================
# KPI CARDS
# ==============================
st.subheader("📊 Air Quality Overview")

c1, c2, c3 = st.columns(3)
c1.metric("PM2.5", round(pm25, 2))
c2.metric("NO2", round(no2, 2))
c3.metric("SO2", round(so2, 2))

# ==============================
# AI PREDICTION (STATIC)
# ==============================
st.subheader("🧠 AI Prediction (Static)")

pred = model.predict(input_df)
result = label_encoder.inverse_transform(pred)[0]
confidence = np.max(model.predict_proba(input_df))

if pm25 > 0.7:
    st.error(f"🔴 High Pollution - {result}")
elif pm25 > 0.4:
    st.warning(f"🟠 Moderate Pollution - {result}")
else:
    st.success(f"🟢 Safe - {result}")

st.write(f"Confidence: {round(confidence * 100, 2)}%")

# ==============================
# GRAPHS SECTION
# ==============================
col1, col2 = st.columns(2)

# TREND GRAPH
with col1:
    st.subheader("📈 PM2.5 Trend")

    city_df = df[df["city"] == city]

    color = "red" if pm25 > 0.7 else "green"

    fig = px.line(city_df.sort_values("timestamp"),
                  x="timestamp", y="pm25")

    fig.update_traces(line=dict(color=color, width=4))
    st.plotly_chart(fig, use_container_width=True)

# PIE CHART
with col2:
    st.subheader("🥧 Source Distribution")

    fig2 = px.pie(city_df, names="source_label",
                  title=f"{city} Sources")

    st.plotly_chart(fig2, use_container_width=True)

# ==============================
# MAP (MODULE 5 STYLE)
# ==============================
st.subheader("🗺 Pollution Map")

def safe_encode(c):
    if c in city_encoder.classes_:
        return city_encoder.transform([c])[0]
    return -1

df["city_encoded"] = df["city"].apply(safe_encode)

X = df[features]
pred_map = model.predict(X)
df["predicted_source"] = label_encoder.inverse_transform(pred_map)

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

# HEATMAP
HeatMap([
    [row["latitude"], row["longitude"], row["pm25"]]
    for _, row in df.iterrows()
]).add_to(m)

# MARKERS
for _, row in df.iterrows():
    folium.Marker(
        [row["latitude"], row["longitude"]],
        popup=f"{row['city']} | {row['predicted_source']} | PM2.5: {row['pm25']}"
    ).add_to(m)

map_data = st_folium(m, width=1200, height=500)

# CLICK FEATURE
if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]

    st.success(f"📍 Selected: {lat:.4f}, {lon:.4f}")

    df["dist"] = ((df["latitude"] - lat)**2 +
                  (df["longitude"] - lon)**2)

    nearest = df.loc[df["dist"].idxmin()]

    st.write("Nearest Data:")
    st.write(nearest[["city", "pm25", "predicted_source"]])

# ==============================
# DOWNLOAD REPORT
# ==============================
st.subheader("📥 Download Report")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    file_name="pollution_report.csv",
    mime="text/csv"
)