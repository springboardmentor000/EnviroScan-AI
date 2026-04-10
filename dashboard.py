import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import requests

# ============================
# PATH SETUP
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "outputs", "predictions.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_model.pkl")
MAP_PATH = os.path.join(ROOT_DIR, "outputs", "pollution_map.html")

# fallback
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(BASE_DIR, "outputs", "predictions.csv")

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

if not os.path.exists(MAP_PATH):
    MAP_PATH = os.path.join(BASE_DIR, "outputs", "pollution_map.html")

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="AI EnviroScan", layout="wide")

st.markdown(
    "<h1 style='text-align:center;color:#00ADB5;'>🌍 AI EnviroScan Dashboard</h1>",
    unsafe_allow_html=True
)

# ============================
# LOAD DATA
# ============================
if not os.path.exists(DATA_PATH):
    st.error(f"❌ File not found: {DATA_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

df['timestamp'] = pd.to_datetime(df['timestamp'])

# ============================
# SOURCE MAP
# ============================
source_map = {
    0: "Agricultural 🌾",
    1: "Burning 🔥",
    2: "Industrial 🏭",
    3: "Natural 🌿",
    4: "Vehicular 🚗"
}

# ============================
# LIVE DATA FUNCTIONS
# ============================
API_KEY = "0905378d18d6b90dcd9e1054ed0c2586"

def get_live_pollution(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    res = requests.get(url).json()
    comp = res['list'][0]['components']
    return {
        "pm2_5": comp['pm2_5'],
        "pm10": comp['pm10'],
        "no2": comp['no2'],
        "co": comp['co'],
        "so2": comp['so2'],
        "o3": comp['o3']
    }

def get_live_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    res = requests.get(url).json()
    return {
        "temp": res['main']['temp'],
        "humidity": res['main']['humidity'],
        "pressure": res['main']['pressure'],
        "wind": res['wind']['speed']
    }

# ============================
# SIDEBAR
# ============================
page = st.sidebar.radio(" Navigate", [
    "📊 Dashboard",
    "🤖 Prediction",
    "🌐 Live Data",
    "🗺️ Map",
    "📥 Download"
])

# ============================
# DASHBOARD
# ============================
if page == "📊 Dashboard":

    st.sidebar.header(" Filters")

    city = st.sidebar.selectbox("Select City", df['city'].unique())
    filtered_df = df[df['city'] == city]

    st.subheader(f" City: {city}")

    col1, col2, col3 = st.columns(3)
    col1.metric("PM2.5", f"{filtered_df['pm2_5'].mean():.2f}")
    col2.metric("NO2", f"{filtered_df['no2'].mean():.2f}")
    col3.metric("CO", f"{filtered_df['co'].mean():.2f}")

    avg_pm = filtered_df['pm2_5'].mean()

    if avg_pm > 70:
        st.error("🚨 Severe Pollution")
    elif avg_pm > 40:
        st.warning("⚠️ Moderate Pollution")
    else:
        st.success("✅ Safe")

    fig = px.line(filtered_df, x='timestamp', y=['pm2_5','pm10','no2'])
    st.plotly_chart(fig, use_container_width=True)

    source_counts = filtered_df['predicted_source'].value_counts().reset_index()
    source_counts.columns = ['source', 'count']

    fig2 = px.pie(source_counts, names='source', values='count')
    st.plotly_chart(fig2)

# ============================
# PREDICTION
# ============================
elif page == "🤖 Prediction":

    st.subheader("🤖 Smart Prediction")

    mode = st.radio("Select Mode", ["Quick", "Advanced"])

    city = st.selectbox("Select City", df['city'].unique())
    city_data = df[df['city'] == city].iloc[-1]

    if mode == "Quick":
        pm2_5 = st.slider("PM2.5", float(df.pm2_5.min()), float(df.pm2_5.max()), float(df.pm2_5.mean()))
        no2 = st.slider("NO2", float(df.no2.min()), float(df.no2.max()), float(df.no2.mean()))
        co = st.slider("CO", float(df.co.min()), float(df.co.max()), float(df.co.mean()))

        pm10 = city_data['pm10']
        so2 = city_data['so2']
        o3 = city_data['o3']
        temp = city_data['temp']
        humidity = city_data['humidity']
        wind = city_data['wind']
        dist_road = city_data['dist_road']
        dist_industry = city_data['dist_industry']
        dist_dump = city_data['dist_dump']
        dist_farm = city_data['dist_farm']

    else:
        pm2_5 = st.slider("PM2.5", float(df.pm2_5.min()), float(df.pm2_5.max()), float(df.pm2_5.mean()))
        pm10 = st.slider("PM10", float(df.pm10.min()), float(df.pm10.max()), float(df.pm10.mean()))
        no2 = st.slider("NO2", float(df.no2.min()), float(df.no2.max()), float(df.no2.mean()))
        co = st.slider("CO", float(df.co.min()), float(df.co.max()), float(df.co.mean()))
        so2 = st.slider("SO2", float(df.so2.min()), float(df.so2.max()), float(df.so2.mean()))
        o3 = st.slider("O3", float(df.o3.min()), float(df.o3.max()), float(df.o3.mean()))
        temp = st.slider("Temperature", float(df.temp.min()), float(df.temp.max()), float(df.temp.mean()))
        humidity = st.slider("Humidity", float(df.humidity.min()), float(df.humidity.max()), float(df.humidity.mean()))
        wind = st.slider("Wind Speed", float(df.wind.min()), float(df.wind.max()), float(df.wind.mean()))
        dist_road = st.slider("Distance to Road", float(df.dist_road.min()), float(df.dist_road.max()), float(df.dist_road.mean()))
        dist_industry = st.slider("Distance to Industry", float(df.dist_industry.min()), float(df.dist_industry.max()), float(df.dist_industry.mean()))
        dist_dump = st.slider("Distance to Dump", float(df.dist_dump.min()), float(df.dist_dump.max()), float(df.dist_dump.mean()))
        dist_farm = st.slider("Distance to Farm", float(df.dist_farm.min()), float(df.dist_farm.max()), float(df.dist_farm.mean()))

    if st.button("🚀 Predict"):

        input_data = np.array([[ 
            city_data['lat'], city_data['lon'],
            pm2_5, pm10, no2, co, so2, o3,
            temp, humidity, wind,
            dist_road, dist_industry, dist_dump, dist_farm
        ]])

        pred_num = model.predict(input_data)[0]
        pred = source_map.get(pred_num, "Unknown")

        confidence = (
            np.max(model.predict_proba(input_data)) * 100
            if hasattr(model, "predict_proba") else 95
        )

        st.success(f"🌟 Predicted Source: {pred}")
        st.progress(int(confidence))
        st.info(f"Confidence: {confidence:.2f}%")

# ============================
# LIVE DATA PAGE
# ============================
elif page == "🌐 Live Data":

    st.subheader("🌐 Real-Time Pollution and Weather Data")

    city = st.selectbox("Select City", df['city'].unique())
    city_data = df[df['city'] == city].iloc[-1]

    if st.button("Fetch Live Data"):

        pollution = get_live_pollution(city_data['lat'], city_data['lon'])
        weather = get_live_weather(city_data['lat'], city_data['lon'])

        st.success("Live data fetched successfully")

        st.markdown("### Pollution Data")
        col1, col2, col3 = st.columns(3)
        col1.metric("PM2.5", pollution['pm2_5'])
        col2.metric("PM10", pollution['pm10'])
        col3.metric("NO2", pollution['no2'])

        col4, col5, col6 = st.columns(3)
        col4.metric("CO", pollution['co'])
        col5.metric("SO2", pollution['so2'])
        col6.metric("O3", pollution['o3'])

        st.markdown("### Weather Data")
        col7, col8, col9 = st.columns(3)
        col7.metric("Temperature (°C)", weather['temp'])
        col8.metric("Humidity", weather['humidity'])
        col9.metric("Wind Speed", weather['wind'])

        st.metric("Pressure", weather['pressure'])

# ============================
# MAP
# ============================
elif page == "🗺️ Map":

    st.subheader("🗺️ Pollution Map")

    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, 'r', encoding='utf-8') as f:
            map_html = f.read()
        st.components.v1.html(map_html, height=500)
    else:
        st.error("❌ Map file not found")

# ============================
# DOWNLOAD
# ============================
elif page == "📥 Download":

    st.subheader("📥 Download Data")

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="pollution_data.csv",
        mime='text/csv'
    )