import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# ============================
#  PATH SETUP
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "outputs", "predictions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
MAP_PATH = os.path.join(BASE_DIR, "outputs", "pollution_map.html")

# ============================
#  PAGE CONFIG
# ============================
st.set_page_config(page_title="AI EnviroScan", layout="wide")

st.markdown(
    "<h1 style='text-align:center;color:#00ADB5;'> AI EnviroScan Dashboard</h1>",
    unsafe_allow_html=True
)

# ============================
#  LOAD DATA
# ============================
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

df['timestamp'] = pd.to_datetime(df['timestamp'])

# ============================
#  SOURCE MAPPING
# ============================
source_map = {
    0: "Agricultural 🌾",
    1: "Burning 🔥",
    2: "Industrial 🏭",
    3: "Natural 🌿",
    4: "Vehicular 🚗"
}

# ============================
# SIDEBAR NAVIGATION (SLIDE UI)
# ============================
page = st.sidebar.radio(" Navigate", [
    " Dashboard",
    " Prediction",
    " Map",
    " Download"
])

# ============================
#  DASHBOARD PAGE
# ============================
if page == " Dashboard":

    st.sidebar.header("🔍 Filters")

    city = st.sidebar.selectbox("Select City", df['city'].unique())

    filtered_df = df[df['city'] == city]

    st.subheader(f" City: {city}")

    col1, col2, col3 = st.columns(3)
    col1.metric("PM2.5", f"{filtered_df['pm2_5'].mean():.2f}")
    col2.metric("NO2", f"{filtered_df['no2'].mean():.2f}")
    col3.metric("CO", f"{filtered_df['co'].mean():.2f}")

    # Alert
    avg_pm = filtered_df['pm2_5'].mean()

    if avg_pm > 70:
        st.error("🚨 Severe Pollution")
    elif avg_pm > 40:
        st.warning("⚠️ Moderate Pollution")
    else:
        st.success("✅ Safe")

    # Graph
    fig = px.line(filtered_df, x='timestamp', y=['pm2_5','pm10','no2'])
    st.plotly_chart(fig, use_container_width=True)

    # Pie
    source_counts = filtered_df['predicted_source'].value_counts().reset_index()
    source_counts.columns = ['source', 'count']

    fig2 = px.pie(source_counts, names='source', values='count')
    st.plotly_chart(fig2)

# ============================
#  PREDICTION PAGE
# ============================
elif page == " Prediction":

    st.subheader(" Smart Prediction")

    mode = st.radio("Select Mode", ["Quick", "Advanced"])

    city = st.selectbox("Select City", df['city'].unique())
    city_data = df[df['city'] == city].iloc[-1]

    if mode == "Quick":
        pm2_5 = st.slider("PM2.5", float(df.pm2_5.min()), float(df.pm2_5.max()), float(df.pm2_5.mean()))
        no2 = st.slider("NO2", float(df.no2.min()), float(df.no2.max()), float(df.no2.mean()))
        co = st.slider("CO", float(df.co.min()), float(df.co.max()), float(df.co.mean()))

        pm10 = df['pm10'].mean()
        so2 = df['so2'].mean()
        o3 = df['o3'].mean()
        temp = df['temp'].mean()
        humidity = df['humidity'].mean()
        wind = df['wind'].mean()

    else:
        pm2_5 = st.slider("PM2.5", float(df.pm2_5.min()), float(df.pm2_5.max()), float(df.pm2_5.mean()))
        pm10 = st.slider("PM10", float(df.pm10.min()), float(df.pm10.max()), float(df.pm10.mean()))
        no2 = st.slider("NO2", float(df.no2.min()), float(df.no2.max()), float(df.no2.mean()))
        co = st.slider("CO", float(df.co.min()), float(df.co.max()), float(df.co.mean()))
        so2 = st.slider("SO2", float(df.so2.min()), float(df.so2.max()), float(df.so2.mean()))
        o3 = st.slider("O3", float(df.o3.min()), float(df.o3.max()), float(df.o3.mean()))
        temp = st.slider("Temperature", float(df.temp.min()), float(df.temp.max()), float(df.temp.mean()))
        humidity = st.slider("Humidity", float(df.humidity.min()), float(df.humidity.max()), float(df.humidity.mean()))
        wind = st.slider("Wind", float(df.wind.min()), float(df.wind.max()), float(df.wind.mean()))

    if st.button(" Predict"):

        input_data = np.array([[ 
            city_data['lat'], city_data['lon'],
            pm2_5, pm10, no2, co, so2, o3,
            temp, humidity, wind,
            df['dist_road'].mean(),
            df['dist_industry'].mean(),
            df['dist_dump'].mean(),
            df['dist_farm'].mean()
        ]])

        pred_num = model.predict(input_data)[0]
        pred = source_map.get(pred_num)

        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(input_data)) * 100
        else:
            confidence = 95

        st.success(f" Predicted Source: {pred}")
        st.progress(int(confidence))
        st.info(f"Confidence: {confidence:.2f}%")

        st.info(" Model Used: Best ML Model (Auto Selected)")

# ============================
#  MAP PAGE
# ============================
elif page == " Map":

    st.subheader(" Pollution Map")

    with open(MAP_PATH, 'r', encoding='utf-8') as f:
        map_html = f.read()

    st.components.v1.html(map_html, height=500)

# ============================
#  DOWNLOAD PAGE
# ============================
elif page == " Download":

    st.subheader(" Download Data")

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="pollution_data.csv",
        mime='text/csv'
    )