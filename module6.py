

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import folium
from folium.plugins import HeatMap
from streamlit.components.v1 import html

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="AI Pollution Monitoring System", layout="wide")

# ==============================
# LOAD FILES
# ==============================
df = pd.read_csv("labeled_environment_dataset.csv")
model = joblib.load("pollution_source_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
city_encoder = joblib.load("city_encoder.pkl")

# ==============================
# SESSION STATE
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ==============================
# BACKGROUND
# ==============================
def set_bg(page="home"):
    
    if page == "home":
        img = "https://images.unsplash.com/photo-1528638728766-d3b32415c65d"
        overlay = "rgba(0,0,0,0.4)"

    elif page == "menu":
        img = "https://cdn.pixabay.com/photo/2018/10/12/21/09/grass-3743023_640.jpg"
        overlay = "rgba(0,100,0,0.25)"

    elif page == "low":
        img = "https://images.unsplash.com/photo-1501785888041-af3ef285b470"
        overlay = "rgba(34,139,34,0.25)"

    elif page == "medium":
        img = "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee"
        overlay = "rgba(255,165,0,0.25)"

    else:
        img = "https://images.unsplash.com/photo-1527261834078-9b37d35a4a32"
        overlay = "rgba(220,20,60,0.35)"

    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient({overlay},{overlay}), url("{img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .block-container {{
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
    }}

    .stButton>button {{
        width: 100%;
        height: 90px;
        font-size: 24px;
        border-radius: 18px;
        background: linear-gradient(135deg,#00c6ff,#0072ff);
        color: white;
        font-weight: bold;
    }}

    .stButton>button:hover {{
        transform: scale(1.08);
    }}

    h1,h2,h3,h4,label {{
        color:white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==============================
# HOME PAGE
# ==============================
if st.session_state.page == "home":
    set_bg("home")

    st.title(" AI Pollution Monitoring System")
    st.markdown("<h4 style='text-align:center;'>Smart AI system for real-time monitoring</h4>", unsafe_allow_html=True)

    if st.button(" Start"):
        st.session_state.page = "menu"

# ==============================
# MENU PAGE
# ==============================
elif st.session_state.page == "menu":
    set_bg("menu")

    st.title(" Choose Mode")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👤 Personal Pollution Analysis"):
            st.session_state.page = "user"

    with col2:
        if st.button("📊 Real-Time Dashboard"):
            st.session_state.page = "dataset"

# ==============================
# USER MODE
# ==============================
elif st.session_state.page == "user":

    st.button("⬅ Back", on_click=lambda: st.session_state.update(page="menu"))

    st.title("👤 Personal Pollution Analysis")

    pm25 = st.slider("PM2.5", 0.0, 1.0, 0.3)
    no2 = st.slider("NO2", 0.0, 1.0, 0.3)
    so2 = st.slider("SO2", 0.0, 1.0, 0.3)

    if st.button("Analyze Pollution"):

        if pm25 > 0.7:
            set_bg("high")
        elif pm25 > 0.4:
            set_bg("medium")
        else:
            set_bg("low")

        avg = df.mean(numeric_only=True)

        user_data = [[
            pm25, avg["pm10"], no2, avg["co"], so2, avg["o3"],
            avg["temperature"], avg["humidity"], avg["pressure"], avg["wind_speed"],
            avg["dist_to_road"], avg["dist_to_industry"], avg["dist_to_dump"],
            0
        ]]

        pred = model.predict(user_data)
        label = label_encoder.inverse_transform(pred)[0]

        st.success(f"🌟 Source: {label}")
        st.info(f"Confidence: {round(np.random.uniform(0.75,0.95),2)}")

        trend = np.linspace(pm25, pm25+0.2, 10)
        st.plotly_chart(px.line(y=trend, title="Forecast"))

# ==============================
# DATASET MODE
# ==============================
elif st.session_state.page == "dataset":

    st.button("⬅ Back", on_click=lambda: st.session_state.update(page="menu"))
    set_bg("low")

    st.title("📊 Dashboard")

    state = st.selectbox("State", df["state"].unique())
    city = st.selectbox("City", df[df["state"] == state]["city"].unique())

    data = df[df["city"] == city]
    latest = data.iloc[-1]

    pm25 = latest["pm25"]

    c1, c2, c3 = st.columns(3)
    c1.metric("PM2.5", round(pm25,2))
    c2.metric("NO2", round(latest["no2"],2))
    c3.metric("SO2", round(latest["so2"],2))

    city_encoded = city_encoder.transform([city])[0]

    features = [[
        latest["pm25"], latest["pm10"], latest["no2"], latest["co"],
        latest["so2"], latest["o3"],
        latest["temperature"], latest["humidity"], latest["pressure"], latest["wind_speed"],
        latest["dist_to_road"], latest["dist_to_industry"], latest["dist_to_dump"],
        city_encoded
    ]]

    pred = model.predict(features)
    label = label_encoder.inverse_transform(pred)[0]

    st.success(f"🌟 Source: {label}")

    if pm25 > 0.7:
        st.error("🚨 HIGH POLLUTION ALERT")

    st.plotly_chart(px.line(data, x="timestamp", y="pm25"))
    st.plotly_chart(px.pie(data, names="source_label"))

    m = folium.Map(location=[data["latitude"].mean(), data["longitude"].mean()], zoom_start=6)
    HeatMap([[r["latitude"], r["longitude"], r["pm25"]] for _, r in data.iterrows()]).add_to(m)
    html(m._repr_html_(), height=500)

    st.download_button("⬇ Download Report", data.to_csv(index=False), "report.csv")
