import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import os
from datetime import datetime

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="EnviroScan — Pollution Dashboard",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F5C99;
        text-align: center;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-box {
        background: #f0f4ff;
        border-left: 5px solid #1F5C99;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
    }
    .alert-red {
        background: #ffe5e5;
        border-left: 5px solid #E74C3C;
        padding: 12px 16px;
        border-radius: 6px;
        color: #c0392b;
        font-weight: bold;
    }
    .alert-green {
        background: #e8f8e8;
        border-left: 5px solid #2ECC71;
        padding: 12px 16px;
        border-radius: 6px;
        color: #1a7a1a;
        font-weight: bold;
    }
    .source-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA AND MODELS
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Hyderabad_pollution_labeled.csv")
    real_df = df[~df["station_id"].astype(str).str.startswith("SIM")].copy()
    for col in ["pm25","pm10","no2","so2","o3","co"]:
        real_df[col] = real_df[col].fillna(real_df[col].median())
    return real_df

@st.cache_resource
def load_models():
    model = joblib.load("models/random_forest.pkl")
    le    = joblib.load("models/label_encoder.pkl")
    return model, le

df    = load_data()
model, le = load_models()

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------
SOURCE_COLORS = {
    "Vehicular":    "#E74C3C",
    "Industrial":   "#3498DB",
    "Agricultural": "#2ECC71",
    "Burning":      "#F39C12",
    "Natural":      "#9B59B6",
}
FEATURES = [
    "pm25","pm10","no2","so2","o3","co",
    "temp","humidity","wind_speed","wind_direction",
    "road_count","has_major_road",
    "industrial_zone_count","near_industrial_zone",
    "dump_site_count","near_dump_site",
    "agricultural_count","near_agricultural_area",
    "hour","month"
]

def aqi_color(pm25):
    if pm25 <= 30:    return "#2ECC71"
    elif pm25 <= 60:  return "#F1C40F"
    elif pm25 <= 90:  return "#F39C12"
    elif pm25 <= 120: return "#E74C3C"
    elif pm25 <= 250: return "#8E44AD"
    else:             return "#2C3E50"

def aqi_label(pm25):
    if pm25 <= 30:    return "Good"
    elif pm25 <= 60:  return "Satisfactory"
    elif pm25 <= 90:  return "Moderate"
    elif pm25 <= 120: return "Poor"
    elif pm25 <= 250: return "Very Poor"
    else:             return "Severe"

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/environment.png", width=80)
st.sidebar.title("EnviroScan")
st.sidebar.markdown("*AI-Powered Pollution Source Identifier*")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Dashboard",
    "Maps",
    "Predict Source",
    "Station Analysis",
    "Alerts",
    "Download Report"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}")
st.sidebar.markdown("Hyderabad, Telangana")

# -------------------------------------------------
# PAGE 1: DASHBOARD
# -------------------------------------------------
if page == "Dashboard":
    st.markdown('<div class="main-title">EnviroScan</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI-Powered Pollution Source Identifier — Hyderabad, India</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stations Monitored", len(df))
    with col2:
        avg_pm25 = round(df["pm25"].mean(), 1)
        st.metric("Avg PM2.5", f"{avg_pm25} µg/m³", delta=f"{aqi_label(avg_pm25)}")
    with col3:
        worst = df.loc[df["pm25"].idxmax(), "station_name"].split(",")[0]
        st.metric("Most Polluted", worst)
    with col4:
        best = df.loc[df["pm25"].idxmin(), "station_name"].split(",")[0]
        st.metric("Cleanest Station", best)

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Station Summary")
        summary = df[["station_name","pm25","no2","so2","co","pollution_source","source_confidence"]].copy()
        summary["aqi"]        = summary["pm25"].apply(aqi_label)
        summary["confidence"] = (summary["source_confidence"] * 100).astype(int).astype(str) + "%"
        summary = summary.rename(columns={
            "station_name": "Station", "pm25": "PM2.5",
            "no2": "NO2", "so2": "SO2", "co": "CO",
            "pollution_source": "Source", "aqi": "AQI"
        })
        st.dataframe(summary[["Station","PM2.5","NO2","SO2","CO","Source","AQI","confidence"]],
                     use_container_width=True, height=400)

    with col_right:
        st.subheader("Source Distribution")
        source_counts = df["pollution_source"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        colors = [SOURCE_COLORS.get(s, "#999") for s in source_counts.index]
        ax.pie(source_counts.values, labels=source_counts.index,
               autopct="%1.0f%%", colors=colors, startangle=140)
        ax.set_title("Pollution Sources")
        st.pyplot(fig)
        plt.close()

    st.subheader("PM2.5 Levels Across Stations")
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    bar_colors = [aqi_color(v) for v in df["pm25"]]
    names = [n.split(",")[0].split("-")[0].strip() for n in df["station_name"]]
    bars  = ax2.bar(names, df["pm25"], color=bar_colors, edgecolor="white")
    ax2.axhline(y=60,  color="#F39C12", linestyle="--", alpha=0.7, label="Moderate threshold (60)")
    ax2.axhline(y=90,  color="#E74C3C", linestyle="--", alpha=0.7, label="Poor threshold (90)")
    ax2.set_ylabel("PM2.5 (µg/m³)")
    ax2.set_title("PM2.5 Levels by Station")
    ax2.legend(fontsize=8)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    for bar, val in zip(bars, df["pm25"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.0f}", ha="center", fontsize=7)
    st.pyplot(fig2)
    plt.close()

# -------------------------------------------------
# PAGE 2: MAPS
# -------------------------------------------------
elif page == "Maps":
    st.title("Interactive Pollution Maps")
    st.markdown("Explore pollution levels and sources across Hyderabad.")

    map_type = st.selectbox("Select Map", [
        "PM2.5 Heatmap",
        "Pollution Source Map",
        "Combined Map"
    ])

    m = folium.Map(location=[17.3850, 78.4867], zoom_start=11, tiles="CartoDB positron")

    if map_type in ["PM2.5 Heatmap", "Combined Map"]:
        heat_data = [[r["latitude"], r["longitude"], r["pm25"]] for _, r in df.iterrows()]
        HeatMap(heat_data, radius=40, blur=25,
                gradient={"0.2":"blue","0.4":"lime","0.6":"yellow","0.8":"orange","1.0":"red"}
                ).add_to(m)

    if map_type in ["Pollution Source Map", "Combined Map"]:
        for _, row in df.iterrows():
            source = row["pollution_source"]
            color  = SOURCE_COLORS.get(source, "#999")
            popup_html = f"""
            <div style='font-family:Arial;width:200px;'>
                <b>{row['station_name']}</b><br>
                <span style='background:{color};color:white;padding:2px 8px;border-radius:4px;'>
                    {source}</span><br>
                PM2.5: {row['pm25']} µg/m³ — {aqi_label(row['pm25'])}<br>
                NO2: {round(row['no2'],1)} | SO2: {round(row['so2'],1)}<br>
                Confidence: {int(row['source_confidence']*100)}%
            </div>
            """
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=12, color="white", weight=2,
                fill=True, fill_color=color, fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=230),
                tooltip=f"{source} | {row['station_name'].split(',')[0]}"
            ).add_to(m)

    st_folium(m, width=1100, height=550)

# -------------------------------------------------
# PAGE 3: PREDICT SOURCE
# -------------------------------------------------
elif page == "Predict Source":
    st.title("Predict Pollution Source")
    st.markdown("Enter pollution and weather values to predict the source.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Pollutants")
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 60.0)
        pm10 = st.slider("PM10 (µg/m³)",  0.0, 500.0, 100.0)
        no2  = st.slider("NO2 (µg/m³)",   0.0, 200.0, 30.0)
        so2  = st.slider("SO2 (µg/m³)",   0.0, 100.0, 10.0)
        o3   = st.slider("O3 (µg/m³)",    0.0, 200.0, 40.0)
        co   = st.slider("CO (mg/m³)",     0.0, 1500.0, 1.0)

    with col2:
        st.subheader("Weather")
        temp      = st.slider("Temperature (°C)",   10.0, 45.0, 26.0)
        humidity  = st.slider("Humidity (%)",         10,  100,  40)
        wind_spd  = st.slider("Wind Speed (m/s)",    0.0, 15.0,  3.0)
        wind_dir  = st.slider("Wind Direction (°)",    0,  359,  180)

    with col3:
        st.subheader("Nearby Features")
        road_count    = st.number_input("Road Count",           0, 1000, 200)
        has_road      = st.selectbox("Has Major Road?",         [1, 0])
        ind_count     = st.number_input("Industrial Zones",     0, 100,  5)
        near_ind      = st.selectbox("Near Industrial Zone?",   [0, 1])
        dump_count    = st.number_input("Dump Sites",           0, 20,   0)
        near_dump     = st.selectbox("Near Dump Site?",         [0, 1])
        agri_count    = st.number_input("Agricultural Areas",   0, 50,   2)
        near_agri     = st.selectbox("Near Agricultural Area?", [0, 1])
        hour          = st.slider("Hour of Day", 0, 23, 12)
        month         = st.slider("Month", 1, 12, 3)

    if st.button("Predict Pollution Source", use_container_width=True):
        input_data = pd.DataFrame([[
            pm25, pm10, no2, so2, o3, co,
            temp, humidity, wind_spd, wind_dir,
            road_count, has_road,
            ind_count, near_ind,
            dump_count, near_dump,
            agri_count, near_agri,
            hour, month
        ]], columns=FEATURES)

        pred_encoded = model.predict(input_data)[0]
        pred_proba   = model.predict_proba(input_data)[0]
        pred_source  = le.inverse_transform([pred_encoded])[0]
        confidence   = round(pred_proba.max() * 100, 1)
        color        = SOURCE_COLORS.get(pred_source, "#999")

        st.markdown("---")
        st.subheader("Prediction Result")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown(f"""
            <div style='background:{color};color:white;padding:20px;border-radius:10px;text-align:center;'>
                <div style='font-size:1.5rem;font-weight:bold;'>{pred_source}</div>
                <div>Pollution Source</div>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            st.metric("Confidence", f"{confidence}%")
            st.metric("AQI Level", aqi_label(pm25))
        with col_r3:
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            classes = le.classes_
            bar_c   = [SOURCE_COLORS.get(c, "#999") for c in classes]
            ax3.barh(classes, pred_proba * 100, color=bar_c)
            ax3.set_xlabel("Probability (%)")
            ax3.set_title("Source Probabilities")
            st.pyplot(fig3)
            plt.close()

        if pm25 > 90:
            st.markdown(f"""<div class='alert-red'>
                ALERT: PM2.5 is {pm25} µg/m³ — {aqi_label(pm25)} level!
                Immediate action recommended for {pred_source} source.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='alert-green'>
                PM2.5 is {pm25} µg/m³ — {aqi_label(pm25)} — within acceptable range.
            </div>""", unsafe_allow_html=True)

# -------------------------------------------------
# PAGE 4: STATION ANALYSIS
# -------------------------------------------------
elif page == "Station Analysis":
    st.title("Station Analysis")

    station = st.selectbox("Select a Station", df["station_name"].tolist())
    row     = df[df["station_name"] == station].iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Station Info")
        st.markdown(f"**Station:** {row['station_name']}")
        st.markdown(f"**Location:** {row['latitude']:.4f}, {row['longitude']:.4f}")
        source = row["pollution_source"]
        st.markdown(f"**Predicted Source:** {source}")
        st.markdown(f"**Confidence:** {int(row['source_confidence']*100)}%")
        st.markdown(f"**AQI:** {aqi_label(row['pm25'])}")

        st.subheader("Weather")
        st.markdown(f"**Temperature:** {row['temp']}°C")
        st.markdown(f"**Humidity:** {row['humidity']}%")
        st.markdown(f"**Wind Speed:** {row['wind_speed']} m/s")
        st.markdown(f"**Wind Direction:** {row['wind_direction']}°")

    with col2:
        st.subheader("Pollutant Levels")
        pollutants = {"PM2.5": row["pm25"], "PM10": row["pm10"],
                      "NO2": row["no2"],   "SO2": row["so2"],
                      "O3": row["o3"],     "CO": row["co"]}
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        bar_c = [aqi_color(row["pm25"])] * len(pollutants)
        ax4.bar(pollutants.keys(), pollutants.values(),
                color=bar_c, edgecolor="white")
        ax4.set_title(f"Pollutant Levels — {station.split(',')[0]}")
        ax4.set_ylabel("Concentration")
        plt.xticks(rotation=20)
        st.pyplot(fig4)
        plt.close()

    st.subheader("Station Location")
    m4 = folium.Map(location=[row["latitude"], row["longitude"]], zoom_start=14)
    folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=station,
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m4)
    st_folium(m4, width=700, height=300)

# -------------------------------------------------
# PAGE 5: ALERTS
# -------------------------------------------------
elif page == "Alerts":
    st.title("Pollution Alerts")
    st.markdown("Stations where pollution crosses safe thresholds.")

    threshold = st.slider("Set PM2.5 Alert Threshold (µg/m³)", 30, 200, 60)

    alerts = df[df["pm25"] > threshold].sort_values("pm25", ascending=False)

    if len(alerts) == 0:
        st.success(f"No stations exceed {threshold} µg/m³. Air quality is acceptable!")
    else:
        st.error(f"{len(alerts)} station(s) exceed the {threshold} µg/m³ threshold!")
        for _, row in alerts.iterrows():
            source = row["pollution_source"]
            st.markdown(f"""
            <div class='alert-red'>
                <b>{row['station_name']}</b><br>
                PM2.5: <b>{row['pm25']} µg/m³</b> — {aqi_label(row['pm25'])}<br>
                Source: {source} | Confidence: {int(row['source_confidence']*100)}%
            </div><br>
            """, unsafe_allow_html=True)

    st.subheader("All Stations AQI Status")
    for _, row in df.sort_values("pm25", ascending=False).iterrows():
        col_a, col_b, col_c, col_d = st.columns([3, 1, 1, 1])
        with col_a:
            st.write(row["station_name"].split(",")[0])
        with col_b:
            st.write(f"{row['pm25']} µg/m³")
        with col_c:
            st.write(aqi_label(row["pm25"]))
        with col_d:
            st.write(row["pollution_source"])

# -------------------------------------------------
# PAGE 6: DOWNLOAD REPORT
# -------------------------------------------------
elif page == "Download Report":
    st.title("Download Pollution Report")
    st.markdown("Export station data as a CSV report.")

    report_df = df[["station_name","latitude","longitude","timestamp",
                    "pm25","pm10","no2","so2","o3","co",
                    "temp","humidity","wind_speed",
                    "pollution_source","source_confidence","aqi_category"]].copy()
    report_df["source_confidence"] = (report_df["source_confidence"] * 100).astype(int).astype(str) + "%"
    report_df.columns = ["Station","Lat","Lon","Timestamp",
                         "PM2.5","PM10","NO2","SO2","O3","CO",
                         "Temp(°C)","Humidity(%)","Wind(m/s)",
                         "Predicted Source","Confidence","AQI Category"]

    st.dataframe(report_df, use_container_width=True)

    csv = report_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV Report",
        data=csv,
        file_name=f"EnviroScan_Report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("---")
    st.markdown(f"**Total Stations:** {len(report_df)}")
    st.markdown(f"**Report Generated:** {datetime.now().strftime('%d %b %Y, %H:%M')}")
    st.markdown(f"**Location:** Hyderabad, Telangana, India")