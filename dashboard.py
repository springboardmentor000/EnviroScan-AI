import os
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="EnviroScan Dashboard", layout="wide")

# -----------------------------------
# SIMPLE CLEAN STYLING
# -----------------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
    background-color: #eef2f7;
}
.simple-header {
    background: white;
    padding: 18px 22px;
    border-radius: 16px;
    border: 1px solid #e6ebf2;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-bottom: 16px;
}
.simple-header h1 {
    margin: 0;
    font-size: 34px;
    color: #0f172a;
}
.simple-header p {
    margin: 6px 0 0 0;
    font-size: 17px;
    color: #475569;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD DATA
# -----------------------------------
df = pd.read_csv("data/processed/labeled_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp")

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = None
if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")

# -----------------------------------
# HELPER FUNCTION
# -----------------------------------
def get_aqi_status(pm25):
    if pm25 <= 12:
        return "Good"
    elif pm25 <= 35.4:
        return "Moderate"
    elif pm25 <= 55.4:
        return "Sensitive Group Risk"
    elif pm25 <= 150.4:
        return "Poor Air Quality"
    elif pm25 <= 250.4:
        return "Very Poor"
    return "Severe"

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.header("Dashboard Filters")

city_list = sorted(df["location"].dropna().unique().tolist())
pollutant_list = [c for c in ["pm2_5", "pm10", "no2", "so2", "o3", "co"] if c in df.columns]

selected_city = st.sidebar.selectbox("Select Location", city_list)
selected_pollutant = st.sidebar.selectbox("Select Pollutant", pollutant_list)

filtered_df = df[df["location"] == selected_city].copy()

if filtered_df.empty:
    st.warning("No data found for selected city.")
    st.stop()

latest_row = filtered_df.sort_values("timestamp").iloc[-1]
latest_pm25 = float(latest_row["pm2_5"])
aqi_status = get_aqi_status(latest_pm25)
last_updated = str(latest_row["timestamp"]).split(".")[0]

if len(filtered_df) > 1:
    prev_row = filtered_df.sort_values("timestamp").iloc[-2]
    pm25_delta = round(latest_row["pm2_5"] - prev_row["pm2_5"], 2)
    pm10_delta = round(latest_row.get("pm10", 0) - prev_row.get("pm10", 0), 2)
    temp_delta = round(latest_row.get("temperature", 0) - prev_row.get("temperature", 0), 1)
    hum_delta = round(latest_row.get("humidity", 0) - prev_row.get("humidity", 0), 1)
    wind_delta = round(latest_row.get("wind_speed", 0) - prev_row.get("wind_speed", 0), 1)
else:
    pm25_delta = pm10_delta = temp_delta = hum_delta = wind_delta = None

dominant_source = (
    latest_row["pollution_source"]
    if "pollution_source" in latest_row
    else "Unknown"
)

# -----------------------------------
# DAILY 10-DAY WEATHER SUMMARY
# -----------------------------------
filtered_df["date"] = filtered_df["timestamp"].dt.date

daily_weather = (
    filtered_df.groupby("date", as_index=False)
    .agg(
        temp_min=("temperature", "min"),
        temp_max=("temperature", "max"),
        temp_avg=("temperature", "mean"),
        humidity_avg=("humidity", "mean"),
        wind_avg=("wind_speed", "mean"),
        pm25_avg=("pm2_5", "mean"),
    )
)

daily_weather["date"] = pd.to_datetime(daily_weather["date"])
daily_weather = daily_weather.sort_values("date")
last_10_days = daily_weather.tail(10).copy()

# -----------------------------------
# SIMPLE TOP HEADER
# -----------------------------------
st.markdown(f"""
<div class="simple-header">
    <h1>EnviroScan Dashboard</h1>
    <p>Location: <b>{selected_city}</b></p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# TABS
# -----------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Air Quality", "Weather", "Map", "Prediction", "Raw Data"]
)

# -----------------------------------
# TAB 1: OVERVIEW
# -----------------------------------
with tab1:
    st.subheader(f"Overview - {selected_city}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PM2.5", round(latest_row["pm2_5"], 2), delta=pm25_delta, delta_color="inverse")
    c2.metric("PM10", round(latest_row.get("pm10", 0), 2), delta=pm10_delta, delta_color="inverse")
    c3.metric("Temperature", f"{round(latest_row.get('temperature', 0), 1)} °C", delta=f"{temp_delta} °C" if temp_delta is not None else None)
    c4.metric("Humidity", f"{round(latest_row.get('humidity', 0), 1)} %", delta=f"{hum_delta} %" if hum_delta is not None else None)
    c5.metric("Wind Speed", f"{round(latest_row.get('wind_speed', 0), 1)} m/s", delta=f"{wind_delta} m/s" if wind_delta is not None else None)

    c6, c7, c8 = st.columns(3)
    c6.metric("Air Condition", aqi_status)
    c7.metric("Likely Source", dominant_source)
    c8.metric("Last Updated", last_updated)

    st.markdown("---")

    st.subheader("Temperature Trend")
    fig_temp_top = px.line(
        filtered_df,
        x="timestamp",
        y="temperature",
        title="Temperature Over Time",
        markers=True
    )
    st.plotly_chart(fig_temp_top, use_container_width=True)

# -----------------------------------
# TAB 2: AIR QUALITY
# -----------------------------------
with tab2:
    st.subheader(f"Current Air Quality - {selected_city}")

    st.markdown("""
    <style>
    .pollutant-card {
        background: white; border-radius: 8px; padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 15px;
        display: flex; justify-content: space-between; align-items: center;
        border-left: 5px solid #ccc;
    }
    .pollutant-card.poor { border-left-color: #ef4444; }
    .pollutant-card.fair { border-left-color: #f59e0b; }
    .pollutant-card.excellent { border-left-color: #10b981; }
    
    .pollutant-info { max-width: 70%; }
    .pollutant-title { font-size: 18px; font-weight: bold; color: #1e293b; margin-bottom: 4px; }
    .pollutant-status { font-size: 14px; font-weight: bold; margin-bottom: 8px; text-transform: uppercase; }
    .pollutant-status.poor { color: #ef4444; }
    .pollutant-status.fair { color: #f59e0b; }
    .pollutant-status.excellent { color: #10b981; }
    .pollutant-desc { font-size: 14px; color: #475569; }
    .pollutant-value-box { text-align: right; }
    .pollutant-val { font-size: 28px; font-weight: 800; color: #0f172a; }
    .pollutant-unit { font-size: 14px; color: #64748b; }
    </style>
    """, unsafe_allow_html=True)

    aqi_col, desc_col = st.columns([1, 2])
    
    with aqi_col:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = latest_pm25,
            title = {'text': "Current AQI<br><span style='font-size:0.8em;color:gray'>Based on PM2.5</span>"},
            gauge = {
                'axis': {'range': [None, 300], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 50], 'color': '#10b981'},
                    {'range': [50, 100], 'color': '#f59e0b'},
                    {'range': [100, 150], 'color': '#f97316'},
                    {'range': [150, 200], 'color': '#ef4444'},
                    {'range': [200, 300], 'color': '#8b5cf6'}],
            }
        ))
        fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with desc_col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        status_color = "#10b981" if aqi_status in ["Good"] else "#f59e0b" if aqi_status in ["Moderate"] else "#ef4444"
        st.markdown(f"<h2 style='color: {status_color}; margin-bottom:0;'>{aqi_status}</h2>", unsafe_allow_html=True)
        pollutant_vals = {
            "PM2.5": latest_row.get("pm2_5", 0),
            "PM10": latest_row.get("pm10", 0),
            "NO2": latest_row.get("no2", 0),
            "SO2": latest_row.get("so2", 0),
            "O3": latest_row.get("o3", 0)
        }
        primary_pollutant = max(pollutant_vals, key=pollutant_vals.get) if pollutant_vals else "various pollutants"
        
        aqi_messages = {
            "Good": "Air quality is considered satisfactory, and air pollution poses little or no risk. It's a great day to be active outside!",
            "Moderate": "Air quality is acceptable; however, there may be a moderate health concern for a very small number of people who are unusually sensitive.",
            "Sensitive Group Risk": "Members of sensitive groups may experience health effects. The general public is not likely to be affected.",
            "Poor Air Quality": "Health effects can be immediately felt by sensitive groups. Healthy individuals may experience difficulty breathing and throat irritation with prolonged exposure. Limit outdoor activity.",
            "Very Poor": "Health warnings of emergency conditions are in effect. The entire population is more likely to be affected. Avoid outdoor activities.",
            "Severe": "Health alert: everyone may experience more serious health effects. Stay indoors and limit physical exertion."
        }
        base_msg = aqi_messages.get(aqi_status, "The air has reached a level of pollution typical for this AQI range.")
        
        desc_text = f"Right now in <b>{selected_city}</b>, the primary pollutant of concern is <b>{primary_pollutant}</b>, largely driven by <b>{str(dominant_source).lower()}</b> sources. {base_msg}"

        st.markdown(f"<p style='font-size: 16px; color: #475569;'>{desc_text}</p>", unsafe_allow_html=True)

    st.markdown("### Current Pollutants")

    def get_status_class(val, thresh_fair, thresh_poor):
        if val <= thresh_fair: return "excellent"
        elif val <= thresh_poor: return "fair"
        return "poor"

    pollutants_data = [
        {"name": "PM10", "val": latest_row.get("pm10", 0), "unit": "µg/m³", "fair": 50, "poor": 100, "desc": "Particulate Matter are inhalable pollutant particles with a diameter less than 10 micrometers."},
        {"name": "PM2.5", "val": latest_row.get("pm2_5", 0), "unit": "µg/m³", "fair": 30, "poor": 60, "desc": "Fine Particulate Matter are inhalable pollutant particles with a diameter less than 2.5 micrometers. They can enter lungs and bloodstream."},
        {"name": "NO2", "val": latest_row.get("no2", 0), "unit": "µg/m³", "fair": 40, "poor": 80, "desc": "Breathing in high levels of Nitrogen Dioxide increases the risk of respiratory problems."},
        {"name": "SO2", "val": latest_row.get("so2", 0), "unit": "µg/m³", "fair": 20, "poor": 50, "desc": "Exposure to Sulfur Dioxide can lead to throat and eye irritation and aggravate asthma."}
    ]

    for p in pollutants_data:
        s_class = get_status_class(p["val"], p["fair"], p["poor"])
        st.markdown(f'''
        <div class="pollutant-card {s_class}">
            <div class="pollutant-info">
                <div class="pollutant-title">{p["name"]}</div>
                <div class="pollutant-status {s_class}">{s_class.capitalize()}</div>
                <div class="pollutant-desc">{p["desc"]}</div>
            </div>
            <div class="pollutant-value-box">
                <div class="pollutant-val">{round(p["val"], 1)}</div>
                <div class="pollutant-unit">{p["unit"]}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# -----------------------------------
# TAB 3: WEATHER
# -----------------------------------
with tab3:
    st.subheader(f"Weather - {selected_city}")

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Temperature", f"{round(latest_row.get('temperature', 0), 1)} °C")
    w2.metric("Humidity", f"{round(latest_row.get('humidity', 0), 1)} %")
    w3.metric("Wind Speed", f"{round(latest_row.get('wind_speed', 0), 1)} m/s")
    w4.metric("Wind Direction", f"{round(latest_row.get('wind_direction', 0), 1)}°")

    st.markdown("---")
    st.subheader("Last 10 Days Weather Data")

    if len(last_10_days) < 10:
        st.info(f"Only {len(last_10_days)} day(s) of data are available in your current dataset.")

    display_10 = last_10_days.copy()
    display_10["temp_min"] = display_10["temp_min"].round(1)
    display_10["temp_max"] = display_10["temp_max"].round(1)
    display_10["temp_avg"] = display_10["temp_avg"].round(1)
    display_10["humidity_avg"] = display_10["humidity_avg"].round(1)
    display_10["wind_avg"] = display_10["wind_avg"].round(1)
    display_10["pm25_avg"] = display_10["pm25_avg"].round(1)

    display_10 = display_10.rename(columns={
        "date": "Date",
        "temp_min": "Min Temp (°C)",
        "temp_max": "Max Temp (°C)",
        "temp_avg": "Avg Temp (°C)",
        "humidity_avg": "Avg Humidity (%)",
        "wind_avg": "Avg Wind Speed",
        "pm25_avg": "Avg PM2.5"
    })

    st.dataframe(display_10, use_container_width=True)

# -----------------------------------
# TAB 4: MAP
# -----------------------------------
with tab4:
    st.subheader("Regional Pollution Radar (Animation)")
    
    # Use global df to show all locations over time
    map_df = df.copy()
    
    if "latitude" in map_df.columns and "longitude" in map_df.columns:
        map_df["Air Condition"] = map_df["pm2_5"].apply(get_aqi_status)
        map_df["time_str"] = map_df["timestamp"].dt.strftime("%Y-%m-%d %H:00")
        map_df = map_df.sort_values("timestamp")
        
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="pm2_5",
            size="pm2_5",
            animation_frame="time_str",
            color_continuous_scale=px.colors.sequential.YlOrRd,
            range_color=[0, 150],
            hover_name="location",
            hover_data={
                "pm2_5": True,
                "Air Condition": True,
                "pollution_source": True,
                "timestamp": False,
                "time_str": False,
                "latitude": False,
                "longitude": False
            },
            zoom=5,
            center={"lat": 17.3850, "lon": 78.4867},
            height=600,
            title="Pollution Hotspots Radar Over Time"
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Latitude and longitude columns not found.")

# -----------------------------------
# TAB 5: PREDICTION
# -----------------------------------
with tab5:
    st.subheader("Prediction Center")

    p1, p2 = st.columns([1, 1])

    with p1:
        pm2_5 = st.slider("PM2.5", 0.0, 250.0, float(round(filtered_df["pm2_5"].mean(), 2)))
        no2 = st.slider("NO2", 0.0, 150.0, float(round(filtered_df["no2"].mean(), 2)))
        so2 = st.slider("SO2", 0.0, 100.0, float(round(filtered_df["so2"].mean(), 2)))
        o3 = st.slider("O3", 0.0, 150.0, float(round(filtered_df["o3"].mean(), 2)))

    with p2:
        st.write("Adjust the pollutant values and predict the likely pollution source.")

        if st.button("Predict Pollution Source"):
            if model is None:
                st.error("Model not found. Please train the model first.")
            else:
                input_data = pd.DataFrame(
                    [[pm2_5, no2, so2, o3]],
                    columns=["pm2_5", "no2", "so2", "o3"]
                )
                prediction = model.predict(input_data)[0]
                st.success(f"Predicted Source: {prediction}")

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_data)[0]
                    confidence = max(probs) * 100
                    st.info(f"Confidence Score: {confidence:.2f}%")

# -----------------------------------
# TAB 6: RAW DATA
# -----------------------------------
with tab6:
    st.subheader(f"Dataset Preview - {selected_city}")
    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {selected_city} Data",
        data=csv_data,
        file_name=f"{selected_city}_pollution_data.csv",
        mime="text/csv"
    )