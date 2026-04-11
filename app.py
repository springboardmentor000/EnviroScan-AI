import streamlit as st
import pandas as pd
import joblib
import datetime
from streamlit_option_menu import option_menu
import osmnx as ox
import sys
sys.path.append("../..")

import config

API_KEY = config.OWM_API_KEY

st.set_page_config(layout="wide")

# Load model
model = joblib.load("model/final_model.pkl")

# Sidebar
with st.sidebar:
    page = option_menu(
    menu_title=None, #required
    options=["Home", "Manual Input", "Map","Data View"], #required
    default_index=0, #optional
    icons=["house", "pencil-square", "map", "bar-chart"]
    )

# HOME
if page == "Home":

    import requests
    import plotly.express as px
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
    import datetime

    st.title("Environment Scan")
    st.markdown("### Pollution Insights Dashboard")

    city_input = st.text_input("Enter City (e.g., Delhi, Mumbai)")


    # Store state
    if "selected_city" not in st.session_state:
        st.session_state.selected_city = None

    if st.button("Analyze"):
        st.session_state.selected_city = city_input

    # Use stored value
    if st.session_state.selected_city:

        city = st.session_state.selected_city

        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={API_KEY}"
        geo = requests.get(geo_url).json()

        if len(geo) == 0:
            st.error("City not found")
        else:
            lat = geo[0]["lat"]
            lon = geo[0]["lon"]

            st.success(f"Showing data for {city}")

            # CURRENT DATA
            air_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
            air = requests.get(air_url).json()
            comp = air["list"][0]["components"]
            #  WEATHER DATA (ADD THIS)
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            weather = requests.get(weather_url).json()

            temp = weather["main"]["temp"]
            humidity = weather["main"]["humidity"]
            wind_speed = weather["wind"]["speed"]

            st.subheader("Current Pollution Levels")

            col1, col2, col3 = st.columns(3)

            col1.metric("PM2.5", comp["pm2_5"])
            col2.metric("PM10", comp["pm10"])
            col3.metric("NO2", comp["no2"])

            col1.metric("SO2", comp["so2"])
            col2.metric("CO", comp["co"])
            col3.metric("O3", comp["o3"])

            # HEATMAP
            st.subheader("Pollution Heatmap")

            m = folium.Map(location=[lat, lon], zoom_start=11)

            pollution_value = (
                comp["pm2_5"] + comp["pm10"] +
                comp["no2"] + comp["so2"] +
                comp["co"] + comp["o3"]
            )

            heat_data = [
                [lat, lon, pollution_value]

            ]

            HeatMap(
                heat_data,
                gradient={
                    0.2: 'green',    
                    0.4: 'yellow',   
                    0.6: 'orange',   
                    0.8: 'red'      
                },
                min_opacity=0.5,
                radius=40,
                blur=25,
                max_zoom=1
            ).add_to(m)

            st_folium(m, width=800, height=400)

            # 5 DAYS 
            st.subheader("Last 5 Days Pollution Trends")

            history_data = []

            for i in range(5, 0, -1):
                start = int(datetime.datetime.now().timestamp()) - i * 86400
                end = int(datetime.datetime.now().timestamp()) - (i - 1) * 86400

                hist_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={API_KEY}"
                hist = requests.get(hist_url).json()

                if "list" in hist and len(hist["list"]) > 0:

                    avg = {
                        "Day": f"{i} days ago",
                        "pm2.5": sum([x["components"]["pm2_5"] for x in hist["list"]]) / len(hist["list"]),
                        "no2": sum([x["components"]["no2"] for x in hist["list"]]) / len(hist["list"]),
                        "so2": sum([x["components"]["so2"] for x in hist["list"]]) / len(hist["list"]),
                        "co": sum([x["components"]["co"] for x in hist["list"]]) / len(hist["list"]),
                        "o3": sum([x["components"]["o3"] for x in hist["list"]]) / len(hist["list"]),
                    }

                    history_data.append(avg)

            if history_data:
                hist_df = pd.DataFrame(history_data)

                for col in ["pm2.5", "no2", "so2", "co", "o3"]:
                    fig = px.line(hist_df, x="Day", y=col, markers=True, title=f"{col.upper()} Trend")
                    st.plotly_chart(fig, use_container_width=True)

            # PREDICTION
            st.subheader("Predicted Pollution Source")

            model = joblib.load("model/final_model.pkl")

            # static features
            road = 30
            industry = 5
            agriculture = 3
            dump = 1

            total_pollution = (
                comp["pm2_5"] + comp["pm10"] +
                comp["no2"] + comp["so2"] +
                comp["co"] + comp["o3"]
            )

            pm_ratio = comp["pm2_5"] / (comp["pm10"] + 1)
            wind_dispersion = total_pollution / (wind_speed + 1)
            humidity_pm = humidity * comp["pm2_5"]

            input_data = pd.DataFrame([{
                "pm2.5 value": comp["pm2_5"],
                "pm10 value": comp["pm10"],
                "no2 value": comp["no2"],
                "so2 value": comp["so2"],
                "co value": comp["co"],
                "o3 value": comp["o3"],
                "temperature": temp,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "latitude": lat,
                "longitude": lon,
                "hour": datetime.datetime.now().hour,
                "day": datetime.datetime.now().day,
                "month": datetime.datetime.now().month,
                "weekday": datetime.datetime.now().weekday(),
                "road_count_2km": road,
                "industry_count_5km": industry,
                "agriculture_count_5km": agriculture,
                "dump_count_5km": dump,
                "total_pollution": total_pollution,
                "pm_ratio": pm_ratio,
                "wind_dispersion": wind_dispersion,
                "humidity_pm": humidity_pm
            }])

            input_data = input_data[model.feature_names_in_]

            pred = model.predict(input_data)[0]

            labels = {
                0: "Agricultural",
                1: "Industrial",
                2: "Natural ",
                3: "Vehicular "
            }

            st.success(f"Predicted Source: {labels.get(pred)}")
# -----------------------------
# MANUAL INPUT PAGE
# -----------------------------
elif page == "Manual Input":

    st.header("Manual Input Prediction")

    # Sliders
    pm25 = st.slider("PM2.5", 0, 300, 100)
    pm10 = st.slider("PM10", 0, 300, 150)
    no2 = st.slider("NO2", 0, 100, 40)
    so2 = st.slider("SO2", 0, 100, 20)
    co = st.slider("CO", 0.0, 5.0, 1.0)
    o3 = st.slider("O3", 0, 100, 30)

    temperature = st.slider("Temperature", 0, 50, 30)
    humidity = st.slider("Humidity", 0, 100, 60)
    wind_speed = st.slider("Wind Speed", 0, 10, 3)

    road = st.slider("Road Count", 0, 50, 10)
    industry = st.slider("Industry Count", 0, 20, 5)
    agriculture = st.slider("Agriculture Count", 0, 20, 3)
    dump = st.slider("Dump Count", 0, 10, 1)

    # Feature engineering
    total_pollution = pm25 + pm10 + no2 + so2 + co + o3
    pm_ratio = pm25 / (pm10 + 1)
    wind_dispersion = total_pollution / (wind_speed + 1)
    humidity_pm = humidity * pm25

    now = datetime.datetime.now()

    input_data = pd.DataFrame([{
        "pm2.5 value": pm25,
        "pm10 value": pm10,
        "no2 value": no2,
        "so2 value": so2,
        "co value": co,
        "o3 value": o3,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "latitude": 28.61,
        "longitude": 77.20,
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "weekday": now.weekday(),
        "road_count_2km": road,
        "industry_count_5km": industry,
        "agriculture_count_5km": agriculture,
        "dump_count_5km": dump,
        "total_pollution": total_pollution,
        "pm_ratio": pm_ratio,
        "wind_dispersion": wind_dispersion,
        "humidity_pm": humidity_pm
    }])

    # Fix column mismatch
    input_data = input_data[model.feature_names_in_]

    if st.button("Predict"):
        pred = model.predict(input_data)[0]

        labels = {
                0: "Agricultural",
                1: "Industrial",
                2: "Natural ",
                3: "Vehicular "
            }

        st.success(f"Predicted Source: {labels.get(pred, 'Unknown')}")

# -----------------------------
# MAP PAGE
# -----------------------------

elif page == "Map":

    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium

    st.header("India Pollution Map")
    if st.button("Refresh Live Data"):

        with st.spinner("Fetching latest data..."):
            import os
            os.system("python scripts/milestone_3/live_data.py")

        st.success("Data updated! Reloading...")

        st.rerun()

    option = st.selectbox(
        "Select Map Type",
        ["Combined","Pollution Heatmap", "Pollution Source Map"]
    )

    df = pd.read_csv("data/live_data.csv")

    now = datetime.datetime.now()

    # Feature engineering
    df["total_pollution"] = (
        df["pm2.5 value"] + df["pm10 value"] +
        df["no2 value"] + df["so2 value"] +
        df["co value"] + df["o3 value"]
    )

    df["pm_ratio"] = df["pm2.5 value"] / (df["pm10 value"] + 1)
    df["wind_dispersion"] = df["total_pollution"] / (df["wind_speed"] + 1)
    df["humidity_pm"] = df["humidity"] * df["pm2.5 value"]

    df["hour"] = now.hour
    df["day"] = now.day
    df["month"] = now.month
    df["weekday"] = now.weekday()

    X = df[model.feature_names_in_]
    df["predicted_source"] = model.predict(X)

    m = folium.Map(location=[22.97, 78.65], zoom_start=5)

    # HEATMAP
    if option == "Pollution Heatmap":
        heat_data = [
            [row["latitude"], row["longitude"], row["pm2.5 value"]]
            for _, row in df.iterrows()
        ]
        HeatMap(heat_data).add_to(m)

    # SOURCE MAP
    elif option == "Pollution Source Map":

        labels = {
            0: "Agricultural Burning",
            1: "Industrial",
            2: "Natural",
            3: "Vehicular"
        }

        icons = {
            0: ("leaf", "green"),        # Agriculture
            1: ("industry", "red"),      # Industrial
            2: ("cloud", "blue"),        # Natural
            3: ("car", "orange")         # Vehicular
        }
        for _, row in df.iterrows():

            icon_name, color = icons.get(row["predicted_source"], ("info-sign", "gray"))
            popup_html = f"""
                <b> City:</b> {row['city']}<br>
                <b> Source:</b> {labels.get(row['predicted_source'])}<br><br>

                <b> PM2.5:</b> {row['pm2.5 value']}<br>
                <b> PM10:</b> {row['pm10 value']}<br>
                <b> NO2:</b> {row['no2 value']}<br>
                <b> SO2:</b> {row['so2 value']}<br>
                <b> CO:</b> {row['co value']}<br>
                <b> O3:</b> {row['o3 value']}<br><br>
                <b>Date & Time:</b> {row['timestamp']}<br>
                """
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=color, icon=icon_name, prefix='fa')
            ).add_to(m)

    # COMBINED
    elif option == "Combined":

        heat_data = [
            [row["latitude"], row["longitude"], row["pm2.5 value"]]
            for _, row in df.iterrows()
        ]

        HeatMap(heat_data).add_to(m)
        colors = {0: "blue", 1: "red", 2: "green", 3: "purple"}
        labels = {
            0: "Agricultural Buring",
            1: "Industrial",
            2: "Natural",
            3: "Vehicular"
        }
        icons = {
            0: ("seedling", "green"),        # Agriculture
            1: ("industry", "red"),      # Industrial
            2: ("tree", "darkgreen"),        # Natural
            3: ("car", "orange")         # Vehicular
        }
        for _, row in df.iterrows():

            icon_name, color = icons.get(row["predicted_source"], ("info-sign", "gray"))
            popup_html = f"""
                <b>City:</b> {row['city']}<br>
                <b>Source:</b> {labels.get(row['predicted_source'])}<br><br>

                <b>PM2.5:</b> {row['pm2.5 value']}<br>
                <b>PM10:</b> {row['pm10 value']}<br>
                <b>NO2:</b> {row['no2 value']}<br>
                <b>SO2:</b> {row['so2 value']}<br>
                <b>CO:</b> {row['co value']}<br>
                <b>O3:</b> {row['o3 value']}<br><br>
                <b>Date & Time:</b> {row['timestamp']}<br>
                """
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                 popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=color, icon=icon_name, prefix='fa')
            ).add_to(m)

    st_folium(m, width=1000, height=600)

# -----------------------------
# DATA VIEW
# -----------------------------
elif page == "Data View":
    if st.button("Refresh Live Data"):

        with st.spinner("Fetching latest data..."):
            import os
            os.system("python scripts/milestone_3/live_data.py")

        st.success(" Data updated! Reloading...")

        st.rerun()
    df = pd.read_csv("data/live_data.csv")
    st.dataframe(df)