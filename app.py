

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.express as px

st.set_page_config(page_title="EnviroScan Dashboard", layout="wide")

st.title(" EnviroScan: Pollution Intelligence Dashboard")


data_path = "Infosys_dataset.csv"
model_path = "pollution_model.pkl"

if not os.path.exists(data_path):
    st.error(" Dataset not found")
    st.stop()

df = pd.read_csv(data_path)

if not os.path.exists(model_path):
    st.error(" Model not found")
    st.stop()

model = joblib.load(model_path)

st.sidebar.title(" Controls")

city_filter = st.sidebar.multiselect(
    "Select Cities",
    options=df["city"].dropna().unique(),
    default=df["city"].dropna().unique()[:5]
)

filtered_df = df[df["city"].isin(city_filter)]


if filtered_df.shape[0] < 2:
    st.warning(" Not enough data. Please select more cities.")
    st.stop()

st.subheader(" Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg PM2.5", round(filtered_df["PM2_5"].mean(), 2))
col2.metric("Avg AQI", round(filtered_df["AQI_score"].mean(), 2))
col3.metric("Max PM2.5", round(filtered_df["PM2_5"].max(), 2))
col4.metric("Cities Selected", len(city_filter))


st.subheader(" Pollution Alert")

latest = filtered_df.tail(1)
aqi_val = latest["AQI_score"].values[0]

if aqi_val > 0.7:
    st.error(" HIGH POLLUTION ALERT!")
elif aqi_val > 0.4:
    st.warning(" Moderate Pollution")
else:
    st.success(" Air Quality Normal")


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [" Overview", " Graph Analysis", " Correlation", " Map", " Prediction"]
)


with tab1:
    st.subheader(" Dataset Overview")

    st.dataframe(filtered_df)

    st.download_button(
        label="⬇ Download Data",
        data=filtered_df.to_csv(index=False),
        file_name="pollution_data.csv",
        mime="text/csv"
    )


with tab2:
    st.subheader(" Pollution Graph Analysis")

    df_plot = filtered_df.reset_index()

    st.markdown("###  Pollution Trend")

    fig1 = px.line(
        df_plot,
        x="index",
        y=["PM2_5", "PM10", "NO2"],
        color="city",
        markers=True
    )

    st.plotly_chart(fig1, use_container_width=True)

    if len(filtered_df) > 5:
        st.markdown("###  PM2.5 Distribution")

        fig2 = px.histogram(
            filtered_df,
            x="PM2_5",
            nbins=20
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("###  Pollution Spread")

    fig3 = px.box(
        filtered_df,
        y=["PM2_5", "PM10", "NO2"]
    )

    st.plotly_chart(fig3, use_container_width=True)


with tab3:
    st.subheader(" Feature Correlation")

    numeric_cols = [
        "PM2_5", "PM10", "NO2", "CO", "SO2", "O3",
        "AQI_score", "temperature_C", "humidity_%",
        "wind_speed_mps", "latitude", "longitude"
    ]

    numeric_cols = [col for col in numeric_cols if col in filtered_df.columns]

    if len(filtered_df) > 2:
        corr = filtered_df[numeric_cols].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap"
        )

        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning(" Not enough data for correlation")


with tab4:
    st.subheader(" Geospatial Pollution Map")

    fig_map = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        color="source_domain",
        size="PM2_5",
        hover_name="city",
        zoom=4
    )

    fig_map.update_layout(mapbox_style="open-street-map")

    st.plotly_chart(fig_map, use_container_width=True)


with tab5:
    st.subheader(" Predict Pollution Source")

    mode = st.radio("Choose Mode", ["City Based", "Manual Input"])

   
    if mode == "City Based":

        selected_city = st.selectbox("Select City", df["city"].unique())

        city_data = df[df["city"] == selected_city].tail(1)

        if city_data.shape[0] == 0:
            st.error("No data available for this city")
        else:
            st.write("###  Latest City Data")
            st.dataframe(city_data)

            row = city_data.iloc[0]

            input_data = np.array([[
                row["PM2_5"],
                row["PM10"],
                row["NO2"],
                row["CO"],
                row["SO2"],
                row["O3"],
                row["AQI_score"],
                row["temperature_C"],
                row["humidity_%"],
                row["wind_speed_mps"]
            ]])

            if st.button("Predict for City"):
                pred = model.predict(input_data)
                st.success(f" Predicted Source for {selected_city}: {pred[0]}")


    else:

        col1, col2, col3 = st.columns(3)

        pm25 = col1.slider("PM2.5", 0.0, 300.0, 50.0)
        pm10 = col2.slider("PM10", 0.0, 300.0, 60.0)
        no2 = col3.slider("NO2", 0.0, 200.0, 40.0)

        col4, col5, col6 = st.columns(3)

        co = col4.slider("CO", 0.0, 10.0, 1.0)
        so2 = col5.slider("SO2", 0.0, 200.0, 20.0)
        o3 = col6.slider("O3", 0.0, 200.0, 30.0)

        col7, col8, col9 = st.columns(3)

        temp = col7.slider("Temperature", 0.0, 50.0, 25.0)
        hum = col8.slider("Humidity", 0.0, 100.0, 60.0)
        wind = col9.slider("Wind Speed", 0.0, 20.0, 5.0)

        aqi = (pm25 + pm10 + no2 + co + so2 + o3) / 6

        if st.button("Predict"):
            input_data = np.array([[pm25, pm10, no2, co, so2, o3, aqi, temp, hum, wind]])
            pred = model.predict(input_data)

            st.success(f" Predicted Source: {pred[0]}")


st.markdown("---")
st.caption(" EnviroScan | AI + Geospatial Intelligence")