import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="EnviroScan Dashboard", layout="wide")

st.title("EnviroScan - Air Pollution Monitoring Dashboard")

df = pd.read_csv("data/processed/labeled_dataset.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

model = joblib.load("model.pkl")

st.sidebar.header("Filters")

city = st.sidebar.selectbox("Select City", df["location"].unique())
pollutant = st.sidebar.selectbox("Select Pollutant", ["pm2_5", "no2", "so2", "o3"])

filtered_df = df[df["location"] == city]

col1, col2, col3 = st.columns(3)

col1.metric("Avg PM2.5", round(filtered_df["pm2_5"].mean(), 2))
col2.metric("Avg NO2", round(filtered_df["no2"].mean(), 2))
col3.metric("Avg SO2", round(filtered_df["so2"].mean(), 2))

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Graphs", "Map", "Prediction"])


with tab1:
    st.subheader("Pollution Trend")

    fig1 = px.line(filtered_df, x="timestamp", y=pollutant, title="Pollution Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Pollution Source Distribution")

    source_count = filtered_df["pollution_source"].value_counts().reset_index()
    source_count.columns = ["Source", "Count"]

    fig2 = px.pie(source_count, values="Count", names="Source")
    st.plotly_chart(fig2, use_container_width=True)


with tab2:
    st.subheader("Graph Analysis")

    x_axis = st.selectbox("Select X Axis", df.columns)
    y_axis = st.selectbox("Select Y Axis", df.columns)

    fig3 = px.scatter(filtered_df, x=x_axis, y=y_axis, color="pollution_source")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Correlation Heatmap")

    corr = filtered_df.corr(numeric_only=True)
    fig4 = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader("Pollution Map")

    with open("data/processed/pollution_map.html", "r", encoding="utf-8") as f:
        html_data = f.read()

    st.components.v1.html(html_data, height=500)

with tab4:
    st.subheader("Predict Pollution Source (ML Model)")

    pm2_5 = st.slider("PM2.5", 0, 200, 50)
    no2 = st.slider("NO2", 0, 100, 20)
    so2 = st.slider("SO2", 0, 100, 10)
    o3 = st.slider("O3", 0, 100, 20)

    if st.button("Predict"):
        input_data = [[pm2_5, no2, so2, o3]]
        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Source: {prediction}")

st.subheader("Dataset Preview")
st.dataframe(filtered_df)