import streamlit as st
import pandas as pd
import joblib

st.title("AI Pollution Source Prediction Dashboard")

st.write("Predict the source of air pollution using machine learning.")

# =====================================
# Load trained model
# =====================================

model = joblib.load("pollution_source_model.pkl")

# =====================================
# Sidebar inputs
# =====================================

st.sidebar.header("Input Pollution Data")

PM2_5 = st.sidebar.slider("PM2.5", 0.0, 300.0, 50.0)
PM10 = st.sidebar.slider("PM10", 0.0, 400.0, 80.0)
NO2 = st.sidebar.slider("NO2", 0.0, 200.0, 40.0)
CO = st.sidebar.slider("CO", 0.0, 10.0, 0.8)
SO2 = st.sidebar.slider("SO2", 0.0, 100.0, 10.0)
O3 = st.sidebar.slider("O3", 0.0, 200.0, 30.0)

temperature_C = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 30.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 3.0)

dist_to_road = st.sidebar.slider("Distance to Road (meters)", 0, 1000, 100)
dist_to_industry = st.sidebar.slider("Distance to Industry (meters)", 0, 2000, 500)
dist_to_farmland = st.sidebar.slider("Distance to Farmland (meters)", 0, 2000, 500)

# =====================================
# Prepare input data
# =====================================

input_data = pd.DataFrame([{
    'PM2_5': PM2_5,
    'PM10': PM10,
    'NO2': NO2,
    'CO': CO,
    'SO2': SO2,
    'O3': O3,
    'temperature_C': temperature_C,
    'humidity_%': humidity,
    'wind_speed_mps': wind_speed,
    'dist_to_road': dist_to_road,
    'dist_to_industry': dist_to_industry,
    'dist_to_farmland': dist_to_farmland
}])

st.subheader("Input Data")
st.write(input_data)

# =====================================
# Prediction
# =====================================

if st.button("Predict Pollution Source"):

    prediction = model.predict(input_data)

    st.subheader("Predicted Pollution Source")

    st.success(prediction[0])

# =====================================
# Information Section
# =====================================

st.markdown("---")

st.write("""
### Pollution Source Categories

Vehicular → traffic emissions  
Industrial → factory pollution  
Agricultural → farming activity  
Burning → waste or biomass burning  
Natural → dust or natural sources
""")

st.write("Developed using Machine Learning (Random Forest).")