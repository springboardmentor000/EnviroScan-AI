import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("enviro_data_with_distance.csv")
df = df.dropna()

# -------------------------------
# SOURCE LABEL
# -------------------------------
def label_source(row):
    if row["Dist_Industrial"] < 1000 and row["SO2"] > 5:
        return "Industrial"
    elif row["Dist_Farmland"] < 1000 and row["PM2.5"] > 20:
        return "Agricultural"
    elif row["Dist_Road"] < 500 and row["NO2"] > 10:
        return "Vehicular"
    else:
        return "Natural"

df["Source"] = df.apply(label_source, axis=1)

# -------------------------------
# MODEL
# -------------------------------
X = df[[
    "Dist_Road", "Dist_Industrial", "Dist_Farmland",
    "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)",
    "PM2.5", "PM10", "NO2", "SO2"
]]

y = df["Source"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Controls")

cities = df["City"].unique()
selected_city = st.sidebar.selectbox("Select City", cities)

city_df = df[df["City"] == selected_city]

# -------------------------------
# TITLE
# -------------------------------
st.title("EnviroScan")

# -------------------------------
# BAR GRAPH (SMALL)
# -------------------------------
st.subheader("PM2.5 by City")

city_pm25 = df.groupby("City")["PM2.5"].mean()

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(city_pm25.index, city_pm25.values)
ax.set_xticklabels(city_pm25.index, rotation=45)
ax.set_ylabel("PM2.5")

st.pyplot(fig, use_container_width=False)

# -------------------------------
# SCATTER (SMALL)
# -------------------------------
st.subheader("PM2.5 vs NO2")

city_data = df.groupby("City")[["PM2.5", "NO2"]].mean()

fig, ax = plt.subplots(figsize=(5, 3))
ax.scatter(city_data["PM2.5"], city_data["NO2"])

for i, city in enumerate(city_data.index):
    ax.text(city_data["PM2.5"].iloc[i], city_data["NO2"].iloc[i], city, fontsize=8)

ax.set_xlabel("PM2.5")
ax.set_ylabel("NO2")

st.pyplot(fig, use_container_width=False)

# -------------------------------
# PIE CHART (SMALL)
# -------------------------------
st.subheader(f"Source Distribution - {selected_city}")

source_counts = city_df["Source"].value_counts()

fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%')

st.pyplot(fig, use_container_width=False)

# -------------------------------
# INPUT
# -------------------------------
st.subheader("Input")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.slider("PM2.5", 0.0, 500.0, 50.0)
    no2 = st.slider("NO2", 0.0, 300.0, 30.0)
    temp = st.slider("Temperature", 0.0, 50.0, 25.0)

with col2:
    pm10 = st.slider("PM10", 0.0, 1000.0, 100.0)
    so2 = st.slider("SO2", 0.0, 100.0, 10.0)
    humidity = st.slider("Humidity", 0, 100, 40)

# -------------------------------
# PREDICT
# -------------------------------
if st.button("Predict"):

    input_data = [[
        200, 500, 800,
        temp, humidity, 3,
        pm25, pm10, no2, so2
    ]]

    prediction = model.predict(input_data)
    result = le.inverse_transform(prediction)[0]

    st.write("Predicted Source:", result)