import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("enviro_data_with_distance.csv")

# ✅ NOW print
print(df["City"].unique())
print("Total unique cities:", df["City"].nunique())

# -------------------------------
# 1️⃣ BAR GRAPH (PM2.5 per City)
# -------------------------------
# Group by city and take average
city_pm25 = df.groupby("City")["PM2.5"].mean()

plt.figure()
plt.bar(city_pm25.index, city_pm25.values)
plt.title("Average PM2.5 by City")
plt.xlabel("City")
plt.ylabel("PM2.5")
plt.xticks(rotation=45)
plt.show()

# -------------------------------
# 2️⃣ SCATTER PLOT (PM2.5 vs NO2)
# -------------------------------
city_data = df.groupby("City")[["PM2.5", "NO2"]].mean()

plt.figure()
plt.scatter(city_data["PM2.5"], city_data["NO2"])

for i, city in enumerate(city_data.index):
    plt.text(city_data["PM2.5"].iloc[i], city_data["NO2"].iloc[i], city)

plt.title("PM2.5 vs NO2 (City-wise)")
plt.xlabel("PM2.5")
plt.ylabel("NO2")
plt.show()

# -------------------------------
# 3️⃣ PIE CHART (Source Distribution)
# -------------------------------
# Check if Source column exists
if "Source" in df.columns:
    source_counts = df["Source"].value_counts()

    plt.figure()
    plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%')
    plt.title("Pollution Source Distribution")
    plt.show()
else:
    print("⚠️ Source column not found. Run model.py first.")