import pandas as pd
import matplotlib.pyplot as plt

# ==============================

# LOAD DATASET

# ==============================

df = pd.read_csv("labeled_environment_dataset.csv")
print("Dataset Loaded:", df.shape)

df["timestamp"] = pd.to_datetime(df["timestamp"])

fig, ax = plt.subplots(1,3, figsize=(15,5))

# ==============================

# 1. TOTAL POLLUTION SOURCES

# ==============================

source_counts = df["source_label"].value_counts()

ax[0].bar(source_counts.index, source_counts.values)

ax[0].set_title("Total Pollution Sources")
ax[0].set_xlabel("Source")
ax[0].set_ylabel("Count")

# ==============================

# 2. INDUSTRIAL vs AGRICULTURAL vs VEHICULAR

# ==============================

sources = ["Industrial","Agricultural","Vehicular"]

counts = [
(df["source_label"]=="Industrial").sum(),
(df["source_label"]=="Agricultural").sum(),
(df["source_label"]=="Vehicular").sum()
]

ax[1].bar(sources, counts)

ax[1].set_title("Industrial vs Agricultural vs Vehicular")
ax[1].set_xlabel("Source")
ax[1].set_ylabel("Count")

# ==============================

# 3. WORKING vs NON-WORKING DAYS

# ==============================

df["day_type"] = df["timestamp"].dt.dayofweek.apply(
lambda x: "Working Day" if x < 5 else "Non Working Day"
)

day_counts = df.groupby("day_type")["pm25"].mean()

# ensure both bars appear

day_counts = day_counts.reindex(["Working Day","Non Working Day"]).fillna(0)

ax[2].bar(day_counts.index, day_counts.values, color=["blue","pink"])

ax[2].set_title("Working vs Non-Working Pollution")
ax[2].set_xlabel("Day Type")
ax[2].set_ylabel("Average PM2.5")

plt.tight_layout()
plt.show()
