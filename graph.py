import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("labeled_environment_dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ==============================
# FIGURE
# ==============================

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Graphical Representation of Pollution Analysis", fontsize=16, fontweight='bold')

# ==============================
# 1. TOTAL POLLUTION SOURCES
# ==============================

source_counts = df["source_label"].value_counts()

colors1 = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0"]

ax[0].bar(source_counts.index, source_counts.values, color=colors1)
ax[0].set_title("Total Pollution Sources")
ax[0].set_xlabel("Source")
ax[0].set_ylabel("Count")
ax[0].tick_params(axis='x', rotation=25)

# ==============================
# 2. ALL STATES POLLUTION (NEW)
# ==============================

state_pollution = df.groupby("state")["pm25"].mean().sort_values(ascending=False)

colors2 = plt.cm.viridis(range(len(state_pollution)))  # gradient colors

ax[1].bar(state_pollution.index, state_pollution.values, color=colors2)
ax[1].set_title("Pollution by State (Avg PM2.5)")
ax[1].set_xlabel("State")
ax[1].set_ylabel("PM2.5 Level")

# 🔥 FIX OVERLAP
ax[1].tick_params(axis='x', rotation=60, labelsize=8)

# ==============================
# 3. WORKING vs NON-WORKING
# ==============================

df["day_type"] = df["timestamp"].dt.dayofweek.apply(
    lambda x: "Working Day" if x < 5 else "Non Working Day"
)

day_counts = df.groupby("day_type")["pm25"].mean()

colors3 = ["#00ACC1", "#FF7043"]

ax[2].bar(day_counts.index, day_counts.values, color=colors3)
ax[2].set_title("Working vs Non-Working Pollution")
ax[2].set_xlabel("Day Type")
ax[2].set_ylabel("Average PM2.5")

# ==============================
# FINAL LAYOUT
# ==============================

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()