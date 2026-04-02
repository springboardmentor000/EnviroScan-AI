#  VISUALIZATIONS (matplotlib)
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
df = pd.read_csv("vij_hyd_labelled_dataset.csv")
OUTPUT_DIR = "plots/"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Color palette per station ──
STATION_COLORS = {
    3409492: "#E74C3C",   # HB Colony
    3409392: "#3498DB"   # Kanuru
    
}

pollutants_raw = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
existing_p     = [c for c in pollutants_raw if c in df.columns]
# Create missing columns for plotting (temporary)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["is_dry_season"] = df["month"].isin([11,12,1,2]).astype(int)

# Spatial distance features (use station lat/lon to city center)
CITY_CENTER = (16.5062, 80.6480)
df["dist_to_center_km"] = np.sqrt(
    (df["lat"] - CITY_CENTER[0])**2 + (df["lon"] - CITY_CENTER[1])**2
)

# ──────────────────────────────────────────
# PLOT 1: Pollutant Distribution (Boxplot)
# ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
fig.suptitle("Pollutant Distribution Across All Stations", fontsize=14, fontweight="bold")

for i, col in enumerate(existing_p):
    data_by_station = [
        df[df["location_id"] == sid][col].dropna().values
        for sid in df["location_id"].unique()
    ]
    labels = [
        str(sid) for sid in df["location_id"].unique()
    ]
    bp = axes[i].boxplot(data_by_station, patch_artist=True, labels=labels)
    colors = list(STATION_COLORS.values())
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[i].set_title(col.upper(), fontweight="bold")
    axes[i].set_xlabel("Station ID")
    axes[i].set_ylabel("Value (normalized)")
    axes[i].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}01_pollutant_boxplot.png", dpi=150)
plt.close()
print("✅ Plot 1 saved: pollutant_boxplot.png")

# ──────────────────────────────────────────
# PLOT 2: Hourly Pollution Pattern
# ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
fig.suptitle("Average Pollutant Level by Hour of Day", fontsize=14, fontweight="bold")

for i, col in enumerate(existing_p):
    hourly = df.groupby("hour")[col].mean()
    axes[i].plot(hourly.index, hourly.values, color="#2C3E50", linewidth=2, marker="o", markersize=4)
    axes[i].axvspan(7, 10,  alpha=0.15, color="red",    label="Rush AM")
    axes[i].axvspan(17, 20, alpha=0.15, color="orange", label="Rush PM")
    axes[i].set_title(col.upper(), fontweight="bold")
    axes[i].set_xlabel("Hour of Day")
    axes[i].set_ylabel("Mean Value")
    axes[i].set_xticks(range(0, 24, 2))
    if i == 0:
        axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}02_hourly_pattern.png", dpi=150)
plt.close()
print("✅ Plot 2 saved: hourly_pattern.png")

# ──────────────────────────────────────────
# PLOT 3: Pollution Source Label Distribution
# ──────────────────────────────────────────
if "pollution_source" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pollution Source Label Distribution", fontsize=14, fontweight="bold")

    label_counts = df["pollution_source"].value_counts()
    bar_colors   = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    # Bar chart
    axes[0].bar(label_counts.index, label_counts.values,
                color=bar_colors[:len(label_counts)], edgecolor="black", alpha=0.85)
    axes[0].set_title("Overall Label Counts")
    axes[0].set_xlabel("Pollution Source")
    axes[0].set_ylabel("Count")
    for j, (k, v) in enumerate(label_counts.items()):
        axes[0].text(j, v + 10, str(v), ha="center", fontsize=9)

    # Pie chart
    axes[1].pie(label_counts.values, labels=label_counts.index,
                colors=bar_colors[:len(label_counts)],
                autopct="%1.1f%%", startangle=140,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Label Share (%)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}03_label_distribution.png", dpi=150)
    plt.close()
    print("✅ Plot 3 saved: label_distribution.png")

# ──────────────────────────────────────────
# PLOT 4: All Pollutants + Weather by Day of Week
# ──────────────────────────────────────────
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

all_plot_cols = (
    [c for c in ["pm2_5", "pm10", "no2", "so2", "co", "o3"] if c in df.columns] +
    [c for c in ["temperature_c", "humidity", "pressure_hpa",
                 "wind_speed_ms", "wind_direction"] if c in df.columns]
)

# ── 4a: Pollutants ──
pollutant_plot_cols = [c for c in ["pm2_5", "pm10", "no2", "so2", "co", "o3"] if c in df.columns]
n = len(pollutant_plot_cols)
cols_per_row = 3
rows = (n + cols_per_row - 1) // cols_per_row

fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 5 * rows))
axes = axes.flatten()
fig.suptitle("Pollutants by Day of Week", fontsize=14, fontweight="bold")

for i, col in enumerate(pollutant_plot_cols):
    weekly = df.groupby("day_of_week")[col].mean()
    bars   = axes[i].bar(range(7), weekly.values, color="#3498DB",
                         edgecolor="black", alpha=0.8)
    axes[i].set_xticks(range(7))
    axes[i].set_xticklabels(days, fontsize=9)
    axes[i].set_title(f"Avg {col.upper()} by Day", fontweight="bold")
    axes[i].set_ylabel("Mean (normalized)")
    axes[i].grid(axis="y", alpha=0.3)
    for idx in [5, 6]:                          # highlight weekends red
        bars[idx].set_color("#E74C3C")
        bars[idx].set_alpha(0.85)

# hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}06a_pollutants_by_weekday.png", dpi=150)
plt.close()
print("✅ Plot 4a saved: pollutants_by_weekday.png")

# ── 4b: Weather ──
weather_plot_cols = [c for c in ["temperature_c", "humidity", "pressure_hpa",
                                  "wind_speed_ms", "wind_direction"] if c in df.columns]
n = len(weather_plot_cols)
rows = (n + cols_per_row - 1) // cols_per_row

fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 5 * rows))
axes = axes.flatten()
fig.suptitle("Weather Features by Day of Week", fontsize=14, fontweight="bold")

for i, col in enumerate(weather_plot_cols):
    weekly = df.groupby("day_of_week")[col].mean()
    bars   = axes[i].bar(range(7), weekly.values, color="#2ECC71",
                         edgecolor="black", alpha=0.8)
    axes[i].set_xticks(range(7))
    axes[i].set_xticklabels(days, fontsize=9)
    axes[i].set_title(f"Avg {col.upper()} by Day", fontweight="bold")
    axes[i].set_ylabel("Mean (normalized)")
    axes[i].grid(axis="y", alpha=0.3)
    for idx in [5, 6]:                          # highlight weekends red
        bars[idx].set_color("#E74C3C")
        bars[idx].set_alpha(0.85)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}06b_weather_by_weekday.png", dpi=150)
plt.close()
print("✅ Plot 4b saved: weather_by_weekday.png")

# ──────────────────────────────────────────
# PLOT 5: Correlation Heatmap
# ──────────────────────────────────────────
corr_cols = [c for c in existing_p +
             ["temperature_c", "humidity", "wind_speed_ms",
              "hour", "is_dry_season", "dist_to_center_km"]
             if c in df.columns]

corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, shrink=0.8)

ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr_cols, fontsize=9)
ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")

for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        val = corr_matrix.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="black" if abs(val) < 0.7 else "white")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}05_correlation_heatmap.png", dpi=150)
plt.close()
print("✅ Plot 5 saved: correlation_heatmap.png")

print(f"\n📁 All plots saved to → {OUTPUT_DIR}")
