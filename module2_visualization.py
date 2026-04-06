import pandas as pd
import matplotlib.pyplot as plt

print(" Running Visualization...")

# =========================================
#  LOAD CLEANED DATA 
# =========================================
df = pd.read_csv("data/cleaned_data.csv")

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')

# =========================================
#  STYLE SETTINGS 
# =========================================
plt.style.use('ggplot')  # nice theme

# =========================================
#  CREATE 4 SIMPLE GRAPHS
# =========================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# =========================================
#  1. HOURLY PM2.5 TREND (CITY-WISE)
# =========================================
for city in df['city'].unique():
    city_data = df[df['city'] == city]
    hourly_avg = city_data.groupby('hour')['pm2_5'].mean().sort_index()
    
    axs[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', label=city)

axs[0, 0].set_title("PM2.5 Hourly Trend")
axs[0, 0].set_xlabel("Hour")
axs[0, 0].set_ylabel("PM2.5")
axs[0, 0].legend()

# =========================================
#  2. POLLUTION LEVEL DISTRIBUTION
# =========================================
colors = ['green', 'orange', 'red', 'purple']
df['pollution_level'].value_counts().plot(kind='bar', ax=axs[0, 1], color=colors)

axs[0, 1].set_title("Pollution Levels")
axs[0, 1].set_xlabel("Level")
axs[0, 1].set_ylabel("Count")

# =========================================
#  3. CITY-WISE AVG PM2.5
# =========================================
df.groupby('city')['pm2_5'].mean().plot(kind='bar', ax=axs[1, 0], color='skyblue')

axs[1, 0].set_title("City-wise Avg PM2.5")
axs[1, 0].set_xlabel("City")

# =========================================
#  4. TEMP vs PM2.5 (SCATTER)
# =========================================
axs[1, 1].scatter(df['temp'], df['pm2_5'], alpha=0.6)

axs[1, 1].set_title("Temperature vs PM2.5")
axs[1, 1].set_xlabel("Temperature")
axs[1, 1].set_ylabel("PM2.5")

# =========================================
#  SAVE + SHOW
# =========================================
plt.tight_layout()
plt.savefig("data/final_visualization.png")

print(" Visualization Updated Automatically!")