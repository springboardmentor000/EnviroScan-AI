import pandas as pd
import numpy as np

print("Running Module 2: Data Cleaning...")

# =========================================
#  LOAD RAW DATA
# =========================================
df = pd.read_csv("data/raw_data.csv")

print("Initial Data Shape:", df.shape)

# =========================================
#  1. STANDARDIZE DATA TYPES
# =========================================
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['timestamp'])

df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
df = df.dropna(subset=['lat','lon'])

# =========================================
#  2. REMOVE EXACT DUPLICATES
# =========================================
df = df.drop_duplicates(subset=['city','timestamp'], keep='last')
print("After exact duplicate removal:", df.shape)

# =========================================
#  3. CREATE DATE + HOUR
# =========================================
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour

# =========================================
#  4. KEEP ONLY 1 ENTRY PER HOUR PER CITY
# =========================================
df = df.sort_values('timestamp')
df = df.drop_duplicates(subset=['city','date','hour'], keep='last')
print("After hourly filtering:", df.shape)

# =========================================
#  5. STANDARDIZE VALUES
# =========================================
pollutants = ['pm2_5','pm10','no2','co','so2','o3']
weather_cols = ['temp','humidity','wind']

df[pollutants + weather_cols] = df[pollutants + weather_cols].round(2)

# =========================================
#  6. HANDLE INVALID VALUES 
# =========================================

# Pollution values cannot be <= 0
df[pollutants] = df[pollutants].apply(lambda x: x.where(x > 0, np.nan))

# Weather values → 0 means API failure → convert to NaN
df['temp'] = df['temp'].replace(0, np.nan)
df['humidity'] = df['humidity'].replace(0, np.nan)
df['wind'] = df['wind'].replace(0, np.nan)

# =========================================
#  7. HANDLE MISSING VALUES 
# =========================================
df = df.sort_values(['city','timestamp'])

# Step 1: Interpolate within each city (time-based)
df[pollutants + weather_cols] = df.groupby('city')[pollutants + weather_cols].transform(
    lambda group: group.interpolate(method='linear', limit_direction='both')
)

# Step 2: Fill remaining NaN with city median
df[pollutants + weather_cols] = df.groupby('city')[pollutants + weather_cols].transform(
    lambda group: group.fillna(group.median())
)

# Step 3: Final fallback (global median)
df[pollutants + weather_cols] = df[pollutants + weather_cols].fillna(
    df[pollutants + weather_cols].median()
)

# =========================================
#  8. TEMPORAL FEATURES
# =========================================
df['day_of_week'] = df['timestamp'].dt.dayofweek

def get_season(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Summer"
    elif month in [6,7,8]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

df['season'] = df['timestamp'].dt.month.apply(get_season)

# =========================================
#  9. POLLUTION LEVEL FEATURE
# =========================================
def pollution_level(pm):
    if pm <= 30:
        return "Low"
    elif pm <= 60:
        return "Moderate"
    elif pm <= 120:
        return "High"

    

df['pollution_level'] = df['pm2_5'].apply(pollution_level)

# =========================================
#  10. SPATIAL FEATURES CLEANING
# =========================================
spatial_cols = ['dist_road','dist_industry','dist_dump','dist_farm']

df[spatial_cols] = df[spatial_cols].fillna(3000)
df[spatial_cols] = df[spatial_cols].astype(float)

# =========================================
#  11. FINAL SORTING
# =========================================
df = df.sort_values(by="timestamp")

# =========================================
#  12. FINAL COLUMN ORDER
# =========================================
final_columns = [
    'city','lat','lon','timestamp',
    'pm2_5','pm10','no2','co','so2','o3',
    'temp','humidity','wind','wind_dir',
    'dist_road','dist_industry','dist_dump','dist_farm',
    'hour','day_of_week','season',
    'pollution_level'
]

df = df[final_columns]

# =========================================
#  13. FINAL CHECK
# =========================================
print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFinal Data Shape:", df.shape)

# =========================================
#  SAVE CLEANED DATA
# =========================================
df.to_csv("data/cleaned_data.csv", index=False)

print("\n Module 2 Completed Successfully")