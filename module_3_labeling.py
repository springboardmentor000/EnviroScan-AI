import pandas as pd
import os

print("Starting Module 3: Pollution Source Labeling\n")

# ==============================
# 1️⃣ LOAD DATASET
# ==============================

df = pd.read_csv("module2_cleaned_features.csv")

print("Dataset loaded successfully")
print("Rows:", len(df))
print()

# ==============================
# 2️⃣ STANDARDIZE COLUMN NAMES
# ==============================

# make all column names lowercase
df.columns = df.columns.str.lower()

# rename possible variations
df.rename(columns={
    "temp": "temperature",
    "wind": "wind_speed"
}, inplace=True)

# ==============================
# 3️⃣ REMOVE DUPLICATES
# ==============================

df = df.drop_duplicates()

print("Duplicates removed if any\n")

# ==============================
# 4️⃣ POLLUTION LABEL RULES
# ==============================

def assign_source(row):

    # Vehicular pollution
    if row.get('dist_to_road', 9999) < 300 and row.get('no2', 0) > 0.6:
        return "Vehicular"

    # Industrial pollution
    elif row.get('dist_to_industry', 9999) < 500 and row.get('so2', 0) > 0.6:
        return "Industrial"

    # Agricultural pollution
    elif row.get('dist_to_farmland', 9999) < 700 and row.get('pm2_5', 0) > 0.6 and row.get('season') in ['Summer','Autumn']:
        return "Agricultural"

    # Burning pollution
    elif row.get('co', 0) > 0.6 and row.get('pm2_5', 0) > 0.6:
        return "Burning"

    # Natural pollution
    else:
        return "Natural"


# ==============================
# 5️⃣ APPLY LABELING
# ==============================

df["pollution_source"] = df.apply(assign_source, axis=1)

print("Pollution source labeling completed\n")

# ==============================
# 6️⃣ SHOW DISTRIBUTION
# ==============================

print("Pollution Source Distribution:\n")
print(df["pollution_source"].value_counts())
print()

# ==============================
# 7️⃣ SAVE / APPEND FINAL DATASET
# ==============================

file_name = "final_pollution_dataset.csv"

# remove duplicates before saving
df = df.drop_duplicates()

if os.path.exists(file_name):

    df.to_csv(file_name, mode='a', header=False, index=False)
    print("Data appended to existing dataset")

else:

    df.to_csv(file_name, mode='w', header=True, index=False)
    print("New dataset created")

print("Saved file:", file_name)

print("\nModule 3 Completed Successfully")