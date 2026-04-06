import pandas as pd

print(" Running Module 3: Source Labeling...\n")

# =========================================
#  LOAD CLEANED DATA
# =========================================
df = pd.read_csv("data/cleaned_data.csv")

print(" Data Loaded:", df.shape)

# =========================================
#  IMPROVED LABELING FUNCTION (BALANCED)
# =========================================
def label_source(row):

    #  Burning (relaxed)
    if row['pm2_5'] > 60 and row['co'] > 300:
        return "Burning"

    #  Vehicular
    elif row['dist_road'] < 60 and (row['no2'] > 8 or row['co'] > 250):
        return "Vehicular"

    # Industrial (relaxed)
    elif row['so2'] > 4 or row['co'] > 400:
        return "Industrial"

    #  Agricultural (relaxed)
    elif row['pm2_5'] > 30 and row['season'] == "Summer":
        return "Agricultural"

    #  Natural (default)
    else:
        return "Natural"

# =========================================
#  APPLY LABELING
# =========================================
df['pollution_source'] = df.apply(label_source, axis=1)

# =========================================
#  CHECK DISTRIBUTION
# =========================================
print("\n Pollution Source Distribution:\n")
print(df['pollution_source'].value_counts())

# =========================================
#  SAVE LABELED DATA
# =========================================
df.to_csv("data/labeled_data.csv", index=False)

print("\n Module 3 Completed Successfully")