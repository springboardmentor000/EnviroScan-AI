# ==========================================
# MODULE 3: SOURCE LABELING AND SIMULATION
# ==========================================

import pandas as pd

# ==============================
# LOAD CLEAN DATASET
# ==============================

df = pd.read_csv("clean_environment_dataset.csv")

print("Dataset loaded:", df.shape)


# ==============================
# SOURCE LABELING FUNCTION
# ==============================

def label_source(row):

    # Vehicular pollution
    if row["dist_to_road"] < 0.005 and row["no2"] > 0.6:
        return "Vehicular"

    # Industrial pollution
    elif row["dist_to_industry"] < 0.005 and row["so2"] > 0.6:
        return "Industrial"

    # Agricultural pollution
    elif row["pm25"] > 0.6 and row["season"] == 2:
        return "Agricultural"

    # Burning pollution
    elif row["pm25"] > 0.7 and row["dist_to_dump"] < 0.005:
        return "Burning"

    # Natural pollution
    else:
        return "Natural"


# ==============================
# APPLY LABELING RULES
# ==============================

df["source_label"] = df.apply(label_source, axis=1)


# ==============================
# CHECK LABEL DISTRIBUTION
# ==============================

print(df["source_label"].value_counts())


# ==============================
# SAVE LABELED DATASET
# ==============================

df.to_csv("labeled_environment_dataset.csv", index=False)

print("Labeled dataset saved successfully!")

print(df.head())