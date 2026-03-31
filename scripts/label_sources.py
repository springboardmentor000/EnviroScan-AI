# scripts/label_sources.py

import pandas as pd

def label_sources():

    df = pd.read_csv("data/processed/engineered_data.csv")

    def classify(row):

        # Vehicular (if near roads and some NO2 present)
        if row["roads_count"] > 1000 and row["NO2"] >= 1:
            return "Vehicular"

        # Industrial (if near industry and SO2 present)
        elif row["industrial_count"] > 0 and row["SO2"] >= 1:
            return "Industrial"

        # Agricultural (if farmland nearby and PM present)
        elif row["farmland_count"] > 0 and row["PM2.5"] >= 1:
            return "Agricultural"

        # Otherwise natural
        else:
            return "Natural"

    df["pollution_source"] = df.apply(classify, axis=1)

    df.to_csv("data/processed/final_dataset.csv", index=False)

    print("✔ Pollution source labeled successfully")