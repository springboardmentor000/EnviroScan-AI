import pandas as pd


def label_sources():
    print("🏷 Adding pollution sources...")

    df = pd.read_csv("data/processed/final_dataset.csv")

    def assign_source(row):

        if row["no2"] > 30:
            return "Vehicular"

        elif row["so2"] > 20:
            return "Industrial"

        elif row["pm2_5"] > 80:
            return "Agricultural"

        elif row["o3"] > 50:
            return "Natural"

        else:
            return "Mixed"

    df["pollution_source"] = df.apply(assign_source, axis=1)

    df.to_csv("data/processed/labeled_dataset.csv", index=False)

    print("✔ Pollution sources added")