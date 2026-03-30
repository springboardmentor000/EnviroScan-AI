from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("../../../data/processed/labeled_dataset.csv")

le = LabelEncoder()

df.drop(columns=["wind_direction"], inplace=True)

df["pollution_source"] = le.fit_transform(df["pollution_source"])

df.to_csv("../../../data/final_dataset.csv", index=False)

print("Encoding completed")
print(df["pollution_source"].value_counts())
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Encoding Mapping:")
for label, code in mapping.items():
    print(f"{label} -> {code}")
