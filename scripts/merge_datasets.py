import pandas as pd

def merge_datasets():
    df = pd.read_csv("data/processed/engineered_features.csv")
    df.to_csv("data/processed/final_dataset.csv", index=False)
    print("Final dataset saved → data/processed/final_dataset.csv")