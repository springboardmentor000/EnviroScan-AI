# scripts/export_excel.py

import pandas as pd

def export_excel():

    df = pd.read_csv("data/processed/final_dataset.csv")
    df.to_excel("data/processed/final_dataset.xlsx", index=False)

    print("✔ Excel exported (20 rows)")