import pandas as pd

def export_to_excel():
    print("Converting final dataset to Excel...")

    df = pd.read_csv("data/processed/final_dataset.csv")

    df.to_excel("data/processed/final_dataset.xlsx", index=False)

    print("Excel file created successfully")