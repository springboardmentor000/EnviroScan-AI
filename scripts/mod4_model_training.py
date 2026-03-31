import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    print("Training model...")

    df = pd.read_csv("data/processed/labeled_dataset.csv")

    features = ["pm2_5", "no2", "so2", "o3"]
    X = df[features]
    y = df["pollution_source"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Model Accuracy:", accuracy)

    joblib.dump(model, "model.pkl")

    print("Model saved as model.pkl")


if __name__ == "__main__":
    train_model()