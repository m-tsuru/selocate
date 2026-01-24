import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import sys
import os

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Check for required fields
    required_cols = ['t', 'bssid', 'rssi', 'y']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return df

def preprocess_data(df):
    print("Preprocessing data (Y only)...")

    # Pivot features
    features = df.pivot_table(index='t', columns='bssid', values='rssi', aggfunc='mean')
    features = features.fillna(-100)

    # Get target (y only)
    targets = df.groupby('t')['y'].mean()

    # Align
    dataset = features.join(targets, how='inner')

    X = dataset[features.columns]
    y = dataset['y'] # Series (1D)

    print(f"Dataset shape: {dataset.shape}")
    print(f"Features: {X.shape[1]} BSSIDs")

    return X, y

def train_model(X, y):
    print("Training Random Forest Regressor (Y only)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    print("Training complete.")
    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_y.py <trace_file.json> [model_output.joblib]")
        # fallback for dev
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        input_file = os.path.join(project_root, 'trace-1769108851.json')
        output_file = os.path.join(script_dir, 'model_y.joblib')
        if not os.path.exists(input_file):
             print("Error: Input file not specified and default trace file not found.")
             sys.exit(1)
    else:
        input_file = sys.argv[1]
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
        else:
            output_file = "model_y.joblib"

    try:
        df = load_data(input_file)
        X, y = preprocess_data(df)

        # Train
        model = train_model(X, y)

        # Save model and feature names
        artifact = {
            'model': model,
            'features': X.columns.tolist()
        }

        print(f"Saving model to {output_file}...")
        joblib.dump(artifact, output_file)
        print("Done.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
