import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import sys
import os

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Check for required fields
    required_cols = ['t', 'bssid', 'rssi', 'x', 'y', 'z']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return df

def preprocess_data(df):
    print("Preprocessing data...")

    # 1. Group by timestamp 't' to create fingerprint vectors
    # We want a single row per timestamp with columns for each BSSID's RSSI
    # And target columns for x, y, z (taking the mean if there are slight variations, or first)

    # Pivot: Index=t, Columns=bssid, Values=rssi
    # Note: Aggregate function 'mean' handles duplicates (same bssid at same time? shouldn't happen usually but robust)
    features = df.pivot_table(index='t', columns='bssid', values='rssi', aggfunc='mean')

    # Fill missing BSSIDs with a low RSSI value (e.g., -100 dBm)
    features = features.fillna(-100)

    # Get targets (x, y, z) for each timestamp
    # Assuming x, y, z are constant for a given timestamp
    targets = df.groupby('t')[['x', 'y', 'z']].mean()

    # Align index just in case
    # join='inner' ensures we only keep timestamps that have both features and targets
    dataset = features.join(targets, how='inner')

    X = dataset[features.columns]
    y = dataset[['x', 'y', 'z']]

    print(f"Dataset shape: {dataset.shape}")
    print(f"Features: {X.shape[1]} BSSIDs")

    return X, y

def train_model(X, y):
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    print("Training complete.")
    return model

def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py <trace_file.json> [model_output.joblib]")
        # fallback for dev
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        input_file = os.path.join(project_root, 'trace-1769108851.json')
        output_file = os.path.join(script_dir, 'model.joblib')
        if not os.path.exists(input_file):
             print("Error: Input file not specified and default trace file not found.")
             sys.exit(1)
    else:
        input_file = sys.argv[1]
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
        else:
            output_file = "model.joblib"

    try:
        df = load_data(input_file)
        X, y = preprocess_data(df)

        # Train
        model = train_model(X, y)

        # Save model and feature names
        # We need the feature names (BSSIDs) to ensure the prediction input has same columns in same order
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
