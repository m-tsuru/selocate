import json
import numpy as np
import pandas as pd
import joblib
import sys
import os

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Check for required fields for prediction (t, bssid, rssi)
    required_cols = ['t', 'bssid', 'rssi']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return df

def predict_location(df, model, feature_names):
    print("Preprocessing for prediction...")

    # Pivot to create feature vectors
    # Same as training: Index=t, Columns=bssid, Values=rssi
    features = df.pivot_table(index='t', columns='bssid', values='rssi', aggfunc='mean')

    # Reindex to match the training feature set
    # - missing BSSIDs in input -> filled with -100
    # - extra BSSIDs in input -> dropped
    X = features.reindex(columns=feature_names, fill_value=-100)

    print(f"Prediction dataset shape: {X.shape}")

    # Predict
    preds = model.predict(X)

    # Combine with timestamps for result
    results = pd.DataFrame(preds, columns=['pred_x', 'pred_y', 'pred_z'], index=X.index)

    return results

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict.py <trace_file.json> <model_file.joblib>")
        sys.exit(1)

    input_file = sys.argv[1]
    model_file = sys.argv[2]

    try:
        # Load model artifact
        print(f"Loading model from {model_file}...")
        artifact = joblib.load(model_file)
        model = artifact['model']
        feature_names = artifact['features']

        # Load and predict
        df = load_data(input_file)
        results = predict_location(df, model, feature_names)

        print("\nPredictions (first 5):")
        print(results.head())

        # Optional: Save to CSV
        output_csv = input_file + ".predictions.csv"
        results.to_csv(output_csv)
        print(f"\nSaved predictions to {output_csv}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
