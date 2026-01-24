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

def predict_location_y(df, model, feature_names):
    print("Preprocessing for prediction (Y only)...")

    # Pivot to create feature vectors
    features = df.pivot_table(index='t', columns='bssid', values='rssi', aggfunc='mean')

    # Reindex to match the training feature set
    X = features.reindex(columns=feature_names, fill_value=-100)

    print(f"Prediction dataset shape: {X.shape}")

    # Predict
    # Model might predict [x, y, z] or just [y] depending on how it was trained
    preds = model.predict(X)

    # Check shape
    if preds.ndim == 1:
        # Assuming the model was trained specifically for Y
        preds_y = preds
    else:
        # Index 1 is Y
        preds_y = preds[:, 1]

    # Combine with timestamps for result
    results = pd.DataFrame(preds_y, columns=['pred_y'], index=X.index)

    return results

def main():
    if len(sys.argv) < 3:
        print("Usage: python predict_y.py <trace_file.json> <model_file.joblib>")
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
        results = predict_location_y(df, model, feature_names)

        print("\nPredictions (Y only, first 5):")
        print(results.head())

        # Optional: Save to CSV
        output_csv = input_file + ".predictions_y.csv"
        results.to_csv(output_csv)
        print(f"\nSaved Y-predictions to {output_csv}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
