import json
import argparse
import numpy as np
import pandas as pd
import sys
import os

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Predict location based on single BSSID RSSI")
    parser.add_argument("model_file", help="Path to JSON trace file to use as model")
    parser.add_argument("target_file", help="Path to JSON trace file to predict")
    parser.add_argument("bssid", help="BSSID to use for prediction")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors (rssi space) to average")
    parser.add_argument("--output", help="Output file for predictions (CSV)", default=None)

    args = parser.parse_args()

    # Load data
    model_df = load_data(args.model_file)
    target_df = load_data(args.target_file)

    if model_df is None or target_df is None:
        sys.exit(1)

    # Filter by BSSID
    model_bssid = model_df[model_df['bssid'] == args.bssid].copy()
    target_bssid = target_df[target_df['bssid'] == args.bssid].copy()

    print(f"Model data for {args.bssid}: {len(model_bssid)} points")
    print(f"Target data for {args.bssid}: {len(target_bssid)} points")

    if len(model_bssid) == 0:
        print(f"Error: BSSID {args.bssid} not found in model file.")
        sys.exit(1)

    if len(target_bssid) == 0:
        print(f"Error: BSSID {args.bssid} not found in target file.")
        sys.exit(1)

    # Prediction Loop
    predictions = []

    # We want to map RSSI -> (x, y, z)
    # Since RSSI is discrete and noisy, we find k entries in model with closest RSSI
    # Optimization: Sort model by RSSI for faster lookup? Or just linear scan since N is small?
    # Assuming trace files are not huge, linear scan or vectorization is fine.

    model_rssi = model_bssid['rssi'].values
    model_locs = model_bssid[['x', 'y', 'z']].values

    for idx, row in target_bssid.iterrows():
        target_rssi = row['rssi']
        t = row['t']

        # Calculate RSSI difference
        diff = np.abs(model_rssi - target_rssi)

        # Find k nearest based on rssi difference
        # argsort gives indices of smallest diffs
        nearest_indices = np.argsort(diff)[:args.k]

        # Get corresponding locations
        nearest_locs = model_locs[nearest_indices]

        # Average
        pred_x, pred_y, pred_z = np.mean(nearest_locs, axis=0)

        predictions.append({
            't': t,
            'bssid': args.bssid,
            'target_rssi': target_rssi,
            'pred_x': pred_x,
            'pred_y': pred_y,
            'pred_z': pred_z
        })

    # Result DataFrame
    result_df = pd.DataFrame(predictions)

    print("\nPredictions (first 5):")
    print(result_df[['t', 'target_rssi', 'pred_x', 'pred_y']].head())

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"\nSaved predictions to {args.output}")
    else:
        # Default output name
        base = os.path.basename(args.target_file)
        safe_bssid = args.bssid.replace(':', '-')
        out_name = f"{base}.{safe_bssid}.pred.csv"
        result_df.to_csv(out_name, index=False)
        print(f"\nSaved predictions to {out_name}")

if __name__ == "__main__":
    main()
