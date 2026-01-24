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
    parser = argparse.ArgumentParser(description="Predict location by averaging predictions from multiple BSSIDs")
    parser.add_argument("model_file", help="Path to JSON trace file to use as model")
    parser.add_argument("target_file", help="Path to JSON trace file to predict")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors per BSSID")
    parser.add_argument("--output", help="Output file for averaged predictions (CSV)", default=None)

    args = parser.parse_args()

    # Load data
    model_df = load_data(args.model_file)
    target_df = load_data(args.target_file)

    if model_df is None or target_df is None:
        sys.exit(1)

    # 1. Identify valid BSSIDs in model (non-constant RSSI)
    print(" Analyzing model BSSIDs...")
    unique_bssids = model_df['bssid'].unique()
    valid_bssids = []

    for bssid in unique_bssids:
        bssid_data = model_df[model_df['bssid'] == bssid]
        if len(bssid_data) < 2:
            continue

        rssi_std = bssid_data['rssi'].std()

        # Check if RSSI varies. std can be NaN if only 1 point (handled above), or 0 if constant.
        if rssi_std > 0:
            valid_bssids.append(bssid)
        else:
            # print(f"  Skipping {bssid}: Constant RSSI (std=0)")
            pass

    print(f"Found {len(unique_bssids)} unique BSSIDs. {len(valid_bssids)} are valid (non-constant RSSI).")

    # Structure to hold predictions: keyed by timestamp
    # timestamp -> list of [x, y, z]
    predictions_map = {}

    # 2. Iterate each valid BSSID and predict
    print("Running predictions per BSSID...")

    for i, bssid in enumerate(valid_bssids):
        # Progress check
        # if i % 10 == 0: print(f" Processing {i}/{len(valid_bssids)}...")

        # Get Model Data for this BSSID
        model_bssid_df = model_df[model_df['bssid'] == bssid]
        target_bssid_df = target_df[target_df['bssid'] == bssid]

        if len(target_bssid_df) == 0:
            continue

        model_rssi = model_bssid_df['rssi'].values
        model_locs = model_bssid_df[['x', 'y', 'z']].values

        # Predict for each target occurrence
        for _, row in target_bssid_df.iterrows():
            t = row['t']
            target_rssi = row['rssi']

            # 1D-KNN
            diff = np.abs(model_rssi - target_rssi)
            # Ensure k is not larger than dataset
            k = min(args.k, len(model_rssi))

            nearest_indices = np.argsort(diff)[:k]
            nearest_locs = model_locs[nearest_indices]

            pred_x, pred_y, pred_z = np.mean(nearest_locs, axis=0)

            if t not in predictions_map:
                predictions_map[t] = []

            predictions_map[t].append([pred_x, pred_y, pred_z])

    # 3. Average per timestamp
    print("Averaging results...")
    final_results = []

    sorted_timestamps = sorted(predictions_map.keys())

    for t in sorted_timestamps:
        preds = np.array(predictions_map[t])
        # Average
        x_avg, y_avg, z_avg = np.mean(preds, axis=0)

        final_results.append({
            't': t,
            'pred_x': x_avg,
            'pred_y': y_avg,
            'pred_z': z_avg,
            'contributors': len(preds) # How many BSSIDs contributed
        })

    result_df = pd.DataFrame(final_results)

    print("\nAveraged Predictions (first 5):")
    print(result_df.head())

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"\nSaved averaged predictions to {args.output}")
    else:
        base = os.path.basename(args.target_file)
        out_name = f"{base}.avg_pred.csv"
        result_df.to_csv(out_name, index=False)
        print(f"\nSaved averaged predictions to {out_name}")

if __name__ == "__main__":
    main()
