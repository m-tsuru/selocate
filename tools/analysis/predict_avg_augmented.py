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

def dBm_to_Watts(rssi_dbm):
    return 10 ** ((rssi_dbm - 30) / 10)

def Watts_to_dBm(watts):
    # Avoid log(0)
    watts = np.maximum(watts, 1e-20)
    return 10 * np.log10(watts) + 30

def idw_interpolation(x, y, z, xi, yi, power=2):
    """
    Simple IDW for 1D output (z).
    x, y, z: known points (1D arrays)
    xi, yi: grid points (1D arrays or scalars)
    Returns zi (interpolated values at xi, yi)
    """
    # Calculate distances
    # optimizing for vectorization
    # xi, yi should be broadcastable against x, y

    # Reshape for broadcasting
    # known: (N,)
    # query: (M,) -> (M, 1)

    dist = np.sqrt((xi[:, np.newaxis] - x)**2 + (yi[:, np.newaxis] - y)**2)
    dist = np.maximum(dist, 1e-10) # Avoid zero division

    weights = 1.0 / (dist ** power)

    # Weighted average
    # sum(weights * known_z) / sum(weights)
    weighted_sum = np.sum(weights * z, axis=1)
    sum_weights = np.sum(weights, axis=1)

    return weighted_sum / sum_weights

def main():
    parser = argparse.ArgumentParser(description="Predict location with heatmap-augmented data")
    parser.add_argument("model_file", help="Path to JSON trace file to use as model")
    parser.add_argument("target_file", help="Path to JSON trace file to predict")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")
    parser.add_argument("--grid_density", type=int, default=20, help="Number of steps for interpolation grid (per axis)")
    parser.add_argument("--output", help="Output file (CSV)", default=None)

    args = parser.parse_args()

    # Load data
    model_df = load_data(args.model_file)
    target_df = load_data(args.target_file)

    if model_df is None or target_df is None:
        sys.exit(1)

    # 1. Identify valid BSSIDs
    unique_bssids = model_df['bssid'].unique()
    valid_bssids = []

    for bssid in unique_bssids:
        bssid_data = model_df[model_df['bssid'] == bssid]
        if len(bssid_data) >= 2 and bssid_data['rssi'].std() > 0:
            valid_bssids.append(bssid)

    print(f"Valid BSSIDs: {len(valid_bssids)} / {len(unique_bssids)}")

    predictions_map = {}

    # 2. Process each valid BSSID (Augment + Predict)
    print("Augmenting and Predicting...")

    for i, bssid in enumerate(valid_bssids):
        # Model Data
        model_bssid_df = model_df[model_df['bssid'] == bssid]
        target_bssid_df = target_df[target_df['bssid'] == bssid]

        if len(target_bssid_df) == 0:
            continue

        # Coordinates and Values
        mx = model_bssid_df['x'].values
        my = model_bssid_df['y'].values
        mz = model_bssid_df['z'].values
        mrssi = model_bssid_df['rssi'].values

        # --- AUGMENTATION ---
        x_min, x_max = np.min(mx), np.max(mx)
        y_min, y_max = np.min(my), np.max(my)

        # Only augment if there is some area to cover
        span_x = x_max - x_min
        span_y = y_max - y_min

        augmented_rssi = mrssi.copy()
        augmented_x = mx.copy()
        augmented_y = my.copy()
        augmented_z = mz.copy() # z is just carried over

        if span_x > 1e-3 or span_y > 1e-3:
            # Create Grid
            # Use linspace strictly within bounds
            # If span is 0 in one direction (line), we still interpolate along the other?
            # Or just skip augmentation if strictly 0 area?
            # Let's try to grid whatever dimension has spread.

            gx_count = args.grid_density if span_x > 1e-3 else 1
            gy_count = args.grid_density if span_y > 1e-3 else 1

            gx = np.linspace(x_min, x_max, gx_count)
            gy = np.linspace(y_min, y_max, gy_count)

            GX, GY = np.meshgrid(gx, gy)
            flat_gx = GX.flatten()
            flat_gy = GY.flatten()

            # Interpolate RSSI (in Watts)
            power_watts = dBm_to_Watts(mrssi)
            interp_watts = idw_interpolation(mx, my, power_watts, flat_gx, flat_gy, power=2)
            interp_rssi = Watts_to_dBm(interp_watts)

            # Interpolate Z? Or just use mean Z? Usually Z is 0 or constant per floor.
            # Let's use mean Z of the BSSID samples.
            mean_z = np.mean(mz)
            flat_gz = np.full_like(flat_gx, mean_z)

            # Append to augmented arrays
            augmented_rssi = np.concatenate([augmented_rssi, interp_rssi])
            augmented_x = np.concatenate([augmented_x, flat_gx])
            augmented_y = np.concatenate([augmented_y, flat_gy])
            augmented_z = np.concatenate([augmented_z, flat_gz])

        # Combine into (N, 3) location array
        augmented_locs = np.column_stack((augmented_x, augmented_y, augmented_z))

        # --- PREDICTION (KNN) ---
        for _, row in target_bssid_df.iterrows():
            t = row['t']
            trssi = row['rssi']

            diff = np.abs(augmented_rssi - trssi)

            # k can be larger now since we have more points
            k = min(args.k, len(augmented_rssi))

            nearest_indices = np.argsort(diff)[:k]
            nearest_locs = augmented_locs[nearest_indices]

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
        x_avg, y_avg, z_avg = np.mean(preds, axis=0)

        final_results.append({
            't': t,
            'pred_x': x_avg,
            'pred_y': y_avg,
            'pred_z': z_avg,
            'contributors': len(preds)
        })

    result_df = pd.DataFrame(final_results)

    print("\nAveraged Augmented Predictions (first 5):")
    print(result_df.head())

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
    else:
        base = os.path.basename(args.target_file)
        out_name = f"{base}.aug_pred.csv"
        result_df.to_csv(out_name, index=False)
        print(f"\nSaved to {out_name}")

if __name__ == "__main__":
    main()
