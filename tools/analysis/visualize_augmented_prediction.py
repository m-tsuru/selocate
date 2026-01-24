import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
import os

# --- Import logic from predict_avg_augmented.py ---
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
    watts = np.maximum(watts, 1e-20)
    return 10 * np.log10(watts) + 30

def idw_interpolation(x, y, z, xi, yi, power=2):
    dist = np.sqrt((xi[:, np.newaxis] - x)**2 + (yi[:, np.newaxis] - y)**2)
    dist = np.maximum(dist, 1e-10)
    weights = 1.0 / (dist ** power)
    weighted_sum = np.sum(weights * z, axis=1)
    sum_weights = np.sum(weights, axis=1)
    return weighted_sum / sum_weights

def run_prediction(model_df, target_df, k=5, grid_density=20):
    unique_bssids = model_df['bssid'].unique()
    valid_bssids = []
    for bssid in unique_bssids:
        bssid_data = model_df[model_df['bssid'] == bssid]
        if len(bssid_data) >= 2 and bssid_data['rssi'].std() > 0:
            valid_bssids.append(bssid)

    print(f"Valid BSSIDs for prediction: {len(valid_bssids)}")

    predictions_map = {}

    for i, bssid in enumerate(valid_bssids):
        model_bssid_df = model_df[model_df['bssid'] == bssid]
        target_bssid_df = target_df[target_df['bssid'] == bssid]

        if len(target_bssid_df) == 0:
            continue

        mx = model_bssid_df['x'].values
        my = model_bssid_df['y'].values
        mz = model_bssid_df['z'].values
        mrssi = model_bssid_df['rssi'].values

        # Augmentation
        x_min, x_max = np.min(mx), np.max(mx)
        y_min, y_max = np.min(my), np.max(my)
        span_x = x_max - x_min
        span_y = y_max - y_min

        augmented_rssi = mrssi.copy()
        augmented_x = mx.copy()
        augmented_y = my.copy()
        augmented_z = mz.copy()

        if span_x > 1e-3 or span_y > 1e-3:
            gx_count = grid_density if span_x > 1e-3 else 1
            gy_count = grid_density if span_y > 1e-3 else 1
            gx = np.linspace(x_min, x_max, gx_count)
            gy = np.linspace(y_min, y_max, gy_count)
            GX, GY = np.meshgrid(gx, gy)
            flat_gx = GX.flatten()
            flat_gy = GY.flatten()

            power_watts = dBm_to_Watts(mrssi)
            interp_watts = idw_interpolation(mx, my, power_watts, flat_gx, flat_gy, power=2)
            interp_rssi = Watts_to_dBm(interp_watts)
            mean_z = np.mean(mz)
            flat_gz = np.full_like(flat_gx, mean_z)

            augmented_rssi = np.concatenate([augmented_rssi, interp_rssi])
            augmented_x = np.concatenate([augmented_x, flat_gx])
            augmented_y = np.concatenate([augmented_y, flat_gy])
            augmented_z = np.concatenate([augmented_z, flat_gz])

        augmented_locs = np.column_stack((augmented_x, augmented_y, augmented_z))

        # Prediction
        for _, row in target_bssid_df.iterrows():
            t = row['t']
            trssi = row['rssi']
            diff = np.abs(augmented_rssi - trssi)
            k_eff = min(k, len(augmented_rssi))
            nearest_indices = np.argsort(diff)[:k_eff]
            nearest_locs = augmented_locs[nearest_indices]
            pred_x, pred_y, pred_z = np.mean(nearest_locs, axis=0)

            if t not in predictions_map:
                predictions_map[t] = []
            predictions_map[t].append([pred_x, pred_y, pred_z])

    # Average results
    final_results = []
    for t, preds in predictions_map.items():
        preds_arr = np.array(preds)
        x_avg, y_avg, z_avg = np.mean(preds_arr, axis=0)
        final_results.append({'t': t, 'pred_x': x_avg, 'pred_y': y_avg, 'pred_z': z_avg})

    return pd.DataFrame(final_results)

def main():
    parser = argparse.ArgumentParser(description="Visualize Augmented Prediction Accuracy")
    parser.add_argument("model_file", help="Trace file for model")
    parser.add_argument("target_file", help="Trace file for target (must have ground truth)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--grid_density", type=int, default=20)

    args = parser.parse_args()

    model_df = load_data(args.model_file)
    target_df = load_data(args.target_file)

    if model_df is None or target_df is None:
        sys.exit(1)

    print("Running Prediction...")
    pred_df = run_prediction(model_df, target_df, k=args.k, grid_density=args.grid_density)

    if pred_df.empty:
        print("No predictions generated.")
        sys.exit(1)

    # Get Ground Truth (Actual) unique by timestamp
    actual_df = target_df.groupby('t')[['x', 'y', 'z']].mean().reset_index()

    # Merge
    # actual_df: [t, x, y, z]
    # pred_df: [t, pred_x, pred_y, pred_z]
    # No conflict in columns other than 't' which is join key.
    # So 'x' remains 'x', 'pred_x' remains 'pred_x'.
    merged = pd.merge(actual_df, pred_df, on='t', how='inner')

    print(f"Matched {len(merged)} timestamps for comparison.")

    # Calculate Error
    merged['error_dist'] = np.sqrt(
        (merged['x'] - merged['pred_x'])**2 +
        (merged['y'] - merged['pred_y'])**2
    )

    print(f"Mean Error: {merged['error_dist'].mean():.4f} m")
    print(f"Max Error: {merged['error_dist'].max():.4f} m")

    # --- PLOT 1: Actual vs Predicted ---
    plt.figure(figsize=(10, 8))
    plt.scatter(merged['x'], merged['y'], c='blue', label='Actual', alpha=0.6, s=30)
    plt.scatter(merged['pred_x'], merged['pred_y'], c='red', label='Predicted', alpha=0.6, s=30)

    # Draw connection lines
    for _, row in merged.iterrows():
        plt.plot([row['x'], row['pred_x']], [row['y'], row['pred_y']], 'gray', alpha=0.3, linewidth=0.5)

    plt.title("Actual vs Predicted Locations")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)

    out_sct = "prediction_comparison.png"
    plt.savefig(out_sct)
    print(f"Saved {out_sct}")
    plt.close()

    # --- PLOT 2: Error Heatmap ---
    plt.figure(figsize=(10, 8))

    x = merged['x'].values
    y = merged['y'].values
    z = merged['error_dist'].values

    # Grid for visualization
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    if x_max - x_min < 1e-3: x_max += 1; x_min -= 1
    if y_max - y_min < 1e-3: y_max += 1; y_min -= 1

    grid_x, grid_y = np.mgrid[
        x_min:x_max:100j,
        y_min:y_max:100j
    ]

    try:
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

        plt.imshow(grid_z.T, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='hot', alpha=0.8)
        cbar = plt.colorbar()
        cbar.set_label('Prediction Error (m)')
        plt.scatter(x, y, c='black', s=5, alpha=0.3)
        plt.title("Prediction Error Heatmap")
        plt.xlabel("X")
        plt.ylabel("Y")

        out_err = "prediction_error_heatmap.png"
        plt.savefig(out_err)
        print(f"Saved {out_err}")
    except Exception as e:
        print(f"Could not generate heatmap (maybe not enough points?): {e}")

if __name__ == "__main__":
    main()
