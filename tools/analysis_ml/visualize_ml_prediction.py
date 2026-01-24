import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.interpolate import griddata
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
    parser = argparse.ArgumentParser(description="Visualize ML Prediction Accuracy")
    parser.add_argument("model_file", help="Path to joblib model file")
    parser.add_argument("target_file", help="Path to JSON trace file for prediction (target)")

    args = parser.parse_args()

    # 1. Load Model
    print(f"Loading model from {args.model_file}...")
    try:
        artifact = joblib.load(args.model_file)
        model = artifact['model']
        feature_names = artifact['features']
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # 2. Load Target Data
    df = load_data(args.target_file)
    if df is None:
        sys.exit(1)

    print("Preprocessing...")
    # Pivot to match training features
    features = df.pivot_table(index='t', columns='bssid', values='rssi', aggfunc='mean')

    # Reindex to match model features
    X = features.reindex(columns=feature_names, fill_value=-100)

    print(f"Prediction dataset shape: {X.shape}")

    # Predict
    print("Predicting...")
    preds = model.predict(X)

    # Handle output shape (N, 3) or (N, )
    if preds.ndim == 1:
        # Assuming only Y was predicted? User asked for verification of predict.py which is (x,y,z) usually
        # But if they passed a Y-only model, this script might crash if it expects 3 cols.
        # Let's handle general case if possible, but for visualize which needs X/Y, we assume X/Y exist.
        print("Model predicted 1D output. Cannot visualize 2D map comparison.")
        sys.exit(1)

    pred_df = pd.DataFrame(preds, columns=['pred_x', 'pred_y', 'pred_z'], index=X.index)

    # 3. Ground Truth
    actual_df = df.groupby('t')[['x', 'y', 'z']].mean()

    # Merge (join on index 't')
    merged = pd.merge(actual_df, pred_df, left_index=True, right_index=True)

    print(f"Matched {len(merged)} timestamps.")

    # 4. Calculate Error
    merged['error_dist'] = np.sqrt(
        (merged['x'] - merged['pred_x'])**2 +
        (merged['y'] - merged['pred_y'])**2
    )

    print(f"Mean Error: {merged['error_dist'].mean():.4f} m")
    print(f"Max Error: {merged['error_dist'].max():.4f} m")

    # 5. Plot 1: Comparison
    plt.figure(figsize=(10, 8))
    plt.scatter(merged['x'], merged['y'], c='blue', label='Actual', alpha=0.6, s=30)
    plt.scatter(merged['pred_x'], merged['pred_y'], c='red', label='Predicted', alpha=0.6, s=30)

    for t, row in merged.iterrows():
        plt.plot([row['x'], row['pred_x']], [row['y'], row['pred_y']], 'gray', alpha=0.3, linewidth=0.5)

    plt.title("ML Prediction: Actual vs Predicted")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)

    out_sct = "ml_prediction_comparison.png"
    plt.savefig(out_sct)
    print(f"Saved {out_sct}")
    plt.close()

    # 6. Plot 2: Error Heatmap
    if len(merged) >= 4: # Need enough points for interpolation
        plt.figure(figsize=(10, 8))
        x = merged['x'].values
        y = merged['y'].values
        z = merged['error_dist'].values

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
            plt.title("ML Prediction Error Heatmap")
            plt.xlabel("X")
            plt.ylabel("Y")

            out_err = "ml_prediction_error_heatmap.png"
            plt.savefig(out_err)
            print(f"Saved {out_err}")
        except Exception as e:
            print(f"Heatmap generation failed: {e}")

if __name__ == "__main__":
    main()
