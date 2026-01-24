import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def idw_interpolation(x, y, z, xi, yi, power=2):
    """
    Inverse Distance Weighting interpolation.
    """
    # xi, yi are grid coordinates (meshgrids)
    # x, y, z are known data points

    # Flatten grid for calculation
    xi_flat = xi.flatten()
    yi_flat = yi.flatten()

    # Calculate distances between all grid points and all known points
    # dist shape: (n_grid_points, n_known_points)
    # This can be memory intensive if grid is very large.
    # For this task, assuming reasonable size.

    # Optimization: iterate over grid points or known points depending on counts?
    # Vectorized approach:
    dist = np.sqrt(
        (xi_flat[:, np.newaxis] - x) ** 2 + (yi_flat[:, np.newaxis] - y) ** 2
    )

    # Avoid division by zero
    dist = np.maximum(dist, 1e-10)

    weights = 1.0 / (dist**power)

    # Calculate weighted average
    zi_flat = np.sum(weights * z, axis=1) / np.sum(weights, axis=1)

    return zi_flat.reshape(xi.shape)


def dBm_to_Watts(rssi_dbm):
    """
    Convert RSSI (dBm) to Power (Watts).
    Formula: P(W) = 10^((dBm - 30) / 10)
    """
    return 10 ** ((rssi_dbm - 30) / 10)


def main():
    # Input file configuration
    # Assuming the script is run from project root, or we look for the file relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    trace_file_path = os.path.join(project_root, "trace-1769142448.json")

    if not os.path.exists(trace_file_path):
        # Fallback check current directory or args?
        if len(sys.argv) > 1:
            trace_file_path = sys.argv[1]
        else:
            print(f"Error: Trace file not found at {trace_file_path}")
            return

    print(f"Loading data from {trace_file_path}...")
    try:
        with open(trace_file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return

    # Organize data by BSSID
    # Structure: bssid -> list of {x, y, rssi, ssid}
    bssid_data = {}

    for point in data:
        # Check for required fields
        if (
            "bssid" not in point
            or "x" not in point
            or "y" not in point
            or "rssi" not in point
        ):
            continue

        bssid = point["bssid"]
        if not bssid:
            continue

        if bssid not in bssid_data:
            bssid_data[bssid] = {"ssid": point.get("ssid", "Unknown"), "points": []}

        bssid_data[bssid]["points"].append(
            {"x": point["x"], "y": point["y"], "rssi": point["rssi"]}
        )

    # Prepare output directory
    output_dir = script_dir
    os.makedirs(output_dir, exist_ok=True)

    # Process each BSSID
    for bssid, info in bssid_data.items():
        points = info["points"]
        ssid = info["ssid"]

        if len(points) < 3:
            print(f"Skipping {bssid} ({ssid}): Not enough points ({len(points)})")
            continue

        print(f"Processing {bssid} ({ssid})... {len(points)} points")

        x = np.array([p["x"] for p in points])
        y = np.array([p["y"] for p in points])
        rssi = np.array([p["rssi"] for p in points])

        # Convert RSSI to Watts
        power_watts = dBm_to_Watts(rssi)

        # Define Grid
        # Create a grid with some margin around the data
        margin = (
            1.0  # meters? Unit unclear but x,y seem small float values based on preview
        )
        # Checking preview: x=0.0, y=0.0 mostly, then x=-0.02, y=0.06. Assuming meters.
        # If points are very close, margin should be small.
        # Let's adjust margin dynamically.

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        span_x = x_max - x_min
        span_y = y_max - y_min

        # If span is 0 (all points at same location), we can't make a heatmap
        if span_x < 1e-3 and span_y < 1e-3:
            print(f"Skipping {bssid}: All points at same location.")
            continue

        padding = max(span_x, span_y) * 0.1
        if padding == 0:
            padding = 1.0

        grid_x_min = x_min - padding
        grid_x_max = x_max + padding
        grid_y_min = y_min - padding
        grid_y_max = y_max + padding

        # Resolution
        # Create 100x100 grid or similar
        resolution = 100
        xi = np.linspace(grid_x_min, grid_x_max, resolution)
        yi = np.linspace(grid_y_min, grid_y_max, resolution)
        XI, YI = np.meshgrid(xi, yi)

        # Interpolate
        # ZI = interpolated power in Watts
        ZI = idw_interpolation(x, y, power_watts, XI, YI, power=2)

        # Plotting
        plt.figure(figsize=(10, 8))

        # Use log scale for color since power drops off rapidly?
        # User asked to convert to Watts (Access Number/Linear) for drawing.
        # "W（真数）に直して描画してください" -> Draw using Watts.
        # It might look very peaked.

        # Plot heatmap
        plt.pcolormesh(XI, YI, ZI, shading="auto", cmap="viridis")
        cbar = plt.colorbar()
        cbar.set_label("Signal Power (Watts)")

        # Overlay original points
        plt.scatter(x, y, c="red", s=20, edgecolors="black", label="Sample Points")

        plt.title(f"Heatmap for {ssid}\n({bssid})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.axis("equal")

        # Save filename formatting
        safe_bssid = bssid.replace(":", "-")
        outfile = os.path.join(output_dir, f"heatmap_{safe_bssid}.png")
        plt.savefig(outfile)
        plt.close()

        print(f"Saved {outfile}")


if __name__ == "__main__":
    main()
