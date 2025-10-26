import numpy as np
import pandas as pd
import os

# --- Configuration ---
num_frames = 10
bodyparts = [
    "L1A", "L1B", "L1C", "L1D", "L1E",
    "L2A", "L2B", "L2C", "L2D", "L2E",
    "L3A", "L3B", "L3C", "L3D", "L3E",
    "R1A", "R1B", "R1C", "R1D", "R1E",
    "R2A", "R2B", "R2C", "R2D", "R2E",
    "R3A", "R3B", "R3C", "R3D", "R3E"
]
scorer = "anipose"
noise_std = 3.0  # Gaussian noise standard deviation in pixels [for predictions]

# --- Helper to generate CSV in anipose-style multi-index format ---
def generate_pose_csv(cam_name, output_path, noisy=False, base_coords=None):
    cols = pd.MultiIndex.from_tuples(
        [(scorer, bp, c) for bp in bodyparts for c in ["x", "y"]],
        names=["scorer", "bodyparts", "coords"]
    )
    
    data = []
    for f in range(num_frames):
        # Simulate (x,y) coordinates for each bodypart
        if base_coords is None:
            # Random GT around some arbitrary image center
            frame_coords = np.random.uniform(100, 400, size=(len(bodyparts), 2))
        else:
            frame_coords = base_coords[f]
        
        if noisy:
            frame_coords += np.random.normal(0, noise_std, frame_coords.shape)
        
        data.append(frame_coords.flatten())
    
    index = [f"labeled-data/frame_{f:08d}_{cam_name}.png" for f in range(num_frames)]
    df = pd.DataFrame(data, index=index, columns=cols)
    df.to_csv(output_path)
    print(f"Saved: {output_path}")

# --- Generate ground-truth for both cams ---
base_folder = "fly_toy"
os.makedirs(base_folder, exist_ok=True)
base_cam1 = [np.random.uniform(100, 400, size=(len(bodyparts), 2)) for _ in range(num_frames)]
base_cam2 = [coords + np.random.normal(0, 10, coords.shape) for coords in base_cam1]  # slightly different view

generate_pose_csv("CamA_GT", f"{base_folder}/CamA_GT.csv", noisy=False, base_coords=base_cam1)
generate_pose_csv("CamB_GT", f"{base_folder}/CamB_GT.csv", noisy=False, base_coords=base_cam2)

# --- Generate predictions (GT + noise) ---
generate_pose_csv("CamA_pred", f"{base_folder}/CamA_pred.csv", noisy=True, base_coords=base_cam1)
generate_pose_csv("CamB_pred", f"{base_folder}/CamB_pred.csv", noisy=True, base_coords=base_cam2)
