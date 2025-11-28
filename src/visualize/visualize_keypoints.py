import cv2
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import re
from omegaconf import OmegaConf

###############################################################################
# COMPUTE PROJECT ROOT DYNAMICALLY
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]


###############################################################################
# LOAD CONFIG
###############################################################################

def load_config(dataset: str):
    cfg_path = PROJECT_ROOT / "configs" / f"config_{dataset}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return OmegaConf.load(cfg_path)


###############################################################################
# CAMERA PARSING
###############################################################################

def extract_camera_from_filename(fname: str):
    m = re.search(r"Cam-[A-Z]", fname)
    if not m:
        raise ValueError(f"Camera not found in filename: {fname}")
    return m.group(0)


###############################################################################
# DLC-STYLE CSV LOADING
###############################################################################

def load_dlc_style_csv(csv_path: Path):
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    new_cols = []
    for scorer, kp, coord in df.columns:
        new_cols.append(f"{kp}_{coord}")
    df.columns = new_cols

    df._source_path = csv_path  # store to re-read raw header
    return df.reset_index(drop=True)



###############################################################################
# LOADING FUNCTIONS
###############################################################################

def load_ground_truth(cfg, cam: str, ood=True):
    suffix = "_new" if ood else ""
    gt_path = Path(cfg.data.gt_data_dir) / f"CollectedData_{cam}{suffix}.csv"

    if not gt_path.exists():
        print(f"[WARN] Ground truth missing: {gt_path}")
        return None

    return load_dlc_style_csv(gt_path)


def load_original_predictions(cfg, cam: str, ood=True):
    """
    Path: <preds_data_dir>/predictions_<cam>_new.csv
    """
    suffix = "_new" if ood else ""
    pred_path = Path(cfg.data.preds_data_dir) / f"predictions_{cam}{suffix}.csv"

    if not pred_path.exists():
        print(f"[WARN] Original predictions missing: {pred_path}")
        return None

    return load_dlc_style_csv(pred_path)


def load_refined_predictions(cfg, cam: str, model: str, seed: int, ood=True):
    """
    outputs/<dataset>/<MODEL>_<SEED>/<model_lower>/evaluations/refiner_predictions/
       predictions_Cam-A_new_refined.csv
    """
    suffix = "_new" if ood else ""
    model_upper = f"{model.upper()}_{seed}"
    model_lower = model.lower()

    refined_path = (
        PROJECT_ROOT /
        "outputs" /
        cfg.data.dataset_name /
        model_upper /
        model_lower /
        "evaluations" /
        "refiner_predictions" /
        f"predictions_{cam}{suffix}_refined.csv"
    )

    if not refined_path.exists():
        print(f"[WARN] Refined predictions missing: {refined_path}")
        return None

    return load_dlc_style_csv(refined_path)


###############################################################################
# DRAWING FUNCTIONS
###############################################################################

def overlay_points(frame, pts, keypoints, color, prefix):
    if pts is None:
        return frame

    for kp in keypoints:
        x = pts.get(f"{kp}_x", None)
        y = pts.get(f"{kp}_y", None)
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            continue

        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        cv2.putText(frame, f"{prefix}{kp}",
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return frame


def overlay_skeleton(frame, pts, skeleton, color):
    if pts is None:
        return frame
    for a, b in skeleton:
        xa, ya = pts.get(f"{a}_x", None), pts.get(f"{a}_y", None)
        xb, yb = pts.get(f"{b}_x", None), pts.get(f"{b}_y", None)

        if None in (xa, ya, xb, yb):
            continue
        if np.isnan(xa) or np.isnan(ya) or np.isnan(xb) or np.isnan(yb):
            continue
        cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)

    return frame

def filter_rows_for_video(df, video_prefix):
    """
    Filters DLC-format dataframe rows corresponding to a specific video.
    The first column contains 'labeled-data/<prefix>/imgXXXXX.png'
    """
    # Extract filenames column (first column before flattening)
    # So reload raw CSV with index_col=None:
    
    
    filenames = df.iloc[:, 0].astype(str)  # first column = image file path
    mask = filenames.str.contains(video_prefix)
    df_filtered = df[mask.values].reset_index(drop=True)
    return df_filtered



###############################################################################
# MAIN VISUALIZATION
###############################################################################

def draw_legend(frame):
    """
    Draws a static color legend at the top-right corner of the frame.
    """

    # Legend text + colors (BGR)
    entries = [
        ("Ground Truth", (0, 255, 0)),     # green
        ("Original Pred", (0, 0, 255)),    # red
        ("Refined Pred", (255, 255, 0)),   # cyan/yellow
    ]

    # Positioning
    start_x = frame.shape[1] - 220   # distance from right
    start_y = 20                     # distance from top
    line_height = 25

    # Background box
    cv2.rectangle(
        frame,
        (start_x - 10, start_y - 10),
        (start_x + 200, start_y + 10 + line_height * len(entries)),
        (0, 0, 0),
        thickness=-1
    )

    # Draw each legend item
    for i, (label, color) in enumerate(entries):
        y = start_y + i * line_height

        # Color marker square
        cv2.rectangle(frame, (start_x, y), (start_x + 15, y + 15), color, -1)

        # Text
        cv2.putText(frame, label,
                    (start_x + 25, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def visualize(video_path: str, cfg, model: str, seed: int, ood: bool):

    video_path = Path(video_path)
    fname = video_path.stem

    cam = extract_camera_from_filename(fname)

    print(f"[INFO] Camera   = {cam}")
    print(f"[INFO] Model    = {model.upper()}")
    print(f"[INFO] Seed     = {seed}")
    print(f"[INFO] OOD      = {ood}")
    print(f"[INFO] Fname    = {fname}")

    keypoints = cfg.data.keypoint_names
    skeleton = cfg.data.skeleton

    # Load data (any may be None)
    gt = load_ground_truth(cfg, cam, ood)
    orig = load_original_predictions(cfg, cam, ood)
    refined = load_refined_predictions(cfg, cam, model, seed, ood)

    gt = filter_rows_for_video(gt, fname) if gt is not None else None
    orig = filter_rows_for_video(orig, fname) if orig is not None else None
    refined = filter_rows_for_video(refined, fname) if refined is not None else None

    datasets = [df for df in [gt, orig, refined] if df is not None]
    if len(datasets) == 0:
        raise RuntimeError("No data available to visualize.")

    min_frames = min(len(df) for df in datasets)
    print(f"[INFO] NFrames    = {min_frames}")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model_folder = f"{model.upper()}_{seed}"
    out_dir = PROJECT_ROOT / "outputs" / "visualizations" / model_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{fname}_overlay.mp4"

    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

    for idx in range(min_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame.ndim < 3 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        gt_pts = gt.iloc[idx].to_dict() if gt is not None else None
        orig_pts = orig.iloc[idx].to_dict() if orig is not None else None
        refined_pts = refined.iloc[idx].to_dict() if refined is not None else None

        # skeletons
        frame = overlay_skeleton(frame, gt_pts, skeleton, (0, 200, 0))
        frame = overlay_skeleton(frame, orig_pts, skeleton, (0, 0, 200))
        frame = overlay_skeleton(frame, refined_pts, skeleton, (200, 200, 0))

        # points
        frame = overlay_points(frame, gt_pts, keypoints, (0, 255, 0), "GT:")
        frame = overlay_points(frame, orig_pts, keypoints, (0, 0, 255), "O:")
        frame = overlay_points(frame, refined_pts, keypoints, (255, 255, 0), "R:")

        frame = draw_legend(frame)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[DONE] Saved visualization → {out_path}")


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True,
                        help="Path to the .mp4 video")

    parser.add_argument("--dataset", required=True,
                        choices=["fly", "chickadee"])

    parser.add_argument("--model", required=True,
                        choices=["MLP", "GNN"])

    parser.add_argument("--seed", required=True, type=int)

    parser.add_argument("--ood", default=True, action="store_true",
                        help="Use _new files (default=True)")

    args = parser.parse_args()

    cfg = load_config(args.dataset)
    visualize(args.video, cfg, args.model, args.seed, args.ood)
