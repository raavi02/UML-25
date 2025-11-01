"""Column name schemas & simple validators."""
import pandas as pd
KEYPOINT_NAMES = [
    'L1A', 'L1B', 'L1C', 'L1D', 'L1E',
    'L2A', 'L2B', 'L2C', 'L2D', 'L2E', 
    'L3A', 'L3B', 'L3C', 'L3D', 'L3E',
    'R1A', 'R1B', 'R1C', 'R1D', 'R1E',
    'R2A', 'R2B', 'R2C', 'R2D', 'R2E',
    'R3A', 'R3B', 'R3C', 'R3D', 'R3E'
]

# Ground truth has 60 columns (30 keypoints × 2 coordinates)
GT_COLUMNS = [f"{kp}_{coord}" for kp in KEYPOINT_NAMES for coord in ['x', 'y']]

# Predictions have 90 columns (30 keypoints × 3: x, y, likelihood)
PRED_COLUMNS = [f"{kp}_{coord}" for kp in KEYPOINT_NAMES for coord in ['x', 'y', 'likelihood']]

def validate_gt_columns(df: pd.DataFrame) -> bool:
    """Validate ground truth dataframe has correct columns."""
    return all(col in df.columns for col in GT_COLUMNS)

def validate_pred_columns(df: pd.DataFrame) -> bool:
    """Validate predictions dataframe has correct columns."""
    return all(col in df.columns for col in PRED_COLUMNS)