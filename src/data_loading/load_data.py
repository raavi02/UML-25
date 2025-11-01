from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
import glob

import pandas as pd
import numpy as np

from src.data_loading.schemas import GT_COLUMNS, PRED_COLUMNS, KEYPOINT_NAMES, validate_gt_columns, validate_pred_columns
from src.utils.logging import logger

def load_gt_data(base_path: str = "data/fly", ood: bool = False) -> Dict[str, pd.DataFrame]:
    """Load ground truth data for all cameras."""
    gt_data = {}
    suffix = "_new" if ood else ""
    
    for cam in ['A', 'B', 'C', 'D', 'E', 'F']:
        file_pattern = f"CollectedData_Cam-{cam}{suffix}.csv"
        if ood:
            path = Path(base_path) / "fly_ground_truth_OOD" / file_pattern
        else:
            path = Path(base_path) / "fly_ground_truth" / file_pattern
        
        if path.exists():
            # Skip first 3 rows (multi-index header) and read
            df = pd.read_csv(path, header=[0, 1, 2])
            # Flatten multi-index columns
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Select only coordinate columns (x, y for each keypoint)
            coord_cols = [col for col in df.columns if any(kp in col and coord in col 
                         for kp in KEYPOINT_NAMES
                         for coord in ['x', 'y'])]
            df = df[coord_cols]
            gt_data[f"cam_{cam}"] = df
            logger.info(f"Loaded GT data for camera {cam}: {df.shape}")
    
    return gt_data



def load_pred_data(base_path: str = "data/fly", ood: bool = False) -> Dict[str, pd.DataFrame]:
    """Load prediction data for all cameras."""
    pred_data = {}
    suffix = "_new" if ood else ""
    
    for cam in ['A', 'B', 'C', 'D', 'E', 'F']:
        file_pattern = f"predictions_Cam-{cam}{suffix}.csv"
        if ood:
            path = Path(base_path) / "fly_predictions_OOD" / file_pattern
        else:
            path = Path(base_path) / "fly_predictions" / file_pattern
        
        if path.exists():
            # Skip first 3 rows (multi-index header) and read
            df = pd.read_csv(path, header=[0, 1, 2])
            
            # Flatten multi-index columns
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Select coordinate and likelihood columns
            coord_cols = [col for col in df.columns if any(kp in col for kp in KEYPOINT_NAMES)]
            df = df[coord_cols]
            pred_data[f"cam_{cam}"] = df
            logger.info(f"Loaded prediction data for camera {cam}: {df.shape}")
    
    return pred_data

def prepare_mlp_data(gt_data: Dict[str, pd.DataFrame], 
                    pred_data: Dict[str, pd.DataFrame],
                    use_confidence: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for MLP training.
    
    Returns:
        X: (n_samples, n_features) - input features (coordinates + optionally confidence)
        y: (n_samples, 60) - target coordinates
    """
    X_list, y_list = [], []
    
    for cam in gt_data.keys():
        if cam in pred_data:
            gt_df = gt_data[cam]
            pred_df = pred_data[cam]
            
            # Ensure same number of samples
            assert len(gt_df) == len(pred_df), "Length of ground truth and predictions not equal!"
            pred_coords = []
            for i in range(0, pred_df.shape[1], 3):
                pred_coords.extend([i, i+1])  # x, y columns
            X_coords = pred_df.iloc[:, pred_coords].values
            if use_confidence:
                confidence_cols = [i+2 for i in range(0, pred_df.shape[1], 3)]
                confidences = pred_df.iloc[:, confidence_cols].values
                X = np.concatenate([X_coords, confidences], axis=1)
            else:
                X = X_coords
            y = gt_df.values         
            X_list.append(X)
            y_list.append(y)
    
    return np.vstack(X_list), np.vstack(y_list)