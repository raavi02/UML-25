from __future__ import annotations

from omegaconf import DictConfig
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from src.utils.logging import logger


def load_gt_data(cfg_d : DictConfig, ood: bool = False) -> Dict[str, pd.DataFrame]:
    """Load ground truth data for all cameras."""
    gt_data = {}
    suffix = "_new" if ood else ""
    data_dir = cfg_d.data.gt_data_dir
    for cam in cfg_d.data.view_names:
        path = Path(data_dir) / f"CollectedData_{cam}{suffix}.csv"
        
        if path.exists():
            # Skip first 3 rows (multi-index header) and read
            df = pd.read_csv(path, header=[0, 1, 2])
            # Flatten multi-index columns
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Select only coordinate columns (x, y for each keypoint)
            coord_cols = [col for col in df.columns if any(kp in col and coord in col 
                         for kp in cfg_d.data.keypoint_names
                         for coord in ['x', 'y'])]
            df = df[coord_cols]
            gt_data[f"{cam}"] = df
            logger.info(f"Loaded GT data for {cam}{suffix}: {df.shape}")
    
    return gt_data


def load_pred_data(cfg_d : DictConfig, ood: bool = False) -> Dict[str, pd.DataFrame]:
    """Load prediction data for all cameras."""
    pred_data = {}
    suffix = "_new" if ood else ""
    data_dir = cfg_d.data.preds_data_dir
    for cam in cfg_d.data.view_names:
        path = Path(data_dir) / f"predictions_{cam}{suffix}.csv"
        
        if path.exists():
            # Skip first 3 rows (multi-index header) and read
            df = pd.read_csv(path, header=[0, 1, 2])
            
            # Flatten multi-index columns
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Select coordinate and likelihood columns
            coord_cols = [col for col in df.columns if any(kp in col for kp in cfg_d.data.keypoint_names)]
            df = df[coord_cols]
            pred_data[f"{cam}"] = df
            logger.info(f"Loaded prediction data for {cam}{suffix}: {df.shape}")
    
    return pred_data

def prepare_data(gt_data: Dict[str, pd.DataFrame],
                 pred_data: Dict[str, pd.DataFrame],
                 use_confidence: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for MLP training.
    
    Returns:
        X: (n_samples, 2K + (K if use_confidence else 0))
        y: (n_samples, 2K)
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
    X = np.vstack(X_list)
    y = np.vstack(y_list)
    logger.info(f"Shape of input X: {X.shape}")
    logger.info(f"Shape of prediction y: {y.shape}")
    return X, y