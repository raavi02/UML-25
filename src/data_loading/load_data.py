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


def prepare_loo_data_from_preds(
    pred_data: Dict[str, pd.DataFrame],
    use_confidence: bool = True,
    n_drops_per_pose: int = 1,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for the leave-one-out Pose MLP.

    For each frame (and each camera), we:
      1. Extract the full predicted pose as coords (and optionally confidences).
      2. Choose `n_drops_per_pose` keypoint indices to "drop".
      3. Build an input vector:
           X_example = [pose_features, mask]
         where:
           - pose_features = [x_0, y_0, ..., x_{K-1}, y_{K-1}, (conf_0, ..., conf_{K-1})]
           - mask          = [m_0, ..., m_{K-1}], with m_j = 1 if keypoint j is DROPPED, else 0.
      4. The target y_example is the original coords (2K):
           y_example = [x_0, y_0, ..., x_{K-1}, y_{K-1}]

    Returned shapes:
        X_loo: (n_examples, 2K + (K if use_confidence else 0) + K)
        y_loo: (n_examples, 2K)

    Notes:
      - We do NOT use GT here: this is purely self-supervised on predictions.
      - Multiple cameras are simply concatenated along the batch dimension.
    """
    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    X_list = []
    y_list = []

    for cam, df in pred_data.items():
        if df is None or df.empty:
            continue

        num_frames, num_cols = df.shape

        # We expect columns grouped as (x, y, likelihood) per keypoint
        if num_cols % 3 != 0:
            raise ValueError(
                f"Prediction DataFrame for camera {cam} has {num_cols} columns, "
                "which is not divisible by 3 (x, y, likelihood per keypoint)."
            )

        K = num_cols // 3  # number of keypoints

        # Indices for x,y coords in the flattened columns
        coord_cols = []
        for i in range(0, num_cols, 3):
            coord_cols.extend([i, i + 1])  # x, y

        coords = df.iloc[:, coord_cols].values.astype(np.float32)  # (num_frames, 2K)

        if use_confidence:
            conf_cols = [i + 2 for i in range(0, num_cols, 3)]
            confs = df.iloc[:, conf_cols].values.astype(np.float32)  # (num_frames, K)
            pose_features = np.concatenate([coords, confs], axis=1)  # (num_frames, 2K + K)
        else:
            pose_features = coords  # (num_frames, 2K)

        logger.info(
            f"LOO prep for camera {cam}: num_frames={num_frames}, "
            f"K={K}, pose_feature_dim={pose_features.shape[1]}"
        )

        # For each frame, create n_drops_per_pose training examples
        for t in range(num_frames):
            pose_feat_t = pose_features[t]        # (2K [+ K_conf])
            coords_t = coords[t]                  # (2K) → target

            # choose distinct keypoints to drop
            drop_indices = rng.choice(K, size=n_drops_per_pose, replace=False)

            for j in drop_indices:
                mask = np.zeros(K, dtype=np.float32)
                mask[j] = 1.0                     # mark keypoint j as dropped

                X_example = np.concatenate([pose_feat_t, mask], axis=0)  # (2K [+K_conf] + K)
                y_example = coords_t.copy()                                # (2K)

                X_list.append(X_example)
                y_list.append(y_example)

    if not X_list:
        raise ValueError("No prediction data found to build LOO dataset.")

    X_loo = np.stack(X_list, axis=0)
    y_loo = np.stack(y_list, axis=0)

    logger.info(f"Shape of LOO input X: {X_loo.shape}")
    logger.info(f"Shape of LOO target y: {y_loo.shape}")

    return X_loo, y_loo
