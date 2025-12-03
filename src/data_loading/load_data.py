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
    data_dir = getattr(cfg_d.data, "gt_data_dir_ood", cfg_d.data.gt_data_dir) if ood else cfg_d.data.gt_data_dir
    for cam in cfg_d.data.view_names:
        path = Path(data_dir) / f"CollectedData_{cam}{suffix}.csv"
        
        if path.exists():
            logger.info(f"Loading GT file for {cam}{suffix}: {path}")
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
        else:
            logger.warning(f"GT file missing for {cam}{suffix}: {path}")
    
    return gt_data


def load_pred_data(cfg_d : DictConfig, ood: bool = False) -> Dict[str, pd.DataFrame]:
    """Load prediction data for all cameras."""
    pred_data = {}
    suffix = "_new" if ood else ""
    data_dir = getattr(cfg_d.data, "preds_data_dir_ood", cfg_d.data.preds_data_dir) if ood else cfg_d.data.preds_data_dir
    for cam in cfg_d.data.view_names:
        path = Path(data_dir) / f"predictions_{cam}{suffix}.csv"
        
        if path.exists():
            logger.info(f"Loading predictions file for {cam}{suffix}: {path}")
            # Skip first 3 rows (multi-index header) and read
            df = pd.read_csv(path, header=[0, 1, 2])
            
            # Flatten multi-index columns
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Select coordinate and likelihood columns
            coord_cols = [col for col in df.columns if any(kp in col for kp in cfg_d.data.keypoint_names)]
            df = df[coord_cols]
            pred_data[f"{cam}"] = df
            logger.info(f"Loaded prediction data for {cam}{suffix}: {df.shape}")
        else:
            logger.warning(f"Predictions file missing for {cam}{suffix}: {path}")
    
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

            # Drop rows with NaNs in either GT or predictions to avoid NaN losses.
            valid_mask = ~np.isnan(y).any(axis=1) & ~np.isnan(X).any(axis=1)
            if not valid_mask.all():
                dropped = (~valid_mask).sum()
                logger.warning(
                    f"Dropping {dropped} / {len(valid_mask)} rows with NaNs for camera {cam}."
                )
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) == 0:
                logger.warning(f"No valid samples left for camera {cam} after NaN filtering.")
                continue

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


def prepare_loo_eval_data(
    gt_data: Dict[str, pd.DataFrame],
    pred_data: Dict[str, pd.DataFrame],
    use_confidence: bool = True,
    n_drops_per_pose: int = 1,
    random_state: int | None = None,
):
    """
    Build leave-one-out evaluation examples that align predictions with ground truth.

    Returns:
        X_loo_eval: (n_examples, 2K [+K_conf] + K)
        y_gt:       (n_examples, 2K) ground-truth coordinates
        base_preds: (n_examples, 2K) original predictor outputs
        dropped:    (n_examples,) index of the dropped keypoint for each example
    """

    rng = np.random.RandomState(random_state) if random_state is not None else np.random

    X_list, y_gt_list, base_pred_list, dropped_list = [], [], [], []

    for cam, gt_df in gt_data.items():
        if cam not in pred_data:
            logger.warning(f"Skipping camera {cam} for LOO eval; no predictions available.")
            continue

        pred_df = pred_data[cam]
        if pred_df is None or pred_df.empty or gt_df is None or gt_df.empty:
            logger.warning(f"Skipping camera {cam} for LOO eval; empty GT or predictions.")
            continue

        if len(gt_df) != len(pred_df):
            raise ValueError(
                f"GT/pred length mismatch for camera {cam}: "
                f"GT={len(gt_df)}, Pred={len(pred_df)}"
            )

        num_cols = pred_df.shape[1]
        if num_cols % 3 != 0:
            raise ValueError(
                f"Prediction DataFrame for camera {cam} has {num_cols} columns, "
                "expected groups of (x, y, likelihood)."
            )

        K = num_cols // 3
        if n_drops_per_pose > K:
            raise ValueError(
                f"n_drops_per_pose ({n_drops_per_pose}) cannot exceed number of keypoints ({K})."
            )

        coord_cols = []
        for i in range(0, num_cols, 3):
            coord_cols.extend([i, i + 1])

        coords_pred = pred_df.iloc[:, coord_cols].values.astype(np.float32)
        coords_gt = gt_df.values.astype(np.float32)

        if use_confidence:
            conf_cols = [i + 2 for i in range(0, num_cols, 3)]
            confs = pred_df.iloc[:, conf_cols].values.astype(np.float32)
            pose_features_all = np.concatenate([coords_pred, confs], axis=1)
        else:
            pose_features_all = coords_pred

        # Drop any rows containing NaNs in either GT or predictions
        valid_mask = (
            ~np.isnan(coords_pred).any(axis=1)
            & ~np.isnan(coords_gt).any(axis=1)
            & ~np.isnan(pose_features_all).any(axis=1)
        )

        if not valid_mask.all():
            dropped = (~valid_mask).sum()
            logger.warning(
                f"Dropping {dropped} / {len(valid_mask)} rows with NaNs for camera {cam} during LOO eval."
            )

        coords_pred = coords_pred[valid_mask]
        coords_gt = coords_gt[valid_mask]
        pose_features_all = pose_features_all[valid_mask]

        for t in range(coords_pred.shape[0]):
            pose_feat_t = pose_features_all[t]
            coords_pred_t = coords_pred[t]
            coords_gt_t = coords_gt[t]

            drop_indices = rng.choice(K, size=n_drops_per_pose, replace=False)

            for j in drop_indices:
                mask = np.zeros(K, dtype=np.float32)
                mask[j] = 1.0

                X_example = np.concatenate([pose_feat_t, mask], axis=0)

                X_list.append(X_example)
                y_gt_list.append(coords_gt_t)
                base_pred_list.append(coords_pred_t)
                dropped_list.append(int(j))

    if not X_list:
        raise ValueError("No data available to build LOO evaluation set.")

    X_loo_eval = np.stack(X_list, axis=0)
    y_gt = np.stack(y_gt_list, axis=0)
    base_preds = np.stack(base_pred_list, axis=0)
    dropped = np.array(dropped_list, dtype=np.int64)

    logger.info(
        f"LOO eval set shapes — X: {X_loo_eval.shape}, y_gt: {y_gt.shape}, base_preds: {base_preds.shape}"
    )

    return X_loo_eval, y_gt, base_preds, dropped
