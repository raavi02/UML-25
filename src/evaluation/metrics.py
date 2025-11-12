from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def calculate_mpjpe(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Per Joint Position Error (in pixels)."""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def calculate_pck(y_pred: np.ndarray, y_true: np.ndarray, threshold: float = 5.0) -> float:
    """Percentage of Correct Keypoints within threshold (in pixels)."""
    distances = np.sqrt(np.sum((y_pred - y_true) ** 2, axis=1))
    return np.mean(distances <= threshold) * 100


def calculate_improvement(original_pred: np.ndarray, 
                        refined_pred: np.ndarray, 
                        ground_truth: np.ndarray) -> Dict[str, float]:
    """Calculate improvement metrics."""
    original_error = calculate_mpjpe(original_pred, ground_truth)
    refined_error = calculate_mpjpe(refined_pred, ground_truth)
    
    return {
        'original_mpjpe': original_error,
        'refined_mpjpe': refined_error,
        'absolute_improvement': original_error - refined_error,
        'relative_improvement': (original_error - refined_error) / original_error * 100
    }


def evaluate_model(model: torch.nn.Module, 
                  X_test: np.ndarray, 
                  y_test: np.ndarray,
                  device: str = 'cpu') -> Dict[str, Any]:
    """Comprehensive model evaluation."""
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_tensor).cpu().numpy()
    
    # Calculate various metrics
    metrics = {}
    
    # Store predictions for further analysis
    metrics['y_pred'] = y_pred
    metrics['y_true'] = y_test
    
    # Basic errors
    metrics['mpjpe'] = calculate_mpjpe(y_pred, y_test)
    metrics['pck_5'] = calculate_pck(y_pred, y_test, threshold=5.0)
    metrics['pck_10'] = calculate_pck(y_pred, y_test, threshold=10.0)
    
    # Per-keypoint errors
    keypoint_errors = np.sqrt(np.mean((y_pred - y_test) ** 2, axis=0))
    metrics['per_keypoint_errors'] = keypoint_errors.tolist()
    metrics['worst_keypoint_error'] = np.max(keypoint_errors)
    metrics['best_keypoint_error'] = np.min(keypoint_errors)
    
    # Mean error per coordinate type (x vs y)
    x_errors = keypoint_errors[::2]  # Every even index (0, 2, 4, ...)
    y_errors = keypoint_errors[1::2]  # Every odd index (1, 3, 5, ...)
    metrics['mean_x_error'] = np.mean(x_errors)
    metrics['mean_y_error'] = np.mean(y_errors)
    
    return metrics