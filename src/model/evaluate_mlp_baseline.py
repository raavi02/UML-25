"""Script to evaluate a trained MLP model on test data."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.data_loading.load_data import (load_gt_data, load_pred_data,
                                        prepare_data)
from src.evaluation.metrics import calculate_improvement, evaluate_model
from src.model.mlp import ResidualMLP
from src.utils.logging import logger


def evaluate_saved_model(model_path: str, data_path: str = "data/fly", use_confidence: bool = True):
    """Evaluate a saved model on OOD test data."""
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = ResidualMLP.load_from_checkpoint(model_path)
    model.eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Using device: {device}")
    
    # Load test data
    logger.info("Loading OOD test data...")
    gt_test = load_gt_data(data_path, ood=True)
    pred_test = load_pred_data(data_path, ood=True)
    
    X_test, y_test = prepare_data(gt_test, pred_test, use_confidence=use_confidence)
    
    if len(X_test) == 0:
        logger.error("No test data available!")
        return
    
    logger.info(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, device=str(device))
    
    # Calculate improvement over original predictions
    original_pred = X_test[:, :60]  # Original coordinates
    improvement_metrics = calculate_improvement(original_pred, metrics['y_pred'], y_test)
    metrics.update(improvement_metrics)
    
    # Print results
    print("\n" + "="*60)
    print("MLP BASELINE - FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {Path(model_path).name}")
    print(f"Test samples: {len(X_test)}")
    print(f"\nKEY METRICS:")
    print(f"  MPJPE: {metrics['mpjpe']:.4f} pixels")
    print(f"  PCK@5: {metrics['pck_5']:.2f}%")
    print(f"  PCK@10: {metrics['pck_10']:.2f}%")
    print(f"\nIMPROVEMENT OVER ORIGINAL PREDICTIONS:")
    print(f"  Original MPJPE: {metrics['original_mpjpe']:.4f} pixels")
    print(f"  Refined MPJPE:  {metrics['refined_mpjpe']:.4f} pixels")
    print(f"  Absolute Improvement: {metrics['absolute_improvement']:.4f} pixels")
    print(f"  Relative Improvement: {metrics['relative_improvement']:.2f}%")
    print(f"\nKEYPOINT ANALYSIS:")
    print(f"  Best keypoint error: {metrics['best_keypoint_error']:.4f} pixels")
    print(f"  Worst keypoint error: {metrics['worst_keypoint_error']:.4f} pixels")
    print("="*60)
    
    # Save detailed results
    results_dir = Path("mlp_results") / "evaluations"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"eval_{Path(model_path).stem}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'tolist'):
            serializable_metrics[key] = value.tolist()
        else:
            serializable_metrics[key] = value
    
    with open(results_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    logger.info(f"Detailed results saved to: {results_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MLP model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/fly',
                       help='Path to data directory')
    parser.add_argument('--use_confidence', action='store_true', default=True,
                       help='Use confidence scores as input features')
    
    args = parser.parse_args()
    
    evaluate_saved_model(
        model_path=args.model_path,
        data_path=args.data_path,
        use_confidence=args.use_confidence
    )


if __name__ == "__main__":
    main()