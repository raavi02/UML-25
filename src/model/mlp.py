# src/models/mlp.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json
import numpy as np
import torch
import pytorch_lightning as pl
from src.utils.logging import logger

from src.model.mlp_baseline import ResidualMLP, MLPDataModule
from src.evaluation.metrics import evaluate_model, calculate_improvement
from src.data_loading.load_data import prepare_mlp_data

# ---- factories used by GridSearchRunner ----

def _num_keypoints(cfg: Any) -> int:
    # Works with DictConfig (OmegaConf) or plain dict
    return int(cfg.data.num_keypoints)

def build_mlp_model(params: Dict[str, Any], n_keypoints: int) -> pl.LightningModule:
    use_conf = params.get("use_confidence", True)
    input_dim = n_keypoints * 2 + (n_keypoints if use_conf else 0)
    return ResidualMLP(
        input_dim=input_dim,
        n_keypoints=n_keypoints,
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

def build_mlp_datamodule(params: Dict[str, Any], splits: Dict[str, Any]) -> pl.LightningDataModule:
    use_conf = params.get("use_confidence", True)

    X_all, y_all = prepare_mlp_data(splits["gt_train"], splits["pred_train"], use_confidence=use_conf)
    tr_idx = splits["tr_idx"]
    va_idx = splits["va_idx"]
    assert tr_idx.max() < len(X_all) and va_idx.max() < len(X_all), \
        f"indices out of bounds: X_all len={len(X_all)}, max tr={tr_idx.max()}, max va={va_idx.max()}"

    X_train, y_train = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[va_idx], y_all[va_idx]

    return MLPDataModule(X_train, y_train, X_val, y_val, batch_size=params["batch_size"])

# ---- evaluation for the selected best model ----

def evaluate_mlp_checkpoint(best_path: str, X_test: np.ndarray, y_test: np.ndarray,
                            output_dir: str | Path = "mlp_results") -> Dict[str, float]:
    """Evaluate a trained MLP checkpoint on given test arrays and log detailed metrics."""

    output_dir = Path(output_dir)
    eval_dir = output_dir / "mlp" / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load model + device
    # -------------------------
    logger.info(f"Loading model from: {best_path}")
    model = ResidualMLP.load_from_checkpoint(best_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # -------------------------
    # Early sanity check
    # -------------------------
    if len(X_test) == 0:
        logger.error("No test data available!")
        return {}

    logger.info(f"Test data shape - X_test: {X_test.shape}, y_test: {y_test.shape}")

    # -------------------------
    # Forward pass + metrics
    # -------------------------
    with torch.no_grad():
        y_pred = model(torch.as_tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()

    metrics = evaluate_model(model, X_test, y_test, device=str(device))

    # Add improvement metrics
    base_cols = model.n_keypoints * 2
    original_pred = X_test[:, :base_cols]
    improvement_metrics = calculate_improvement(original_pred, y_pred, y_test)
    metrics.update(improvement_metrics)

    # -------------------------
    # Print summary to console
    # -------------------------
    print("\n" + "=" * 60)
    print("MLP BASELINE - FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {Path(best_path).name}")
    print(f"Test samples: {len(X_test)}")
    print("\nKEY METRICS:")
    print(f"  MPJPE: {metrics['mpjpe']:.4f} pixels")
    print(f"  PCK@5: {metrics['pck_5']:.2f}%")
    print(f"  PCK@10: {metrics['pck_10']:.2f}%")
    print("\nIMPROVEMENT OVER ORIGINAL PREDICTIONS:")
    print(f"  Original MPJPE: {metrics.get('original_mpjpe', np.nan):.4f} pixels")
    print(f"  Refined MPJPE:  {metrics.get('refined_mpjpe', np.nan):.4f} pixels")
    print(f"  Absolute Improvement: {metrics.get('absolute_improvement', np.nan):.4f} pixels")
    print(f"  Relative Improvement: {metrics.get('relative_improvement', np.nan):.2f}%")
    print("\nKEYPOINT ANALYSIS:")
    print(f"  Best keypoint error: {metrics.get('best_keypoint_error', np.nan):.4f} pixels")
    print(f"  Worst keypoint error: {metrics.get('worst_keypoint_error', np.nan):.4f} pixels")
    print("=" * 60)

    # -------------------------
    # Log summary (to file and logger)
    # -------------------------
    logger.info("\n=== TEST SET METRICS ===")
    logger.info(f"MPJPE: {metrics['mpjpe']:.4f} pixels")
    logger.info(f"PCK@5: {metrics['pck_5']:.2f}%")
    logger.info(f"PCK@10: {metrics['pck_10']:.2f}%")
    if 'relative_improvement' in metrics:
        logger.info(f"Relative Improvement: {metrics['relative_improvement']:.2f}%")
    logger.info("=" * 40)

    # -------------------------
    # Save to JSON
    # -------------------------
    results_file = eval_dir / f"eval_{Path(best_path).stem}.json"
    serializable_metrics = {k: (v.tolist() if hasattr(v, "tolist") else v)
                            for k, v in metrics.items()}
    with open(results_file, "w") as f:
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"Detailed results saved to: {results_file}")

    return metrics

# ---- convenience function we can call from pipeline ----

def run_mlp_grid_search(
    splits: Dict[str, Any],
    output_dir: str | Path,
    max_epochs: int,
    n_keypoints: int,
) -> Dict[str, Any]:
    from src.model.grid_search import GridSearchRunner

    param_grid = {
        "hidden_dims": [[512, 256], [256, 128], [512, 256, 128]],
        "dropout": [0.1, 0.2],
        "learning_rate": [1e-3, 5e-4],
        "weight_decay": [1e-4, 1e-5],
        "use_confidence": [True, False],
        "batch_size": [64, 128],
    }

    runner = GridSearchRunner(
        output_dir=output_dir,
        param_grid=param_grid,
        build_model=build_mlp_model,
        build_datamodule=build_mlp_datamodule,
        monitor_metric="val_loss",
        mode="min",
        patience=15,
        max_epochs=max_epochs,
        n_keypoints=n_keypoints,
    )

    results = runner.run(splits)
    if not results:
        raise RuntimeError("No successful MLP runs.")
    best = min(results, key=lambda r: r["best_score"])
    return best  # {params, best_score, best_path}


def create_summary_report(final_results: dict, output_dir: str):
    """Create a human-readable summary report."""
    report_path = Path(output_dir) / 'summary_report.txt'

    with open(report_path, 'w') as f:
        f.write("MLP Baseline - Summary Report\n")
        f.write("=" * 50 + "\n\n")

        f.write("BEST HYPERPARAMETERS:\n")
        for key, value in final_results['best_hyperparameters'].items():
            f.write(f"  {key}: {value}\n")

        f.write(f"\nVALIDATION PERFORMANCE:\n")
        f.write(f"  Best Validation Loss: {final_results['best_validation_loss']:.6f}\n")

        if 'test_metrics' in final_results:
            metrics = final_results['test_metrics']
            f.write(f"\nTEST SET PERFORMANCE:\n")
            f.write(f"  MPJPE: {metrics.get('mpjpe', 'N/A'):.4f} pixels\n")
            f.write(f"  PCK@5: {metrics.get('pck_5', 'N/A'):.2f}%\n")
            f.write(f"  PCK@10: {metrics.get('pck_10', 'N/A'):.2f}%\n")

            if 'original_mpjpe' in metrics and 'refined_mpjpe' in metrics:
                f.write(f"\nIMPROVEMENT OVER ORIGINAL:\n")
                f.write(f"  Original MPJPE: {metrics['original_mpjpe']:.4f} pixels\n")
                f.write(f"  Refined MPJPE:  {metrics['refined_mpjpe']:.4f} pixels\n")
                f.write(f"  Absolute Improvement: {metrics['absolute_improvement']:.4f} pixels\n")
                f.write(f"  Relative Improvement: {metrics['relative_improvement']:.2f}%\n")

        f.write(f"\nDATA STATISTICS:\n")
        for split, shape in final_results['data_shapes'].items():
            f.write(f"  {split}: {shape}\n")

    logger.info(f"Summary report saved to: {report_path}")
