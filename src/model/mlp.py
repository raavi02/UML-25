# src/models/mlp.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
import json
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from src.utils.logging import logger
from src.model.grid_search import GridSearchRunner
from src.evaluation.metrics import evaluate_model, calculate_improvement
from src.data_loading.load_data import prepare_data, prepare_loo_eval_data, load_pred_data
from src.model.loo_plotting import save_loo_recon_plots

# ---- factories used by GridSearchRunner ----

def _num_keypoints(cfg: Any) -> int:
    # Works with DictConfig (OmegaConf) or plain dict
    return int(cfg.data.num_keypoints)

def build_mlp_model(
    params: Dict[str, Any],
    n_keypoints: int,
    mode: str = "refine",
    loo_input_dim: int | None = None,
) -> pl.LightningModule:
    """
    mode:
      - 'refine': ResidualMLP (current baseline, GT-supervised refiner)
      - 'loo':    PoseMLPLOO (self-supervised leave-one-out pose model)
    """
    use_conf = params.get("use_confidence", True)

    if mode == "refine":
        input_dim = n_keypoints * 2 + (n_keypoints if use_conf else 0)
        return ResidualMLP(
            input_dim=input_dim,
            n_keypoints=n_keypoints,
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

    elif mode == "loo":
        if loo_input_dim is None:
            raise ValueError("loo_input_dim must be provided when mode='loo'.")
        return PoseMLPLOO(
            input_dim=loo_input_dim,
            n_keypoints=n_keypoints,
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_mlp_datamodule(
    params: Dict[str, Any],
    splits: Dict[str, Any],
    mode: str = "refine",
) -> pl.LightningDataModule:
    use_conf = params.get("use_confidence", True)

    if mode == "refine":
        X_all, y_all = prepare_data(
            splits["gt_train"],
            splits["pred_train"],
            use_confidence=use_conf,
        )
    elif mode == "loo":
        from src.data_loading.load_data import prepare_loo_data_from_preds
        X_all, y_all = prepare_loo_data_from_preds(
            pred_data=splits["pred_train"],
            use_confidence=use_conf,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    tr_idx = splits["tr_idx"]
    va_idx = splits["va_idx"]

    X_train, y_train = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[va_idx], y_all[va_idx]

    return MLPDataModule(X_train, y_train, X_val, y_val, batch_size=params["batch_size"])

# ---- evaluation for the selected best model ----

def evaluate_mlp_checkpoint(best_path: str, X_test: np.ndarray, y_test: np.ndarray,
                            output_dir: str | Path = "mlp_results", cfg_d=None, save_preds=False) -> Dict[str, float]:
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
    logger.info(f"MPJPE: {metrics['mpjpe']:.4f}")
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

    if save_preds:
        save_predictions(cfg_d, eval_dir, y_pred)

    return metrics


def evaluate_mlp_loo_checkpoint(
    best_path: str,
    gt_data: Dict[str, Any],
    pred_data: Dict[str, Any],
    output_dir: str | Path = "mlp_results",
    use_confidence: bool = True,
    n_drops_per_pose: int = 1,
    keypoint_names: List[str] | None = None,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained LOO MLP by comparing reconstruction to ground truth on dropped joints.
    """

    output_dir = Path(output_dir)
    eval_dir = output_dir / "mlp" / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing LOO evaluation data (GT + predictions)...")
    X_eval, y_gt, base_preds, _ = prepare_loo_eval_data(
        gt_data=gt_data,
        pred_data=pred_data,
        use_confidence=use_confidence,
        n_drops_per_pose=n_drops_per_pose,
    )

    logger.info(f"Loading LOO model from: {best_path}")
    model = PoseMLPLOO.load_from_checkpoint(best_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    # Forward pass
    with torch.no_grad():
        y_recon = model(torch.as_tensor(X_eval, dtype=torch.float32, device=device)).cpu().numpy()

    K = model.n_keypoints
    mask = X_eval[:, -K:]  # (N, K)

    base_coords = base_preds.reshape(-1, K, 2)
    recon_coords = y_recon.reshape(-1, K, 2)
    gt_coords = y_gt.reshape(-1, K, 2)

    base_errors = np.sqrt(((base_coords - gt_coords) ** 2).sum(axis=2))
    recon_errors = np.sqrt(((recon_coords - gt_coords) ** 2).sum(axis=2))
    recon_deltas = np.sqrt(((recon_coords - base_coords) ** 2).sum(axis=2))

    mask_bool = mask > 0.5
    if not mask_bool.any():
        logger.error("No dropped keypoints found in eval mask; cannot compute metrics.")
        return {}

    dropped_base_errors = base_errors[mask_bool]
    dropped_recon_errors = recon_errors[mask_bool]
    dropped_deltas = recon_deltas[mask_bool]

    normalized_deltas = dropped_deltas / (dropped_base_errors + 1e-8)
    normalized_recon_error = dropped_recon_errors / (dropped_base_errors + 1e-8)

    corr_delta_vs_base = np.nan
    corr_recon_vs_base = np.nan
    if dropped_base_errors.size > 1:
        corr_delta_vs_base = float(np.corrcoef(dropped_deltas, dropped_base_errors)[0, 1])
        corr_recon_vs_base = float(np.corrcoef(dropped_recon_errors, dropped_base_errors)[0, 1])

    # Bin analysis: group by base-error magnitude and summarize reconstruction behavior
    bin_stats = []
    try:
        quantiles = np.linspace(0, 1, 6)  # 5 bins
        edges = np.quantile(dropped_base_errors, quantiles)
        edges = np.unique(edges)
        if len(edges) >= 2:
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                if i == len(edges) - 2:
                    mask_bin = (dropped_base_errors >= lo) & (dropped_base_errors <= hi)
                else:
                    mask_bin = (dropped_base_errors >= lo) & (dropped_base_errors < hi)

                if not mask_bin.any():
                    continue

                bin_stats.append({
                    "bin_low": float(lo),
                    "bin_high": float(hi),
                    "count": int(mask_bin.sum()),
                    "mean_base_error": float(dropped_base_errors[mask_bin].mean()),
                    "mean_recon_error": float(dropped_recon_errors[mask_bin].mean()),
                    "mean_delta": float(dropped_deltas[mask_bin].mean()),
                    "mean_recon_base_ratio": float(normalized_recon_error[mask_bin].mean()),
                })
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Failed to compute LOO bin stats: {e}")

    per_k_base = []
    per_k_recon = []
    per_k_delta = []
    for k in range(K):
        mask_k = mask_bool[:, k]
        if mask_k.any():
            per_k_base.append(float(base_errors[mask_k, k].mean()))
            per_k_recon.append(float(recon_errors[mask_k, k].mean()))
            per_k_delta.append(float(recon_deltas[mask_k, k].mean()))
        else:
            per_k_base.append(np.nan)
            per_k_recon.append(np.nan)
            per_k_delta.append(np.nan)

    metrics: Dict[str, Any] = {
        "base_mpjpe": float(dropped_base_errors.mean()),
        "reconstruction_mpjpe": float(dropped_recon_errors.mean()),
        "delta_magnitude": float(dropped_deltas.mean()),
        "normalized_delta_mean": float(normalized_deltas.mean()),
        "normalized_delta_median": float(np.median(normalized_deltas)),
        "normalized_recon_error_mean": float(normalized_recon_error.mean()),
        "corr_delta_vs_base_error": corr_delta_vs_base,
        "corr_recon_vs_base_error": corr_recon_vs_base,
        "binned_error_stats": bin_stats,
        "per_keypoint_base_error": per_k_base,
        "per_keypoint_reconstruction_error": per_k_recon,
        "per_keypoint_delta": per_k_delta,
        "num_eval_examples": int(X_eval.shape[0]),
        "eval_X_shape": tuple(X_eval.shape),
    }

    print("\n" + "=" * 60)
    print("MLP LOO - EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {Path(best_path).name}")
    print(f"Eval examples (pose x drops): {X_eval.shape[0]}")
    print("\nDROPPED JOINT METRICS:")
    print(f"  Base pred MPJPE: {metrics['base_mpjpe']:.4f} pixels")
    print(f"  LOO recon MPJPE: {metrics['reconstruction_mpjpe']:.4f} pixels")
    print(f"  Delta magnitude (|recon - pred|): {metrics['delta_magnitude']:.4f} pixels")
    print(f"  Normalized delta (mean): {metrics['normalized_delta_mean']:.4f}")
    print(f"  Recon/Base error ratio (mean): {metrics['normalized_recon_error_mean']:.4f}")
    if not np.isnan(metrics["corr_delta_vs_base_error"]):
        print(f"  Corr(delta, base_error): {metrics['corr_delta_vs_base_error']:.4f}")
    if not np.isnan(metrics["corr_recon_vs_base_error"]):
        print(f"  Corr(recon_error, base_error): {metrics['corr_recon_vs_base_error']:.4f}")
    if metrics.get("binned_error_stats"):
        print("\n  Mean recon error by base-error bins:")
        for b in metrics["binned_error_stats"]:
            print(
                f"    [{b['bin_low']:.2f}, {b['bin_high']:.2f}] (n={b['count']}): "
                f"base={b['mean_base_error']:.3f}, recon={b['mean_recon_error']:.3f}, "
                f"ratio={b['mean_recon_base_ratio']:.3f}"
            )
    print("=" * 60)

    results_file = eval_dir / f"loo_eval_{Path(best_path).stem}.json"
    serializable_metrics = {k: (v.tolist() if hasattr(v, "tolist") else v)
                            for k, v in metrics.items()}
    with open(results_file, "w") as f:
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"LOO evaluation saved to: {results_file}")

    if save_plots:
        save_loo_recon_plots(
            eval_dir=eval_dir,
            keypoint_names=keypoint_names or [f"kp_{i}" for i in range(K)],
            base_errors=base_errors,
            recon_errors=recon_errors,
            drop_mask=mask_bool,
            tag="mlp",
        )

    return metrics

# ---- convenience function we can call from pipeline ----

def run_mlp_grid_search(
    splits: Dict[str, Any],
    output_dir: str | Path,
    max_epochs: int,
    n_keypoints: int,
    use_confidence_options: List[bool] | None = None,
    mode: str = "refine",
) -> Dict[str, Any]:
    """
    Run grid search over MLP hyperparameters.

    mode:
      - 'refine': ResidualMLP supervised refiner
      - 'loo':    PoseMLPLOO self-supervised leave-one-out model
    """
    use_conf_opts = use_confidence_options if use_confidence_options is not None else [True, False]

    param_grid = {
        "hidden_dims": [[512, 256], [256, 128], [512, 256, 128]],
        "dropout": [0.1, 0.2],
        "learning_rate": [1e-3, 5e-4],
        "weight_decay": [1e-4, 1e-5],
        "use_confidence": use_conf_opts,
        "batch_size": [64, 128],
    }

    def _build_model_for_params(params: Dict[str, Any], n_k: int) -> pl.LightningModule:
        use_conf = params.get("use_confidence", True)

        if mode == "refine":
            loo_input_dim = None
        else:
            # coords (2K) + optional conf (K) + mask (K)
            base_dim = 2 * n_k + (n_k if use_conf else 0)
            loo_input_dim = base_dim + n_k

        return build_mlp_model(
            params=params,
            n_keypoints=n_k,
            mode=mode,
            loo_input_dim=loo_input_dim,
        )

    def _build_datamodule_for_params(params: Dict[str, Any], splits_: Dict[str, Any]) -> pl.LightningDataModule:
        return build_mlp_datamodule(
            params=params,
            splits=splits_,
            mode=mode,
        )

    runner = GridSearchRunner(
        output_dir=output_dir,
        param_grid=param_grid,
        build_model=_build_model_for_params,
        build_datamodule=_build_datamodule_for_params,
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
    logger.info(f"Best MLP ({mode}) grid-search result: {best}")
    return best


def create_summary_report(final_results: dict, output_dir: str):
    """Create a human-readable summary report."""
    report_path = Path(output_dir) / 'summary_report.txt'

    def _fmt_float(val, fmt: str = ".4f") -> str:
        """Safely format a value as float; fall back to 'N/A'."""
        if isinstance(val, (int, float, np.floating)):
            return format(val, fmt)
        return "N/A"

    with open(report_path, 'w') as f:
        f.write("MLP Baseline - Summary Report\n")
        f.write("=" * 50 + "\n\n")

        # -----------------------
        # Best hyperparameters
        # -----------------------
        f.write("BEST HYPERPARAMETERS:\n")
        for key, value in final_results.get('best_hyperparameters', {}).items():
            f.write(f"  {key}: {value}\n")

        # -----------------------
        # Validation performance
        # -----------------------
        f.write("\nVALIDATION PERFORMANCE:\n")
        val_loss = final_results.get('best_validation_loss', None)
        f.write(
            f"  Best Validation Loss: "
            f"{_fmt_float(val_loss, '.6f') if val_loss is not None else 'N/A'}\n"
        )

        # -----------------------
        # Test performance
        # -----------------------
        metrics = final_results.get('test_metrics', None)
        if metrics and isinstance(metrics, dict) and len(metrics) > 0:
            f.write("\nTEST SET PERFORMANCE:\n")
            mpjpe = metrics.get('mpjpe', None)
            pck_5 = metrics.get('pck_5', None)
            pck_10 = metrics.get('pck_10', None)

            if mpjpe is not None:
                f.write(f"  MPJPE: {_fmt_float(mpjpe, '.4f')} pixels\n")
            if pck_5 is not None:
                f.write(f"  PCK@5: {_fmt_float(pck_5, '.2f')}%\n")
            if pck_10 is not None:
                f.write(f"  PCK@10: {_fmt_float(pck_10, '.2f')}%\n")

            # Improvement block (only if these exist and look numeric)
            orig = metrics.get('original_mpjpe', None)
            ref = metrics.get('refined_mpjpe', None)
            abs_imp = metrics.get('absolute_improvement', None)
            rel_imp = metrics.get('relative_improvement', None)

            if orig is not None and ref is not None:
                f.write("\nIMPROVEMENT OVER ORIGINAL:\n")
                f.write(f"  Original MPJPE: {_fmt_float(orig, '.4f')} pixels\n")
                f.write(f"  Refined MPJPE:  {_fmt_float(ref, '.4f')} pixels\n")
                f.write(f"  Absolute Improvement: {_fmt_float(abs_imp, '.4f')} pixels\n")
                f.write(f"  Relative Improvement: {_fmt_float(rel_imp, '.2f')}%\n")

            # LOO reconstruction block
            base_mpjpe = metrics.get('base_mpjpe', None)
            recon_mpjpe = metrics.get('reconstruction_mpjpe', None)
            if base_mpjpe is not None:
                f.write("\nLOO RECONSTRUCTION (dropped joints):\n")
                f.write(f"  Base pred MPJPE: {_fmt_float(base_mpjpe, '.4f')} pixels\n")
                if recon_mpjpe is not None:
                    f.write(f"  LOO recon MPJPE: {_fmt_float(recon_mpjpe, '.4f')} pixels\n")
                if metrics.get('normalized_delta_mean', None) is not None:
                    f.write(
                        f"  Normalized delta (mean): {_fmt_float(metrics['normalized_delta_mean'], '.4f')}\n"
                    )
                if metrics.get('normalized_recon_error_mean', None) is not None:
                    f.write(
                        f"  Recon/Base error ratio (mean): {_fmt_float(metrics['normalized_recon_error_mean'], '.4f')}\n"
                    )
                if metrics.get('corr_delta_vs_base_error', None) is not None:
                    f.write(
                        f"  Corr(delta, base_error): {_fmt_float(metrics['corr_delta_vs_base_error'], '.4f')}\n"
                    )
                if metrics.get('corr_recon_vs_base_error', None) is not None:
                    f.write(
                        f"  Corr(recon_error, base_error): {_fmt_float(metrics['corr_recon_vs_base_error'], '.4f')}\n"
                    )
                if metrics.get('binned_error_stats'):
                    f.write("  Mean recon error by base-error bins:\n")
                    for b in metrics['binned_error_stats']:
                        f.write(
                            f"    [{_fmt_float(b['bin_low'], '.2f')}, {_fmt_float(b['bin_high'], '.2f')}] "
                            f"(n={b['count']}): base={_fmt_float(b['mean_base_error'], '.3f')}, "
                            f"recon={_fmt_float(b['mean_recon_error'], '.3f')}, "
                            f"ratio={_fmt_float(b['mean_recon_base_ratio'], '.3f')}\n"
                        )
        else:
            # This covers LOO mode and/or when OOD eval is disabled
            f.write("\nTEST SET PERFORMANCE:\n")
            f.write("  No test metrics available (evaluation skipped or not applicable).\n")

        # -----------------------
        # Data statistics
        # -----------------------
        f.write("\nDATA STATISTICS:\n")
        for split, shape in final_results.get('data_shapes', {}).items():
            f.write(f"  {split}: {shape}\n")

    logger.info(f"Summary report saved to: {report_path}")

import matplotlib.pyplot as plt

def save_predictions(cfg_d, eval_dir, prediction_array, ood=True):
    """
    Save refined predictions and pixel error CSV.
    ALSO compare per-frame pixel error against original predictions
    and plot both on aligned CDF-style plots.
    """

    suffix = "_new" if ood else ""
    data_dir = cfg_d.data.gt_data_dir
    pred_original_dir = cfg_d.data.preds_data_dir  # ORIGINAL pre-refinement preds
    output_dir = Path(eval_dir)

    idx = 0

    # Prepare folders
    pred_dir = output_dir / "refiner_predictions"
    stats_dir = output_dir / "stats"
    pred_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Load original predictions ONCE using your function definition
    pred_original_dict = load_pred_data(cfg_d, ood=ood)

    for cam in cfg_d.data.view_names:

        # ---------------- LOAD GT ----------------
        path = Path(data_dir) / f"CollectedData_{cam}{suffix}.csv"
        if not path.exists():
            print(f"GT not found for {cam}, skipping")
            continue

        df_gt = pd.read_csv(path, header=[0, 1, 2])
        flat_cols = ['_'.join(col).strip() for col in df_gt.columns.values]
        df_gt.columns = flat_cols

        coord_cols = [
            col for col in flat_cols
            if any(kp in col and coord in col
                   for kp in cfg_d.data.keypoint_names
                   for coord in ['x', 'y'])
        ]

        num_rows = df_gt.shape[0]
        num_coords_per_frame = len(coord_cols)

        # Slice refined predictions; guard against length mismatches
        preds_refined = prediction_array[idx:idx + num_rows, :num_coords_per_frame]
        rows = preds_refined.shape[0]
        idx += rows

        if rows != num_rows:
            print(
                f"Prediction rows ({rows}) != GT rows ({num_rows}) "
                f"for {cam}{suffix}; skipping save for this camera."
            )
            continue

        preds_refined = preds_refined.astype(np.float64, copy=False)

        # ---------------- SAVE REFINED PREDICTIONS CSV ----------------
        df_pred_ref = df_gt.copy()
        df_pred_ref.loc[:, coord_cols] = preds_refined

        col_tuples = [tuple(c.split("_")) for c in flat_cols]
        df_pred_ref.columns = pd.MultiIndex.from_tuples(col_tuples)

        out_pred = pred_dir / f"predictions_{cam}{suffix}_refined.csv"
        df_pred_ref.to_csv(out_pred, index=False)
        print(f"Saved refined predictions → {out_pred}")

        # =====================================================================
        # --------------------- PIXEL ERRORS (REFINED) -----------------------
        # =====================================================================
        gt_xy = df_gt.loc[:, coord_cols].to_numpy()
        refined_errs = []
        for k in range(0, num_coords_per_frame, 2):
            err = np.sqrt(((preds_refined[:, k:k+2] - gt_xy[:, k:k+2]) ** 2).sum(axis=1))
            refined_errs.append(err)
        refined_errs = np.vstack(refined_errs).T  # N × K

        # =====================================================================
        # ------------------ PIXEL ERRORS (ORIGINAL PREDICTIONS) --------------
        # =====================================================================
        if cam not in pred_original_dict:
            print(f"No original predictions for {cam}, skipping stats plots.")
            continue

        df_orig = pred_original_dict[cam]  # Already flattened & filtered

        # Extract only xy (remove likelihood columns)
        orig_coord_cols = []
        for kp in cfg_d.data.keypoint_names:
            orig_coord_cols.extend([f"heatmap_multiview_transformer_tracker_{kp}_x", f"heatmap_multiview_transformer_tracker_{kp}_y"])

        orig_xy = df_orig[orig_coord_cols].to_numpy()

        orig_errs = []
        for k in range(0, num_coords_per_frame, 2):
            err = np.sqrt(((orig_xy[:, k:k+2] - gt_xy[:, k:k+2]) ** 2).sum(axis=1))
            orig_errs.append(err)
        orig_errs = np.vstack(orig_errs).T  # N × K

        # =====================================================================
        # ------------------------- SORT ORDER (ORIGINAL) ---------------------
        # =====================================================================
        # The difficulty ordering is based on ORIGINAL pixel error
        # This is 1D per frame → we choose L2 norm across keypoints or max?
        # The user wants CDF-like sorting per keypoint → so sort per keypoint individually.

        keypoints = cfg_d.data.keypoint_names

        cam_stats_dir = stats_dir / f"{cam}"
        cam_stats_dir.mkdir(exist_ok=True)

        # =====================================================================
        # ------------------- PLOTS: ORIGINAL vs REFINED ----------------------
        # =====================================================================
        for i, kp in enumerate(keypoints):

            orig_kp = orig_errs[:, i]
            refined_kp = refined_errs[:, i]

            # Sort frames by ORIGINAL difficulty
            order = np.argsort(orig_kp)

            orig_sorted = orig_kp[order]
            refined_sorted = refined_kp[order]

            # --------------------- PLOT ---------------------
            plt.figure(figsize=(7, 5))
            plt.plot(orig_sorted, label="Original Error", linewidth=2)
            plt.plot(refined_sorted, label="Refined Error", linewidth=2)
            plt.xlabel("Frames (sorted by original difficulty)")
            plt.ylabel("Pixel Error")
            plt.title(f"{cam} — {kp}\nOriginal vs Refined Pixel Error")
            plt.grid(alpha=0.3)
            plt.legend()

            out_plot = cam_stats_dir / f"pixel_error_compare_{cam}_{kp}.png"
            plt.savefig(out_plot, dpi=150, bbox_inches="tight")
            plt.close()

        print(f"Saved comparison stats → {cam_stats_dir}")




class MLPDataModule(pl.LightningDataModule):
    """Data module for MLP training."""

    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 batch_size: int = 64):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # Convert to PyTorch tensors
        self.train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train),
            torch.FloatTensor(self.y_train)
        )
        self.val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val),
            torch.FloatTensor(self.y_val)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class ResidualMLP(pl.LightningModule):
    """Residual MLP for keypoint refinement."""

    def __init__(self,
                 input_dim: int,
                 n_keypoints: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout: float = 0.1,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.n_keypoints = int(n_keypoints)

        self.layers = nn.ModuleList()
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer - predicts coordinate adjustments
        self.output_layer = nn.Linear(prev_dim, self.n_keypoints * 2)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.n_keypoints * 2
        residual = x[:, :base]  # Original coordinates

        for layer in self.layers:
            x = layer(x)

        adjustments = self.output_layer(x)

        # Residual connection: original coords + learned adjustments
        return residual + adjustments

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)

        # Calculate MPJPE (Mean Per Joint Position Error)
        mpjpe = torch.sqrt(F.mse_loss(y_pred, y, reduction='none')).mean(dim=1).mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mpjpe', mpjpe, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

class PoseMLPLOO(pl.LightningModule):
    """
    Pose-level MLP for leave-one-out (LOO) reconstruction.

    Input:  per-example pose vector, e.g. [x_0, y_0, ..., x_{K-1}, y_{K-1}, mask_0, ..., mask_{K-1}]
            where mask_i = 1 if keypoint i is DROPPED, 0 otherwise.
    Output: predicted coordinates for ALL keypoints [x_hat_0, y_hat_0, ..., x_hat_{K-1}, y_hat_{K-1}].

    During training, we only compute loss on the dropped keypoint(s), as indicated by the mask.
    """

    def __init__(
            self,
            input_dim: int,
            n_keypoints: int,
            hidden_dims: List[int] = [512, 256, 128],
            dropout: float = 0.1,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_keypoints = int(n_keypoints)

        self.layers = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output full pose: x,y for each keypoint
        self.output_layer = nn.Linear(prev_dim, self.n_keypoints * 2)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No residual connection here; we just output a reconstructed pose
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)  # [B, 2K]

    def _compute_loss(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        x: [B, D]  (includes mask in last K dims)
        y_true: [B, 2K]  (original pose coords)

        We only compute loss on the DROPPED keypoint(s) where mask == 1.
        """
        B = x.shape[0]
        K = self.n_keypoints
        # mask is assumed to be last K features
        mask = x[:, -K:]  # [B, K], 1 = dropped keypoint

        y_pred = self(x)  # [B, 2K]

        # Reshape for convenience
        y_pred_reshaped = y_pred.view(B, K, 2)  # [B, K, 2]
        y_true_reshaped = y_true.view(B, K, 2)  # [B, K, 2]

        # Expand mask to (B, K, 2) so we apply the loss only to dropped joints
        mask_expanded = mask.unsqueeze(-1).expand_as(y_true_reshaped)  # [B, K, 2]

        diff = (y_pred_reshaped - y_true_reshaped) ** 2  # [B, K, 2]
        masked_diff = diff * mask_expanded  # [B, K, 2]

        # Avoid divide-by-zero if no dropped joints in a batch
        denom = mask_expanded.sum()
        if denom < 1.0:
            # Fallback: plain MSE on all joints
            return F.mse_loss(y_pred, y_true)

        loss = masked_diff.sum() / denom
        return loss

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        loss = self._compute_loss(x, y_true)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        loss = self._compute_loss(x, y_true)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
