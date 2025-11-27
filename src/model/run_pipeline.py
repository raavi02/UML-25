from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict
import json

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from src.data_loading.load_data import (
    load_gt_data,
    load_pred_data,
    prepare_data,
)
from src.model.mlp import (
    run_mlp_grid_search,
    evaluate_mlp_checkpoint,
    save_predictions,
    create_summary_report as create_mlp_summary_report,
)
from src.model.gnn import (
    run_gnn_grid_search,
    evaluate_gnn_checkpoint,
    create_summary_report as create_gnn_summary_report,
)
from src.utils.io import load_cfgs
from src.utils.logging import logger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# --------------------------------------------------------------------------- #
# Main GNN/MLP Pipeline
# --------------------------------------------------------------------------- #

def pipeline(config_file: str, for_seed: int | None = None) -> None:
    # -------------------------------------------
    # Setup
    # -------------------------------------------

    # load cfg (pipeline yaml) and cfg_d (data yaml)
    cfg_pipe, cfg_d = load_cfgs(config_file)

    # Define directories
    dataset = cfg_pipe.dataset_name
    data_dir = cfg_pipe.dataset_dir
    base_outputs_dir = Path(cfg_pipe.outputs_dir)
    gt_dir = os.path.join(data_dir, "ground_truth")
    preds_dir = os.path.join(data_dir, "predictions")

    # Dataset parameters
    cameras = cfg_d.data.view_names
    keypoints = cfg_d.data.keypoint_names
    skeleton = cfg_d.data.skeleton
    use_conf_flag = bool(getattr(cfg_pipe.train, "use_confidence", True))
    run_grid_search = bool(getattr(cfg_pipe.train, "run_grid_search", True))
    run_eval_on_ood = bool(getattr(cfg_pipe.train, "run_eval_on_ood", True))
    mode = getattr(cfg_pipe.train, "mode", "refine")  # "refine" or "loo"

    # Train parameters
    model_type = cfg_pipe.train.model_type
    seed = cfg_pipe.train.seed if for_seed is None else for_seed
    max_epochs = cfg_pipe.train.max_epochs
    val_ratio = cfg_pipe.train.val_ratio
    pl.seed_everything(seed, workers=True)

    outputs_dir = base_outputs_dir / f"{model_type}_{seed}"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("--------------DATASET PARAMETERS------------")
    print(f"Dataset: {dataset}")
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {outputs_dir}")
    print(f"Ground Truth Directory: {gt_dir}")
    print(f"Predictions Directory: {preds_dir}")
    print(f"Camera views: {cameras}")
    print(f"Keypoints: {keypoints}")
    print(f"Bones: {skeleton}")
    print("--------------------------------------------")

    print("--------------MODEL PARAMETERS------------")
    print(f"Network Type: {model_type}")
    print(f"Mode: {mode}")
    print(f"Seed: {seed}")
    print(f"Max Epochs: {max_epochs}")
    print(f"Val Ratio: {val_ratio}")
    print("------------------------------------------")

    # -------------------------------------------
    # Load raw data
    # -------------------------------------------
    logger.info("Loading train/test data...")
    gt_train = load_gt_data(cfg_d.copy(), ood=False)
    pred_train = load_pred_data(cfg_d.copy(), ood=False)
    if run_eval_on_ood:
        gt_test = load_gt_data(cfg_d.copy(), ood=True)
        pred_test = load_pred_data(cfg_d.copy(), ood=True)
    else:
        gt_test, pred_test = {}, {}

    # -------------------------------------------
    # Build index split (using supervised-style features just to get N)
    # -------------------------------------------
    logger.info("Preparing temporary features for index split...")
    X_all_tmp, y_all_tmp = prepare_data(
        gt_data=gt_train,
        pred_data=pred_train,
        use_confidence=use_conf_flag,
    )
    feature_dim = X_all_tmp.shape[1]
    idx = np.arange(len(X_all_tmp))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=val_ratio,
        random_state=seed,
    )
    logger.info(f"Data shape (temp) - X: {X_all_tmp.shape}, y: {y_all_tmp.shape}")
    logger.info(f"Train/val split - {len(tr_idx)} train, {len(va_idx)} val")

    splits: Dict[str, Any] = {
        "gt_train": gt_train,
        "pred_train": pred_train,
        "gt_val": gt_train,
        "pred_val": pred_train,
        "gt_test": gt_test,
        "pred_test": pred_test,
        "tr_idx": tr_idx,
        "va_idx": va_idx,
        "test_indices": None,
    }

    # -------------------------------------------
    # Train & Evaluate
    # -------------------------------------------
    if model_type == "MLP":

        cache_file = outputs_dir / "best_result.json"

        if not run_grid_search:
            if not cache_file.exists():
                raise FileNotFoundError(
                    f"run_grid_search is False and cached params not found at {cache_file}. "
                    "Run once with run_grid_search=True to generate best_result.json."
                )
            logger.info(
                f"Skipping grid search (run_grid_search=False). "
                f"Loading cached params → {cache_file.name}."
            )
            with open(cache_file, "r") as f:
                best = json.load(f)

        elif cache_file.exists() and not getattr(cfg_pipe.train, "force_grid", False):
            logger.info(
                f"Found cached best params → {cache_file.name}, skipping grid search."
            )
            with open(cache_file, "r") as f:
                best = json.load(f)

        else:
            logger.info("Running grid search for best params...")
            best = run_mlp_grid_search(
                splits=splits,
                output_dir=outputs_dir,
                max_epochs=max_epochs,
                n_keypoints=cfg_d.data.num_keypoints,
                use_confidence_options=[use_conf_flag],
                mode=mode,
            )
            with open(cache_file, "w") as f:
                json.dump(best, f, indent=2)
            logger.info(f"Saved best grid-search result → {cache_file}")

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------
        use_conf = best["params"]["use_confidence"]

        if run_eval_on_ood and mode == "refine":
            # In refine mode, we can evaluate L2 vs GT on OOD test set
            X_test, y_test = prepare_data(
                gt_data=splits["gt_test"],
                pred_data=splits["pred_test"],
                use_confidence=use_conf,
            )

            metrics = evaluate_mlp_checkpoint(
                best_path=best["best_path"],
                X_test=X_test,
                y_test=y_test,
                output_dir=outputs_dir,
                cfg_d=cfg_d,
                save_preds=cfg_pipe.train.save_predictions
            )
              
        else:
            # In LOO mode we don't yet have a standard L2-to-GT evaluation;
            # or if run_eval_on_ood is False, we skip test eval.
            X_test = np.empty((0, feature_dim))
            y_test = np.empty((0, cfg_d.data.num_keypoints * 2))
            metrics = {}

    elif model_type == "GNN":

        cache_file = outputs_dir / "best_result.json"

        # ---------- grid search / cached best ----------
        if not run_grid_search:
            if not cache_file.exists():
                raise FileNotFoundError(
                    f"run_grid_search is False and cached params not found at {cache_file}. "
                    "Run once with run_grid_search=True to generate best_result.json."
                )
            logger.info(
                f"Skipping grid search (run_grid_search=False). "
                f"Loading cached params → {cache_file.name}."
            )
            with open(cache_file, "r") as f:
                best = json.load(f)

        elif cache_file.exists() and not getattr(cfg_pipe.train, "force_grid", False):
            logger.info(
                f"Found cached best params → {cache_file.name}, skipping grid search."
            )
            with open(cache_file, "r") as f:
                best = json.load(f)

        else:
            logger.info("Running grid search for best GNN params...")
            best = run_gnn_grid_search(
                splits=splits,
                output_dir=outputs_dir,
                max_epochs=max_epochs,
                n_keypoints=cfg_d.data.num_keypoints,
                skeleton=skeleton,
                keypoints=keypoints,
                mode=mode,
                use_confidence_options=[use_conf_flag],
            )
            with open(cache_file, "w") as f:
                json.dump(best, f, indent=2)
            logger.info(f"Saved best GNN grid-search result → {cache_file}")

        # ---------- evaluation ----------
        use_conf = best["params"]["use_confidence"]

        if run_eval_on_ood and mode == "refine":
            X_test, y_test = prepare_data(
                gt_data=splits["gt_test"],
                pred_data=splits["pred_test"],
                use_confidence=use_conf,
            )

            metrics = evaluate_gnn_checkpoint(
                best_path=best["best_path"],
                X_test=X_test,
                y_test=y_test,
                output_dir=outputs_dir,
                cfg_d=cfg_d,
                save_preds=cfg_pipe.train.save_predictions
            )
        else:
            X_test = np.empty((0, feature_dim))
            y_test = np.empty((0, cfg_d.data.num_keypoints * 2))
            metrics = {}

    # -------------------------------------------
    # Collect results and export summary
    # -------------------------------------------
    final_results = {
        "best_hyperparameters": best["params"],
        "best_validation_loss": float(best["best_score"]),
        "test_metrics": metrics,
        "data_shapes": {
            "train": (len(splits["tr_idx"]), feature_dim),
            "val": (len(splits["va_idx"]), feature_dim),
            "test": X_test.shape,
        },
        "mode": mode,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    if model_type == "MLP":
        create_mlp_summary_report(final_results, outputs_dir)
    elif model_type == "GNN":
        create_gnn_summary_report(final_results, outputs_dir)
    logger.info("Summary report created.")
    logger.info(f"Path: {Path(cfg_pipe.outputs_dir) / 'summary_report.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    pipeline(args.config)
