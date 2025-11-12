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
from src.data_loading.load_data import (load_gt_data, load_pred_data,
                                        prepare_mlp_data)
from src.model.mlp import run_mlp_grid_search, evaluate_mlp_checkpoint, create_summary_report
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

    # load cfg (pipeline yaml) and cfg_lp (lp yaml)
    cfg_pipe, cfg_d = load_cfgs(config_file)  # cfg_lp is a DictConfig, cfg_pipe is not

    # Define directories
    dataset = cfg_pipe.dataset_name
    data_dir = cfg_pipe.dataset_dir
    base_outputs_dir = Path(cfg_pipe.outputs_dir)
    gt_dir = os.path.join(data_dir, f'{dataset}_ground_truth')
    preds_dir = os.path.join(data_dir, f'{dataset}_predictions')

    # Dataset parameters
    cameras = cfg_d.data.view_names
    keypoints = cfg_d.data.keypoint_names
    skeleton = cfg_d.data.skeleton

    # Train parameters
    model_type = cfg_pipe.train.model_type
    seed = cfg_pipe.train.seed
    max_epochs = cfg_pipe.train.max_epochs
    val_ratio = cfg_pipe.train.val_ratio
    pl.seed_everything(cfg_pipe.train.seed, workers=True)

    outputs_dir = base_outputs_dir / f"{model_type}_{seed}"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f'--------------DATASET PARAMETERS------------')
    print(f'Dataset: {dataset}')
    print(f'Data Directory: {data_dir}')
    print(f'Output Directory: {outputs_dir}')
    print(f'Ground Truth Directory: {gt_dir}')
    print(f'Predictions Directory: {preds_dir}')
    print(f'Camera views: {cameras}')
    print(f'Keypoints: {keypoints}')
    print(f'Bones: {skeleton}')
    print(f'--------------------------------------------')

    print(f'--------------MODEL PARAMETERS------------')
    print(f'Network Type: {model_type}')
    print(f'Seed: {seed}')
    print(f'Max Epochs: {max_epochs}')
    print(f'Val Ratio: {val_ratio}')
    print(f'------------------------------------------')

    # -------------------------------------------
    # Load raw data
    # -------------------------------------------
    logger.info("Loading train/test data...")
    gt_train = load_gt_data(cfg_d.copy(), ood=False)
    pred_train = load_pred_data(cfg_d.copy(), ood=False)
    gt_test = load_gt_data(cfg_d.copy(), ood=True)
    pred_test = load_pred_data(cfg_d.copy(), ood=True)

    # Build index split
    logger.info("Preparing temporary features for index split...")
    X_all, y_all = prepare_mlp_data(
        gt_data=gt_train,
        pred_data=pred_train,
        use_confidence=True
    )
    idx = np.arange(len(X_all))
    tr_idx, va_idx = train_test_split(idx, test_size=cfg_pipe.train.val_ratio,
                                      random_state=seed)
    logger.info(f"Data shape - X: {X_all.shape}, y: {y_all.shape}")
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
    if cfg_pipe.train.model_type == "MLP":

        cache_file = outputs_dir / "best_result.json"

        if cache_file.exists() and not getattr(cfg_pipe.train, "force_grid", False):
            logger.info(f"Found cached best params → {cache_file.name}, skipping grid search.")
            with open(cache_file, "r") as f:
                best = json.load(f)
        else:
            logger.info("Running grid search for best params...")
            best = run_mlp_grid_search(
                splits=splits,
                output_dir=outputs_dir,
                max_epochs=cfg_pipe.train.max_epochs,
                n_keypoints=cfg_d.data.num_keypoints,
            )
            with open(cache_file, "w") as f:
                json.dump(best, f, indent=2)
            logger.info(f"Saved best grid-search result → {cache_file}")

        def debug_split_dict(name, d):
            logger.info(f"[DEBUG] {name}: {list(d.keys())}")
            for k, v in d.items():
                logger.info(f"[DEBUG] {name}[{k}] rows={len(v)} cols={v.shape[1]}")

        logger.info("Sanity-checking test splits…")
        debug_split_dict("gt_test", splits["gt_test"])
        debug_split_dict("pred_test", splits["pred_test"])

        common = set(splits["gt_test"].keys()) | set(splits["pred_test"].keys())
        logger.info(f"[DEBUG] common cams: {sorted(common)}")

        # Build test arrays with winning feature layout
        use_conf = best["params"]["use_confidence"]
        X_test, y_test = prepare_mlp_data(
            splits["gt_test"], splits["pred_test"],
            use_confidence=use_conf
        )

        # Evaluate on test data (OOD)
        metrics = evaluate_mlp_checkpoint(
            best_path=best["best_path"],
            X_test=X_test,
            y_test=y_test,
            output_dir=outputs_dir,
        )

    elif cfg_pipe.train.model_type == "GNN":
        # from src.models.gnn.experiment import run_gnn_grid_search, evaluate_gnn_checkpoint
        # best = run_gnn_grid_search(splits=splits, output_dir=outputs_dir / "gnn", max_epochs=cfg_pipe.train.max_epochs)
        # X_test, y_test = prepare_gnn_data(...)
        # metrics = evaluate_gnn_checkpoint(best["best_path"], X_test, y_test)
        raise NotImplementedError("GNN path to be added.")
    else:
        raise ValueError("model_type must be 'MLP' or 'GNN'")

    # Collect results and export
    final_results = {
        "best_hyperparameters": best["params"],
        "best_validation_loss": float(best["best_score"]),
        "test_metrics": metrics,
        "data_shapes": {
            "train": (len(splits["tr_idx"]), X_test.shape[1]),
            "val": (len(splits["va_idx"]), X_test.shape[1]),
            "test": X_test.shape,
        },
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    create_summary_report(final_results, outputs_dir)
    logger.info("Summary report created.")
    logger.info(f"Path: {Path(cfg_pipe.outputs_dir) / 'summary_report.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()
    pipeline(args.config)
