# src/models/gnn.py
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd
from typing import Any, Dict, Optional, List, Tuple
import json
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch.utils.data import TensorDataset, DataLoader

from src.utils.logging import logger

from src.evaluation.metrics import evaluate_model, calculate_improvement
from src.data_loading.load_data import prepare_data, load_pred_data

def convert_labels_to_indices(keypoints: List[str], skeleton: List[Tuple[str, str]]) -> List[Tuple[int, int]]:
    keypoint_to_index = {kp: idx for idx, kp in enumerate(keypoints)}
    skeleton_indices = []
    for joint_a, joint_b in skeleton:
        idx_a = keypoint_to_index.get(joint_a)
        idx_b = keypoint_to_index.get(joint_b)
        if idx_a is not None and idx_b is not None:
            skeleton_indices.append((idx_a, idx_b))
    return np.array(skeleton_indices).T  # shape (2, num_edges)

def _num_keypoints(cfg: Any) -> int:
    # Works with DictConfig (OmegaConf) or plain dict
    return int(cfg.data.num_keypoints)

def build_gnn_model(
    params: Dict[str, Any],
    n_keypoints: int,
    mode: str = "refine",
    loo_input_dim: int | None = None,
    skeleton: List[Tuple[str, str]] | None = None,
    keypoints: List[str] | None = None,
) -> pl.LightningModule:
    """
    mode:
      - 'refine': ResidualGNN supervised refiner (residual on coords)
      - 'loo':    PoseGNNLOO self-supervised leave-one-out model
    """
    use_conf = params.get("use_confidence", True)

    if keypoints is None or skeleton is None:
        raise ValueError("build_gnn_model requires keypoints and skeleton lists.")

    # Convert labeled skeleton to index-based edge_index (2, num_edges)
    skeleton_indices = convert_labels_to_indices(keypoints, skeleton)

    if mode == "refine":
        input_dim = n_keypoints * 2 + (n_keypoints if use_conf else 0)
        return ResidualGNN(
            input_dim=input_dim,
            n_keypoints=n_keypoints,
            hidden_dims=params["hidden_dims"],
            hidden_heads=params["hidden_heads"],
            skeleton=skeleton_indices,
            negative_slope=params.get("negative_slope", 0.2),
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            use_gat=params.get("use_gat", False)
        )

    elif mode == "loo":
        if loo_input_dim is None:
            raise ValueError("loo_input_dim must be provided when mode='loo'.")
        return PoseGNNLOO(
            input_dim=loo_input_dim,
            n_keypoints=n_keypoints,
            skeleton=skeleton_indices,
            hidden_dims=params["hidden_dims"],
            hidden_heads=params["hidden_heads"],
            negative_slope=params.get("negative_slope", 0.2),
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            use_confidence=use_conf,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_gnn_datamodule(
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

    return GNNDataModule(
        X_train, y_train, X_val, y_val,
        batch_size=params["batch_size"],
    )

def evaluate_gnn_checkpoint(best_path: str, X_test: np.ndarray, y_test: np.ndarray,
                            output_dir: str | Path = "gnn_results", cfg_d=None, save_preds=False) -> Dict[str, float]:
    """Evaluate a trained GNN checkpoint on given test arrays and log detailed metrics."""

    output_dir = Path(output_dir)
    eval_dir = output_dir / "gnn" / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load model + device
    # -------------------------
    logger.info(f"Loading model from: {best_path}")
    model = ResidualGNN.load_from_checkpoint(best_path)
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
    print("GNN BASELINE - FINAL EVALUATION RESULTS")
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

# ---- convenience function we can call from pipeline ----

def run_gnn_grid_search(
    splits: Dict[str, Any],
    output_dir: str | Path,
    max_epochs: int,
    n_keypoints: int,
    skeleton: List[Tuple[str, str]],
    keypoints: List[str],
    mode: str = "refine",
    use_confidence_options: List[bool] | None = None,
    use_gat = False
) -> Dict[str, Any]:
    from src.model.grid_search import GridSearchRunner

    # Ensure plain Python containers (in case cfg objects are passed)
    keypoints_list = list(keypoints)
    skeleton_list = [tuple(edge) for edge in skeleton]

    use_conf_opts = use_confidence_options if use_confidence_options is not None else [True, False]
    if use_gat:
        param_grid = {
            "hidden_dims": [[16, 8], [8, 4], [16, 8, 4]],
            "hidden_heads": [[8, 4], [4, 2], [8, 4, 2]],
            "dropout": [0.1, 0.2],
            "negative_slope": [0.0, 0.2],
            "learning_rate": [1e-3, 5e-4],
            "weight_decay": [1e-4, 1e-5],
            "use_confidence": use_conf_opts,
            "batch_size": [64, 128],
            "use_gat": [use_gat]
        }
    else:
        param_grid = {
            "hidden_dims": [[32, 32], [64, 32], [64, 32, 32]],
            "hidden_heads": [[1]],
            "dropout": [0.1, 0.2],
            "negative_slope": [0.0, 0.2],
            "learning_rate": [1e-3, 5e-4],
            "weight_decay": [1e-4, 1e-5],
            "use_confidence": use_conf_opts,
            "batch_size": [64, 128],
            "use_gat": [use_gat]
        }


    def _build_model_for_params(params: Dict[str, Any], n_k: int) -> pl.LightningModule:
        use_conf = params.get("use_confidence", True)

        if mode == "refine":
            loo_input_dim = None
        else:
            # coords (2K) + optional conf (K) + mask (K)
            base_dim = 2 * n_k + (n_k if use_conf else 0)
            loo_input_dim = base_dim + n_k

        return build_gnn_model(
            params=params,
            n_keypoints=n_k,
            mode=mode,
            loo_input_dim=loo_input_dim,
            skeleton=skeleton_list,
            keypoints=keypoints_list,
        )

    def _build_datamodule_for_params(
        params: Dict[str, Any],
        splits_: Dict[str, Any],
    ) -> pl.LightningDataModule:
        return build_gnn_datamodule(
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
        raise RuntimeError("No successful GNN runs.")
    best = min(results, key=lambda r: r["best_score"])
    return best  # {params, best_score, best_path}


def create_summary_report(final_results: dict, output_dir: str):
    """Create a human-readable summary report."""
    report_path = Path(output_dir) / 'summary_report.txt'

    with open(report_path, 'w') as f:
        f.write("GNN Baseline - Summary Report\n")
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

        # Slice refined predictions
        preds_refined = prediction_array[idx:idx + num_rows, :num_coords_per_frame]
        idx += num_rows

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





class GNNDataModule(pl.LightningDataModule):
    """Data module for GNN training."""

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


class ResidualGNN(pl.LightningModule):
    """Residual GNN for keypoint refinement."""

    def __init__(self,
                 input_dim: int,
                 n_keypoints: int,
                 hidden_dims: List[int] = [16, 8],
                 hidden_heads: List[int] = [8,4],
                 skeleton: np.ndarray=None,
                 negative_slope: float = 0.2,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 use_gat: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.n_keypoints = int(n_keypoints)
        # self.skeleton = torch.tensor(skeleton, dtype=torch.long)  # shape (2, num_edges)
        self.register_buffer("skeleton", torch.tensor(skeleton, dtype=torch.long))
        self.layers = nn.ModuleList()
        self.use_gat = use_gat
        prev_dim = input_dim//n_keypoints
        prev_head=1
        if not self.use_gat:
            hidden_heads = [1] * len(hidden_dims)   # <-- force heads=1 for GCN

        # Build hidden layers
        for hidden_dim,hidden_head in zip(hidden_dims,hidden_heads):
            if self.use_gat:
                conv = GATConv(in_channels=prev_dim*prev_head,
                                       out_channels=hidden_dim,
                                       heads=hidden_head,
                                       negative_slope=negative_slope,
                                       concat=True)
                out_dim_for_norm = hidden_dim * hidden_head
                prev_head = hidden_head
            else:
                conv = GCNConv(
                    in_channels=prev_dim * prev_head,
                    out_channels=hidden_dim,
                )
                out_dim_for_norm = hidden_dim
                prev_head = 1  # CHANGED: no heads in GCN, so keep head factor = 1

            self.layers.append(conv)
            self.layers.append(nn.BatchNorm1d(out_dim_for_norm))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer - predicts coordinate adjustments
        self.output_layer = nn.Linear(prev_dim*prev_head, 2)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.n_keypoints * 2
        residual = x[:, :base]  # Original coordinates
        x=x.reshape(-1, self.n_keypoints, 2)

        #We now need to reshape x for GNN processing of shape (keypoints*batch_size,2),
        # creating edge_index for each sample in the batch

        batch_size = x.shape[0]
        x = x.reshape(batch_size * self.n_keypoints, 2)
        # edge_index_list=[]
        # for i in range(batch_size):
        #     edge_index_list.append(self.skeleton + i*self.n_keypoints)
        # edge_index=torch.cat(edge_index_list,dim=1)  # shape (2, num_edges * batch_size)
        edge_index = torch.cat(
        [self.skeleton + i * self.n_keypoints for i in range(batch_size)],
        dim=1
        )
        for layer in self.layers:
            x = layer(x, edge_index=edge_index) if isinstance(layer, (GATConv, GCNConv)) else layer(x)

        #Reshape back to (batch_size, n_keypoints, feature_dim) for the linear layer
        x = x.reshape(batch_size, self.n_keypoints, -1)
        adjustments = self.output_layer(x)

        #Reshape adjustments back to (batch_size, n_keypoints*2)
        adjustments=adjustments.reshape(batch_size, -1)

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


class PoseGNNLOO(pl.LightningModule):
    """
    Graph-based pose model for leave-one-out (LOO) reconstruction.

    Input x:  [coords (2K), optional conf (K), mask (K)]
              shape (B, 2K + (K if use_confidence else 0) + K)

      - coords: flattened [x0, y0, ..., x_{K-1}, y_{K-1}]
      - confs:  [c0, ..., c_{K-1}]  (optional)
      - mask:   [m0, ..., m_{K-1}], where m_j = 1 if keypoint j is DROPPED

    Target y_true: original coords, shape (B, 2K).

    We compute loss only on dropped keypoints (mask == 1), mirroring PoseMLPLOO.
    """

    def __init__(
        self,
        input_dim: int,
        n_keypoints: int,
        skeleton: np.ndarray,
        hidden_dims: List[int] = [16, 8],
        hidden_heads: List[int] = [8, 4],
        negative_slope: float = 0.2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_confidence: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_keypoints = int(n_keypoints)
        self.use_confidence = bool(use_confidence)

        # edge_index (2, num_edges)
        self.skeleton = torch.tensor(skeleton, dtype=torch.long)

        # Per-node feature dimension = input_dim // K
        feat_dim = input_dim // self.n_keypoints

        self.layers = nn.ModuleList()
        prev_dim = feat_dim
        prev_head = 1

        for hidden_dim, hidden_head in zip(hidden_dims, hidden_heads):
            # GATConv over all nodes in the batch graph
            self.layers.append(
                GATConv(
                    in_channels=prev_dim * prev_head,
                    out_channels=hidden_dim,
                    heads=hidden_head,
                    negative_slope=negative_slope,
                    concat=True,
                )
            )
            # BatchNorm over (num_nodes_in_batch, C)
            self.layers.append(nn.BatchNorm1d(hidden_dim * hidden_head))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim
            prev_head = hidden_head

        # Per-node output: 2D coords
        self.output_layer = nn.Linear(prev_dim * prev_head, 2)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    # ---- helpers to parse input into node features + mask ----

    def _split_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, D = 2K + (K if use_conf else 0) + K)

        Returns:
          node_feats: (B, K, F)  where F = 3 or 4
          mask:       (B, K)     1 = dropped keypoint
          coords_true:(B, 2K)    flattened true coords (for convenience)
        """
        B = x.shape[0]
        K = self.n_keypoints

        base = 2 * K
        coords_flat = x[:, :base]  # (B, 2K)
        coords = coords_flat.view(B, K, 2)  # (B, K, 2)

        if self.use_confidence:
            conf_start = base
            conf_end = conf_start + K
            conf = x[:, conf_start:conf_end].view(B, K, 1)  # (B, K, 1)
            mask = x[:, conf_end : conf_end + K].view(B, K)  # (B, K)
            mask_unsq = mask.unsqueeze(-1)  # (B, K, 1)
            node_feats = torch.cat([coords, conf, mask_unsq], dim=-1)  # (B, K, 4)
        else:
            # no conf: coords + mask
            mask = x[:, base : base + K].view(B, K)  # (B, K)
            mask_unsq = mask.unsqueeze(-1)  # (B, K, 1)
            node_feats = torch.cat([coords, mask_unsq], dim=-1)  # (B, K, 3)

        return node_feats, mask, coords_flat

    # ---- forward & loss ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns: predicted coords, shape (B, 2K)
        """
        node_feats, mask, coords_flat = self._split_input(x)
        B, K, F = node_feats.shape

        # Flatten nodes into a single batch graph
        h = node_feats.reshape(B * K, F)

        # Build batched edge_index by offsetting node ids per example
        edge_index_list = []
        for i in range(B):
            edge_index_list.append(self.skeleton + i * K)
        edge_index = torch.cat(edge_index_list, dim=1)  # (2, E_total)

        for layer in self.layers:
            if isinstance(layer, GATConv):
                h = layer(h, edge_index=edge_index)
            else:
                h = layer(h)

        # Back to (B, K, feat_dim)
        h = h.reshape(B, K, -1)
        coords_pred = self.output_layer(h)  # (B, K, 2)
        return coords_pred.reshape(B, 2 * K)  # (B, 2K)

    def _compute_loss(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Only penalize dropped keypoints (mask == 1).
        """
        node_feats, mask, _ = self._split_input(x)
        B, K = mask.shape

        y_pred = self.forward(x)  # (B, 2K)

        y_pred_reshaped = y_pred.view(B, K, 2)
        y_true_reshaped = y_true.view(B, K, 2)

        mask_expanded = mask.unsqueeze(-1).expand_as(y_true_reshaped)  # (B, K, 2)
        diff = (y_pred_reshaped - y_true_reshaped) ** 2
        masked_diff = diff * mask_expanded

        denom = mask_expanded.sum()
        if denom < 1.0:
            # Fallback: plain MSE on all joints if for some reason no drops in batch
            return F.mse_loss(y_pred, y_true)

        loss = masked_diff.sum() / denom
        return loss

    # ---- Lightning hooks ----

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
