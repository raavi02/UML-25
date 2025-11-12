from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Callable, Dict, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class GridSearchRunner:
    """
    Model-agnostic grid search.

    build_model:      (params, n_keypoints) -> pl.LightningModule
    build_datamodule: (params, splits) -> pl.LightningDataModule
    """
    def __init__(
        self,
        output_dir: str | Path,
        param_grid: Dict[str, List[Any]],
        build_model: Callable[[Dict[str, Any], int], pl.LightningModule],
        build_datamodule: Callable[[Dict[str, Any], Dict[str, Any]], pl.LightningDataModule],
        monitor_metric: str = "val_loss",
        mode: str = "min",
        patience: int = 15,
        max_epochs: int = 100,
        accelerator: str = "auto",
        n_keypoints: int | None = None,   # <-- only used for model
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.param_grid = param_grid
        self.build_model = build_model
        self.build_datamodule = build_datamodule
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.patience = patience
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.n_keypoints = int(n_keypoints) if n_keypoints is not None else None

    def generate_param_combinations(self) -> List[Dict[str, Any]]:
        import itertools
        keys = list(self.param_grid.keys())
        vals = [self.param_grid[k] for k in keys]
        combos: List[Dict[str, Any]] = []
        for vs in itertools.product(*vals):
            d = dict(zip(keys, vs))
            slug = hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]
            d["_slug"] = slug
            combos.append(d)
        return combos

    def run(self, splits: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for params in self.generate_param_combinations():
            try:
                res = self._train_single(params, splits)
                results.append(res)
                self._save_results(results)
            except Exception as e:
                print(f"[GridSearchRunner] Failed {params}: {e}")
        return results

    def _train_single(self, params: Dict[str, Any], splits: Dict[str, Any]) -> Dict[str, Any]:
        # n_keypoints -> model only
        model = self.build_model(params, self.n_keypoints)
        datamodule = self.build_datamodule(params, splits)

        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            monitor=self.monitor_metric,
            dirpath=ckpt_dir,
            filename=f"model-{params['_slug']}",
            save_top_k=1,
            mode=self.mode,
        )
        early_cb = EarlyStopping(
            monitor=self.monitor_metric,
            patience=self.patience,
            mode=self.mode
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_cb, early_cb],
            enable_progress_bar=True,
            log_every_n_steps=10,
            accelerator=self.accelerator,
        )
        trainer.fit(model, datamodule)

        clean_params = {k: v for k, v in params.items() if k != "_slug"}
        return {
            "params": clean_params,
            "best_score": float(checkpoint_cb.best_model_score.item()),
            "best_path": str(checkpoint_cb.best_model_path),
        }

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        serializable = [
            {"params": r["params"], "best_score": float(r["best_score"]), "best_path": r["best_path"]}
            for r in results
        ]
        serializable.sort(key=lambda x: x["best_score"])
        with open(self.output_dir / "grid_search_results.json", "w") as f:
            json.dump(serializable, f, indent=2)
