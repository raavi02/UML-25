"""Tiny I/O helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from omegaconf import OmegaConf, DictConfig


def read_csv(p: str | Path) -> pd.DataFrame:
    return pd.read_csv(p)

def write_csv(df: pd.DataFrame, p: str | Path) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

def project_root() -> Path:
    """Resolve the repository root assuming this file lives at root/model/run_pipeline.py."""
    return Path(__file__).resolve().parents[1]

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def load_cfgs(config_file: str | Path) -> Tuple[DictConfig, DictConfig]:
    config_file = Path(config_file).resolve()

    # Load pipeline config
    cfg_pipe: DictConfig = OmegaConf.load(config_file)

    # Resolve dataset_config path relative to the pipeline config file
    dataset_config_raw = str(cfg_pipe.dataset_config)  # or cfg_pipe["dataset_config"]
    dataset_config_path = Path(dataset_config_raw)
    if not dataset_config_path.is_absolute():
        dataset_config_path = config_file.parent / dataset_config_path

    # Load dataset config
    cfg_d: DictConfig = OmegaConf.load(dataset_config_path)

    return cfg_pipe, cfg_d
