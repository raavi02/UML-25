"""Tiny I/O helpers."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml
from omegaconf import DictConfig



def read_csv(p: str): return pd.read_csv(p)
def write_csv(df, p: str): Path(p).parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False)


def project_root() -> Path:
    """Resolve the repository root assuming this file lives at root/model/gnn_refiner.py."""
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def load_cfgs(config_file: str):
    # Load pipeline config file
    with open(config_file, "r") as file:
        cfg_pipe = yaml.safe_load(file)
    cfg_pipe = DictConfig(cfg_pipe)

    # Load lightning pose config file from the path specified in pipeline config
    dataset_config_path = cfg_pipe.get("dataset_config")
    with open(dataset_config_path, "r") as file:
        dataset_cfg = yaml.safe_load(file)

    cfg_d = DictConfig(dataset_cfg)
    return cfg_pipe, cfg_d
