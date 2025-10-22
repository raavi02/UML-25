"""Config loading & merge helpers."""
from pathlib import Path
import yaml

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)
