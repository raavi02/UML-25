"""Lightweight logger + seeding."""
import json, random, numpy as np
def log_json(d: dict, path: str):
    with open(path, "w") as f: json.dump(d, f, indent=2)
def seed_everything(s: int = 42):
    random.seed(s); np.random.seed(s)
