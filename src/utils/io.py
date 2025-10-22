"""Tiny I/O helpers."""
from pathlib import Path
import pandas as pd

def read_csv(p: str): return pd.read_csv(p)
def write_csv(df, p: str): Path(p).parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False)
