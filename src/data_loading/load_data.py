from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd

from src.data_loading.load_calib import load_calibration
from src.data_loading.schemas import GT_COLUMNS, PRED_COLUMNS
from src.model.gnn_refiner import logger


def load_gt(dataset_dir: str) -> pd.DataFrame:
    # TODO: glob real files; return validated dataframe
    return pd.DataFrame(columns=GT_COLUMNS)


def load_predictions(dataset_dir: str) -> pd.DataFrame:
    # TODO: glob real files; return validated dataframe
    return pd.DataFrame(columns=PRED_COLUMNS)
