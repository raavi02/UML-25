import pandas as pd
from .schemas import PRED_COLUMNS
def load_predictions(dataset_dir: str) -> pd.DataFrame:
    # TODO: glob real files; return validated dataframe
    return pd.DataFrame(columns=PRED_COLUMNS)
