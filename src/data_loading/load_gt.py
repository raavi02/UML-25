import pandas as pd
from .schemas import GT_COLUMNS
def load_gt(dataset_dir: str) -> pd.DataFrame:
    # TODO: glob real files; return validated dataframe
    return pd.DataFrame(columns=GT_COLUMNS)
