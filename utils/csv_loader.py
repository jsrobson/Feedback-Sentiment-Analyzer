import pandas as pd
from pathlib import Path

class CSVLoader:
    def __init__(self, fpath: str):
        self.filepath = Path(fpath)

    def load(self):
        self._validate_path()
        return pd.read_csv(self.filepath)

    def _validate_path(self):
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")
        if not self.filepath.suffix.lower() == ".csv":
            raise ValueError(f"Invalid file type: {self.filepath.suffix}")