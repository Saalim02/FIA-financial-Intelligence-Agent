from io import BytesIO
from uuid import uuid4

import pandas as pd


class DatasetService:
    def __init__(self):
        self.datasets: dict[str, pd.DataFrame] = {}

    def load_dataset(self, filename: str, contents: bytes) -> tuple[str, pd.DataFrame]:
        filename_lower = filename.lower()

        if filename_lower.endswith(".csv"):
            df = pd.read_csv(BytesIO(contents))

        elif filename_lower.endswith(".xlsx"):
            df = pd.read_excel(
                BytesIO(contents),
                engine="openpyxl"
            )

        elif filename_lower.endswith(".xls"):
            df = pd.read_excel(BytesIO(contents))

        else:
            raise ValueError(
                "Only CSV, XLSX, and XLS files are supported."
            )

        if df.empty:
            raise ValueError("Dataset is empty.")

        dataset_id = str(uuid4())

        self.datasets[dataset_id] = df

        return dataset_id, df

    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        if dataset_id not in self.datasets:
            raise KeyError(f"Dataset not found: {dataset_id}")

        return self.datasets[dataset_id]
