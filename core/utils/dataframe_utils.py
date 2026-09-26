import pandas as pd


def is_numeric_column(df: pd.DataFrame, column: str) -> bool:
    """Return True if the given DataFrame column contains numeric data."""
    if column not in df.columns:
        return False

    return pd.api.types.is_numeric_dtype(df[column])


def validate_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return columns that are missing from the DataFrame."""
    return [column for column in columns if column not in df.columns]
