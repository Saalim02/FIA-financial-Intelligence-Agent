"""Convert analytical results into values accepted by strict JSON encoders."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """Recursively replace missing/non-finite numeric values with JSON null."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, (pd.Series, pd.Index)):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.generic):
        item = value.item()
        return item if item is value else json_safe(item)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value
