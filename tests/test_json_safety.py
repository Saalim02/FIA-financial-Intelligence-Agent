import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

from api.json_safety import json_safe
from api.routes import query as query_route


def test_json_safe_recursively_converts_nonfinite_and_pandas_missing_values():
    payload = {
        "nan": float("nan"),
        "positive_infinity": float("inf"),
        "negative_infinity": np.float64(float("-inf")),
        "nested": [{"missing": pd.NA}, (np.float32(2.5), np.int64(3))],
        "array": np.array([0.0, float("nan"), 125.5]),
    }

    safe = json_safe(payload)

    assert safe == {
        "nan": None,
        "positive_infinity": None,
        "negative_infinity": None,
        "nested": [{"missing": None}, [2.5, 3]],
        "array": [0.0, None, 125.5],
    }
    assert safe["array"][0] == 0.0
    assert safe["array"][2] == 125.5
    assert all(value is None or not (isinstance(value, float) and not math.isfinite(value)) for value in safe["array"])


@pytest.mark.parametrize("value", [0, 0.0, 125.5, "financial value"])
def test_json_safe_preserves_valid_values(value):
    assert json_safe(value) == value


def test_query_endpoint_returns_http_200_with_json_safe_analysis(monkeypatch):
    class NonFiniteQueryService:
        def answer_query(self, dataset_id, question, session_id=None, document_path=None):
            return {
                "answer": "Analysis completed with missing periods.",
                "result": {"nan": float("nan"), "inf": np.float64(float("inf")), "zero": 0.0},
                "evidence": [{"source": "local analysis", "findings": ["finite"]}],
                "intent": "trend_analysis",
                "route": "trend",
                "visualization": {
                    "chart_type": "line", "title": "Amount over time", "x": ["2022", "2023"],
                    "y": [10.0, float("nan")],
                },
            }

    monkeypatch.setattr(query_route, "query_service", NonFiniteQueryService())
    result = query_route.query_dataset(
        query_route.QueryRequest(dataset_id="local", question="Show trend"),
    )
    response = JSONResponse(content=jsonable_encoder(result.model_dump()))

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["result"]["result"] == {"nan": None, "inf": None, "zero": 0.0}
    assert payload["result"]["visualization"]["y"] == [10.0, None]
    assert payload["result"]["answer"] == "Analysis completed with missing periods."
