import pandas as pd

from core.graph.graph import query_understanding


def test_query_understanding_resolves_multiple_financial_metrics():
    df = pd.DataFrame(
        {
            "Sales": [1000.0, 2000.0, 3000.0],
            "Profit": [100.0, 200.0, 300.0],
            "Region": ["North", "South", "East"],
        }
    )

    result = query_understanding(
        {
            "question": "What was the total sales and profit in the dataset?",
            "df": df,
        }
    )

    assert result["intent"] == "financial_analysis"
    assert result["route"] == "financial"

    assert result["metrics"]["metrics"] == [
        "Sales",
        "Profit",
    ]
