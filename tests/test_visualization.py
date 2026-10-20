import pandas as pd
import pytest

from core.engines.financial_analyzer import FinancialAnalyzer
from core.engines.time_analyzer import TimeAnalyzer
from core.graph.graph import query_understanding


def test_trend_chart_detects_requested_date_and_numeric_metric_dynamically():
    frame = pd.DataFrame({
        "Imported Timestamp": ["2023-01-03", "2024-01-03", "2023-02-03", "2024-02-03"],
        "Posting Date": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01"],
        "Units": [2, 5, 3, 9],
        "Margin": [0.2, 0.3, 0.25, 0.4],
    })

    result = TimeAnalyzer().analyze(frame, entities=["Units"], question="Show monthly Units by Posting Date")

    assert result["visualization"]["chart_type"] == "line"
    assert result["visualization"]["metric"] == "Units"
    assert result["visualization"]["time_column"] == "Posting Date"
    assert result["visualization"]["granularity"] == "month"
    assert result["visualization"]["x"] == ["2024-01", "2024-02"]
    assert result["visualization"]["y"] == [7.0, 12.0]


def test_yearly_trend_chart_uses_detected_schema_and_backend_summary():
    frame = pd.DataFrame({
        "TransactionDate": ["2020-01-01", "2021-01-01", "2022-01-01"],
        "Operating Amount": [80.0, 72.0, 60.0],
    })

    result = TimeAnalyzer().analyze(frame, entities=["Operating Amount"], question="Plot operating amount by year")
    chart = result["visualization"]

    assert chart["time_column"] == "TransactionDate"
    assert chart["metric"] == "Operating Amount"
    assert chart["x"] == ["2020", "2021", "2022"]
    assert chart["y"] == [80.0, 72.0, 60.0]
    assert chart["summary"] == {
        "direction": "decreasing", "start": 80.0, "end": 60.0, "change_percent": -25.0,
    }


def test_group_by_chart_uses_arbitrary_dataset_dimension_and_metric():
    frame = pd.DataFrame({
        "Territory": ["North", "South", "North", "West"],
        "Net Amount": [100.0, 80.0, 50.0, 40.0],
    })

    chart = FinancialAnalyzer().chart_data(
        frame, {"metrics": ["Net Amount"]}, "Show Net Amount by Territory",
    )

    assert chart is not None
    assert chart["chart_type"] == "bar"
    assert chart["dimension"] == "Territory"
    assert chart["metric"] == "Net Amount"
    assert chart["x"] == ["North", "South", "West"]
    assert chart["y"] == [150.0, 80.0, 40.0]


def test_normal_total_query_does_not_produce_visualization():
    frame = pd.DataFrame({"Event Date": ["2024-01-01", "2024-02-01"], "Units": [4, 9]})

    assert FinancialAnalyzer().chart_data(frame, {"metrics": ["Units"]}, "What is the total Units?") is None


@pytest.mark.parametrize("question", ["Show monthly units", "Show yearly units"])
def test_monthly_and_yearly_wording_routes_to_trend_analysis(question):
    frame = pd.DataFrame({
        "Recorded At": ["2023-01-01", "2024-01-01"],
        "Units": [2, 5],
    })

    route = query_understanding({"question": question, "df": frame})

    assert route["route"] == "trend"
    assert route["intent"] == "trend_analysis"


def test_schema_resolved_trend_visualization():
    frame = pd.DataFrame({
        "Order Date": [
            "2023-01-01",
            "2023-02-01",
            "2023-03-01",
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
        ],
        "Sales": [
            100.0,
            120.0,
            140.0,
            160.0,
            180.0,
            200.0,
        ],
    })

    resolution = query_understanding({
        "question": "Show me the Sales trend",
        "df": frame,
    })

    analysis = TimeAnalyzer().analyze(
        frame,
        entities=resolution["entities"],
        question="Show me the Sales trend",
    )

    chart = analysis["visualization"]

    assert resolution["route"] == "trend"
    assert chart["metric"] == "Sales"
    assert chart["time_column"] == "Order Date"
    assert chart["x"] and len(chart["x"]) == len(chart["y"])
    assert chart["summary"]["direction"] in {
        "increasing",
        "decreasing",
        "stable",
    }


def test_trend_preserves_missing_period_as_null_without_inventing_zero():
    frame = pd.DataFrame({
        "Recorded At": ["2022-01-01", "2023-01-01", "2024-01-01"],
        "Operating Amount": [10.0, float("nan"), 30.0],
    })

    chart = TimeAnalyzer().analyze(
        frame, entities=["Operating Amount"], question="Show yearly Operating Amount trend",
    )["visualization"]

    assert chart["x"] == ["2022", "2023", "2024"]
    assert chart["y"] == [10.0, None, 30.0]
    assert chart["summary"]["start"] == 10.0
    assert chart["summary"]["end"] == 30.0


def test_group_visualization_converts_nonfinite_and_preserves_zero_and_values():
    frame = pd.DataFrame({
        "Region": ["Zero"] * 4 + ["Missing"] * 4 + ["Finite"] * 4,
        "Amount": [0.0] * 4 + [float("nan")] * 4 + [125.5, 10.0, 5.0, 0.0],
    })

    chart = FinancialAnalyzer().chart_data(frame, {"metrics": ["Amount"]}, "Show Amount by Region")

    assert chart["x"] == ["Finite", "Zero", "Missing"]
    assert chart["y"] == [140.5, 0.0, None]
