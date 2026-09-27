import pandas as pd

from core.engines.correlation import CorrelationEngine


def test_correlation_preserves_negative_direction():

    df = pd.DataFrame({
        "revenue": [1, 2, 3, 4, 5],
        "cost": [5, 4, 3, 2, 1]
    })

    engine = CorrelationEngine()
    result = engine.run(df)

    assert result["strongest_pair"] == ("revenue", "cost")
    assert result["correlation"] == -1.0
    assert result["direction"] == "negative"

