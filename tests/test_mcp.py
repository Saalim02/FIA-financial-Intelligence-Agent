from core.graph.graph import financial_graph
from core.graph.nodes import mcp_market_node
from core.guardrails.schema import GuardrailResult


class MockGuardrailClassifier:

    def classify(self, query):
        return GuardrailResult(
            is_safe=True,
            is_prompt_injection=False,
            is_pii=False,
            is_irrelevant=False,
            reason="Valid financial analysis question",
        )

def test_mcp_market_node():
    result = mcp_market_node({
        "mcp_symbol": "AAPL"
    })

    assert result["result"]["symbol"] == "AAPL"
    assert result["result"]["price"] == 150.25
    assert result["result"]["currency"] == "USD"
    assert result["result"]["source"] == "mock_market_provider"


def test_mcp_graph_routing():
    result = financial_graph.invoke({
        "question": "What is the current market price of AAPL?",
        "mcp_symbol": "AAPL",
        "guardrail_classifier": MockGuardrailClassifier(),
    })

    assert result["intent"] == "market_data"
    assert result["route"] == "mcp"
    assert result["mcp_result"]["symbol"] == "AAPL"
    assert result["mcp_result"]["price"] == 150.25
