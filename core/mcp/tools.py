from datetime import datetime, timezone


class MarketDataTool:
    """Provide external market data through the MCP layer."""

    name = "market_data"

    def get_quote(self, symbol: str) -> dict:
        symbol = symbol.strip().upper()

        if not symbol:
            raise ValueError("Symbol cannot be empty.")

        # Controlled mock response for architecture testing.
        return {
            "symbol": symbol,
            "price": 150.25,
            "currency": "USD",
            "change_percent": 1.42,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "mock_market_provider",
        }
