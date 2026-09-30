from core.mcp.tools import MarketDataTool


class MCPClient:
    """Simple MCP-style client for external financial tools."""

    def __init__(self):
        self.market_data = MarketDataTool()

    def get_market_quote(self, symbol: str) -> dict:
        return self.market_data.get_quote(symbol)
