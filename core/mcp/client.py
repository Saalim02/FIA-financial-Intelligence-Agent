import logging
import time

from core.mcp.tools import MarketDataTool
from core.observability.logging import get_logger, log_event


logger = get_logger(__name__)


class MCPClient:
    """Simple MCP-style client for external financial tools."""

    def __init__(self):
        self.market_data = MarketDataTool()

    def get_market_quote(self, symbol: str) -> dict:
        start = time.perf_counter()

        log_event(
            logger,
            logging.INFO,
            "mcp_tool_started",
            tool="market_data",
            symbol=symbol,
        )

        try:
            result = self.market_data.get_quote(symbol)

            latency_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            log_event(
                logger,
                logging.INFO,
                "mcp_tool_completed",
                tool="market_data",
                symbol=symbol,
                latency_ms=latency_ms,
            )

            return result

        except Exception as exc:
            latency_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            log_event(
                logger,
                logging.ERROR,
                "mcp_tool_failed",
                tool="market_data",
                symbol=symbol,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )

            raise
