from __future__ import annotations

import re


FIA_GREETING = "Hello, this is FIA. How can I help you?"

FIA_IDENTITY = (
    "FIA refers to Financial Intelligence Agent. "
    "How can I help you?"
)

FIA_SCOPE = (
    "I'm FIA, the Financial Intelligence Agent. "
    "I can help with your uploaded financial data, financial documents, "
    "market data, and my AI capabilities."
)

OUT_OF_SCOPE = (
    "I'm FIA, the Financial Intelligence Agent. "
    "I can help with your uploaded financial data, financial documents, "
    "market data, and my AI capabilities. "
    "That question is outside my scope. "
    "How can I help you with your financial data?"
)


class FIAController:
    """Controls FIA-specific identity, scope and deterministic responses."""

    IDENTITY_TERMS = {
        "what is fia",
        "what does fia mean",
        "what is meant by fia",
        "meaning of fia",
        "what do you mean by fia",
    }

    CAPABILITIES = {
        "model": "FIA uses the Qwen 2.5 Instruct model with a QLoRA adapter for financial reasoning.",
        "qlora": "Yes. FIA uses a QLoRA fine-tuned adapter with its Qwen model for financial reasoning.",
        "rag": "RAG means retrieval-augmented generation. FIA retrieves relevant financial document passages to ground its answers.",
        "mcp": "MCP connects FIA to external tools and market data sources when they are configured.",
        "langgraph": "LangGraph orchestrates FIA's guarded workflow, including financial analysis, document retrieval, and tool use.",
        "how": "FIA sends financial questions through input guardrails and a LangGraph workflow, then returns evidence-backed answers checked by output guardrails.",
    }

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())

    def handle_identity(self, text: str) -> str | None:
        normalized = self.normalize(text)

        if normalized in self.IDENTITY_TERMS:
            return FIA_IDENTITY

        return None

    def handle_capability(self, text: str) -> str | None:
        normalized = self.normalize(text)
        if "qlora" in normalized or "fine-tun" in normalized or "fine tun" in normalized:
            return self.CAPABILITIES["qlora"]
        if "rag" in normalized or "retrieval augmented" in normalized:
            return self.CAPABILITIES["rag"]
        if "mcp" in normalized:
            return self.CAPABILITIES["mcp"]
        if "langgraph" in normalized:
            return self.CAPABILITIES["langgraph"]
        if "model" in normalized or "using" in normalized and "you" in normalized:
            return self.CAPABILITIES["model"]
        if "how does fia work" in normalized or "how do you work" in normalized:
            return self.CAPABILITIES["how"]
        return None

    def out_of_scope_response(self) -> str:
        return OUT_OF_SCOPE

    @staticmethod
    def missing_dataset_response() -> str:
        return "Please upload or select a financial dataset first."
