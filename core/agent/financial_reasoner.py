import json

from pydantic import BaseModel

from core.agent.llm_interface import LLMInterface


class FinancialReasoningResult(BaseModel):
    answer: str


class FinancialReasoner:
    """Evidence-grounded reasoning over a swappable model provider."""
    def __init__(self, llm):
        self.llm = llm

    def reason(
        self,
        question: str,
        evidence: list[dict],
        conversation_history: list[dict] | None = None,
    ) -> str:
        system_prompt = """
You are a financial reasoning assistant.

Answer the user's question using ONLY the supplied evidence.

You may perform simple arithmetic or derive values from the evidence when
the calculation is directly supported by the evidence.

For example, if evidence says revenue increased from 100 to 120,
you may calculate:

(120 - 100) / 100 * 100 = 20%

Do not invent facts, sources, dates, or financial values.

Return a concise business-oriented answer.
Include the relevant calculation or interpretation when useful.
"""

        user_prompt = json.dumps(
            {
                "question": question,
                "conversation_history": conversation_history or [],
                "evidence": evidence,
            },
            ensure_ascii=False,
            default=str,
        )

        if hasattr(self.llm, "reason"):
            return self.llm.reason(system_prompt, user_prompt)
        result = self.llm.generate_structured(system_prompt, user_prompt, FinancialReasoningResult)
        return result.answer
