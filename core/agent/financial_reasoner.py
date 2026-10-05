import json

from pydantic import BaseModel

from core.agent.llm_interface import LLMInterface


class FinancialReasoningResult(BaseModel):
    answer: str


class FinancialReasoner:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def reason(self, question: str, evidence: list[dict]) -> str:
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
                "evidence": evidence,
            },
            ensure_ascii=False,
            default=str,
        )

        result = self.llm.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=FinancialReasoningResult,
        )

        return result.answer
