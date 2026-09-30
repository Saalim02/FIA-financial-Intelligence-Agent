from core.agent.llm_interface import LLMInterface
from core.guardrails.output_schema import OutputGuardrailResult


OUTPUT_GUARDRAIL_SYSTEM_PROMPT = """
You are an output security and factuality classifier for a Financial Intelligence Agent.

Evaluate the generated answer against the user's question and the supplied evidence.

Classify using these rules:

1. is_safe:
   True only when the answer is safe and appropriate to return.

2. is_grounded:
   True when the important claims in the answer are supported by the supplied evidence.

3. contains_unsupported_claims:
   True when the answer makes factual claims that are not supported by the supplied evidence.

4. contains_pii:
   True when the answer exposes sensitive personally identifiable information,
   authentication secrets, financial account credentials, or similar sensitive data.

Do not assume information is supported simply because it sounds plausible.

A response can be concise and still be grounded.

Return only the structured classification.
"""


class OutputGuardrailClassifier:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def classify(
        self,
        question: str,
        answer: str,
        evidence: list[dict],
    ) -> OutputGuardrailResult:

        user_prompt = f"""
User question:
{question}

Generated answer:
{answer}

Evidence:
{evidence}
"""

        return self.llm.generate_structured(
            system_prompt=OUTPUT_GUARDRAIL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=OutputGuardrailResult,
        )
