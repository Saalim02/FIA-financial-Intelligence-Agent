from core.guardrails.output_classifier import OutputGuardrailClassifier
from core.guardrails.output_schema import OutputGuardrailResult


class MockLLM:
    def generate_structured(
        self,
        system_prompt,
        user_prompt,
        response_model,
    ):
        return OutputGuardrailResult(
            is_safe=True,
            is_grounded=True,
            contains_unsupported_claims=False,
            contains_pii=False,
            reason="Answer is supported by the supplied financial evidence.",
        )


def test_output_guardrail_classifier():
    classifier = OutputGuardrailClassifier(MockLLM())

    result = classifier.classify(
        question="What was the revenue growth?",
        answer="Revenue increased from 100 million to 120 million.",
        evidence=[
            {
                "source": "financial_report",
                "findings": "Revenue increased from 100 million to 120 million."
            }
        ],
    )

    assert isinstance(result, OutputGuardrailResult)
    assert result.is_safe is True
    assert result.is_grounded is True
    assert result.contains_unsupported_claims is False
    assert result.contains_pii is False

