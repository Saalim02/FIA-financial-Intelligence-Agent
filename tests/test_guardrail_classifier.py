from core.guardrails.classifier import GuardrailClassifier
from core.guardrails.schema import GuardrailResult


class MockLLM:

    def generate_structured(
        self,
        system_prompt,
        user_prompt,
        response_model,
    ):
        return GuardrailResult(
            is_safe=True,
            is_prompt_injection=False,
            is_pii=False,
            is_irrelevant=False,
            reason="Valid financial analysis question",
        )


def test_guardrail_classifier():

    classifier = GuardrailClassifier(MockLLM())

    result = classifier.classify(
        "What was the revenue growth in 2025?"
    )

    assert isinstance(result, GuardrailResult)
    assert result.is_safe is True
    assert result.is_prompt_injection is False
    assert result.is_pii is False
    assert result.is_irrelevant is False
