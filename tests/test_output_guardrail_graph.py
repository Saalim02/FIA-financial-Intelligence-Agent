from core.graph.graph import build_graph
import pandas as pd

class MockInputGuardrailClassifier:
    def classify(self, query):
        class Result:
            is_safe = True
            is_prompt_injection = False
            is_pii = False
            is_irrelevant = False
            reason = "Valid financial analysis question"

            def model_dump(self):
                return {
                    "is_safe": self.is_safe,
                    "is_prompt_injection": self.is_prompt_injection,
                    "is_pii": self.is_pii,
                    "is_irrelevant": self.is_irrelevant,
                    "reason": self.reason,
                }

        return Result()


class MockOutputGuardrailClassifier:
    def classify(self, question, answer, evidence):
        class Result:
            is_safe = True
            is_grounded = True
            contains_unsupported_claims = False
            contains_pii = False
            reason = "Answer is supported by supplied evidence."

            def model_dump(self):
                return {
                    "is_safe": self.is_safe,
                    "is_grounded": self.is_grounded,
                    "contains_unsupported_claims": self.contains_unsupported_claims,
                    "contains_pii": self.contains_pii,
                    "reason": self.reason,
                }

        return Result()

class MockUnsafeOutputGuardrailClassifier:
    def classify(self, question, answer, evidence):
        class Result:
            is_safe = False
            is_grounded = False
            contains_unsupported_claims = True
            contains_pii = False
            reason = "Answer contains unsupported financial claims."

            def model_dump(self):
                return {
                    "is_safe": self.is_safe,
                    "is_grounded": self.is_grounded,
                    "contains_unsupported_claims": self.contains_unsupported_claims,
                    "contains_pii": self.contains_pii,
                    "reason": self.reason,
                }

        return Result()

def test_unsafe_output_is_rejected():
    graph = build_graph()

    state = {
        "question": "What is the revenue?",
        "df": pd.DataFrame(
            {
                "revenue": [100000, 120000],
            }
        ),
        "guardrail_classifier": MockInputGuardrailClassifier(),
        "output_guardrail_classifier": MockUnsafeOutputGuardrailClassifier(),
    }

    result = graph.invoke(state)

    assert result["output_guardrail_safe"] is False
    assert "blocked by the output guardrail" in result["answer"].lower()

def test_output_guardrail_graph_path():
    graph = build_graph()

    state = {
        "question": "What is the revenue?",
        "guardrail_classifier": MockInputGuardrailClassifier(),
        "output_guardrail_classifier": MockOutputGuardrailClassifier(),
        "df": pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2025-01-01"]
                ),
                "revenue": [100000, 120000],
            }
        ),
    }   
