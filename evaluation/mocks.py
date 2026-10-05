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
