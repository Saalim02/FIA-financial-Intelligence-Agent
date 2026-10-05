import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import os

from dotenv import load_dotenv

load_dotenv()

from core.agent.financial_reasoner import FinancialReasoner
from core.agent.llm_interface import LLMInterface
from core.graph.graph import build_graph
from evaluation.dataset import EVALUATION_DATASET
from evaluation.mocks import (
    MockInputGuardrailClassifier,
    MockOutputGuardrailClassifier,
)


class AgentEvaluator:

    def __init__(self):
        self.graph = build_graph()
        self.df = pd.read_csv("test_financial_data.csv")

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for financial reasoning evaluation."
            )

        self.reasoner = FinancialReasoner(
            LLMInterface(
                api_key=api_key,
                model="gpt-4o-mini",
            )
        )

    def build_state(self, item):

        state = {
            "question": item["question"],
            "guardrail_classifier": MockInputGuardrailClassifier(),
            "output_guardrail_classifier": MockOutputGuardrailClassifier(),
            "reasoning_llm": self.reasoner,
        }

        route = item["expected_route"]

        if route == "financial":
            state.update({
                "df": self.df,
                "metrics": {
                    "revenue": "revenue",
                    "profit": "profit",
                },
            })

        elif route == "trend":
            state.update({
                "df": self.df,
                "entities": [
                    "revenue"
                ],
            })

        elif route == "anomaly":
            state.update({
                "df": self.df,
                "entities": [
                    "revenue"
                ],
            })

        elif route == "rag":
            state.update({
                "document_path":
                    "data/documents/test_financial_report.pdf"
            })

        elif route == "mcp":
            state.update({
                "mcp_symbol": "AAPL"
            })

        return state


    def run(self):

        results = []

        for item in EVALUATION_DATASET:

            state = self.build_state(item)

            output = self.graph.invoke(state)

            results.append({
                "id": item["id"],
                "question": item["question"],
                "route": output.get("route"),
                "answer": output.get("answer"),
                "evidence": output.get("evidence", []),
                "error": output.get("error"),
                "reference": item.get("reference"),
            })

        return results


if __name__ == "__main__":

    evaluator = AgentEvaluator()

    results = evaluator.run()

    for result in results:
        print("\n---")
        print(result)
