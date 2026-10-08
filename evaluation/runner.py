import sys
from pathlib import Path
import argparse
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
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


def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV or Excel datasets based on file extension."""

    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    suffix = dataset_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(dataset_path)

    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(dataset_path)

    else:
        raise ValueError(
            f"Unsupported dataset format: {suffix}. "
            "Supported formats: .csv, .xls, .xlsx"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    print(
        f"\nDATASET LOADED: {dataset_path.name}"
    )
    print(
        f"Rows: {len(df)} | Columns: {len(df.columns)}"
    )

    return df


class AgentEvaluator:

    def __init__(self, dataset_path: str):
        self.graph = build_graph()

        # Load the actual dataset supplied by the user.
        self.df = load_dataset(dataset_path)

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


def main():

    parser = argparse.ArgumentParser(
        description="Run Financial Intelligence Agent evaluation."
    )

    parser.add_argument(
        "--dataset",
        default="test_financial_data.xls",
        help="Path to CSV/XLS/XLSX dataset.",
    )

    args = parser.parse_args()

    evaluator = AgentEvaluator(args.dataset)

    results = evaluator.run()

    for result in results:
        print("\n---")
        print(result)


if __name__ == "__main__":
    main()
