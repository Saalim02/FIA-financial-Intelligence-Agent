import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import (
    EvaluationDataset,
    SingleTurnSample,
    evaluate,
)

from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

from evaluation.runner import AgentEvaluator


def extract_contexts(evidence):

    contexts = []

    for item in evidence:

        source = item.get("source", "unknown")
        findings = item.get("findings", [])

        if isinstance(findings, list):

            text = (
                f"{source} analysis findings: "
                + " ".join(str(x) for x in findings)
            )

            contexts.append(text)

        elif isinstance(findings, dict):

            contexts.append(
                f"{source} result: {findings}"
            )

        else:
            contexts.append(
                f"{source} result: {str(findings)}"
            )

    return contexts



def build_ragas_dataset(results):

    samples = []

    for item in results:

        if not item.get("reference"):
            continue

        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=item["answer"],
                retrieved_contexts=extract_contexts(item["evidence"]),
                reference=item["reference"],
            )
        )

    return EvaluationDataset(
        samples=samples
    )


def run_ragas():

    load_dotenv()

    chat = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
    )

    llm = LangchainLLMWrapper(chat)


    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    )


    results = [r for r in AgentEvaluator().run() if r["id"] == "rag_001"]

    dataset = build_ragas_dataset(results)


    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(weights=[1.0, 0.0]),
    ]


    evaluation_result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=True
    )


    print(evaluation_result)



if __name__ == "__main__":
    run_ragas()
