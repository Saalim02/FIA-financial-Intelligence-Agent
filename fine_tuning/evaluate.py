"""Compare base and QLoRA Qwen on the shared behavioral test split.

Local metrics run without an external judge. Set RUN_RAGAS_LLM_EVAL=1 to also
use this repository's RAGAS evaluator (which makes paid OpenAI calls).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fine_tuning.inference import read_cases
from fine_tuning.model_loader import QwenConfig, load_qwen

TEST_PATH = Path(__file__).resolve().parent / "datasets" / "test.jsonl"


def evaluate_mode(mode: str, cases: list[dict]) -> dict:
    config = QwenConfig.from_env()
    config = QwenConfig(config.base_model, config.adapter_path, mode == "finetuned",
                        config.four_bit, config.device, mode == "finetuned")
    model, tokenizer, adapter_loaded = load_qwen(config)
    import torch
    rows, latency, token_counts = [], [], []
    memory_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    for case in cases:
        messages = case["messages"]
        prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(input_ids=prompt, max_new_tokens=192, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        latency.append(time.perf_counter() - start)
        generated = output[0, prompt.shape[-1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        token_counts.append(int(generated.numel()))
        reference = messages[-1]["content"]
        terms = set(re.findall(r"[a-z0-9.%]+", reference.lower()))
        answer_terms = set(re.findall(r"[a-z0-9.%]+", answer.lower()))
        overlap = len(terms & answer_terms) / max(1, len(terms))
        # An exact value absent from the user/evidence context is a measurable risk signal.
        context = " ".join(m["content"] for m in messages[:-1])
        values = re.findall(r"(?<![\w])\$?\d+(?:,\d{3})*(?:\.\d+)?%?", answer)
        unsupported_values = [v for v in values if v.replace(",", "") not in context.replace(",", "")]
        rows.append({"question": messages[1]["content"], "reference": reference, "answer": answer,
                     "reference_term_recall": round(overlap, 4),
                     "unsupported_numeric_values": unsupported_values,
                     "unsupported_numeric_count": len(unsupported_values)})
    memory_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    return {"mode": mode, "adapter_loaded": adapter_loaded,
            "mean_latency_seconds": round(sum(latency) / max(1, len(latency)), 4),
            "generated_tokens": token_counts,
            "gpu_memory_start_bytes": memory_start, "gpu_memory_peak_bytes": memory_peak,
            "mean_reference_term_recall": round(sum(x["reference_term_recall"] for x in rows) / max(1, len(rows)), 4),
            "unsupported_numeric_total": sum(x["unsupported_numeric_count"] for x in rows), "cases": rows}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "evaluation_results.json")
    args = parser.parse_args()
    cases = read_cases(args.test)
    if not cases:
        raise ValueError(f"No test cases found in {args.test}")
    results = {"base": evaluate_mode("base", cases), "finetuned": evaluate_mode("finetuned", cases)}
    if os.getenv("RUN_RAGAS_LLM_EVAL", "").lower() in {"1", "true", "yes"}:
        # Reuse the project's RAGAS dataset/sample construction and metric setup.
        from dotenv import load_dotenv
        load_dotenv()
        from evaluation.ragas_evaluator import build_ragas_dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms.base import LangchainLLMWrapper
        from ragas.metrics import AnswerCorrectness, AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
        from fine_tuning.inference import run
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("RUN_RAGAS_LLM_EVAL requires OPENAI_API_KEY")
        chat = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
        for mode in ("base", "finetuned"):
            generated, _ = run(mode, args.test)
            ragas_rows = []
            for case, prediction in zip(cases, generated):
                user = case["messages"][1]["content"]
                evidence_text = user.split("Evidence:", 1)[1].strip() if "Evidence:" in user else user
                ragas_rows.append({"question": prediction["question"], "answer": prediction["answer"],
                                   "reference": prediction["reference"],
                                   "evidence": [{"source": "behavioral test evidence", "findings": [evidence_text]}]})
            dataset = build_ragas_dataset(ragas_rows)
            results[mode]["ragas"] = str(evaluate(
                dataset,
                metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall(),
                         AnswerCorrectness(weights=[1.0, 0.0])],
                llm=chat, embeddings=embeddings,
            ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: {key: value for key, value in item.items() if key != "cases"} for k, item in results.items()}, indent=2))
    print(f"Detailed results: {args.output}")


if __name__ == "__main__":
    main()
