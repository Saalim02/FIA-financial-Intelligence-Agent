"""Run the same behavioral test cases through base or adapter-backed Qwen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fine_tuning.model_loader import QwenConfig, load_qwen


def read_cases(path: Path):
    cases = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(case.get("messages"), list) or len(case["messages"]) != 3:
                raise ValueError(f"Invalid messages at {path}:{line_no}")
            cases.append(case)
    return cases


def run(mode: str, test_path: Path):
    config = QwenConfig.from_env()
    config = QwenConfig(config.base_model, config.adapter_path, mode == "finetuned",
                        config.four_bit, config.device, mode == "finetuned")
    model, tokenizer, adapter_loaded = load_qwen(config)
    import torch
    results = []
    for case in read_cases(test_path):
        msgs = case["messages"]
        inputs = tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, max_new_tokens=192, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        answer = tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=True).strip()
        results.append({"question": msgs[1]["content"], "reference": msgs[2]["content"], "answer": answer})
    return results, adapter_loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("base", "finetuned"), default="finetuned")
    parser.add_argument("--test", type=Path, default=Path(__file__).resolve().parent / "datasets" / "test.jsonl")
    args = parser.parse_args()
    results, adapter_loaded = run(args.mode, args.test)
    print(json.dumps({"mode": args.mode, "adapter_loaded": adapter_loaded, "results": results}, indent=2))


if __name__ == "__main__":
    main()
