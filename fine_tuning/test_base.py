from pathlib import Path
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = Path(__file__).resolve().parent / "output"
TEST_PATH = Path(__file__).resolve().parent / "datasets" / "test.jsonl"


def load_test_cases():
    cases = []

    with TEST_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            cases.append(json.loads(line))

    return cases


def main():
    print("=" * 70)
    print("BASE QWEN - TEST INFERENCE")
    print("=" * 70)

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Tokenizer loaded")

    print("\nLoading base model in 4-bit...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("✓ Base model loaded")
    
    model = base_model
    model.eval()


    test_cases = load_test_cases()

    print(f"\nTest cases: {len(test_cases)}")

    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    for index, case in enumerate(test_cases, start=1):

        messages = case["messages"]

        prompt_messages = [
            message
            for message in messages
            if message["role"] != "assistant"
        ]

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        answer = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        expected = messages[-1]["content"]

        print(f"\nTest {index}")
        print("-" * 70)

        print("Question:")
        print(messages[1]["content"])

        print("\nExpected:")
        print(expected)

        print("\nBase Qwen:")
        print(answer)

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
