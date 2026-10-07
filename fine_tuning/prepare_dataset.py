import json
import random
from pathlib import Path


OUTPUT_DIR = Path("fine_tuning/datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


EXAMPLES = [
    {
        "question": "What was the revenue growth?",
        "evidence": "Revenue increased from 100 million to 120 million.",
        "answer": (
            "Revenue increased by 20 million, representing 20% growth."
        ),
    },
    {
        "question": "What happened to operating profit?",
        "evidence": "Operating profit increased by 15 percent.",
        "answer": (
            "Operating profit increased by 15% according to the supplied evidence."
        ),
    },
    {
        "question": "Did revenue increase?",
        "evidence": "Revenue increased from 100 million to 120 million.",
        "answer": (
            "Yes. Revenue increased from 100 million to 120 million."
        ),
    },
    {
        "question": "What was the revenue change in absolute terms?",
        "evidence": "Revenue increased from 100 million to 120 million.",
        "answer": (
            "Revenue increased by 20 million."
        ),
    },
    {
        "question": "What is the percentage revenue growth?",
        "evidence": "Revenue increased from 100 million to 120 million.",
        "answer": (
            "Revenue growth was 20%."
        ),
    },
    {
        "question": "Can you determine next year's revenue?",
        "evidence": "The supplied evidence only contains historical revenue.",
        "answer": (
            "The supplied evidence does not provide enough information "
            "to determine next year's revenue."
        ),
    },
    {
        "question": "Does the evidence support a revenue increase?",
        "evidence": "Revenue increased from 100 million to 120 million.",
        "answer": (
            "Yes. The evidence directly supports a revenue increase "
            "from 100 million to 120 million."
        ),
    },
    {
        "question": "What financial conclusion can be drawn?",
        "evidence": "Revenue increased from 100 million to 120 million.",
        "answer": (
            "The supplied evidence shows that revenue increased by "
            "20 million, or 20%."
        ),
    },
    {
        "question": "What is the profit trend?",
        "evidence": "Profit increased in January, February, and March.",
        "answer": (
            "Profit shows an increasing trend across the supplied periods."
        ),
    },
    {
        "question": "Is there enough evidence to calculate revenue growth?",
        "evidence": "Revenue was 120 million.",
        "answer": (
            "No. A starting revenue value is required to calculate "
            "revenue growth."
        ),
    },
]


SYSTEM_PROMPT = """You are a financial intelligence assistant.

Follow these rules:
1. Use only the supplied financial evidence.
2. Never invent financial values, dates, sources, or facts.
3. Perform simple calculations only when the required values are present.
4. Clearly state when the evidence is insufficient.
5. Keep answers concise and business-oriented.
6. Separate evidence-based conclusions from unsupported speculation.
"""


def convert(example):
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{example['question']}\n\n"
                    f"Evidence:\n{example['evidence']}"
                ),
            },
            {
                "role": "assistant",
                "content": example["answer"],
            },
        ]
    }


def main():
    examples = [convert(example) for example in EXAMPLES]

    random.seed(42)
    random.shuffle(examples)

    total = len(examples)

    train_end = int(total * 0.7)
    validation_end = int(total * 0.85)

    train = examples[:train_end]
    validation = examples[train_end:validation_end]
    test = examples[validation_end:]

    files = {
        "train.jsonl": train,
        "validation.jsonl": validation,
        "test.jsonl": test,
    }

    for filename, records in files.items():
        path = OUTPUT_DIR / filename

        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")

        print(f"{path}: {len(records)} examples")


if __name__ == "__main__":
    main()
