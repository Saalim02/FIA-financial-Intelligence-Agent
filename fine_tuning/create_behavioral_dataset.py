import json
import random
from pathlib import Path


OUTPUT_DIR = Path(__file__).resolve().parent / "datasets"

random.seed(42)


SYSTEM_PROMPT = """You are a financial intelligence assistant.

Follow these rules:
1. Use only the supplied financial evidence.
2. Never invent financial values, dates, sources, or facts.
3. Perform simple calculations only when the required values are present.
4. Clearly state when the evidence is insufficient.
5. Keep answers concise and business-oriented.
6. Separate evidence-based conclusions from unsupported speculation.
"""


def example(question, evidence, answer):
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Evidence:\n{evidence}"
                ),
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]
    }


def build_examples():
    examples = []

    # ---------------------------------------------------------
    # Revenue growth
    # ---------------------------------------------------------
    for old, new in [
        (100, 120),
        (150, 180),
        (200, 250),
        (80, 100),
        (300, 330),
        (500, 550),
        (125, 150),
        (240, 300),
        (400, 360),
        (600, 540),
    ]:
        change = new - old
        percentage = (change / old) * 100

        direction = "increased" if change > 0 else "decreased"

        examples.append(
            example(
                "What was the revenue growth?",
                f"Revenue changed from {old} million to {new} million.",
                (
                    f"Revenue {direction} by {abs(change):g} million, "
                    f"representing a {abs(percentage):g}% "
                    f"{'increase' if change > 0 else 'decrease'}."
                ),
            )
        )

    # ---------------------------------------------------------
    # Profit percentage
    # ---------------------------------------------------------
    for profit_change in [5, 8, 10, 12, 15, 18, 20]:
        examples.append(
            example(
                "What happened to operating profit?",
                f"Operating profit increased by {profit_change} percent.",
                (
                    f"Operating profit increased by "
                    f"{profit_change}% according to the supplied evidence."
                ),
            )
        )

    # ---------------------------------------------------------
    # Absolute change
    # ---------------------------------------------------------
    for old, new in [
        (100, 125),
        (250, 300),
        (500, 450),
        (800, 1000),
        (1200, 1100),
    ]:
        change = new - old

        examples.append(
            example(
                "What was the absolute change in revenue?",
                f"Revenue changed from {old} million to {new} million.",
                f"The absolute change in revenue was {abs(change):g} million.",
            )
        )

    # ---------------------------------------------------------
    # Margin
    # ---------------------------------------------------------
    for revenue, profit in [
        (100, 20),
        (200, 40),
        (500, 75),
        (1000, 150),
        (800, 160),
    ]:
        margin = (profit / revenue) * 100

        examples.append(
            example(
                "What was the profit margin?",
                (
                    f"Revenue was {revenue} million and "
                    f"profit was {profit} million."
                ),
                f"The profit margin was {margin:g}%.",
            )
        )

    # ---------------------------------------------------------
    # Trend
    # ---------------------------------------------------------
    trend_cases = [
        (
            "Revenue increased from 100 million to 120 million to 140 million.",
            "Revenue shows an increasing trend over the supplied periods.",
        ),
        (
            "Profit increased from 20 million to 25 million to 30 million.",
            "Profit shows an increasing trend over the supplied periods.",
        ),
        (
            "Revenue decreased from 300 million to 270 million to 240 million.",
            "Revenue shows a decreasing trend over the supplied periods.",
        ),
        (
            "Revenue was 100 million, 102 million, and 101 million.",
            "Revenue appears relatively stable over the supplied periods.",
        ),
    ]

    for evidence, answer in trend_cases:
        examples.append(
            example(
                "What is the financial trend?",
                evidence,
                answer,
            )
        )

    # ---------------------------------------------------------
    # Insufficient evidence
    # ---------------------------------------------------------
    insufficient_cases = [
        (
            "What will revenue be next year?",
            "Revenue was 100 million this year.",
            "The supplied evidence is insufficient to determine next year's revenue.",
        ),
        (
            "What is the profit margin?",
            "Revenue was 200 million.",
            "The supplied evidence is insufficient to calculate the profit margin because profit is not provided.",
        ),
        (
            "What was the revenue growth percentage?",
            "Revenue was 150 million.",
            "The supplied evidence is insufficient to calculate revenue growth because a comparison value is not provided.",
        ),
        (
            "Why did profit decrease?",
            "Profit decreased by 10 percent.",
            "The supplied evidence shows that profit decreased by 10%, but it does not explain why.",
        ),
    ]

    for question, evidence, answer in insufficient_cases:
        examples.append(
            example(question, evidence, answer)
        )

    # ---------------------------------------------------------
    # Anomaly
    # ---------------------------------------------------------
    anomaly_cases = [
        (
            "Are there any unusual revenue values?",
            "Revenue values were 100, 105, 102, 300, and 104 million.",
            "The 300 million value appears unusually high compared with the other supplied values.",
        ),
        (
            "Is there an unusual profit value?",
            "Profit values were 20, 22, 21, 19, and 80 million.",
            "The 80 million value appears unusually high compared with the other supplied values.",
        ),
    ]

    for question, evidence, answer in anomaly_cases:
        examples.append(
            example(question, evidence, answer)
        )

    # ---------------------------------------------------------
    # Evidence grounding
    # ---------------------------------------------------------
    grounding_cases = [
        (
            "Did revenue increase?",
            "Revenue increased from 100 million to 120 million.",
            "Yes. The supplied evidence shows that revenue increased from 100 million to 120 million.",
        ),
        (
            "Did operating profit improve?",
            "Operating profit increased by 15 percent.",
            "Yes. Operating profit increased by 15% according to the supplied evidence.",
        ),
    ]

    for question, evidence, answer in grounding_cases:
        examples.append(
            example(question, evidence, answer)
        )

    # ---------------------------------------------------------
    # Unsupported speculation
    # ---------------------------------------------------------
    speculation_cases = [
        (
            "Why did revenue increase?",
            "Revenue increased from 100 million to 120 million.",
            "The supplied evidence confirms that revenue increased, but it does not explain why.",
        ),
        (
            "Will revenue continue increasing?",
            "Revenue increased from 100 million to 120 million.",
            "The supplied evidence does not establish whether revenue will continue increasing.",
        ),
    ]

    for question, evidence, answer in speculation_cases:
        examples.append(
            example(question, evidence, answer)
        )

    return examples


def write_jsonl(path, examples):
    with path.open("w", encoding="utf-8") as file:
        for item in examples:
            file.write(json.dumps(item) + "\n")


def main():
    examples = build_examples()

    random.shuffle(examples)

    train_size = int(len(examples) * 0.8)
    validation_size = int(len(examples) * 0.1)

    train = examples[:train_size]
    validation = examples[
        train_size:train_size + validation_size
    ]
    test = examples[
        train_size + validation_size:
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(
        OUTPUT_DIR / "behavioral_train.jsonl",
        train,
    )

    write_jsonl(
        OUTPUT_DIR / "behavioral_validation.jsonl",
        validation,
    )

    write_jsonl(
        OUTPUT_DIR / "behavioral_test.jsonl",
        test,
    )

    print("=" * 70)
    print("BEHAVIORAL DATASET CREATED")
    print("=" * 70)
    print(f"Total examples: {len(examples)}")
    print(f"Training:        {len(train)}")
    print(f"Validation:      {len(validation)}")
    print(f"Test:            {len(test)}")
    print()
    print("Dataset files:")
    print(OUTPUT_DIR / "behavioral_train.jsonl")
    print(OUTPUT_DIR / "behavioral_validation.jsonl")
    print(OUTPUT_DIR / "behavioral_test.jsonl")


if __name__ == "__main__":
    main()
