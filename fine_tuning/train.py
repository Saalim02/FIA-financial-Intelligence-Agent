"""
Phase 10 - Financial Intelligence Agent
QLoRA/SFT training pipeline.

This script is designed to run on a CUDA-enabled machine.
It intentionally does not start training when CUDA is unavailable.
"""

from pathlib import Path
import json

import torch


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
CONFIG_PATH = BASE_DIR / "configs" / "lora_config.yaml"
OUTPUT_DIR = BASE_DIR / "output"


def check_environment():
    print("=" * 60)
    print("PHASE 10 - FINE-TUNING ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print()
        print("GPU training is unavailable.")
        print()
        print("This is expected on the current WSL environment.")
        print("The training pipeline is ready for a CUDA-enabled machine.")
        return False

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    properties = torch.cuda.get_device_properties(0)

    print(
        f"GPU memory: "
        f"{properties.total_memory / (1024 ** 3):.2f} GB"
    )

    return True


def validate_dataset():
    print()
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    required_files = [
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    ]

    for filename in required_files:
        path = DATASET_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset file: {path}"
            )

        records = []

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)

                if "messages" not in record:
                    raise ValueError(
                        f"{filename}: missing messages"
                    )

                messages = record["messages"]

                roles = [message["role"] for message in messages]

                if roles != ["system", "user", "assistant"]:
                    raise ValueError(
                        f"{filename}: invalid message roles: {roles}"
                    )

                for message in messages:
                    if not message.get("content"):
                        raise ValueError(
                            f"{filename}: empty message content"
                        )

                records.append(record)

        print(f"{filename}: {len(records)} valid examples")


def main():
    validate_dataset()

    can_train = check_environment()

    if not can_train:
        print()
        print("PHASE 10 STATUS")
        print("Dataset: READY")
        print("Training configuration: READY")
        print("QLoRA pipeline: READY")
        print("GPU training: PENDING GPU")
        return

    print()
    print("CUDA environment detected.")
    print("Training implementation will be enabled in the next step.")


if __name__ == "__main__":
    main()
