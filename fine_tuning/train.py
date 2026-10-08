"""
Phase 10 - Financial Intelligence Agent
QLoRA + SFT training pipeline.

Target:
    Qwen/Qwen2.5-1.5B-Instruct

Hardware:
    RTX 3050 Laptop GPU, 4 GB VRAM
"""

from pathlib import Path
import json

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
CONFIG_PATH = BASE_DIR / "configs" / "lora_config.yaml"
OUTPUT_DIR = BASE_DIR / "output_behavioral"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_dataset():
    required_files = [
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    ]

    for filename in required_files:
        path = DATASET_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}")

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                record = json.loads(line)

                if "messages" not in record:
                    raise ValueError(
                        f"{filename}:{line_number} missing messages"
                    )

                roles = [
                    message["role"]
                    for message in record["messages"]
                ]

                if roles != ["system", "user", "assistant"]:
                    raise ValueError(
                        f"{filename}:{line_number} invalid roles: {roles}"
                    )

                for message in record["messages"]:
                    if not message.get("content"):
                        raise ValueError(
                            f"{filename}:{line_number} empty message"
                        )

        print(f"✓ {filename} validated")


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this QLoRA training run."
        )

    print()
    print("GPU:", torch.cuda.get_device_name(0))

    properties = torch.cuda.get_device_properties(0)

    print(
        f"VRAM: "
        f"{properties.total_memory / (1024 ** 3):.2f} GB"
    )


def format_dataset(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def main():
    print("=" * 70)
    print("PHASE 10 - QLoRA/SFT TRAINING")
    print("=" * 70)

    config = load_config()

    validate_dataset()
    check_gpu()

    model_name = config["model"]["name"]
    max_seq_length = config["model"]["max_seq_length"]

    print()
    print("Model:", model_name)
    print("Max sequence length:", max_seq_length)

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Tokenizer loaded")

    print()
    print("Loading 4-bit quantized model...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    print("✓ Model loaded")

    print(
        f"GPU memory allocated: "
        f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB"
    )

    print()
    print("Preparing LoRA...")

    lora_cfg = config["lora"]

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("✓ LoRA configuration ready")

    print()
    print("Loading training dataset...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATASET_DIR / "train.jsonl"),
            "validation": str(DATASET_DIR / "validation.jsonl"),
        },
    )

    dataset = dataset.map(
        lambda example: format_dataset(example, tokenizer),
        remove_columns=["messages"],
    )

    print(
        f"Train examples: {len(dataset['train'])}"
    )

    print(
        f"Validation examples: {len(dataset['validation'])}"
    )

    print("Dataset columns:", dataset["train"].column_names)

    training_cfg = config["training"]

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=training_cfg["learning_rate"],
        logging_steps=training_cfg["logging_steps"],
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        max_grad_norm=0.3,
        warmup_ratio=0.05,

        # TRL 0.12.2 uses max_seq_length
        max_seq_length=max_seq_length,

        # Dataset contains a "text" column
        dataset_text_field="text",
    )

    print()
    print("Training configuration:")
    print("  Batch size: 1")
    print("  Gradient accumulation: 8")
    print("  Sequence length:", max_seq_length)
    print("  FP16: enabled")
    print("  Gradient checkpointing: enabled")
    print("  8-bit optimizer: enabled")
    print("  Epochs:", training_cfg["num_train_epochs"])

    print()
    print("=" * 70)
    print("STARTING QLoRA TRAINING")
    print("=" * 70)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print()
    print("Adapter saved to:")
    print(OUTPUT_DIR)

    print(
        f"Final GPU memory allocated: "
        f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB"
    )


if __name__ == "__main__":
    main()