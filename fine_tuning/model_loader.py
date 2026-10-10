"""Lazy Qwen 2.5 model loading helpers shared by API and CLI inference."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_PATH = Path(__file__).resolve().parent / "output"


class ModelConfigurationError(RuntimeError):
    """Invalid model or adapter configuration."""


@dataclass(frozen=True)
class QwenConfig:
    base_model: str = DEFAULT_BASE_MODEL
    adapter_path: Path = DEFAULT_ADAPTER_PATH
    load_adapter: bool = True
    four_bit: bool = True
    device: str = "auto"
    require_adapter: bool = False

    @classmethod
    def from_env(cls) -> "QwenConfig":
        def flag(name: str, default: bool) -> bool:
            value = os.getenv(name)
            return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}
        return cls(
            base_model=os.getenv("QWEN_BASE_MODEL", DEFAULT_BASE_MODEL),
            adapter_path=Path(os.getenv("QWEN_ADAPTER_PATH", str(DEFAULT_ADAPTER_PATH))).expanduser(),
            load_adapter=flag("QWEN_LOAD_ADAPTER", True),
            four_bit=flag("QWEN_4BIT", True),
            device=os.getenv("QWEN_DEVICE", "auto"),
            require_adapter=flag("QWEN_REQUIRE_ADAPTER", False),
        )


def adapter_available(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").is_file() and any(
        (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def validate_adapter_config(config: QwenConfig) -> bool:
    available = config.load_adapter and adapter_available(config.adapter_path)
    if config.load_adapter and not available:
        if config.require_adapter:
            raise ModelConfigurationError(
                f"QLoRA adapter is required but no valid adapter exists at {config.adapter_path}."
            )
        import logging
        logging.getLogger(__name__).warning("QLoRA adapter not found; using base Qwen model.")
    return available


def load_qwen(config: QwenConfig | None = None):
    """Return (model, tokenizer, adapter_loaded); imports heavyweight deps on demand."""
    config = config or QwenConfig.from_env()
    use_adapter = validate_adapter_config(config)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError as exc:
        raise ModelConfigurationError(
            "Qwen inference requires torch, transformers, peft and (for 4-bit) bitsandbytes; install requirements.txt."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"device_map": config.device, "torch_dtype": torch.float16}
    if config.four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **kwargs)
    model.config.use_cache = True
    if use_adapter:
        model = PeftModel.from_pretrained(model, str(config.adapter_path))
    model.eval()
    return model, tokenizer, use_adapter


def load_base_qwen(config: QwenConfig | None = None):
    config = config or QwenConfig.from_env()
    return load_qwen(QwenConfig(config.base_model, config.adapter_path, False, config.four_bit, config.device, False))


def load_finetuned_qwen(config: QwenConfig | None = None):
    config = config or QwenConfig.from_env()
    return load_qwen(QwenConfig(config.base_model, config.adapter_path, True, config.four_bit, config.device, True))
