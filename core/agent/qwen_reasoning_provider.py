"""Lazy, thread-safe local Qwen provider for FinancialReasoner."""
from __future__ import annotations

import json
import threading
import time

from fine_tuning.model_loader import QwenConfig, adapter_available, load_qwen, validate_adapter_config


class QwenReasoningProvider:
    def __init__(self, config: QwenConfig | None = None, loader=load_qwen):
        self.config = config or QwenConfig.from_env()
        self._adapter_available = validate_adapter_config(self.config) if self.config.load_adapter else False
        self._loader = loader
        self._model = None
        self._tokenizer = None
        self._adapter_loaded = False
        self._lock = threading.RLock()

    def _ensure_loaded(self):
        with self._lock:
            if self._model is None:
                self._model, self._tokenizer, self._adapter_loaded = self._loader(self.config)

    def reason(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_loaded()

        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        encoded = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        if hasattr(encoded, "items"):
            inputs = {
                key: value.to(self._model.device)
                for key, value in encoded.items()
            }
        else:
            inputs = {
                "input_ids": encoded.to(self._model.device)
            }


        with self._lock, torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[-1]

        answer = self._tokenizer.decode(
            output[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()

        if not answer:
            raise RuntimeError(
                "Qwen returned an empty financial reasoning response."
           )

        return answer

    def status(self) -> dict:
        return {
            "provider": "qwen", "base_model": self.config.base_model,
            "adapter_loaded": self._adapter_loaded,
            "adapter_available": self._adapter_available,
            "quantized": self.config.four_bit,
            "device": str(getattr(self._model, "device", self.config.device)),
            "mode": "finetuned" if self._adapter_loaded else "finetuned_pending" if self._adapter_available else "base_model",
        }
