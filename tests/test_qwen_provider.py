import json

import pytest

from fine_tuning.model_loader import ModelConfigurationError, QwenConfig, adapter_available, validate_adapter_config
from core.agent.qwen_reasoning_provider import QwenReasoningProvider


def test_adapter_detection_requires_peft_config_and_weights(tmp_path):
    assert not adapter_available(tmp_path)
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert not adapter_available(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    assert adapter_available(tmp_path)


def test_missing_optional_adapter_falls_back_to_base(tmp_path, caplog):
    config = QwenConfig(adapter_path=tmp_path, load_adapter=True, require_adapter=False)
    assert validate_adapter_config(config) is False
    assert "QLoRA adapter not found; using base Qwen model." in caplog.text


def test_missing_required_adapter_is_configuration_error(tmp_path):
    config = QwenConfig(adapter_path=tmp_path, load_adapter=True, require_adapter=True)
    with pytest.raises(ModelConfigurationError, match="required"):
        validate_adapter_config(config)


def test_qwen_provider_loads_once_and_reports_adapter_state(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    class Tokenizer:
        eos_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            import torch
            return torch.tensor([1])
        def decode(self, tokens, **kwargs):
            return "Evidence indicates a 20% increase."
    class Model:
        device = "cpu"
        calls = 0
        def generate(self, **kwargs):
            self.calls += 1
            return [[1, 2, 3]]
    model = Model()
    provider = QwenReasoningProvider(QwenConfig(adapter_path=tmp_path), loader=lambda _: (model, Tokenizer(), True))
    # Supply an inference-mode stub so this unit test does not depend on a CUDA-capable install.
    import torch
    assert provider.status()["adapter_available"] is True
    assert provider.status()["adapter_loaded"] is False
    assert provider.reason("system", json.dumps({"evidence": "Revenue grew 20%"}))
    assert provider.status()["adapter_loaded"] is True
    assert model.calls == 1
