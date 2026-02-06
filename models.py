import gc
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ModelConfig:
    short_name: str
    hf_id: str
    tier: str
    quantize_4bit: bool
    is_thinking: bool = False
    trust_remote_code: bool = False


MODEL_REGISTRY: list[ModelConfig] = [
    # 3B tier
    ModelConfig("llama3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", "3b", False),
    ModelConfig("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", "3b", False),
    ModelConfig("phi3.5-mini", "microsoft/Phi-3.5-mini-instruct", "3b", False),
    # 7B tier
    ModelConfig("mistral-7b", "mistralai/Mistral-7B-Instruct-v0.3", "7b", False),
    ModelConfig("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "7b", False),
    ModelConfig("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "7b", False),
    # 30B tier
    ModelConfig("qwen2.5-32b", "Qwen/Qwen2.5-32B-Instruct", "30b", True),
    ModelConfig("mistral-small", "mistralai/Mistral-Small-Instruct-2409", "30b", True),
    # 70B tier
    ModelConfig("llama3.1-70b", "meta-llama/Llama-3.1-70B-Instruct", "70b", True),
    ModelConfig("qwen2.5-72b", "Qwen/Qwen2.5-72B-Instruct", "70b", True),
    # Thinking tier
    ModelConfig("qwq-32b", "Qwen/QwQ-32B-Preview", "thinking", True, is_thinking=True),
    ModelConfig("deepseek-r1", "deepseek-ai/DeepSeek-R1", "thinking", True, is_thinking=True, trust_remote_code=True),
]

_NAME_INDEX = {m.short_name: m for m in MODEL_REGISTRY}
_TIER_INDEX: dict[str, list[ModelConfig]] = {}
for _m in MODEL_REGISTRY:
    _TIER_INDEX.setdefault(_m.tier, []).append(_m)


def get_models(selector: str) -> list[ModelConfig]:
    """Resolve a selector to a list of ModelConfig objects.

    selector can be:
      - "all"           -> every registered model
      - "3b"/"7b"/...   -> all models in that tier
      - a short name    -> that single model
    """
    if selector == "all":
        return list(MODEL_REGISTRY)
    if selector in _TIER_INDEX:
        return list(_TIER_INDEX[selector])
    if selector in _NAME_INDEX:
        return [_NAME_INDEX[selector]]
    valid = sorted(set(["all"] + list(_TIER_INDEX.keys()) + list(_NAME_INDEX.keys())))
    raise ValueError(f"Unknown model selector: {selector!r}. Valid: {valid}")


def load_model_and_tokenizer(config: ModelConfig):
    """Load model and tokenizer with appropriate dtype/quantization."""
    kwargs: dict = {"device_map": "auto"}
    if config.trust_remote_code:
        kwargs["trust_remote_code"] = True

    if config.quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id, trust_remote_code=config.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(config.hf_id, **kwargs)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer):
    """Free GPU memory occupied by the model."""
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
