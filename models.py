import gc
import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    short_name: str
    hf_id: str
    tier: str
    is_thinking: bool = False
    trust_remote_code: bool = False


MODEL_REGISTRY: list[ModelConfig] = [
    # 3B tier
    ModelConfig("llama3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", "3b"),
    ModelConfig("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", "3b"),
    ModelConfig("phi3.5-mini", "microsoft/Phi-3.5-mini-instruct", "3b"),
    # 7B tier
    ModelConfig("mistral-7b", "mistralai/Mistral-7B-Instruct-v0.3", "7b"),
    ModelConfig("llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "7b"),
    ModelConfig("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "7b"),
    # 30B tier
    ModelConfig("qwen2.5-32b", "Qwen/Qwen2.5-32B-Instruct", "30b"),
    ModelConfig("mistral-small", "mistralai/Mistral-Small-Instruct-2409", "30b"),
    # 70B tier
    ModelConfig("llama3.1-70b", "meta-llama/Llama-3.1-70B-Instruct", "70b"),
    ModelConfig("qwen2.5-72b", "Qwen/Qwen2.5-72B-Instruct", "70b"),
    # Thinking 7B tier
    ModelConfig("deepseek-r1-14b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "thinking-7b", is_thinking=True),
    # Thinking 30B tier
    ModelConfig("qwq-32b", "Qwen/QwQ-32B-Preview", "thinking-30b", is_thinking=True),
    ModelConfig("deepseek-r1-32b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "thinking-30b", is_thinking=True),
    # Thinking 70B tier
    ModelConfig("deepseek-r1-70b", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "thinking-70b", is_thinking=True),
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
    # "thinking" selects all thinking-* tiers
    if selector == "thinking":
        return [m for m in MODEL_REGISTRY if m.tier.startswith("thinking")]
    if selector in _NAME_INDEX:
        return [_NAME_INDEX[selector]]
    valid = sorted(set(["all"] + list(_TIER_INDEX.keys()) + list(_NAME_INDEX.keys())))
    raise ValueError(f"Unknown model selector: {selector!r}. Valid: {valid}")


def load_model_and_tokenizer(config: ModelConfig, *, quantize_4bit: bool = False):
    """Load model and tokenizer, optionally with NF4 4-bit quantization."""
    kwargs: dict = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16,
    }
    if config.trust_remote_code:
        kwargs["trust_remote_code"] = True

    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs.pop("torch_dtype", None)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["quantization_config"] = bnb_config
        logger.info(
            "4-bit quantization enabled: load_in_4bit=%s, quant_type=%s",
            bnb_config.load_in_4bit, bnb_config.bnb_4bit_quant_type,
        )

    # Limit GPU memory to avoid OOM; overflow goes to CPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        max_gpu = f"{int(gpu_mem * 0.85 / 1e9)}GiB"
        kwargs["max_memory"] = {0: max_gpu, "cpu": "100GiB"}
        logger.info("max_memory: GPU=%s, CPU=100GiB", max_gpu)

    logger.info("from_pretrained kwargs: %s", {k: str(v)[:80] for k, v in kwargs.items()})

    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_id, trust_remote_code=config.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(config.hf_id, **kwargs)

    # Verify quantization was actually applied
    if quantize_4bit:
        attn = model.model.layers[0].self_attn
        # Phi-3.5 uses fused qkv_proj; most others use separate q_proj
        for proj_name in ("q_proj", "qkv_proj", "o_proj"):
            if hasattr(attn, proj_name):
                sample_layer = getattr(attn, proj_name)
                break
        else:
            sample_layer = None
        layer_type = type(sample_layer).__name__ if sample_layer else "unknown"
        logger.info("Quantization check: layer type = %s", layer_type)
        if "4bit" not in layer_type.lower() and "bnb" not in layer_type.lower():
            logger.warning(
                "QUANTIZATION NOT APPLIED — layer is %s, expected Linear4bit. "
                "Model is loaded in full precision!", layer_type,
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer):
    """Free GPU memory occupied by the model."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
