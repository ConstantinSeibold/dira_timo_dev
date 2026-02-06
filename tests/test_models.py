import pytest

from models import MODEL_REGISTRY, ModelConfig, get_models


class TestModelRegistry:
    def test_registry_not_empty(self):
        assert len(MODEL_REGISTRY) > 0

    def test_unique_short_names(self):
        names = [m.short_name for m in MODEL_REGISTRY]
        assert len(names) == len(set(names))

    def test_thinking_models_flagged(self):
        thinking = [m for m in MODEL_REGISTRY if m.is_thinking]
        assert len(thinking) >= 2
        for m in thinking:
            assert m.tier == "thinking"


class TestGetModels:
    def test_all(self):
        models = get_models("all")
        assert len(models) == len(MODEL_REGISTRY)

    def test_tier_3b(self):
        models = get_models("3b")
        assert all(m.tier == "3b" for m in models)
        assert len(models) == 3

    def test_tier_thinking(self):
        models = get_models("thinking")
        assert all(m.is_thinking for m in models)

    def test_single_name(self):
        models = get_models("qwen2.5-3b")
        assert len(models) == 1
        assert models[0].short_name == "qwen2.5-3b"

    def test_invalid_selector(self):
        with pytest.raises(ValueError, match="Unknown model selector"):
            get_models("nonexistent")
