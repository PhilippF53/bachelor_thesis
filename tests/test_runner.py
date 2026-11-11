from types import SimpleNamespace

import pytest

from src.config import ModelConfig
from src.inference.runner import ContextOverflowError, InferenceRunner, check_loaded_context


def test_run_stops_if_the_model_is_loaded_with_a_smaller_context_than_configured():
    # codellama:70b-instruct: Ollama loads 2048 tokens although 8192 were requested
    with pytest.raises(ContextOverflowError, match="context of 2048 tokens"):
        check_loaded_context("codellama:70b-instruct", 8192, 2048)


@pytest.mark.parametrize("loaded", [8192, 16384, None])
def test_loaded_context_as_configured_or_unknown_passes(loaded):
    check_loaded_context("model:tag", 8192, loaded)


def test_loaded_model_reads_placement_and_context_from_ollama_ps():
    listed = [
        SimpleNamespace(model="other:1", size=5, size_vram=5, context_length=8192),
        SimpleNamespace(model="m:1", size=10, size_vram=10, context_length=2048),
    ]
    runner = InferenceRunner.__new__(InferenceRunner)  # only what _loaded_model needs, no config or DB
    runner.llm = SimpleNamespace(client=SimpleNamespace(ps=lambda: SimpleNamespace(models=listed)))
    runner.model = ModelConfig(name="m", tag="m:1", format="chat")
    assert runner._loaded_model() == {"size": 10, "size_vram": 10, "fully_on_gpu": True, "context_length": 2048}
    runner.model = ModelConfig(name="x", tag="x:1", format="chat")
    assert runner._loaded_model() == {}  # not loaded
