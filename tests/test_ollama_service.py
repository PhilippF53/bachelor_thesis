import ollama
import pytest

from src.services.ollama_service import REPEAT_LIMIT_REASON, OllamaService


class _FailingClient:
    def __init__(self, message: str) -> None:
        self.message = message

    def _fail(self, **kwargs):
        raise ollama.ResponseError(self.message, 500)

    generate = chat = _fail


def _service(message: str) -> OllamaService:
    service = OllamaService("dummy:latest", {})
    service.client = _FailingClient(message)
    return service


@pytest.mark.parametrize("call", [
    lambda s: s.generate_raw("prompt"),
    lambda s: s.chat([{"role": "user", "content": "prompt"}]),
])
def test_repeat_limit_becomes_empty_answer(call):
    resp = call(_service("prediction aborted, token repeat limit reached"))
    assert resp.content == ""
    assert resp.done_reason == REPEAT_LIMIT_REASON
    assert resp.wall_s >= 0


def test_other_server_errors_still_raise():
    with pytest.raises(ollama.ResponseError):
        _service("model runner has unexpectedly stopped").generate_raw("prompt")
