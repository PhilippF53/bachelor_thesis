import time
from dataclasses import asdict, dataclass

import ollama

_NS = 1e9


@dataclass
class LLMResponse:
    content: str
    thinking: str | None
    prompt_tokens: int | None
    output_tokens: int | None
    wall_s: float
    total_duration_s: float | None
    load_duration_s: float | None
    prompt_eval_duration_s: float | None
    eval_duration_s: float | None
    done_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class OllamaService:
    def __init__(self, model_name: str, options: dict, think: bool | str | None = None,
                 keep_alive: str = "30m") -> None:
        self.model_name = model_name
        self.options = options
        self.think = think
        self.keep_alive = keep_alive
        self.client = ollama.Client()

    def _wrap(self, response, content: str, thinking: str | None, wall_s: float) -> LLMResponse:
        def sec(v):
            return None if v is None else v / _NS

        return LLMResponse(
            content=content,
            thinking=thinking or None,
            prompt_tokens=response.prompt_eval_count,
            output_tokens=response.eval_count,
            wall_s=wall_s,
            total_duration_s=sec(response.total_duration),
            load_duration_s=sec(response.load_duration),
            prompt_eval_duration_s=sec(response.prompt_eval_duration),
            eval_duration_s=sec(response.eval_duration),
            done_reason=response.done_reason,
        )

    def chat(self, messages: list[dict]) -> LLMResponse:
        start = time.perf_counter()
        response = self.client.chat(model=self.model_name, messages=messages, think=self.think,
                                    options=self.options, keep_alive=self.keep_alive)
        wall = time.perf_counter() - start
        return self._wrap(response, response.message.content or "", response.message.thinking, wall)

    def generate_raw(self, prompt: str) -> LLMResponse:
        start = time.perf_counter()
        response = self.client.generate(model=self.model_name, prompt=prompt, raw=True,
                                        options=self.options, keep_alive=self.keep_alive)
        wall = time.perf_counter() - start
        return self._wrap(response, response.response or "", response.thinking, wall)

    def model_info(self) -> dict:
        listed = {m.model: m for m in self.client.list().models}
        entry = listed.get(self.model_name)
        shown = self.client.show(self.model_name)
        return {
            "tag": self.model_name,
            "digest": entry.digest if entry else None,
            "size_bytes": entry.size if entry else None,
            "details": shown.details.model_dump() if shown.details else None,
            "capabilities": shown.capabilities,
            "parameters": shown.parameters,
            "context_length": next(
                (v for k, v in (shown.modelinfo or {}).items() if k.endswith(".context_length")), None
            ),
        }

    def unload(self) -> None:
        self.client.generate(model=self.model_name, prompt="", keep_alive=0)

    @staticmethod
    def server_version() -> str:
        import httpx

        return httpx.get(f"{ollama.Client()._client.base_url}/api/version", timeout=10).json()["version"]
