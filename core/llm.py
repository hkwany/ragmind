"""可插拔 LLM 层，带 token 核算。

设计要点：
1. 每次调用返回 prompt_tokens / completion_tokens，成本账才能算。
2. 流式（SSE）和非流式统一到同一个接口，上层不用关心 provider。
3. mock provider 用于离线跑通 + 压测，语义上「把检索到的资料拼进答案」，
   这样冒烟测试能验证「答案确实带着检索上下文」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Any

from core.config import settings


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    finish_reason: str = "stop"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BaseLLM:
    model: str = "base"

    async def generate(self, messages: list[dict], **kwargs: Any) -> LLMResult:  # pragma: no cover
        raise NotImplementedError

    async def stream(self, messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:  # pragma: no cover
        raise NotImplementedError
        yield ""  # pragma: no cover


class MockLLM(BaseLLM):
    """确定性 mock：把 system 里塞进的『上下文』摘出来拼成答案。

    作用：离线验证检索链路 —— 如果检索返回了正确 chunk，mock 答案里就会出现它。
    """

    model = "mock-model"

    async def generate(self, messages: list[dict], **kwargs: Any) -> LLMResult:
        ctx = self._extract_context(messages)
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = (
            f"[mock answer] 基于以下资料：{ctx[:200] if ctx else '(无上下文)'}\n"
            f"问题：{user[:120]}"
        )
        return LLMResult(text=text, prompt_tokens=len(ctx) // 4, completion_tokens=len(text) // 4, model=self.model)

    async def stream(self, messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        r = await self.generate(messages, **kwargs)
        for i in range(0, len(r.text), 20):
            yield r.text[i : i + 20]

    @staticmethod
    def _extract_context(messages: list[dict]) -> str:
        for m in messages:
            if m["role"] == "system" and m.get("_context"):
                return str(m["_context"])
        return ""


class OpenAICompatLLM(BaseLLM):
    """OpenAI 兼容（也覆盖 vLLM）。需要 openai 包；用 requests 而非 SDK 也行，这里用官方 SDK。"""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None) -> None:
        import openai  # 延迟导入

        self.model = model or settings.llm_model
        self.client = openai.AsyncOpenAI(
            base_url=base_url or settings.llm_base_url, api_key=api_key or settings.llm_api_key
        )

    async def generate(self, messages: list[dict], **kwargs: Any) -> LLMResult:
        messages = self._strip_context(messages)
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", settings.generation_temperature),
            max_tokens=kwargs.get("max_tokens", settings.max_new_tokens),
        )
        ch = resp.choices[0]
        return LLMResult(
            text=ch.message.content or "",
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=resp.model or self.model,
            finish_reason=ch.finish_reason or "stop",
        )

    async def stream(self, messages: list[dict], **kwargs: Any) -> AsyncIterator[str]:
        messages = self._strip_context(messages)
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", settings.generation_temperature),
            max_tokens=kwargs.get("max_tokens", settings.max_new_tokens),
            stream=True,
        )
        async for chunk in resp:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    @staticmethod
    def _strip_context(messages: list[dict]) -> list[dict]:
        # 把内部用的 _context 字段去掉，避免发给模型
        return [{k: v for k, v in m.items() if k != "_context"} for m in messages]


def get_llm() -> BaseLLM:
    if settings.llm_provider in ("openai", "vllm"):
        return OpenAICompatLLM()
    return MockLLM()
