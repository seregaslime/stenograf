"""Общий контракт LLM-клиентов. Локальная Ollama и OpenAI-совместимый API
взаимозаменяемы за роутером (llm/router.py): у обоих одна сигнатура generate/status."""
from typing import Protocol


class LlmError(Exception):
    """Понятная пользователю ошибка LLM (показывается в UI)."""


class LlmClient(Protocol):
    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.4,
        num_predict: int | None = None,
    ) -> str: ...

    async def status(self) -> dict: ...
