"""Выбор провайдера LLM: локальная Ollama (по умолчанию) или OpenAI-совместимый
API. Роль-ориентирован — вызывающий код просит summarize()/hint(), а роутер сам
подставляет активного клиента и его модель, чтобы имя модели не протекало наружу.
Провайдер читается на каждый вызов, поэтому переключение в настройках действует
сразу, в том числе во время идущей встречи."""
from .base import LlmClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from ..config import Settings


class LlmRouter:
    def __init__(self, cfg: Settings):
        self._cfg = cfg
        self._ollama = OllamaClient(cfg)
        self._openai = OpenAIClient(cfg)

    @property
    def provider(self) -> str:
        return self._cfg.llm_provider

    def _client(self) -> LlmClient:
        return self._openai if self._cfg.llm_provider == "api" else self._ollama

    @property
    def summary_model_name(self) -> str:
        if self._cfg.llm_provider == "api":
            return self._cfg.llm_api_summary_model
        return self._cfg.summary_model

    @property
    def hints_model_name(self) -> str:
        if self._cfg.llm_provider == "api":
            return self._cfg.llm_api_hints_model
        return self._cfg.hints_model

    async def summarize(self, prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
        return await self._client().generate(
            self.summary_model_name, prompt, system=system, temperature=temperature
        )

    async def hint(self, prompt: str, system: str | None = None, temperature: float = 0.5) -> str:
        return await self._client().generate(
            self.hints_model_name, prompt, system=system, temperature=temperature
        )

    async def status(self) -> dict:
        """Доступность активного провайдера."""
        return await self._client().status()

    async def local_status(self) -> dict:
        """Доступность локальной Ollama независимо от выбранного провайдера
        (в настройках всегда полезно видеть, поднят ли локальный резерв)."""
        return await self._ollama.status()
