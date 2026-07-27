"""Клиент OpenAI-совместимого API (chat/completions).

Опциональная альтернатива локальной Ollama: один протокол покрывает и внутренний
сервер инференса организации (vLLM/TGI/llama.cpp-server), и внешние сервисы
(OpenAI, Groq, OpenRouter), и локальные (LM Studio, Ollama через /v1). Адрес и
ключ берутся только из настроек сервера (server/.env) — на клиент не уходят."""
import logging
import re

import httpx

from ..config import Settings
from .base import LlmError

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)  # на случай qwen3 за API


class OpenAIClient:
    def __init__(self, cfg: Settings):
        self._cfg = cfg

    @property
    def _base(self) -> str:
        return self._cfg.llm_api_base_url.rstrip("/")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._cfg.llm_api_key:
            headers["Authorization"] = f"Bearer {self._cfg.llm_api_key}"
        return headers

    async def status(self) -> dict:
        """Отвечает ли endpoint и какие модели он декларирует (GET /models)."""
        if not self._cfg.llm_api_base_url:
            return {"reachable": False, "models": []}
        try:
            # внешний API может отвечать не мгновенно (сеть) — таймаут щедрее локального
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.get(f"{self._base}/models", headers=self._headers())
                response.raise_for_status()
                models = [m.get("id") for m in response.json().get("data", [])]
                return {"reachable": True, "models": [m for m in models if m]}
        except Exception:
            return {"reachable": False, "models": []}

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.4,
        num_predict: int | None = None,
    ) -> str:
        if not self._cfg.llm_api_base_url:
            raise LlmError("Не задан адрес API (STENOGRAF_LLM_API_BASE_URL).")
        if not model:
            raise LlmError(
                "Не задана модель API (STENOGRAF_LLM_API_SUMMARY_MODEL / _HINTS_MODEL)."
            )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if num_predict:
            payload["max_tokens"] = num_predict

        timeout = httpx.Timeout(600.0, connect=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self._base}/chat/completions", json=payload, headers=self._headers()
                )
        except httpx.ConnectError as exc:
            raise LlmError(
                f"API недоступен по адресу {self._base}. Проверьте STENOGRAF_LLM_API_BASE_URL."
            ) from exc

        if response.status_code in (401, 403):
            raise LlmError("API отклонил ключ — проверьте STENOGRAF_LLM_API_KEY.")
        if response.status_code == 404:
            raise LlmError(f"Модель «{model}» недоступна на этом API.")
        if response.status_code != 200:
            raise LlmError(f"API вернул ошибку {response.status_code}: {response.text[:200]}")

        try:
            text = response.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError("API вернул ответ в неожиданном формате.") from exc
        return _THINK_RE.sub("", text).strip()
