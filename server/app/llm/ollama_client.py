"""Клиент Ollama (локальная LLM). Никаких внешних API — данные не покидают машину/сервер."""
import logging
import re

import httpx

from ..config import Settings

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaError(Exception):
    """Понятная пользователю ошибка LLM (показывается в UI)."""


class OllamaClient:
    def __init__(self, cfg: Settings):
        self._cfg = cfg

    async def status(self) -> dict:
        """Доступен ли Ollama и какие модели скачаны."""
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(f"{self._cfg.ollama_url}/api/tags")
                response.raise_for_status()
                models = [m["name"] for m in response.json().get("models", [])]
                return {"reachable": True, "models": models}
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
        # think у qwen3 не выключаем (рассуждения вслух попадут в ответ) и num_predict
        # не ограничиваем (лимит съедят «мысли», ответ окажется пустым)
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self._cfg.llm_keep_alive,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if num_predict:
            payload["options"]["num_predict"] = num_predict

        timeout = httpx.Timeout(600.0, connect=5.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{self._cfg.ollama_url}/api/generate", json=payload)
        except httpx.ConnectError as exc:
            raise OllamaError(
                f"Ollama недоступен по адресу {self._cfg.ollama_url}. "
                "Запустите его командой `ollama serve`."
            ) from exc

        if response.status_code == 404:
            raise OllamaError(
                f"Модель «{model}» не найдена в Ollama. Скачайте её: `ollama pull {model}`."
            )
        if response.status_code != 200:
            raise OllamaError(f"Ollama вернул ошибку {response.status_code}: {response.text[:200]}")

        text = response.json().get("response", "")
        return _THINK_RE.sub("", text).strip()
