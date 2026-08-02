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

    def _suitable(self, raw: dict) -> bool:
        """Годится ли модель для наших задач.

        Судим ТОЛЬКО по тому, что прислал провайдер, — никаких зашитых списков
        имён: у другого провайдера модели будут называться иначе.

        Отсеиваем два класса:
        - не текст→текст (whisper принимает аудио, orpheus отдаёт речь);
        - контекст меньше нужного (llama-prompt-guard — 512 токенов,
          allam-2-7b — 4096, а одно только окно подсказок это ~16k).

        Если поля отсутствуют, модель не прячем: судить не по чему, а молча
        урезать список опаснее, чем показать лишнее.
        """
        modal_in = raw.get("input_modalities")
        modal_out = raw.get("output_modalities")
        if modal_in is not None and "text" not in modal_in:
            return False
        if modal_out is not None and "text" not in modal_out:
            return False
        context = raw.get("context_window") or raw.get("context_length")
        if context is not None and context < self._cfg.llm_api_min_context_tokens:
            return False
        return True

    async def status(self) -> dict:
        """Отвечает ли endpoint и какие модели годятся (GET /models).

        `models` — только пригодные, их и показываем в выборе.
        `models_info` — те же с размером контекста, чтобы UI мог его показать.
        `models_rejected` — сколько отсеяли (для честного «показано N из M»).
        """
        empty = {"reachable": False, "models": [], "models_info": [], "models_rejected": 0}
        if not self._cfg.llm_api_base_url:
            return empty
        try:
            # внешний API может отвечать не мгновенно (сеть) — таймаут щедрее локального
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.get(f"{self._base}/models", headers=self._headers())
                response.raise_for_status()
                raw_models = [m for m in response.json().get("data", []) if m.get("id")]
        except Exception:
            return empty

        info = [
            {
                "id": m["id"],
                "context_window": m.get("context_window") or m.get("context_length"),
            }
            for m in raw_models
            if self._suitable(m)
        ]
        info.sort(key=lambda m: (-(m["context_window"] or 0), m["id"]))
        return {
            "reachable": True,
            "models": [m["id"] for m in info],
            "models_info": info,
            "models_rejected": len(raw_models) - len(info),
        }

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
