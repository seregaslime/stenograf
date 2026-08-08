"""Клиент OpenAI-совместимого API (chat/completions).

Опциональная альтернатива локальной Ollama: один протокол покрывает и внутренний
сервер инференса организации (vLLM/TGI/llama.cpp-server), и внешние сервисы
(OpenAI, Groq, OpenRouter), и локальные (LM Studio, Ollama через /v1). Адрес и
ключ берутся только из настроек сервера (server/.env) — на клиент не уходят."""
import asyncio
import logging
import re
import time

import httpx

from ..config import Settings
from .base import LlmError

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)  # на случай qwen3 за API


def _detail(response: httpx.Response) -> str:
    """Причина отказа человеческими словами.

    OpenAI-совместимые провайдеры кладут её в error.message — там написано, чего
    именно не хватило («Limit 8000, Requested 16324»). Сырое тело ответа, которое
    показывалось раньше, для пользователя бесполезно.
    """
    try:
        message = response.json().get("error", {}).get("message")
    except Exception:
        message = None
    return message or response.text[:200]


def _parse_reset(raw: str | None) -> float:
    """«1.2s», «120ms», «2m59.56s» → секунды. Формат провайдера, не ISO."""
    if not raw:
        return 0.0
    factors = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(
        float(value) * factors[unit]
        for value, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", raw)
    )


class OpenAIClient:
    def __init__(self, cfg: Settings):
        self._cfg = cfg
        # Остаток минутного лимита по данным провайдера. Спать фиксированную
        # минуту оказалось недостаточно: окно скользящее, и после крупного
        # запроса за 60 секунд восстанавливается не всё. На длинной встрече это
        # выглядело так, что первый фрагмент проходит, а дальше всё хуже.
        self._remaining: int | None = None
        self._reset_s = 0.0
        self._checked_at = 0.0

    def _remember_limits(self, response: httpx.Response) -> None:
        raw = response.headers.get("x-ratelimit-remaining-tokens")
        if raw is None:
            return
        try:
            self._remaining = int(raw)
        except ValueError:
            return
        self._reset_s = _parse_reset(response.headers.get("x-ratelimit-reset-tokens"))
        self._checked_at = time.monotonic()

    def _reserve(self, model: str) -> int:
        limit = self._cfg.llm_api_tpm_limits.get(model) or self._cfg.llm_api_tpm_fallback
        return int(max(limit, 0) * self._cfg.llm_api_output_share)

    async def _wait_for_budget(self, need: int) -> None:
        """Ждёт, пока провайдер не восстановит лимит под запрос такого размера.

        Спрашиваем у того, кто знает, вместо того чтобы спать наугад: сколько
        осталось и через сколько вернётся, сказано в заголовках прошлого ответа.
        """
        if self._remaining is None or need <= self._remaining:
            return
        wait = self._reset_s - (time.monotonic() - self._checked_at)
        if wait <= 0:
            return
        log.info("Ждём восстановления лимита: нужно %d токенов, осталось %d, "
                 "провайдер обещает через %.1f с", need, self._remaining, wait)
        await asyncio.sleep(wait + 1)  # секунда сверху: их часы и наши не совпадают
        self._remaining = None  # после ожидания оценка устарела

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

    async def token_limit(self, model: str) -> int | None:
        """Сколько токенов в минуту разрешено этой модели, или None если узнать
        не вышло.

        В списке моделей лимита нет — провайдер сообщает его только заголовком
        ответа. Поэтому шлём самый дешёвый запрос, какой возможен: одно слово и
        один токен в ответе. Делается один раз при выборе модели, чтобы первая
        же встреча шла с правильным бюджетом, а не выясняла его, упираясь в
        лимит посреди разговора.

        Ответ с ошибкой тоже годится: заголовки лимитов приходят и с ним.
        """
        if not (self._cfg.llm_api_base_url and model):
            return None
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "1"}],
            "max_tokens": 1,
            "temperature": 0,
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=5.0)) as client:
                response = await client.post(
                    f"{self._base}/chat/completions", json=payload, headers=self._headers()
                )
        except httpx.HTTPError:
            return None  # сеть у пользователя нестабильна — не повод ломать сохранение настроек
        self._remember_limits(response)
        raw = response.headers.get("x-ratelimit-limit-tokens")
        try:
            return int(raw) if raw else None
        except ValueError:
            return None

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
        # max_tokens намеренно НЕ задаём по умолчанию. Провайдер списывает с
        # минутного лимита фиксированный резерв (2500) независимо от того,
        # просили мы что-то или нет, — но сам предел ответа у модели куда выше
        # (max_completion_tokens = 65536). Выставив max_tokens = 2500, мы
        # списание не уменьшаем, а модели связываем руки: у рассуждающих моделей
        # рассуждения идут из того же лимита, и на текст ответа не остаётся
        # ничего — фрагмент возвращается пустым с finish_reason=length.
        if num_predict:
            payload["max_tokens"] = num_predict

        # Сколько этот запрос будет стоить: вход плюс место под ответ, которое
        # провайдер списывает независимо от того, воспользуется им модель или нет.
        need = int((len(system or "") + len(prompt)) / self._cfg.chars_per_token)
        await self._wait_for_budget(need + self._reserve(model))

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
        except httpx.TimeoutException as exc:
            raise LlmError(
                "API не ответил вовремя. Внешние сервисы работают через VPN — "
                "проверьте, что туннель поднят."
            ) from exc
        except httpx.HTTPError as exc:
            # Связь чаще рвётся на середине запроса, чем не открывается вовсе.
            # Без этой ветки обрыв улетал мимо LlmError и убивал фоновую задачу
            # резюме, оставляя встречу в вечном "summarizing".
            raise LlmError(f"Связь с API оборвалась: {exc}") from exc

        # Остаток лимита читаем с ЛЮБОГО ответа, включая отказ: отказ по лимиту
        # как раз и несёт самые свежие цифры.
        self._remember_limits(response)

        if response.status_code in (401, 403):
            raise LlmError("API отклонил ключ — проверьте STENOGRAF_LLM_API_KEY.")
        if response.status_code == 404:
            raise LlmError(f"Модель «{model}» недоступна на этом API.")
        if response.status_code in (413, 429):
            # Тарифный лимит, а не поломка: 413 — запрос сам по себе крупнее
            # лимита токенов в минуту, 429 — лимит выбран предыдущими запросами.
            # Различать их пользователю незачем, а знать, что делать, — нужно.
            raise LlmError(
                "Запрос не уложился в лимит тарифа. Укоротите встречу, подождите "
                f"минуту или выберите модель с большим лимитом. Ответ API: {_detail(response)}"
            )
        if response.status_code != 200:
            raise LlmError(f"API вернул ошибку {response.status_code}: {_detail(response)}")

        try:
            choice = response.json()["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LlmError("API вернул ответ в неожиданном формате.") from exc

        # Модель упёрлась в предел длины ответа и оборвалась на полуслове. Молча
        # брать такой текст нельзя: у длинной встречи он уходит дальше в сведение
        # фрагментов, и в протоколе не хватает куска разговора — а выглядит
        # протокол целым. Один раз мы так уже потеряли половину третьего
        # фрагмента и заметили только вручную.
        if choice.get("finish_reason") == "length":
            log.warning(
                "Ответ модели «%s» оборван по пределу длины (%d символов) — "
                "часть содержания потеряна", model, len(text)
            )
            raise LlmError(
                "Модель не уместила ответ в отведённую длину. Для длинной встречи "
                "это значит, что часть разговора не попала бы в протокол."
            )
        return _THINK_RE.sub("", text).strip()
