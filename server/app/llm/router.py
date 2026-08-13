"""Выбор провайдера LLM: локальная Ollama (по умолчанию) или OpenAI-совместимый
API. Роль-ориентирован — вызывающий код просит summarize()/hint(), а роутер сам
подставляет активного клиента и его модель, чтобы имя модели не протекало наружу.
Провайдер читается на каждый вызов, поэтому переключение в настройках действует
сразу, в том числе во время идущей встречи."""
from dataclasses import dataclass

from ..config import Settings, save_tpm_limits
from .base import LlmClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient


@dataclass(frozen=True)
class Budget:
    """Сколько контекста не жалко отдать модели и насколько подробный промпт.
    У локальной модели окно маленькое, у API — большое (см. Settings)."""
    summary_chars: int  # 0 = без ограничения
    hints_chars: int
    detailed: bool      # развёрнутый промпт: больше секций и примеров
    # Потолок одного запроса подсказки в токенах — вместе с промптом, не только
    # разговор. 0 = тарифного лимита нет (локальная модель), режем по hints_chars.
    hints_tokens: int = 0
    # Потолок одного запроса резюме. Оно идёт после встречи и в одиночку, поэтому
    # ему достаётся весь минутный лимит, а не доля как подсказкам. 0 = без лимита.
    summary_tokens: int = 0


class LlmRouter:
    def __init__(self, cfg: Settings):
        self._cfg = cfg
        self._ollama = OllamaClient(cfg)
        self._openai = OpenAIClient(cfg)

    @property
    def provider(self) -> str:
        return self._cfg.llm_provider

    @property
    def budget(self) -> Budget:
        """Свойство, а не поле: провайдера могут переключить посреди встречи,
        и глубина подсказок должна поменяться сразу."""
        if self._cfg.llm_provider == "api":
            return Budget(
                self._cfg.summary_max_chars_api, self._cfg.hints_window_chars_api, True,
                self._hints_tokens(), self._tpm_limit(self._cfg.llm_api_summary_model),
            )
        return Budget(
            self._cfg.summary_max_chars_local, self._cfg.hints_window_chars, False
        )

    def _hints_tokens(self) -> int:
        """Сколько токенов можно отдать одной подсказке, чтобы сумма за минуту
        не превысила тарифный лимит.

        Считать расход в реальном времени не нужно: частота подсказок ограничена
        сверху минимальным интервалом между ними, поэтому достаточно поделить
        минутный лимит на максимальное число запросов в минуту — сумма тогда не
        превысит лимит по построению.
        """
        limit = self._tpm_limit(self._cfg.llm_api_hints_model)
        if limit <= 0:
            return 0  # лимита нет (внутренний сервер организации) — не ограничиваем
        per_minute = 60 / max(self._cfg.hints_min_gap_s, 1.0)
        return int(limit / per_minute)

    def _tpm_limit(self, model: str) -> int:
        """Сколько токенов можно ОТПРАВИТЬ этой модели за минуту.

        Из минутного лимита вычитаем резерв под ответ: провайдер засчитывает в
        лимит и его тоже. Без вычета запрос на 5400 токенов при лимите 8000
        получал отказ «Requested 8476» — недостающие три тысячи и были местом,
        отведённым модели на ответ.
        """
        limit = self._cfg.llm_api_tpm_limits.get(model) or self._cfg.llm_api_tpm_fallback
        if limit <= 0:
            return 0  # лимита нет — делить не из чего
        return int(limit * (1 - self._cfg.llm_api_output_share))

    @property
    def chars_per_token(self) -> float:
        return self._cfg.chars_per_token

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

    async def ensure_tpm_limit(self) -> None:
        """Измеряет лимит модели резюме, если он ещё неизвестен.

        Проба делается при сохранении настроек, но у того, кто настроил API
        раньше её появления, лимит не измерен никогда — и он молча работает на
        запасном значении. Разница не косметическая: 6000 вместо 8000 урезает
        бюджет входа на четверть и удваивает число фрагментов у длинной встречи.
        """
        if self._cfg.llm_provider != "api":
            return
        model = self._cfg.llm_api_summary_model
        if not model or model in self._cfg.llm_api_tpm_limits:
            return
        limit = await self._openai.token_limit(model)
        if limit:
            save_tpm_limits({model: limit})

    async def summarize(self, prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
        return await self._client().generate(
            self.summary_model_name, prompt, system=system, temperature=temperature
        )

    async def hint(self, prompt: str, system: str | None = None, temperature: float = 0.5,
                   on_delta=None) -> str:
        """on_delta — печатать ответ по мере генерации.

        Работает только у внешнего API: у локальной Ollama свой протокол, и
        стриминг там пришлось бы делать отдельно. Провайдера переключают в
        настройках, поэтому вызывающий не обязан знать, кто сейчас активен, —
        с локальной моделью он просто получит ответ целиком, как раньше.
        """
        client = self._client()
        if on_delta is not None and isinstance(client, OpenAIClient):
            return await client.generate_streaming(
                self.hints_model_name, prompt, system=system,
                temperature=temperature, on_delta=on_delta,
            )
        return await client.generate(
            self.hints_model_name, prompt, system=system, temperature=temperature
        )

    async def status(self) -> dict:
        """Доступность активного провайдера."""
        return await self._client().status()

    async def local_status(self) -> dict:
        """Доступность локальной Ollama независимо от выбранного провайдера
        (в настройках всегда полезно видеть, поднят ли локальный резерв)."""
        return await self._ollama.status()
