"""Провайдер LLM: OpenAI-совместимый клиент, роутер local↔api и персист выбора.

HTTP мокается httpx.MockTransport — без сети и без внешних зависимостей.
Асинхронные вызовы гоняются через asyncio.run (pytest-asyncio в проекте нет).
"""
import asyncio
import json

import httpx
import pytest

from app import config
from app.config import Settings
from app.llm import openai_client as openai_mod
from app.llm.base import LlmError
from app.llm.openai_client import OpenAIClient
from app.llm.router import LlmRouter


# Сохранять можно только поддерживаемый хост (см. LLM_API_ALLOWED_HOSTS):
# чужой провайдер не сообщает контекст модели.
GROQ = "https://api.groq.com/openai/v1"
GROQ_ALT = "https://api.groq.com/openai/v1/"


def _api_cfg(tmp_path, **over) -> Settings:
    base = dict(
        data_dir=tmp_path, _env_file=None, llm_provider="api",
        llm_api_base_url="http://api.local/v1", llm_api_key="secret",
        llm_api_summary_model="sum-m", llm_api_hints_model="hint-m",
    )
    base.update(over)
    return Settings(**base)


def _patch_transport(monkeypatch, handler) -> None:
    """Подменяет httpx.AsyncClient на клиент с MockTransport (перехват запросов)."""
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(openai_mod.httpx, "AsyncClient", factory)


# ------------------------------------------------------------------ OpenAIClient

def test_openai_generate_builds_request_and_parses(tmp_path, monkeypatch):
    """Запрос к OpenAI-совместимому API уходит на /chat/completions с моделью, ролями и температурой; ответ разбирается и обрезается.
    """
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "  ответ  "}}]})

    _patch_transport(monkeypatch, handler)
    out = asyncio.run(
        OpenAIClient(_api_cfg(tmp_path)).generate(
            "sum-m", "привет", system="ты ассистент", temperature=0.3
        )
    )
    assert out == "ответ"  # обрезаны пробелы
    assert captured["url"] == "http://api.local/v1/chat/completions"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"]["model"] == "sum-m"
    assert captured["body"]["temperature"] == 0.3
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "ты ассистент"},
        {"role": "user", "content": "привет"},
    ]


def test_openai_generate_strips_think(tmp_path, monkeypatch):
    """Блок <think> вырезается и из ответа внешнего API (там тоже может стоять qwen3)."""
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>ммм</think>итог"}}]})

    _patch_transport(monkeypatch, handler)
    out = asyncio.run(OpenAIClient(_api_cfg(tmp_path)).generate("m", "p"))
    assert out == "итог"


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_openai_http_errors_raise_llmerror(tmp_path, monkeypatch, status):
    """Коды 401/403/404/500 превращаются в понятную пользователю ошибку, а не в сырое исключение.
    """
    def handler(request):
        return httpx.Response(status, json={"error": "nope"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LlmError):
        asyncio.run(OpenAIClient(_api_cfg(tmp_path)).generate("m", "p"))


def test_openai_connect_error_raises_llmerror(tmp_path, monkeypatch):
    """Недоступный адрес API даёт понятное сообщение с указанием, что проверить."""
    def handler(request):
        raise httpx.ConnectError("boom")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LlmError):
        asyncio.run(OpenAIClient(_api_cfg(tmp_path)).generate("m", "p"))


@pytest.mark.parametrize("status", [413, 429])
def test_openai_rate_limit_explains_what_to_do(tmp_path, monkeypatch, status):
    """Тарифный лимит — не поломка, и сообщение должно это отражать.

    Groq на бесплатном тарифе отвечает 413, когда один запрос крупнее лимита
    токенов в минуту. Пользователю нужен не код ошибки, а причина (сколько
    просили и сколько можно) и что с этим делать.
    """
    def handler(request):
        return httpx.Response(status, json={"error": {"message": "Limit 8000, Requested 16324"}})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LlmError) as exc:
        asyncio.run(OpenAIClient(_api_cfg(tmp_path)).generate("m", "p"))
    assert "Limit 8000, Requested 16324" in str(exc.value)
    assert "Укоротите встречу" in str(exc.value)


def test_openai_timeout_raises_llmerror(tmp_path, monkeypatch):
    """Таймаут превращается в LlmError, а не улетает наружу.

    Раньше ловился только ConnectError, поэтому упавший посреди запроса VPN
    убивал фоновую задачу резюме и встреча навсегда оставалась «summarizing».
    """
    def handler(request):
        raise httpx.ReadTimeout("too slow")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LlmError) as exc:
        asyncio.run(OpenAIClient(_api_cfg(tmp_path)).generate("m", "p"))
    assert "VPN" in str(exc.value)


def test_openai_broken_connection_raises_llmerror(tmp_path, monkeypatch):
    """Обрыв связи на середине ответа — тоже LlmError, а не сырое исключение."""
    def handler(request):
        raise httpx.RemoteProtocolError("server disconnected")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LlmError):
        asyncio.run(OpenAIClient(_api_cfg(tmp_path)).generate("m", "p"))


def test_token_limit_read_from_header(tmp_path, monkeypatch):
    """Лимит токенов в минуту берётся из заголовка ответа.

    В списке моделей его нет — провайдер сообщает лимит только так. Меряем при
    выборе модели, чтобы первая встреча шла с правильным бюджетом.
    """
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"x-ratelimit-limit-tokens": "8000"},
            json={"choices": [{"message": {"content": "1"}}]},
        )

    _patch_transport(monkeypatch, handler)
    limit = asyncio.run(OpenAIClient(_api_cfg(tmp_path)).token_limit("m"))
    assert limit == 8000
    assert seen["body"]["max_tokens"] == 1  # проба должна быть дешёвой


def test_token_limit_survives_error_response(tmp_path, monkeypatch):
    """Заголовки лимитов приходят и с ошибкой — ответ 429 тоже годится."""
    def handler(request):
        return httpx.Response(429, headers={"x-ratelimit-limit-tokens": "6000"}, json={})

    _patch_transport(monkeypatch, handler)
    assert asyncio.run(OpenAIClient(_api_cfg(tmp_path)).token_limit("m")) == 6000


@pytest.mark.parametrize("handler_kind", ["no_header", "network"])
def test_token_limit_returns_none_when_unknown(tmp_path, monkeypatch, handler_kind):
    """Проба необязательна: не вышло — работаем на запасном значении.

    Сеть у пользователя нестабильна (внешние API только через VPN), и упавшая
    проба не должна мешать сохранению настроек.
    """
    def handler(request):
        if handler_kind == "network":
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"choices": [{"message": {"content": "1"}}]})

    _patch_transport(monkeypatch, handler)
    assert asyncio.run(OpenAIClient(_api_cfg(tmp_path)).token_limit("m")) is None


def test_hints_budget_derived_from_measured_limit(tmp_path):
    """Бюджет подсказки = минутный лимит ÷ максимум запросов в минуту.

    Частота ограничена сверху hints_min_gap_s, поэтому сумма за минуту не
    превысит лимит по построению — считать расход в реальном времени не нужно.
    """
    cfg = Settings(data_dir=tmp_path, _env_file=None)
    cfg.llm_provider = "api"
    cfg.llm_api_hints_model = "groq-model"
    cfg.llm_api_tpm_limits = {"groq-model": 8000}
    cfg.hints_min_gap_s = 15.0  # → не чаще 4 запросов в минуту

    assert LlmRouter(cfg).budget.hints_tokens == 2000

    # Модель с большим лимитом получает пропорционально больше — без правки настроек
    cfg.llm_api_tpm_limits = {"groq-model": 40_000}
    assert LlmRouter(cfg).budget.hints_tokens == 10_000


def test_hints_budget_falls_back_when_limit_unknown(tmp_path):
    """Лимит не измерен — берём запасное значение, а не считаем «без лимита»."""
    cfg = Settings(data_dir=tmp_path, _env_file=None)
    cfg.llm_provider = "api"
    cfg.llm_api_hints_model = "неизмеренная"
    cfg.llm_api_tpm_fallback = 6000
    cfg.hints_min_gap_s = 15.0

    assert LlmRouter(cfg).budget.hints_tokens == 1500


def test_local_provider_has_no_token_budget(tmp_path):
    """У локальной модели тарифного лимита нет — режем по символам, как раньше."""
    cfg = Settings(data_dir=tmp_path, _env_file=None)
    cfg.llm_provider = "local"
    assert LlmRouter(cfg).budget.hints_tokens == 0


def test_openai_bad_shape_raises_llmerror(tmp_path, monkeypatch):
    """Ответ неожиданной формы не роняет сервер, а даёт внятную ошибку."""
    def handler(request):
        return httpx.Response(200, json={"unexpected": 1})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(LlmError):
        asyncio.run(OpenAIClient(_api_cfg(tmp_path)).generate("m", "p"))


def test_openai_requires_base_url_and_model(tmp_path):
    """Без адреса API или без имени модели запрос не отправляется."""
    with pytest.raises(LlmError):
        asyncio.run(OpenAIClient(_api_cfg(tmp_path, llm_api_base_url="")).generate("m", "p"))
    with pytest.raises(LlmError):
        asyncio.run(OpenAIClient(_api_cfg(tmp_path)).generate("", "p"))  # пустая модель


# ------------------------------------------------------------------ LlmRouter

class _FakeClient:
    def __init__(self, name: str):
        self.name = name
        self.calls = []

    async def generate(self, model, prompt, system=None, temperature=0.4, num_predict=None):
        self.calls.append((model, prompt, system, temperature))
        return f"{self.name}:{model}"

    async def status(self):
        return {"reachable": True, "models": [self.name]}


def _router_with_fakes(cfg) -> LlmRouter:
    r = LlmRouter(cfg)
    r._ollama = _FakeClient("ollama")
    r._openai = _FakeClient("openai")
    return r


def test_router_local_dispatch(tmp_path):
    """При провайдере local роутер зовёт клиента Ollama и подставляет локальное имя модели."""
    r = _router_with_fakes(Settings(data_dir=tmp_path, _env_file=None))  # provider=local по умолчанию
    assert r.provider == "local"
    out = asyncio.run(r.hint("p", system="s"))
    assert out == f"ollama:{r.hints_model_name}"
    assert r._ollama.calls and not r._openai.calls


def test_router_api_dispatch_uses_api_model(tmp_path):
    """При провайдере api роутер зовёт OpenAI-клиента и подставляет модель, заданную для API."""
    r = _router_with_fakes(_api_cfg(tmp_path))
    assert r.provider == "api"
    assert r.summary_model_name == "sum-m" and r.hints_model_name == "hint-m"
    out = asyncio.run(r.summarize("p"))
    assert out == "openai:sum-m"
    assert r._openai.calls and not r._ollama.calls


# ------------------------------------------------------------------ persist (config.save/load_llm_choice)

def test_save_llm_choice_local_persists(tmp_path, monkeypatch):
    """Выбор локального провайдера сохраняется в llm.json и применяется к настройкам."""
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(config.settings, "llm_provider", "api")
    config.save_llm_choice("local")
    assert json.loads((tmp_path / "llm.json").read_text())["provider"] == "local"
    assert config.settings.llm_provider == "local"


def test_save_llm_choice_persists_creds(tmp_path, monkeypatch):
    """Адрес, ключ и обе модели сохраняются в llm.json — их вводят в настройках приложения, а не в .env.
    """
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(config.settings, "llm_api_base_url", "")
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    config.save_llm_choice(
        "api", api_base_url=GROQ, api_key="secret",
        summary_model="sum-m", hints_model="hint-m",
    )
    data = json.loads((tmp_path / "llm.json").read_text())
    assert data == {
        "provider": "api", "api_base_url": GROQ,
        "api_key": "secret", "summary_model": "sum-m", "hints_model": "hint-m",
        # измеренные лимиты живут рядом с выбором: их дописывает save_tpm_limits
        # уже после сохранения, поэтому здесь они пустые
        "tpm_limits": {},
    }
    assert config.settings.llm_api_base_url == GROQ
    assert config.settings.llm_api_summary_model == "sum-m"


def test_tpm_limits_persist_and_reload(tmp_path, monkeypatch):
    """Измеренный лимит переживает перезапуск сервера — мерить каждый раз незачем."""
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(config.settings, "llm_api_base_url", "")
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    monkeypatch.setattr(config.settings, "llm_api_tpm_limits", {})
    config.save_llm_choice("api", api_base_url=GROQ, api_key="k", hints_model="hint-m")
    config.save_tpm_limits({"hint-m": 8000})

    monkeypatch.setattr(config.settings, "llm_api_tpm_limits", {})
    config.load_llm_choice()
    assert config.settings.llm_api_tpm_limits == {"hint-m": 8000}


def test_save_tpm_limits_without_saved_choice_does_not_crash(tmp_path, monkeypatch):
    """Лимиты без сохранённого выбора записывать некуда — молча выходим."""
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(config.settings, "llm_api_tpm_limits", {})
    config.save_tpm_limits({"m": 8000})  # llm.json ещё нет
    assert not (tmp_path / "llm.json").exists()


def test_save_llm_choice_empty_key_keeps_existing(tmp_path, monkeypatch):
    """Пустой ключ при повторном сохранении не затирает сохранённый (клиент
    присылает ключ только когда его меняют)."""
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(config.settings, "llm_api_base_url", GROQ)
    monkeypatch.setattr(config.settings, "llm_api_key", "kept")
    config.save_llm_choice(
        "api", api_base_url=GROQ_ALT, api_key="",
        summary_model="s", hints_model="h",
    )
    assert config.settings.llm_api_key == "kept"          # ключ не тронут
    assert config.settings.llm_api_base_url == GROQ_ALT  # адрес обновлён
    assert json.loads((tmp_path / "llm.json").read_text())["api_key"] == "kept"


def test_save_llm_choice_rejects_unknown(tmp_path, monkeypatch):
    """Неизвестное имя провайдера отвергается на уровне конфига."""
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    with pytest.raises(ValueError):
        config.save_llm_choice("gguf")


def test_save_llm_choice_api_requires_config(tmp_path, monkeypatch):
    """Провайдер api без адреса и ключа сохранить нельзя."""
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(config.settings, "llm_api_base_url", "")
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    with pytest.raises(ValueError):
        config.save_llm_choice("api")


def test_save_and_load_llm_choice_api_roundtrip(tmp_path, monkeypatch):
    """Выбор api переживает перезапуск: пишется на диск и читается обратно."""
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(config.settings, "llm_api_base_url", GROQ)
    monkeypatch.setattr(config.settings, "llm_api_key", "k")
    monkeypatch.setattr(config.settings, "llm_provider", "local")
    config.save_llm_choice("api")
    assert config.settings.llm_provider == "api"

    monkeypatch.setattr(config.settings, "llm_provider", "local")  # сброс, затем чтение с диска
    config.load_llm_choice()
    assert config.settings.llm_provider == "api"


def test_load_llm_choice_falls_back_when_api_unconfigured(tmp_path, monkeypatch):
    """Если сохранён api, но конфигурация подключения пропала, сервер честно откатывается на локальную модель.
    """
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    (tmp_path / "llm.json").write_text(json.dumps({"provider": "api"}))
    monkeypatch.setattr(config.settings, "llm_api_base_url", "")
    monkeypatch.setattr(config.settings, "llm_api_key", "")
    monkeypatch.setattr(config.settings, "llm_provider", "local")
    config.load_llm_choice()
    assert config.settings.llm_provider == "local"  # api сохранён, но конфиг пропал → local


# ------------------------------------------------------------------ бюджет контекста

def test_budget_switches_with_provider(tmp_path):
    """Бюджет — свойство, а не поле: провайдера меняют посреди встречи, и
    глубина подсказок/резюме обязана поменяться сразу."""
    cfg = Settings(data_dir=tmp_path, _env_file=None)  # local по умолчанию
    router = LlmRouter(cfg)
    assert router.budget.hints_chars == cfg.hints_window_chars
    assert router.budget.summary_chars == cfg.summary_max_chars_local
    assert router.budget.detailed is False

    cfg.llm_provider = "api"
    assert router.budget.hints_chars == cfg.hints_window_chars_api
    assert router.budget.summary_chars == cfg.summary_max_chars_api  # 0 = без лимита
    assert router.budget.detailed is True
    assert router.budget.hints_chars > cfg.hints_window_chars


# ------------------------------------------------------------------ фильтр моделей

def _models_response(*models) -> dict:
    return {"data": list(models)}


def _model(mid, ctx=131072, mi=("text",), mo=("text",)) -> dict:
    return {"id": mid, "context_window": ctx,
            "input_modalities": list(mi), "output_modalities": list(mo)}


def test_status_filters_unsuitable_models(tmp_path, monkeypatch):
    """Судим по данным провайдера, а не по зашитым именам моделей.

    Реальный ответ Groq содержит whisper (принимает аудио), orpheus (отдаёт
    речь), prompt-guard (512 токенов) и allam (4096) — всё это молча сломалось
    бы на первой встрече, поэтому в список выбора не попадает.
    """
    def handler(request):
        return httpx.Response(200, json=_models_response(
            _model("llama-3.3-70b-versatile"),
            _model("openai/gpt-oss-120b"),
            _model("whisper-large-v3", ctx=448, mi=("audio",), mo=("transcription",)),
            _model("canopylabs/orpheus-v1-english", ctx=4000, mo=("speech",)),
            _model("meta-llama/llama-prompt-guard-2-86m", ctx=512),
            _model("allam-2-7b", ctx=4096),
        ))

    _patch_transport(monkeypatch, handler)
    status = asyncio.run(OpenAIClient(_api_cfg(tmp_path)).status())

    assert status["reachable"] is True
    assert status["models"] == ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
    assert status["models_rejected"] == 4


def test_status_keeps_models_without_metadata(tmp_path, monkeypatch):
    """Провайдер не сообщил контекст и модальности — судить не по чему,
    прятать нельзя: молча урезать список опаснее, чем показать лишнее."""
    def handler(request):
        return httpx.Response(200, json=_models_response(
            {"id": "gpt-4o-mini"}, {"id": "some-model"},
        ))

    _patch_transport(monkeypatch, handler)
    status = asyncio.run(OpenAIClient(_api_cfg(tmp_path)).status())
    assert set(status["models"]) == {"gpt-4o-mini", "some-model"}
    assert status["models_rejected"] == 0
    assert all(m["context_window"] is None for m in status["models_info"])


def test_status_reports_context_and_sorts_by_it(tmp_path, monkeypatch):
    def handler(request):
        return httpx.Response(200, json=_models_response(
            _model("small", ctx=32_768), _model("big", ctx=131_072),
        ))

    _patch_transport(monkeypatch, handler)
    info = asyncio.run(OpenAIClient(_api_cfg(tmp_path)).status())["models_info"]
    assert [m["id"] for m in info] == ["big", "small"]  # сначала самые вместительные
    assert info[0]["context_window"] == 131_072


def test_min_context_threshold_is_configurable(tmp_path, monkeypatch):
    """Порог берётся из настроек: поднимаем — модель выпадает из списка."""
    def handler(request):
        return httpx.Response(200, json=_models_response(_model("mid", ctx=32_768)))

    _patch_transport(monkeypatch, handler)
    cfg = _api_cfg(tmp_path, llm_api_min_context_tokens=65_536)
    status = asyncio.run(OpenAIClient(cfg).status())
    assert status["models"] == [] and status["models_rejected"] == 1


# ------------------------------------------------------------------ ограничение провайдера

def test_only_groq_host_is_supported():
    from app.config import api_host_supported
    assert api_host_supported("https://api.groq.com/openai/v1")
    assert api_host_supported("https://api.groq.com/openai/v1/")
    assert not api_host_supported("https://api.openai.com/v1")
    assert not api_host_supported("http://ai.corp.local:8000/v1")
    assert not api_host_supported("")


def test_save_llm_choice_rejects_foreign_host(tmp_path, monkeypatch):
    """Чужой провайдер не сообщает контекст — принять его значит пустить
    пользователя выбирать модель вслепую."""
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    with pytest.raises(ValueError, match="Groq"):
        config.save_llm_choice(
            "api", api_base_url="https://api.openai.com/v1", api_key="k",
            summary_model="m", hints_model="m",
        )
