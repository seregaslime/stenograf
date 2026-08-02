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
        "api", api_base_url="http://api.local/v1", api_key="secret",
        summary_model="sum-m", hints_model="hint-m",
    )
    data = json.loads((tmp_path / "llm.json").read_text())
    assert data == {
        "provider": "api", "api_base_url": "http://api.local/v1",
        "api_key": "secret", "summary_model": "sum-m", "hints_model": "hint-m",
    }
    assert config.settings.llm_api_base_url == "http://api.local/v1"
    assert config.settings.llm_api_summary_model == "sum-m"


def test_save_llm_choice_empty_key_keeps_existing(tmp_path, monkeypatch):
    """Пустой ключ при повторном сохранении не затирает сохранённый (клиент
    присылает ключ только когда его меняют)."""
    monkeypatch.setattr(config.settings, "data_dir", tmp_path)
    monkeypatch.setattr(config.settings, "llm_api_base_url", "http://x/v1")
    monkeypatch.setattr(config.settings, "llm_api_key", "kept")
    config.save_llm_choice(
        "api", api_base_url="http://y/v1", api_key="",
        summary_model="s", hints_model="h",
    )
    assert config.settings.llm_api_key == "kept"          # ключ не тронут
    assert config.settings.llm_api_base_url == "http://y/v1"  # адрес обновлён
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
    monkeypatch.setattr(config.settings, "llm_api_base_url", "http://x/v1")
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
