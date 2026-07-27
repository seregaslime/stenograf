"""Юнит-тесты клиента локальной LLM (llm/ollama_client.py) через httpx.MockTransport."""
import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.llm import ollama_client as oll_mod
from app.llm.base import LlmError
from app.llm.ollama_client import OllamaClient


def _patch(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*a, **k):
        k["transport"] = httpx.MockTransport(handler)
        return real(*a, **k)

    monkeypatch.setattr(oll_mod.httpx, "AsyncClient", factory)


def _cfg(tmp_path):
    return Settings(data_dir=tmp_path, _env_file=None)


def test_generate_success_builds_request(tmp_path, monkeypatch):
    cap = {}

    def handler(req):
        cap["url"] = str(req.url)
        cap["body"] = json.loads(req.content)
        return httpx.Response(200, json={"response": "  ответ  "})

    _patch(monkeypatch, handler)
    out = asyncio.run(OllamaClient(_cfg(tmp_path)).generate(
        "qwen3:4b", "prompt", system="sys", temperature=0.3))
    assert out == "ответ"
    assert cap["url"].endswith("/api/generate")
    assert cap["body"]["model"] == "qwen3:4b"
    assert cap["body"]["system"] == "sys"
    assert cap["body"]["stream"] is False
    assert cap["body"]["options"]["temperature"] == 0.3


def test_generate_strips_think(tmp_path, monkeypatch):
    _patch(monkeypatch, lambda req: httpx.Response(200, json={"response": "<think> х</think>итог"}))
    assert asyncio.run(OllamaClient(_cfg(tmp_path)).generate("m", "p")) == "итог"


def test_generate_404_model_missing(tmp_path, monkeypatch):
    _patch(monkeypatch, lambda req: httpx.Response(404, text="not found"))
    with pytest.raises(LlmError):
        asyncio.run(OllamaClient(_cfg(tmp_path)).generate("missing", "p"))


def test_generate_connect_error(tmp_path, monkeypatch):
    def handler(req):
        raise httpx.ConnectError("down")

    _patch(monkeypatch, handler)
    with pytest.raises(LlmError):
        asyncio.run(OllamaClient(_cfg(tmp_path)).generate("m", "p"))


def test_status_reachable_lists_models(tmp_path, monkeypatch):
    _patch(monkeypatch, lambda req: httpx.Response(
        200, json={"models": [{"name": "qwen3:4b"}, {"name": "qwen3:1.7b"}]}))
    st = asyncio.run(OllamaClient(_cfg(tmp_path)).status())
    assert st["reachable"] is True
    assert "qwen3:4b" in st["models"]


def test_status_unreachable(tmp_path, monkeypatch):
    def handler(req):
        raise httpx.ConnectError("down")

    _patch(monkeypatch, handler)
    st = asyncio.run(OllamaClient(_cfg(tmp_path)).status())
    assert st["reachable"] is False and st["models"] == []
