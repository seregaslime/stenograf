"""Вопрос модели во время встречи не должен останавливать чтение сокета.

Регрессия, пойманная руками 04.09.2026 на живой встрече: обработчик вопроса
вызывался прямо в цикле чтения, и пока локальная модель думала 45 секунд,
сервер не читал сокет вообще. Клиент всё это время слал аудио встречи, буфер
отправки переполнялся, и браузерный движок рвал соединение — человек посреди
встречи получал «соединение с сервером прервано».

С внешним API это не всплывало: он отвечает потоком, и сокет не простаивает.
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.llm.router import LlmRouter

ЗАДЕРЖКА_МОДЕЛИ_С = 1.0


@pytest.fixture()
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture()
def медленная_модель(monkeypatch):
    """Модель, которая думает долго и отвечает одним куском — как локальная."""
    async def hint(self, prompt, system=None, temperature=0.5, on_delta=None):
        await asyncio.sleep(ЗАДЕРЖКА_МОДЕЛИ_С)
        return "Сроки обсуждали в прошлый вторник."

    monkeypatch.setattr(LlmRouter, "hint", hint)


def _начать_встречу(ws) -> None:
    ws.send_json({"type": "start", "title": "Проверка вопроса", "record_audio": False,
                  "hints": False, "summarize": False, "meeting_mode": "work"})
    assert ws.receive_json()["type"] == "ready"


def test_пока_модель_думает_сервер_читает_сокет(client, медленная_модель):
    """Ключевая проверка: команда, отправленная СЛЕДОМ за вопросом, получает
    ответ раньше, чем сам вопрос. Значит цикл чтения не заблокирован."""
    with client.websocket_connect("/ws/live") as ws:
        _начать_встречу(ws)
        ws.send_json({"type": "ask", "question": "Какие сроки?", "segment_ids": []})
        ws.send_json({"type": "не-такая-команда"})

        первое = ws.receive_json()
        assert первое["type"] == "error", "цикл чтения стоит, пока модель думает"
        assert "Неизвестная команда" in первое["message"]

        второе = ws.receive_json()
        assert второе["type"] == "answer"
        assert второе["text"] == "Сроки обсуждали в прошлый вторник."
        ws.send_json({"type": "stop"})


def test_аудио_принимается_во_время_ответа(client, медленная_модель):
    """То же самое, но кадрами аудио — именно они переполняли буфер."""
    with client.websocket_connect("/ws/live") as ws:
        _начать_встречу(ws)
        ws.send_json({"type": "ask", "question": "Что решили?", "segment_ids": []})
        # Полсекунды звука с обоих каналов, пока модель думает
        for _ in range(25):
            ws.send_bytes(b"\x00" + b"\x00" * 640)
            ws.send_bytes(b"\x01" + b"\x00" * 640)
        assert ws.receive_json()["type"] == "answer"
        ws.send_json({"type": "stop"})


def test_второй_вопрос_получает_отказ_а_не_вторую_генерацию(client, медленная_модель):
    """Задачи запускаются параллельно, поэтому защита «модель ещё отвечает»
    обязана срабатывать до первого await — иначе оба вопроса уйдут к модели."""
    with client.websocket_connect("/ws/live") as ws:
        _начать_встречу(ws)
        ws.send_json({"type": "ask", "question": "Первый", "segment_ids": []})
        ws.send_json({"type": "ask", "question": "Второй", "segment_ids": []})

        события = [ws.receive_json(), ws.receive_json()]
        типы = [с["type"] for с in события]
        assert типы.count("answer_error") == 1, типы
        assert типы.count("answer") == 1, типы
        отказ = next(с for с in события if с["type"] == "answer_error")
        assert "ещё отвечает" in отказ["message"]
        ws.send_json({"type": "stop"})
