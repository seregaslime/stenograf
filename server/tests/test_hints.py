"""Движок подсказок во время встречи: чистые функции (дедуп, решение триггера)
и поведение _emit_hint (отправка, дедуп, бэкофф при ошибках, «подсказать сейчас»).
Без сети и без реального LLM — через фейковый роутер.
"""
import asyncio

from app.llm.base import LlmError
from app.ws import LiveSession


def _session(cfg, llm=None):
    s = LiveSession(
        ws=None, cfg=cfg, transcriber=None, embedder=None,
        registry=None, llm=llm, on_meeting_ended=lambda mid: None,
    )
    s._sent = []

    async def _capture(payload):
        s._sent.append(payload)

    s._send = _capture  # перехватываем исходящие события вместо WebSocket
    return s


class _FakeLLM:
    def __init__(self, reply="Подсказка", fail=False):
        self.reply = reply
        self.fail = fail
        self.calls = 0

    async def hint(self, prompt, system=None, temperature=0.5):
        self.calls += 1
        if self.fail:
            raise LlmError("нет связи с LLM")
        return self.reply


def _fill_context(s):
    s._recent.extend(f"Спикер: реплика номер {i} с достаточным объёмом текста" for i in range(6))


# ------------------------------------------------------------------ чистые функции

def test_is_duplicate_catches_near_repeat(cfg):
    s = _session(cfg)
    s._recent_hints.append("Уточните сроки запуска беты и критерии готовности.")
    assert s._is_duplicate("уточните Сроки запуска беты и критерии готовности!")
    assert not s._is_duplicate("Кто отвечает за тестирование авторизации?")


def test_is_duplicate_empty_memory(cfg):
    assert not _session(cfg)._is_duplicate("Любая подсказка")


def test_should_hint_needs_enough_new_text(cfg):
    s = _session(cfg)
    s._last_hint_at = 1000.0 - cfg.hints_min_gap_s
    s._chars_since_hint = cfg.hints_min_new_chars - 1
    assert not s._should_hint(1000.0)
    s._chars_since_hint = cfg.hints_min_new_chars
    assert s._should_hint(1000.0)


def test_should_hint_respects_min_gap(cfg):
    s = _session(cfg)
    s._chars_since_hint = cfg.hints_min_new_chars * 5
    s._last_hint_at = 1000.0 - cfg.hints_min_gap_s / 2
    assert not s._should_hint(1000.0)


def test_should_hint_respects_backoff(cfg):
    s = _session(cfg)
    s._chars_since_hint = cfg.hints_min_new_chars * 5
    s._last_hint_at = 0.0
    s._hint_backoff_until = 2000.0
    assert not s._should_hint(1000.0)
    assert s._should_hint(2000.0)


# ------------------------------------------------------------------ поведение _emit_hint

def test_emit_hint_sends_then_dedups(cfg):
    llm = _FakeLLM("Уточните сроки и ответственных.")
    s = _session(cfg, llm)
    _fill_context(s)
    asyncio.run(s._emit_hint())
    asyncio.run(s._emit_hint())  # тот же ответ — второй раз не шлём
    hints = [m for m in s._sent if m["type"] == "hint"]
    assert len(hints) == 1
    assert llm.calls == 2  # LLM звали дважды, дубль отсеян на клиентской стороне


def test_emit_hint_error_is_not_permanent(cfg):
    s = _session(cfg, _FakeLLM(fail=True))
    _fill_context(s)
    s._hints_enabled = True
    asyncio.run(s._emit_hint())
    assert s._hint_fail_streak == 1
    assert s._hint_backoff_until > 0
    assert s._hints_enabled is True  # одна ошибка НЕ выключает подсказки
    assert sum(m["type"] == "hint_error" for m in s._sent) == 1


def test_emit_hint_disables_after_max_fails(cfg):
    s = _session(cfg, _FakeLLM(fail=True))
    _fill_context(s)
    s._hints_enabled = True
    for _ in range(cfg.hints_max_fails):
        asyncio.run(s._emit_hint())
    assert s._hints_enabled is False


def test_hint_now_forces_even_when_disabled(cfg):
    s = _session(cfg, _FakeLLM("Мгновенная подсказка."))
    _fill_context(s)
    s._hints_enabled = False
    asyncio.run(s._on_command({"type": "hint_now"}))
    assert any(m["type"] == "hint" for m in s._sent)
