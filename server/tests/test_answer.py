"""Ответ на вопрос участника из окна чата (LiveSession._answer).

Отличается от подсказки принципиально: тему задаёт человек, ответ обязателен,
молчать нельзя. Проверяем сборку промпта, фильтр выделенных реплик по встрече
и поведение при ошибках. Без сети и без реального LLM.
"""
import asyncio

import pytest

from app.db import crud
from app.db.database import init_db, session_scope
from app.llm.base import LlmError
from app.llm.router import Budget
from app.ws import LiveSession


@pytest.fixture(scope="module", autouse=True)
def _schema():
    init_db()


class _FakeLLM:
    def __init__(self, reply="SLA — соглашение об уровне сервиса.", fail=False):
        self.reply = reply
        self.fail = fail
        self.seen: list[tuple[str, str]] = []

    @property
    def budget(self) -> Budget:
        return Budget(12_000, 2500, False)

    async def hint(self, prompt, system=None, temperature=0.5):
        self.seen.append((system or "", prompt))
        if self.fail:
            raise LlmError("нет связи с LLM")
        return self.reply


def _session(cfg, llm, meeting_id):
    s = LiveSession(
        ws=None, cfg=cfg, transcriber=None, embedder=None,
        registry=None, llm=llm, on_meeting_ended=lambda mid: None,
    )
    s._meeting_id = meeting_id
    s._meeting_title = "Планёрка"
    s._sent = []

    async def _capture(payload):
        s._sent.append(payload)

    s._send = _capture
    return s


def _meeting(texts):
    """Встреча с репликами; возвращает (meeting_id, [id реплик])."""
    with session_scope() as db:
        meeting = crud.create_meeting(db, "Планёрка", False)
        speaker = crud.get_or_create_self_speaker(db)
        ids = [
            crud.add_segment(db, meeting.id, speaker.id, "mic", float(i), i + 1.0, text).id
            for i, text in enumerate(texts)
        ]
        return meeting.id, ids


def test_quoted_segments_go_into_the_prompt(cfg):
    """Выделенные реплики попадают в промпт отдельным блоком «вопрос про них»."""
    meeting_id, ids = _meeting(["обсудим SLA по сервису", "а что такое SLA", "поехали дальше"])
    llm = _FakeLLM()
    s = _session(cfg, llm, meeting_id)

    asyncio.run(s._answer(question="объясни термин", segment_ids=[ids[1]]))

    _, prompt = llm.seen[-1]
    assert "показал именно на эти реплики" in prompt
    assert "а что такое SLA" in prompt
    assert "объясни термин" in prompt
    assert s._sent[-1] == {"type": "answer", "text": llm.reply}


def test_segments_from_another_meeting_are_ignored(cfg):
    """id приходят от клиента — чужая встреча через них вытянуться не должна.

    Без фильтра по meeting_id в crud.segments_by_ids произвольные числа отдали
    бы текст постороннего разговора прямо в промпт.
    """
    _, foreign_ids = _meeting(["секретное обсуждение зарплат"])
    meeting_id, _ = _meeting(["наша встреча про релиз"])
    llm = _FakeLLM()
    s = _session(cfg, llm, meeting_id)

    asyncio.run(s._answer(question="что это", segment_ids=foreign_ids))

    _, prompt = llm.seen[-1]
    assert "секретное обсуждение" not in prompt


def test_question_without_selection_still_works(cfg):
    """Спросить можно и без выделения — тогда контекстом идёт весь разговор."""
    meeting_id, _ = _meeting(["обсудили сроки релиза"])
    llm = _FakeLLM()
    s = _session(cfg, llm, meeting_id)
    s._recent.extend(["Сергей: обсудили сроки релиза на следующий квартал"])

    asyncio.run(s._answer(question="что такое кубернетес", segment_ids=[]))

    _, prompt = llm.seen[-1]
    assert "что такое кубернетес" in prompt
    assert "показал именно на эти реплики" not in prompt
    assert s._sent[-1]["type"] == "answer"


def test_empty_question_without_selection_is_rejected(cfg):
    """Пустой вопрос без выделения — спрашивать не о чем, LLM не зовём."""
    meeting_id, _ = _meeting(["что-то сказали"])
    llm = _FakeLLM()
    s = _session(cfg, llm, meeting_id)

    asyncio.run(s._answer(question="   ", segment_ids=[]))

    assert not llm.seen
    assert s._sent[-1]["type"] == "answer_error"


def test_llm_error_reaches_the_user(cfg):
    """Ошибка связи показывается сразу: человек спросил и ждёт ответа.

    Бэкофф фоновых подсказок здесь намеренно не участвует — глушить явные
    вопросы из-за неудач автоматического цикла нельзя.
    """
    meeting_id, _ = _meeting(["реплика"])
    llm = _FakeLLM(fail=True)
    s = _session(cfg, llm, meeting_id)

    asyncio.run(s._answer(question="почему", segment_ids=[]))

    assert s._sent[-1]["type"] == "answer_error"
    assert "нет связи" in s._sent[-1]["message"]


def test_second_question_while_answering_is_refused(cfg):
    """Пока модель отвечает, второй вопрос не запускаем — иначе два запроса
    в лимит токенов в минуту вместо одного."""
    meeting_id, _ = _meeting(["реплика"])
    llm = _FakeLLM()
    s = _session(cfg, llm, meeting_id)
    s._hint_in_flight = True

    asyncio.run(s._answer(question="вопрос", segment_ids=[]))

    assert not llm.seen
    assert "ещё отвечает" in s._sent[-1]["message"]


def test_garbage_ids_do_not_crash(cfg):
    """Клиент может прислать что угодно — до запроса в БД доходят только числа."""
    meeting_id, ids = _meeting(["нормальная реплика"])
    llm = _FakeLLM()
    s = _session(cfg, llm, meeting_id)

    asyncio.run(s._answer(question="что", segment_ids=["abc", None, {}, ids[0]]))

    _, prompt = llm.seen[-1]
    assert "нормальная реплика" in prompt
