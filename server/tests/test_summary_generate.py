"""Составление протокола встречи (llm/summary.generate_summary).

Это ядро фичи «протокол по завершении», и до сих пор оно проверялось только
косвенно — через e2e. Здесь LLM подменена фейком, поэтому тесты быстрые и
проверяют именно логику: что уходит в модель, что сохраняется в БД и что
происходит при ошибках.
"""
import asyncio

import pytest

from app.db import crud
from app.db.database import init_db, session_scope
from app.db.models import Meeting
from app.llm.base import LlmError
from app.llm.router import Budget
from app.llm.summary import generate_summary


@pytest.fixture(scope="module", autouse=True)
def _schema():
    """generate_summary работает с общей БД (не с фикстурой db_session), поэтому
    таблицы нужно создать явно — в приложении это делает lifespan."""
    init_db()


class _FakeLlm:
    """Подставной роутер: помнит, с чем его позвали."""

    def __init__(self, reply="## Краткий итог\nОбсудили релиз.", fail=False,
                 detailed=False, summary_chars=12_000):
        self.reply = reply
        self.fail = fail
        self.seen: list[tuple[str, str]] = []  # (prompt, system)
        self._budget = Budget(summary_chars, 2500, detailed)

    @property
    def budget(self) -> Budget:
        return self._budget

    @property
    def summary_model_name(self) -> str:
        return "fake-model"

    async def summarize(self, prompt, system=None, temperature=0.3):
        self.seen.append((prompt, system or ""))
        if self.fail:
            raise LlmError("Ollama недоступен")
        return self.reply


def _meeting(mode="work", texts=("привет коллеги", "обсудим релиз")):
    with session_scope() as db:
        m = crud.create_meeting(db, "Планёрка", False, mode)
        speaker = crud.get_or_create_self_speaker(db)
        for i, text in enumerate(texts):
            crud.add_segment(db, m.id, speaker.id, "mic", float(i), i + 1.0, text)
        crud.end_meeting(db, m.id, status="summarizing")
        return m.id


def _stored(meeting_id):
    with session_scope() as db:
        m = db.get(Meeting, meeting_id)
        return m.status, m.summary, m.summary_model, m.summary_error


def test_summary_saved_with_model_name():
    """Успешный протокол сохраняется вместе с именем модели, встреча закрывается."""
    llm = _FakeLlm()
    meeting_id = _meeting()
    asyncio.run(generate_summary(llm, meeting_id))

    status, summary, model, error = _stored(meeting_id)
    assert status == "done"
    assert summary == "## Краткий итог\nОбсудили релиз."
    assert model == "fake-model" and error is None


def test_transcript_and_participants_reach_the_model():
    """В промпт уходят реплики с таймкодами и статистика участников."""
    llm = _FakeLlm()
    asyncio.run(generate_summary(llm, _meeting()))

    prompt, system = llm.seen[0]
    assert "[00:00] Вы: привет коллеги" in prompt
    assert "Вы (2 реплик)" in prompt
    assert "русском" in system  # протокол требуем по-русски


def test_meeting_mode_changes_sections():
    """Тип встречи доезжает из БД до промпта: у собеседования свои разделы."""
    llm = _FakeLlm()
    asyncio.run(generate_summary(llm, _meeting(mode="interview")))

    prompt, _ = llm.seen[0]
    assert "Что стоит подтянуть" in prompt
    assert "Принятые решения" not in prompt


def test_api_budget_asks_for_more_detail():
    """У API контекст большой — просим таймкоды и раздел открытых вопросов."""
    llm = _FakeLlm(detailed=True)
    asyncio.run(generate_summary(llm, _meeting()))

    prompt, system = llm.seen[0]
    assert "таймкод" in system
    assert "Открытые вопросы" in prompt


def test_long_transcript_truncated_by_budget():
    """Маленький бюджет режет транскрипт, а не отправляет всё подряд."""
    llm = _FakeLlm(summary_chars=400)
    meeting_id = _meeting(texts=[f"реплика номер {i} " + "текст " * 20 for i in range(60)])
    asyncio.run(generate_summary(llm, meeting_id))

    prompt, _ = llm.seen[0]
    assert "часть транскрипта опущена" in prompt


def test_llm_error_saved_as_message_not_crash():
    """Недоступная LLM не роняет задачу: встреча закрывается с понятной ошибкой."""
    llm = _FakeLlm(fail=True)
    meeting_id = _meeting()
    asyncio.run(generate_summary(llm, meeting_id))

    status, summary, model, error = _stored(meeting_id)
    assert status == "done"
    assert summary is None and model is None
    assert "Ollama недоступен" in error


def test_meeting_without_speech_is_not_sent_to_llm():
    """Встречу без распознанной речи в модель не отправляем — сразу ошибка."""
    llm = _FakeLlm()
    with session_scope() as db:
        meeting_id = crud.create_meeting(db, "Тишина", False).id

    asyncio.run(generate_summary(llm, meeting_id))

    status, summary, _, error = _stored(meeting_id)
    assert status == "done" and summary is None
    assert "не содержит распознанной речи" in error
    assert not llm.seen  # LLM вообще не звали


def test_missing_meeting_is_ignored():
    """Встречу удалили, пока задача ждала очереди — тихо выходим, без исключения."""
    llm = _FakeLlm()
    asyncio.run(generate_summary(llm, 999_999))
    assert not llm.seen
