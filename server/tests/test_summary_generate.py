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

    # Роутер отдаёт их summary.py для расчёта размера фрагмента и пауз
    chars_per_token = 2.5
    rate_pause_s = 60.0

    @property
    def summary_model_name(self) -> str:
        return "fake-model"

    async def summarize(self, prompt, system=None, temperature=0.3):
        self.seen.append((prompt, system or ""))
        if self.fail:
            raise LlmError("Ollama недоступен")
        return self.reply


def _raising(exc):
    """Подменяет summarize так, чтобы он падал заданным исключением."""
    async def _summarize(prompt, system=None, temperature=0.3):
        raise exc
    return _summarize


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


def test_unexpected_error_also_closes_the_meeting():
    """Любое исключение, а не только LlmError, закрывает встречу.

    Раньше ловился один LlmError, поэтому таймаут по VPN или обрыв связи убивали
    фоновую задачу молча: встреча навсегда оставалась "summarizing", а клиент
    крутил спиннер, опрашивая её каждые 4 секунды.
    """
    llm = _FakeLlm()
    llm.summarize = _raising(TimeoutError("Соединение с API истекло"))
    meeting_id = _meeting()

    asyncio.run(generate_summary(llm, meeting_id))

    status, summary, model, error = _stored(meeting_id)
    assert status == "done"
    assert summary is None and model is None
    assert "TimeoutError" in error  # тип сбоя виден
    # Текст исключения наружу не отдаём: в сообщениях httpx попадается адрес API,
    # а summary_error уходит на клиент и показывается в интерфейсе.
    assert "Соединение с API истекло" not in error


def test_cancellation_leaves_status_to_the_replacing_task():
    """Отмена статус не трогает и пробрасывается дальше.

    Отменяют задачу только из _schedule_summary, и сразу за отменой стартует
    новая. Поставь мы здесь "done" — затёрли бы "summarizing" уже запущенной
    замены, и клиент перестал бы опрашивать посреди живой генерации.
    """
    llm = _FakeLlm()
    llm.summarize = _raising(asyncio.CancelledError())
    meeting_id = _meeting()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(generate_summary(llm, meeting_id))

    status, _, _, error = _stored(meeting_id)
    assert status == "summarizing"  # ждём новую задачу, а не закрываемся
    assert error is None


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


# ------------------------------------------------------------------ длинная встреча

def test_split_by_lines_keeps_replicas_whole():
    """Режем по границам реплик: разорванная пополам фраза бессмысленна."""
    from app.llm.summary import split_by_lines

    transcript = "\n".join(f"[00:0{i}] Сергей: реплика номер {i}" for i in range(6))
    chunks = split_by_lines(transcript, 60)

    assert len(chunks) > 1
    assert "\n".join(chunks) == transcript          # ничего не потеряли
    for chunk in chunks:
        for line in chunk.split("\n"):
            assert line.startswith("[00:0")          # каждая строка — целая реплика


def test_split_keeps_overlong_replica_in_its_own_chunk():
    """Реплика длиннее лимита рвать некуда — уходит своим куском целиком."""
    from app.llm.summary import split_by_lines

    long_line = "[00:00] Сергей: " + "очень длинная фраза " * 20
    chunks = split_by_lines(long_line + "\n[00:30] Куратор: коротко", 100)

    assert chunks[0] == long_line


def test_long_meeting_goes_in_several_requests(monkeypatch):
    """Транскрипт больше бюджета — заметки по фрагментам, потом сведение.

    Проверяем и что пауза между запросами выдерживается: лимит провайдера
    считается за минуту, без паузы фрагменты упрутся в него так же, как один
    большой запрос.
    """
    slept = []

    async def _no_wait(seconds):
        slept.append(seconds)

    monkeypatch.setattr("app.llm.summary.asyncio.sleep", _no_wait)

    llm = _FakeLlm(summary_chars=0)
    llm._budget = Budget(0, 2500, True, 0, summary_tokens=900)  # ~2250 символов на запрос
    meeting_id = _meeting(texts=[f"довольно длинная реплика номер {i} про релиз" for i in range(90)])

    asyncio.run(generate_summary(llm, meeting_id))

    status, summary, _, error = _stored(meeting_id)
    assert status == "done" and error is None and summary
    assert len(llm.seen) > 2                    # несколько фрагментов + сведение
    assert slept and all(s == 60.0 for s in slept)
    assert "Фрагмент" in llm.seen[-1][0]        # последний запрос сводит заметки


def test_short_meeting_still_goes_in_one_request():
    """Влезающая встреча идёт как раньше — одним запросом, без пауз."""
    llm = _FakeLlm()
    llm._budget = Budget(0, 2500, True, 0, summary_tokens=8000)
    meeting_id = _meeting()

    asyncio.run(generate_summary(llm, meeting_id))

    assert len(llm.seen) == 1


def test_no_tariff_limit_means_no_splitting():
    """У локальной модели тарифного лимита нет — деление не включается."""
    llm = _FakeLlm(summary_chars=0)
    meeting_id = _meeting(texts=[f"реплика {i} с текстом" for i in range(200)])

    asyncio.run(generate_summary(llm, meeting_id))

    assert len(llm.seen) == 1
