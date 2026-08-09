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

    async def hint(self, prompt, system=None, temperature=0.5, on_delta=None):
        self.seen.append((system or "", prompt))
        if self.fail:
            raise LlmError("нет связи с LLM")
        if on_delta is not None:
            # как настоящий поток: сначала мысли, потом ответ по кускам
            await on_delta("reasoning", "думаю")
            for кусок in self.reply.split(" "):
                await on_delta("text", кусок + " ")
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
    """Пока модель отвечает на вопрос, второй вопрос не запускаем — иначе два
    запроса в лимит токенов в минуту вместо одного."""
    meeting_id, _ = _meeting(["реплика"])
    llm = _FakeLLM()
    s = _session(cfg, llm, meeting_id)
    s._explicit_in_flight = True

    asyncio.run(s._answer(question="вопрос", segment_ids=[]))

    assert not llm.seen
    assert "ещё отвечает" in s._sent[-1]["message"]


def test_background_hint_does_not_block_the_question(cfg):
    """Фоновая подсказка вопрос не отбивает — она необязательна, а он нет.

    Раньше флаг занятости был один на подсказки и вопросы, и подсказка, ждущая
    восстановления минутного лимита (десятки секунд), отвечала человеку «Модель
    ещё отвечает на прошлый вопрос…» — про вопрос, которого он не задавал.
    """
    meeting_id, _ = _meeting(["реплика"])
    llm = _FakeLLM()
    s = _session(cfg, llm, meeting_id)
    s._hint_in_flight = True  # занято фоновой подсказкой, а не человеком

    asyncio.run(s._answer(question="вопрос", segment_ids=[]))

    assert llm.seen  # вопрос дошёл до модели
    assert s._sent[-1]["type"] == "answer"


def test_garbage_ids_do_not_crash(cfg):
    """Клиент может прислать что угодно — до запроса в БД доходят только числа."""
    meeting_id, ids = _meeting(["нормальная реплика"])
    llm = _FakeLLM()
    s = _session(cfg, llm, meeting_id)

    asyncio.run(s._answer(question="что", segment_ids=["abc", None, {}, ids[0]]))

    _, prompt = llm.seen[-1]
    assert "нормальная реплика" in prompt


def test_answer_is_printed_as_it_goes(cfg):
    """Ответ на вопрос печатается по кускам, а мысли идут отдельным событием.

    Провайдер шлёт их разными полями (content и reasoning), и смешивать нельзя:
    ответ показывается как ответ, рассуждения — под спойлером.
    """
    meeting_id, _ = _meeting(["реплика"])
    llm = _FakeLLM("SLA — соглашение об уровне сервиса.")
    s = _session(cfg, llm, meeting_id)

    asyncio.run(s._answer(question="что такое SLA", segment_ids=[]))

    куски = [m for m in s._sent if m["type"] == "answer_delta"]
    мысли = [m for m in s._sent if m["type"] == "answer_reasoning"]
    assert "".join(m["text"] for m in куски).strip() == llm.reply
    assert мысли, "мысли модели провайдер отдаёт даром — выбрасывать их незачем"
    assert s._sent[-1] == {"type": "answer", "text": llm.reply}  # итог приходит последним


def test_selection_without_question_asks_to_explain_the_content(cfg):
    """Показал на реплику без вопроса — просим разобрать сказанное.

    Живой случай: участник выделил реплику с термином, а модель ответила, что
    этот вопрос задали для проверки связи. Формально верно — мы просили
    объяснить, «о чём эти реплики», то есть чем они являются в разговоре. А
    человек выделяет то, чего не понял, и ждёт разбора.
    """
    meeting_id, ids = _meeting(["Скажите, что такое ВВП?"])
    llm = _FakeLLM()
    s = _session(cfg, llm, meeting_id)

    asyncio.run(s._answer(question="", segment_ids=[ids[0]]))

    _, prompt = llm.seen[-1]
    assert "объясни термины" in prompt.lower()
    assert "ответь на него" in prompt.lower()   # прозвучавший вопрос не игнорируем
    assert "о чём эти реплики" not in prompt.lower()


def test_typed_question_is_not_replaced(cfg):
    """Свой вопрос участника подменять нечем — он и есть тема."""
    meeting_id, ids = _meeting(["обсудим SLA"])
    llm = _FakeLLM()
    s = _session(cfg, llm, meeting_id)

    asyncio.run(s._answer(question="кто за это отвечает", segment_ids=[ids[0]]))

    _, prompt = llm.seen[-1]
    assert "кто за это отвечает" in prompt
    assert "объясни термины" not in prompt.lower()
