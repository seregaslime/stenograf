"""Юнит-тесты операций с БД (db/crud.py) на настоящем PostgreSQL (фикстура db_session)."""
import pytest
from sqlalchemy.exc import IntegrityError

from app.db import crud
from app.db.models import Chunk, Document, Meeting, Speaker


def test_self_speaker_created_once(db_session):
    """Профиль владельца «Вы» создаётся один раз: повторный вызов возвращает тот же id, а не плодит дубли.
    """
    a = crud.get_or_create_self_speaker(db_session)
    assert a.is_self and a.name == "Вы"
    b = crud.get_or_create_self_speaker(db_session)
    assert b.id == a.id  # не дублируется


def test_create_speaker_default_name(db_session):
    """Новому спикеру даётся имя-заглушка «Спикер N» и он не помечается владельцем."""
    s = crud.create_speaker(db_session)
    assert s.name == f"Спикер {s.id}"
    assert s.is_self is False


def test_rename_speaker(db_session):
    """Переименование обрезает пробелы по краям и сохраняется в БД; для несуществующего id возвращается None.
    """
    s = crud.create_speaker(db_session)
    assert crud.rename_speaker(db_session, s.id, "  Иван  ").name == "Иван"  # обрезка
    crud.rename_speaker(db_session, s.id, "   ")  # пустое имя не затирает
    assert db_session.get(Speaker, s.id).name == "Иван"
    assert crud.rename_speaker(db_session, 9999, "X") is None


def test_create_and_end_meeting(db_session):
    """Встреча создаётся в статусе live с фолбэком названия; завершение проставляет время один раз и повторно его не сдвигает.
    """
    m = crud.create_meeting(db_session, title="   ", record_audio=True)
    assert m.title == "Встреча"  # фолбэк на пустом названии
    assert m.status == "live" and m.ended_at is None
    ended = crud.end_meeting(db_session, m.id, status="done")
    assert ended.status == "done" and ended.ended_at is not None
    first_end = ended.ended_at
    crud.end_meeting(db_session, m.id, status="summarizing")  # повторно — не меняет
    assert db_session.get(Meeting, m.id).ended_at == first_end
    assert db_session.get(Meeting, m.id).status == "done"
    assert crud.end_meeting(db_session, 9999) is None


def test_add_segment_and_list_meetings(db_session):
    """Список встреч отдаёт название, число реплик и признак наличия резюме."""
    m = crud.create_meeting(db_session, "Планёрка", False)
    s = crud.create_speaker(db_session)
    crud.add_segment(db_session, m.id, s.id, "mic", 0.0, 1.0, "привет", 0.9)
    crud.add_segment(db_session, m.id, s.id, "system", 1.0, 2.0, "мир")
    row = next(r for r in crud.list_meetings(db_session) if r["id"] == m.id)
    assert row["segments_count"] == 2
    assert row["has_summary"] is False
    assert row["title"] == "Планёрка"


def test_meeting_segments_ordered_and_dict(db_session):
    """Реплики возвращаются по возрастанию времени; в словаре близость округлена до 3 знаков, а спикер может быть None.
    """
    m = crud.create_meeting(db_session, "M", False)
    s = crud.get_or_create_self_speaker(db_session)
    crud.add_segment(db_session, m.id, s.id, "mic", 2.0, 3.0, "второй")
    crud.add_segment(db_session, m.id, s.id, "mic", 0.0, 1.0, "первый", 0.87654)
    segs = crud.meeting_segments(db_session, m.id)
    assert [x.text for x in segs] == ["первый", "второй"]  # по start_s
    d = crud.segment_to_dict(segs[0])
    assert d["similarity"] == 0.877  # округление до 3 знаков
    assert d["speaker"]["is_self"] is True
    anon = crud.add_segment(db_session, m.id, None, "mic", 5.0, 6.0, "аноним")
    assert crud.segment_to_dict(anon)["speaker"] is None


def test_reassign_segments(db_session):
    """Перенос реплик между спикерами возвращает число перенесённых; повторный перенос уже ничего не двигает.
    """
    m = crud.create_meeting(db_session, "M", False)
    a, b = crud.create_speaker(db_session), crud.create_speaker(db_session)
    crud.add_segment(db_session, m.id, a.id, "mic", 0.0, 1.0, "1")
    crud.add_segment(db_session, m.id, a.id, "mic", 1.0, 2.0, "2")
    assert crud.reassign_segments(db_session, a.id, b.id) == 2
    assert crud.reassign_segments(db_session, a.id, None) == 0  # уже перенесены
    assert all(x.speaker_id == b.id for x in crud.meeting_segments(db_session, m.id))


def test_list_speakers_counts_and_order(db_session):
    """Владелец «Вы» идёт первым в списке; у каждого спикера считаются реплики,
    встречи и число отпечатков голоса."""
    self_sp = crud.get_or_create_self_speaker(db_session)
    guest = crud.create_speaker(db_session)
    m = crud.create_meeting(db_session, "M", False)
    crud.add_segment(db_session, m.id, self_sp.id, "mic", 0.0, 1.0, "a")
    crud.add_segment(db_session, m.id, guest.id, "system", 1.0, 2.0, "b")
    rows = crud.list_speakers(db_session)
    assert rows[0]["is_self"] is True  # «Вы» первым
    by_id = {r["id"]: r for r in rows}
    assert by_id[self_sp.id]["segments_count"] == 1
    assert by_id[self_sp.id]["meetings_count"] == 1
    assert by_id[guest.id]["voiceprints_count"] == 0


def test_id_удалённого_профиля_не_достаётся_новому(db_session):
    """Последовательность id назад не отматывается.

    На этом держится разделение фоновых задач и файлов: пока по удалённому
    профилю ещё может дописывать что-то фоновая задача, новый профиль не должен
    получить его id и «унаследовать» чужие образцы голоса. В SQLite это
    приходилось просить отдельно (sqlite_autoincrement), PostgreSQL так делает
    сам — но проверяем, потому что держится оно на поведении СУБД, а не на
    нашем коде, и молча поменяться может только здесь.
    """
    первый = crud.create_speaker(db_session)
    db_session.flush()
    был = первый.id

    db_session.delete(первый)
    db_session.flush()

    второй = crud.create_speaker(db_session)
    db_session.flush()
    assert второй.id != был


# --- часть реплики другому спикеру ---

# «Да, согласен. Нет, погоди.» — пауза перед «Нет,» с 1.0 до 1.4 с
СЛОВА = [[0.2, 0.4, "Да,"], [0.5, 1.0, "согласен."], [1.4, 1.6, "Нет,"], [1.7, 2.1, "погоди."]]


def test_выделение_в_середине_даёт_три_куска():
    куски = crud.word_parts(СЛОВА, 0.0, 2.3, 1, 2)
    assert [(н, к, [с for _, _, с in слова], в) for н, к, слова, в in куски] == [
        (0.0, 0.45, ["Да,"], False),
        (0.45, 1.65, ["согласен.", "Нет,"], True),
        (1.65, 2.3, ["погоди."], False),
    ]


def test_граница_посередине_паузы_а_края_реплики_на_месте():
    """Разрез по паузе перед «Нет,» (1.0–1.4) — ровно посередине, а начало и
    конец реплики не сдвигаются к словам: запас VAD по краям остаётся."""
    [первый, второй] = crud.word_parts(СЛОВА, 0.0, 2.3, 2, 3)
    assert (первый[0], первый[1]) == (0.0, 1.2)
    assert (второй[0], второй[1]) == (1.2, 2.3)


def test_выделена_вся_реплика_кусок_один():
    [(начало, конец, слова, выделено)] = crud.word_parts(СЛОВА, 0.0, 2.3, 0, 3)
    assert (начало, конец, выделено) == (0.0, 2.3, True)
    assert слова == СЛОВА


def test_границы_не_идут_назад_при_наложении_слов():
    """Модель ставит время по кадрам, и конец слова бывает позже начала
    следующего. Граница тогда всё равно между краями, а куски не выворачиваются."""
    наложены = [[0.2, 0.9, "раз"], [0.6, 1.0, "два"], [0.95, 1.2, "три"]]
    куски = crud.word_parts(наложены, 0.0, 1.3, 1, 1)
    границы = [(н, к) for н, к, _, _ in куски]
    assert all(н <= к for н, к in границы)
    assert all(левый[1] == правый[0] for левый, правый in zip(границы, границы[1:]))


def test_реплика_делится_и_первый_кусок_остаётся_той_же_строкой(db_session):
    """Исходная строка — первый кусок: на её id ссылаются куски поиска."""
    встреча = crud.create_meeting(db_session, "деление", False)
    сатир, бабка = crud.create_speaker(db_session), crud.create_speaker(db_session)
    реплика = crud.add_segment(db_session, встреча.id, сатир.id, "system", 10.0, 12.3,
                               "Да, согласен. Нет, погоди.", 0.61,
                               words=[[10 + н, 10 + к, с] for н, к, с in СЛОВА])

    куски = crud.reassign_words(db_session, реплика, 2, 3, бабка.id)

    assert куски[0].id == реплика.id
    assert [(к.text, к.speaker_id, к.similarity) for к in куски] == [
        ("Да, согласен.", сатир.id, 0.61),
        ("Нет, погоди.", бабка.id, None),  # назначено рукой — мерить нечего
    ]
    assert [(к.start_s, к.end_s) for к in куски] == [(10.0, 11.2), (11.2, 12.3)]
    assert куски[1].speaker.name == бабка.name
    assert [s.id for s in crud.meeting_segments(db_session, встреча.id)] == [к.id for к in куски]


def test_выделение_в_начале_не_уносит_остаток_реплики(db_session):
    """Первый кусок — исходная строка, и ей спикер меняется первым. Нашлось в
    браузере: остаток реплики копировал уже нового спикера и уходил вместе с
    выделением, а тесты с выделением в середине и в конце этого не видели."""
    встреча = crud.create_meeting(db_session, "деление с начала", False)
    сатир, бабка = crud.create_speaker(db_session), crud.create_speaker(db_session)
    реплика = crud.add_segment(db_session, встреча.id, сатир.id, "system", 0.0, 2.3,
                               "Да, согласен. Нет, погоди.", 0.61, words=СЛОВА)

    куски = crud.reassign_words(db_session, реплика, 0, 1, бабка.id)

    assert [(к.text, к.speaker_id, к.similarity) for к in куски] == [
        ("Да, согласен.", бабка.id, None),
        ("Нет, погоди.", сатир.id, 0.61),
    ]


# --- куски поиска: из встречи или из документа ---

def _кусок(**источник) -> Chunk:
    return Chunk(text="кусок", model="bge-m3", vector=[1.0, 0.0, 0.0], **источник)


def test_у_куска_ровно_один_источник(db_session):
    """Кусок из встречи и документа сразу нашёлся бы в поиске дважды под разными
    подписями, а ничей — привёл бы в никуда. Держит это база, а не код."""
    встреча = crud.create_meeting(db_session, "Планёрка", False)
    документ = Document(title="Регламент", text="текст")
    db_session.add(документ)
    db_session.flush()

    for лишний in (_кусок(meeting_id=встреча.id, document_id=документ.id), _кусок()):
        with db_session.begin_nested():
            db_session.add(лишний)
            with pytest.raises(IntegrityError):
                db_session.flush()


def test_удаление_документа_уносит_только_его_куски(db_session):
    встреча = crud.create_meeting(db_session, "Планёрка", False)
    сегмент = crud.add_segment(db_session, встреча.id, None, "mic", 0, 1, "реплика")
    документ = Document(title="Регламент", text="текст")
    db_session.add(документ)
    db_session.flush()
    db_session.add_all([
        _кусок(meeting_id=встреча.id, first_segment_id=сегмент.id,
               last_segment_id=сегмент.id, start_s=0.0),
        _кусок(document_id=документ.id),
    ])
    db_session.flush()

    db_session.delete(документ)
    db_session.flush()
    db_session.expire_all()
    assert [к.meeting_id for к in db_session.query(Chunk)] == [встреча.id]
