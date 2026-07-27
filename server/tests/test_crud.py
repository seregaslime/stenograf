"""Юнит-тесты операций с БД (db/crud.py) на in-memory SQLite (фикстура db_session)."""
from app.db import crud
from app.db.models import Meeting, Speaker


def test_self_speaker_created_once(db_session):
    a = crud.get_or_create_self_speaker(db_session)
    assert a.is_self and a.name == "Вы"
    b = crud.get_or_create_self_speaker(db_session)
    assert b.id == a.id  # не дублируется


def test_create_speaker_default_name(db_session):
    s = crud.create_speaker(db_session)
    assert s.name == f"Спикер {s.id}"
    assert s.is_self is False


def test_rename_speaker(db_session):
    s = crud.create_speaker(db_session)
    assert crud.rename_speaker(db_session, s.id, "  Иван  ").name == "Иван"  # обрезка
    crud.rename_speaker(db_session, s.id, "   ")  # пустое имя не затирает
    assert db_session.get(Speaker, s.id).name == "Иван"
    assert crud.rename_speaker(db_session, 9999, "X") is None


def test_create_and_end_meeting(db_session):
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
    m = crud.create_meeting(db_session, "Планёрка", False)
    s = crud.create_speaker(db_session)
    crud.add_segment(db_session, m.id, s.id, "mic", 0.0, 1.0, "привет", 0.9)
    crud.add_segment(db_session, m.id, s.id, "system", 1.0, 2.0, "мир")
    row = next(r for r in crud.list_meetings(db_session) if r["id"] == m.id)
    assert row["segments_count"] == 2
    assert row["has_summary"] is False
    assert row["title"] == "Планёрка"


def test_meeting_segments_ordered_and_dict(db_session):
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
    m = crud.create_meeting(db_session, "M", False)
    a, b = crud.create_speaker(db_session), crud.create_speaker(db_session)
    crud.add_segment(db_session, m.id, a.id, "mic", 0.0, 1.0, "1")
    crud.add_segment(db_session, m.id, a.id, "mic", 1.0, 2.0, "2")
    assert crud.reassign_segments(db_session, a.id, b.id) == 2
    assert crud.reassign_segments(db_session, a.id, None) == 0  # уже перенесены
    assert all(x.speaker_id == b.id for x in crud.meeting_segments(db_session, m.id))


def test_list_speakers_counts_and_order(db_session):
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
