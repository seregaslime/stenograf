"""Объединение спикеров ПОСРЕДИ идущей встречи.

Окно разговора уехало в приложение вместе с подсказками, поэтому проверки про
переименование строк «Имя: текст» отсюда удалены — их место теперь в клиенте.
Осталось то, что относится к диаризации: донор коротких реплик, недавний
говоривший и счётчик участников.

Слияние приходит по REST (страница «Спикеры» открыта окном поверх встречи), а
живая сессия держит id спикеров в памяти. Профиль-источник при этом удаляется из
базы, и без уведомления сессия продолжала бы им пользоваться.

Опасность тихая: PRAGMA foreign_keys в SQLite по умолчанию выключен и в проекте
не включается, поэтому реплика, приписанная удалённому спикеру, не упала бы с
ошибкой — она молча записалась бы ссылкой в никуда и не нашлась бы ни у одного
участника.
"""
import pytest

from app.diarization.registry import MatchResult
from app.ws import LiveSession, notify_speakers_merged


def _session(cfg):
    s = LiveSession(
        ws=None, cfg=cfg, transcriber=None, embedder=None,
        registry=None, llm=None, on_meeting_ended=lambda mid: None,
    )
    s._meeting_id = 1
    return s


def _match(speaker_id: int, name: str) -> MatchResult:
    return MatchResult(speaker_id, name, False, 0.9, False)


@pytest.fixture()
def session(cfg):
    s = _session(cfg)
    s._participants.update({3: 12, 5: 4})
    s._recent_speakers.update({3: 10.0, 5: 20.0})
    s._last_by_channel["mic"] = (_match(3, "Спикер 3"), 20.0)
    return s


def test_short_replica_goes_to_the_surviving_speaker(session):
    """Донор коротких реплик перестаёт указывать на удалённый профиль.

    Иначе следующее «ага» уехало бы на спикера, которого больше нет.
    """
    session.on_speakers_merged(3, 5, "Иван", ["Спикер 3", "Спикер 5"])

    donor = session._short_segment_donor("mic", 21.0)
    assert donor is not None
    assert donor.speaker_id == 5 and donor.name == "Иван"


def test_participants_do_not_split_in_two(session):
    """Счётчик реплик складывается: иначе модель увидит лишнего участника."""
    session.on_speakers_merged(3, 5, "Иван", ["Спикер 3", "Спикер 5"])

    assert 3 not in session._participants
    assert session._participants[5] == 16


def test_recent_prior_forgets_the_deleted_speaker(session):
    """Приор «кто говорил недавно» тоже чистится — id уже не существует."""
    session.on_speakers_merged(3, 5, "Иван", ["Спикер 3", "Спикер 5"])

    assert 3 not in session._recent_speakers


def test_merging_a_speaker_with_itself_changes_nothing(session):
    """Защита от вырожденного случая: считать 12 реплик дважды нельзя."""
    session.on_speakers_merged(5, 5, "Иван", [])

    assert session._participants[5] == 4


def test_notify_reaches_only_running_meetings(cfg):
    """Рассылка идёт по живым сессиям; завершённая её не получает."""
    from app import ws as ws_mod

    live, finished = _session(cfg), _session(cfg)
    live._participants[3] = 2
    finished._participants[3] = 2
    ws_mod._LIVE_SESSIONS.add(live)
    try:
        notify_speakers_merged(3, 5, "Иван", [])
    finally:
        ws_mod._LIVE_SESSIONS.discard(live)

    assert live._participants[5] == 2 and 3 not in live._participants
    assert finished._participants[3] == 2  # её никто не трогал


