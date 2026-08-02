"""Юнит-тесты форматирования транскрипта для резюме (llm/summary.py)."""
from types import SimpleNamespace

from app.llm.summary import MAX_TRANSCRIPT_CHARS, _mmss, build_transcript


def _seg(name, start_s, text):
    speaker = SimpleNamespace(name=name) if name else None
    return SimpleNamespace(speaker=speaker, start_s=start_s, text=text)


def test_mmss():
    """Секунды форматируются в MM:SS для меток времени в транскрипте."""
    assert _mmss(0) == "00:00"
    assert _mmss(65) == "01:05"
    assert _mmss(3599) == "59:59"


def test_build_transcript_lines_and_participants():
    """Каждая реплика идёт строкой «[MM:SS] Имя: текст», участники считаются по числу реплик."""
    segs = [_seg("Иван", 0, "привет"), _seg("Мария", 61, "да"), _seg("Иван", 120, "ок")]
    transcript, participants = build_transcript(segs)
    assert "[00:00] Иван: привет" in transcript
    assert "[01:01] Мария: да" in transcript
    assert participants.startswith("Иван (2 реплик)")  # most_common первым
    assert "Мария (1 реплик)" in participants


def test_build_transcript_unknown_speaker():
    """Реплика без опознанного спикера подписывается «Неизвестный»."""
    transcript, _ = build_transcript([_seg(None, 0, "аноним")])
    assert "Неизвестный: аноним" in transcript


def test_build_transcript_truncates_long():
    """Слишком длинный транскрипт ужимается: остаются начало и хвост, середина помечается пропуском.
    """
    segs = [_seg("Иван", i, "x" * 500) for i in range(100)]  # заведомо > лимита
    transcript, _ = build_transcript(segs)
    assert len(transcript) <= MAX_TRANSCRIPT_CHARS + 200
    assert "часть транскрипта опущена" in transcript


def _long_segments(n=200):
    return [_seg("Иван", i, f"реплика номер {i} " + "текст " * 20) for i in range(n)]


def test_build_transcript_without_limit_keeps_everything():
    """max_chars=0 — экспорт и API-провайдер получают полную расшифровку."""
    transcript, _ = build_transcript(_long_segments(), 0)
    assert "опущена из-за длины" not in transcript
    assert "реплика номер 0 " in transcript and "реплика номер 199 " in transcript


def test_build_transcript_honours_custom_limit():
    """Лимит длины берётся из аргумента — так провайдеры получают разный бюджет контекста."""
    transcript, _ = build_transcript(_long_segments(), 1000)
    assert "опущена из-за длины" in transcript
    assert len(transcript) < 1200


def test_build_transcript_default_limit_unchanged():
    """Дефолт остался прежним — поведение локального провайдера не изменилось."""
    transcript, _ = build_transcript(_long_segments(400))
    assert len(transcript) <= MAX_TRANSCRIPT_CHARS + 100
