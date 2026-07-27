"""Юнит-тесты форматирования транскрипта для резюме (llm/summary.py)."""
from types import SimpleNamespace

from app.llm.summary import MAX_TRANSCRIPT_CHARS, _mmss, build_transcript


def _seg(name, start_s, text):
    speaker = SimpleNamespace(name=name) if name else None
    return SimpleNamespace(speaker=speaker, start_s=start_s, text=text)


def test_mmss():
    assert _mmss(0) == "00:00"
    assert _mmss(65) == "01:05"
    assert _mmss(3599) == "59:59"


def test_build_transcript_lines_and_participants():
    segs = [_seg("Иван", 0, "привет"), _seg("Мария", 61, "да"), _seg("Иван", 120, "ок")]
    transcript, participants = build_transcript(segs)
    assert "[00:00] Иван: привет" in transcript
    assert "[01:01] Мария: да" in transcript
    assert participants.startswith("Иван (2 реплик)")  # most_common первым
    assert "Мария (1 реплик)" in participants


def test_build_transcript_unknown_speaker():
    transcript, _ = build_transcript([_seg(None, 0, "аноним")])
    assert "Неизвестный: аноним" in transcript


def test_build_transcript_truncates_long():
    segs = [_seg("Иван", i, "x" * 500) for i in range(100)]  # заведомо > лимита
    transcript, _ = build_transcript(segs)
    assert len(transcript) <= MAX_TRANSCRIPT_CHARS + 200
    assert "часть транскрипта опущена" in transcript
