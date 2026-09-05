"""Сборка транскрипта встречи для выгрузки.

Жила в llm/summary.py, потому что тем же текстом кормили модель. Модели уехали
в приложение, а выгрузка осталась: человек скачивает полную расшифровку, и
резать её по бюджету контекста здесь нечем и незачем.
"""
from collections import Counter


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def build_transcript(segments) -> tuple[str, str]:
    """Возвращает (текст транскрипта, строку со статистикой участников)."""
    lines = []
    counter: Counter[str] = Counter()
    for segment in segments:
        name = segment.speaker.name if segment.speaker else "Неизвестный"
        counter[name] += 1
        lines.append(f"[{_mmss(segment.start_s)}] {name}: {segment.text}")
    participants = ", ".join(f"{name} ({n} реплик)" for name, n in counter.most_common())
    return "\n".join(lines), participants
