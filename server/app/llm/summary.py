"""Итог встречи от локальной LLM: краткий итог → темы → решения → задачи."""
import logging
from collections import Counter

from ..db import crud
from ..db.database import session_scope
from ..db.models import Meeting
from . import prompts
from .base import LlmError
from .router import LlmRouter

log = logging.getLogger(__name__)

# qwen3:4b с контекстом 8k токенов: ~12000 символов русского текста влезает с запасом
MAX_TRANSCRIPT_CHARS = 12_000


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def build_transcript(segments) -> tuple[str, str]:
    """Возвращает (текст транскрипта, строка со статистикой участников)."""
    lines = []
    counter: Counter[str] = Counter()
    for segment in segments:
        name = segment.speaker.name if segment.speaker else "Неизвестный"
        counter[name] += 1
        lines.append(f"[{_mmss(segment.start_s)}] {name}: {segment.text}")
    transcript = "\n".join(lines)
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        head = transcript[: MAX_TRANSCRIPT_CHARS // 4]
        tail = transcript[-(MAX_TRANSCRIPT_CHARS - len(head)):]
        transcript = head + "\n[... часть транскрипта опущена из-за длины ...]\n" + tail
    participants = ", ".join(f"{name} ({n} реплик)" for name, n in counter.most_common())
    return transcript, participants


async def generate_summary(llm: LlmRouter, meeting_id: int) -> None:
    with session_scope() as db:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            return
        segments = crud.meeting_segments(db, meeting_id)
        title = meeting.title
        date = meeting.started_at.strftime("%d.%m.%Y %H:%M") if meeting.started_at else ""
        if not segments:
            meeting.status = "done"
            meeting.summary_error = "Встреча не содержит распознанной речи."
            return

    transcript, participants = build_transcript(segments)
    prompt = prompts.SUMMARY_TEMPLATE.format(
        title=title, date=date, participants=participants, transcript=transcript
    )
    try:
        summary = await llm.summarize(
            prompt, system=prompts.SUMMARY_SYSTEM, temperature=0.3
        )
        error = None
    except LlmError as exc:
        summary, error = None, str(exc)
        log.warning("Резюме встречи #%d не создано: %s", meeting_id, exc)

    with session_scope() as db:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            return
        meeting.status = "done"
        meeting.summary = summary
        meeting.summary_model = llm.summary_model_name if summary else None
        meeting.summary_error = error
    if summary:
        log.info("Резюме встречи #%d готово", meeting_id)
