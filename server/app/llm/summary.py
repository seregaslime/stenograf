"""Итог встречи от локальной LLM: краткий итог → темы → решения → задачи."""
import asyncio
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

# Меньше этого фрагменты не режем. На каждый уходит минута паузы, поэтому куски
# по паре тысяч символов превратили бы встречу на 40 минут в получасовое
# ожидание — молча. Лучше честно сказать, что тариф не тянет.
MIN_CHUNK_CHARS = 4_000


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60:02d}:{int(seconds) % 60:02d}"


def build_transcript(segments, max_chars: int = MAX_TRANSCRIPT_CHARS) -> tuple[str, str]:
    """Возвращает (текст транскрипта, строка со статистикой участников).

    max_chars=0 — не усекать (для API с большим контекстом и для экспорта, где
    выбрасывать середину нельзя: пользователь скачивает полную расшифровку).
    """
    lines = []
    counter: Counter[str] = Counter()
    for segment in segments:
        name = segment.speaker.name if segment.speaker else "Неизвестный"
        counter[name] += 1
        lines.append(f"[{_mmss(segment.start_s)}] {name}: {segment.text}")
    transcript = "\n".join(lines)
    if max_chars and len(transcript) > max_chars:
        head = transcript[: max_chars // 4]
        tail = transcript[-(max_chars - len(head)):]
        transcript = head + "\n[... часть транскрипта опущена из-за длины ...]\n" + tail
    participants = ", ".join(f"{name} ({n} реплик)" for name, n in counter.most_common())
    return transcript, participants


def split_by_lines(transcript: str, max_chars: int) -> list[str]:
    """Режет транскрипт на куски не длиннее max_chars — ПО ГРАНИЦАМ РЕПЛИК.

    Резать по символам нельзя: фраза разорвётся пополам, и обе половины станут
    бессмыслицей — а именно по ним модель и будет составлять заметки. Реплика
    длиннее лимита целиком уходит в свой кусок: рвать её всё равно некуда.
    """
    if max_chars <= 0:
        return [transcript]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in transcript.split("\n"):
        if current and size + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


async def _summarize_in_parts(
    llm: LlmRouter, chunks: list[str], *, mode: str, title: str, date: str,
    participants: str, detailed: bool, pause_s: float, on_progress,
) -> str:
    """Длинная встреча: заметки по каждому фрагменту → сведение в протокол.

    Пауза между запросами обязательна и не является перестраховкой: лимит
    провайдера считается за минуту, поэтому два куска подряд упрутся в него
    так же, как один большой запрос.

    Промежуточные заметки живут в памяти: при обрыве связи резюме
    пересоздаётся целиком. Хранить их в БД значило бы менять схему ради задачи
    на три минуты — оно того не стоит, но если встречи станут по три часа, к
    этому придётся вернуться.
    """
    notes = []
    for index, chunk in enumerate(chunks, start=1):
        if index > 1:
            await asyncio.sleep(pause_s)
        on_progress(index, len(chunks))
        system, prompt = prompts.build_chunk_prompt(
            mode=mode, title=title, part=index, total=len(chunks), transcript=chunk,
        )
        notes.append(f"— Фрагмент {index} —\n" + await llm.summarize(prompt, system=system))

    await asyncio.sleep(pause_s)
    on_progress(len(chunks) + 1, len(chunks) + 1)
    system, prompt = prompts.build_reduce_prompt(
        mode=mode, title=title, date=date, participants=participants,
        notes="\n\n".join(notes), detailed=detailed,
    )
    return await llm.summarize(prompt, system=system, temperature=0.3)


async def generate_summary(llm: LlmRouter, meeting_id: int, on_progress=None) -> None:
    with session_scope() as db:
        meeting = db.get(Meeting, meeting_id)
        if meeting is None:
            return
        segments = crud.meeting_segments(db, meeting_id)
        title = meeting.title
        mode = meeting.meeting_mode
        date = meeting.started_at.strftime("%d.%m.%Y %H:%M") if meeting.started_at else ""
        if not segments:
            meeting.status = "done"
            meeting.summary_error = "Встреча не содержит распознанной речи."
            return

    # Бюджет читается здесь, а не в __init__: провайдера могли переключить
    # уже после встречи (резюме перегенерируют кнопкой в истории).
    budget = llm.budget
    transcript, participants = build_transcript(segments, budget.summary_chars)
    system, prompt = prompts.build_summary_prompt(
        mode=mode, title=title, date=date,
        participants=participants, transcript=transcript,
        detailed=budget.detailed,
    )
    # Влезаем ли одним запросом. Считаем вместе с промптом: транскрипт, обрезанный
    # ровно по бюджету, уедет к провайдеру вместе с инструкциями и всё равно
    # получит 413 — на подсказках эта же ошибка уже была.
    chunks = [transcript]
    if budget.summary_tokens:
        limit_chars = int(budget.summary_tokens * llm.chars_per_token) - len(system) - len(prompt) + len(transcript)
        if len(transcript) > limit_chars:
            if limit_chars < MIN_CHUNK_CHARS:
                # Бюджет меньше самого промпта. Резать по крохам нельзя: на
                # каждый кусок уходит минута паузы, и встреча на 40 минут
                # превратилась бы в получасовое молчаливое ожидание.
                with session_scope() as db:
                    meeting = db.get(Meeting, meeting_id)
                    if meeting is not None:
                        meeting.status = "done"
                        meeting.summary_error = (
                            "Лимит тарифа слишком мал для протокола этой встречи. "
                            "Выберите модель с большим лимитом токенов в минуту."
                        )
                return
            chunks = split_by_lines(transcript, limit_chars)
            log.info("Встреча #%d: транскрипт %d символов — режем на %d фрагментов",
                     meeting_id, len(transcript), len(chunks))

    try:
        if len(chunks) > 1:
            summary = await _summarize_in_parts(
                llm, chunks, mode=mode, title=title, date=date,
                participants=participants, detailed=budget.detailed,
                pause_s=llm.rate_pause_s, on_progress=on_progress or (lambda *_: None),
            )
        else:
            summary = await llm.summarize(prompt, system=system, temperature=0.3)
        error = None
    except LlmError as exc:
        summary, error = None, str(exc)
        log.warning("Резюме встречи #%d не создано: %s", meeting_id, exc)
    except asyncio.CancelledError:
        # Статус НАМЕРЕННО не трогаем. Отменяют нас только из _schedule_summary,
        # и сразу за отменой стартует новая задача — она статус и разрешит.
        # Поставив здесь "done", мы затёрли бы "summarizing" уже запущенной
        # замены: клиент перестал бы опрашивать посреди живой генерации.
        log.info("Резюме встречи #%d отменено — статус разрешит новая задача", meeting_id)
        raise
    except Exception as exc:
        # Ловим всё остальное: раньше любое исключение кроме LlmError (таймаут
        # по VPN, обрыв связи, ошибка БД) убивало фоновую задачу молча, встреча
        # навсегда оставалась "summarizing", а клиент опрашивал её каждые 4
        # секунды и показывал спиннер до бесконечности.
        # Текст исключения на клиент не отдаём: в сообщениях httpx попадается
        # адрес API, а он не должен покидать сервер. Имени класса хватает, чтобы
        # понять характер сбоя; подробности — в журнале.
        summary, error = None, (
            f"Не удалось создать резюме ({type(exc).__name__}). "
            "Подробности в журнале сервера."
        )
        log.exception("Резюме встречи #%d упало неожиданно", meeting_id)

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
