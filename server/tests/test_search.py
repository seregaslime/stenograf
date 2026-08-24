"""Поиск по прошлым встречам: нарезка на куски, индексация и подбор ближайших.

Эмбеддер подменён: в быстром прогоне не должно быть ни Ollama, ни моделей.
Качество поиска проверяется не здесь, а замером на эталоне
(scripts/eval_search.py) — тест отвечает «работает/сломано», а не «насколько
хорошо».
"""
import asyncio

import numpy as np
import pytest

from app import search
from app.db.models import Chunk, Meeting, Segment
from app.llm.base import LlmError
from app.llm.ollama_client import OllamaClient


def _segment(i: int, text: str, start: float = 0.0) -> Segment:
    return Segment(id=i, meeting_id=1, channel="mic", start_s=start, end_s=start + 1, text=text)


def подставить(monkeypatch, таблица: dict[str, list[float]], размер: int = 3) -> None:
    """Эмбеддер, который выдаёт заранее заданный вектор по подстроке в тексте.

    Так тест задаёт «смысл» явно: близость получается такой, какой мы её
    назначили, и проверяется именно логика поиска, а не поведение модели.
    """
    async def embed(self, model, texts):
        векторы = []
        for текст in texts:
            for ключ, вектор in таблица.items():
                if ключ in текст:
                    векторы.append(вектор)
                    break
            else:
                векторы.append([0.0] * размер)
        return векторы

    monkeypatch.setattr(OllamaClient, "embed", embed)


# ------------------------------------------------------------------ нарезка

def test_chunks_glue_replicas_until_limit():
    """Короткие реплики склеиваются в кусок: вектор от «Да-да, согласен» — шум."""
    сегменты = [_segment(i, "тридцать символов ровно тут да", i) for i in range(1, 5)]
    куски = search.build_chunks(сегменты, max_chars=60)
    assert len(куски) == 2
    assert куски[0]["first_segment_id"] == 1 and куски[0]["last_segment_id"] == 2


def test_chunk_keeps_replica_whole():
    """Граница куска проходит по реплике, а не по символу: разрезанная посреди
    фразы реплика теряет смысл вместе с вектором."""
    длинная = "а" * 200
    куски = search.build_chunks([_segment(1, длинная), _segment(2, "хвост")], max_chars=50)
    assert куски[0]["text"] == длинная
    assert куски[1]["text"] == "хвост"


def test_chunk_skips_empty_segments():
    """Пустые реплики в кусок не попадают — иначе в тексте копятся лишние пробелы."""
    куски = search.build_chunks([_segment(1, "  "), _segment(2, "есть текст")], max_chars=500)
    assert len(куски) == 1 and куски[0]["text"] == "есть текст"


def test_no_chunks_from_silence():
    """Встреча без единой реплики кусков не даёт и к модели не ходит."""
    assert search.build_chunks([], max_chars=500) == []


# ------------------------------------------------------------------ индексация

def test_index_meeting_stores_normalized_vectors(db_session, cfg, monkeypatch):
    """Векторы кладутся L2-нормированными: после этого близость — обычное
    скалярное произведение, без деления на длины при каждом поиске."""
    подставить(monkeypatch, {"бюджет": [3.0, 4.0, 0.0]})
    meeting = Meeting(id=1, title="Планёрка", status="done")
    db_session.add(meeting)
    db_session.add(_segment(1, "обсудили бюджет на квартал"))
    db_session.flush()

    assert asyncio.run(search.index_meeting(db_session, cfg, meeting)) == 1
    кусок = db_session.query(Chunk).one()
    вектор = np.frombuffer(кусок.vector, dtype=np.float32)
    assert float(np.linalg.norm(вектор)) == pytest.approx(1.0, abs=1e-6)
    assert кусок.model == cfg.search_embed_model


def test_reindex_replaces_old_chunks(db_session, cfg, monkeypatch):
    """Повторная индексация не копит дубли: иначе один и тот же разговор
    занимал бы всю выдачу."""
    подставить(monkeypatch, {"бюджет": [1.0, 0.0, 0.0]})
    meeting = Meeting(id=1, title="Планёрка", status="done")
    db_session.add(meeting)
    db_session.add(_segment(1, "обсудили бюджет"))
    db_session.flush()

    asyncio.run(search.index_meeting(db_session, cfg, meeting))
    asyncio.run(search.index_meeting(db_session, cfg, meeting))
    assert db_session.query(Chunk).count() == 1


def test_reindex_missing_skips_already_indexed(db_session, cfg, monkeypatch):
    """Проиндексированную встречу второй раз не считаем: эмбеддинги стоят
    времени и памяти, а поиск дёргает эту проверку на каждый запрос."""
    вызовов = []

    async def embed(self, model, texts):
        вызовов.append(len(texts))
        return [[1.0, 0.0, 0.0]] * len(texts)

    monkeypatch.setattr(OllamaClient, "embed", embed)
    db_session.add(Meeting(id=1, title="Планёрка", status="done"))
    db_session.add(_segment(1, "обсудили бюджет"))
    db_session.flush()

    asyncio.run(search.reindex_missing(db_session, cfg))
    asyncio.run(search.reindex_missing(db_session, cfg))
    assert len(вызовов) == 1


def test_reindex_recounts_after_model_change(db_session, cfg, monkeypatch):
    """Сменили модель — куски пересчитываются: векторы разных моделей лежат в
    разных пространствах, и сравнивать их между собой бессмысленно."""
    подставить(monkeypatch, {"бюджет": [1.0, 0.0, 0.0]})
    db_session.add(Meeting(id=1, title="Планёрка", status="done"))
    db_session.add(_segment(1, "обсудили бюджет"))
    db_session.flush()
    asyncio.run(search.reindex_missing(db_session, cfg))

    cfg.search_embed_model = "другая-модель"
    asyncio.run(search.reindex_missing(db_session, cfg))
    куски = db_session.query(Chunk).all()
    assert len(куски) == 1 and куски[0].model == "другая-модель"


def test_live_meeting_is_not_indexed(db_session, cfg, monkeypatch):
    """Идущая встреча не индексируется: она ещё дописывается, и куски пришлось
    бы пересчитывать после каждой реплики."""
    подставить(monkeypatch, {"бюджет": [1.0, 0.0, 0.0]})
    db_session.add(Meeting(id=1, title="Идёт", status="live"))
    db_session.add(_segment(1, "обсудили бюджет"))
    db_session.flush()
    assert asyncio.run(search.reindex_missing(db_session, cfg)) == 0


# ------------------------------------------------------------------ поиск

def _наполнить(db_session, cfg, monkeypatch) -> None:
    подставить(monkeypatch, {
        "бюджет": [1.0, 0.0, 0.0],
        "сроки": [0.0, 1.0, 0.0],
        "найм": [0.0, 0.0, 1.0],
        "деньги": [1.0, 0.0, 0.0],   # вопрос про деньги = кусок про бюджет
    })
    for i, (тема, текст) in enumerate([
        ("бюджет", "обсудили бюджет на квартал"),
        ("сроки", "перенесли сроки сдачи"),
        ("найм", "обсудили найм разработчика"),
    ], start=1):
        db_session.add(Meeting(id=i, title=f"Встреча про {тема}", status="done"))
        db_session.add(Segment(id=i, meeting_id=i, channel="mic",
                               start_s=0.0, end_s=1.0, text=текст))
    db_session.flush()
    asyncio.run(search.reindex_missing(db_session, cfg))


def test_search_finds_by_meaning_not_by_words(db_session, cfg, monkeypatch):
    """Вопрос другими словами находит нужный разговор — ради этого всё и делается."""
    _наполнить(db_session, cfg, monkeypatch)
    (лучший, *_) = asyncio.run(search.search(db_session, cfg, "куда делись деньги", limit=3))
    assert лучший["meeting_title"] == "Встреча про бюджет"
    assert лучший["similarity"] == pytest.approx(1.0, abs=1e-6)


def test_search_respects_limit(db_session, cfg, monkeypatch):
    """limit ограничивает выдачу: в интерфейсе показывается несколько цитат, не все."""
    _наполнить(db_session, cfg, monkeypatch)
    assert len(asyncio.run(search.search(db_session, cfg, "деньги", limit=2))) == 2


def test_empty_query_returns_nothing(db_session, cfg, monkeypatch):
    """Пустой запрос не идёт к модели: пустое поле в интерфейсе — не повод
    будить эмбеддер и ждать загрузки модели."""
    async def embed(self, model, texts):
        raise AssertionError("к модели ходить не должны")

    monkeypatch.setattr(OllamaClient, "embed", embed)
    assert asyncio.run(search.search(db_session, cfg, "   ")) == []


def test_search_without_index_returns_nothing(db_session, cfg, monkeypatch):
    """Пустая база — пустая выдача, а не падение."""
    подставить(monkeypatch, {"деньги": [1.0, 0.0, 0.0]})
    assert asyncio.run(search.search(db_session, cfg, "деньги")) == []


def test_dimension_mismatch_is_survived(db_session, cfg, monkeypatch):
    """Куски от модели с другой размерностью не роняют поиск.

    Так бывает, если модель сменили, а пересчёт не прошёл целиком: старые
    векторы 3-мерные, новый запрос 5-мерный. Умножение матриц упало бы с
    невнятной ошибкой посреди запроса пользователя.
    """
    _наполнить(db_session, cfg, monkeypatch)
    подставить(monkeypatch, {"деньги": [1.0, 0.0, 0.0, 0.0, 0.0]}, размер=5)
    assert asyncio.run(search.search(db_session, cfg, "деньги")) == []


def test_mixed_dimensions_are_survived(db_session, cfg, monkeypatch):
    """Куски разной длины под одним именем модели не роняют поиск.

    Так бывает, когда модель перекачали новой версией, а пересчёт прошёл
    наполовину: склейка векторов в матрицу не делится нацело, и вместо пустой
    выдачи пользователь получил бы 500 посреди запроса.
    """
    _наполнить(db_session, cfg, monkeypatch)
    чужой = db_session.query(Chunk).first()
    чужой.vector = np.ones(5, dtype=np.float32).tobytes()  # 5 чисел вместо трёх
    db_session.flush()

    подставить(monkeypatch, {"деньги": [1.0, 0.0, 0.0]})
    найдено = asyncio.run(search.search(db_session, cfg, "деньги"))
    assert найдено  # уцелевшие куски по-прежнему ищутся
    assert all(r["meeting_id"] != чужой.meeting_id for r in найдено)


def test_missing_model_error_is_explained(db_session, cfg, monkeypatch):
    """Не скачанная модель эмбеддингов объясняется словами: это чинится одной
    командой, и она должна быть в тексте ошибки."""
    async def embed(self, model, texts):
        raise LlmError(f"Модель «{model}» не найдена в Ollama. Скачайте её: `ollama pull {model}`.")

    monkeypatch.setattr(OllamaClient, "embed", embed)
    db_session.add(Meeting(id=1, title="Планёрка", status="done"))
    db_session.add(_segment(1, "обсудили бюджет"))
    db_session.flush()
    with pytest.raises(LlmError, match="ollama pull"):
        asyncio.run(search.reindex_missing(db_session, cfg))
