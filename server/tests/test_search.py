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


# ------------------------------------------------------------------ ответ по найденному

class _Модель:
    """Роутер-подставка: запоминает промпт и отдаёт заготовленный ответ."""

    def __init__(self, ответ: str = "На встрече 14 августа решили двигать сдачу."):
        self.ответ, self.промпт, self.system = ответ, None, None

    async def summarize(self, prompt, system=None, temperature=0.3):
        self.промпт, self.system = prompt, system
        return f"  {self.ответ}  "  # с пробелами: ответ модели обрезается


def test_answer_uses_found_fragments(db_session, cfg, monkeypatch):
    """Модель получает найденные фрагменты, а ответ возвращается вместе с ними.

    Цитаты идут рядом с ответом всегда: модель не помнит встреч, она
    пересказывает показанное, и проверить её можно только по ним.
    """
    _наполнить(db_session, cfg, monkeypatch)
    модель = _Модель()
    итог = asyncio.run(search.answer(db_session, cfg, модель, "куда делись деньги", limit=1))

    assert итог["answer"] == "На встрече 14 августа решили двигать сдачу."
    assert итог["results"] and итог["results"][0]["meeting_title"] == "Встреча про бюджет"
    assert "обсудили бюджет на квартал" in модель.промпт   # фрагмент попал в промпт
    assert "Встреча про бюджет" in модель.промпт           # и с указанием встречи
    assert "куда делись деньги" in модель.промпт           # и сам вопрос


def test_answer_can_use_hints_model(db_session, cfg, monkeypatch):
    """Настройка переключает ответ на модель подсказок.

    На слабой машине это решает: замер на M3 дал 36 секунд у модели протокола
    против 13 у модели подсказок при одинаково верном ответе.
    """
    _наполнить(db_session, cfg, monkeypatch)
    cfg.search_answer_model = "hints"

    звали = []

    class Роутер:
        async def summarize(self, prompt, system=None, temperature=0.3):
            звали.append("summary")
            return "ответ протоколом"

        async def hint(self, prompt, system=None, temperature=0.5, on_delta=None):
            звали.append("hints")
            return "ответ подсказкой"

    итог = asyncio.run(search.answer(db_session, cfg, Роутер(), "деньги", limit=1))
    assert звали == ["hints"] and итог["answer"] == "ответ подсказкой"


def test_answer_without_hits_does_not_wake_model(db_session, cfg, monkeypatch):
    """Ничего не нашлось — модель не зовём.

    Локальная модель грузится в память секундами, и будить её ради ответа
    «нечего пересказывать» — это ожидание на ровном месте.
    """
    подставить(monkeypatch, {"деньги": [1.0, 0.0, 0.0]})
    модель = _Модель()
    итог = asyncio.run(search.answer(db_session, cfg, модель, "деньги"))
    assert итог == {"answer": "", "results": []}
    assert модель.промпт is None


def test_answer_forbids_general_knowledge(db_session, cfg, monkeypatch):
    """В правилах промпта запрещено достраивать ответ из общих знаний.

    Это отличие от чата по текущей встрече, где это как раз разрешено: там
    человек спрашивает про мир, здесь — про договорённости, и придуманная
    договорённость хуже её отсутствия.
    """
    _наполнить(db_session, cfg, monkeypatch)
    модель = _Модель()
    asyncio.run(search.answer(db_session, cfg, модель, "деньги", limit=1))
    assert "в записях встреч этого нет" in модель.system


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
