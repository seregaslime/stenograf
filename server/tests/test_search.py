"""Нарезка разговора на куски и устойчивость поиска к чужим размерностям.

Векторы считает приложение, здесь их нет вовсе. На сервере осталось то, для
чего модель не нужна: нарезка (она про содержимое встречи) и сравнение готовых
векторов. Качество поиска проверяется не здесь, а замером на эталоне
(scripts/eval_search.py) — тест отвечает «работает/сломано», а не «насколько
хорошо».
"""
import numpy as np
from sqlalchemy import text

from app import search
from app.db.models import Chunk, Meeting, Segment


def _segment(i: int, text: str, start: float = 0.0) -> Segment:
    return Segment(id=i, meeting_id=1, channel="mic", start_s=start, end_s=start + 1, text=text)


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


# ------------------------------------------------- чужие размерности в базе

def _положить(db, вектор: list[float], meeting_id: int = 1) -> None:
    """Кусок со всей роднёй: встреча и сегмент, на который он ссылается.

    Сегмент здесь не для красоты — на него смотрит внешний ключ. Пока база была
    SQLite, она ключи по умолчанию не проверяла, и кусок спокойно ссылался в
    пустоту; PostgreSQL такую вставку отбивает.
    """
    встреча = Meeting(id=meeting_id, title="Планёрка", status="done")
    db.add(встреча)
    сегмент = _segment(meeting_id, "про деньги и сроки")
    сегмент.meeting_id = meeting_id
    db.add(сегмент)
    db.flush()
    db.add(Chunk(
        meeting_id=meeting_id, model="bge-m3", text="про деньги и сроки",
        first_segment_id=сегмент.id, last_segment_id=сегмент.id, start_s=0.0,
        vector=вектор,
    ))
    db.flush()


def test_чужая_размерность_не_роняет_поиск(db_session):
    """Модель сменили, а пересчёт не прошёл целиком: старые векторы трёхмерные,
    новый запрос пятимерный. pgvector отказывается сравнивать векторы разной
    длины — ошибкой посреди запроса человека, если не отсеять их заранее."""
    _положить(db_session, [1.0, 0.0, 0.0])
    найдено = search.search_by_vector(
        db_session, "bge-m3", [1.0, 0.0, 0.0, 0.0, 0.0], limit=5,
    )
    assert найдено == []


def test_уцелевшие_куски_ищутся_несмотря_на_чужие(db_session):
    """Модель перекачали новой версией, пересчёт прошёл наполовину: чужие по
    длине куски отсеиваются, а свои находятся, — вместо отказа всего запроса."""
    _положить(db_session, [1.0, 0.0, 0.0], meeting_id=1)
    _положить(db_session, [1.0, 1.0, 1.0, 1.0, 1.0], meeting_id=2)

    найдено = search.search_by_vector(db_session, "bge-m3", [1.0, 0.0, 0.0], limit=5)
    assert [к["meeting_id"] for к in найдено] == [1]


def test_база_ранжирует_так_же_как_прежний_перебор_в_numpy(db_session):
    """Переезд на pgvector не должен поменять выдачу: те же векторы — тот же
    порядок и те же близости, что давало перемножение матрицы в numpy.

    Векторы случайные, но с фиксированным зерном: проверяется арифметика
    сравнения, а не смысл, и порядок должен совпасть до последнего куска.
    """
    rng = np.random.default_rng(17)
    векторы = rng.standard_normal((12, 16)).astype(np.float32)
    for номер, вектор in enumerate(векторы, start=1):
        _положить(db_session, list(вектор / np.linalg.norm(вектор)), meeting_id=номер)
    запрос = rng.standard_normal(16).astype(np.float32)

    найдено = search.search_by_vector(db_session, "bge-m3", list(запрос), limit=5)

    нормированные = векторы / np.linalg.norm(векторы, axis=1, keepdims=True)
    близости = нормированные @ (запрос / np.linalg.norm(запрос))
    ожидаемые = np.argsort(-близости)[:5] + 1
    assert [к["meeting_id"] for к in найдено] == list(ожидаемые)
    assert [к["similarity"] for к in найдено] == [round(float(близости[i - 1]), 3) for i in ожидаемые]


# ------------------------------------------- хранение векторов и оценка диска

def test_векторы_лежат_в_строке_а_не_в_отдельном_хранилище(db_session):
    """Вынесенные в TOAST векторы перебор достаёт по одному, а база считает
    такой перебор дешёвым и не берёт индекс. Замер на 50 тысячах кусков: 188 мс
    с выносом, 59 мс без. MAIN, а не PLAIN, — иначе длинный вектор не запишется."""
    хранение = db_session.execute(text(
        "SELECT attstorage FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'vector'")).scalar()
    assert хранение == "m"


def test_база_считает_диск_ssd(db_session):
    """С оценкой для диска с головкой (4) база на 50 тысячах кусков выбирала
    перебор за 120 мс вместо индекса за 36 мс."""
    assert db_session.execute(text("SHOW random_page_cost")).scalar() == "1.1"


def test_длинный_вектор_записывается(db_session):
    """Вектор на 4096 чисел (qwen3-embedding:8b) — 16 КБ, больше страницы базы.
    Со строгим хранением в строке запись упала бы; MAIN выносит то, что не влезло."""
    _положить(db_session, [1.0] + [0.0] * 4095)
    найдено = search.search_by_vector(db_session, "bge-m3", [1.0] + [0.0] * 4095, limit=5)
    assert [к["meeting_id"] for к in найдено] == [1]


# ------------------------------------------------------------ индекс HNSW

def _план(db, dims: int, by_owner: bool = False) -> str:
    """План запроса поиска — в том виде, в каком его выберет база после пяти
    одинаковых запросов: драйвер тогда переходит на заготовленный запрос с
    общим планом, где параметры неизвестны. Здесь общий план выбран сразу."""
    sql = search.search_sql(dims, by_owner)
    параметры = [":q", ":model", ":limit"] + ([":owner_id"] if by_owner else [])
    for номер, имя in enumerate(параметры, start=1):
        sql = sql.replace(имя, f"${номер}")
    db.execute(text("SET LOCAL enable_seqscan = off"))  # на пяти строках перебор дешевле всегда
    db.execute(text("SET LOCAL plan_cache_mode = force_generic_plan"))
    db.execute(text(f"PREPARE поиск AS {sql}"))
    вектор = "[" + ",".join(["1"] + ["0"] * (dims - 1)) + "]"
    аргументы = f"'{вектор}', 'bge-m3', 5" + (", 1" if by_owner else "")
    план = "\\n".join(r[0] for r in db.execute(text(f"EXPLAIN EXECUTE поиск({аргументы})")))
    db.execute(text("DEALLOCATE поиск"))
    return план


def test_индексация_строит_индекс_и_поиск_его_берёт(db_session):
    """Главный риск индекса — что база его молча не возьмёт: приведение длины и
    условие в запросе обязаны совпасть с индексом дословно, в том числе в общем
    плане, где параметры неизвестны."""
    встреча = Meeting(id=1, title="Планёрка", status="done")
    db_session.add(встреча)
    сегмент = _segment(1, "про сроки")
    db_session.add(сегмент)
    db_session.flush()
    search.store_vectors(db_session, "bge-m3", встреча, [{
        "first_segment_id": 1, "last_segment_id": 1, "start_s": 0.0,
        "text": "про сроки", "vector": [1.0, 0.0, 0.0],
    }])

    индексы = db_session.execute(text(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'")).scalars().all()
    assert "chunks_vector_hnsw_3" in индексы
    assert "chunks_vector_hnsw_3" in _план(db_session, 3)
    assert "chunks_vector_hnsw_3" in _план(db_session, 3, by_owner=True)


def test_длинный_вектор_без_индекса(db_session):
    """Длиннее 2000 чисел pgvector индекс не строит — такие векторы ищутся
    перебором, и попытка создать индекс не должна ронять индексацию."""
    search.ensure_index(db_session, 4096)
    индексы = db_session.execute(text(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'")).scalars().all()
    assert "chunks_vector_hnsw_4096" not in индексы


def test_фильтр_после_индекса_не_съедает_выдачу(db_session):
    """Индекс ищет ближайших среди всех кусков этой длины, а другая модель
    отсеивается после. Здесь рядом с вопросом лежат 150 кусков чужой модели —
    больше, чем кандидатов смотрит индекс, — а свои пять дальше. Без
    итеративного поиска индекс отдал бы одних чужих, и выдача была бы пустой.

    16 чисел, а не 3–4, как в соседних тестах: на крошечной размерности среди
    полутора сотен почти одинаковых векторов граф индекса оставлял свою точку
    недостижимой — индекс приблизительный, и на таких данных это видно ярче
    всего. Граф строится со случайностью, поэтому устойчивость проверена
    прогоном 20 раз подряд."""
    rng = np.random.default_rng(3)
    вопрос = np.zeros(16, dtype=np.float32)
    вопрос[0] = 1.0
    search.ensure_index(db_session, 16)
    for номер in range(1, 156):
        чужой = номер <= 150
        шум = rng.normal(scale=0.05 if чужой else 0.15, size=16)
        _положить(db_session, list(вопрос + шум), meeting_id=номер)
        if чужой:
            db_session.execute(text("UPDATE chunks SET model = 'другая' WHERE meeting_id = :m"),
                               {"m": номер})
    db_session.execute(text("ANALYZE chunks"))
    db_session.execute(text("SET LOCAL enable_seqscan = off"))

    найдено = search.search_by_vector(db_session, "bge-m3", list(вопрос), limit=5)
    assert sorted(к["meeting_id"] for к in найдено) == list(range(151, 156))
    близости = [к["similarity"] for к in найдено]
    assert близости == sorted(близости, reverse=True)


# ------------------------------------------------------ нарезка документа

def test_короткие_абзацы_склеиваются_и_ничего_не_теряется():
    """Заголовок в одну строку сам по себе ничего не значит — он склеивается со
    следующими абзацами в один кусок, пока тот не наберёт предел."""
    текст = "# Созвоны\n\nПо вторникам в 11.\n\n" + "Длинный абзац про релиз. " * 10
    куски = search.document_chunks(текст, max_chars=120)
    assert куски[0].startswith("# Созвоны\nПо вторникам в 11.")
    assert all(len(к) <= 120 for к in куски)
    assert "".join(куски).replace("\n", "").replace(" ", "") == текст.replace("\n", "").replace(" ", "")


def test_длинный_абзац_режется_по_предложениям():
    абзац = " ".join(f"Предложение номер {i} про сроки." for i in range(20))
    куски = search.document_chunks(абзац, max_chars=100)
    assert all(len(к) <= 100 for к in куски)
    assert all(к.endswith(".") for к in куски)  # ни одно предложение не разорвано


def test_строка_без_точек_режется_по_словам():
    """Выгрузка из таблицы или лога — одна «фраза» на килобайт без точек."""
    строка = " ".join(f"слово{i}" for i in range(300))
    куски = search.document_chunks(строка, max_chars=100)
    assert all(len(к) <= 100 for к in куски)
    assert " ".join(куски).split() == строка.split()  # ни одно слово не разрезано и не потеряно


def test_пустой_документ_не_даёт_кусков():
    assert search.document_chunks("\n\n   \n", max_chars=600) == []
