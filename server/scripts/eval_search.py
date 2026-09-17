"""Замер качества поиска по встречам: попадает ли нужный разговор в топ-N и на каком месте.

Зачем отдельный скрипт, а не тест: тест отвечает «работает/сломано», а здесь
нужно число, по которому выбирают модель эмбеддингов и длину куска. Устроен как
scripts/regress.py — тот же приём, только меряется не WER, а доля попаданий.

Эталон (tests/fixtures/search/meetings.json) выдуман намеренно: вопросы заданы
ДРУГИМИ словами, чем сказано в разговоре, — на таком наборе поиск по подстроке
проваливается, а поиск по смыслу обязан справляться.

Адрес модели задаётся здесь, а не берётся из настроек сервера: с 05.09.2026 их
там нет вовсе — эмбеддинги считает приложение своей моделью. Замер остался
серверным, потому что мерит он не модель, а нарезку и подбор: их код здесь.

Что печатает по каждой модели:
  - в топ-N — доля вопросов, где нужная встреча среди первых N;
  - первым — доля, где она на первом месте: человек читает верхнюю цитату;
  - MRR — среднее 1/место нужной встречи (1 — всегда первая, 0.5 — в среднем
    вторая): различает модели, которые обе попадают в топ-3, но одна ставит
    нужное первым, а другая третьим. Одних «попаданий в топ-3» не хватило:
    bge-m3 набирала 100%, и сравнить её было не с чем (17.09.2026);
  - то же отдельно по трудным вопросам ("hard" в эталоне).

Запуск из папки server (Ollama — там, где она у вас стоит):
    .venv/bin/python scripts/eval_search.py
    .venv/bin/python scripts/eval_search.py --model bge-m3 --model paraphrase-multilingual
    .venv/bin/python scripts/eval_search.py --url http://192.168.1.50:11434
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "search" / "meetings.json"

# Длину куска берём из настроек сервера — это его параметр, он там и остался.
# Через getattr: в контейнере код приложения может быть старше правки (образ
# собран раньше), и обращение к новому полю уронило бы замер там, где он как раз
# и нужен — рядом с моделями.
ЧАНК = getattr(settings, "search_chunk_chars", 600)

# А вот модель и её адрес — уже не дело сервера. Дефолты здесь означают «самое
# частое место, где стоит Ollama» и «модель, которой мерили до сих пор»:
# bge-m3 обошла вдвое более лёгкую paraphrase-multilingual не на этом эталоне
# (там они равны), а на длинных кусках — у неё вход 128 токенов, и кусок на 1313
# символов она молча обрезает: близость к запросу упала с 0.639 до 0.187 против
# 0.728 → 0.579 у bge-m3.
#
# 17.09.2026 с ней сравнили qwen3-embedding:0.6b (та же длина вектора, 1024) на
# эталоне с трудными вопросами, Ollama на ПК:
#   bge-m3                    — все: первым 92%, MRR 0.962; трудные: первым 90%, MRR 0.950
#   qwen3-embedding:0.6b      — все: первым 92%, MRR 0.947; трудные: первым 80%, MRR 0.863
#   то же с инструкцией Qwen  — трудные: первым 70% (английская), 60% (русская)
# bge-m3 осталась: на трудных вопросах qwen ошибается чаще, а инструкция, которую
# авторы qwen3-embedding советуют ставить перед вопросом, здесь только мешает.
# Эталон мал (26 вопросов, разница — один-два вопроса), вывод предварительный.
URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
МОДЕЛЬ = "bge-m3"


def нарезать(встреча: dict, max_chars: int) -> list[str]:
    """Куски из реплик — та же логика, что в app.search.build_chunks.

    Повторена здесь намеренно: скрипт должен запускаться в контейнере, где кода
    приложения может ещё не быть (образ собран раньше правки), — как это уже
    сделано в regress.py.
    """
    куски, текущий, длина = [], [], 0
    for реплика in встреча["replicas"]:
        текущий.append(реплика)
        длина += len(реплика)
        if длина >= max_chars:
            куски.append(" ".join(текущий))
            текущий, длина = [], 0
    if текущий:
        куски.append(" ".join(текущий))
    return куски


async def векторы(url: str, model: str, тексты: list[str]) -> np.ndarray:
    """Эмбеддинги пачкой, запросом к Ollama напрямую.

    Своим запросом, а не клиентским кодом из client/src/llm: замер живёт на
    сервере и запускается там, где стоят модели. Ровно та же причина, по которой
    продублирована нарезка.
    """
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0)) as client:
        ответ = await client.post(f"{url.rstrip('/')}/api/embed",
                                  json={"model": model, "input": тексты})
        ответ.raise_for_status()
        v = np.array(ответ.json()["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


async def места(url: str, model: str, данные: dict, max_chars: int, instruct: str = "") -> list[int]:
    """Место нужной встречи в выдаче для каждого вопроса, начиная с 1.

    Место — среди встреч, а не кусков: у встречи их может быть несколько, и
    человек видит разговор, а не кусок. Встреча стоит на месте своего лучшего куска.
    """
    тексты, откуда = [], []
    for встреча in данные["meetings"]:
        for кусок in нарезать(встреча, max_chars):
            тексты.append(кусок)
            откуда.append(встреча["id"])

    куски = await векторы(url, model, тексты)
    # Инструкция — только перед вопросом, куски идут как есть: так устроены
    # модели вроде qwen3-embedding, у которых вопрос и документ кодируются по-разному
    префикс = f"Instruct: {instruct}\nQuery: " if instruct else ""
    запросы = await векторы(url, model, [префикс + в["q"] for в in данные["questions"]])

    результат = []
    for вопрос, запрос in zip(данные["questions"], запросы):
        порядок: list[str] = []
        for i in np.argsort(-(куски @ запрос)):
            if откуда[i] not in порядок:
                порядок.append(откуда[i])
        результат.append(порядок.index(вопрос["expect"]) + 1)
    return результат


def сводка(места_: list[int], top: int) -> str:
    в_топе = sum(м <= top for м in места_) / len(места_)
    первым = sum(м == 1 for м in места_) / len(места_)
    mrr = sum(1 / м for м in места_) / len(места_)
    return f"в топ-{top}: {в_топе:4.0%}   первым: {первым:4.0%}   MRR: {mrr:.3f}"


async def main() -> None:
    parser = argparse.ArgumentParser(description="Замер поиска по встречам")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--model", action="append", default=[],
                        help="можно указать несколько раз — сравнить модели")
    parser.add_argument("--chunk-chars", type=int, default=ЧАНК)
    parser.add_argument("--url", default=URL, help=f"адрес Ollama (по умолчанию {URL})")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--instruct", default="",
                        help="инструкция перед вопросом (для qwen3-embedding), куски без неё")
    args = parser.parse_args()

    данные = json.loads(args.fixtures.read_text(encoding="utf-8"))
    модели = args.model or [МОДЕЛЬ]
    print(f"Эталон: {len(данные['meetings'])} встреч, {len(данные['questions'])} вопросов; "
          f"кусок ≤ {args.chunk_chars} символов, попадание в топ-{args.top}\n")

    трудные = [i for i, в in enumerate(данные["questions"]) if в.get("hard")]
    for model in модели:
        места_ = await места(args.url, model, данные, args.chunk_chars, args.instruct)
        print(f"  {model:26} все {len(места_):2}:      {сводка(места_, args.top)}")
        if трудные:
            print(f"  {'':26} трудные {len(трудные):2}:  {сводка([места_[i] for i in трудные], args.top)}")
        for вопрос, место in zip(данные["questions"], места_):
            if место > 1:
                print(f"      на {место}-м месте: «{вопрос['q']}» (ждали {вопрос['expect']})")


if __name__ == "__main__":
    asyncio.run(main())
