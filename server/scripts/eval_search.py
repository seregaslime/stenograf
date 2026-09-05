"""Замер качества поиска по встречам: попадает ли нужный разговор в топ-N.

Зачем отдельный скрипт, а не тест: тест отвечает «работает/сломано», а здесь
нужно число, по которому выбирают модель эмбеддингов и длину куска. Устроен как
scripts/regress.py — тот же приём, только меряется не WER, а доля попаданий.

Эталон (tests/fixtures/search/meetings.json) выдуман намеренно: вопросы заданы
ДРУГИМИ словами, чем сказано в разговоре, — на таком наборе поиск по подстроке
проваливается, а поиск по смыслу обязан справляться.

Адрес модели задаётся здесь, а не берётся из настроек сервера: с 05.09.2026 их
там нет вовсе — эмбеддинги считает приложение своей моделью. Замер остался
серверным, потому что мерит он не модель, а нарезку и подбор: их код здесь.

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


async def замер(url: str, model: str, данные: dict, max_chars: int, top: int) -> tuple[float, list[str]]:
    """Доля вопросов, для которых нужная встреча попала в топ-N. И список промахов."""
    тексты, откуда = [], []
    for встреча in данные["meetings"]:
        for кусок in нарезать(встреча, max_chars):
            тексты.append(кусок)
            откуда.append(встреча["id"])

    куски = await векторы(url, model, тексты)
    запросы = await векторы(url, model, [в["q"] for в in данные["questions"]])

    попаданий, промахи = 0, []
    for вопрос, запрос in zip(данные["questions"], запросы):
        близости = куски @ запрос
        лучшие = [откуда[i] for i in np.argsort(-близости)[:top]]
        if вопрос["expect"] in лучшие:
            попаданий += 1
        else:
            промахи.append(f"«{вопрос['q']}» → {лучшие} (ждали {вопрос['expect']})")
    return попаданий / len(данные["questions"]), промахи


async def main() -> None:
    parser = argparse.ArgumentParser(description="Замер поиска по встречам")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--model", action="append", default=[],
                        help="можно указать несколько раз — сравнить модели")
    parser.add_argument("--chunk-chars", type=int, default=ЧАНК)
    parser.add_argument("--url", default=URL, help=f"адрес Ollama (по умолчанию {URL})")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    данные = json.loads(args.fixtures.read_text(encoding="utf-8"))
    модели = args.model or [МОДЕЛЬ]
    print(f"Эталон: {len(данные['meetings'])} встреч, {len(данные['questions'])} вопросов; "
          f"кусок ≤ {args.chunk_chars} символов, попадание в топ-{args.top}\n")

    for model in модели:
        доля, промахи = await замер(args.url, model, данные, args.chunk_chars, args.top)
        print(f"  {model:26} попаданий: {доля:.0%}")
        for промах in промахи:
            print(f"      мимо: {промах}")


if __name__ == "__main__":
    asyncio.run(main())
