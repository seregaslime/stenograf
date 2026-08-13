"""Синтезирует эталонный набор для регрессионного прогона.

Запускается редко и только на macOS: голоса берутся из системного `say`. Файлы
кладутся в tests/fixtures/regress и коммитятся — чтобы regress.py работал у
кого угодно и где угодно, включая контейнер, где лежат модели.

Голоса синтезированные намеренно. Живые «звучания» из data/samples — это записи
участников встреч, а репозиторий публичный: класть туда чужие голоса нельзя.
Синтез снимает вопрос и заодно даёт то, чего у живых записей нет, — известную
заранее расшифровку, то есть настоящий эталон для WER.

Запуск из папки server:
    .venv/bin/python scripts/make_regress_fixtures.py
"""
import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "regress"

# Фразы «совещательные»: с именами, числами и терминами — на них ASR и ошибается.
SPEECH = [
    "Коллеги, давайте начнём совещание по проекту Стенограф",
    "Илья Петрович предлагает перенести дедлайн на середину сентября",
    "Бюджет на третий квартал уже согласован с руководством",
    "Нужно связать распознавание речи с базой данных",
    "Протокол встречи будет готов к завтрашнему утру",
    "Качество распознавания зависит от микрофона и фонового шума",
    "Есть вопросы по диаризации голосов и определению спикеров",
    "Подведём итоги и назначим следующую встречу на конец недели",
]

# Три «участника». Милена говорит по-русски, остальные с акцентом — для
# диаризации это неважно: ECAPA сравнивает тембр, а не слова.
VOICES = ["Milena", "Daniel", "Anna"]
VOICE_PHRASES = [
    "Коллеги, давайте начнём встречу и обсудим план на неделю",
    "У меня есть несколько вопросов по текущему статусу проекта",
    "Предлагаю перенести обсуждение бюджета на следующий раз",
    "Хорошо, тогда я подготовлю отчёт к завтрашнему утру",
]


def synth(voice: str, text: str, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["say", "-v", voice, "-o", str(path), "--data-format=LEI16@16000", text],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  голос {voice} недоступен: {result.stderr.decode().strip()}")
        return False
    return True


def main() -> None:
    if sys.platform != "darwin":
        sys.exit("Нужен macOS: синтез идёт системной командой say.")

    reference = {"speech": [], "voices": {}}

    print(f"Речь для WER ({len(SPEECH)} фраз, голос Milena)...")
    for i, text in enumerate(SPEECH):
        path = FIXTURES / "speech" / f"{i:02d}.wav"
        if synth("Milena", text, path):
            reference["speech"].append({"file": f"speech/{i:02d}.wav", "text": text})

    print(f"Голоса для диаризации ({len(VOICES)} × {len(VOICE_PHRASES)})...")
    for voice in VOICES:
        files = []
        for i, text in enumerate(VOICE_PHRASES):
            path = FIXTURES / "voices" / voice.lower() / f"{i:02d}.wav"
            if not synth(voice, text, path):
                break
            files.append(f"voices/{voice.lower()}/{i:02d}.wav")
        if files:
            reference["voices"][voice.lower()] = files

    (FIXTURES / "reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    размер = sum(p.stat().st_size for p in FIXTURES.rglob("*.wav"))
    print(f"\nГотово: {len(list(FIXTURES.rglob('*.wav')))} файлов, "
          f"{размер / 1024 / 1024:.1f} МБ в {FIXTURES}")


if __name__ == "__main__":
    main()
