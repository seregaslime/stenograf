"""Замер разреза реплик по смене голоса на настоящей записи встречи.

Порог speaker_turn_threshold подбирается здесь. Синтез для этого не годится:
у голосов `say` близость своих фраз 0.76 и выше, и на таком наборе любой порог
выглядит хорошим — ровно так уже ошиблись с порогом тишины VAD, который по
синтезу опустили, а живая запись опровергла.

Что нужно: встреча, записанная с галочкой «Записывать аудио встречи», — от неё
берутся реплики из базы и звук с диска. Звук в репозиторий не кладётся (это
аудио встречи), поэтому скрипт только локальный, в CI его нет.

Что печатает — для каждой реплики не короче 2.5 с:
  - самую низкую близость голоса слева и справа (то, с чем сравнивается порог);
  - дорожку окон: на какой из голосов этой встречи похоже каждое окно;
  - разрежет ли её текущий порог.
И сводку: где кончаются реплики с двумя голосами и начинаются монологи — порог
должен встать между ними с запасом, а не на границе.

Честно об эталоне: «два голоса в реплике» определяется сравнением окон с
профилями голосов этой встречи, а не на слух. Это независимо от проверяемого
метода — он профилей не знает, — но это всё же вычисленная разметка. Спорные
реплики стоит послушать.

Запуск из папки server:
    .venv/bin/python scripts/eval_turns.py --meeting 7
"""
import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, text  # noqa: E402

from app.config import SAMPLE_RATE, settings  # noqa: E402
from app.diarization.embedder import VoiceEmbedder  # noqa: E402
from app.diarization.turns import HOP_S, WINDOW_S, local_similarities  # noqa: E402

МИН_РЕПЛИКА_С = WINDOW_S + 3 * HOP_S  # меньше — не хватает окон на две пары
ОТРЫВ = 0.10  # насколько окно должно быть ближе к одному голосу, чем к другому


def читать_wav(путь: Path) -> np.ndarray:
    with wave.open(str(путь)) as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--meeting", type=int, required=True, help="номер записанной встречи")
    args = parser.parse_args()
    порог = settings.speaker_turn_threshold

    engine = create_engine(settings.database_url)
    with engine.connect() as c:
        строка = c.execute(text("select audio_dir from meetings where id = :m"),
                           {"m": args.meeting}).fetchone()
        if строка is None or not строка[0]:
            sys.exit(f"Встреча {args.meeting} не записана: нет audio_dir. "
                     "Нужна встреча с галочкой «Записывать аудио встречи».")
        папка = Path(строка[0])
        реплики = c.execute(text(
            "select s.id, s.start_s, s.end_s, s.channel, s.speaker_id from segments s "
            "where s.meeting_id = :m order by s.start_s"), {"m": args.meeting}).fetchall()
        участники = sorted({r[4] for r in реплики if r[4] is not None})
        имена = dict(c.execute(text("select id, name from speakers where id = any(:ids)"),
                               {"ids": участники}).fetchall())
        отпечатки: dict[int, list[np.ndarray]] = {}
        for sid, вектор in c.execute(text(
                "select speaker_id, centroid from voiceprints where speaker_id = any(:ids)"),
                {"ids": участники}):
            v = np.frombuffer(вектор, dtype=np.float32)
            отпечатки.setdefault(sid, []).append(v / np.linalg.norm(v))

    # Буквами — пока их хватает, дальше номерами: на большой встрече участников
    # бывает больше десяти, и падать из-за этого замер не должен.
    буквы = {sid: ("АБВГДЕЖЗИК"[i] if i < 10 else str(i)) for i, sid in enumerate(отпечатки)}
    print("Голоса встречи: " + ", ".join(f"{буквы[s]} — {имена.get(s, s)}" for s in буквы))

    каналы = {имя: читать_wav(папка / f"{имя}.wav") for имя in ("mic", "system")}
    длина = min(len(v) for v in каналы.values())
    каналы["mixed"] = (каналы["mic"][:длина] + каналы["system"][:длина]) / 2

    эмбеддер = VoiceEmbedder(settings)
    эмбеддер.load()

    def метка(вектор: np.ndarray) -> str:
        близости = sorted(((max(float(вектор @ w) for w in ws), s) for s, ws in отпечатки.items()),
                          reverse=True)
        if len(близости) > 1 and близости[0][0] - близости[1][0] < ОТРЫВ:
            return "?"
        return буквы[близости[0][1]]

    итог = []
    for sid, начало, конец, канал, _ in реплики:
        if конец - начало < МИН_РЕПЛИКА_С or канал not in каналы:
            continue
        звук, окна, t = каналы[канал], [], начало
        while t + WINDOW_S <= конец:
            окна.append(эмбеддер.embed(звук[int(t * SAMPLE_RATE):int((t + WINDOW_S) * SAMPLE_RATE)]))
            t += HOP_S
        окна = np.array(окна)
        близости = local_similarities(окна)
        if not близости:
            continue
        дорожка = "".join(метка(в) for в in окна)
        двое = len(set(дорожка) - {"?"}) > 1
        итог.append((min(b for b, _ in близости), sid, дорожка, двое))

    print(f"\n{'реплика':>8}  {'близость':>8}  {'разрежет':>8}  дорожка")
    for близость, sid, дорожка, двое in sorted(итог):
        отметка = "да" if близость < порог else ""
        хвост = "   ← два голоса" if двое else ""
        print(f"{'#' + str(sid):>8}  {близость:>8.3f}  {отметка:>8}  {дорожка}{хвост}")

    с_двумя = [b for b, _, _, д in итог if д]
    с_одним = [b for b, _, _, д in итог if not д]
    print(f"\nпорог {порог}")
    if с_двумя:
        print(f"  реплик с двумя голосами: {len(с_двумя)}, разрежет {sum(b < порог for b in с_двумя)}; "
              f"самая похожая из них {max(с_двумя):.3f}")
    if с_одним:
        print(f"  монологов: {len(с_одним)}, ложно разрежет {sum(b < порог for b in с_одним)}; "
              f"самый непохожий {min(с_одним):.3f}")


if __name__ == "__main__":
    main()
