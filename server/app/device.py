"""Выбор вычислительного устройства для моделей.

Одно место на все три загрузчика (whisper, GigaAM, ECAPA). Раньше «cpu» было
зашито в каждом из них, и у куратора на Windows всё считалось процессором, хотя
видеокарта стояла, — отсюда и жалоба на тормоза.

Здесь же спрятана разница в поддержке: движки умеют не одно и то же, и «мы
выбрали cuda» не значит, что её примут все трое.
"""
import logging

log = logging.getLogger(__name__)

DEVICES = ("auto", "cuda", "mps", "cpu")


def _available(name: str) -> bool:
    """Есть ли устройство на самом деле.

    torch импортируем внутри: он тяжёлый, а выбор устройства спрашивают и там,
    где моделей нет (например, /api/health). Любая поломка импорта — это «нет»,
    а не падение сервера: остаться на процессоре хуже, чем не запуститься.
    """
    try:
        import torch
    except Exception:
        return False
    try:
        if name == "cuda":
            return torch.cuda.is_available()
        if name == "mps":
            return torch.backends.mps.is_available()
    except Exception:
        return False
    return False


def resolve(preference: str = "auto") -> str:
    """Какое устройство использовать: auto → cuda → cpu.

    Metal (`mps`) сам не выбирается, хотя и поддерживается: на Apple Silicon
    для whisper в проекте есть отдельный движок mlx, а GigaAM и ECAPA на Metal
    не мерялись — а на процессоре мерялись и работают быстрее реального
    времени. Молча переводить рабочую машину на непроверенный путь ради
    предположительного ускорения не стоит; кому надо — ставит `mps` руками.

    Запрошенное явно, но отсутствующее — не ошибка, а откат на процессор с
    записью в журнал: конфиг переезжает с машины на машину (у Сергея Mac, у
    куратора Windows с видеокартой, на сервере организации может не быть ни
    той, ни другой), и падать из-за этого нельзя.
    """
    preference = (preference or "auto").strip().lower()
    if preference not in DEVICES:
        log.warning("Неизвестное устройство «%s» — считаем на процессоре", preference)
        return "cpu"
    if preference == "cpu":
        return "cpu"
    if preference == "auto":
        return "cuda" if _available("cuda") else "cpu"
    if _available(preference):
        return preference
    log.warning("Устройство «%s» недоступно на этой машине — считаем на процессоре",
                preference)
    return "cpu"


def for_ctranslate2(device: str) -> str:
    """faster-whisper работает через CTranslate2, а тот знает только cpu и cuda.

    Metal он не поддерживает вовсе, поэтому на Маке whisper остаётся на
    процессоре — для Apple Silicon в проекте есть отдельный движок mlx.
    """
    return device if device == "cuda" else "cpu"


def compute_type(device: str, preference: str = "auto") -> str:
    """Точность вычислений для faster-whisper.

    На процессоре int8 — он для того и квантован. На видеокарте int8 не даёт
    ничего, кроме потери качества: float16 там и быстрее, и точнее.
    """
    preference = (preference or "auto").strip().lower()
    if preference != "auto":
        return preference
    return "float16" if device == "cuda" else "int8"
