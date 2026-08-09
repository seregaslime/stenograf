"""Выбор вычислительного устройства.

Замечание куратора №4: у него на Windows всё считалось процессором при живой
видеокарте, потому что "cpu" было зашито в трёх загрузчиках. Здесь проверяется
только логика выбора — саму видеокарту в тестах не потрогать.
"""
import pytest

from app import device as device_mod


@pytest.fixture()
def gpus(monkeypatch):
    """Подменяет «что есть на этой машине»: тесты гоняются и на Маке, и в CI."""
    present: set[str] = set()
    monkeypatch.setattr(device_mod, "_available", lambda name: name in present)
    return present


def test_auto_prefers_the_graphics_card(gpus):
    gpus.add("cuda")
    assert device_mod.resolve("auto") == "cuda"


def test_auto_falls_back_to_cpu_without_a_card(gpus):
    assert device_mod.resolve("auto") == "cpu"


def test_auto_does_not_take_metal_by_itself(gpus):
    """Metal сам не выбирается: GigaAM и ECAPA на нём не мерялись, а на
    процессоре мерялись и работают быстрее реального времени. Для whisper на
    Apple Silicon в проекте есть отдельный движок mlx."""
    gpus.add("mps")
    assert device_mod.resolve("auto") == "cpu"
    assert device_mod.resolve("mps") == "mps"  # но руками — пожалуйста


def test_missing_device_falls_back_instead_of_crashing(gpus):
    """Конфиг переезжает с машины на машину: у Сергея Mac, у куратора Windows
    с видеокартой, на сервере организации может не быть ни того, ни другого."""
    assert device_mod.resolve("cuda") == "cpu"


def test_nonsense_value_is_not_fatal(gpus):
    assert device_mod.resolve("видеокарта") == "cpu"
    assert device_mod.resolve("") == "cpu"


def test_whisper_never_gets_metal():
    """faster-whisper работает через CTranslate2, а тот знает только cpu и cuda:
    отдав ему mps, получили бы падение при загрузке модели."""
    assert device_mod.for_ctranslate2("mps") == "cpu"
    assert device_mod.for_ctranslate2("cuda") == "cuda"
    assert device_mod.for_ctranslate2("cpu") == "cpu"


def test_precision_follows_the_device():
    """int8 на видеокарте не даёт ничего, кроме потери качества."""
    assert device_mod.compute_type("cuda") == "float16"
    assert device_mod.compute_type("cpu") == "int8"
    # заданное руками уважаем как есть
    assert device_mod.compute_type("cuda", "int8_float16") == "int8_float16"


def test_broken_torch_is_treated_as_no_gpu(monkeypatch):
    """Импорт torch может упасть (битая установка, неполный образ) — это «нет
    видеокарты», а не повод не запустить сервер."""
    import builtins

    real_import = builtins.__import__

    def explode(name, *args, **kwargs):
        if name == "torch":
            raise RuntimeError("сломанная установка")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", explode)
    assert device_mod.resolve("auto") == "cpu"
    assert device_mod.resolve("cuda") == "cpu"
