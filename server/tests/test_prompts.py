"""Сборка промптов: режимы встречи, право промолчать (SKIP), разбор ответа.

Чистые функции без сети и без БД — самый дешёвый и самый ценный слой проверок:
формулировки промпта тут единственное, что отделяет полезную подсказку от воды.
"""
import pytest

from app.llm import prompts


# ------------------------------------------------------------------ режимы

def test_normalize_mode_falls_back_to_default():
    """Пустой, неизвестный или отсутствующий режим (старый клиент) превращается в режим по умолчанию.
    """
    for value in (None, "", "gibberish", "WORK"):
        assert prompts.normalize_mode(value) == prompts.DEFAULT_MODE
    for key in prompts.MODES:
        assert prompts.normalize_mode(key) == key


def test_every_mode_has_focus_and_sections():
    """У каждого режима встречи заполнены и фокус подсказок, и разделы протокола."""
    for mode in prompts.MODES.values():
        assert mode.label and mode.hint_focus and mode.summary_sections


# ------------------------------------------------------------------ промпт подсказок

def _hint(**over):
    params = dict(mode="work", transcript="Иван: тест", previous="—")
    params.update(over)
    return prompts.build_hint_prompt(**params)


def test_hint_prompt_lists_all_five_categories():
    """В системном промпте перечислены все пять разрешённых типов подсказки."""
    system, _ = _hint()
    for word in ("ОТВЕТ", "ТЕРМИН", "РИСК", "ПРОБЕЛ", "ЗАДАЧА"):
        assert word in system


def test_hint_prompt_modes_differ():
    """Режимы дают разные промпты; собеседование написано со стороны соискателя, переговоры — про уступки.
    """
    focuses = {key: _hint(mode=key)[0] for key in prompts.MODES}
    assert len(set(focuses.values())) == len(prompts.MODES)
    assert "кандидат" in focuses["interview"]  # собеседование — со стороны соискателя
    assert "уступк" in focuses["negotiation"]


def test_allow_skip_toggles_skip_rule_everywhere():
    """Главная гарантия: по кнопке «Подсказать сейчас» слова SKIP быть не должно
    нигде — ни в system, ни в примерах, иначе модель промолчит на явный запрос."""
    s_auto, u_auto = _hint(allow_skip=True)
    s_force, u_force = _hint(allow_skip=False)
    assert prompts.SKIP_TOKEN in s_auto and prompts.SKIP_TOKEN in u_auto
    assert prompts.SKIP_TOKEN not in s_force and prompts.SKIP_TOKEN not in u_force
    assert "обязателен" in s_force


def test_detailed_prompt_carries_context_and_is_longer():
    """Развёрнутый вариант (для API) длиннее и включает название встречи и участников."""
    _, short = _hint(detailed=False)
    _, long_ = _hint(detailed=True, title="Планёрка", participants="Иван (3 реплик)")
    assert len(long_) > len(short)
    assert "Планёрка" in long_ and "Иван (3 реплик)" in long_


def test_hint_prompt_includes_transcript_and_previous():
    """В промпт попадают и текущий фрагмент разговора, и ранее выданные подсказки (против повторов).
    """
    _, prompt = _hint(transcript="Мария: что такое SLA?", previous="Прошлая подсказка")
    assert "Мария: что такое SLA?" in prompt and "Прошлая подсказка" in prompt


# ------------------------------------------------------------------ разбор ответа

@pytest.mark.parametrize("raw", [
    None, "", "   ", "SKIP", "skip", " SKIP. ", "**SKIP**", "`SKIP`", "SKIP.",
    "Ок.", "—", "Нет.",
])
def test_parse_hint_treats_as_silence(raw):
    """Разные формы молчания модели (SKIP, пустой ответ, «Ок.») распознаются как «подсказки нет».
    """
    assert prompts.parse_hint(raw) is None


def test_parse_hint_keeps_real_hint_containing_skip_word():
    """Жадный матчинг съел бы полезную подсказку со словом skip внутри."""
    text = "Не пропускайте код-ревью — skip тут дорого обойдётся команде."
    assert prompts.parse_hint(text) == text


def test_parse_hint_strips_prefix_and_restores_capital():
    """Служебные префиксы вроде «Подсказка:» срезаются, первая буква возвращается заглавной."""
    assert prompts.parse_hint("Подсказка: уточните срок задачи.") == "Уточните срок задачи."
    assert prompts.parse_hint("Совет: назначьте ответственного.") == "Назначьте ответственного."


def test_parse_hint_respects_min_chars():
    """Слишком короткий ответ считается молчанием; порог настраивается."""
    assert prompts.parse_hint("Коротко", min_chars=100) is None
    assert prompts.parse_hint("Коротко", min_chars=3) == "Коротко"


# ------------------------------------------------------------------ промпт резюме

def test_summary_sections_depend_on_mode():
    """Разделы протокола зависят от типа встречи: у планёрки решения, у собеседования — что подтянуть, у переговоров — позиции сторон.
    """
    def sections(mode):
        return prompts.build_summary_prompt(
            mode=mode, title="T", date="D", participants="P", transcript="X"
        )[1]

    assert "Принятые решения" in sections("work")          # регрессия на текущий формат
    assert "Что стоит подтянуть" in sections("interview")  # собеседование — глазами соискателя
    assert "Позиции сторон" in sections("negotiation")


def test_summary_detailed_adds_instructions():
    """Для API протокол просит больше: таймкоды у решений и раздел открытых вопросов."""
    common = dict(mode="work", title="T", date="D", participants="P", transcript="X")
    sys_short, prompt_short = prompts.build_summary_prompt(**common, detailed=False)
    sys_long, prompt_long = prompts.build_summary_prompt(**common, detailed=True)
    assert len(sys_long) > len(sys_short) and len(prompt_long) > len(prompt_short)
    assert "таймкод" in sys_long
    assert "Открытые вопросы" in prompt_long


def test_both_prompts_warn_about_asr_noise():
    """Распознавание русское и коверкает английские термины: «UDP» приходит как
    «уд». Без предупреждения модель честно объясняет, что такое «уд» — реальный
    случай с живой встречи.
    """
    hint_system, _ = _hint()
    summary_system, _ = prompts.build_summary_prompt(
        mode="work", title="T", date="D", participants="P", transcript="X"
    )
    for system in (hint_system, summary_system):
        assert "распознаванием речи" in system
        assert "UDP" in system          # пример искажения
        assert "не угадывай" in system  # и защита от выдумывания термина


# ------------------------------------------------------------------ честность протокола

@pytest.mark.parametrize("builder", ["summary", "reduce"])
def test_summary_forbids_the_four_ways_to_lie(builder):
    """Правила против выдумок есть и в обычном протоколе, и в сведении фрагментов.

    Все четыре запрета появились не из общих соображений, а из разбора реального
    протокола встречи 03.08.2026: модель выдумала все шесть таймкодов, назначила
    ответственным того, кто задачу попросил, записала уже проделанное
    нагрузочное тестирование в планы и превратила отвергнутое предложение
    (whisper по API) в «принятое решение».
    """
    if builder == "summary":
        system, _ = prompts.build_summary_prompt(
            mode="work", title="Т", date="", participants="", transcript="х", detailed=True,
        )
    else:
        system, _ = prompts.build_reduce_prompt(
            mode="work", title="Т", date="", participants="", notes="х", detailed=True,
        )
    assert "НЕ придумывай таймкоды" in system
    assert "НЕ выдавай предложенное за принятое" in system
    assert "НЕ записывай в задачи то, что уже сделано" in system
    assert "НЕ назначай ответственного" in system
    assert "НЕ достраивай названия" in system


def test_detailed_summary_no_longer_demands_a_timecode():
    """Раньше промпт ВЕЛЕЛ ставить таймкод у каждого решения, не сказав откуда.

    Модель послушалась и выдала правдоподобный ряд выдуманных меток — это и был
    источник ошибки. Теперь таймкод разрешён, но только скопированный.
    """
    system, _ = prompts.build_summary_prompt(
        mode="work", title="Т", date="", participants="", transcript="х", detailed=True,
    )
    assert "указывай таймкод" not in system
    assert "СКОПИРОВАВ" in system


def test_chunk_prompt_preserves_timecodes_for_the_reduce_pass():
    """Заметки по фрагменту должны нести таймкоды: при сведении транскрипта уже
    не будет, и восстановить их будет неоткуда."""
    system, _ = prompts.build_chunk_prompt(
        mode="work", title="Т", part=1, total=2, transcript="[00:10] Вы: раз",
    )
    assert "таймкода" in system and "скопированного" in system
    assert "### ПРЕДЛОЖЕНО" in system


def test_chunk_notes_come_in_fixed_sections():
    """Заметки по фрагменту выдаются разделами, а не свободным списком.

    Свободный список заставлял сведение решать, что важно, — и оно выбрасывало
    целые куски разговора. С разделами переносить нечего решать.
    """
    system, _ = prompts.build_chunk_prompt(
        mode="work", title="Т", part=1, total=2, transcript="[00:10] Вы: раз",
    )
    for section in ("### РЕШЕНО", "### ПРЕДЛОЖЕНО", "### СДЕЛАНО", "### ФАКТЫ", "### ВОПРОСЫ"):
        assert section in system
    assert "«нет»" in system  # пустой раздел разрешён, выдумывать в него не надо


def test_reduce_is_told_to_carry_over_not_retell():
    """Сведение переносит помеченное, а не пересказывает.

    Заметки уже прошли одно сжатие; второе стирало именно то, ради чего протокол
    и составляют — на встрече 03.08.2026 так пропало замечание куратора про
    коммит на 4500 строк, при том что в заметках фрагмента оно было.
    """
    system, _ = prompts.build_reduce_prompt(
        mode="work", title="Т", date="", participants="", notes="х", detailed=True,
    )
    assert "ПЕРЕНЕСТИ, а не пересказать" in system
    assert "Ни одного не выбрасывай" in system
    assert "В принятые решения это не переводится никогда" in system
    assert "Своими словами пиши только «Краткий итог»" in system
