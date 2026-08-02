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
