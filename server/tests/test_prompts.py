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


def test_every_mode_has_label_and_focus():
    """У каждого режима заполнены название и фокус подсказок.

    Разделов протокола у режимов больше нет: протокол один на все типы,
    а тип встречи попадает в промпт строкой в шапке.
    """
    for mode in prompts.MODES.values():
        assert mode.label and mode.hint_focus


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


# ------------------------------------------------------------------ промпт протокола

def _protocol(**over):
    params = dict(mode="work", title="Планёрка", date="03.08.2026",
                  participants="Вы (5 реплик)", text="[00:10] Вы: раз")
    params.update(over)
    return prompts.build_protocol_prompt(**params)


def test_protocol_sections_are_the_same_for_every_mode():
    """Разделы протокола больше не зависят от типа встречи.

    Своя вёрстка под каждый тип стоила 300–500 символов промпта, а при лимите
    8000 токенов в минуту это столько же символов разговора, которые не влезли.
    Тип встречи остался строкой в шапке — модель его видит.
    """
    prompts_by_mode = {key: _protocol(mode=key)[1] for key in prompts.MODES}
    sections = [p.split("Расшифровка:")[0].split("формате:")[1] for p in prompts_by_mode.values()]
    assert len(set(sections)) == 1                       # разделы одни на всех
    assert "Тип встречи: рабочая встреча" in prompts_by_mode["work"]
    assert "Тип встречи: переговоры" in prompts_by_mode["negotiation"]


def test_protocol_forbids_the_ways_to_lie():
    """Запреты против выдумок — каждый из разбора реальных протоколов.

    03.08.2026: модель выдумала все шесть таймкодов, записала уже проделанное
    нагрузочное тестирование в планы, превратила отвергнутое предложение в
    решение. 09.08.2026: дважды назначила ответственным куратора, который
    задачу попросил, — поэтому ответственных не пишем вовсе.
    """
    system, _ = _protocol()
    assert "Таймкод ставь только скопированный" in system
    assert "результат, а не" in system
    assert "задачей не" in system and "становится" in system
    assert "Ответственных и сроки не пиши вообще" in system
    assert "не достраивай" in system


def test_protocol_has_no_decisions_section():
    """Раздела решений нет намеренно: модель ставила туда описания и
    рекомендации через раз, а неверное «принятое решение» в протоколе хуже
    отсутствующего. Стоит обсудить с куратором — в исходном задании он был."""
    _, prompt = _protocol()
    assert "Принятые решения" not in prompt
    assert "Ответственный" not in prompt
    for section in ("## Краткий итог", "## Ключевые темы", "## Задачи", "## Открытые вопросы"):
        assert section in prompt


def test_fragment_is_marked_as_a_fragment():
    """Фрагмент длинной встречи помечается, чтобы модель не искала в нём итогов."""
    _, whole = _protocol()
    _, part = _protocol(part=2, total=5)
    assert "фрагмент 2 из 5" in part
    assert "фрагмент" not in whole


def test_same_prompt_serves_the_merge():
    """Склейка протоколов фрагментов идёт тем же промптом: для него это просто
    текст, по которому надо составить протокол."""
    _, merge = _protocol(text="— Фрагмент 1 —\n## Принятые решения\n- [05:18] что-то")
    assert "Составь протокол" in merge
    assert "Фрагмент 1" in merge


def test_protocol_warns_about_asr_noise():
    """Распознавание русское и коверкает английские термины: «UDP» приходит как
    «уд». Без предупреждения модель честно объясняет, что такое «уд».
    """
    system, _ = _protocol()
    assert "распознаванием речи" in system and "UDP" in system and "не угадывай" in system


def test_both_prompts_warn_about_asr_noise():
    """То же предупреждение и в подсказках, и в протоколе."""
    hint_system, _ = _hint()
    protocol_system, _ = _protocol()
    for system in (hint_system, protocol_system):
        assert "распознаванием речи" in system
        assert "UDP" in system
        assert "не угадывай" in system
