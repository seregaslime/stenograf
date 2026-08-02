"""Движок подсказок во время встречи: чистые функции (дедуп, решение триггера)
и поведение _emit_hint (отправка, дедуп, бэкофф при ошибках, «подсказать сейчас»).
Без сети и без реального LLM — через фейковый роутер.
"""
import asyncio

from app.llm.base import LlmError
from app.llm.router import Budget
from app.ws import LiveSession


def _session(cfg, llm=None):
    s = LiveSession(
        ws=None, cfg=cfg, transcriber=None, embedder=None,
        registry=None, llm=llm, on_meeting_ended=lambda mid: None,
    )
    s._sent = []

    async def _capture(payload):
        s._sent.append(payload)

    s._send = _capture  # перехватываем исходящие события вместо WebSocket
    return s


class _FakeLLM:
    # Ответ по умолчанию — правдоподобная подсказка: короче hints_min_len_chars
    # parse_hint считает молчанием, и тест проверял бы не то.
    def __init__(self, reply="Уточните срок задачи и ответственного.", fail=False,
                 detailed=False):
        self.reply = reply
        self.fail = fail
        self.calls = 0
        self.seen: list[tuple[str, str]] = []  # (system, prompt) последних вызовов
        self._detailed = detailed

    @property
    def budget(self) -> Budget:
        return Budget(12_000, 2500, self._detailed)

    async def hint(self, prompt, system=None, temperature=0.5):
        self.calls += 1
        self.seen.append((system or "", prompt))
        if self.fail:
            raise LlmError("нет связи с LLM")
        return self.reply


def _fill_context(s):
    s._recent.extend(f"Спикер: реплика номер {i} с достаточным объёмом текста" for i in range(6))


# ------------------------------------------------------------------ чистые функции

def test_is_duplicate_catches_near_repeat(cfg):
    """Почти дословный повтор недавней подсказки считается дублем (регистр и знаки не важны), а другая по смыслу — нет.
    """
    s = _session(cfg)
    s._recent_hints.append("Уточните сроки запуска беты и критерии готовности.")
    assert s._is_duplicate("уточните Сроки запуска беты и критерии готовности!")
    assert not s._is_duplicate("Кто отвечает за тестирование авторизации?")


def test_is_duplicate_empty_memory(cfg):
    """При пустой памяти подсказок дублей быть не может."""
    assert not _session(cfg)._is_duplicate("Любая подсказка")


def test_should_hint_needs_enough_new_text(cfg):
    """Подсказка не запускается, пока не накопилось достаточно нового разговора."""
    s = _session(cfg)
    s._last_hint_at = 1000.0 - cfg.hints_min_gap_s
    s._chars_since_hint = cfg.hints_min_new_chars - 1
    assert not s._should_hint(1000.0)
    s._chars_since_hint = cfg.hints_min_new_chars
    assert s._should_hint(1000.0)


def test_should_hint_respects_min_gap(cfg):
    """Между подсказками выдерживается минимальный интервал, даже если текста много."""
    s = _session(cfg)
    s._chars_since_hint = cfg.hints_min_new_chars * 5
    s._last_hint_at = 1000.0 - cfg.hints_min_gap_s / 2
    assert not s._should_hint(1000.0)


def test_should_hint_respects_backoff(cfg):
    """В период бэкоффа после ошибок LLM подсказки не запрашиваются."""
    s = _session(cfg)
    s._chars_since_hint = cfg.hints_min_new_chars * 5
    s._last_hint_at = 0.0
    s._hint_backoff_until = 2000.0
    assert not s._should_hint(1000.0)
    assert s._should_hint(2000.0)


# ------------------------------------------------------------------ поведение _emit_hint

def test_emit_hint_sends_then_dedups(cfg):
    """Первая подсказка доходит до клиента, повторная с тем же текстом отсеивается дедупом."""
    llm = _FakeLLM("Уточните сроки и ответственных.")
    s = _session(cfg, llm)
    _fill_context(s)
    asyncio.run(s._emit_hint())
    asyncio.run(s._emit_hint())  # тот же ответ — второй раз не шлём
    hints = [m for m in s._sent if m["type"] == "hint"]
    assert len(hints) == 1
    assert llm.calls == 2  # LLM звали дважды, дубль отсеян на клиентской стороне


def test_emit_hint_error_is_not_permanent(cfg):
    """Одна ошибка LLM не выключает подсказки: наращивается бэкофф и шлётся одно уведомление."""
    s = _session(cfg, _FakeLLM(fail=True))
    _fill_context(s)
    s._hints_enabled = True
    asyncio.run(s._emit_hint())
    assert s._hint_fail_streak == 1
    assert s._hint_backoff_until > 0
    assert s._hints_enabled is True  # одна ошибка НЕ выключает подсказки
    assert sum(m["type"] == "hint_error" for m in s._sent) == 1


def test_emit_hint_disables_after_max_fails(cfg):
    """После серии ошибок подряд подсказки отключаются, чтобы не долбить недоступный сервис."""
    s = _session(cfg, _FakeLLM(fail=True))
    _fill_context(s)
    s._hints_enabled = True
    for _ in range(cfg.hints_max_fails):
        asyncio.run(s._emit_hint())
    assert s._hints_enabled is False


def test_hint_now_forces_even_when_disabled(cfg):
    """Кнопка «Подсказать сейчас» срабатывает даже при выключенных подсказках."""
    s = _session(cfg, _FakeLLM("Мгновенная подсказка."))
    _fill_context(s)
    s._hints_enabled = False
    asyncio.run(s._on_command({"type": "hint_now"}))
    assert any(m["type"] == "hint" for m in s._sent)


# ------------------------------------------------------------------ право промолчать (SKIP)

def test_skip_is_not_sent_to_client(cfg):
    """Модель промолчала — клиент не должен увидеть ничего."""
    s = _session(cfg, _FakeLLM("SKIP"))
    _fill_context(s)
    asyncio.run(s._emit_hint())
    assert not [m for m in s._sent if m["type"] == "hint"]
    assert not s._recent_hints  # молчание не занимает память подсказок


def test_skip_shortens_then_grows_the_gap(cfg):
    """После молчания пауза короче обычной, но серия молчаний наращивает её до потолка."""
    s = _session(cfg, _FakeLLM("SKIP"))
    _fill_context(s)
    asyncio.run(s._emit_hint())
    assert s._skip_streak == 1
    assert s._hint_gap_s == cfg.hints_skip_gap_s  # после SKIP ждём меньше обычного

    for _ in range(20):  # серия SKIP растит паузу, но не выше потолка
        asyncio.run(s._emit_hint())
    assert s._hint_gap_s == cfg.hints_skip_max_gap_s


def test_skip_does_not_spam_llm(cfg):
    """Сразу после SKIP новый запрос не уходит: нужна свежая порция текста."""
    s = _session(cfg, _FakeLLM("SKIP"))
    _fill_context(s)
    asyncio.run(s._emit_hint())
    now = s._last_hint_at
    assert not s._should_hint(now)
    s._chars_since_hint = cfg.hints_min_new_chars
    assert not s._should_hint(now)  # текст есть, но пауза не вышла
    assert s._should_hint(now + cfg.hints_skip_gap_s)


def test_skip_keeps_context_and_error_state(cfg):
    """SKIP — не ошибка: бэкофф не трогается, история реплик остаётся,
    подсказки не выключаются даже после длинной серии молчания."""
    s = _session(cfg, _FakeLLM("SKIP"))
    _fill_context(s)
    s._hints_enabled = True
    before = list(s._recent)
    for _ in range(cfg.hints_max_fails + 3):
        asyncio.run(s._emit_hint())
    assert list(s._recent) == before          # контекст жив
    assert s._hint_fail_streak == 0 and s._hint_backoff_until == 0.0
    assert s._hints_enabled is True           # молчание ≠ сбой связи


def test_real_hint_resets_skip_streak(cfg):
    """Первая же реальная подсказка обнуляет счётчик молчаний и возвращает обычную паузу."""
    llm = _FakeLLM("SKIP")
    s = _session(cfg, llm)
    _fill_context(s)
    asyncio.run(s._emit_hint())
    assert s._skip_streak == 1
    llm.reply = "Уточните срок задачи и ответственного."
    asyncio.run(s._emit_hint())
    assert s._skip_streak == 0 and s._hint_gap_s == cfg.hints_min_gap_s


# ------------------------------------------------------------------ кнопка «Подсказать сейчас»

def test_force_prompt_has_no_skip_rule(cfg):
    """В промпте по кнопке слова SKIP нет вовсе — на явный запрос модель молчать не вправе."""
    llm = _FakeLLM("Подсказка по кнопке для проверки.")
    s = _session(cfg, llm)
    _fill_context(s)
    asyncio.run(s._emit_hint(force=True))
    system, prompt = llm.seen[-1]
    assert "SKIP" not in system and "SKIP" not in prompt


def test_force_reports_instead_of_staying_silent(cfg):
    """Раньше кнопка молча ничего не делала — теперь всегда отвечает."""
    s = _session(cfg, _FakeLLM("SKIP"))
    _fill_context(s)
    asyncio.run(s._emit_hint(force=True))
    assert any(m["type"] == "hint_error" for m in s._sent)

    empty = _session(cfg, _FakeLLM())  # контекста нет вовсе
    asyncio.run(empty._emit_hint(force=True))
    assert any(m["type"] == "hint_error" for m in empty._sent)


def test_force_bypasses_dedup(cfg):
    """По кнопке подсказка доставляется, даже если совпадает с предыдущей."""
    s = _session(cfg, _FakeLLM("Одна и та же подсказка про сроки."))
    _fill_context(s)
    asyncio.run(s._emit_hint())            # авто — доставлено
    asyncio.run(s._emit_hint())            # авто-дубль — отсеян
    asyncio.run(s._emit_hint(force=True))  # по кнопке — доставлено несмотря на дубль
    assert sum(m["type"] == "hint" for m in s._sent) == 2


# ------------------------------------------------------------------ контекст промпта

def test_prompt_carries_meeting_context(cfg):
    """В промпт уходят название встречи, участники и режим — без этого модель не понимает темы.
    """
    llm = _FakeLLM("Уточните срок задачи и ответственного.")
    s = _session(cfg, llm)
    s._meeting_title = "Планёрка отдела"
    s._mode = "interview"
    s._participants.update({"Вы": 3, "Интервьюер": 5})
    _fill_context(s)
    asyncio.run(s._emit_hint())
    system, prompt = llm.seen[-1]
    assert "Планёрка отдела" in prompt
    assert "Интервьюер (5 реплик)" in prompt
    assert "кандидат" in system  # режим собеседования доехал до промпта


def test_window_follows_provider_budget(cfg):
    """Окно режется бюджетом провайдера, а не размером деки."""
    llm = _FakeLLM("Уточните срок задачи и ответственного.")
    s = _session(cfg, llm)
    s._recent.extend(f"Спикер: реплика {i} с текстом подлиннее" for i in range(300))
    asyncio.run(s._emit_hint())
    _, prompt = llm.seen[-1]
    assert len(prompt) < 20_000  # local-бюджет 2500 символов транскрипта
    assert "реплика 299" in prompt  # берётся хвост разговора


def test_short_question_reaches_the_trigger(cfg):
    """Регрессия на живой сценарий: человек задаёт короткий вопрос и замолкает,
    ожидая подсказку. При старом пороге 200 символов счётчик вставал на 16 и
    подсказка не приходила никогда — самый ценный повод оказывался недостижим.
    """
    s = _session(cfg)
    s._last_hint_at = 0.0
    s._chars_since_hint = len("а что такое SLA?")
    assert s._should_hint(cfg.hints_min_gap_s + 1)


def test_backchannel_does_not_trigger(cfg):
    """Поддакивания подсказку не запускают — иначе модель дёргалась бы на «угу»."""
    s = _session(cfg)
    s._last_hint_at = 0.0
    for filler in ("угу", "да", "ага, понятно"):
        s._chars_since_hint = len(filler)
        assert not s._should_hint(cfg.hints_min_gap_s + 1), filler


def test_gap_still_limits_rate(cfg):
    """Порог символов НЕ управляет частотой — её держит пауза. Иначе низкий
    порог означал бы запрос на каждую реплику."""
    s = _session(cfg)
    s._chars_since_hint = cfg.hints_min_new_chars * 10  # текста накопилось с избытком
    s._last_hint_at = 1000.0
    assert not s._should_hint(1000.0 + cfg.hints_min_gap_s / 2)  # рано
    assert s._should_hint(1000.0 + cfg.hints_min_gap_s)          # пора


# ------------------------------------------------------------------ граница «уже отвечено»

def _say(s, line: str) -> None:
    """Реплика в транскрипт — как это делает _process_segment."""
    s._recent.append(line)
    s._lines_total += 1


def test_second_hint_reacts_only_to_new_text(cfg):
    """Регрессия на живой сценарий: задал вопрос — получил ответ, задал второй —
    получил ответы на ОБА. Модель не знала, что первый уже закрыт, потому что
    её собственные подсказки в транскрипт не попадают."""
    llm = _FakeLLM("SLA — соглашение об уровне сервиса, около 43 минут простоя.")
    s = _session(cfg, llm)
    _say(s, "Сергей: коллеги, обсудим надёжность сервиса в следующем квартале")
    _say(s, "Сергей: а что такое SLA? часто слышу термин, но не понимаю точно")
    asyncio.run(s._emit_hint(force=True))

    _say(s, "Сергей: ладно, поехали дальше по задачам на эту неделю")
    _say(s, "Сергей: слушайте, а что такое CI/CD? тоже постоянно всплывает")
    asyncio.run(s._emit_hint())

    _, prompt = llm.seen[-1]
    context, new = prompt.split("НОВОЕ с прошлой подсказки")
    assert "что такое SLA" in context      # старый вопрос ушёл в контекст
    assert "ты подсказал:" in context      # и помечен как уже отвеченный
    assert "что такое CI/CD" in new        # реагировать надо только на новый
    assert "что такое SLA" not in new


def test_skip_moves_the_boundary(cfg):
    """Модель промолчала — значит текст посмотрела. Не сдвинуть границу означало
    бы гонять отвергнутый фрагмент по кругу, пока она не надумает подсказку."""
    s = _session(cfg, _FakeLLM("SKIP"))
    _fill_context(s)
    asyncio.run(s._emit_hint())
    assert s._hinted_at_line == s._lines_total


def test_llm_error_keeps_the_boundary(cfg):
    """При сбое связи модель текста не видела — граница обязана остаться,
    иначе реплики молча выпадут из рассмотрения навсегда."""
    s = _session(cfg, _FakeLLM(fail=True))
    _fill_context(s)
    before = s._hinted_at_line
    asyncio.run(s._emit_hint())
    assert s._hinted_at_line == before


def test_force_looks_at_whole_conversation(cfg):
    """Кнопка «Подсказать сейчас» игнорирует границу: человек попросил явно,
    значит смотрим на разговор целиком, даже если нового ничего не было."""
    llm = _FakeLLM("Стоит зафиксировать срок и ответственного по задаче.")
    s = _session(cfg, llm)
    _fill_context(s)
    asyncio.run(s._emit_hint())          # авто — сдвинет границу
    asyncio.run(s._emit_hint(force=True))  # кнопка — без деления
    assert "НОВОЕ с прошлой подсказки" not in llm.seen[-1][1]


def test_boundary_survives_deque_overflow(cfg):
    """Дека реплик переполняется и теряет старое слева. Счётчик монотонный,
    поэтому граница не съезжает — сохранённый индекс указывал бы не туда."""
    s = _session(cfg)
    for i in range(cfg.hints_recent_maxlen + 50):
        _say(s, f"Спикер: реплика номер {i} с достаточным объёмом текста")
    s._hinted_at_line = s._lines_total - 3   # подсказка была три реплики назад
    earlier, new = s._split_window(budget_chars=100_000, force=False)
    assert new.count("\n") == 2              # ровно три последние реплики
    assert f"реплика номер {s._lines_total - 1}" in new
    assert f"реплика номер {s._lines_total - 4}" in earlier


def test_force_keeps_hint_markers(cfg):
    """«Смотреть на весь разговор» не значит «забыть свои ответы».

    Первая реализация на force выбрасывала пометки о выданных подсказках, и
    модель снова видела закрытые вопросы открытыми — ровно та проблема, ради
    которой границу и делали.
    """
    llm = _FakeLLM("SLA — соглашение об уровне сервиса, около 43 минут простоя.")
    s = _session(cfg, llm)
    _say(s, "Сергей: обсудим надёжность нашего сервиса в этом квартале, коллеги")
    _say(s, "Сергей: а что такое SLA? часто слышу термин, но не понимаю смысл")
    asyncio.run(s._emit_hint(force=True))

    llm.reply = "CI/CD — автоматическая сборка и доставка кода по коммиту."
    _say(s, "Сергей: ладно, поехали дальше по задачам на эту неделю")
    _say(s, "Сергей: слушайте, а что такое CI/CD? тоже часто всплывает")
    asyncio.run(s._emit_hint(force=True))

    _, prompt = llm.seen[-1]
    assert "ты подсказал:" in prompt      # модель знает, что на SLA уже ответила
    assert "что такое CI/CD" in prompt    # и видит новый вопрос
