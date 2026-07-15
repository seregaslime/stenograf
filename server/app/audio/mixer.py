"""Микшер каналов: микрофон + системный звук → один поток единой шкалы времени.

Зачем: распознавание и VAD работают одним проходом по смешанному звуку
(совет куратора), а эхо из колонок перестаёт создавать дубли — копия голоса
в миксе совпадает по времени с оригиналом. При этом сырые каналы микшер
запоминает в скользящем окне: по энергии (RMS) каждого канала внутри окна
речевого сегмента диаризация понимает, откуда пришёл голос — из комнаты
(микрофон) или из звонка (система).

Выравнивание: оба потока идут с одной машины с одинаковой частотой, поэтому
достаточно совмещать их по количеству семплов. Если один канал отстаёт больше
чем на max_lag (сеть икнула или источник замолчал навсегда), недостающее
заполняется тишиной, а «долг» канала запоминается — опоздавшие семплы затем
отбрасываются, чтобы шкала не разъезжалась.
"""
import numpy as np

from ..config import SAMPLE_RATE, Settings

CHANNELS = ("mic", "system")


class ChannelMixer:
    def __init__(self, cfg: Settings, history_s: float = 40.0):
        self._max_lag = int(cfg.mixer_max_lag_ms / 1000 * SAMPLE_RATE)
        self._dominance_ratio = cfg.speaker_channel_dominance
        self._buffers: dict[str, np.ndarray] = {c: np.empty(0, dtype=np.float32) for c in CHANNELS}
        self._debt: dict[str, int] = {c: 0 for c in CHANNELS}  # сколько опоздавших семплов выкинуть
        self._active: set[str] = set()
        # Холодный старт: ждём второй канал (или max_lag), чтобы шкалы совпали с нуля
        self._started = False
        # История выровненных каналов для RMS-доминанты (единая шкала с миксом)
        self._history_cap = int(history_s * SAMPLE_RATE)
        self._history: dict[str, np.ndarray] = {c: np.empty(0, dtype=np.float32) for c in CHANNELS}
        self._history_start = 0  # номер первого семпла истории в единой шкале
        self._emitted = 0        # всего семплов выдано в микс

    def feed(self, channel: str, chunk: np.ndarray) -> list[np.ndarray]:
        """Принимает чанк канала, возвращает готовые куски смешанного потока."""
        self._active.add(channel)
        if self._debt[channel]:
            drop = min(self._debt[channel], len(chunk))
            self._debt[channel] -= drop
            chunk = chunk[drop:]
        if len(chunk):
            self._buffers[channel] = np.concatenate([self._buffers[channel], chunk])
        return self._drain()

    def flush(self) -> list[np.ndarray]:
        """Конец встречи: выдать всё накопленное, дополнив отставший канал тишиной."""
        return self._drain(force=True)

    def _drain(self, force: bool = False) -> list[np.ndarray]:
        lengths = [len(self._buffers[c]) for c in self._active]
        if not lengths:
            return []
        longest = max(lengths)
        if not self._started:
            # оба канала на месте (обычный случай: приходят в первые же кадры)
            # или второго не будет — тогда не ждём его дольше max_lag
            if len(self._active) == len(CHANNELS) or longest > self._max_lag or force:
                self._started = True
            else:
                return []
        if force:
            ready = longest
        elif len(self._active) < len(CHANNELS):
            ready = longest  # единственный канал идёт как есть
        else:
            ready = min(lengths)
            if longest - ready > self._max_lag:
                # отстающий канал молчит слишком долго — не задерживаем распознавание
                ready = longest - self._max_lag
        if ready == 0:
            return []

        mixed = np.zeros(ready, dtype=np.float32)
        for channel in CHANNELS:
            buffer = self._buffers[channel]
            take = min(ready, len(buffer)) if channel in self._active else 0
            aligned = np.zeros(ready, dtype=np.float32)
            if take:
                aligned[:take] = buffer[:take]
                self._buffers[channel] = buffer[take:]
            if channel in self._active and take < ready:
                self._debt[channel] += ready - take  # канал дополнен тишиной — «долг»
            mixed += aligned
            self._history[channel] = np.concatenate([self._history[channel], aligned])

        # подрезаем историю до окна
        overflow = len(self._history[CHANNELS[0]]) - self._history_cap
        if overflow > 0:
            for channel in CHANNELS:
                self._history[channel] = self._history[channel][overflow:]
            self._history_start += overflow

        self._emitted += ready
        return [np.clip(mixed, -1.0, 1.0)]

    # ------------------------------------------------------------- доминанта

    def dominance(self, start_s: float, end_s: float) -> str:
        """Чей голос в окне сегмента: 'mic' / 'system' / 'mixed' (непонятно/оба)."""
        if len(self._active) < 2:
            return next(iter(self._active), "mixed")
        lo = max(int(start_s * SAMPLE_RATE) - self._history_start, 0)
        hi = max(int(end_s * SAMPLE_RATE) - self._history_start, 0)
        rms = {}
        for channel in CHANNELS:
            window = self._history[channel][lo:hi]
            rms[channel] = float(np.sqrt(np.mean(window**2))) if len(window) else 0.0
        mic, system = rms["mic"], rms["system"]
        if mic > system * self._dominance_ratio:
            return "mic"
        if system > mic * self._dominance_ratio:
            return "system"
        return "mixed"

    def dominance_spans(
        self, start_s: float, end_s: float, window_s: float, min_run_s: float,
    ) -> list[tuple[float, float]]:
        """Участки стабильной доминанты внутри сегмента: [(start, end), ...].

        Один участок — резать нечего. Окна с 'mixed' (наложение голосов, тихое
        место) наследуют метку соседа, а серии короче min_run_s вливаются в
        соседнюю: кашель в микрофон посреди чужой фразы — не смена говорящего.
        """
        n_windows = int((end_s - start_s) / window_s)
        if len(self._active) < 2 or n_windows < 2:
            return [(start_s, end_s)]

        labels = []
        for i in range(n_windows):
            lo = start_s + i * window_s
            hi = end_s if i == n_windows - 1 else lo + window_s  # хвост — в последнее окно
            labels.append(self.dominance(lo, hi))
        for i in range(1, n_windows):
            if labels[i] == "mixed":
                labels[i] = labels[i - 1]
        for i in range(n_windows - 2, -1, -1):
            if labels[i] == "mixed":
                labels[i] = labels[i + 1]
        if labels[0] == "mixed":  # весь сегмент неразличим
            return [(start_s, end_s)]

        runs: list[list] = []  # [метка, число окон]
        for label in labels:
            if runs and runs[-1][0] == label:
                runs[-1][1] += 1
            else:
                runs.append([label, 1])

        min_windows = max(1, round(min_run_s / window_s))
        while len(runs) > 1:
            shortest = min(range(len(runs)), key=lambda i: runs[i][1])
            if runs[shortest][1] >= min_windows:
                break
            neighbours = [i for i in (shortest - 1, shortest + 1) if 0 <= i < len(runs)]
            target = max(neighbours, key=lambda i: runs[i][1])
            runs[target][1] += runs[shortest][1]
            del runs[shortest]
            merged: list[list] = []  # слияние могло уравнять метки соседей
            for label, count in runs:
                if merged and merged[-1][0] == label:
                    merged[-1][1] += count
                else:
                    merged.append([label, count])
            runs = merged

        spans = []
        edge, seen = start_s, 0
        for _, count in runs:
            seen += count
            cut = end_s if seen == n_windows else start_s + seen * window_s
            spans.append((edge, cut))
            edge = cut
        return spans
