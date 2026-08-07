import { useEffect, useRef, useState } from "react";
import type { Page } from "../App";
import { LiveClient } from "../api/live";
import {
  AudioEngine,
  type CaptureHandle,
  type SystemSource,
  listAudioInputs,
  looksLikeLoopback,
} from "../audio/capture";
import Transcript, { formatTime } from "../components/Transcript";
import { isDebugMode, platform } from "../store";
import {
  MEETING_MODE_LABELS,
  type HealthDto,
  type LiveEvent,
  type MeetingMode,
  type SegmentDto,
} from "../types";

type Phase = "setup" | "starting" | "live" | "stopping";

function LevelMeter({ label, value }: { label: string; value: number }) {
  const width = Math.min(100, Math.round(value * 260)); // RMS речи ~0.05–0.3
  return (
    <div className="level-meter">
      <span>{label}</span>
      <div className="level-track">
        <div className="level-fill" style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

export default function LivePage({
  navigate,
  health,
  onPhaseChange,
}: {
  navigate: (page: Page) => void;
  health: HealthDto | null;
  onPhaseChange?: (phase: string | null) => void;
}) {
  const [phase, setPhase] = useState<Phase>("setup");
  const [title, setTitle] = useState("");
  const [recordAudio, setRecordAudio] = useState(false);
  const [summarizeWanted, setSummarizeWanted] = useState(true);
  const [hintsWanted, setHintsWanted] = useState(false);
  const [meetingMode, setMeetingMode] = useState<MeetingMode>("work");
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [micId, setMicId] = useState("");
  // "auto" — автозахват через Electron (без драйверов), "" — выключен, иначе deviceId
  const [sysId, setSysId] = useState(platform() !== "web" ? "auto" : "");
  const [error, setError] = useState("");
  const [warning, setWarning] = useState("");
  const [segments, setSegments] = useState<SegmentDto[]>([]);
  const [hintList, setHintList] = useState<{ text: string; at: string }[]>([]);
  const [hintError, setHintError] = useState("");
  const [hintsOn, setHintsOn] = useState(false); // подсказки включены прямо сейчас (тумблер)
  // Чат с моделью: вопрос участника и ответы. Отдельно от подсказок — там модель
  // говорит сама, здесь спрашивает человек.
  const [chatLog, setChatLog] = useState<{ role: "you" | "model"; text: string }[]>([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [pickedIds, setPickedIds] = useState<Set<number>>(new Set());
  const [micLevel, setMicLevel] = useState(0);
  const [sysLevel, setSysLevel] = useState(0);
  const [sysActive, setSysActive] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [liveTitle, setLiveTitle] = useState("");

  const clientRef = useRef<LiveClient | null>(null);
  const engineRef = useRef<AudioEngine | null>(null);
  const handlesRef = useRef<CaptureHandle[]>([]);
  const meetingIdRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const finishedRef = useRef(false);
  const chatRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true); // стоит ли человек внизу ленты (см. onChatScroll)
  const phaseRef = useRef(phase);
  phaseRef.current = phase;

  const isElectron = platform() !== "web";

  useEffect(() => {
    onPhaseChange?.(phase === "setup" || phase === "stopping" ? null : phase);
  }, [phase, onPhaseChange]);

  useEffect(() => {
    listAudioInputs().then((list) => {
      setDevices(list);
      if (!isElectron) {
        // в браузере автозахвата нет — сразу предлагаем виртуальный кабель
        const loopback = list.find(looksLikeLoopback);
        if (loopback) setSysId(loopback.deviceId);
      }
    });
  }, [isElectron]);

  // Автопрокрутка только когда человек и так стоял внизу. Иначе она утаскивает
  // экран ровно тогда, когда он отлистал вверх выделить реплику для вопроса.
  //
  // Положение запоминаем в обработчике прокрутки, а не в эффекте: эффект
  // выполняется уже ПОСЛЕ вставки новой реплики, и длинное сообщение само
  // вытолкнуло бы себя из зоны «внизу».
  useEffect(() => {
    const el = chatRef.current;
    if (el && stickToBottomRef.current) el.scrollTop = el.scrollHeight;
  }, [segments]);

  function togglePicked(id: number) {
    setPickedIds((previous) => {
      const next = new Set(previous);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function ask() {
    const text = question.trim();
    if ((!text && pickedIds.size === 0) || asking) return;
    // «реплик: N» вместо «N реплик» — обходит склонение, как уже сделано
    // в истории встреч (HistoryPage)
    const shown = text || `Объясни выделенное — реплик: ${pickedIds.size}`;
    setChatLog((previous) => [...previous, { role: "you", text: shown }]);
    clientRef.current?.ask(text, [...pickedIds]);
    setQuestion("");
    setPickedIds(new Set()); // выделение одноразовое: вопрос задан, лента снова живая
    setAsking(true);
  }

  function onChatScroll() {
    const el = chatRef.current;
    if (el) stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }

  useEffect(() => () => {
    if (clientRef.current?.connected && (phaseRef.current === "live" || phaseRef.current === "starting")) {
      clientRef.current?.stop();
    }
    cleanup();
    onPhaseChange?.(null);
  }, []);

  function cleanup() {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    handlesRef.current.forEach((handle) => handle.stop());
    handlesRef.current = [];
    engineRef.current?.close();
    engineRef.current = null;
    clientRef.current?.close();
    clientRef.current = null;
    setMicLevel(0);
    setSysLevel(0);
    setSysActive(false);
  }

  function finish(meetingId: number | null) {
    if (finishedRef.current) return;
    finishedRef.current = true;
    cleanup();
    setPhase("setup");
    if (meetingId != null) navigate({ name: "meeting", id: meetingId });
  }

  function onEvent(event: LiveEvent) {
    switch (event.type) {
      case "ready":
        meetingIdRef.current = event.meeting_id;
        setLiveTitle(event.title);
        void startCaptures();
        break;
      case "segment":
        setSegments((previous) => [...previous, event.segment]);
        break;
      case "hint":
        setHintError(""); // подсказка пришла — снимаем баннер прошлой ошибки
        setHintList((previous) => [
          ...previous,
          { text: event.text, at: new Date().toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" }) },
        ]);
        break;
      case "hint_error":
        setHintError(event.message);
        break;
      case "answer":
        setAsking(false);
        setChatLog((previous) => [...previous, { role: "model", text: event.text }]);
        break;
      case "answer_error":
        setAsking(false);
        setChatLog((previous) => [
          ...previous,
          { role: "model", text: `⚠ ${event.message}` },
        ]);
        break;
      case "stopped":
        finish(event.meeting_id);
        break;
      case "error":
        setError(event.message);
        break;
    }
  }

  function onWsClose() {
    if (finishedRef.current) return;
    if (phaseRef.current === "stopping") {
      finish(meetingIdRef.current);
    } else if (phaseRef.current === "live" || phaseRef.current === "starting") {
      setError("Соединение с сервером прервано. Уже распознанная часть встречи сохранена.");
      cleanup();
      setPhase("setup");
    }
  }

  async function startCaptures() {
    const engine = engineRef.current!; // микрофон уже захвачен в start()
    if (sysId !== "") {
      try {
        const source: SystemSource =
          sysId === "auto" ? { kind: "auto" } : { kind: "device", deviceId: sysId };
        const system = await engine.startSystem(source, {
          onChunk: (pcm) => clientRef.current?.sendAudio(1, pcm),
          onLevel: setSysLevel,
        });
        handlesRef.current.push(system);
        setSysActive(true);
      } catch (exc) {
        setWarning(`Системный звук не захвачен (${(exc as Error).message}) — пишем только микрофон.`);
      }
    }
    const startedAt = Date.now();
    timerRef.current = setInterval(
      () => setElapsed(Math.floor((Date.now() - startedAt) / 1000)),
      1000,
    );
    setPhase("live");
  }

  async function start() {
    setError("");
    setWarning("");
    setHintError("");
    setSegments([]);
    setHintList([]);
    setHintsOn(hintsWanted); // стартовое состояние тумблера = выбор на экране настройки
    setElapsed(0);
    finishedRef.current = false;
    meetingIdRef.current = null;
    setPhase("starting");

    // Микрофон захватываем ДО создания встречи: если его нет (нет разрешения,
    // устройство отвалилось), в истории не должна оставаться пустая встреча
    const engine = new AudioEngine();
    engineRef.current = engine;
    try {
      const mic = await engine.startMic(micId || undefined, {
        onChunk: (pcm) => clientRef.current?.sendAudio(0, pcm),
        onLevel: setMicLevel,
      });
      handlesRef.current.push(mic);
    } catch (exc) {
      setError(`Микрофон недоступен: ${(exc as Error).message}`);
      cleanup();
      setPhase("setup");
      return;
    }

    const client = new LiveClient(onEvent, onWsClose);
    clientRef.current = client;
    try {
      await client.connect();
    } catch (exc) {
      setError(
        `${(exc as Error).message}. Проверьте, что сервер запущен, и адрес в настройках верный.`,
      );
      cleanup();
      setPhase("setup");
      return;
    }
    client.start({
      title: title.trim() || "Встреча",
      record_audio: recordAudio,
      hints: hintsWanted,
      summarize: summarizeWanted,
      meeting_mode: meetingMode,
    });
  }

  function stop() {
    setPhase("stopping");
    handlesRef.current.forEach((handle) => handle.stop());
    handlesRef.current = [];
    if (clientRef.current?.connected) {
      clientRef.current.stop(); // ответ придёт событием "stopped"
    } else {
      finish(meetingIdRef.current);
    }
  }

  // ---------------------------------------------------------------- setup

  if (phase === "setup" || phase === "starting") {
    return (
      <div className="content">
        <div className="setup-grid">
          <h1>Новая встреча</h1>
          <p className="page-sub">
            Речь с микрофона и из звонка превращается в текст в реальном времени.
            Все данные обрабатываются локально.
          </p>
          {error && <div className="banner error">{error}</div>}
          {!health && (
            <div className="banner warn">
              Сервер недоступен — запустите его или проверьте адрес в настройках.
            </div>
          )}
          <div className="card">
            <label className="field">
              <span>Название встречи</span>
              <input
                className="input"
                placeholder="Например: Планёрка отдела"
                value={title}
                onChange={(event) => setTitle(event.target.value)}
              />
            </label>
            <div className="setup-row">
              <label className="field">
                <span>Микрофон</span>
                <select
                  className="input"
                  value={micId}
                  onChange={(event) => setMicId(event.target.value)}
                >
                  <option value="">По умолчанию</option>
                  {devices
                    .filter((device) => !looksLikeLoopback(device))
                    .map((device) => (
                      <option key={device.deviceId} value={device.deviceId}>
                        {device.label || "Микрофон"}
                      </option>
                    ))}
                </select>
              </label>
              <label className="field">
                <span>Системный звук (звонок, видео)</span>
                <select
                  className="input"
                  value={sysId}
                  onChange={(event) => setSysId(event.target.value)}
                >
                  {isElectron && <option value="auto">Автоматически — звук системы</option>}
                  <option value="">— выключен —</option>
                  {devices.map((device) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || "Устройство"}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {sysId !== "" && (
              <div className="banner warn" style={{ marginTop: 10 }}>
                🎧 В наушниках надёжнее: звук звонка из колонок попадает в микрофон,
                смешивается с вашим голосом и мешает точно определять, кто говорит.
              </div>
            )}
            <details className="help">
              <summary>Про захват системного звука</summary>
              <p style={{ margin: "8px 0 0" }}>
                «Автоматически» работает в приложении на macOS 13+ и Windows без драйверов:
                при первом запуске macOS попросит разрешить{" "}
                <b>«Запись экрана и звука системы»</b> — разрешите и перезапустите приложение.
              </p>
              <p style={{ margin: "8px 0 0" }}>
                Запасной вариант для старых macOS или запуска в браузере — виртуальный
                аудиокабель:
              </p>
              <ol>
                <li>Установите драйвер: <code>brew install blackhole-2ch</code></li>
                <li>
                  В «Настройка Audio-MIDI» создайте «Устройство с несколькими выходами»
                  (динамики + BlackHole 2ch) и назначьте его выходом звука.
                </li>
                <li>Здесь выберите «BlackHole 2ch» как источник системного звука.</li>
              </ol>
            </details>
            <label className="field" style={{ marginTop: 14 }}>
              <span>Тип встречи</span>
              <select
                className="input"
                value={meetingMode}
                onChange={(event) => setMeetingMode(event.target.value as MeetingMode)}
              >
                {(Object.keys(MEETING_MODE_LABELS) as MeetingMode[]).map((mode) => (
                  <option key={mode} value={mode}>
                    {MEETING_MODE_LABELS[mode]}
                  </option>
                ))}
              </select>
              <span className="hint">
                влияет на то, что подсказывает ИИ и из каких разделов состоит протокол
              </span>
            </label>
            <div style={{ marginTop: 10 }}>
              <label className="check">
                <input
                  type="checkbox"
                  checked={recordAudio}
                  onChange={(event) => setRecordAudio(event.target.checked)}
                />
                <span className="box">✓</span>
                Записывать аудио встречи
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={summarizeWanted}
                  onChange={(event) => setSummarizeWanted(event.target.checked)}
                />
                <span className="box">✓</span>
                Составить протокол по завершении
                <span className="hint">если выключить — резюме можно создать позже на странице встречи</span>
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={hintsWanted}
                  onChange={(event) => setHintsWanted(event.target.checked)}
                />
                <span className="box">✓</span>
                Подсказки ИИ во время встречи
                <span className="hint">экспериментально, нагружает память</span>
              </label>
            </div>
            <div style={{ marginTop: 18, display: "flex", gap: 12, alignItems: "center" }}>
              <button
                className="btn primary big"
                onClick={start}
                disabled={phase === "starting" || !health || !health.asr.loaded}
              >
                {phase === "starting" ? <span className="spinner" /> : "▶"} Начать встречу
              </button>
              {health && !health.asr.loaded && (
                <span style={{ color: "var(--muted)", fontSize: 13 }}>
                  <span className="spinner" /> модель распознавания загружается — обычно до минуты
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------- live

  return (
    <div className="live-layout">
      <div className="live-header">
        <span className="rec-dot" />
        <span className="live-title">{liveTitle}</span>
        {meetingMode !== "work" && (
          <span className="chip">{MEETING_MODE_LABELS[meetingMode].split(" (")[0]}</span>
        )}
        <span className="live-timer">{formatTime(elapsed)}</span>
        <LevelMeter label="Микрофон" value={micLevel} />
        {sysActive && <LevelMeter label="Система" value={sysLevel} />}
        <span className="live-spacer" />
        <button className="btn danger" onClick={stop} disabled={phase === "stopping"}>
          {phase === "stopping" ? <span className="spinner" /> : "■"} Завершить
        </button>
      </div>
      {(error || warning) && (
        <div style={{ padding: "12px 24px 0" }}>
          {error && <div className="banner error">{error}</div>}
          {warning && (
            <div className="banner warn">
              {warning}
              {warning.includes("Запись экрана") && (
                <button
                  className="btn small"
                  style={{ marginLeft: 10 }}
                  onClick={() => window.stenograf?.openScreenSettings?.()}
                >
                  Открыть настройки
                </button>
              )}
            </div>
          )}
        </div>
      )}
      <div className="live-main">
        <div className="live-chat" ref={chatRef} onScroll={onChatScroll}>
          {segments.length > 0 ? (
            <Transcript
              segments={segments}
              debug={isDebugMode()}
              selectedIds={pickedIds}
              onToggle={togglePicked}
            />
          ) : (
            <div className="empty">
              <div className="big-icon">🎙️</div>
              Говорите — распознанные реплики появятся здесь
            </div>
          )}
        </div>
        {/* Панель показывается всегда, а не только когда подсказки выбрали до
            старта: решение «а не спросить ли модель» приходит посреди встречи,
            и тогда переиграть выбор было уже нельзя. */}
        <div className="hints-panel">
          <div className="hints-title">
            💡 Подсказки ИИ
            <span className="hint" style={{ fontWeight: 400, marginLeft: 8 }}>
              молчит, когда сказать нечего
            </span>
          </div>
          <div style={{ display: "flex", gap: 10, alignItems: "center", margin: "8px 0 12px" }}>
            <label className="check" style={{ margin: 0 }}>
              <input
                type="checkbox"
                checked={hintsOn}
                onChange={(event) => {
                  setHintsOn(event.target.checked);
                  clientRef.current?.setHints(event.target.checked);
                }}
              />
              <span className="box">✓</span>
              Включены
            </label>
            <button className="btn small" onClick={() => clientRef.current?.requestHint()}>
              Подсказать сейчас
            </button>
          </div>
          {hintError && <div className="banner warn">{hintError}</div>}
          {hintList.length === 0 && !hintError && (
            <div className="empty" style={{ padding: "20px 8px" }}>
              {hintsOn
                ? "Модель слушает разговор…"
                : "Подсказки выключены. Включите тумблер, чтобы модель подсказывала сама, или спросите разово кнопкой."}
            </div>
          )}
          {[...hintList].reverse().slice(0, 8).map((hint, index) => (
            <div className="hint-card" key={`${hint.at}-${index}`}>
              {hint.text}
              <div className="hint-time">{hint.at}</div>
            </div>
          ))}

          <div className="ask-block">
            <div className="hints-title">
              💬 Спросить модель
              <span className="hint" style={{ fontWeight: 400, marginLeft: 8 }}>
                видит разговор целиком
              </span>
            </div>
            {chatLog.length === 0 && (
              <div className="empty" style={{ padding: "14px 8px" }}>
                Задайте вопрос или отметьте реплики кнопкой «?» и спросите про них.
              </div>
            )}
            {chatLog.map((message, index) => (
              <div className={`ask-msg ${message.role}`} key={index}>
                {message.text}
              </div>
            ))}
            {asking && (
              <div className="ask-msg model">
                <span className="spinner" /> Думает…
              </div>
            )}
            {pickedIds.size > 0 && (
              <div className="ask-picked">
                Выделено реплик: {pickedIds.size}
                <button className="btn small" onClick={() => setPickedIds(new Set())}>
                  Снять
                </button>
              </div>
            )}
            <div className="ask-row">
              <input
                className="input"
                value={question}
                placeholder={pickedIds.size > 0 ? "Вопрос про выделенное…" : "Ваш вопрос…"}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => event.key === "Enter" && ask()}
              />
              <button
                className="btn small"
                onClick={ask}
                disabled={asking || (!question.trim() && pickedIds.size === 0)}
              >
                Спросить
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
