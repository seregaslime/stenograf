import { useEffect, useRef, useState } from "react";
import { api } from "../api/rest";
import { OllamaClient } from "../llm/ollama";
import { OpenAiClient } from "../llm/openai";
import { loadLlmSettings, saveLlmSettings } from "../llm/settings";
import { DEFAULT_SERVER_URL, getSetting, isDebugMode, platform, setSetting } from "../store";
import type { AsrStateDto, HealthDto, LlmModelInfo } from "../types";

const ENGINE_LABELS: Record<string, string> = {
  faster_whisper: "CPU (faster-whisper)",
  mlx: "GPU Metal (mlx)",
  gigaam: "GigaAM (Сбер)",
};

// Куратор жаловался, что «всё лагает»: у него всё считалось процессором при
// живой видеокарте. Пишем словами, а не кодом устройства.
const DEVICE_LABELS: Record<string, string> = {
  cuda: "видеокарта NVIDIA",
  mps: "GPU Apple (Metal)",
  cpu: "процессор",
};

const MODEL_HINTS: Record<string, string> = {
  tiny: "самая лёгкая, много ошибок на русском",
  base: "лёгкая, но заметно больше ошибок (имена, окончания)",
  small: "лучшее качество среди whisper",
  v3_e2e_rnnt: "точнее, с пунктуацией (только русский)",
  v3_e2e_ctc: "быстрее, с пунктуацией (только русский)",
};

/** Единственный поддерживаемый провайдер API: только он сообщает размер
 *  контекста модели, без которого нельзя отсеять непригодные. */
const API_BASE_URL_DEFAULT = "https://api.groq.com/openai/v1";

export default function SettingsPage({ onServerChange }: { onServerChange: () => void }) {
  const [url, setUrl] = useState(getSetting("serverUrl", DEFAULT_SERVER_URL));
  const [token, setToken] = useState(getSetting("serverToken"));
  const [debug, setDebug] = useState(isDebugMode());
  const [health, setHealth] = useState<HealthDto | null>(null);
  const [testError, setTestError] = useState("");
  const [testing, setTesting] = useState(false);

  const [asr, setAsr] = useState<AsrStateDto | null>(null);
  const [engine, setEngine] = useState("faster_whisper");
  const [model, setModel] = useState("small");
  const [asrError, setAsrError] = useState("");
  const [applying, setApplying] = useState(false);
  const unmounted = useRef(false);

  const [tpmLimits, setTpmLimits] = useState<Record<string, number>>({});
  const [embedModel, setEmbedModel] = useState("bge-m3");
  const [provider, setProvider] = useState<"local" | "api">("local");
  const [llmError, setLlmError] = useState("");
  const [applyingLlm, setApplyingLlm] = useState(false);
  // Настройки внешнего API (вводятся здесь; ключ обратно с сервера не приходит)
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [apiModels, setApiModels] = useState<string[]>([]);
  const [modelsInfo, setModelsInfo] = useState<LlmModelInfo[]>([]);
  const [modelsRejected, setModelsRejected] = useState(0);
  const [summaryModel, setSummaryModel] = useState("");
  const [hintsModel, setHintsModel] = useState("");
  const [probing, setProbing] = useState(false);
  const [probeError, setProbeError] = useState("");
  // Настройки локальной модели — те же, что у API. Раньше адрес Ollama и её
  // модели задавались только переменными окружения сервера.
  const [ollamaUrl, setOllamaUrl] = useState("");
  const [localModels, setLocalModels] = useState<string[]>([]);
  const [localSummaryModel, setLocalSummaryModel] = useState("");
  const [localHintsModel, setLocalHintsModel] = useState("");
  // Роль модели для ответов по прошлым встречам — общая для обоих провайдеров

  async function test() {
    setTesting(true);
    setTestError("");
    setHealth(null);
    try {
      setHealth(await api.health());
      await refreshAsr(true);
      await refreshLlm(true);
    } catch (exc) {
      setTestError((exc as Error).message);
    } finally {
      setTesting(false);
    }
  }

  async function refreshAsr(resetSelects: boolean) {
    try {
      const state = await api.asr();
      setAsr(state);
      if (resetSelects) {
        setEngine(state.engine);
        setModel(state.model);
      }
      return state;
    } catch {
      setAsr(null); // старый сервер без /api/asr — карточку просто не показываем
      return null;
    }
  }

  useEffect(() => {
    unmounted.current = false; // StrictMode в dev монтирует дважды — флаг надо вернуть
    void test();
    return () => {
      unmounted.current = true;
    };
  }, []);

  function save() {
    setSetting("serverUrl", url.trim().replace(/\/+$/, "") || DEFAULT_SERVER_URL);
    setSetting("serverToken", token.trim());
    onServerChange();
    void test();
  }

  async function applyAsr() {
    setApplying(true);
    setAsrError("");
    try {
      let state = await api.setAsr(engine, model);
      setAsr(state);
      // модель грузится в фоне — опрашиваем сервер, пока не поднимется
      for (let i = 0; i < 200 && !state.loaded && !state.error; i++) {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        if (unmounted.current) return;
        const fresh = await refreshAsr(false);
        if (fresh === null) break;
        state = fresh;
      }
      if (state.error) setAsrError(state.error);
    } catch (exc) {
      setAsrError((exc as Error).message);
    } finally {
      setApplying(false);
    }
  }


  /** «llama-3.3-70b — 131k контекст · 8k токенов/мин».
   *
   *  Размер контекста показывает, потянет ли модель наши промпты, а лимит
   *  токенов в минуту — сколько разговора влезет в одну подсказку. Упираются
   *  на практике во второе, поэтому без него первая цифра вводит в заблуждение.
   *  Лимит известен только для сохранённых моделей: он измеряется при
   *  сохранении настроек. */
  function modelLabel(id: string): string {
    const context = modelsInfo.find((m) => m.id === id)?.context_window;
    const tpm = tpmLimits[id];
    const parts = [
      context ? `${Math.round(context / 1024)}k контекст` : "",
      tpm ? `${Math.round(tpm / 1000)}k токенов/мин` : "",
    ].filter(Boolean);
    return parts.length ? `${id} — ${parts.join(" · ")}` : id;
  }

  /**
   * Настройки моделей читаются из приложения, а не с сервера: он про модели
   * больше ничего не знает — ни адреса, ни ключа, ни выбора.
   */
  function refreshLlm(resetSelect: boolean) {
    const s = loadLlmSettings();
    if (resetSelect) {
      setProvider(s.provider);
      setBaseUrl(s.apiBaseUrl || API_BASE_URL_DEFAULT);
      setSummaryModel(s.apiSummaryModel);
      setHintsModel(s.apiHintsModel);
      setApiKey(s.apiKey);
      setOllamaUrl(s.ollamaUrl);
      setLocalSummaryModel(s.localSummaryModel);
      setLocalHintsModel(s.localHintsModel);
      setEmbedModel(s.embedModel);
    }
    setTpmLimits(s.tpmLimits ?? {});
    return s;
  }

  async function probeLocal() {
    setProbing(true);
    setProbeError("");
    try {
      // Спрашиваем саму Ollama: сервер посредником больше не работает
      // Пустой список бывает и у живой Ollama (модели не скачаны), поэтому
      // «отвечает» и «есть модели» — разные вопросы, и спрашиваем их отдельно.
      const client = new OllamaClient({ url: ollamaUrl.trim() });
      const models = await client.models();
      const res = { models, reachable: models.length > 0 || (await client.reachable()) };
      setLocalModels(res.models);
      if (!res.reachable) {
        setProbeError("Ollama по этому адресу не отвечает — проверьте, что она запущена.");
        return;
      }
      // Пустой список — не поломка, а не скачанные модели. Без подсказки это
      // выглядит одинаково, и человек идёт искать несуществующую ошибку.
      if (!res.models.length) {
        setProbeError("Ollama отвечает, но моделей нет. Скачайте: ollama pull qwen3:4b");
        return;
      }
      if (!localSummaryModel || !res.models.includes(localSummaryModel))
        setLocalSummaryModel(res.models[0]);
      if (!localHintsModel || !res.models.includes(localHintsModel))
        setLocalHintsModel(res.models[0]);
    } catch (exc) {
      setProbeError((exc as Error).message);
    } finally {
      setProbing(false);
    }
  }

  async function probe() {
    setProbing(true);
    setProbeError("");
    try {
      // Спрашиваем провайдера сами: ключ на сервер не уходит вовсе
      const пригодные = await new OpenAiClient({
        baseUrl: baseUrl.trim(),
        apiKey: apiKey || loadLlmSettings().apiKey,
      }).models();
      const res = { models: пригодные.map((m) => m.id) };
      setApiModels(res.models);
      setModelsInfo(пригодные.map((m) => ({ id: m.id, context_window: m.context ?? 0 })));
      setModelsRejected(0);
      if (!res.models.length) {
        setProbeError(
          "Подходящих моделей не нашлось: у всех либо слишком маленькое контекстное " +
            "окно, либо они работают не с текстом.",
        );
        return;
      }
      if (res.models.length) {
        if (!summaryModel || !res.models.includes(summaryModel)) setSummaryModel(res.models[0]);
        if (!hintsModel || !res.models.includes(hintsModel)) setHintsModel(res.models[0]);
      }
    } catch (exc) {
      setProbeError((exc as Error).message);
    } finally {
      setProbing(false);
    }
  }

  async function applyLlm() {
    setApplyingLlm(true);
    setLlmError("");
    try {
      saveLlmSettings({
        provider,
        ollamaUrl: ollamaUrl.trim(),
        localSummaryModel,
        localHintsModel,
        apiBaseUrl: baseUrl.trim(),
        ...(apiKey ? { apiKey } : {}), // пусто — не затираем уже сохранённый
        apiSummaryModel: summaryModel,
        apiHintsModel: hintsModel,
        embedModel: embedModel.trim() || "bge-m3",
      });
      // Лимит токенов в минуту меряем сразу: иначе первая же встреча пойдёт с
      // бюджетом наугад и выяснит его, упершись в отказ провайдера.
      if (provider === "api" && summaryModel) {
        const client = new OpenAiClient({ baseUrl: baseUrl.trim(), apiKey: loadLlmSettings().apiKey });
        const лимит = await client.tokenLimit(summaryModel);
        if (лимит) {
          const обновлённые = { ...loadLlmSettings().tpmLimits, [summaryModel]: лимит };
          saveLlmSettings({ tpmLimits: обновлённые });
          setTpmLimits(обновлённые);
        }
      }
      refreshLlm(false);
    } catch (exc) {
      setLlmError((exc as Error).message);
    } finally {
      setApplyingLlm(false);
    }
  }

  function toggleDebug(checked: boolean) {
    setDebug(checked);
    setSetting("debug", checked ? "1" : "0");
  }

  return (
    <div className="content">
      <h1>Настройки</h1>
      <p className="page-sub">
        Сервер распознаёт речь и хранит встречи. Модель языка — своя у каждого:
        её адрес и ключ живут здесь, в приложении.
      </p>

      <div className="card settings-block">
        <h2 style={{ marginBottom: 12 }}>Сервер распознавания</h2>
        <label className="field">
          <span>
            Адрес сервера (локальный или сервер организации — например{" "}
            <code>http://ai.corp.local:8765</code>)
          </span>
          <input
            className="input"
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            placeholder={DEFAULT_SERVER_URL}
          />
        </label>
        <label className="field">
          <span>
            Токен доступа — если сервер его требует. Личный сервер обходится без
            токена, общий выдаёт его командой <code>python -m app.users add</code>
          </span>
          <input
            className="input"
            type="password"
            value={token}
            onChange={(event) => setToken(event.target.value)}
            placeholder="не требуется"
            autoComplete="off"
          />
        </label>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn primary" onClick={save}>
            Сохранить и проверить
          </button>
          <button className="btn" onClick={test} disabled={testing}>
            {testing ? <span className="spinner" /> : "Проверить"}
          </button>
        </div>
        {testError && (
          <div className="banner error" style={{ marginTop: 12 }}>
            {testError}. Сервер запускается командой <code>uvicorn app.main:app</code> в папке{" "}
            <code>server</code>.
          </div>
        )}
        {health && health.authorized === false && (
          <div className="hint" style={{ marginTop: 14 }}>
            Сервер отвечает, но не признаёт нас: он закрыт токеном. Версия{" "}
            {health.version}. Введите токен доступа, который выдал администратор
            сервера.
          </div>
        )}
        {health && health.asr && (
          <div style={{ marginTop: 14 }}>
            <div className="kv">
              <span className="k">Версия сервера</span>
              <span>{health.version}</span>
            </div>
            <div className="kv">
              <span className="k">Распознавание речи</span>
              <span>
                {health.asr.model} · {ENGINE_LABELS[health.asr.engine] ?? health.asr.engine}{" "}
                {health.asr.device ? `· ${DEVICE_LABELS[health.asr.device] ?? health.asr.device} ` : ""}
                {health.asr.loaded ? "· загружена" : "· грузится…"}
              </span>
            </div>
            <div className="kv">
              <span className="k">Идентификация голосов</span>
              <span>
                {health.diarization?.loaded
                  ? `ECAPA · ${DEVICE_LABELS[health.diarization?.device ?? ""] ?? health.diarization?.device ?? ""} · загружена`
                  : "грузится…"}
              </span>
            </div>
          </div>
        )}
      </div>

      {asr && (
        <div className="card settings-block">
          <h2 style={{ marginBottom: 12 }}>Распознавание речи</h2>
          <label className="field">
            <span>Движок</span>
            <select
              className="input"
              value={engine}
              onChange={(event) => {
                const next = event.target.value;
                setEngine(next);
                const models = asr.models_by_engine[next] ?? [];
                if (!models.includes(model)) setModel(models[0] ?? "");
              }}
            >
              <option value="gigaam" disabled={!asr.engines.gigaam}>
                GigaAM (Сбер) — лучшее качество для русского
                {asr.engines.gigaam ? "" : " (не установлен на сервере)"}
              </option>
              <option value="faster_whisper">whisper CPU — работает везде</option>
              <option value="mlx" disabled={!asr.engines.mlx}>
                whisper GPU Metal — разгружает процессор
                {asr.engines.mlx ? "" : " (недоступен на этом сервере)"}
              </option>
            </select>
          </label>
          <label className="field">
            <span>Модель</span>
            <select
              className="input"
              value={model}
              onChange={(event) => setModel(event.target.value)}
            >
              {(asr.models_by_engine[engine] ?? []).map((m) => (
                <option key={m} value={m}>
                  {m} — {MODEL_HINTS[m] ?? ""}
                </option>
              ))}
            </select>
          </label>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <button
              className="btn primary"
              onClick={applyAsr}
              disabled={applying || (engine === asr.engine && model === asr.model)}
            >
              {applying ? <span className="spinner" /> : "Применить"}
            </button>
            <span style={{ color: "var(--muted)", fontSize: 12.5 }}>
              {applying || asr.loading
                ? "модель загружается…"
                : `сейчас: ${asr.model} · ${ENGINE_LABELS[asr.engine] ?? asr.engine}${asr.loaded ? " · загружена" : ""}`}
            </span>
          </div>
          {asrError && (
            <div className="banner error" style={{ marginTop: 12 }}>
              {asrError}
            </div>
          )}
        </div>
      )}

      {(
        <div className="card settings-block">
          <h2 style={{ marginBottom: 12 }}>Модель для подсказок и резюме</h2>
          <label className="field">
            <span>Провайдер LLM</span>
            <select
              className="input"
              value={provider}
              onChange={(event) => setProvider(event.target.value as "local" | "api")}
            >
              <option value="local">Локальная (Ollama) — данные не покидают контур</option>
              <option value="api">Внешний API (OpenAI-совместимый)</option>
            </select>
          </label>

          {provider === "local" && (
            <>
              <label className="field">
                <span>Адрес Ollama</span>
                <input
                  className="input"
                  value={ollamaUrl}
                  onChange={(event) => setOllamaUrl(event.target.value)}
                  placeholder="http://127.0.0.1:11434"
                />
                <span className="hint">
                  Ollama может работать не на этой машине: в контуре с
                  докером — соседним контейнером (<code>http://ollama:11434</code>),
                  в организации — на отдельном сервере
                </span>
              </label>
              <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 6 }}>
                <button className="btn" onClick={probeLocal} disabled={probing || !ollamaUrl.trim()}>
                  {probing ? <span className="spinner" /> : "Запросить модели"}
                </button>
                <span style={{ color: "var(--muted)", fontSize: 12.5 }}>
                  {localModels.length
                    ? `скачанных моделей: ${localModels.length}`
                    : "введите адрес и запросите список"}
                </span>
              </div>
              {probeError && (
                <div className="banner error" style={{ marginBottom: 10 }}>
                  {probeError}
                </div>
              )}
              <label className="field">
                <span>Модель для протокола (резюме)</span>
                <select
                  className="input"
                  value={localSummaryModel}
                  onChange={(event) => setLocalSummaryModel(event.target.value)}
                >
                  {localSummaryModel && !localModels.includes(localSummaryModel) && (
                    <option value={localSummaryModel}>{localSummaryModel}</option>
                  )}
                  {!localSummaryModel && <option value="">— выберите модель —</option>}
                  {localModels.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
                <span className="hint">
                  Протокол собирается из всей встречи, поэтому модель нужна
                  покрупнее: qwen3:4b и выше
                </span>
              </label>
              <label className="field">
                <span>Модель для подсказок (можно ту же; полегче — быстрее)</span>
                <select
                  className="input"
                  value={localHintsModel}
                  onChange={(event) => setLocalHintsModel(event.target.value)}
                >
                  {localHintsModel && !localModels.includes(localHintsModel) && (
                    <option value={localHintsModel}>{localHintsModel}</option>
                  )}
                  {!localHintsModel && <option value="">— выберите модель —</option>}
                  {localModels.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
                <span className="hint">
                  Подсказка должна успеть за разговором: на 8 ГБ памяти это
                  qwen3:1.7b, модель покрупнее не успевает
                </span>
              </label>
            </>
          )}

          {provider === "api" && (
            <>
              <label className="field">
                <span>Адрес API (пока поддерживается только Groq)</span>
                <input
                  className="input"
                  value={baseUrl}
                  onChange={(event) => setBaseUrl(event.target.value)}
                  placeholder="https://api.groq.com/openai/v1"
                />
                <span className="hint">
                  Groq сообщает размер контекста каждой модели — без этого нельзя
                  проверить, потянет ли модель наши промпты
                </span>
              </label>
              <label className="field">
                <span>API-ключ</span>
                <input
                  className="input"
                  type="password"
                  value={apiKey}
                  onChange={(event) => setApiKey(event.target.value)}
                  placeholder={apiKey ? "•••• сохранён (оставьте пустым)" : "gsk_…"}
                />
              </label>
              <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 6 }}>
                <button className="btn" onClick={probe} disabled={probing || !baseUrl.trim()}>
                  {probing ? <span className="spinner" /> : "Запросить модели"}
                </button>
                <span style={{ color: "var(--muted)", fontSize: 12.5 }}>
                  {apiModels.length
                    ? `подходящих моделей: ${apiModels.length}` +
                      (modelsRejected
                        ? ` · ${modelsRejected} скрыто (мало контекста или не текст)`
                        : "")
                    : "введите адрес и ключ, затем запросите список"}
                </span>
              </div>
              {probeError && (
                <div className="banner error" style={{ marginBottom: 10 }}>
                  {probeError}
                </div>
              )}
              <label className="field">
                <span>Модель для протокола (резюме)</span>
                <select
                  className="input"
                  value={summaryModel}
                  onChange={(event) => setSummaryModel(event.target.value)}
                >
                  {summaryModel && !apiModels.includes(summaryModel) && (
                    <option value={summaryModel}>{summaryModel}</option>
                  )}
                  {!summaryModel && <option value="">— выберите модель —</option>}
                  {apiModels.map((m) => (
                    <option key={m} value={m}>
                      {modelLabel(m)}
                    </option>
                  ))}
                </select>
                {/* Предупреждение, а не запрет: список моделей приходит от
                    провайдера, у следующего API он будет другим, и прятать по
                    именам мы бы всё равно ничего не смогли. */}
                <span className="hint">
                  Берите модель, предназначенную именно для текста → текста.
                  Агентные связки (в списке Groq это groq/compound) добавляют к
                  запросу своё и обращаются к модели по нескольку раз: тот же
                  протокол обходится им вдвое дороже, чем показывают счётчики, и
                  встреча не собирается даже короткая.
                </span>
              </label>
              <label className="field">
                <span>Модель для подсказок (можно ту же; полегче — быстрее)</span>
                <select
                  className="input"
                  value={hintsModel}
                  onChange={(event) => setHintsModel(event.target.value)}
                >
                  {hintsModel && !apiModels.includes(hintsModel) && (
                    <option value={hintsModel}>{hintsModel}</option>
                  )}
                  {!hintsModel && <option value="">— выберите модель —</option>}
                  {apiModels.map((m) => (
                    <option key={m} value={m}>
                      {modelLabel(m)}
                    </option>
                  ))}
                </select>
              </label>
            </>
          )}

          <label className="field">
            <span>Модель эмбеддингов для поиска по встречам</span>
            <input
              className="input"
              value={embedModel}
              onChange={(event) => setEmbedModel(event.target.value)}
              placeholder="bge-m3"
            />
            <span className="hint">
              Считается локальной Ollama даже при выбранном внешнем API: это не
              разговорная модель, у провайдеров она тарифицируется отдельно.
              Скачать: <code>ollama pull bge-m3</code>
            </span>
          </label>

          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <button
              className="btn primary"
              onClick={applyLlm}
              disabled={
                applyingLlm ||
                // Пустой адрес сервер игнорирует (пустое = «не меняли»), и
                // «Применить» выглядело бы как молчаливый отказ.
                (provider === "local" && !ollamaUrl.trim()) ||
                (provider === "api" &&
                  (!baseUrl.trim() ||
                    !summaryModel ||
                    !hintsModel ||
                    !apiKey))
              }
            >
              {applyingLlm ? <span className="spinner" /> : "Применить"}
            </button>
            <span style={{ color: "var(--muted)", fontSize: 12.5 }}>
              {provider === "api"
                ? `сейчас: внешний API · ${summaryModel || "модель не задана"}`
                : `сейчас: локальная модель · ${localSummaryModel || "модель не задана"}`}
            </span>
          </div>
          <p style={{ color: "var(--muted)", fontSize: 12.5, marginTop: 10 }}>
            Ключ хранится на сервере (<code>server/data/llm.json</code>) и обратно на клиент не
            отдаётся. При включённом API на указанный endpoint отправляется текст транскрипта
            (аудио — никогда).
          </p>
          {llmError && (
            <div className="banner error" style={{ marginTop: 12 }}>
              {llmError}
            </div>
          )}
        </div>
      )}

      <div className="card settings-block">
        <h2 style={{ marginBottom: 12 }}>Приложение</h2>
        <label className="check">
          <input
            type="checkbox"
            checked={debug}
            onChange={(event) => toggleDebug(event.target.checked)}
          />
          <span className="box">✓</span>
          Режим отладки
          <span className="hint">показывать канал и близость голоса у каждой реплики</span>
        </label>
        <div className="kv" style={{ marginTop: 10 }}>
          <span className="k">Платформа</span>
          <span>{platform()}</span>
        </div>
      </div>

      <div className="card settings-block">
        <h2 style={{ marginBottom: 12 }}>Приватность</h2>
        <p style={{ color: "var(--muted)", fontSize: 13.5 }}>
          Распознавание речи и определение голосов всегда выполняются локальными моделями
          (whisper/GigaAM, ECAPA-TDNN) — аудио не покидает вашу машину или сервер организации.
          Составление протоколов и подсказки по умолчанию тоже локальны (Ollama). Внешний
          LLM-API — отдельная опция, выключенная по умолчанию; при её включении на указанный в
          настройках сервера endpoint отправляется текст транскрипта (аудио — никогда).
        </p>
      </div>
    </div>
  );
}
