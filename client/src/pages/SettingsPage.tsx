import { useEffect, useRef, useState } from "react";
import { api } from "../api/rest";
import { saveLlmSettings } from "../llm/settings";
import { DEFAULT_SERVER_URL, getSetting, isDebugMode, platform, setSetting } from "../store";
import type { AsrStateDto, HealthDto, LlmModelInfo, LlmStateDto } from "../types";

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

  const [llm, setLlm] = useState<LlmStateDto | null>(null);
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
  const [searchAnswerModel, setSearchAnswerModel] = useState("summary");

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

  function fillLlmForm(state: LlmStateDto) {
    setProvider(state.provider);
    // Ничего не сохранено — подставляем адрес поддерживаемого провайдера,
    // чтобы не заставлять человека печатать его руками
    setBaseUrl(state.api_base_url || state.api_base_url_default || "");
    // именно API-модели: при активном local здесь не должны оказаться имена Ollama
    setSummaryModel(state.api_summary_model ?? "");
    setHintsModel(state.api_hints_model ?? "");
    setApiKey(""); // ключ с сервера не приходит — пустое поле = «оставить прежний»
    setOllamaUrl(state.ollama_url ?? "");
    setLocalSummaryModel(state.local_summary_model ?? "");
    setLocalHintsModel(state.local_hints_model ?? "");
    setSearchAnswerModel(state.search_answer_model || "summary");
    // models из status() — это модели АКТИВНОГО провайдера, и класть их надо
    // в свой список: у api они уже отфильтрованы по пригодности, у local это
    // просто скачанные Ollama. Свалив их в одну переменную, мы бы показывали
    // имена qwen3 в списке моделей API сразу после переключения провайдера.
    const модели = state.models ?? [];
    setApiModels(state.provider === "api" ? модели : []);
    setLocalModels(state.provider === "local" ? модели : []);
    setModelsInfo(state.models_info ?? []);
    setModelsRejected(state.models_rejected ?? 0);
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
    const tpm = llm?.api_tpm_limits?.[id];
    const parts = [
      context ? `${Math.round(context / 1024)}k контекст` : "",
      tpm ? `${Math.round(tpm / 1000)}k токенов/мин` : "",
    ].filter(Boolean);
    return parts.length ? `${id} — ${parts.join(" · ")}` : id;
  }

  async function refreshLlm(resetSelect: boolean) {
    try {
      const state = await api.llm();
      setLlm(state);
      if (resetSelect) fillLlmForm(state);
      return state;
    } catch {
      setLlm(null); // старый сервер без /api/llm — карточку просто не показываем
      return null;
    }
  }

  async function probeLocal() {
    setProbing(true);
    setProbeError("");
    try {
      const res = await api.probeOllama(ollamaUrl.trim());
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
      const res = await api.probeModels(baseUrl.trim(), apiKey || undefined);
      if (!res.reachable) {
        setProbeError("API недоступен или отклонил ключ — проверьте адрес и ключ.");
        return;
      }
      setApiModels(res.models);
      setModelsInfo(res.models_info ?? []);
      setModelsRejected(res.models_rejected ?? 0);
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
      const state = await api.setLlm({
        provider,
        api_base_url: baseUrl.trim(),
        api_key: apiKey || undefined, // пусто — сервер оставит сохранённый ключ
        summary_model: summaryModel,
        hints_model: hintsModel,
        ollama_url: ollamaUrl.trim(),
        local_summary_model: localSummaryModel,
        local_hints_model: localHintsModel,
        search_answer_model: searchAnswerModel,
      });
      setLlm(state);
      fillLlmForm(state);
      // Те же настройки — в хранилище приложения: протокол теперь считает оно,
      // и адрес модели с ключом нужны ему самому. Сервер узнаёт о них
      // последний раз — на следующем шаге серии он про модели забудет совсем.
      saveLlmSettings({
        provider,
        ollamaUrl: ollamaUrl.trim(),
        localSummaryModel,
        localHintsModel,
        apiBaseUrl: baseUrl.trim(),
        ...(apiKey ? { apiKey } : {}), // пусто — не затираем уже сохранённый
        apiSummaryModel: summaryModel,
        apiHintsModel: hintsModel,
        tpmLimits: state.api_tpm_limits ?? {},
      });
    } catch (exc) {
      setLlmError((exc as Error).message);
      if (llm) setProvider(llm.provider); // сервер отклонил выбор — вернуть селект
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
      <p className="page-sub">Клиент лёгкий — все модели работают на сервере.</p>

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
            <div className="kv">
              <span className="k">Локальная LLM (Ollama)</span>
              <span>
                {health.ollama?.reachable
                  ? `доступна · ${health.ollama.models.length ? health.ollama.models.join(", ") : "нет моделей"}`
                  : "недоступна — резюме работать не будет"}
              </span>
            </div>
            <div className="kv">
              <span className="k">Модель резюме</span>
              <span>{health.summary_model}</span>
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

      {llm && (
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
                  placeholder={llm.api_configured ? "•••• сохранён (оставьте пустым)" : "gsk_…"}
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
            <span>Модель для ответов по прошлым встречам</span>
            <select
              className="input"
              value={searchAnswerModel}
              onChange={(event) => setSearchAnswerModel(event.target.value)}
            >
              <option value="summary">Как для протокола — точнее</option>
              <option value="hints">Как для подсказок — быстрее</option>
            </select>
            <span className="hint">
              Поиск по истории встреч отдаёт найденное модели, и она отвечает по
              нему. Модель протокола обычно крупнее: на локальной машине ответ
              занимает втрое больше времени (замер на M3: 36 секунд против 13)
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
                    !(apiKey || llm.api_configured)))
              }
            >
              {applyingLlm ? <span className="spinner" /> : "Применить"}
            </button>
            <span style={{ color: "var(--muted)", fontSize: 12.5 }}>
              {llm.provider === "api"
                ? `сейчас: внешний API${llm.reachable ? " · доступен" : " · недоступен"} · ${
                    llm.summary_model || "модель не задана"
                  }`
                : `сейчас: локальная Ollama${llm.reachable ? " · доступна" : " · недоступна"}`}
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
