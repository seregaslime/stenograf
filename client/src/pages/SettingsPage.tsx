import { useEffect, useRef, useState } from "react";
import { api } from "../api/rest";
import { DEFAULT_SERVER_URL, getSetting, isDebugMode, platform, setSetting } from "../store";
import type { AsrStateDto, HealthDto, LlmStateDto } from "../types";

const ENGINE_LABELS: Record<string, string> = {
  faster_whisper: "CPU (faster-whisper)",
  mlx: "GPU Metal (mlx)",
  gigaam: "GigaAM (Сбер)",
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

  async function refreshLlm(resetSelect: boolean) {
    try {
      const state = await api.llm();
      setLlm(state);
      if (resetSelect) setProvider(state.provider);
      return state;
    } catch {
      setLlm(null); // старый сервер без /api/llm — карточку просто не показываем
      return null;
    }
  }

  async function applyLlm() {
    setApplyingLlm(true);
    setLlmError("");
    try {
      const state = await api.setLlm(provider);
      setLlm(state);
      setProvider(state.provider);
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
        {health && (
          <div style={{ marginTop: 14 }}>
            <div className="kv">
              <span className="k">Версия сервера</span>
              <span>{health.version}</span>
            </div>
            <div className="kv">
              <span className="k">Распознавание речи</span>
              <span>
                {health.asr.model} · {ENGINE_LABELS[health.asr.engine] ?? health.asr.engine}{" "}
                {health.asr.loaded ? "· загружена" : "· грузится…"}
              </span>
            </div>
            <div className="kv">
              <span className="k">Идентификация голосов</span>
              <span>{health.diarization.loaded ? "ECAPA · загружена" : "грузится…"}</span>
            </div>
            <div className="kv">
              <span className="k">Локальная LLM (Ollama)</span>
              <span>
                {health.ollama.reachable
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
              <option value="api" disabled={!llm.api_configured}>
                Внешний API (OpenAI-совместимый)
                {llm.api_configured ? "" : " — не настроен в server/.env"}
              </option>
            </select>
          </label>
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <button
              className="btn primary"
              onClick={applyLlm}
              disabled={applyingLlm || provider === llm.provider}
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
          {!llm.api_configured && (
            <p style={{ color: "var(--muted)", fontSize: 12.5, marginTop: 10 }}>
              Чтобы включить внешний API, задайте <code>STENOGRAF_LLM_API_BASE_URL</code>,{" "}
              <code>STENOGRAF_LLM_API_KEY</code> и модели{" "}
              <code>STENOGRAF_LLM_API_SUMMARY_MODEL</code> /{" "}
              <code>STENOGRAF_LLM_API_HINTS_MODEL</code> в <code>server/.env</code> и перезапустите
              сервер. Ключ остаётся на сервере и на клиент не передаётся.
            </p>
          )}
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
