import { useEffect, useState } from "react";
import { api } from "../api/rest";
import { DEFAULT_SERVER_URL, getSetting, isDebugMode, platform, setSetting } from "../store";
import type { HealthDto } from "../types";

export default function SettingsPage({ onServerChange }: { onServerChange: () => void }) {
  const [url, setUrl] = useState(getSetting("serverUrl", DEFAULT_SERVER_URL));
  const [debug, setDebug] = useState(isDebugMode());
  const [health, setHealth] = useState<HealthDto | null>(null);
  const [testError, setTestError] = useState("");
  const [testing, setTesting] = useState(false);

  async function test() {
    setTesting(true);
    setTestError("");
    setHealth(null);
    try {
      setHealth(await api.health());
    } catch (exc) {
      setTestError((exc as Error).message);
    } finally {
      setTesting(false);
    }
  }

  useEffect(() => {
    void test();
  }, []);

  function save() {
    setSetting("serverUrl", url.trim().replace(/\/+$/, "") || DEFAULT_SERVER_URL);
    onServerChange();
    void test();
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
                whisper {health.asr.model} {health.asr.loaded ? "· загружен" : "· грузится…"}
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
          Стенограф не использует внешние API: распознавание речи, определение голосов и
          составление протоколов выполняются локальными моделями (whisper, ECAPA-TDNN, Ollama)
          на вашей машине или на сервере организации. Записи и транскрипты не покидают контур.
        </p>
      </div>
    </div>
  );
}
