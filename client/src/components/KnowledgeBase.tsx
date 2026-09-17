import { useEffect, useRef, useState } from "react";

import { api } from "../api/rest";
import type { DocumentDto } from "../types";

/** Байты файла в base64 — кусками: String.fromCharCode на мегабайт сразу
 *  упирается в предел числа аргументов функции и падает. */
export function toBase64(bytes: Uint8Array): string {
  let binary = "";
  const шаг = 0x8000;
  for (let i = 0; i < bytes.length; i += шаг) {
    binary += String.fromCharCode(...bytes.subarray(i, i + шаг));
  }
  return btoa(binary);
}

/**
 * Свои документы в базе знаний: загрузить txt или md, увидеть список, удалить.
 *
 * Здесь же, на странице истории, а не отдельным экраном: документ ищется тем
 * же поиском, что и встречи, и человек загружает его ради этого поиска.
 * Полноценный экран управления — пункт 5б. Векторы документа посчитаются при
 * следующем поиске, как и у новой встречи.
 */
export default function KnowledgeBase() {
  const [documents, setDocuments] = useState<DocumentDto[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const выбор = useRef<HTMLInputElement>(null);

  const load = () =>
    api.documents().then(setDocuments).catch((exc: Error) => setError(exc.message));

  useEffect(() => {
    void load();
  }, []);

  async function upload(file: File) {
    setBusy(true);
    setError("");
    try {
      const байты = new Uint8Array(await file.arrayBuffer());
      await api.uploadDocument(file.name, toBase64(байты));
      await load();
    } catch (exc) {
      setError((exc as Error).message);
    } finally {
      setBusy(false);
      if (выбор.current) выбор.current.value = "";  // тот же файл можно выбрать снова
    }
  }

  async function remove(document: DocumentDto) {
    if (!confirm(`Удалить документ «${document.title}» из базы знаний?`)) return;
    try {
      await api.deleteDocument(document.id);
      await load();
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  return (
    <div className="card settings-block">
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <h2 style={{ margin: 0, flex: 1 }}>База знаний</h2>
        <input
          ref={выбор}
          type="file"
          accept=".txt,.md"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void upload(file);
          }}
        />
        <button className="btn small" disabled={busy} onClick={() => выбор.current?.click()}>
          {busy ? <span className="spinner" /> : "Загрузить документ"}
        </button>
      </div>
      <span className="hint">
        Регламенты, ТЗ, заметки в .txt и .md до мегабайта — поиск найдёт в них ответ вместе со встречами
      </span>
      {error && <div className="banner error" style={{ marginTop: 10 }}>{error}</div>}
      {documents?.map((document) => (
        <div key={document.id} className="list-item" style={{ marginTop: 8 }}>
          <div className="grow">
            <div>{document.title}</div>
            <div className="meta">{document.chars.toLocaleString("ru-RU")} символов</div>
          </div>
          <button className="btn small danger" onClick={() => void remove(document)}>
            Удалить
          </button>
        </div>
      ))}
    </div>
  );
}
