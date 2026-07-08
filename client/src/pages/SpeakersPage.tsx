import { useEffect, useRef, useState } from "react";
import { api } from "../api/rest";
import Avatar from "../components/Avatar";
import type { SpeakerDto } from "../types";

export default function SpeakersPage() {
  const [speakers, setSpeakers] = useState<SpeakerDto[] | null>(null);
  const [selected, setSelected] = useState<number[]>([]);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState("");
  const [playingSample, setPlayingSample] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const load = () =>
    api
      .speakers()
      .then((list) => {
        setSpeakers(list);
        setError("");
      })
      .catch((exc: Error) => setError(exc.message));

  useEffect(() => {
    void load();
    return () => audioRef.current?.pause();
  }, []);

  function toggleSelect(id: number) {
    setSelected((previous) =>
      previous.includes(id)
        ? previous.filter((x) => x !== id)
        : previous.length < 2
          ? [...previous, id]
          : previous,
    );
  }

  function playSample(sampleId: number) {
    audioRef.current?.pause();
    if (playingSample === sampleId) {
      setPlayingSample(null);
      return;
    }
    const audio = new Audio(api.sampleUrl(sampleId));
    audioRef.current = audio;
    audio.onended = () => setPlayingSample(null);
    audio.onerror = () => {
      setPlayingSample(null);
      setError("Не удалось воспроизвести образец");
    };
    void audio.play();
    setPlayingSample(sampleId);
  }

  async function saveName(id: number) {
    setEditingId(null);
    const name = editName.trim();
    if (!name) return;
    try {
      await api.renameSpeaker(id, name);
      await load();
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  async function merge(sourceId: number, targetId: number) {
    const source = speakers?.find((s) => s.id === sourceId);
    const target = speakers?.find((s) => s.id === targetId);
    if (!source || !target) return;
    if (
      !confirm(
        `Объединить «${source.name}» с «${target.name}»?\n` +
          `Все реплики «${source.name}» во всех встречах перейдут к «${target.name}», ` +
          `профиль «${source.name}» будет удалён.`,
      )
    )
      return;
    try {
      const result = await api.mergeSpeakers(sourceId, targetId);
      setNotice(`Готово: перенесено реплик — ${result.moved_segments}`);
      setSelected([]);
      await load();
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  const [a, b] = selected.map((id) => speakers?.find((s) => s.id === id)).filter(Boolean) as SpeakerDto[];

  return (
    <div className="content">
      <h1>Спикеры</h1>
      <p className="page-sub">
        Все голоса, которые Стенограф встречал. Если один человек распознался как два профиля
        (например, сменил микрофон) — прослушайте образцы, выберите оба и объедините.
      </p>
      {error && <div className="banner error">{error}</div>}
      {notice && <div className="banner info">{notice}</div>}
      {speakers && speakers.length === 0 && (
        <div className="empty">
          <div className="big-icon">👥</div>
          Спикеры появятся после первой встречи
        </div>
      )}
      <div className="speakers-grid">
        {(speakers ?? []).map((speaker) => (
          <div
            key={speaker.id}
            className={`card speaker-card ${selected.includes(speaker.id) ? "selected" : ""}`}
          >
            <div className="speaker-head">
              <Avatar id={speaker.id} name={speaker.name} isSelf={speaker.is_self} size={42} />
              <div style={{ minWidth: 0, flex: 1 }}>
                {editingId === speaker.id ? (
                  <input
                    className="input"
                    autoFocus
                    value={editName}
                    onChange={(event) => setEditName(event.target.value)}
                    onBlur={() => saveName(speaker.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") void saveName(speaker.id);
                      if (event.key === "Escape") setEditingId(null);
                    }}
                  />
                ) : (
                  <div className="speaker-name">
                    {speaker.name}
                    {speaker.is_self && <span className="chip done">это вы</span>}
                    <button
                      className="icon-btn"
                      title="Переименовать"
                      onClick={() => {
                        setEditingId(speaker.id);
                        setEditName(speaker.name);
                      }}
                    >
                      ✎
                    </button>
                  </div>
                )}
                <div className="speaker-stats">
                  встреч: {speaker.meetings_count} · реплик: {speaker.segments_count}
                </div>
              </div>
            </div>
            {speaker.samples.length > 0 && (
              <div className="sample-row">
                {speaker.samples.map((sample) => (
                  <button
                    key={sample.id}
                    className={`icon-btn ${playingSample === sample.id ? "playing" : ""}`}
                    onClick={() => playSample(sample.id)}
                  >
                    {playingSample === sample.id ? "◼" : "▶"} {sample.duration_s}с
                  </button>
                ))}
              </div>
            )}
            <label className="check">
              <input
                type="checkbox"
                checked={selected.includes(speaker.id)}
                onChange={() => toggleSelect(speaker.id)}
              />
              <span className="box">✓</span>
              <span className="hint">выбрать для объединения</span>
            </label>
          </div>
        ))}
      </div>
      {a && b && (
        <div className="merge-bar">
          <span>
            Объединить <b>{a.name}</b> и <b>{b.name}</b> — кого оставить?
          </span>
          <button className="btn primary small" onClick={() => merge(b.id, a.id)}>
            Оставить «{a.name}»
          </button>
          <button className="btn primary small" onClick={() => merge(a.id, b.id)}>
            Оставить «{b.name}»
          </button>
          <button className="btn small" onClick={() => setSelected([])}>
            Отмена
          </button>
        </div>
      )}
    </div>
  );
}
