import { useEffect, useRef, useState } from "react";
import { api } from "../api/rest";
import Avatar from "../components/Avatar";
import type { SpeakerDto } from "../types";

/** «Звучания» голоса: каждый отпечаток можно прослушать (аудио реплики,
 *  из которой он родился) и удалить, если он путает голоса. */
function VoiceprintButtons({
  speaker,
  playing,
  onPlay,
  onDelete,
}: {
  speaker: SpeakerDto;
  playing: number | null;
  onPlay: (speakerId: number, printId: number) => void;
  onDelete: (speaker: SpeakerDto, printId: number, index: number) => void;
}) {
  if (speaker.voiceprints.length === 0) {
    return <span className="hint" style={{ fontSize: 12 }}>звучаний голоса пока нет</span>;
  }
  return (
    <div className="sample-row">
      {speaker.voiceprints.map((print, index) => (
        <span key={print.id} style={{ display: "inline-flex", gap: 2 }}>
          {print.audio_duration_s != null ? (
            <button
              className={`icon-btn ${playing === print.id ? "playing" : ""}`}
              title={`Звучание №${index + 1}, усреднено из ${print.count} реплик — нажмите, чтобы прослушать`}
              onClick={() => onPlay(speaker.id, print.id)}
            >
              {playing === print.id ? "◼" : "▶"} {print.audio_duration_s}с · {print.count}
            </button>
          ) : (
            <span
              className="icon-btn"
              title={`Звучание №${index + 1} без аудио (создано до обновления), усреднено из ${print.count} реплик`}
            >
              №{index + 1} · {print.count}
            </span>
          )}
          <button
            className="icon-btn"
            title="Удалить звучание (отпечаток и его аудио)"
            onClick={() => onDelete(speaker, print.id, index + 1)}
          >
            ✕
          </button>
        </span>
      ))}
    </div>
  );
}

export default function SpeakersPage() {
  const [speakers, setSpeakers] = useState<SpeakerDto[] | null>(null);
  const [selected, setSelected] = useState<number[]>([]);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState("");
  const [playingPrint, setPlayingPrint] = useState<number | null>(null);
  const [mergeOpen, setMergeOpen] = useState(false);
  const [merging, setMerging] = useState(false);
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

  function playPrint(speakerId: number, printId: number) {
    audioRef.current?.pause();
    if (playingPrint === printId) {
      setPlayingPrint(null);
      return;
    }
    const audio = new Audio(api.voiceprintAudioUrl(speakerId, printId));
    audioRef.current = audio;
    audio.onended = () => setPlayingPrint(null);
    audio.onerror = () => {
      setPlayingPrint(null);
      setError("Не удалось воспроизвести звучание");
    };
    void audio.play();
    setPlayingPrint(printId);
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

  async function removeVoiceprint(speaker: SpeakerDto, printId: number, index: number) {
    const last = speaker.voiceprints.length === 1;
    const ok = confirm(
      `Удалить звучание №${index} у «${speaker.name}»?\n` +
        (last
          ? speaker.is_self
            ? "Это последнее звучание — профиль заново обучится по вашему голосу из микрофона."
            : "Это последнее звучание — голос перестанет узнаваться автоматически."
          : "Профиль и реплики останутся, удалится только это «звучание» голоса."),
    );
    if (!ok) return;
    try {
      await api.deleteVoiceprint(speaker.id, printId);
      await load();
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  async function removeSpeaker(speaker: SpeakerDto) {
    const ok = confirm(
      `Удалить профиль «${speaker.name}»?\n` +
        "Реплики в транскриптах останутся (как «Неизвестный»), " +
        "а звучания голоса будут удалены — при следующей встрече " +
        "этот голос распознается как новый спикер.",
    );
    if (!ok) return;
    try {
      await api.deleteSpeaker(speaker.id);
      setSelected((previous) => previous.filter((x) => x !== speaker.id));
      await load();
    } catch (exc) {
      setError((exc as Error).message);
    }
  }

  const [a, b] = selected
    .map((id) => speakers?.find((s) => s.id === id))
    .filter(Boolean) as SpeakerDto[];

  async function merge() {
    if (!a || !b) return;
    setMerging(true);
    try {
      const result = await api.mergeSpeakers([a.id, b.id]);
      setNotice(
        `Объединено в «${result.name}»: перенесено реплик — ${result.moved_segments}. ` +
          "Оба отпечатка голоса сохранены.",
      );
      setSelected([]);
      setMergeOpen(false);
      await load();
    } catch (exc) {
      setError((exc as Error).message);
      setMergeOpen(false);
    } finally {
      setMerging(false);
    }
  }

  return (
    <div className="content">
      <h1>Спикеры</h1>
      <p className="page-sub">
        Все голоса, которые Стенограф встречал. У каждого — «звучания»: отпечатки голоса,
        которые можно прослушать. Если один человек распознался как два профиля
        (например, сменил микрофон) — выберите оба, прослушайте и объедините.
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
                    {!speaker.is_self && (
                      <button
                        className="btn small danger"
                        title="Удалить профиль"
                        onClick={() => removeSpeaker(speaker)}
                      >
                        Удалить
                      </button>
                    )}
                  </div>
                )}
                <div className="speaker-stats">
                  встреч: {speaker.meetings_count} · реплик: {speaker.segments_count}
                  {speaker.voiceprints_count > 1 &&
                    ` · звучаний голоса: ${speaker.voiceprints_count}`}
                </div>
              </div>
            </div>
            <VoiceprintButtons
              speaker={speaker}
              playing={playingPrint}
              onPlay={playPrint}
              onDelete={removeVoiceprint}
            />
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

      {a && b && !mergeOpen && (
        <div className="merge-bar">
          <span>
            Выбраны <b>{a.name}</b> и <b>{b.name}</b>
          </span>
          <button className="btn primary small" onClick={() => setMergeOpen(true)}>
            Объединить…
          </button>
          <button className="btn small" onClick={() => setSelected([])}>
            Отмена
          </button>
        </div>
      )}

      {mergeOpen && a && b && (
        <div className="modal-overlay" onClick={() => !merging && setMergeOpen(false)}>
          <div className="modal" onClick={(event) => event.stopPropagation()}>
            <h2>Объединение спикеров</h2>
            <p className="modal-sub">
              Прослушайте звучания и убедитесь, что это один человек. Профили сольются в
              один, реплики во всех встречах будут переписаны. Все звучания голоса
              сохранятся — человек будет узнаваться в каждом из них.
            </p>
            <div className="merge-pair">
              {[a, b].map((speaker) => (
                <div className="merge-side" key={speaker.id}>
                  <div className="speaker-head">
                    <Avatar
                      id={speaker.id}
                      name={speaker.name}
                      isSelf={speaker.is_self}
                      size={38}
                    />
                    <div>
                      <div className="speaker-name">{speaker.name}</div>
                      <div className="speaker-stats">
                        встреч: {speaker.meetings_count} · реплик: {speaker.segments_count}
                      </div>
                    </div>
                  </div>
                  <VoiceprintButtons
                    speaker={speaker}
                    playing={playingPrint}
                    onPlay={playPrint}
                    onDelete={removeVoiceprint}
                  />
                </div>
              ))}
            </div>
            <div className="modal-actions">
              <button className="btn" onClick={() => setMergeOpen(false)} disabled={merging}>
                Отмена
              </button>
              <button className="btn primary" onClick={merge} disabled={merging}>
                {merging ? <span className="spinner" /> : "Объединить"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
