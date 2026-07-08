import type { CSSProperties } from "react";

const COLORS = [
  "#ff8a5c", "#3ecf8e", "#f65ca8", "#f6c453",
  "#5cd6f6", "#b18cff", "#7fd18c", "#ff6b6b",
];

export function speakerColor(id: number, isSelf: boolean): string {
  if (isSelf) return "#4f8cff";
  return COLORS[id % COLORS.length];
}

export function initials(name: string): string {
  const words = name.trim().split(/\s+/);
  if (words.length >= 2) {
    // «Спикер 3» → «С3», «Анна Петрова» → «АП»
    return (words[0][0] + words[1][0]).toUpperCase();
  }
  return name.slice(0, 2);
}

export default function Avatar({
  id,
  name,
  isSelf,
  size,
}: {
  id: number;
  name: string;
  isSelf: boolean;
  size?: number;
}) {
  const style: CSSProperties = { background: speakerColor(id, isSelf) };
  if (size) {
    style.width = size;
    style.height = size;
    style.fontSize = size * 0.36;
  }
  return (
    <div className="avatar" style={style} title={name}>
      {initials(name)}
    </div>
  );
}
