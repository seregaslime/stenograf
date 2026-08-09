/** Переименование участника прямо в ленте.
 *
 *  Отдельным файлом от Transcript.test.ts: там чистые функции, здесь нужен DOM
 *  и разметка. Рендерим настоящим react-dom (он и так в зависимостях), без
 *  библиотек для тестирования — новая зависимость ради трёх проверок не окупается.
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Transcript from "./Transcript";
import type { SegmentDto } from "../types";

const segments: SegmentDto[] = [
  {
    id: 1, meeting_id: 1, channel: "mic", start_s: 0, end_s: 1,
    text: "первая", similarity: null,
    speaker: { id: 7, name: "Спикер 7", is_self: false },
  },
  {
    id: 2, meeting_id: 1, channel: "mic", start_s: 2, end_s: 3,
    text: "вторая", similarity: null,
    speaker: { id: 7, name: "Спикер 7", is_self: false },
  },
];

// Без этого React пишет в stderr, что окружение не настроено под act(...):
// он не умеет сам определить, что мы в тестах, и предупреждает на каждый рендер.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

function render(onRename?: (id: number, name: string) => void) {
  act(() => {
    root.render(<Transcript segments={segments} onRename={onRename} />);
  });
}

function nameButton() {
  return container.querySelector<HTMLButtonElement>(".msg-name.as-button");
}

function nameInput() {
  return container.querySelector<HTMLInputElement>(".msg-name-input");
}

function press(input: HTMLInputElement, key: string) {
  act(() => {
    input.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true }));
  });
}

describe("переименование участника в ленте", () => {
  it("без onRename имя остаётся подписью — в истории встречи править нечего", () => {
    render();
    expect(nameButton()).toBeNull();
    expect(container.querySelector(".msg-name")?.textContent).toBe("Спикер 7");
  });

  it("клик по имени открывает поле с текущим именем", () => {
    render(vi.fn());
    act(() => nameButton()!.click());
    expect(nameInput()?.value).toBe("Спикер 7");
  });

  it("Enter сохраняет новое имя", () => {
    const onRename = vi.fn();
    render(onRename);
    act(() => nameButton()!.click());
    const input = nameInput()!;
    input.value = "Иван";
    press(input, "Enter");
    expect(onRename).toHaveBeenCalledWith(7, "Иван");
    expect(nameInput()).toBeNull();  // поле закрылось
  });

  it("Escape отменяет правку", () => {
    const onRename = vi.fn();
    render(onRename);
    act(() => nameButton()!.click());
    const input = nameInput()!;
    input.value = "Ошибся";
    press(input, "Escape");
    expect(onRename).not.toHaveBeenCalled();
  });

  it("пустое имя и имя без изменений не сохраняются", () => {
    const onRename = vi.fn();
    render(onRename);
    act(() => nameButton()!.click());
    const input = nameInput()!;
    input.value = "   ";
    press(input, "Enter");
    expect(onRename).not.toHaveBeenCalled();
  });
});
