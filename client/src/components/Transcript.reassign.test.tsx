/** Выделить слова в пузыре и отдать их другому спикеру.
 *
 *  Нужен настоящий DOM с выделением: какие слова выделены, решается по границам
 *  Range, и чистой функцией без разметки это не проверить. Как и в
 *  Transcript.rename.test.tsx — голый react-dom, без библиотек для тестирования.
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Transcript, { type Reassign } from "./Transcript";
import type { SegmentDto, SpeakerRef } from "../types";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const сатир: SpeakerRef = { id: 7, name: "Сатир", is_self: false };
const бабка: SpeakerRef = { id: 8, name: "Бабка", is_self: false };
const вы: SpeakerRef = { id: 3, name: "Вы", is_self: true };

const реплика: SegmentDto = {
  id: 1, meeting_id: 1, channel: "system", start_s: 5, end_s: 7.3,
  text: "Да, согласен. Нет, погоди.", similarity: 0.6, speaker: сатир,
  words: [[5.2, 5.4, "Да,"], [5.5, 6.0, "согласен."], [6.4, 6.6, "Нет,"], [6.7, 7.1, "погоди."]],
};

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
  window.getSelection()?.removeAllRanges();
});

function render(segment: SegmentDto, onReassign?: Reassign) {
  act(() => {
    root.render(
      <Transcript segments={[segment]} speakers={[сатир, бабка, вы]} onReassign={onReassign} />,
    );
  });
}

const слово = (номер: number) =>
  container.querySelector(`[data-word="${номер}"]`)!.firstChild as Text;

/** Выделить мышью от буквы `с` слова `отНомера` до буквы `до` слова `доНомера` и отпустить кнопку. */
function выделить(отНомера: number, с: number, доНомера: number, до: number) {
  const range = document.createRange();
  range.setStart(слово(отНомера), с);
  range.setEnd(слово(доНомера), до);
  const selection = window.getSelection()!;
  selection.removeAllRanges();
  selection.addRange(range);
  act(() => {
    container.querySelector(".bubble")!.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
  });
}

const меню = () => container.querySelector(".reassign");
const кнопка = (имя: string) =>
  Array.from(container.querySelectorAll<HTMLButtonElement>(".reassign button"))
    .find((b) => b.textContent === имя);

describe("слова реплики другому спикеру", () => {
  it("выделенные слова и спикеры, кроме нынешнего, — в меню под пузырём", () => {
    render(реплика, vi.fn());
    выделить(2, 0, 3, 7);
    expect(меню()?.textContent).toContain("«Нет, погоди.»");
    expect(кнопка("Бабка")).toBeDefined();
    expect(кнопка("Вы")).toBeDefined();
    expect(кнопка("Сатир")).toBeUndefined();  // отдавать тому же — бессмыслица
  });

  it("выбор спикера отправляет номера первого и последнего слова", async () => {
    const onReassign = vi.fn().mockResolvedValue(undefined);
    render(реплика, onReassign);
    выделить(2, 1, 3, 3);  // «ет, пог» — слова задеты частично, но задеты
    await act(async () => кнопка("Бабка")!.click());
    expect(onReassign).toHaveBeenCalledWith(1, 2, 3, 8);
    expect(меню()).toBeNull();
  });

  it("кнопку мыши отпустили за краем пузыря — меню всё равно открывается", () => {
    // Так и выделяют до последнего слова: рука уходит дальше конца строки.
    // Первая версия слушала только сам пузырь и в браузере меню не показывала.
    render(реплика, vi.fn());
    const range = document.createRange();
    range.setStart(слово(2), 0);
    range.setEnd(слово(3), 7);
    window.getSelection()!.removeAllRanges();
    window.getSelection()!.addRange(range);
    act(() => {
      document.body.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
    });
    expect(меню()?.textContent).toContain("«Нет, погоди.»");
  });

  it("нажатие на кнопку спикера сбрасывает выделение, но меню дожидается щелчка", async () => {
    const onReassign = vi.fn().mockResolvedValue(undefined);
    render(реплика, onReassign);
    выделить(2, 0, 3, 7);
    window.getSelection()!.removeAllRanges();  // так ведёт себя браузер при нажатии
    const бабке = кнопка("Бабка")!;
    act(() => {
      бабке.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
    });
    await act(async () => бабке.click());
    expect(onReassign).toHaveBeenCalledWith(1, 2, 3, 8);
  });

  it("слово, которого выделение лишь коснулось краем, не в счёт", () => {
    // Выделение начато ровно за последней буквой «согласен.» — его не выбирали
    render(реплика, vi.fn());
    выделить(1, "согласен.".length, 2, 4);
    expect(меню()?.textContent).toContain("«Нет,»");
    expect(меню()?.textContent).not.toContain("согласен");
  });

  it("выделение, вылезшее за пузырь, — это копирование, а не переназначение", () => {
    render(реплика, vi.fn());
    const range = document.createRange();
    // от подписи с именем (она над пузырём) до середины реплики
    range.setStartBefore(container.querySelector(".msg-name")!);
    range.setEnd(слово(1), 4);
    window.getSelection()!.removeAllRanges();
    window.getSelection()!.addRange(range);
    act(() => {
      container.querySelector(".bubble")!.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
    });
    expect(меню()).toBeNull();
  });

  it("щелчок без выделения меню не открывает — текст по-прежнему можно просто читать", () => {
    render(реплика, vi.fn());
    выделить(1, 3, 1, 3);
    expect(меню()).toBeNull();
  });

  it("ошибка сервера остаётся в меню, реплика не пропадает", async () => {
    const onReassign = vi.fn().mockRejectedValue(new Error("Спикер не найден"));
    render(реплика, onReassign);
    выделить(0, 0, 0, 3);
    await act(async () => кнопка("Бабка")!.click());
    expect(меню()?.textContent).toContain("Спикер не найден");
  });

  it("без времени слов и без onReassign текст выводится как раньше", () => {
    render({ ...реплика, words: null }, vi.fn());
    expect(container.querySelector("[data-word]")).toBeNull();
    render(реплика);
    expect(container.querySelector("[data-word]")).toBeNull();
    expect(container.querySelector(".bubble")?.textContent).toBe(реплика.text);
  });

  it("слова в пузыре склеиваются в тот же текст — копирование не меняется", () => {
    render(реплика, vi.fn());
    expect(container.querySelector(".bubble")?.textContent).toBe(реплика.text);
  });
});
