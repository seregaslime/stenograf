import { describe, expect, it } from "vitest";

import type { SegmentDto } from "../types";
import { formatTime, groupSegments } from "./Transcript";

describe("formatTime", () => {
  it("форматирует секунды меньше минуты", () => {
    expect(formatTime(0)).toBe("00:00");
    expect(formatTime(5)).toBe("00:05");
    expect(formatTime(59)).toBe("00:59");
  });

  it("форматирует минуты:секунды", () => {
    expect(formatTime(65)).toBe("01:05");
    expect(formatTime(600)).toBe("10:00");
  });

  it("добавляет часы начиная с 3600 с", () => {
    expect(formatTime(3600)).toBe("1:00:00");
    expect(formatTime(3661)).toBe("1:01:01");
  });

  it("зажимает отрицательные к нулю и отбрасывает дробь", () => {
    expect(formatTime(-10)).toBe("00:00");
    expect(formatTime(9.9)).toBe("00:09");
  });
});

describe("groupSegments", () => {
  let nextId = 1;
  const seg = (speakerId: number | null, start: number, end: number): SegmentDto => ({
    id: nextId++,
    meeting_id: 1,
    channel: "mic",
    start_s: start,
    end_s: end,
    text: "реплика",
    similarity: null,
    speaker: speakerId === null ? null : { id: speakerId, name: `С${speakerId}`, is_self: false },
  });

  it("склеивает подряд идущие реплики одного спикера", () => {
    const groups = groupSegments([seg(1, 0, 2), seg(1, 3, 5), seg(1, 6, 8)]);
    expect(groups.map((g) => g.length)).toEqual([3]);
  });

  it("рвёт группу на смене спикера", () => {
    const groups = groupSegments([seg(1, 0, 2), seg(2, 3, 5), seg(1, 6, 8)]);
    expect(groups.map((g) => g.length)).toEqual([1, 1, 1]);
  });

  it("рвёт группу после паузы длиннее 30 секунд", () => {
    // Пауза считается от конца предыдущей реплики, а не от её начала: иначе
    // длинный монолог рвался бы сам по себе.
    const groups = groupSegments([seg(1, 0, 10), seg(1, 39, 41), seg(1, 100, 102)]);
    expect(groups.map((g) => g.length)).toEqual([2, 1]);
  });

  it("не считает неизвестных спикеров одним человеком по ошибке", () => {
    // speaker === null у всех неопознанных — но они идут подряд и в ленте
    // выглядят как один «Неизвестный», так и группируем.
    const groups = groupSegments([seg(null, 0, 2), seg(null, 3, 5)]);
    expect(groups.map((g) => g.length)).toEqual([2]);
  });

  it("пустой список даёт пустой результат", () => {
    expect(groupSegments([])).toEqual([]);
  });
});
