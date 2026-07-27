import { describe, expect, it } from "vitest";

import { formatTime } from "./Transcript";

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
