/**
 * Сверка с difflib из Python: порог дедупликации 0.85 подбирался под этот
 * алгоритм, и «похожая» мера означала бы другой порог. Эталон снят с сервера.
 */
import { describe, expect, it } from "vitest";

import golden from "./similarity-golden.json";
import { similarityRatio } from "./similarity";

describe("похожесть строк совпадает с difflib", () => {
  it.each(golden.map((с) => [с.a.slice(0, 40), с] as const))("«%s…»", (_имя, случай) => {
    expect(similarityRatio(случай.a, случай.b)).toBeCloseTo(случай.ratio, 10);
  });
});

describe("свойства меры", () => {
  it("одинаковые строки — единица", () => {
    expect(similarityRatio("подсказка", "подсказка")).toBe(1);
  });

  it("порог 0.85 ловит перефразировку и пропускает разное", () => {
    const дубль = similarityRatio(
      "уточните срок задачи у ивана",
      "уточните срок этой задачи у ивана",
    );
    const другое = similarityRatio(
      "уточните срок задачи у ивана",
      "sla это соглашение об уровне сервиса",
    );
    expect(дубль).toBeGreaterThanOrEqual(0.85);
    expect(другое).toBeLessThan(0.85);
  });
});
