/** Выбор модели: список скачанных, недоступная Ollama и модель не из списка. */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ModelSelect from "./ModelSelect";

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

function render(value: string, models: string[], onChange = vi.fn()) {
  act(() => {
    root.render(<ModelSelect value={value} models={models} placeholder="bge-m3" onChange={onChange} />);
  });
  return onChange;
}

const варианты = () =>
  Array.from(container.querySelectorAll("option")).map((o) => o.textContent);

describe("выбор модели", () => {
  it("список скачанных моделей — выпадающим списком, а не полем ввода", () => {
    render("bge-m3", ["qwen3:4b", "bge-m3"]);
    expect(container.querySelector("select")).not.toBeNull();
    expect(container.querySelector("input")).toBeNull();
    expect(варианты()).toEqual(["qwen3:4b", "bge-m3"]);
    expect(container.querySelector("select")!.value).toBe("bge-m3");
  });

  it("выбранная модель не из списка остаётся и помечена", () => {
    // Ollama переехала на другую машину — подменять выбор человека нельзя
    render("qwen3-embedding:0.6b", ["bge-m3"]);
    expect(варианты()).toEqual(["qwen3-embedding:0.6b (не скачана)", "bge-m3"]);
    expect(container.querySelector("select")!.value).toBe("qwen3-embedding:0.6b");
  });

  it("список пуст (Ollama не ответила) — остаётся поле ввода", () => {
    render("bge-m3", []);
    expect(container.querySelector("input")?.value).toBe("bge-m3");
    expect(container.querySelector("select")).toBeNull();
  });

  it("ничего не выбрано — подсказка вместо пустой строки", () => {
    render("", ["bge-m3"]);
    expect(варианты()).toEqual(["— выберите модель —", "bge-m3"]);
  });

  it("выбор и ввод сообщаются наверх", () => {
    const список = render("bge-m3", ["bge-m3", "nomic-embed-text"]);
    const select = container.querySelector("select")!;
    select.value = "nomic-embed-text";
    act(() => select.dispatchEvent(new Event("change", { bubbles: true })));
    expect(список).toHaveBeenCalledWith("nomic-embed-text");

    const ввод = render("bge", []);
    const input = container.querySelector("input")!;
    // React следит за value сам — меняем через системный сеттер, иначе он
    // считает, что значение не менялось, и onChange не сработает
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "bge-m3");
    act(() => input.dispatchEvent(new Event("input", { bubbles: true })));
    expect(ввод).toHaveBeenCalledWith("bge-m3");
  });
});
