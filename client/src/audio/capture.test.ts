import { describe, expect, it } from "vitest";

import { looksLikeLoopback } from "./capture";

const dev = (label: string) => ({ label }) as MediaDeviceInfo;

describe("looksLikeLoopback", () => {
  it("узнаёт виртуальные loopback-драйверы", () => {
    for (const label of ["BlackHole 2ch", "Loopback Audio", "VB-Cable", "VBCable", "Soundflower (2ch)", "Virtual Mic"]) {
      expect(looksLikeLoopback(dev(label))).toBe(true);
    }
  });

  it("не путает обычные устройства с loopback", () => {
    for (const label of ["MacBook Pro Microphone", "AirPods", "USB Audio Device"]) {
      expect(looksLikeLoopback(dev(label))).toBe(false);
    }
  });
});
