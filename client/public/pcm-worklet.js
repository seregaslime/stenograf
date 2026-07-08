// AudioWorklet: собирает входной звук в кадры PCM16 по 100 мс (1600 сэмплов при 16 кГц)
// и отдаёт их в основной поток вместе с RMS-уровнем для индикатора громкости.
class Pcm16Processor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Int16Array(1600);
    this._filled = 0;
    this._sumSquares = 0;
    this._count = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      const sample = Math.max(-1, Math.min(1, channel[i]));
      this._buffer[this._filled++] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      this._sumSquares += sample * sample;
      this._count++;
      if (this._filled === this._buffer.length) {
        const out = this._buffer.slice();
        const level = Math.sqrt(this._sumSquares / this._count);
        this.port.postMessage({ pcm: out.buffer, level }, [out.buffer]);
        this._filled = 0;
        this._sumSquares = 0;
        this._count = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm16", Pcm16Processor);
