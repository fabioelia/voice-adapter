class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(2048); this.n = 0; }
  process(inputs) {
    const inp = inputs[0];
    if (inp && inp.length) {
      const ch = inp.length, len = inp[0].length;
      for (let i = 0; i < len; i++) {
        let s = 0; for (let c = 0; c < ch; c++) s += inp[c][i];
        this.buf[this.n++] = s / ch;
        if (this.n === this.buf.length) { this.port.postMessage(this.buf.slice()); this.n = 0; }
      }
    }
    return true;
  }
}
registerProcessor('mic-tap', Tap);
