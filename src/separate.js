// Isolate voices from music and effects with TIGER-DnR (separator.worker.js).
import { ORT_URL, ORT_WASM, SEPARATOR_URL, SEP_SR, SR } from './config.js';
import { decodeRange, resample } from './audio.js';
import { device, sepWorker } from './runtime.js';
import { fmtTime, log, setProgress, setStage } from './ui.js';

// The model under-estimates speech it is unsure of (non-English speech came out at 0.84x its level).
// Music and effects are uncorrelated with speech, so per 1 s window the scale that leaves the rest
// uncorrelated with the estimate is applied (clamped 1–1.4; 1 where there is little speech).
function speechGain(mid, d, s, e) {
  const W = SEP_SR, H = SEP_SR >> 2, cs = [], vs = [];
  for (let c = s; c < e + H; c += H) {
    const lo = Math.max(0, c - W / 2), hi = Math.min(d.length, c + W / 2);
    let md = 0, dd = 0, mm = 0;
    for (let i = lo; i < hi; i++) { md += mid[i] * d[i]; dd += d[i] * d[i]; mm += mid[i] * mid[i]; }
    const n = Math.max(1, hi - lo), rd = Math.sqrt(dd / n), rm = Math.sqrt(mm / n);
    cs.push(c); vs.push(rd >= 0.01 && rd >= 0.25 * rm ? Math.min(1.4, Math.max(1, md / dd)) : 1);
  }
  const g = new Float32Array(e - s);
  for (let i = s, k = 0; i < e; i++) {
    while (k + 1 < cs.length && cs[k + 1] <= i) k++;
    const c0 = cs[k], c1 = cs[k + 1] ?? c0, t = c1 > c0 ? (i - c0) / (c1 - c0) : 0;
    g[i - s] = vs[k] + ((vs[k + 1] ?? vs[k]) - vs[k]) * Math.max(0, Math.min(1, t));
  }
  return g;
}

let loaded = null;
/** Speech of the first `maxSec` seconds of a file, isolated from music and effects, at 16 kHz. */
export async function isolateVoice(file, maxSec, hop = 6) {
  setStage(`Decoding ${file.name}`); setProgress(null);
  const { chans: [L, R] } = await decodeRange(file, 0, maxSec, SEP_SR, 2);
  const n = L.length, mid = new Float32Array(n);
  for (let i = 0; i < n; i++) mid[i] = (L[i] + R[i]) / 2;
  if (!loaded) { loaded = await sepWorker.call('load', { ortUrl: ORT_URL, wasmBase: ORT_WASM, modelUrl: SEPARATOR_URL, device: device() }); log(`Voice separator ready on ${loaded}.`); }
  setStage('Separating voices from music and effects'); setProgress(0);
  const out = new Float32Array(Math.floor((n * SR) / SEP_SR)), BLOCK = 60 * SEP_SR, PAD = 6 * SEP_SR, t0 = performance.now();
  for (let s = 0; s < n; s += BLOCK) {
    const a = Math.max(0, s - PAD), b = Math.min(n, s + BLOCK + PAD), e = Math.min(n, s + BLOCK);
    const block = mid.slice(a, b);
    const d = await sepWorker.call('dialogue', { mono: block, hop }, { transfer: [block.buffer], onProgress: (m) => {
      const done = (s + (e - s) * m.value) / n, secs = (performance.now() - t0) / 1000;
      setProgress(done, `${fmtTime((done * n) / SEP_SR)} of ${fmtTime(n / SEP_SR)}${secs > 15 && done > 0.02 ? ` · ${fmtTime((secs / done) * (1 - done))} left` : ''}`);
    } });
    const g = speechGain(mid.subarray(a, b), d, s - a, e - a);
    for (let i = s; i < e; i++) d[i - a] *= g[i - s];
    const o = Math.floor((s * SR) / SEP_SR), d16 = resample(d.subarray(s - a, e - a), SEP_SR, SR);
    out.set(d16.subarray(0, Math.max(0, Math.min(d16.length, out.length - o))), o);
  }
  log(`Separated ${fmtTime(n / SEP_SR)} in ${fmtTime((performance.now() - t0) / 1000)}.`);
  return out;
}
