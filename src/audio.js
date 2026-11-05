// Decoding, resampling, playback, microphone and WAV.
import { SR } from './config.js';

let ctx = null, out = null, tapReady = false;
export function audioContext() {
  if (!ctx) { ctx = new AudioContext(); out = ctx.createGain(); out.connect(ctx.destination); }
  if (ctx.state !== 'running') ctx.resume();
  return ctx;
}

const playing = new Set();
/** Play mono samples at `when` (AudioContext time, default now) at `rate`. */
export function play(samples, sr, { when, rate = 1 } = {}) {
  const c = audioContext(), b = c.createBuffer(1, samples.length, sr);
  b.copyToChannel(samples, 0);
  const src = c.createBufferSource();
  src.buffer = b; src.playbackRate.value = rate; src.connect(out);
  src.start(when ?? c.currentTime + 0.05);
  playing.add(src); src.onended = () => playing.delete(src);
  return src;
}
export function stopAll() { for (const s of playing) { try { s.stop(); } catch (e) { /* already stopped */ } } playing.clear(); }

export function resample(a, from, to) {
  if (from === to) return a;
  const n = Math.floor((a.length * to) / from), o = new Float32Array(n), r = from / to;
  for (let i = 0; i < n; i++) { const p = i * r, k = Math.floor(p), f = p - k; o[i] = a[k] * (1 - f) + (a[k + 1] ?? a[k]) * f; }
  return o;
}
export function concat(parts) {
  const o = new Float32Array(parts.reduce((n, x) => n + x.length, 0)); let i = 0;
  for (const x of parts) { o.set(x, i); i += x.length; }
  return o;
}
export const rms = (a) => { let e = 0; for (const x of a) e += x * x; return Math.sqrt(e / (a.length || 1)); };

/** Decode `durSec` seconds from `startSec` at `rate` with `channels` (folded down by the browser). */
export async function decodeRange(file, startSec, durSec, rate, channels) {
  const c = new AudioContext();
  try {
    const d = await c.decodeAudioData(await file.arrayBuffer());
    const len = Math.max(1, Math.floor(Math.min(durSec, d.duration - startSec) * rate));
    const off = new OfflineAudioContext(channels, len, rate), src = off.createBufferSource();
    src.buffer = d; src.connect(off.destination); src.start(0, startSec, len / rate);
    const r = await off.startRendering();
    return { chans: Array.from({ length: channels }, (_, k) => r.getChannelData(Math.min(k, r.numberOfChannels - 1))), duration: d.duration, layout: d.numberOfChannels };
  } catch (e) { throw new Error(`Could not decode ${file.name}: ${e.message || e}`); }
  finally { c.close(); }
}
export const decodeTo16k = async (file) => (await decodeRange(file, 0, Infinity, SR, 1)).chans[0];

/** Keep only frames with speech energy, so voice prints are not diluted by pauses. */
export function speechOnly(pcm) {
  const f = 480, keep = [];
  let peak = 0;
  for (let i = 0; i < pcm.length; i += f) peak = Math.max(peak, rms(pcm.subarray(i, i + f)));
  const th = Math.max(0.004, peak * 0.08);
  for (let i = 0; i < pcm.length; i += f) { const fr = pcm.subarray(i, Math.min(pcm.length, i + f)); if (rms(fr) > th) keep.push(fr); }
  return concat(keep);
}
/** Median pitch (Hz) of 16 kHz speech by autocorrelation, 80–400 Hz; null when unvoiced. */
export function medianPitch(pcm) {
  const est = [], N = 1024;
  for (let s = 0; s + N < pcm.length; s += 1600) {
    const x = pcm.subarray(s, s + N);
    let e = 0; for (let i = 0; i < N; i++) e += x[i] * x[i];
    if (e / N < 1e-4) continue;
    let best = 0, lag = 0;
    for (let L = 40; L <= 200; L++) { let c = 0; for (let i = 0; i + L < N; i++) c += x[i] * x[i + L]; if (c > best) { best = c; lag = L; } }
    if (lag && best / e > 0.35) est.push(SR / lag);
  }
  if (!est.length) return null;
  est.sort((a, b) => a - b);
  return est[est.length >> 1];
}
export function trimSilence(a, sr) {
  let s = 0, e = a.length;
  while (s < e && Math.abs(a[s]) < 0.004) s++;
  while (e > s && Math.abs(a[e - 1]) < 0.004) e--;
  return a.subarray(Math.max(0, s - Math.round(sr * 0.02)), Math.min(a.length, e + Math.round(sr * 0.06)));
}

/** Record the microphone; returns { stop(): Float32Array at 16 kHz }. */
export async function recordMic() {
  const c = audioContext();
  if (!tapReady) { await c.audioWorklet.addModule(new URL('./workers/capture.worklet.js', import.meta.url)); tapReady = true; }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: true, autoGainControl: true } });
  const src = c.createMediaStreamSource(stream), tap = new AudioWorkletNode(c, 'mic-tap', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
  const sink = c.createGain(); sink.gain.value = 0; tap.connect(sink); sink.connect(c.destination); src.connect(tap);
  const chunks = [];
  tap.port.onmessage = (ev) => chunks.push(ev.data);
  return {
    stop() {
      try { src.disconnect(); tap.disconnect(); sink.disconnect(); } catch (e) { /* already disconnected */ }
      stream.getTracks().forEach((t) => t.stop());
      return resample(concat(chunks), c.sampleRate, SR);
    },
  };
}

/** 16-bit mono WAV. */
export function wavBlob(f32, sr) {
  const h = new DataView(new ArrayBuffer(44)), bytes = f32.length * 2;
  const w = (o, s) => [...s].forEach((ch, i) => h.setUint8(o + i, ch.charCodeAt(0)));
  w(0, 'RIFF'); h.setUint32(4, 36 + bytes, true); w(8, 'WAVE'); w(12, 'fmt ');
  h.setUint32(16, 16, true); h.setUint16(20, 1, true); h.setUint16(22, 1, true);
  h.setUint32(24, sr, true); h.setUint32(28, sr * 2, true); h.setUint16(32, 2, true); h.setUint16(34, 16, true);
  w(36, 'data'); h.setUint32(40, bytes, true);
  const pcm = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) { const v = Math.max(-1, Math.min(1, f32[i])); pcm[i] = v < 0 ? v * 0x8000 : v * 0x7fff; }
  return new Blob([h, pcm], { type: 'audio/wav' });
}
