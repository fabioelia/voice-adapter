// Kokoro-82M: speech recognition setup, stock and trained voices, synthesis.
import { ASR_MODELS, KOKORO_MODEL, KOKORO_SR, KOKORO_URL, ORT_URL, ORT_WASM, TRAIN_MODELS, TRANSFORMERS_URL } from './config.js';
import { trimSilence } from './audio.js';
import { asrWorker, device, trainWorker, ttsWorker } from './runtime.js';
import { $, log } from './ui.js';

let kokoroDevice = null;
export async function ensureKokoro() {
  if (!kokoroDevice) { kokoroDevice = await ttsWorker.call('load', { libUrl: KOKORO_URL, model: KOKORO_MODEL, device: device() }); log(`Kokoro ready on ${kokoroDevice}.`); }
  return kokoroDevice;
}
export async function ensureAsr() {
  const m = ASR_MODELS.find((x) => x.id === $('asrModel').value) || ASR_MODELS[0];
  const d = $('device').value === 'auto' ? device() : $('device').value;
  return asrWorker.call('loadAsr', { libUrl: TRANSFORMERS_URL, model: m.id, big: !!m.big, device: d });
}

const registered = new Set();
/** Make a voice table (search-trained or stock-shaped) available to Kokoro under `name`. */
export async function registerVoice(name, data) { await ttsWorker.call('registerVoice', { name, data, lang: 'a' }); registered.add(name); }
export const isRegistered = (name) => registered.has(name);
export const forget = (name) => registered.delete(name);

export async function tokenize(text, voice) { await ensureKokoro(); return ttsWorker.call('tokenize', { text, voice }); }

const stock = new Map();
export async function stockTable(name) {
  if (!stock.has(name)) {
    const r = await fetch(`https://huggingface.co/${KOKORO_MODEL}/resolve/main/voices/${name}.bin`);
    if (!r.ok) throw new Error(`Could not download voice ${name}`);
    stock.set(name, new Float32Array(await r.arrayBuffer()));
  }
  return stock.get(name);
}
/** The style row Kokoro uses for an input of n tokens. */
export const styleRow = (table, n) => { const r = 256 * Math.min(Math.max(n - 1, 0), 509); return table.slice(r, r + 256); };

// The split graphs and training step used by deep training (models/kokoro-train/).
const trainLoaded = new Set();
export async function ensureTrainModels(graphs) {
  const need = graphs.filter((g) => !trainLoaded.has(g));
  if (!need.length) return;
  const r = await trainWorker.call('load', { ortUrl: ORT_URL, wasmBase: ORT_WASM, base: TRAIN_MODELS, graphs: need, device: device() });
  need.forEach((g) => trainLoaded.add(g));
  log(`Training models ready: ${Object.entries(r.devices).map(([k, v]) => `${k} on ${v}`).join(', ')}.`);
}

/**
 * Speak `text` with a voice: a stock name, or a trained voice { id, kind, data | base+deltas }.
 * pitch shifts by playback rate (pace is compensated); returns { audio, sr, rate, dur }.
 */
export async function synth(voice, text, { pitch = 1, pace = 1 } = {}) {
  await ensureKokoro();
  let audio, sr = KOKORO_SR;
  if (typeof voice === 'object' && voice.kind === 'neural') {
    await ensureTrainModels(['frontend', 'prosody', 'decoder']);
    const ids = await tokenize(text, voice.base);
    if (!ids.length) return { audio: new Float32Array(0), sr, rate: pitch, dur: 0 };
    const style = styleRow(await stockTable(voice.base), ids.length);
    const speed = Math.max(0.5, Math.min(2, ((voice.speed || 1) * pace) / pitch));
    audio = (await trainWorker.call('render', { ids, style, speed, styleDelta: voice.styleDelta, biasDelta: voice.biasDelta, f0Ratio: voice.f0Ratio || 1 })).audio;
  } else {
    const name = typeof voice === 'object' ? voice.id : voice;
    if (typeof voice === 'object' && !registered.has(name)) await registerVoice(name, voice.data);
    const r = await ttsWorker.call('synth', { text, voice: name, speed: Math.max(0.5, Math.min(2, pace / pitch)) });
    audio = r.audio; sr = r.sr;
  }
  const trimmed = trimSilence(audio, sr);
  return { audio: trimmed, sr, rate: pitch, dur: trimmed.length / sr / pitch };
}
