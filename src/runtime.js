// Workers and compute device.
import { $, log, onDownload, setProgress, setStage } from './ui.js';

export const gpu = { available: false };
export async function detectGPU() {
  try { if (navigator.gpu) gpu.available = !!(await navigator.gpu.requestAdapter()); } catch (e) { /* no WebGPU */ }
  return gpu.available;
}
/** 'webgpu' or 'wasm', from the Compute setting. */
export const device = () => ($('device').value === 'wasm' ? 'wasm' : $('device').value === 'webgpu' || gpu.available ? 'webgpu' : 'wasm');

class WorkerClient {
  constructor(file) { this.url = new URL(`./workers/${file}`, import.meta.url); this.w = null; this.seq = 0; this.pending = new Map(); }
  get worker() {
    if (!this.w) {
      this.w = new Worker(this.url, { type: 'module' });
      this.w.onmessage = (ev) => this.onMessage(ev.data);
      this.w.onerror = (e) => { for (const p of this.pending.values()) p.reject(new Error(e.message || 'Worker crashed')); this.pending.clear(); };
    }
    return this.w;
  }
  call(type, payload, { onProgress, transfer } = {}) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.worker.postMessage({ id, type, payload }, transfer || []);
    });
  }
  onMessage(m) {
    if (m.type === 'download') return onDownload(m);
    if (m.type === 'log') return log(m.message);
    if (m.type === 'status') { setStage(m.message); return log(m.message); }
    const p = this.pending.get(m.id);
    if (!p) return;
    if (m.type === 'progress') return p.onProgress ? p.onProgress(m) : setProgress(m.value, m.message);
    this.pending.delete(m.id);
    if (m.type === 'done') p.resolve(m.result); else p.reject(new Error(m.message));
  }
}

export const asrWorker = new WorkerClient('asr.worker.js');        // Whisper + WeSpeaker
export const ttsWorker = new WorkerClient('tts.worker.js');        // Kokoro
export const sepWorker = new WorkerClient('separator.worker.js');  // TIGER-DnR
export const trainWorker = new WorkerClient('train.worker.js');    // Kokoro split graphs + training step
