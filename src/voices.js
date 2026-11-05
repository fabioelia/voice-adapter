// Saved voices (IndexedDB) and .voice.json files. A voice keeps the talent release it was made under.
import { VOICE_LEN } from './config.js';
import { download } from './ui.js';

function db() {
  return new Promise((res, rej) => {
    const q = indexedDB.open('voice-adapter', 1);
    q.onupgradeneeded = () => q.result.createObjectStore('voices', { keyPath: 'id' });
    q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error);
  });
}
async function tx(mode, fn) {
  const d = await db();
  return new Promise((res, rej) => { const t = d.transaction('voices', mode), r = fn(t.objectStore('voices')); t.oncomplete = () => res(r?.result); t.onerror = () => rej(t.error); });
}
export const listVoices = async () => (await tx('readonly', (s) => s.getAll())) || [];
export const deleteVoice = (id) => tx('readwrite', (s) => s.delete(id));

/** Save a training result as a named voice. `release` is the talent's release or contract reference. */
export async function saveVoice(result, { name, release, clips }) {
  if (!name || !release) throw new Error('Enter the talent name and release reference to save the voice.');
  const v = {
    id: 'voice:' + crypto.randomUUID(), name, createdAt: new Date().toISOString(), release,
    ...(result.kind === 'neural'
      ? { kind: 'neural', base: result.base, styleDelta: result.styleDelta, biasDelta: result.biasDelta, f0Ratio: result.f0Ratio, speed: result.speed }
      : { kind: 'search', data: result.data }),
    match: result.heldOut, baseline: { voice: result.baseline.name, match: result.baseline.toTarget }, steps: result.steps, training: clips,
  };
  await tx('readwrite', (s) => s.put(v));
  return v;
}

const b64 = {
  enc: (arr) => { if (!arr) return undefined; const u = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000)); return btoa(s); },
  f32: (str) => (str ? new Float32Array(Uint8Array.from(atob(str), (c) => c.charCodeAt(0)).buffer) : undefined),
};
export function exportVoice(v) {
  const out = { format: 'voice-adapter', version: 1, ...v, data: b64.enc(v.data), styleDelta: b64.enc(v.styleDelta), biasDelta: b64.enc(v.biasDelta) };
  download(`${v.name.replace(/[^\w-]+/g, '_')}.voice.json`, JSON.stringify(out), 'application/json');
}
export async function importVoice(file) {
  const j = JSON.parse(await file.text());
  if (j.format !== 'voice-adapter') throw new Error('Not a voice file.');
  if (!j.release) throw new Error('This voice has no release reference.');
  const v = { ...j, data: b64.f32(j.data), styleDelta: b64.f32(j.styleDelta), biasDelta: b64.f32(j.biasDelta) };
  const ok = v.kind === 'neural' ? v.styleDelta?.length === 128 && v.biasDelta?.length && v.base : v.data?.length === VOICE_LEN;
  if (!ok) throw new Error('The voice data is damaged.');
  delete v.format; delete v.version;
  await tx('readwrite', (s) => s.put(v));
  return v;
}
