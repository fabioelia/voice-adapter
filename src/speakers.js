// Voice prints (WeSpeaker ResNet34) and grouping speech by speaker.
import { EMB_MODEL, SR, TRANSFORMERS_URL } from './config.js';
import { asrWorker, device } from './runtime.js';
import { speechOnly } from './audio.js';

export const cosine = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
export const unitMean = (vs) => {
  const m = new Float32Array(vs[0].length);
  for (const v of vs) for (let i = 0; i < m.length; i++) m[i] += v[i];
  let s = 0; for (const x of m) s += x * x; s = Math.sqrt(s) || 1;
  return m.map((x) => x / s);
};

export const ensureEmb = () => asrWorker.call('loadEmb', { libUrl: TRANSFORMERS_URL, model: EMB_MODEL, device: device() });
export const embed16k = (pcm) => asrWorker.call('embed', { audio: pcm });

/** Voice print of recordings: mean of 6 s speech windows, at most 50. */
export async function voicePrint(pcms) {
  await ensureEmb();
  const vecs = [];
  for (const pcm of pcms) {
    const sp = speechOnly(pcm);
    for (let s = 0; s + 2 * SR <= sp.length && vecs.length < 50; s += 6 * SR) vecs.push(await embed16k(sp.slice(s, s + 6 * SR)));
  }
  if (!vecs.length) throw new Error('Not enough speech: add at least a few seconds of clear speech.');
  return unitMean(vecs);
}

/** Speech regions of 16 kHz audio from frame energy (reliable on isolated voice). */
export function speechRegions(pcm) {
  const F = 480, n = Math.floor(pcm.length / F), e = new Float32Array(n);
  for (let i = 0; i < n; i++) { let s = 0; for (let j = i * F; j < (i + 1) * F; j++) s += pcm[j] * pcm[j]; e[i] = Math.sqrt(s / F); }
  const sorted = Float32Array.from(e).sort(), floor = sorted[Math.floor(n * 0.2)] || 0, top = sorted[Math.floor(n * 0.95)] || 0;
  const th = Math.max(0.004, floor * 2.5, top * 0.08), regions = [];
  let cur = null, gap = 0;
  for (let i = 0; i < n; i++) {
    if (e[i] > th) { if (!cur) cur = { s: i, e: i }; cur.e = i; gap = 0; }
    else if (cur && ++gap > 8) { regions.push(cur); cur = null; }
  }
  if (cur) regions.push(cur);
  return regions.map((r) => ({ start: (r.s * F) / SR, end: ((r.e + 1) * F) / SR })).filter((r) => r.end - r.start >= 0.6);
}

const nearest = (emb, cents) => { let k = -1, s = -1; cents.forEach((c, i) => { const v = cosine(emb, c.v); if (v > s) { s = v; k = i; } }); return [k, s]; };
/**
 * Group windows { start, end, emb } by voice: assign each to the nearest group or start a new one,
 * refine the centres, then merge groups whose centres are close (one speaker split by emotion).
 */
export function clusterWindows(wins, threshold = 0.4, mergeAt = 0.5) {
  let cents = [];
  for (const w of wins) {
    const [k, s] = nearest(w.emb, cents);
    if (s >= threshold) { const c = cents[k]; c.sum.forEach((x, i) => { c.sum[i] += w.emb[i]; }); c.v = unitMean([c.sum]); }
    else cents.push({ sum: Float32Array.from(w.emb), v: w.emb });
  }
  for (let it = 0; it < 4; it++) {
    for (const w of wins) [w.k, w.sim] = nearest(w.emb, cents);
    cents = cents.map((c, k) => { const m = wins.filter((w) => w.k === k); return m.length ? { v: unitMean(m.map((w) => w.emb)) } : null; });
    const keep = cents.map((c, k) => (c ? k : -1)).filter((k) => k >= 0), remap = new Map(keep.map((k, i) => [k, i]));
    cents = keep.map((k) => cents[k]); for (const w of wins) w.k = remap.get(w.k);
  }
  for (let merged = true; merged && cents.length > 1;) {
    merged = false;
    let bi = -1, bj = -1, bs = mergeAt;
    for (let i = 0; i < cents.length; i++) for (let j = i + 1; j < cents.length; j++) { const v = cosine(cents[i].v, cents[j].v); if (v > bs) { bs = v; bi = i; bj = j; } }
    if (bi >= 0) {
      for (const w of wins) if (w.k === bj) w.k = bi;
      cents[bi] = { v: unitMean(wins.filter((w) => w.k === bi).map((w) => w.emb)) };
      cents.splice(bj, 1); for (const w of wins) if (w.k > bj) w.k--;
      merged = true;
    }
  }
  for (const w of wins) [w.k, w.sim] = nearest(w.emb, cents);
  return cents.map((c, k) => {
    const m = wins.filter((w) => w.k === k);
    return { centroid: c.v, windows: m, seconds: m.reduce((n, w) => n + (w.end - w.start), 0) / 2 + 1 };
  }).sort((a, b) => b.seconds - a.seconds);
}
