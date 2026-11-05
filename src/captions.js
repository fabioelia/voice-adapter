// Captions -> timed training sentences: WebVTT, SRT and TTML/DFXP/IMSC; speaker names from <v Name>,
// TTML agents or "NAME:" prefixes; two-speaker cues split; sound descriptions kept apart from speech.
// Captions are aligned to the speech with Silero VAD.
import { ORT_URL, ORT_WASM, SR, VAD_URL, CACHE } from './config.js';
import { setProgress, setStage } from './ui.js';

// TTML / DFXP / IMSC -> cues [{ start, end, text, speaker }] via DOMParser: clock and offset times
// (frames, ticks), begin offsets inherited from <body>/<div>, <br/>, nested <span>s, ttm:agent speakers.
function ttmlToCues(xml) {
  const doc = new DOMParser().parseFromString(xml, 'application/xml');
  const tt = doc.documentElement;
  if (!tt || tt.localName !== 'tt') return [];
  const attr = (el, name) => { for (const a of el.attributes) if (a.localName === name) return a.value; return null; };
  const frameRate = Number(attr(tt, 'frameRate')) || 30;
  const [mn, md] = (attr(tt, 'frameRateMultiplier') || '1 1').split(/\s+/).map(Number);
  const fps = frameRate * (mn && md ? mn / md : 1);
  const subFrameRate = Number(attr(tt, 'subFrameRate')) || 1;
  const tickRate = Number(attr(tt, 'tickRate')) || (attr(tt, 'frameRate') ? frameRate * subFrameRate : 1);
  const time = (v) => {
    if (v == null) return null;
    v = v.trim();
    let m = v.match(/^(\d+):(\d\d):(\d\d)(?:\.(\d+))?$/);
    if (m) return +m[1] * 3600 + +m[2] * 60 + +m[3] + (m[4] ? Number('0.' + m[4]) : 0);
    m = v.match(/^(\d+):(\d\d):(\d\d):(\d+)(?:\.(\d+))?$/);
    if (m) return +m[1] * 3600 + +m[2] * 60 + +m[3] + (+m[4] + (m[5] ? +m[5] / subFrameRate : 0)) / fps;
    m = v.match(/^(\d+(?:\.\d+)?)(h|ms|m|s|f|t)$/);
    if (m) { const n = +m[1]; return { h: n * 3600, m: n * 60, s: n, ms: n / 1000, f: n / fps, t: n / tickRate }[m[2]]; }
    return null;
  };
  const agents = new Map();
  for (const a of doc.getElementsByTagNameNS('*', 'agent')) {
    const id = a.getAttribute('xml:id') || attr(a, 'id');
    const name = [...a.getElementsByTagNameNS('*', 'name')].map((n) => n.textContent.trim()).find(Boolean);
    if (id) agents.set(id, name || id);
  }
  const speakerOf = (el) => {
    for (let e = el; e && e.nodeType === 1; e = e.parentNode) {
      const ag = attr(e, 'agent'); if (ag) return ag.split(/\s+/).map((x) => agents.get(x) || x).join(' & ');
    }
    return null;
  };
  const textOf = (node) => {
    let out = '';
    for (const c of node.childNodes) {
      if (c.nodeType === 3) out += c.nodeValue;
      else if (c.nodeType === 1) out += c.localName === 'br' ? '\n' : textOf(c);
    }
    return out;
  };
  const cues = [];
  for (const p of doc.getElementsByTagNameNS('*', 'p')) {
    let base = 0;                                   // time containment: begins on ancestors add up
    for (let e = p.parentNode; e && e.nodeType === 1 && e !== tt; e = e.parentNode) base += time(attr(e, 'begin')) || 0;
    const b = time(attr(p, 'begin')), e = time(attr(p, 'end')), d = time(attr(p, 'dur'));
    if (b == null) continue;
    const start = base + b, end = e != null ? base + e : d != null ? start + d : null;
    const text = textOf(p).split('\n').map((l) => l.replace(/\s+/g, ' ').trim()).filter(Boolean).join('\n');
    if (end != null && end > start && text) cues.push({ start, end, text, speaker: speakerOf(p) || [...p.getElementsByTagNameNS('*', 'span')].map(speakerOf).find(Boolean) || null });
  }
  return cues.sort((x, y) => x.start - y.start);
}
// Captions -> dialogue cues [{ start, end, text, speaker }] plus cues.sounds [{ start, end, label }].
// Speakers come from <v Name>, TTML agents or "NAME:" prefixes. A cue holding two speakers
// ("- Hi.\n- Hello.") is split, with its time shared in proportion to each part's length. SDH sound
// descriptions ([door slams], (laughs)) and sung lines (♪ … ♪) become sound events, not dialogue.
const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ' };
const decodeEntities = (x) => x.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (all, e) => {
  if (e[0] === '#') { const c = /^#x/i.test(e) ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10); return Number.isFinite(c) ? String.fromCodePoint(c) : all; }
  return ENTITIES[e.toLowerCase()] ?? all;
});
const SPEAKER_PREFIX = /^([A-Z][A-Z0-9 .'-]{1,30}):\s*(.+)$/;
function cueParts(start, end, raw, speaker, sounds) {
  let t = decodeEntities(raw).replace(/\{\\?[^}]*\}/g, '');
  for (const m of t.matchAll(/\[([^\]]{1,60})\]|\(([^)]{1,60})\)/g)) sounds.push({ start, end, label: (m[1] || m[2]).replace(/\s+/g, ' ').trim().toLowerCase() });
  if (/[♪♫]/.test(t)) sounds.push({ start, end, label: 'music' });
  t = t.replace(/\[[^\]]*\]|\([^)]*\)/g, ' ').split('\n').filter((l) => !/[♪♫]/.test(l)).join('\n');
  const lines = t.split('\n').map((l) => l.replace(/\s+/g, ' ').trim()).filter((l) => l && !/^[-–—\s.…]*$/.test(l));
  if (!lines.length) return [];
  const parts = [];
  const dashed = lines.filter((l) => /^[-–—]\s*\S/.test(l)).length >= 2;
  if (dashed) { for (const l of lines) { if (/^[-–—]/.test(l) || !parts.length) parts.push(l); else parts[parts.length - 1] += ' ' + l; } }
  else parts.push(lines.join(' '));
  const out = parts.map((x) => {
    let txt = x.replace(/^[-–—]\s*/, '').trim(), who = parts.length === 1 ? speaker || null : null;
    const pre = txt.match(SPEAKER_PREFIX);
    if (pre) { who = pre[1].trim(); txt = pre[2].trim(); }
    return { text: txt, speaker: who };
  }).filter((x) => x.text && /[\p{L}\p{N}]/u.test(x.text));
  const total = out.reduce((n, x) => n + x.text.length, 0) || 1;
  let at = start;
  return out.map((x) => { const d = (end - start) * x.text.length / total, c = { start: at, end: at + d, text: x.text, speaker: x.speaker, split: out.length > 1 }; at += d; return c; });
}
export function parseCaptions(text) {
  text = String(text).replace(/^﻿/, '');
  const cues = [], sounds = [];
  if (/^\s*(<\?xml[^>]*>\s*)?(<!--[\s\S]*?-->\s*)*<tt[\s>]/.test(text)) {          // TTML / DFXP / IMSC
    for (const c of ttmlToCues(text)) cues.push(...cueParts(c.start, c.end, c.text, c.speaker, sounds));
  } else {
    const t = (x) => { const m = x.trim().replace(',', '.').match(/(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)/); return m ? (Number(m[1] || 0) * 3600 + Number(m[2]) * 60 + Number(m[3])) : NaN; };
    for (const block of text.replace(/\r/g, '').split(/\n\s*\n/)) {
      const lines = block.split('\n').filter((l) => l.trim());
      const i = lines.findIndex((l) => l.includes('-->')); if (i < 0) continue;
      const [a, b] = lines[i].split('-->'); const start = t(a), end = t(b.split(/\s+/).filter(Boolean)[0] || '');
      if (!(Number.isFinite(start) && Number.isFinite(end) && end > start)) continue;
      let raw = lines.slice(i + 1).join('\n');
      const speaker = (raw.match(/<v(?:\.[^ >]*)?\s+([^>]+)>/) || [])[1] || null;
      raw = raw.replace(/<br\s*\/?>/gi, '\n').replace(/<[^>]+>/g, '');
      cues.push(...cueParts(start, end, raw, speaker, sounds));
    }
  }
  cues.sort((x, y) => x.start - y.start);
  cues.sounds = sounds.sort((x, y) => x.start - y.start);
  return cues;
}

/** Cues -> sentences of up to 12 s (consecutive cues < 0.6 s apart join), each with its 16 kHz audio. */
export function captionUtterances(cues, pcm) {
  const out = []; let cur = null;
  for (const c of cues) {
    if (cur && c.start - cur.end < 0.6 && c.end - cur.start <= 12) { cur.end = c.end; cur.text += ' ' + c.text; }
    else { cur = { start: c.start, end: c.end, text: c.text }; out.push(cur); }
  }
  return out.filter((u) => u.end - u.start >= 1)
    .map((u) => ({ text: u.text, pcm: pcm.slice(Math.max(0, Math.floor((u.start - 0.05) * SR)), Math.min(pcm.length, Math.ceil((u.end + 0.1) * SR))) }))
    .filter((u) => u.pcm.length > SR * 0.8);
}

// Silero VAD speech probability every 32 ms.
let vad = null, ort = null;
async function speechTrack(pcm, seconds) {
  if (!vad) {
    ort = await import(ORT_URL);
    ort.env.wasm.wasmPaths = ORT_WASM;
    let bytes;
    try { const hit = await (await caches.open(CACHE)).match(VAD_URL); if (hit) bytes = new Uint8Array(await hit.arrayBuffer()); } catch (e) { /* no cache */ }
    if (!bytes) {
      const r = await fetch(VAD_URL); if (!r.ok) throw new Error('Could not download the voice-activity model.');
      bytes = new Uint8Array(await r.arrayBuffer());
      try { await (await caches.open(CACHE)).put(VAD_URL, new Response(bytes)); } catch (e) { /* no cache */ }
    }
    vad = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  }
  const W = 512, C = 64, n = Math.floor(Math.min(pcm.length, seconds * SR) / W), out = new Float32Array(n);
  let state = new ort.Tensor('float32', new Float32Array(2 * 128), [2, 1, 128]), ctx = new Float32Array(C);
  const sr = new ort.Tensor('int64', BigInt64Array.from([16000n]), []), buf = new Float32Array(C + W);
  for (let k = 0; k < n; k++) {
    buf.set(ctx, 0); buf.set(pcm.subarray(k * W, (k + 1) * W), C);
    const o = await vad.run({ input: new ort.Tensor('float32', buf.slice(), [1, C + W]), state, sr });
    out[k] = o.output.data[0]; state = o.stateN; ctx = pcm.slice((k + 1) * W - C, (k + 1) * W);
    if (k % 500 === 0) setProgress(k / n, 'Finding speech');
  }
  return out;
}

/**
 * Constant shift between captions and speech: the VAD track correlated with the caption on/off
 * track within ±15 s over the first 4 minutes. Needs 12+ lines; applied only when clearly better.
 */
export async function alignCaptions(cues, pcm) {
  setStage('Aligning captions to the speech'); setProgress(0);
  const hop = 512 / SR, sp = await speechTrack(pcm, 240), n = sp.length, cu = new Float32Array(n);
  for (const c of cues) for (let t = Math.max(0, Math.floor(c.start / hop)); t < Math.min(n, Math.ceil(c.end / hop)); t++) cu[t] = 1;
  const ms = sp.reduce((a, b) => a + b, 0) / n, mc = cu.reduce((a, b) => a + b, 0) / n;
  if (!mc) return { offset: 0, how: 'no caption lines in the first minutes' };
  const inWindow = cues.filter((c) => c.start < n * hop).length;
  if (inWindow < 12) return { offset: 0, how: `only ${inWindow} lines, too few to align automatically` };
  const score = (d) => { let s = 0; for (let t = Math.max(0, d); t < Math.min(n, n + d); t++) s += (sp[t] - ms) * (cu[t - d] - mc); return s; };
  const D = Math.round(15 / hop);
  let best = 0, bestScore = score(0); const s0 = bestScore;
  for (let d = -D; d <= D; d++) { const v = score(d); if (v > bestScore) { bestScore = v; best = d; } }
  const shift = best * hop;
  if (Math.abs(shift) < 0.1) return { offset: 0, how: 'already in sync' };
  if (s0 <= 0 || bestScore / s0 > 1.2) return { offset: shift, how: 'matched to the speech' };
  return { offset: 0, how: 'no clear match' };
}
