// Training audio: files, microphone, voices found in a video, and audio + captions.
import { PALETTE, SR } from './config.js';
import { concat, decodeRange, decodeTo16k, play, recordMic } from './audio.js';
import { alignCaptions, captionUtterances, parseCaptions } from './captions.js';
import { isolateVoice } from './separate.js';
import { clusterWindows, embed16k, ensureEmb, speechRegions } from './speakers.js';
import { $, el, friendlyError, log, setProgress, setStage, showError } from './ui.js';

export const clips = [];            // { name, pcm (16 kHz), utts?: [{ text, pcm }] }
const listeners = [];
export const onClipsChange = (fn) => listeners.push(fn);
function changed() { renderClips(); listeners.forEach((fn) => fn()); }
export const totalSeconds = () => clips.reduce((n, c) => n + c.pcm.length, 0) / SR;

function renderClips() {
  $('clips').replaceChildren(...clips.map((c) => el('div', { class: 'row' }, [
    el('span', { class: 'grow' }, c.name),
    el('span', { class: 'muted' }, `${(c.pcm.length / SR).toFixed(1)} s${c.utts ? ` · ${c.utts.length} lines` : ''}`),
    el('button', { onclick: () => play(c.pcm, SR) }, '▶'),
    el('button', { onclick: () => { clips.splice(clips.indexOf(c), 1); changed(); } }, '✕'),
  ])));
  const s = totalSeconds();
  $('clipsHint').textContent = clips.length ? `${Math.round(s)} s${s < 20 ? ' · at least 20 s needed' : ''}` : '';
}

export async function addFiles(files) {
  for (const f of files) {
    try { clips.push({ name: f.name, pcm: await decodeTo16k(f) }); } catch (e) { showError(e.message); }
  }
  changed();
}

let mic = null;
export async function toggleRecord() {
  if (mic) {
    const pcm = mic.stop(); mic = null;
    $('recordBtn').textContent = 'Record';
    clips.push({ name: `Mic ${clips.filter((c) => c.mic).length + 1}`, pcm, mic: true }); changed();
    return;
  }
  try { mic = await recordMic(); $('recordBtn').replaceChildren(el('span', { class: 'rec' }), 'Stop'); }
  catch (e) { showError(`Microphone unavailable: ${e.message}`); }
}

/** Find the different voices in the first minutes of a video; each can be added as a clip. */
export async function extractVoices(file) {
  showError(''); $('found').textContent = '';
  try {
    const seconds = Number($('scanMinutes').value) * 60;
    const voice16 = $('isolate').checked ? await isolateVoice(file, seconds, Number($('sepQuality').value)) : (await decodeRange(file, 0, seconds, SR, 1)).chans[0];
    setStage('Finding voices'); setProgress(null);
    await ensureEmb();
    const wins = [];
    for (const r of speechRegions(voice16)) {
      const len = r.end - r.start;
      if (len < 2) { if (len >= 0.8) wins.push({ start: r.start, end: r.end }); continue; }
      for (let t = r.start; t + 2 <= r.end + 0.01; t += 1) wins.push({ start: t, end: Math.min(r.end, t + 2) });
    }
    if (!wins.length) throw new Error('No speech found.');
    for (let i = 0; i < wins.length; i++) {
      if (i % 10 === 0) setProgress(i / wins.length, `voice prints ${i + 1}/${wins.length}`);
      wins[i].emb = await embed16k(voice16.slice(Math.floor(wins[i].start * SR), Math.floor(wins[i].end * SR)));
    }
    const voices = clusterWindows(wins).filter((v) => v.seconds >= 2);
    const seg = (w) => voice16.subarray(Math.floor(w.start * SR), Math.floor(w.end * SR));
    $('found').replaceChildren(...voices.map((v, i) => {
      const label = `Voice ${String.fromCharCode(65 + i)}`;
      // The windows that clearly match this voice, merged into segments (up to 5 minutes).
      const audio = () => {
        const segs = v.windows.filter((w) => w.sim >= 0.45).map((w) => [w.start, w.end]).sort((a, b) => a[0] - b[0]), merged = [];
        for (const s of segs) { const last = merged.at(-1); if (last && s[0] <= last[1] + 0.05) last[1] = Math.max(last[1], s[1]); else merged.push([...s]); }
        let total = 0; const parts = [];
        for (const [a, b] of merged) { if (total > 300) break; parts.push(voice16.subarray(Math.floor(a * SR), Math.floor(b * SR))); total += b - a; }
        return concat(parts);
      };
      return el('div', { class: 'row' }, [
        el('span', { class: 'grow', style: `color:${PALETTE[i % PALETTE.length]}` }, label),
        el('span', { class: 'muted' }, `${Math.round(v.seconds)} s`),
        el('button', { onclick: () => play(concat([...v.windows].sort((a, b) => b.sim - a.sim).slice(0, 3).sort((a, b) => a.start - b.start).map(seg)), SR) }, '▶'),
        el('button', { onclick: () => { clips.push({ name: `${file.name} · ${label}`, pcm: audio() }); changed(); } }, 'Add'),
      ]);
    }));
    setStage(`${voices.length} voice${voices.length === 1 ? '' : 's'} in ${file.name}`); setProgress(1);
  } catch (e) { showError(friendlyError(e)); setStage('Extraction failed'); setProgress(0); }
}

/** Audio (or video) + captions: caption lines become training sentences, grouped by speaker. */
export async function importCaptioned(files) {
  showError('');
  const cap = files.find((f) => /\.(vtt|srt|ttml|dfxp|xml)$/i.test(f.name)), media = files.find((f) => f !== cap && !/\.json$/i.test(f.name));
  if (!cap || !media) return showError('Choose the audio or video file and its captions together.');
  try {
    setStage(`Reading ${media.name}`); setProgress(null);
    const cues = parseCaptions(await cap.text());
    if (!cues.length) throw new Error(`No caption lines in ${cap.name}.`);
    const pcm = await decodeTo16k(media), manual = $('captionOffset').value.trim();
    const { offset, how } = manual !== '' && Number.isFinite(Number(manual)) ? { offset: Number(manual), how: 'manual' } : await alignCaptions(cues, pcm);
    if (offset) for (const c of cues) { c.start += offset; c.end += offset; }
    log(`${cap.name}: ${cues.length} lines, offset ${offset.toFixed(2)} s (${how}).`);
    const groups = new Map();
    for (const c of cues) { const k = c.speaker || 'All lines'; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(c); }
    $('found').replaceChildren(...[...groups].sort((a, b) => b[1].length - a[1].length).map(([who, list], i) => {
      const utts = captionUtterances(list, pcm), secs = utts.reduce((n, u) => n + u.pcm.length, 0) / SR;
      return el('div', { class: 'row' }, [
        el('span', { class: 'grow', style: `color:${PALETTE[i % PALETTE.length]}` }, who),
        el('span', { class: 'muted' }, `${utts.length} lines · ${Math.round(secs)} s`),
        el('button', { onclick: () => play(concat(utts.slice(0, 2).map((u) => u.pcm)), SR) }, '▶'),
        el('button', { disabled: !utts.length, onclick: () => { clips.push({ name: `${media.name} · ${who}`, pcm: concat(utts.map((u) => u.pcm)), utts }); changed(); } }, 'Add'),
      ]);
    }));
    setStage(`${cues.length} caption lines`); setProgress(0);
  } catch (e) { showError(friendlyError(e)); setStage('Import failed'); setProgress(0); }
}
