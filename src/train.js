// Training a Kokoro voice from a talent's recordings.
//  quickMatch: searches Kokoro's voice-style space (a (1+1) evolution strategy scored by voice print).
//  deepTrain:  gradient training of the decoder's style layers (AdaSpeech-style: the 58 AdaIN biases
//              and a style offset) with the exported forward+backward graph; Adam runs in the worker.
import { KOKORO_SR, KOKORO_VOICES, PROBES, SR, VOICE_LEN } from './config.js';
import { medianPitch, resample, speechOnly } from './audio.js';
import { ensureAsr, ensureKokoro, ensureTrainModels, registerVoice, stockTable, styleRow, synth, tokenize } from './kokoro.js';
import { asrWorker, trainWorker, ttsWorker } from './runtime.js';
import { cosine, embed16k, ensureEmb, voicePrint } from './speakers.js';
import { log, setProgress, setStage } from './ui.js';

const median = (a) => { if (!a.length) return 0; const s = Float32Array.from(a).sort(); return s[s.length >> 1]; };
/** Stock voices on the talent's side of the pitch range. */
const candidates = (hz, list = KOKORO_VOICES) => list.filter((v) => hz == null || (hz > 170 ? v[1] === 'f' : hz < 140 ? v[1] === 'm' : true));
const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

/**
 * @param job  { clips: [{ pcm, utts? }], steps, stop: false } — set job.stop = true to stop.
 * @param on   { history(values), status(text) } progress callbacks.
 * @returns    { data, heldOut, baseline, steps }
 */
export async function quickMatch(job, on) {
  setStage('Preparing'); setProgress(null);
  await ensureKokoro(); await ensureEmb();
  const target = await voicePrint(job.clips.map((c) => c.pcm));
  const tables = new Map();
  for (const n of KOKORO_VOICES) tables.set(n, await stockTable(n));
  // How much a candidate sounds like the talent over the probe sentences, plus a little consistency
  // across sentences (broken voices drift).
  const score = async (data, probes) => {
    await registerVoice('probe', data);
    const prints = [];
    for (const text of probes) {
      const r = await ttsWorker.call('synth', { text, voice: 'probe', speed: 1 });
      let peak = 0; for (let i = 0; i < r.audio.length; i += 7) peak = Math.max(peak, Math.abs(r.audio[i]));
      const secs = r.audio.length / r.sr;
      if (!(peak > 0.02) || secs > text.length * 0.2 || secs < text.length * 0.025) return { score: -1 };
      prints.push(await embed16k(resample(r.audio, r.sr, SR)));
    }
    const toTarget = prints.reduce((s, p) => s + cosine(p, target), 0) / prints.length;
    let self = 1, n = 0;
    if (prints.length > 1) { self = 0; for (let i = 0; i < prints.length; i++) for (let j = i + 1; j < prints.length; j++) { self += cosine(prints[i], prints[j]); n++; } self /= n; }
    return { score: 0.85 * toTarget + 0.15 * self, toTarget };
  };
  // Mutation scale per element: the spread of the stock voices.
  const all = [...tables.values()], spread = new Float32Array(VOICE_LEN), lo = new Float32Array(VOICE_LEN), hi = new Float32Array(VOICE_LEN);
  for (let i = 0; i < VOICE_LEN; i++) {
    let m = 0, mn = Infinity, mx = -Infinity; for (const v of all) { m += v[i]; mn = Math.min(mn, v[i]); mx = Math.max(mx, v[i]); }
    m /= all.length; let q = 0; for (const v of all) q += (v[i] - m) ** 2;
    spread[i] = Math.sqrt(q / all.length); lo[i] = mn - spread[i]; hi[i] = mx + spread[i];
  }
  const probes = PROBES.slice(0, 2), names = candidates(medianPitch(speechOnly(job.clips[0].pcm)), KOKORO_VOICES.filter((v) => /^[ab]/.test(v)));
  setStage('Comparing stock voices');
  const ranked = [];
  for (let i = 0; i < names.length && !job.stop; i++) { setProgress(i / names.length, names[i]); ranked.push({ name: names[i], ...(await score(tables.get(names[i]), probes)) }); }
  ranked.sort((a, b) => b.score - a.score);
  const baseline = ranked[0];
  log(`Closest stock voices: ${ranked.slice(0, 3).map((r) => `${r.name} ${r.toTarget?.toFixed(3)}`).join(', ')}`);
  // Start from a similarity-weighted blend of the three closest.
  const top = ranked.slice(0, 3).filter((r) => r.score > -1), w = top.map((r) => Math.exp((r.score - top[0].score) * 20)), ws = w.reduce((a, b) => a + b, 0);
  let best = new Float32Array(VOICE_LEN);
  top.forEach((r, k) => { const v = tables.get(r.name); for (let i = 0; i < VOICE_LEN; i++) best[i] += (v[i] * w[k]) / ws; });
  let bestScore = await score(best, probes);
  if (bestScore.score < baseline.score) { best = tables.get(baseline.name).slice(); bestScore = baseline; }
  const history = [bestScore.toTarget]; on.history(history, baseline);
  // Each step moves every style row by the same 256-dim offset, so the whole voice shifts coherently.
  const colSpread = new Float32Array(256);
  for (let i = 0; i < VOICE_LEN; i++) colSpread[i % 256] += spread[i] / 510;
  let sigma = 0.25, fails = 0, step = 0;
  const t0 = performance.now();
  setStage('Training');
  for (step = 1; step <= job.steps && !job.stop; step++) {
    const dir = new Float32Array(256), cand = new Float32Array(VOICE_LEN);
    for (let c = 0; c < 256; c++) dir[c] = sigma * colSpread[c] * gauss();
    for (let i = 0; i < VOICE_LEN; i++) cand[i] = Math.min(hi[i], Math.max(lo[i], best[i] + dir[i % 256]));
    const sc = await score(cand, probes);
    if (sc.score > bestScore.score) { best = cand; bestScore = sc; sigma = Math.min(0.8, sigma * 1.3); fails = 0; }
    else if (++fails >= 6) { sigma = Math.max(0.02, sigma * 0.8); fails = 0; }
    history.push(bestScore.toTarget); job.preview = best;
    const rate = step / ((performance.now() - t0) / 1000);
    setProgress(step / job.steps, `step ${step}/${job.steps} · ${(1 / rate).toFixed(1)} s/step`);
    if (step % 2 === 0) on.history(history, baseline);
  }
  on.history(history, baseline);
  const held = await score(best, [PROBES[2]]);   // a sentence the search never used
  return { kind: 'search', data: best, heldOut: held.toTarget, baseline, steps: step - 1 };
}

/** Whisper segments -> sentences of 1.5–12 s with their audio. */
function utterancesFrom(items, pcm) {
  const out = []; let cur = null;
  for (const it of items) {
    const text = (it.text || '').trim(); if (!text) continue;
    const start = Number(it.start) || 0, end = Number(it.end ?? start + 2);
    if (cur && start - cur.end < 0.6 && end - cur.start <= 12) { cur.text += ' ' + text; cur.end = end; }
    else { cur = { start, end, text }; out.push(cur); }
  }
  return out.filter((u) => u.end - u.start >= 1.5)
    .map((u) => ({ text: u.text, pcm: pcm.slice(Math.max(0, Math.floor((u.start - 0.1) * SR)), Math.min(pcm.length, Math.ceil((u.end + 0.15) * SR))) }));
}

export async function deepTrain(job, on) {
  setStage('Loading training models'); setProgress(null);
  await ensureKokoro(); await ensureEmb(); await ensureAsr();
  await ensureTrainModels(['frontend', 'prosody', 'decoder', 'train_step']);
  const target = await voicePrint(job.clips.map((c) => c.pcm));
  // 1. Starting voice: the closest stock voice on the talent's side of the pitch range.
  setStage('Choosing a starting voice');
  const names = candidates(medianPitch(speechOnly(job.clips[0].pcm)));
  let base = null;
  for (let i = 0; i < names.length && !job.stop; i++) {
    setProgress(i / names.length, names[i]);
    const r = await ttsWorker.call('synth', { text: PROBES[0], voice: names[i], speed: 1 });
    const sim = cosine(await embed16k(resample(r.audio, r.sr, SR)), target);
    if (!base || sim > base.sim) base = { name: names[i], sim };
  }
  const table = await stockTable(base.name);
  log(`Starting from ${base.name} (similarity ${base.sim.toFixed(3)}).`);
  // 2. Sentences (captioned clips bring their own; others are transcribed), aligned with real pitch and energy.
  await trainWorker.call('reset');
  const utts = [];
  for (let i = 0; i < job.clips.length && !job.stop; i++) {
    if (job.clips[i].utts) { utts.push(...job.clips[i].utts); continue; }
    setStage(`Transcribing recording ${i + 1} of ${job.clips.length}`); setProgress(0);
    const items = await asrWorker.call('transcribeFile', { audio: job.clips[i].pcm.slice(), opts: { language: 'en', task: 'transcribe', multilingual: true } });
    utts.push(...utterancesFrom(items, job.clips[i].pcm));
  }
  const f0Real = [], f0Pred = []; let realSpeech = 0, predSpeech = 0, used = 0;
  for (let i = 0; i < utts.length && !job.stop; i++) {
    setStage('Aligning sentences'); setProgress(i / utts.length, `${i + 1} of ${utts.length}`);
    const ids = await tokenize(utts[i].text, base.name);
    if (ids.length < 3) continue;
    const r = await trainWorker.call('prep', { ids, style: styleRow(table, ids.length), audio: resample(utts[i].pcm, SR, KOKORO_SR) });
    if (r.skipped) continue;
    used++; f0Real.push(...r.f0Real); f0Pred.push(...r.f0Pred); realSpeech += r.realSpeech; predSpeech += r.predSpeech;
  }
  if (!used) throw new Error('No usable sentences: deep training needs clear English speech, a few seconds per sentence.');
  const f0Ratio = Math.max(0.6, Math.min(1.6, median(f0Pred) ? median(f0Real) / median(f0Pred) : 1));
  const speed = Math.max(0.75, Math.min(1.3, realSpeech ? predSpeech / realSpeech : 1));
  log(`${used} sentences. Pitch ×${f0Ratio.toFixed(2)}, pace ×${speed.toFixed(2)} relative to ${base.name}.`);
  // 3. Train; score a sentence not in the training audio every 50 steps and keep the best checkpoint
  //    (with a few minutes of audio, similarity on unseen text peaks and then falls).
  const probe = await tokenize(PROBES[1], base.name), probeStyle = styleRow(table, probe.length);
  const evalSim = async (useTrained) => {
    const r = await trainWorker.call('render', { ids: probe, style: probeStyle, speed, f0Ratio, useTrained });
    return cosine(await embed16k(resample(r.audio, KOKORO_SR, SR)), target);
  };
  const baseline = { name: base.name, toTarget: await evalSim(false) };
  const history = [baseline.toTarget]; on.history(history, baseline);
  let done = 0, state = null, best = null;
  setStage('Training');
  while (done < job.steps && !job.stop) {
    state = await trainWorker.call('train', { steps: Math.min(25, job.steps - done), resume: done > 0 }, { onProgress: (m) => {
      setProgress((done + m.step) / job.steps, `step ${done + m.step}/${job.steps} · loss ${m.loss.toFixed(3)} · ${m.secPerStep.toFixed(2)} s/step`);
    } });
    done = state.steps;
    job.preview = { kind: 'neural', base: base.name, styleDelta: state.style, biasDelta: state.bias, f0Ratio, speed };
    if (done % 50 === 0 || done >= job.steps || job.stop) {
      const sim = await evalSim(true);
      history.push(sim); on.history(history, baseline);
      if (!best || sim > best.sim) best = { sim, style: Float32Array.from(state.style), bias: Float32Array.from(state.bias), steps: done };
    }
  }
  if (!state) throw new Error('Stopped before the first step.');
  if (best) log(`Keeping the checkpoint from step ${best.steps} (match ${best.sim.toFixed(3)}).`);
  const style = best ? best.style : state.style, bias = best ? best.bias : state.bias;
  const unseen = await tokenize(PROBES[2], base.name);
  const r = await trainWorker.call('render', { ids: unseen, style: styleRow(table, unseen.length), speed, f0Ratio, styleDelta: style, biasDelta: bias });
  const heldOut = cosine(await embed16k(resample(r.audio, KOKORO_SR, SR)), target);
  return { kind: 'neural', base: base.name, styleDelta: style, biasDelta: bias, f0Ratio, speed, heldOut, baseline, steps: best ? best.steps : done };
}

/** Speak a preview sentence with a result (or a job's latest state). */
export async function previewVoice(v) {
  const voice = v.kind === 'neural' ? v : { id: 'preview', kind: 'search', data: v.data || v };
  if (voice.kind === 'search') await registerVoice('preview', voice.data);
  return synth(voice.kind === 'search' ? 'preview' : voice, 'Hello! This is how I sound after training. What do you think?');
}
