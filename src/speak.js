// Speak typed text in a stock or trained voice; download the result as WAV.
import { KOKORO_SR, KOKORO_VOICES } from './config.js';
import { audioContext, concat, play, resample, stopAll, wavBlob } from './audio.js';
import { synth } from './kokoro.js';
import { $, download, el, friendlyError, showError } from './ui.js';

let stopped = false, last = null;

export function renderVoiceChoices(trained) {
  const cur = $('speakVoice').value || 'af_heart';
  $('speakVoice').replaceChildren(
    ...(trained.length ? [el('optgroup', { label: 'Trained' }, trained.map((v) => el('option', { value: v.id }, v.name)))] : []),
    el('optgroup', { label: 'Stock' }, KOKORO_VOICES.map((v) => el('option', { value: v }, v))));
  $('speakVoice').value = [...$('speakVoice').options].some((o) => o.value === cur) ? cur : 'af_heart';
}

/** Sentences of up to ~220 characters. */
function sentences(text) {
  const out = [];
  for (const para of text.split(/\n+/)) {
    let buf = '';
    for (const part of para.split(/(?<=[.!?…])\s+/)) { if (buf && (buf + ' ' + part).length > 220) { out.push(buf); buf = part; } else buf = buf ? `${buf} ${part}` : part; }
    if (buf.trim()) out.push(buf.trim());
  }
  return out;
}

export async function speak(trained) {
  const text = $('speakText').value.trim(); if (!text) return;
  stopped = false; last = null;
  $('speakBtn').disabled = true; $('speakStop').disabled = false; $('speakSave').disabled = true;
  const id = $('speakVoice').value, voice = trained.find((v) => v.id === id) || id;
  const opts = { pitch: Number($('speakPitch').value), pace: Number($('speakPace').value) };
  try {
    const ctx = audioContext(), parts = [], list = sentences(text);
    let when = ctx.currentTime + 0.1;
    for (let i = 0; i < list.length && !stopped; i++) {
      $('speakStatus').textContent = `${i + 1} of ${list.length}`;
      const d = await synth(voice, list[i], opts);
      if (stopped) break;
      when = Math.max(when, ctx.currentTime + 0.02);
      play(d.audio, d.sr, { when, rate: d.rate });
      when += d.dur + 0.25;
      parts.push(resample(d.audio, Math.round(d.sr * d.rate), d.sr), new Float32Array(Math.round(0.25 * d.sr)));
    }
    if (parts.length) { last = concat(parts); $('speakStatus').textContent = `${(last.length / KOKORO_SR).toFixed(1)} s`; }
  } catch (e) { showError(friendlyError(e)); $('speakStatus').textContent = ''; }
  finally { $('speakBtn').disabled = false; $('speakStop').disabled = true; $('speakSave').disabled = !last; }
}
export function stopSpeaking() { stopped = true; stopAll(); }
export function saveSpeech() { if (last) download('speech.wav', wavBlob(last, KOKORO_SR)); }
