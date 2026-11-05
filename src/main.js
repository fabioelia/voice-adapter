// Wires the page: recordings → training → saved voices → speak.
import { ASR_MODELS } from './config.js';
import { play } from './audio.js';
import { forget, synth } from './kokoro.js';
import { addFiles, clips, extractVoices, importCaptioned, onClipsChange, toggleRecord, totalSeconds } from './recordings.js';
import { detectGPU, trainWorker } from './runtime.js';
import { renderVoiceChoices, saveSpeech, speak, stopSpeaking } from './speak.js';
import { deepTrain, previewVoice, quickMatch } from './train.js';
import { $, el, friendlyError, log, setProgress, setStage, showError } from './ui.js';
import { deleteVoice, exportVoice, importVoice, listVoices, saveVoice } from './voices.js';

let job = null, result = null, voices = [];

// ---- training ----
function drawChart(history = [], baseline) {
  const c = $('chart'), g = c.getContext('2d'), W = c.width, H = c.height, css = getComputedStyle(document.documentElement);
  g.clearRect(0, 0, W, H);
  if (!history.length) return;
  const vals = history.filter((x) => x != null), min = Math.min(...vals, baseline?.toTarget ?? 1) - 0.02, max = Math.max(...vals) + 0.02;
  const y = (v) => H - 10 - ((v - min) / (max - min || 1)) * (H - 20), x = (i) => 8 + (i / Math.max(1, history.length - 1)) * (W - 16);
  if (baseline?.toTarget != null) {
    g.strokeStyle = css.getPropertyValue('--muted'); g.setLineDash([6, 6]); g.beginPath();
    g.moveTo(0, y(baseline.toTarget)); g.lineTo(W, y(baseline.toTarget)); g.stroke(); g.setLineDash([]);
  }
  g.strokeStyle = css.getPropertyValue('--accent'); g.lineWidth = 3; g.beginPath();
  history.forEach((v, i) => (i ? g.lineTo(x(i), y(v)) : g.moveTo(x(i), y(v)))); g.stroke();
  g.fillStyle = css.getPropertyValue('--text'); g.font = '22px system-ui';
  g.fillText(`match ${vals.at(-1).toFixed(3)}`, 12, 28);
  $('trainStatus').textContent = baseline ? `closest stock voice ${baseline.name}: ${baseline.toTarget?.toFixed(3)} (dashed)` : '';
}
function sync() {
  const busy = !!job;
  $('trainBtn').disabled = busy || totalSeconds() < 20;
  $('stopBtn').disabled = !busy;
  $('listenBtn').disabled = !(job?.preview || result);
  $('saveBtn').disabled = busy || !result || !$('talent').value.trim() || !$('release').value.trim();
  for (const id of ['addBtn', 'recordBtn', 'videoBtn', 'captionBtn']) $(id).disabled = busy;
}
async function train() {
  if (job) return;
  job = { clips: clips.slice(), steps: Number($('steps').value), stop: false };
  result = null; showError(''); sync();
  try {
    result = await ($('method').value === 'neural' ? deepTrain : quickMatch)(job, { history: drawChart });
    setStage(`Done · match ${result.heldOut?.toFixed(3)} on an unseen sentence (start ${result.baseline.toTarget?.toFixed(3)})`); setProgress(1);
  } catch (e) { showError(friendlyError(e)); setStage('Training failed'); }
  finally { job = null; sync(); }
}
async function listen() {
  try { const d = await previewVoice(job?.preview || result); play(d.audio, d.sr, { rate: d.rate }); } catch (e) { showError(friendlyError(e)); }
}
async function save() {
  try {
    const v = await saveVoice(result, { name: $('talent').value.trim(), release: $('release').value.trim(), clips: { count: clips.length, seconds: Math.round(totalSeconds()) } });
    log(`Saved voice "${v.name}".`); await refreshVoices();
  } catch (e) { showError(e.message); }
}

// ---- saved voices ----
async function refreshVoices() {
  try { voices = await listVoices(); } catch (e) { voices = []; log(`Saved voices unavailable: ${e.message}`); }
  $('voices').replaceChildren(...(voices.length ? voices.map((v) => el('div', { class: 'row' }, [
    el('span', { class: 'grow' }, v.name),
    el('span', { class: 'muted' }, `${v.kind === 'neural' ? 'deep' : 'quick'} · match ${v.match?.toFixed(2) ?? '?'} · ${v.training?.seconds ?? '?'} s · release ${v.release}`),
    el('button', { onclick: async () => { try { const d = await synth(v, 'Hi, this is my trained voice.'); play(d.audio, d.sr, { rate: d.rate }); } catch (e) { showError(friendlyError(e)); } } }, '▶'),
    el('button', { onclick: () => exportVoice(v) }, 'Export'),
    el('button', { onclick: async () => { if (!confirm(`Delete "${v.name}"?`)) return; await deleteVoice(v.id); forget(v.id); refreshVoices(); } }, '✕'),
  ])) : [el('div', { class: 'muted' }, 'No saved voices.')]));
  renderVoiceChoices(voices);
}

// ---- settings ----
const SETTINGS = 'voice-adapter.settings', FIELDS = ['asrModel', 'device', 'sepQuality', 'method', 'steps'];
function loadSettings() { try { const s = JSON.parse(localStorage.getItem(SETTINGS)) || {}; for (const k of FIELDS) if (s[k] != null) $(k).value = s[k]; } catch (e) { /* defaults */ } }
function saveSettings() { try { localStorage.setItem(SETTINGS, JSON.stringify(Object.fromEntries(FIELDS.map((k) => [k, $(k).value])))); } catch (e) { /* not kept */ } }

function picker(input, fn) { $(input).addEventListener('change', () => { const f = [...$(input).files]; $(input).value = ''; if (f.length) fn(f); }); }

async function init() {
  for (const m of ASR_MODELS) $('asrModel').append(el('option', { value: m.id }, m.label));
  loadSettings();
  for (const k of FIELDS) $(k).addEventListener('change', saveSettings);
  $('stepsVal').textContent = $('steps').value;
  $('steps').addEventListener('input', () => { $('stepsVal').textContent = $('steps').value; });
  $('gpu').textContent = (await detectGPU()) ? 'WebGPU' : 'no WebGPU (CPU)';

  $('addBtn').addEventListener('click', () => $('addInput').click()); picker('addInput', addFiles);
  $('videoBtn').addEventListener('click', () => $('videoInput').click()); picker('videoInput', (f) => extractVoices(f[0]));
  $('captionBtn').addEventListener('click', () => $('captionInput').click()); picker('captionInput', importCaptioned);
  $('recordBtn').addEventListener('click', toggleRecord);
  onClipsChange(sync);

  $('trainBtn').addEventListener('click', train);
  $('stopBtn').addEventListener('click', () => { if (job) { job.stop = true; trainWorker.call('stop').catch(() => {}); } });
  $('listenBtn').addEventListener('click', listen);
  $('saveBtn').addEventListener('click', save);
  for (const id of ['talent', 'release']) $(id).addEventListener('input', sync);

  $('importBtn').addEventListener('click', () => $('importInput').click());
  picker('importInput', async ([f]) => { try { const v = await importVoice(f); log(`Imported "${v.name}".`); refreshVoices(); } catch (e) { showError(`Import failed: ${e.message}`); } });

  for (const k of ['speakPitch', 'speakPace']) $(k).addEventListener('input', () => { $(`${k}Val`).textContent = Number($(k).value).toFixed(2); });
  $('speakBtn').addEventListener('click', () => speak(voices));
  $('speakStop').addEventListener('click', stopSpeaking);
  $('speakSave').addEventListener('click', saveSpeech);

  sync(); refreshVoices();
  window.__voice = { clips, get result() { return result; }, get job() { return job; }, refreshVoices };
}
init();
