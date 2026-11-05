const SR = 16000;
let TF = null, asr = null;
const post = (m, t) => self.postMessage(m, t || []);

async function lib(url) {
  if (TF) return TF;
  TF = await import(url);
  TF.env.allowLocalModels = false;
  try { if (typeof caches === 'undefined') TF.env.useBrowserCache = false; else await caches.open('transformers-cache'); }
  catch (e) { TF.env.useBrowserCache = false; }
  return TF;
}

async function loadAsr(id, cfg) {
  const T = await lib(cfg.libUrl);
  const key = JSON.stringify([cfg.model, cfg.device]);
  if (asr && asr.key === key) return asr.device;
  if (asr) { try { await asr.pipe.dispose(); } catch (e) {} asr = null; }
  const progress_callback = (p) => post({ id, type: 'download', ...p });
  const tries = cfg.device === 'webgpu'
    ? [['webgpu', cfg.big ? { encoder_model: 'fp32', decoder_model_merged: 'q4' } : 'fp32'], ['wasm', 'q8']]
    : [['wasm', 'q8']];
  let err;
  for (const [device, dtype] of tries) {
    try {
      post({ id, type: 'status', message: `Loading ${cfg.model} (${device})` });
      const pipe = await T.pipeline('automatic-speech-recognition', cfg.model, { device, dtype, progress_callback });
      asr = { key, pipe, device };
      return device;
    } catch (e) { err = e; post({ id, type: 'log', message: `Loading on ${device} failed: ${e.message || e}` }); }
  }
  throw err;
}

function genOpts(opts) {
  const g = {};
  if (opts.multilingual) {
    if (opts.language && opts.language !== 'auto') g.language = opts.language;
    g.task = opts.task || 'transcribe';
  }
  return g;
}

let emb = null;
async function loadEmb(id, cfg) {
  const T = await lib(cfg.libUrl);
  if (emb) return emb.device;
  post({ id, type: 'status', message: 'Loading voice-matching model' });
  const processor = await T.AutoProcessor.from_pretrained(cfg.model, { progress_callback: (p) => post({ id, type: 'download', ...p }) });
  let model, device = 'wasm';
  for (const d of cfg.device === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']) {
    try { model = await T.AutoModel.from_pretrained(cfg.model, { device: d, dtype: 'fp32' }); device = d; break; }
    catch (e) { post({ id, type: 'log', message: `Voice-matching model on ${d} failed: ${e.message || e}` }); }
  }
  if (!model) throw new Error('Could not load the voice-matching model.');
  emb = { processor, model, device };
  return device;
}
// One unit-length speaker embedding (WeSpeaker ResNet34) for 16 kHz mono audio.
async function embed(id, { audio }) {
  let a = audio;
  if (a.length < SR) { const p = new Float32Array(SR); p.set(a); a = p; }
  const inputs = await emb.processor(a);
  const out = await emb.model(inputs);
  const t = out.embeddings || out.last_hidden_state || out[Object.keys(out)[0]];
  let data = t.data;
  if (t.dims.length === 3) {
    const [, n, d] = t.dims, m = new Float32Array(d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) m[j] += data[i * d + j] / n;
    data = m;
  }
  let s = 0; for (const x of data) s += x * x; s = Math.sqrt(s) || 1;
  return Float32Array.from(data, (x) => x / s);
}

async function transcribeFile(id, { audio, opts }) {
  const pipe = asr.pipe;
  const chunk_length_s = 30, stride_length_s = 5;
  const win = chunk_length_s * SR, jump = win - 2 * stride_length_s * SR;
  const total = audio.length > win ? 1 + Math.ceil((audio.length - win) / jump) : 1;
  let done = 0;
  const orig = pipe.model.generate;
  pipe.model.generate = async function (...a) {
    post({ id, type: 'progress', value: done / total, message: `Listening: part ${done + 1} of ${total}` });
    const r = await orig.apply(this, a); done++; return r;
  };
  try {
    const out = await pipe(audio, { chunk_length_s, stride_length_s, return_timestamps: true, ...genOpts(opts) });
    return (out.chunks || []).map((c) => ({ text: c.text, start: c.timestamp?.[0] ?? 0, end: c.timestamp?.[1] }));
  } finally { pipe.model.generate = orig; }
}

async function transcribeClip(id, { audio, opts }) {
  const out = await asr.pipe(audio, genOpts(opts));
  return (out.text || '').trim();
}

self.onmessage = async (ev) => {
  const { id, type, payload } = ev.data;
  try {
    let result;
    if (type === 'loadAsr') result = await loadAsr(id, payload);
    else if (type === 'transcribeFile') result = await transcribeFile(id, payload);
    else if (type === 'transcribeClip') result = await transcribeClip(id, payload);
    else if (type === 'loadEmb') result = await loadEmb(id, payload);
    else if (type === 'embed') result = await embed(id, payload);
    else throw new Error('Unknown message ' + type);
    post({ id, type: 'done', result });
  } catch (e) { post({ id, type: 'error', message: (e && e.message) || String(e) }); }
};
