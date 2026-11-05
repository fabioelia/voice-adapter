// Dialogue separation with TIGER-DnR (Xu, Li, Chen, Hu, ICLR 2025; Apache-2.0 weights, ONNX by audiojs).
// The graph maps the STFT of a 12 s mono 44.1 kHz segment to the STFTs of three stems: dialogue, music,
// effects. Only the dialogue stem is used: the music + effects bed is the mixture minus dialogue, so
// every sound the model does not call dialogue (including ones it is unsure about) stays in the bed.
const N = 2048, HOP = 512, BINS = N / 2 + 1, SR = 44100, SEG = 12 * SR, FRAMES = 1 + Math.floor(SEG / HOP);
let ort = null, sess = null;
const post = (m, t) => self.postMessage(m, t || []);

// ---- radix-2 FFT (in place) ----
const rev = new Uint32Array(N), cosT = new Float64Array(N / 2), sinT = new Float64Array(N / 2), win = new Float64Array(N);
(() => {
  const bits = Math.log2(N);
  for (let i = 0; i < N; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
  for (let i = 0; i < N / 2; i++) { cosT[i] = Math.cos(2 * Math.PI * i / N); sinT[i] = Math.sin(2 * Math.PI * i / N); }
  for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N); // periodic Hann (torch.hann_window)
})();
function fft(re, im, inverse) {
  for (let i = 0; i < N; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let size = 2; size <= N; size <<= 1) {
    const half = size >> 1, step = N / size;
    for (let s = 0; s < N; s += size) for (let k = 0; k < half; k++) {
      const c = cosT[k * step], sn = inverse ? sinT[k * step] : -sinT[k * step];
      const a = s + k, b = a + half, tr = re[b] * c - im[b] * sn, ti = re[b] * sn + im[b] * c;
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
}
// Centered STFT with reflect padding, written straight into the model's [2, BINS, FRAMES] input layout.
function stftInto(x, out) {
  const pad = N / 2, len = x.length, re = new Float64Array(N), im = new Float64Array(N);
  const at = (i) => { if (i < 0) i = -i; if (i >= len) i = 2 * len - 2 - i; return x[Math.max(0, Math.min(len - 1, i))]; };
  for (let t = 0; t < FRAMES; t++) {
    const o = t * HOP - pad;
    for (let i = 0; i < N; i++) { re[i] = at(o + i) * win[i]; im[i] = 0; }
    fft(re, im, false);
    for (let f = 0; f < BINS; f++) { out[f * FRAMES + t] = re[f]; out[(BINS + f) * FRAMES + t] = im[f]; }
  }
}
// Inverse of stem 0 (dialogue) of the [6, BINS, FRAMES] output, overlap-added and window-normalised.
function istftStem(y, stem, len) {
  const pad = N / 2, out = new Float64Array(len + N), norm = new Float64Array(len + N), re = new Float64Array(N), im = new Float64Array(N);
  const R = 2 * stem * BINS * FRAMES, I = (2 * stem + 1) * BINS * FRAMES;
  for (let t = 0; t < FRAMES; t++) {
    for (let f = 0; f < BINS; f++) { re[f] = y[R + f * FRAMES + t]; im[f] = y[I + f * FRAMES + t]; }
    for (let f = 1; f < N / 2; f++) { re[N - f] = re[f]; im[N - f] = -im[f]; }
    fft(re, im, true);
    const o = t * HOP;
    for (let i = 0; i < N; i++) { out[o + i] += re[i] / N * win[i]; norm[o + i] += win[i] * win[i]; }
  }
  const s = new Float32Array(len);
  for (let i = 0; i < len; i++) { const n = norm[i + pad]; s[i] = n > 1e-8 ? out[i + pad] / n : 0; }
  return s;
}

async function load(id, p) {
  if (sess) return sess.device;
  ort = await import(p.ortUrl);
  ort.env.wasm.wasmPaths = p.wasmBase;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  let bytes;
  try { const c = await caches.open('voice-adapter-models'); const hit = await c.match(p.modelUrl); if (hit) bytes = new Uint8Array(await hit.arrayBuffer()); } catch (e) {}
  if (!bytes) {
    post({ id, type: 'status', message: 'Downloading the dialogue separation model' });
    const r = await fetch(p.modelUrl);
    if (!r.ok) throw new Error(`Could not download the separation model (${r.status}).`);
    const total = Number(r.headers.get('content-length')) || 0, reader = r.body.getReader(), parts = [];
    let loaded = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); loaded += value.length; post({ id, type: 'download', status: 'progress', name: 'separator', file: p.modelUrl.split('/').pop(), loaded, total }); }
    bytes = new Uint8Array(loaded); let o = 0; for (const v of parts) { bytes.set(v, o); o += v.length; }
    post({ id, type: 'download', status: 'done', name: 'separator', file: p.modelUrl.split('/').pop() });
    try { const c = await caches.open('voice-adapter-models'); await c.put(p.modelUrl, new Response(bytes)); } catch (e) {}
  }
  let err;
  for (const ep of p.device === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']) {
    try { sess = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'all' }); sess.device = ep; return ep; }
    catch (e) { err = e; post({ id, type: 'log', message: `Separator on ${ep} failed: ${e.message || e}` }); }
  }
  throw err;
}

// Dialogue of a mono 44.1 kHz signal: 12 s segments centred every `hop` seconds, cross-faded with Hann
// weights (hop 4 = each sample seen 3 times, as the reference inference does; 6 = twice; 9 = fastest).
async function dialogue(id, { mono, hop = 6 }) {
  const n = mono.length, H = Math.round(hop * SR), out = new Float32Array(n), wsum = new Float32Array(n);
  const w = Float32Array.from({ length: SEG }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * (i + 0.5) / SEG) + 1e-3);
  const seg = new Float32Array(SEG), x = new Float32Array(2 * BINS * FRAMES);
  const centres = []; for (let c = 0; c < n + H; c += H) centres.push(c);
  for (let k = 0; k < centres.length; k++) {
    const st = centres[k] - SEG / 2, a = Math.max(0, st), b = Math.min(n, st + SEG);
    if (b <= a) continue;
    seg.fill(0); seg.set(mono.subarray(a, b), a - st);
    stftInto(seg, x);
    const res = await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', x, [1, 2, BINS, FRAMES]) });
    const d = istftStem(res[sess.outputNames[0]].data, 0, SEG);
    for (let i = a; i < b; i++) { out[i] += d[i - st] * w[i - st]; wsum[i] += w[i - st]; }
    post({ id, type: 'progress', value: (k + 1) / centres.length });
  }
  for (let i = 0; i < n; i++) out[i] /= wsum[i] || 1;
  return out;
}

self.onmessage = async (ev) => {
  const { id, type, payload } = ev.data;
  try {
    let result, transfer;
    if (type === 'load') result = await load(id, payload);
    else if (type === 'dialogue') { result = await dialogue(id, payload); transfer = [result.buffer]; }
    else throw new Error('Unknown message ' + type);
    post({ id, type: 'done', result }, transfer);
  } catch (e) { post({ id, type: 'error', message: (e && e.message) || String(e) }); }
};
