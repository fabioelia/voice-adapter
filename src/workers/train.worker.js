// Runs the split Kokoro graphs (frontend, prosody, decoder) and the exported training step
// (forward + gradients) with ONNX Runtime Web. Adam runs here in JS on the two trainable deltas.
let ort = null, manifest = null, base = '', P = 0, W = 48;
const sess = {}, shared = {};
const dataset = [];
let trainState = null, stopFlag = false;
const post = (m, t) => self.postMessage(m, t || []);

// ---------- model loading ----------
async function cachedBytes(url, label) {
  let c = null;
  try { c = await caches.open('voice-adapter-models'); const hit = await c.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); } catch (e) {}
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Could not download ${url.split('/').pop()} (${r.status}). Is the models/ folder next to the page?`);
  const total = Number(r.headers.get('content-length')) || 0, reader = r.body.getReader(), parts = [];
  let loaded = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); loaded += value.length; post({ type: 'download', status: 'progress', name: 'voice-model', file: label, loaded, total }); }
  post({ type: 'download', status: 'done', name: 'voice-model', file: label });
  const out = new Uint8Array(loaded); let o = 0; for (const v of parts) { out.set(v, o); o += v.length; }
  if (c) try { await c.put(url, new Response(out)); } catch (e) {}
  return out;
}
async function joined(names) {
  const bufs = [];
  // ?v=<build> so updated model files never come from a stale cache entry
  for (const n of names) bufs.push(await cachedBytes(new URL(n, base).href + '?v=' + (manifest.build || '0'), n));
  if (bufs.length === 1) return bufs[0];
  const out = new Uint8Array(bufs.reduce((s, b) => s + b.length, 0)); let o = 0; for (const b of bufs) { out.set(b, o); o += b.length; }
  return out;
}
async function load(p) {
  if (!ort) {
    ort = await import(p.ortUrl);
    ort.env.wasm.wasmPaths = p.wasmBase;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  }
  base = p.base;
  if (!manifest) {
    manifest = await (await fetch(new URL('manifest.json', base), { cache: 'no-cache' })).json();
    P = manifest.train.params; W = manifest.train.crop_frames;
  }
  const devices = {};
  for (const name of p.graphs) {
    if (sess[name]) { devices[name] = sess[name].ep; continue; }
    const g = manifest.graphs[name];
    post({ type: 'status', message: `Loading voice model: ${name}` });
    const model = await joined(g.parts);
    const externalData = [];
    for (const ext of g.external) {
      if (!shared[ext]) shared[ext] = await joined(manifest.external[ext]);
      externalData.push({ path: ext, data: shared[ext] });
    }
    // LSTMs (frontend, prosody) are CPU-only in ORT Web; the decoder and training step use WebGPU.
    const eps = (name === 'decoder' || name === 'train_step') && p.device === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
    sources[name] = { model, externalData };
    await createSession(name, eps);
    devices[name] = sess[name].ep;
  }
  return { devices, params: P, crop: W };
}
const sources = {};
async function createSession(name, eps) {
  const { model, externalData } = sources[name];
  let err;
  for (const ep of eps) {
    try { sess[name] = await ort.InferenceSession.create(model, { executionProviders: [ep], externalData, graphOptimizationLevel: 'all' }); sess[name].ep = ep; return; }
    catch (e) { err = e; post({ type: 'log', message: `Voice model ${name} on ${ep} failed: ${e.message || e}` }); }
  }
  throw err;
}
// Run a graph; if WebGPU fails at run time (device limits, driver issues), rebuild it on the CPU and retry.
async function run(name, feeds) {
  try { return await sess[name].run(feeds); }
  catch (e) {
    if (sess[name].ep !== 'webgpu') throw e;
    post({ type: 'log', message: `Voice model ${name} failed on WebGPU (${(e.message || e).toString().slice(0, 160)}); continuing on the CPU.` });
    try { await sess[name].release(); } catch (_) {}
    await createSession(name, ['wasm']);
    return sess[name].run(feeds);
  }
}
const T32 = (data, dims) => new ort.Tensor('float32', data, dims);

// ---------- Kokoro inference on the split graphs ----------
async function frontend(ids, style, speed) {
  const L = ids.length + 2, input = new BigInt64Array(L);
  ids.forEach((v, i) => { input[i + 1] = BigInt(v); });
  const o = await run('frontend', { input_ids: new ort.Tensor('int64', input, [1, L]), style: T32(style, [1, 256]), speed: T32(new Float32Array([speed]), [1]) });
  const dur = o.dur.data, pd = new Int32Array(L);
  for (let i = 0; i < L; i++) pd[i] = Math.max(1, Math.round(dur[i]));
  return { L, pd, t_en: o.t_en.data, d: o.d.data, dDim: o.d.dims[2] };
}
// Per-frame token index from integer durations.
function frameTokens(pd) { const T = pd.reduce((a, b) => a + b, 0), idx = new Int32Array(T); let t = 0; pd.forEach((n, i) => { for (let k = 0; k < n; k++) idx[t++] = i; }); return idx; }
function gatherCols(src, C, L, idx) { // src [C, L] row-major -> [C, T]
  const T = idx.length, out = new Float32Array(C * T);
  for (let c = 0; c < C; c++) { const so = c * L, oo = c * T; for (let t = 0; t < T; t++) out[oo + t] = src[so + idx[t]]; }
  return out;
}
function gatherRowsT(src, L, D, idx) { // src [L, D] -> [D, T]
  const T = idx.length, out = new Float32Array(D * T);
  for (let t = 0; t < T; t++) { const so = idx[t] * D; for (let k = 0; k < D; k++) out[k * T + t] = src[so + k]; }
  return out;
}
async function prosody(en, D, T, sPros) {
  const o = await run('prosody', { en: T32(en, [1, D, T]), s: T32(sPros, [1, 128]) });
  return { F0: o.F0.data, N: o.N.data };
}
async function decode(asr, F0, N, T, s, biasDelta) {
  const o = await run('decoder', { asr: T32(asr, [1, 512, T]), F0: T32(F0, [1, 2 * T]), N: T32(N, [1, 2 * T]), s: T32(s, [1, 128]), bias_delta: T32(biasDelta, [P]) });
  return o.audio.data;
}
async function render({ ids, style, speed = 1, styleDelta, biasDelta, f0Ratio = 1, useTrained = false }) {
  if (useTrained && trainState) { styleDelta = trainState.style; biasDelta = trainState.bias; }
  const f = await frontend(ids, style, speed), idx = frameTokens(f.pd), T = idx.length;
  const pr = await prosody(gatherRowsT(f.d, f.L, f.dDim, idx), f.dDim, T, style.subarray(128, 256));
  if (f0Ratio !== 1) for (let i = 0; i < pr.F0.length; i++) pr.F0[i] *= f0Ratio;
  const s = Float32Array.from(style.subarray(0, 128));
  if (styleDelta) for (let i = 0; i < 128; i++) s[i] += styleDelta[i];
  const audio = await decode(gatherCols(f.t_en, 512, f.L, idx), pr.F0, pr.N, T, s, biasDelta || new Float32Array(P));
  return { audio: Float32Array.from(audio), f, F0: pr.F0, T };
}

// ---------- DSP: FFT, mel, MFCC, DTW, YIN ----------
const fftCache = new Map();
function fftTables(n) {
  if (fftCache.has(n)) return fftCache.get(n);
  const bits = Math.log2(n), rev = new Uint32Array(n), cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
  for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
  const t = { rev, cos, sin }; fftCache.set(n, t); return t;
}
function fft(re, im) {
  const n = re.length, { rev, cos, sin } = fftTables(n);
  for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let s0 = 0; s0 < n; s0 += size) for (let k = 0; k < half; k++) {
      const c = cos[k * step], sn = -sin[k * step], a = s0 + k, b = a + half;
      const tr = re[b] * c - im[b] * sn, ti = re[b] * sn + im[b] * c;
      re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
    }
  }
}
const hz2mel = (f) => 2595 * Math.log10(1 + f / 700), mel2hz = (m) => 700 * (10 ** (m / 2595) - 1);
function melBank(sr, nfft, nMels) { // HTK scale, no normalisation (torchaudio MelSpectrogram defaults)
  const bins = nfft / 2 + 1, fb = new Float32Array(nMels * bins), mMax = hz2mel(sr / 2), pts = [];
  for (let i = 0; i < nMels + 2; i++) pts.push(mel2hz(mMax * i / (nMels + 1)));
  for (let m = 0; m < nMels; m++) for (let k = 0; k < bins; k++) {
    const f = k * sr / nfft, lo = pts[m], c = pts[m + 1], hi = pts[m + 2];
    fb[m * bins + k] = Math.max(0, Math.min((f - lo) / (c - lo), (hi - f) / (hi - c)));
  }
  return fb;
}
// Power-mel frames, centered with reflect padding: returns { frames, nMels, data[frames*nMels] }.
function melPower(x, sr, nfft, win, hop, nMels) {
  const pad = nfft / 2, len = x.length, frames = 1 + Math.floor(len / hop), bins = nfft / 2 + 1;
  const fb = melBank(sr, nfft, nMels), w = new Float64Array(nfft), off = (nfft - win) >> 1;
  for (let i = 0; i < win; i++) w[off + i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / win);
  const at = (i) => { if (i < 0) i = -i; if (i >= len) i = 2 * len - 2 - i; return x[Math.max(0, Math.min(len - 1, i))]; };
  const re = new Float64Array(nfft), im = new Float64Array(nfft), out = new Float32Array(frames * nMels), pw = new Float64Array(bins);
  for (let t = 0; t < frames; t++) {
    const o = t * hop - pad;
    for (let i = 0; i < nfft; i++) { re[i] = at(o + i) * w[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < bins; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < nMels; m++) { let s = 0; const r = m * bins; for (let k = 0; k < bins; k++) s += fb[r + k] * pw[k]; out[t * nMels + m] = s; }
  }
  return { frames, nMels, data: out };
}
function mfcc(x) { // 40 Hz frames: 19 cepstral coefficients (c1..c19), mean/variance normalised
  const M = melPower(x, 24000, 2048, 1200, 600, 40), F = M.frames, K = 19, out = new Float32Array(F * K);
  for (let t = 0; t < F; t++) for (let k = 1; k <= K; k++) {
    let s = 0; for (let m = 0; m < 40; m++) s += Math.log(M.data[t * 40 + m] + 1e-10) * Math.cos(Math.PI * k * (m + 0.5) / 40);
    out[t * K + k - 1] = s;
  }
  for (let k = 0; k < K; k++) {
    let mu = 0, v = 0; for (let t = 0; t < F; t++) mu += out[t * K + k]; mu /= F;
    for (let t = 0; t < F; t++) v += (out[t * K + k] - mu) ** 2; const sd = Math.sqrt(v / F) + 1e-5;
    for (let t = 0; t < F; t++) out[t * K + k] = (out[t * K + k] - mu) / sd;
  }
  return { F, K, data: out };
}
// DTW with cosine distance; returns for every row of A the mean matched column of B.
function dtwMap(A, B) {
  const n = A.F, m = B.F, K = A.K, norm = (X, i) => { let s = 0; for (let k = 0; k < K; k++) s += X.data[i * K + k] ** 2; return Math.sqrt(s) + 1e-9; };
  const na = Float32Array.from({ length: n }, (_, i) => norm(A, i)), nb = Float32Array.from({ length: m }, (_, j) => norm(B, j));
  const D = new Float32Array(n * m);
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
    let d = 0; for (let k = 0; k < K; k++) d += A.data[i * K + k] * B.data[j * K + k];
    const c = 1 - d / (na[i] * nb[j]);
    const best = i === 0 && j === 0 ? 0 : Math.min(i > 0 ? D[(i - 1) * m + j] : Infinity, j > 0 ? D[i * m + j - 1] : Infinity, i > 0 && j > 0 ? D[(i - 1) * m + j - 1] : Infinity);
    D[i * m + j] = c + best;
  }
  const sum = new Float64Array(n), cnt = new Float64Array(n);
  let i = n - 1, j = m - 1;
  for (;;) {
    sum[i] += j; cnt[i]++;
    if (i === 0 && j === 0) break;
    const a = i > 0 && j > 0 ? D[(i - 1) * m + j - 1] : Infinity, b = i > 0 ? D[(i - 1) * m + j] : Infinity, c = j > 0 ? D[i * m + j - 1] : Infinity;
    if (a <= b && a <= c) { i--; j--; } else if (b <= c) i--; else j--;
  }
  return Float64Array.from(sum, (s, k) => s / Math.max(1, cnt[k]));
}
// YIN pitch at 80 Hz (hop 300 on 24 kHz), unvoiced = 0. The difference function's cross term
// sum_{i<W} x[i]·x[i+τ] is computed for all τ at once with an FFT cross-correlation.
function yin(x, sr = 24000, hop = 300, frame = 1200, fmin = 50, fmax = 500, thr = 0.15) {
  const len = x.length, frames = 1 + Math.floor(len / hop), maxLag = Math.floor(sr / fmin), minLag = Math.floor(sr / fmax);
  const Wn = frame - maxLag, n = 2048;
  const ar = new Float64Array(n), ai = new Float64Array(n), br = new Float64Array(n), bi = new Float64Array(n);
  const d = new Float64Array(maxLag + 1), dn = new Float64Array(maxLag + 1), out = new Float32Array(frames);
  const at = (i) => (i < 0 || i >= len ? 0 : x[i]);
  for (let t = 0; t < frames; t++) {
    const o = t * hop - frame / 2;
    let e = 0; for (let i = 0; i < frame; i++) e += at(o + i) ** 2;
    if (e / frame < 1e-5) { out[t] = 0; continue; }
    ar.fill(0); ai.fill(0); br.fill(0); bi.fill(0);
    for (let i = 0; i < Wn; i++) ar[i] = at(o + i);
    for (let i = 0; i < frame; i++) br[i] = at(o + i);
    fft(ar, ai); fft(br, bi);
    // C = conj(A)·B; inverse FFT via conj(fft(conj(C))) / n
    for (let k = 0; k < n; k++) { const re = ar[k] * br[k] + ai[k] * bi[k], im = ar[k] * bi[k] - ai[k] * br[k]; ar[k] = re; ai[k] = -im; }
    fft(ar, ai);
    let e0 = 0; for (let i = 0; i < Wn; i++) e0 += at(o + i) ** 2;
    let el = e0, cum = 0; dn[0] = 1;
    for (let lag = 1; lag <= maxLag; lag++) {
      el += at(o + lag + Wn - 1) ** 2 - at(o + lag - 1) ** 2;
      d[lag] = Math.max(0, e0 + el - 2 * (ar[lag] / n));
      cum += d[lag];
      dn[lag] = cum > 0 ? d[lag] * lag / cum : 1;
    }
    let lag = 0;
    for (let k = minLag; k <= maxLag; k++) if (dn[k] < thr) { while (k + 1 <= maxLag && dn[k + 1] < dn[k]) k++; lag = k; break; }
    out[t] = lag ? sr / lag : 0;
  }
  const sm = Float32Array.from(out); // median-of-3 on voiced frames to remove octave blips
  for (let t = 1; t < frames - 1; t++) if (out[t] > 0 && out[t - 1] > 0 && out[t + 1] > 0) sm[t] = [out[t - 1], out[t], out[t + 1]].sort((p, q) => p - q)[1];
  return sm;
}
// StyleTTS2 energy: log of the L2 norm of the (linear) mel power across bins, at 80 Hz.
function energyN(x) {
  const M = melPower(x, 24000, 2048, 1200, 300, 80), out = new Float32Array(M.frames);
  for (let t = 0; t < M.frames; t++) { let s = 0; for (let m = 0; m < 80; m++) s += (M.data[t * 80 + m] + 1e-5) ** 2; out[t] = Math.log(Math.sqrt(s)); }
  return out;
}

// ---------- teacher-forcing data from a real recording ----------
async function prep({ ids, style, audio }) {
  const T = Math.floor(audio.length / 600), real = audio.subarray(0, 600 * T);
  if (T < W + 2) return { skipped: 'too short' };
  const base = await render({ ids, style });
  const f = base.f, A = mfcc(base.audio), B = mfcc(real);
  const realOf = dtwMap(A, B), ts = A.F;
  const bounds = [0]; for (const n of f.pd) bounds.push(bounds.at(-1) + n);
  const rb = [0];
  for (let k = 1; k < bounds.length - 1; k++) rb.push(Math.round(realOf[Math.min(bounds[k], ts - 1)]));
  rb.push(T);
  for (let k = 1; k < rb.length; k++) rb[k] = Math.max(rb[k], rb[k - 1]);
  const durs = new Int32Array(f.L); for (let k = 0; k < f.L; k++) durs[k] = rb[k + 1] - rb[k];
  for (let guard = 0; guard < 10000; guard++) {
    let lo = 0, hi = 0; for (let k = 1; k < f.L; k++) { if (durs[k] < durs[lo]) lo = k; if (durs[k] > durs[hi]) hi = k; }
    if (durs[lo] >= 1) break; durs[lo]++; durs[hi]--;
  }
  const idx = frameTokens(durs);
  const asr = gatherCols(f.t_en, 512, f.L, idx);
  const F0 = new Float32Array(2 * T), N = new Float32Array(2 * T), yf = yin(real), nf = energyN(real);
  let nMin = Infinity; for (const v of nf) nMin = Math.min(nMin, v);
  for (let t = 0; t < 2 * T; t++) { F0[t] = t < yf.length ? yf[t] : 0; N[t] = t < nf.length ? nf[t] : nMin; }
  const item = { asr, F0, N, T, audio: Float32Array.from(real), s: Float32Array.from(style.subarray(0, 128)) };
  dataset.push(item);
  const realSpeech = durs.slice(1, -1).reduce((a, b) => a + b, 0), predSpeech = f.pd.slice(1, -1).reduce((a, b) => a + b, 0);
  return { T, items: dataset.length, f0Real: F0.filter((v) => v > 0), f0Pred: base.F0.filter((v) => v > 10), realSpeech, predSpeech };
}

// ---------- training (Adam on bias_delta and style_delta) ----------
async function train({ steps, lrBias = 2e-3, lrStyle = 1e-2, resume = false }, id) {
  if (!dataset.length) throw new Error('No training data prepared.');
  if (!trainState || !resume) trainState = { bias: new Float32Array(P), style: new Float32Array(128), mb: new Float32Array(P), vb: new Float32Array(P), ms: new Float32Array(128), vs: new Float32Array(128), t: 0 };
  const st = trainState, b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const adam = (p, g, m, v, lr) => {
    const c1 = 1 - b1 ** st.t, c2 = 1 - b2 ** st.t;
    for (let i = 0; i < p.length; i++) { m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]; p[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps); }
  };
  stopFlag = false;
  const t0 = performance.now();
  for (let step = 1; step <= steps && !stopFlag; step++) {
    const d = dataset[Math.floor(Math.random() * dataset.length)];
    const a = Math.floor(Math.random() * (d.T - W + 1));
    const feeds = {
      asr: T32(gatherWindow(d.asr, 512, d.T, a, W), [1, 512, W]),
      F0: T32(d.F0.slice(2 * a, 2 * (a + W)), [1, 2 * W]), N: T32(d.N.slice(2 * a, 2 * (a + W)), [1, 2 * W]),
      s_base: T32(d.s, [1, 128]), target: T32(d.audio.slice(600 * a, 600 * (a + W)), [1, 600 * W]),
      bias_delta: T32(Float32Array.from(st.bias), [P]), style_delta: T32(Float32Array.from(st.style), [1, 128]),
    };
    const o = await run('train_step', feeds);
    st.t++;
    adam(st.bias, o.grad_bias.data, st.mb, st.vb, lrBias);
    adam(st.style, o.grad_style.data, st.ms, st.vs, lrStyle);
    post({ id, type: 'progress', step, steps, loss: o.loss.data[0], secPerStep: (performance.now() - t0) / 1000 / step });
  }
  return { bias: Float32Array.from(st.bias), style: Float32Array.from(st.style), steps: st.t };
}
function gatherWindow(src, C, T, a, w) { const out = new Float32Array(C * w); for (let c = 0; c < C; c++) out.set(src.subarray(c * T + a, c * T + a + w), c * w); return out; }

self.onmessage = async (ev) => {
  const { id, type, payload } = ev.data;
  try {
    let result, transfer = [];
    if (type === 'load') result = await load(payload);
    else if (type === 'render') { const r = await render(payload); result = { audio: r.audio }; transfer = [r.audio.buffer]; }
    else if (type === 'prep') result = await prep(payload);
    else if (type === 'reset') { dataset.length = 0; trainState = null; result = true; }
    else if (type === 'train') result = await train(payload, id);
    else if (type === 'stop') { stopFlag = true; result = true; }
    else throw new Error('Unknown message ' + type);
    post({ id, type: 'done', result }, transfer);
  } catch (e) { post({ id, type: 'error', message: (e && e.message) || String(e) }); }
};
