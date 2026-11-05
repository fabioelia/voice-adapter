let tts = null, device = null;
const post = (m, t) => self.postMessage(m, t || []);
// Trained voices are Kokoro style tables (510 × 256 floats, one row per input length), like the stock voice files.
const custom = new Map();
function hookCustomVoices(t) {
  const validate = t._validate_voice.bind(t), fromIds = t.generate_from_ids.bind(t);
  t._validate_voice = (v) => custom.has(v) ? custom.get(v).lang : validate(v);
  t.generate_from_ids = async (ids, opts = {}) => {
    const v = custom.get(opts.voice);
    if (!v) return fromIds(ids, opts);
    const Tensor = ids.constructor;
    const off = 256 * Math.min(Math.max(ids.dims.at(-1) - 2, 0), 509);
    const { waveform } = await t.model({ input_ids: ids, style: new Tensor('float32', v.data.slice(off, off + 256), [1, 256]), speed: new Tensor('float32', [opts.speed ?? 1], [1]) });
    return { audio: waveform.data, sampling_rate: 24000 };
  };
}
self.onmessage = async (ev) => {
  const { id, type, payload } = ev.data;
  try {
    if (type === 'load') {
      if (!tts) {
        const { KokoroTTS } = await import(payload.libUrl);
        const tries = payload.device === 'webgpu' ? [['webgpu', 'fp32'], ['wasm', 'q8']] : [['wasm', 'q8']];
        let err;
        for (const [d, dtype] of tries) {
          try {
            post({ id, type: 'status', message: `Loading Kokoro voices (${d})` });
            tts = await KokoroTTS.from_pretrained(payload.model, { dtype, device: d, progress_callback: (p) => post({ id, type: 'download', ...p }) });
            device = d; break;
          } catch (e) { err = e; post({ id, type: 'log', message: `Kokoro on ${d} failed: ${e.message || e}` }); }
        }
        if (!tts) throw err;
        hookCustomVoices(tts);
      }
      post({ id, type: 'done', result: device });
    } else if (type === 'registerVoice') {
      custom.set(payload.name, { data: payload.data, lang: payload.lang || 'a' });
      post({ id, type: 'done', result: true });
    } else if (type === 'tokenize') {
      // Kokoro's own text normalisation + phonemizer + tokenizer, without running the model.
      let ids = null;
      const g = tts.generate_from_ids;
      tts.generate_from_ids = async (inp) => { ids = Array.from(inp.data, Number); return { audio: new Float32Array(0), sampling_rate: 24000 }; };
      try { await tts.generate(payload.text, { voice: payload.voice || 'af_heart' }); } finally { tts.generate_from_ids = g; }
      post({ id, type: 'done', result: ids ? ids.slice(1, -1) : [] });
    } else if (type === 'synth') {
      const out = await tts.generate(payload.text, { voice: payload.voice, speed: payload.speed });
      const audio = out.audio instanceof Float32Array ? out.audio : Float32Array.from(out.audio);
      post({ id, type: 'done', result: { audio, sr: out.sampling_rate } }, [audio.buffer]);
    } else throw new Error('Unknown message ' + type);
  } catch (e) { post({ id, type: 'error', message: (e && e.message) || String(e) }); }
};
