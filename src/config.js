// Model locations and constants shared by the app.
export const SR = 16000;          // speech models (Whisper, WeSpeaker, Silero VAD) and stored clips
export const SEP_SR = 44100;      // TIGER-DnR separator
export const KOKORO_SR = 24000;

export const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js';
export const KOKORO_URL = 'https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/dist/kokoro.web.js';
export const KOKORO_MODEL = 'onnx-community/Kokoro-82M-v1.0-ONNX';
export const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.webgpu.min.mjs';
export const ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/';
export const EMB_MODEL = 'onnx-community/wespeaker-voxceleb-resnet34-LM';
export const SEPARATOR_URL = 'https://huggingface.co/audiojs/tiger-dnr/resolve/main/tiger.int8.onnx';
export const VAD_URL = 'https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx';
export const TRAIN_MODELS = new URL('../models/kokoro-train/', import.meta.url).href;
export const CACHE = 'voice-adapter-models';

export const ASR_MODELS = [
  { id: 'Xenova/whisper-base', label: 'Whisper base (75 MB)' },
  { id: 'Xenova/whisper-tiny', label: 'Whisper tiny (40 MB)' },
  { id: 'Xenova/whisper-small', label: 'Whisper small (250 MB)' },
  { id: 'onnx-community/whisper-large-v3-turbo', label: 'Whisper large-v3-turbo (1.3 GB, WebGPU)', big: true },
];
export const KOKORO_VOICES = ['af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah', 'af_sky',
  'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa',
  'bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'];
export const VOICE_LEN = 510 * 256;   // a Kokoro voice table: one 256-dim style row per input length
export const PROBES = ['Get ready, the next round starts right now.', 'I never thought we would make it this far together.', 'Where did you hide the map? Tell me the truth.'];
export const PALETTE = ['#ff8a3d', '#5b9cff', '#3ddc97', '#c77dff', '#ffd23f', '#ff6b9a', '#4dd0e1', '#a3e048', '#f4a261'];
