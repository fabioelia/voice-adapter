# Voice Adapter

Train a voice from a talent's recordings and synthesize speech with it, in the browser (WebGPU or WebAssembly).

```
python3 tools/serve.py          # http://localhost:8000/
```

## Pages

- `index.html`: training. Clips can come from:
  - recordings (files or the microphone)
  - voices found in a video
  - audio with captions

  Deep training and Quick match train a Kokoro voice from those clips. Speak, then save, import or export the result.
- `clone.html`: zero-shot cloning with Audio8 from a 5–15 s reference.

## Code

| Path | |
|---|---|
| `src/main.js` | page wiring, training controls, saved voices |
| `src/recordings.js` | clips: files, microphone, voices in a video, audio + captions |
| `src/train.js` | `quickMatch` (voice-space search) and `deepTrain` (gradient training of Kokoro's decoder style layers) |
| `src/kokoro.js` | Kokoro load, stock and trained voices, `synth` |
| `src/speakers.js` | WeSpeaker voice prints, speech regions, speaker clustering |
| `src/separate.js` | voice isolation from music and effects (TIGER-DnR) |
| `src/captions.js` | WebVTT / SRT / TTML parsing, caption-to-speech alignment (Silero VAD) |
| `src/voices.js` | IndexedDB store, `.voice.json` import/export |
| `src/speak.js`, `src/audio.js`, `src/ui.js`, `src/runtime.js`, `src/config.js` | speak panel, audio, DOM, workers, constants |
| `src/workers/` | Whisper + WeSpeaker, Kokoro, separator, training step, mic capture |
| `models/kokoro-train/` | split Kokoro graphs and the training-step graph |
| `models/audio8/` | Audio8 graphs patched for ONNX Runtime Web |
| `tools/kokoro-train/` | export scripts for `models/kokoro-train/` |

## Models

Kokoro-82M (Apache-2.0), Whisper (MIT), WeSpeaker ResNet34 (Apache-2.0), Silero VAD (MIT), TIGER-DnR (Apache-2.0), Audio8-TTS-Preview (Apache-2.0).
