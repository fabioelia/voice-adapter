# Kokoro deep-training models

These scripts build the files in `models/kokoro-train/`, which **Deep training** uses to fine-tune a voice inside the browser. You only need them to rebuild or change those files; the page uses the committed copies.

## How the browser training works

Kokoro-82M is split into three inference graphs plus one training graph, all run with ONNX Runtime Web:

| Graph | What it does | Runs on |
|---|---|---|
| `frontend.onnx` | text tokens + voice style → per-token durations, text features | WebAssembly (LSTMs) |
| `prosody.onnx` | aligned features → pitch (F0) and energy curves | WebAssembly (LSTM) |
| `decoder.onnx` | features + F0 + energy + style → 24 kHz audio, with the trained `bias_delta` as an input | WebGPU |
| `train_step.onnx` | one training step: decoder forward pass, multi-resolution log-mel L1 loss against the real recording, and the **gradients** for the trainable parameters | WebGPU |

ONNX Runtime Web cannot train, so the training step is exported as an ordinary inference graph that outputs the loss and the gradients. `export_train.py` traces the joint forward and backward pass with `torch.func.grad_and_value` and `make_fx`, then exports it. Adam runs in JavaScript.

**What gets trained (AdaSpeech-style speaker adaptation):** a 128-value offset to the voice's decoder style, plus the biases of all 58 AdaIN layers in the decoder and generator. That is 37,524 parameters; every convolution stays frozen, so audio quality is preserved.

**Training data, prepared in the page from the talent's recordings:**
1. Whisper transcribes the recordings into sentences, and Kokoro's tokenizer turns them into phonemes.
2. The starting voice renders each sentence, and DTW on MFCCs maps each token onto the real recording's timeline.
3. Real pitch comes from YIN at 80 Hz, and energy uses StyleTTS2's log-norm of the mel spectrum.
4. Training runs on random 1.2 s windows. Inference also applies the talent's pitch ratio and pace ratio, measured against the starting voice.

**Measured:** this setup was tested on LibriSpeech speaker 1272 with 90 s of audio and 300 steps, scoring WeSpeaker similarity of new sentences against held-out recordings. It scored 0.40–0.42, against 0.07–0.08 for the starting stock voice. Gradients from the ONNX graph match PyTorch with cosine similarity of 0.96–0.999.

## Numerical changes made for ONNX and WebGPU

The exported decoder is Kokoro's, with numerically equivalent rewrites that keep PyTorch, ONNX Runtime CPU and WebGPU in agreement. Each is commented in `export_train.py`:
- The sine source has no random phase or noise, and keeps its phase wrapped so 32-bit precision holds.
- Linear and nearest upsampling are written as Gather and Expand operations.
- `atan2` is rewritten without IsNaN, NonZero or ScatterND.
- The convolution backward pass uses forward convolutions.
- The loss uses a smooth |·| and zero padding.

The training graph uses only operators that ONNX Runtime's WebGPU backend supports.

## Rebuild

Python 3.13 with PyTorch 2.14 works. `kokoro` declares Python < 3.13, so install its model code without dependencies:

```bash
python -m venv venv && . venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime onnxscript numpy librosa soundfile transformers loguru attrs huggingface_hub torchaudio
pip install --no-deps --ignore-requires-python kokoro==0.9.4
python -c "from huggingface_hub import hf_hub_download as d; [d('hexgrad/Kokoro-82M', f, local_dir='kokoro-weights') for f in ['config.json','kokoro-v1_0.pth']]"

python export_train.py      # onnx/train_step.onnx + train_meta.json
python export_infer.py      # onnx/frontend.onnx, prosody.onnx, decoder.onnx
python pack_models.py onnx ../../models/kokoro-train   # fp16 storage, shared weights, <100 MB parts
```

`prep.py` and `train.py` are the PyTorch research prototype that produced the benchmark above. To use them:
- They need LibriSpeech `dev-clean` in `$KOKORO_WORK/media/` and the `phonemizer` npm package in `$KOKORO_WORK/g2p/`.
- Run `MODE=onnx` to train with the exported graph's gradients, as the browser does.

Model license: Kokoro-82M weights are Apache-2.0 (hexgrad/Kokoro-82M).
