"""Prototype: gradient training of Kokoro's decoder style layers on a real speaker.

Trainable: a 128-d offset to the voice's decoder style, plus the biases of every AdaIN `fc`
in the decoder/generator (AdaSpeech-style conditional-norm adaptation). Optional: fc weights.
Loss: multi-resolution log-mel L1 between the decoder's output (teacher-forced with real
durations/F0/energy) and the real recording. Evaluation: WeSpeaker similarity of NEW sentences
to held-out real recordings.
"""
import sys, os, time, random, json
import numpy as np, torch, torchaudio, librosa, onnxruntime as ort
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kload
from prep import frontend, aln, phonemize

S = os.environ.get('KOKORO_WORK', os.path.dirname(os.path.abspath(__file__)))  # work dir: weights, data, outputs
SPK = os.environ.get('SPK', '1272')
STEPS = int(os.environ.get('STEPS', '300'))
TRAIN_W = os.environ.get('TRAIN_W', '0') == '1'
CROP = int(os.environ.get('CROP', '48'))  # 40 Hz frames
torch.manual_seed(0); random.seed(0)

m = kload.load()
data = torch.load(f'{S}/data_{SPK}.pt', weights_only=False)
voice = torch.load(f'{kload.W}/voices/{data["base"]}.pt', weights_only=True)
dec = m.decoder
for p in m.parameters(): p.requires_grad_(False)
fcs = [mod for name, mod in dec.named_modules() if name.endswith('.fc') and isinstance(mod, torch.nn.Linear)]
orig = [(f.weight.detach().clone(), f.bias.detach().clone()) for f in fcs]
params = []
for f in fcs:
    f.bias.requires_grad_(True); params.append(f.bias)
    if TRAIN_W: f.weight.requires_grad_(True); params.append(f.weight)
style_delta = torch.zeros(1, 128, requires_grad=True)
n_train = sum(p.numel() for p in params) + 128
print(f'{len(fcs)} AdaIN fc layers; trainable params: {n_train:,}', flush=True)

mels = [torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=n, hop_length=h, win_length=w, n_mels=80) for n, h, w in [(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]]
def mel_loss(y, x):
    return sum((torch.log(mt(y).clamp(min=1e-5)) - torch.log(mt(x).clamp(min=1e-5))).abs().mean() for mt in mels) / len(mels)

base_row = voice[len(data['train'][0]['ids']) - 1][:, :128]
MODE = os.environ.get('MODE', 'torch')
if MODE == 'onnx':
    # Gradients from the exported ONNX training step (what the browser runs); Adam stays in the host.
    sess = ort.InferenceSession(f'{S}/onnx/train_step.onnx', providers=['CPUExecutionProvider'])
    base_b = torch.cat([f.bias.detach().clone() for f in fcs])
    sizes = [f.bias.numel() for f in fcs]
    bias_delta = torch.zeros_like(base_b, requires_grad=True)
    CROP = json.load(open(f'{S}/onnx/train_meta.json'))['crop_frames']
    opt = torch.optim.Adam([{'params': [bias_delta], 'lr': float(os.environ.get('LR', '2e-3'))}, {'params': [style_delta], 'lr': 1e-2}])
    t0 = time.time(); hist = []
    for step in range(1, STEPS + 1):
        d = random.choice([x for x in data['train'] if x['asr'].shape[1] >= CROP])
        T = d['asr'].shape[1]; a = random.randint(0, T - CROP)
        feeds = dict(asr=d['asr'][None, :, a:a + CROP].numpy(), F0=d['F0'][None, 2 * a:2 * (a + CROP)].numpy(), N=d['N'][None, 2 * a:2 * (a + CROP)].numpy(),
                     s_base=voice[len(d['ids']) - 1][:, :128].numpy(), target=d['audio'][None, 600 * a:600 * (a + CROP)].numpy(),
                     bias_delta=bias_delta.detach().numpy(), style_delta=style_delta.detach().numpy())
        loss, gb, gs = sess.run(None, feeds)
        bias_delta.grad = torch.from_numpy(gb); style_delta.grad = torch.from_numpy(gs)
        opt.step(); hist.append(float(loss))
        if step % 25 == 0 or step == 1:
            print(f'[onnx] step {step} loss {np.mean(hist[-25:]):.4f}  {(time.time() - t0) / step:.2f}s/step', flush=True)
    with torch.no_grad():
        for f, b in zip(fcs, torch.split(base_b + bias_delta.detach(), sizes)): f.bias.copy_(b)
    STEPS_TORCH = 0
else:
    STEPS_TORCH = STEPS
if MODE != 'onnx':
    opt = torch.optim.Adam([{'params': params, 'lr': float(os.environ.get('LR', '2e-3'))}, {'params': [style_delta], 'lr': 1e-2}])
t0 = time.time(); hist = []
for step in range(1, STEPS_TORCH + 1):
    d = random.choice(data['train'])
    T = d['asr'].shape[1]; a = random.randint(0, max(0, T - CROP)); W = min(CROP, T - a)
    s = voice[len(d['ids']) - 1][:, :128] + style_delta
    y = dec(d['asr'][None, :, a:a + W], d['F0'][None, 2 * a:2 * (a + W)], d['N'][None, 2 * a:2 * (a + W)], s).squeeze()
    x = d['audio'][600 * a:600 * (a + W)]
    n = min(len(y), len(x)); loss = mel_loss(y[:n], x[:n])
    opt.zero_grad(); loss.backward(); opt.step()
    hist.append(loss.item())
    if step % 25 == 0 or step == 1:
        print(f'step {step} loss {np.mean(hist[-25:]):.4f}  {(time.time() - t0) / step:.2f}s/step', flush=True)

# ---------------- evaluation ----------------
ws = ort.InferenceSession(f'{S}/wespeaker/onnx/model.onnx')
def embed16(y16):
    fb = torchaudio.compliance.kaldi.fbank(torch.from_numpy(y16).float()[None] * 32768, num_mel_bins=80, sample_frequency=16000, dither=0.0)
    fb = fb - fb.mean(0, keepdim=True)
    e = ws.run(None, {'input_features': fb[None].numpy()})[0].reshape(-1)
    return e / np.linalg.norm(e)
held = np.concatenate([librosa.load(f, sr=16000)[0] for f in data['held']])
target = embed16(held)

# Pitch: shift the predicted contour so its voiced median matches the speaker's.
real_f0 = torch.cat([d['F0'] for d in data['train']]); pred_f0 = torch.cat([d['pred_f0'] for d in data['train']])
ratio = float(real_f0[real_f0 > 0].median() / pred_f0[pred_f0 > 10].median())
print(f'F0 ratio speaker/base = {ratio:.3f}')

SENTS = ['The weather changed quickly, and we had to find shelter before nightfall.',
         'Please remember to lock the door when you leave the studio tonight.',
         'Nobody expected the old bridge to hold, but it carried every one of us across.']
phs = phonemize([t.lower() for t in SENTS])

@torch.no_grad()
def synth(ps, delta=None, f0_ratio=1.0):
    ids = [m.vocab[c] for c in ps if c in m.vocab]
    ref_s = voice[len(ids) - 1]
    dd, t_en, pd = frontend(m, ids, ref_s)
    L = len(ids) + 2; A = aln(pd, L)
    F0, N = m.predictor.F0Ntrain(dd.transpose(-1, -2) @ A, ref_s[:, 128:])
    s = ref_s[:, :128] + (delta if delta is not None else 0)
    return dec(t_en @ A, F0 * f0_ratio, N, s).squeeze().numpy()

def score(tag, **kw):
    sims = []
    for i, ps in enumerate(phs):
        y = synth(ps, **kw)
        if i == 0: import soundfile as sf; sf.write(f'{S}/{SPK}_{tag}.wav', y, 24000)
        sims.append(float(embed16(librosa.resample(y, orig_sr=24000, target_sr=16000)) @ target))
    print(f'{tag:>28}: similarity to held-out real speech {np.mean(sims):.3f}  ({", ".join(f"{v:.2f}" for v in sims)})', flush=True)
    return float(np.mean(sims))

res = {}
res['trained'] = score('trained_' + MODE, delta=style_delta.detach(), f0_ratio=ratio)
res['trained_no_f0'] = score('trained_noF0shift', delta=style_delta.detach())
with torch.no_grad():
    for f, (w, b) in zip(fcs, orig): f.weight.copy_(w); f.bias.copy_(b)
res['base'] = score('base_' + data['base'])
res['base_f0'] = score('base_with_F0shift', f0_ratio=ratio)
json.dump(dict(res, steps=STEPS, train_w=TRAIN_W, trainable=n_train, ratio=ratio), open(f'{S}/result_{SPK}_{STEPS}_{int(TRAIN_W)}_{MODE}.json', 'w'))
