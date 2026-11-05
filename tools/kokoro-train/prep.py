"""Teacher-forcing data for Kokoro decoder adaptation, from real recordings + transcripts.

For each utterance: phonemes -> Kokoro frontend with a base voice (predicted durations),
DTW between the base-voice rendering and the real audio to move token boundaries onto the
real timeline, then real F0 (pYIN) and energy N (StyleTTS2 log-norm) at 80 Hz.
Output: list of dicts with asr [512,T], F0 [2T], N [2T], audio [600T] (24 kHz), saved to a .pt file.
"""
import sys, os, glob, subprocess, json
import numpy as np, torch, librosa, torchaudio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kload

S = os.environ.get('KOKORO_WORK', os.path.dirname(os.path.abspath(__file__)))  # work dir: weights, data, outputs
SPK = sys.argv[1] if len(sys.argv) > 1 else '1272'
BASE = sys.argv[2] if len(sys.argv) > 2 else 'am_eric'
torch.manual_seed(0)

def phonemize(lines):
    out = subprocess.run(['node', f'{S}/g2p/g2p.mjs'], input='\n'.join(lines), capture_output=True, text=True, cwd=f'{S}/g2p').stdout.strip().split('\n')
    fix = lambda p: p.replace('ʲ', 'j').replace('r', 'ɹ').replace('x', 'k').replace('ɬ', 'l')
    return [fix(p) for p in out]

def utterances(spk):
    trans = {}
    for f in glob.glob(f'{S}/media/LibriSpeech/dev-clean/{spk}/*/*.trans.txt'):
        for line in open(f):
            k, t = line.strip().split(' ', 1); trans[k] = t
    files = sorted(glob.glob(f'{S}/media/LibriSpeech/dev-clean/{spk}/*/*.flac'))
    train, held, tot = [], [], 0.0
    for f in files:  # same split as test-media/librispeech-<spk>-{train,heldout}.ogg
        d = librosa.get_duration(path=f)
        if tot < 90: train.append(f); tot += d
        elif len(held) < 3: held.append(f)
    key = lambda f: os.path.basename(f)[:-5]
    return [(f, trans[key(f)]) for f in train], [(f, trans[key(f)]) for f in held]

mel_tf = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
def energy_N(y24):
    mel = mel_tf(torch.from_numpy(y24).float())  # [80, frames]
    return torch.log((mel + 1e-5).norm(dim=0))     # StyleTTS2 log_norm of exp(mel_norm*4-4)

@torch.no_grad()
def frontend(m, ids, ref_s):
    input_ids = torch.LongTensor([[0, *ids, 0]])
    L = input_ids.shape[1]
    mask = torch.zeros(1, L, dtype=torch.bool)
    lens = torch.LongTensor([L])
    bert_dur = m.bert(input_ids, attention_mask=(~mask).int())
    d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
    s = ref_s[:, 128:]
    d = m.predictor.text_encoder(d_en, s, lens, mask)
    x, _ = m.predictor.lstm(d)
    dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1)
    pred_dur = torch.round(dur).clamp(min=1).long().squeeze()
    t_en = m.text_encoder(input_ids, lens, mask)
    return d, t_en, pred_dur

def aln(durs, L):
    idx = torch.repeat_interleave(torch.arange(L), durs)
    a = torch.zeros(L, idx.shape[0]); a[idx, torch.arange(idx.shape[0])] = 1
    return a.unsqueeze(0)

def mfcc(y):
    M = librosa.feature.mfcc(y=y, sr=24000, n_mfcc=20, n_fft=2048, hop_length=600, win_length=1200)[1:]
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-5)

def prep(m, voice, items):
    out = []
    phs = phonemize([t.lower() for _, t in items])
    for (f, text), ps in zip(items, phs):
        ids = [m.vocab[c] for c in ps if c in m.vocab][:500]
        ref_s = voice[len(ids) - 1]
        d, t_en, pred_dur = frontend(m, ids, ref_s)
        L = len(ids) + 2
        with torch.no_grad():
            en = d.transpose(-1, -2) @ aln(pred_dur, L)
            F0p, Np = m.predictor.F0Ntrain(en, ref_s[:, 128:])
            synth = m.decoder(t_en @ aln(pred_dur, L), F0p, Np, ref_s[:, :128]).squeeze().numpy()
        y16, _ = librosa.load(f, sr=16000)
        y = librosa.resample(y16, orig_sr=16000, target_sr=24000)
        T = len(y) // 600; y = y[:600 * T]
        # DTW between base-voice rendering and real speech, at the 40 Hz token-frame rate.
        A, B = mfcc(synth), mfcc(y)
        _, path = librosa.sequence.dtw(X=A, Y=B, metric='cosine')
        path = path[::-1]
        ts = A.shape[1]; real_of = np.zeros(ts)
        for i in range(ts): real_of[i] = path[path[:, 0] == i, 1].mean() if (path[:, 0] == i).any() else np.nan
        real_of = np.nan_to_num(real_of, nan=0)
        bounds = np.concatenate([[0], np.cumsum(pred_dur.numpy())])
        rb = [0] + [int(round(real_of[min(b, ts - 1)])) for b in bounds[1:-1]] + [T]
        rb = np.maximum.accumulate(np.array(rb))
        durs = np.diff(rb)
        # every token needs at least one frame; borrow from the longest
        while (durs < 1).any():
            i = int(np.argmin(durs)); j = int(np.argmax(durs)); durs[i] += 1; durs[j] -= 1
        durs = torch.from_numpy(durs).long()
        assert int(durs.sum()) == T
        asr = (t_en @ aln(durs, L)).squeeze(0)
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=24000, frame_length=1200, hop_length=300, center=True)
        f0 = np.nan_to_num(f0, nan=0.0)
        F0 = torch.zeros(2 * T); F0[:min(2 * T, len(f0))] = torch.from_numpy(f0[:2 * T]).float()
        Nn = energy_N(y); N = torch.full((2 * T,), float(Nn.min())); N[:min(2 * T, len(Nn))] = Nn[:2 * T]
        out.append(dict(asr=asr, F0=F0, N=N, audio=torch.from_numpy(y).float(), text=text, ps=ps, ids=ids,
                        pred_f0=F0p.squeeze(0)))
        print(f'{os.path.basename(f)} T={T} tokens={L} synth={len(synth)/24000:.1f}s real={len(y)/24000:.1f}s', flush=True)
    return out

if __name__ == '__main__':
    m = kload.load()
    voice = torch.load(f'{kload.W}/voices/{BASE}.pt', weights_only=True)
    train, held = utterances(SPK)
    data = prep(m, voice, train)
    torch.save(dict(train=data, held=[f for f, _ in held], base=BASE), f'{S}/data_{SPK}.pt')
    print('saved', len(data), 'utterances,', sum(len(d['audio']) for d in data) / 24000, 's')
