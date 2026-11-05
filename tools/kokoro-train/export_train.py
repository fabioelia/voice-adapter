"""Export one Kokoro decoder-adaptation training step as a plain ONNX inference graph.

Inputs:  asr [1,512,W], F0 [1,2W], N [1,2W], s_base [1,128], target [1,600W],
         bias_delta [P] (offsets for every AdaIN fc bias), style_delta [1,128]
Outputs: loss [], grad_bias [P], grad_style [1,128]
The browser runs this with ONNX Runtime Web (WebGPU) and applies Adam to the two deltas in JS.
"""
import sys, os, math, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, librosa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kload
from kokoro import istftnet

S = os.environ.get('KOKORO_WORK', os.path.dirname(os.path.abspath(__file__)))  # work dir: weights, data, outputs
W = int(os.environ.get('CROP', '48'))
OUT = os.environ.get('OUT', f'{S}/onnx')
os.makedirs(OUT, exist_ok=True)

# ---- deterministic source module (no random phase / noise), as in the published ONNX graph ----
# Ops below are chosen to stay on ONNX Runtime's WebGPU backend: clamps on floats (Clip), relu
# instead of clamp(min=0), x - floor(x) instead of %, and no integer min/max.
def _lerp_at(x, src):
    L = x.shape[-1]
    f0 = torch.clamp(src.floor(), 0, L - 1)
    f1 = torch.clamp(f0 + 1, 0, L - 1)
    w = src - f0
    return x.index_select(-1, f0.long()), x.index_select(-1, f1.long()), w

def interp_down(x, k):
    """F.interpolate(x, scale_factor=1/k, mode='linear') for an integer k, written with Gather and
    arithmetic so ONNX matches PyTorch exactly (ONNX Resize maps coordinates differently)."""
    n = x.shape[-1] // k
    src = F.relu((torch.arange(n, device=x.device).to(x.dtype) + 0.5) * k - 0.5)
    a0, a1, w = _lerp_at(x, src)
    return a0 * (1 - w) + a1 * w

def _f02sine(self, f0_values):
    # Same maths as Kokoro's SineGen (without the random initial phase), but the accumulated
    # phase is wrapped before upsampling so it stays small: sin(2*pi*cycles) of numbers in the
    # thousands loses float32 precision differently in PyTorch and ONNX Runtime.
    k = int(self.upsample_scale)
    rad = f0_values / self.sampling_rate
    rad = rad - rad.floor()
    rad = interp_down(rad.transpose(1, 2), k)                         # [B, H, Ld]
    c = torch.cumsum(rad, dim=-1) * k                                 # cycles at the low rate
    n = c.shape[-1] * k
    src = F.relu((torch.arange(n, device=c.device).to(c.dtype) + 0.5) / k - 0.5)
    a0, a1, w = _lerp_at(c, src)
    cyc = (a0 - a0.floor()) + w * (a1 - a0)                           # wrapped phase, in cycles
    return torch.sin(2 * math.pi * cyc).transpose(1, 2)

def sine_forward(self, f0):
    fn = f0 * torch.arange(1, self.harmonic_num + 2, dtype=f0.dtype, device=f0.device).view(1, 1, -1)
    uv = (f0 > self.voiced_threshold).to(f0.dtype)
    return self._f02sine(fn) * self.sine_amp * uv, uv, None
def source_forward(self, x):
    with torch.no_grad():
        sine_wavs, uv, _ = self.l_sin_gen(x)
    return self.l_tanh(self.l_linear(sine_wavs)), None, uv
def atan2_webgpu(y, x):
    # torch.atan2 exports with IsNaN/NonZero/ScatterND, which the WebGPU backend lacks.
    safe_x = torch.where(x.abs() < 1e-12, torch.full_like(x, 1e-12), x)
    a = torch.atan(y / safe_x)
    return torch.where(x < 0, torch.where(y >= 0, a + math.pi, a - math.pi), a)
def _stft_transform(self, waveform):
    # Kokoro's CustomSTFT.transform, with a WebGPU-friendly atan2 (which already returns +pi for
    # imag == 0, real < 0, so the boolean-mask correction, and its NonZero op, is not needed).
    if self.center:
        waveform = F.pad(waveform, (self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)
    x = waveform.unsqueeze(1)
    real_out = F.conv1d(x, self.weight_forward_real, stride=self.hop_length)
    imag_out = F.conv1d(x, self.weight_forward_imag, stride=self.hop_length)
    return torch.sqrt(real_out ** 2 + imag_out ** 2 + 1e-14), atan2_webgpu(imag_out, real_out)
from kokoro import custom_stft
custom_stft.CustomSTFT.transform = _stft_transform

istftnet.SineGen._f02sine = _f02sine
istftnet.SineGen.forward = sine_forward
istftnet.SourceModuleHnNSF.forward = source_forward

class RepeatUp(nn.Module):
    """nn.Upsample(scale_factor=k, mode='nearest') as Expand + Reshape: exact in ONNX (the
    decomposed form computes float indices that round to an out-of-range value in ONNX Runtime)."""
    def __init__(self, k):
        super().__init__(); self.k = int(k)
    def forward(self, x):
        B, C, L = x.shape
        return x.unsqueeze(-1).expand(B, C, L, self.k).reshape(B, C, L * self.k)

def load_decoder():
    m = kload.KModel(config=f'{kload.W}/config.json', model=f'{kload.W}/kokoro-v1_0.pth', disable_complex=True).eval()
    for mod in m.modules():
        if hasattr(mod, 'weight_g'):
            torch.nn.utils.remove_weight_norm(mod)
    up = m.decoder.generator.f0_upsamp
    m.decoder.generator.f0_upsamp = RepeatUp(up.scale_factor)
    for p in m.parameters(): p.requires_grad_(False)
    return m

class STFTMag(nn.Module):
    """Magnitude STFT as conv1d with a fixed windowed DFT basis (center, reflect pad)."""
    def __init__(self, n_fft, hop, win):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        w = torch.hann_window(win, periodic=True); w = F.pad(w, ((n_fft - win) // 2, n_fft - win - (n_fft - win) // 2))
        k = torch.arange(n_fft // 2 + 1).float()[:, None]; n = torch.arange(n_fft).float()[None]
        ang = 2 * math.pi * k * n / n_fft
        self.register_buffer('basis', torch.cat([torch.cos(ang) * w, -torch.sin(ang) * w])[:, None, :])
    def forward(self, x):  # x [B, T]
        x = F.pad(x[:, None], (self.n_fft // 2, self.n_fft // 2))  # zero pad: its backward is a plain slice
        y = F.conv1d(x, self.basis, stride=self.hop)
        re, im = y.chunk(2, dim=1)
        return torch.sqrt(re * re + im * im + 1e-9)

class MelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.stfts = nn.ModuleList()
        for i, (n, h, w) in enumerate([(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]):
            self.stfts.append(STFTMag(n, h, w))
            self.register_buffer(f'fb{i}', torch.from_numpy(librosa.filters.mel(sr=24000, n_fft=n, n_mels=80, htk=True, norm=None)).float())
    def forward(self, y, x):
        tot = 0
        for i, st in enumerate(self.stfts):
            fb = getattr(self, f'fb{i}')
            my = torch.log(fb @ st(y) ** 2 + 1e-5); mx = torch.log(fb @ st(x) ** 2 + 1e-5)
            tot = tot + torch.sqrt((my - mx) ** 2 + 1e-6).mean()   # smooth |·| (no Sign op)
        return tot / len(self.stfts)

def conv_backward_1d(grad_out, inp, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, mask):
    """aten.convolution_backward for 1-D convs, written with forward convs only (ONNX has no backward op)."""
    s, p, d = stride[0], padding[0], dilation[0]
    k = weight.shape[-1]
    gi = gw = gb = None
    if mask[0]:
        if not transposed:
            L = inp.shape[-1]
            base = (grad_out.shape[-1] - 1) * s - 2 * p + d * (k - 1) + 1
            gi = F.conv_transpose1d(grad_out, weight, None, s, p, L - base, groups, d)
        else:
            gi = F.conv1d(grad_out, weight, None, s, p, d, groups)
    if mask[1]:
        raise NotImplementedError('weight gradients are not needed: conv weights are frozen')
    if mask[2]:
        gb = grad_out.sum(dim=(0, 2))
    return gi, gw, gb

def upsample_nearest1d_backward(grad_out, output_size, input_size, scales=None):
    B, C, L = input_size
    return grad_out.reshape(B, C, L, grad_out.shape[-1] // L).sum(-1)

def decomps():
    # Core ATen decompositions turn most *_backward ops into plain math; the two ops below
    # have no ONNX mapping and no core decomposition, so they are written out by hand.
    table = dict(torch._decomp.core_aten_decompositions())
    # Keep linear upsampling as a native op: its decomposition maps coordinates differently from
    # the kernel when scale_factor is given (1/300 in the sine source) and shifts the output.
    for k in [k for k in table if 'upsample' in str(k) and 'nearest' not in str(k)]:
        table.pop(k)
    table[torch.ops.aten.convolution_backward.default] = conv_backward_1d
    table[torch.ops.aten.upsample_nearest1d_backward.default] = upsample_nearest1d_backward
    return table

def adain_fcs(dec):
    return [(name, mod) for name, mod in dec.named_modules() if name.endswith('.fc') and isinstance(mod, nn.Linear)]

class TrainStep(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec
        self.fcs = adain_fcs(dec)
        self.names = [n + '.bias' for n, _ in self.fcs]
        self.sizes = [m.bias.numel() for _, m in self.fcs]
        self.offsets = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.register_buffer('base_bias', torch.cat([m.bias.detach() for _, m in self.fcs]))
        self.loss = MelLoss()
    def pieces(self, full):
        # One Slice per layer rather than a 58-way Split (and a 58-way Concat in the backward pass):
        # WebGPU shaders may bind at most 10 storage buffers on some devices (e.g. Apple Silicon).
        return [full.narrow(0, o, n) for o, n in zip(self.offsets, self.sizes)]
    def forward(self, asr, F0, N, s_base, target, bias_delta, style_delta):
        def f(bd, sd):
            biases = self.pieces(self.base_bias + bd)
            y = torch.func.functional_call(self.dec, dict(zip(self.names, biases)), (asr, F0, N, s_base + sd), strict=False)
            y = y.reshape(1, -1)[:, :target.shape[1]]
            return self.loss(y, target)
        (gb, gs), loss = torch.func.grad_and_value(f, argnums=(0, 1))(bias_delta, style_delta)
        return loss, gb, gs

class DecodeStep(nn.Module):
    """Inference decoder with the same adaptation inputs (bias_delta, style_delta folded into s)."""
    def __init__(self, dec):
        super().__init__()
        self.t = TrainStep(dec)
    def forward(self, asr, F0, N, s, bias_delta):
        t = self.t
        biases = t.pieces(t.base_bias + bias_delta)
        y = torch.func.functional_call(t.dec, dict(zip(t.names, biases)), (asr, F0, N, s), strict=False)
        return y.reshape(1, -1)

if __name__ == '__main__':
    m = load_decoder()
    step = TrainStep(m.decoder).eval()
    P = int(step.base_bias.numel())
    json.dump({'crop_frames': W, 'bias_sizes': step.sizes, 'bias_names': step.names, 'params': P}, open(f'{OUT}/train_meta.json', 'w'))
    g = torch.Generator().manual_seed(0)
    args = (torch.randn(1, 512, W, generator=g), torch.full((1, 2 * W), 120.0), torch.zeros(1, 2 * W), torch.randn(1, 128, generator=g) * 0.1,
            torch.randn(1, 600 * W, generator=g) * 0.05, torch.zeros(P), torch.zeros(1, 128))
    with torch.no_grad(): pass
    loss, gb, gs = step(*args)
    print('eager: loss', float(loss), '|grad_bias|', float(gb.norm()), '|grad_style|', float(gs.norm()), flush=True)
    # Trace the joint forward+backward into a flat graph of aten ops first; torch.export
    # cannot trace through torch.func transforms directly.
    from torch.fx.experimental.proxy_tensor import make_fx
    gm = make_fx(lambda *a: step(*a), tracing_mode='real', decomposition_table=decomps())(*args)
    print('joint graph nodes:', len(gm.graph.nodes), flush=True)
    class Joint(nn.Module):
        def __init__(self, gm): super().__init__(); self.gm = gm
        def forward(self, asr, F0, N, s_base, target, bias_delta, style_delta): return self.gm(asr, F0, N, s_base, target, bias_delta, style_delta)
    torch.onnx.export(Joint(gm), args, f'{OUT}/train_step.onnx', dynamo=True, opset_version=18,
                      input_names=['asr', 'F0', 'N', 's_base', 'target', 'bias_delta', 'style_delta'],
                      output_names=['loss', 'grad_bias', 'grad_style'], external_data=False)
    print('exported train_step.onnx', os.path.getsize(f'{OUT}/train_step.onnx') / 1e6, 'MB', flush=True)
