"""Export Kokoro as three inference graphs for the browser, matching the training graph.

frontend.onnx : input_ids [1,L] int64, style [1,256], speed [1]  -> dur [L] (float, before rounding),
                t_en [1,512,L], d [1,L,640]
prosody.onnx  : en [1,640,T], s [1,128]                        -> F0 [1,2T], N [1,2T]
decoder.onnx  : asr [1,512,T], F0 [1,2T], N [1,2T], s [1,128], bias_delta [P] -> audio [1,600T]
The JS side rounds durations, builds the alignment (repeat per token) and the matmuls.
"""
import os, json, torch, torch.nn as nn
import export_train as E

S = E.S
OUT = os.environ.get('OUT', f'{S}/onnx')

class Frontend(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, input_ids, style, speed):
        m = self.m
        L = input_ids.shape[1]
        mask = torch.zeros(1, L, dtype=torch.bool)
        lens = torch.full((1,), L, dtype=torch.long)
        bert_dur = m.bert(input_ids, attention_mask=torch.ones_like(input_ids))
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        s = style[:, 128:]
        d = m.predictor.text_encoder(d_en, s, lens, mask)
        x, _ = m.predictor.lstm(d)
        dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1) / speed
        t_en = m.text_encoder(input_ids, lens, mask)
        return dur.reshape(-1), t_en, d

class Prosody(nn.Module):
    def __init__(self, m):
        super().__init__(); self.p = m.predictor
    def forward(self, en, s):
        return self.p.F0Ntrain(en, s)

if __name__ == '__main__':
    m = E.load_decoder()
    m.eval()
    L, T = 40, 120
    ids = torch.randint(1, 170, (1, L)); style = torch.randn(1, 256) * 0.1
    # Frontend and prosody (LSTMs, packed sequences) export cleanly with the TorchScript exporter.
    torch.onnx.export(Frontend(m), (ids, style, torch.ones(1)), f'{OUT}/frontend.onnx', dynamo=False, opset_version=18,
                      input_names=['input_ids', 'style', 'speed'], output_names=['dur', 't_en', 'd'],
                      dynamic_axes={'input_ids': {1: 'L'}, 'dur': {0: 'L'}, 't_en': {2: 'L'}, 'd': {1: 'L'}})
    print('frontend.onnx', os.path.getsize(f'{OUT}/frontend.onnx') / 1e6, 'MB', flush=True)
    torch.onnx.export(Prosody(m), (torch.randn(1, 640, T), torch.randn(1, 128) * 0.1), f'{OUT}/prosody.onnx', dynamo=False, opset_version=18,
                      input_names=['en', 's'], output_names=['F0', 'N'], dynamic_axes={'en': {2: 'T'}, 'F0': {1: 'T2'}, 'N': {1: 'T2'}})
    print('prosody.onnx', os.path.getsize(f'{OUT}/prosody.onnx') / 1e6, 'MB', flush=True)
    DT = torch.export.Dim('T', min=8, max=4000)
    dec = E.DecodeStep(m.decoder).eval()
    P = int(dec.t.base_bias.numel())
    D2 = 2 * DT
    torch.onnx.export(dec, (torch.randn(1, 512, T), torch.full((1, 2 * T), 140.0), torch.zeros(1, 2 * T), torch.randn(1, 128) * 0.1, torch.zeros(P)),
                      f'{OUT}/decoder.onnx', dynamo=True, opset_version=18,
                      input_names=['asr', 'F0', 'N', 's', 'bias_delta'], output_names=['audio'],
                      dynamic_shapes={'asr': {2: DT}, 'F0': {1: D2}, 'N': {1: D2}, 's': None, 'bias_delta': None}, external_data=False)
    print('decoder.onnx', os.path.getsize(f'{OUT}/decoder.onnx') / 1e6, 'MB', flush=True)
