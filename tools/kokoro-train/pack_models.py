"""Package the ONNX graphs for the browser.

- Large float32 weights are stored as float16 and cast back to float32 inside the graph
  (ONNX Runtime folds the casts at load time, so compute stays float32). Halves the download.
- decoder.onnx and train_step.onnx share their decoder weights through one external data file.
- Files larger than 90 MB are split into parts (GitHub's per-file limit is 100 MB); the page joins them.
Writes OUT/manifest.json describing graphs, external data and parts.
"""
import os, sys, json, hashlib
import numpy as np, onnx
from onnx import numpy_helper, helper, TensorProto

SRC = sys.argv[1]
OUT = sys.argv[2]
PART = 90 * 1000 * 1000
os.makedirs(OUT, exist_ok=True)

def to_fp16_storage(model, min_elems=1024):
    g = model.graph
    keep, casts, n_conv = [], [], 0
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= min_elems:
            arr = numpy_helper.to_array(init).astype(np.float16)
            h = numpy_helper.from_array(arr, init.name + '__fp16')
            keep.append(h)
            casts.append(helper.make_node('Cast', [h.name], [init.name], to=TensorProto.FLOAT, name=init.name + '__cast'))
            n_conv += 1
        else:
            keep.append(init)
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)
    return n_conv

def digest(t):
    return hashlib.sha1(numpy_helper.to_array(t).tobytes() + str(list(t.dims)).encode()).hexdigest()

def externalize(model, fname, table, buf, min_bytes=4096):
    """Move large initializers to the shared file `fname`, reusing bytes already in `table`."""
    moved = reused = 0
    for t in model.graph.initializer:
        raw = numpy_helper.to_array(t).tobytes()
        if len(raw) < min_bytes:
            continue
        key = digest(t)
        if key in table:
            off, ln = table[key]; reused += 1
        else:
            pad = (-len(buf)) % 64
            buf.extend(b'\0' * pad)
            off, ln = len(buf), len(raw)
            buf.extend(raw); table[key] = (off, ln); moved += 1
        t.ClearField('raw_data'); t.ClearField('float_data'); t.ClearField('int32_data'); t.ClearField('int64_data')
        t.data_location = TensorProto.EXTERNAL
        del t.external_data[:]
        for k, v in (('location', fname), ('offset', str(off)), ('length', str(ln))):
            e = t.external_data.add(); e.key = k; e.value = v
    return moved, reused

def write_parts(name, data):
    parts = []
    for i in range(0, max(1, len(data)), PART):
        pn = f'{name}.part{len(parts)}' if len(data) > PART else name
        with open(f'{OUT}/{pn}', 'wb') as f: f.write(data[i:i + PART])
        parts.append(pn)
    return parts

manifest = {'version': 1, 'graphs': {}, 'external': {}}
models = {k: onnx.load(f'{SRC}/{k}.onnx') for k in ['frontend', 'prosody', 'decoder', 'train_step']}
for k, m in models.items():
    print(k, 'fp16-stored tensors:', to_fp16_storage(m), flush=True)

shared, table = bytearray(), {}
for k in ['decoder', 'train_step']:
    moved, reused = externalize(models[k], 'decoder_weights.bin', table, shared)
    print(f'{k}: {moved} tensors added to decoder_weights.bin, {reused} reused', flush=True)
manifest['external']['decoder_weights.bin'] = write_parts('decoder_weights.bin', bytes(shared))

for k, m in models.items():
    data = m.SerializeToString()
    manifest['graphs'][k] = {'parts': write_parts(f'{k}.onnx', data),
                             'external': ['decoder_weights.bin'] if k in ('decoder', 'train_step') else []}
manifest['train'] = json.load(open(f'{SRC}/train_meta.json'))
manifest['bytes'] = sum(os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT))
# Build id: the page uses it to cache-bust model files whenever they change.
h = hashlib.sha1()
for f in sorted(os.listdir(OUT)):
    with open(f'{OUT}/{f}', 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''): h.update(chunk)
manifest['build'] = h.hexdigest()[:12]
json.dump(manifest, open(f'{OUT}/manifest.json', 'w'), indent=1)
print('total MB', manifest['bytes'] / 1e6)
for f in sorted(os.listdir(OUT)): print(f'  {f}  {os.path.getsize(f"{OUT}/{f}") / 1e6:.1f} MB')
