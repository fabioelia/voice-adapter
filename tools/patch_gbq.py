"""Replace com.microsoft GatherBlockQuantized (no CPU kernel in ONNX Runtime Web) with graph inputs.

Each GatherBlockQuantized node becomes a float16 graph input `emb_<i>`; the page looks the rows up
itself from the packed 4-bit table in the external data file. Writes <name>.web.onnx (graph only,
same external data) and <name>.web.json describing each input: table, index source, data offsets.
"""
import sys, json, onnx
from onnx import helper, TensorProto
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
g = m.graph
inits = {i.name: i for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}
def ext(t):
    e = {kv.key: kv.value for kv in t.external_data}
    return {'location': e['location'], 'offset': int(e.get('offset', 0)), 'length': int(e['length']), 'dims': list(t.dims), 'dtype': t.data_type}
def describe(name, depth=4):  # short description of how an index tensor is produced (for the JS mapping)
    n = prod.get(name)
    if not n or depth == 0: return name
    return f"{n.op_type}({', '.join(describe(i, depth - 1) for i in n.input if i)})"
from onnx import numpy_helper
def const_of(name):
    n = prod[name]
    for x in n.attribute:
        if x.name == 'value': return numpy_helper.to_array(x.t)
def source(idx):  # index = codes[:, row, :] (+ add)  or  token_id (+ add)
    n, add = prod[idx], 0
    if n.op_type == 'Add': add = int(const_of(n.input[1])); n = prod[n.input[0]]
    if n.op_type == 'Cast': return {'from': n.input[0], 'add': add}
    if n.op_type == 'Gather' and n.input[0] == 'codes': return {'from': 'codes', 'row': int(const_of(n.input[1])), 'add': add}
    raise SystemExit(f'unrecognised index expression for {idx}')
meta, keep_nodes, removed_inits = [], [], set()
for n in g.node:
    if n.op_type != 'GatherBlockQuantized':
        keep_nodes.append(n); continue
    a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
    data, idx, scales, zps = n.input
    j = len(meta)
    name = f'emb_{j}'
    g.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT16, ['batch', 'sequence', inits[data].dims[1]]))
    for c in g.node:
        for k, i in enumerate(c.input):
            if i == n.output[0]: c.input[k] = name
    for o in g.output:
        if o.name == n.output[0]: raise SystemExit('GBQ output is a graph output; not handled')
    meta.append({'input': name, 'node': n.name, 'source': source(idx), 'index_expr': describe(idx), 'block_size': a['block_size'],
                 'weight': ext(inits[data]), 'scales': ext(inits[scales]), 'zero_points': ext(inits[zps])})
    removed_inits.update([data, scales, zps])
del g.node[:]; g.node.extend(keep_nodes)
kept = [i for i in g.initializer if i.name not in removed_inits]
del g.initializer[:]; g.initializer.extend(kept)
onnx.save(m, dst)
json.dump(meta, open(dst.replace('.onnx', '.json'), 'w'), indent=1)
for x in meta: print(x['input'], x['node'], '<-', x['source'])
