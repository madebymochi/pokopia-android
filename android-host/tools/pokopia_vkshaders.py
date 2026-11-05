#!/usr/bin/env python3
"""Pack the translated SPIR-V (android-host/tools/pokopia_spirv.py output) for the Android host's Vulkan replay (android-host/gfx/nvn_vk.c).

Per shader key (the per-stage key hle_nvn.c prog_key computes, column 4 of shader_map.tsv):
  * its SPIR-V, zlib-compressed, with
      - capabilities the translator declares but no shader uses removed (Float64: the Xclipse 920 has no shaderFloat64; FragmentShaderPixelInterlockEXT and its
        extension: no shader calls Begin/EndInvocationInterlock), so pipelines need neither feature (a module that declares OpTypeFloat 64 is reported: it needs a feature the phone lacks);
      - every DescriptorSet decoration set to the stage's own set (vertex 0, pixel 1, geometry 2, hull 3, domain 4, compute 0): the translator puts all stages in set 0 with
        overlapping binding numbers, and one pipeline layout holds them all.
  * its c1 constant segment (the NVN driver's per-shader constants, cbuf slot 1), when not empty.
  * a row:  stage  set  bindings  inputs / outputs  extras
      bindings: space-separated  binding:type:kind[:a[:b]]; inputs / outputs: i<location>:<f|i|u><components>, o<location>:...
        type  u uniform buffer, b storage buffer, c combined image sampler (+ dim 1/2/3/C/R/S, a arrayed, s depth compare, f/i/u sampled type), t uniform texel buffer, i storage image (+ dim, a arrayed, f/i/u sampled type), j storage texel buffer
        kind  sup  Ryujinx support buffer (render scale etc., filled by the host)
              c1   the c1 segment            drv:n   driver segment c0 / c2 read directly (n = bank)
              push NVN2 push constants, c[24] (dmsh_abi.h DMSH_PUSH_BANK)
              ub:r user uniform slot r (NVN BindUniformBuffer index = cbuf slot - 3)
              sb:r storage slot r (from the c0 address table)  sbu:r:off  storage buffer whose address sits in user uniform slot r at byte off
              ubp:r compute stage only: a storage buffer on the address of NVN uniform buffer r (c0 + 0x210 + 16 r; dmsh_build.py 'ubo_ptr', the compute shaders' parameters)
              tx:t:s:ty  texture slot t (0-127 BindTexture, 128+ separate), sampler: ty 0 from the texture handle, 1 separate sampler slot s, 3 constant sampler ID s
              tu:r:off:sr:soff  bindless: texture handle in user uniform slot r at byte off, sampler handle at sr / soff (sr -1: sampler ID constant soff)
              im:r:w image slot r, w 1 if the shader writes it          ?    not decoded
      extras: f1:<binding>  the combined image sampler at this binding reads a 1D texture or texture buffer by integer coordinate that the translator turned into a 2D read at
                            (x & 0x3FFF, x >> 14) (res.json "fetch1d", dmsh_xlate RecordTexelFetch1D): the host must bind a texture buffer there as a 2D image of 16384-texel rows
Decoding follows nxnative's dm/gfx/tools/dmsh_build.py (decode_tex, decode_ubo_tex, reflect; MIT, seedds/nxnative): the translator names each resource after its NVN source (fp_c3, fp_s0, fp_t_tcb_8 ...).

Everything goes into one file, <out>/shaders.pack: about 99,000 shaders would be some 200,000 small files on the phone's shared storage. Format (little-endian; nvn_vk.c refl_load):
  "PKSHPACK", u32 version 1, u32 count, u64 index offset; per shader row bytes, c1 bytes, zlib bytes; index: count entries sorted by key {u64 key (the 16 hex digits as a number), u64 record offset, u32 row, c1, zlib, raw SPIR-V lengths}

Only keys that summary.json marks "ok" are taken; the others, and shaders with more bindings (40) or inputs (32) than nvn_vk.c shader_get keeps, are listed in a [RED] line. Exit code 1 only when no shader has valid SPIR-V or summary.json is missing.
Usage: pokopia_vkshaders.py <profile dir> <shader pkg dir> <spirv dir> <out dir>      Env: J=parallel jobs, SPIRV_VAL=spirv-val with VAL_SAMPLE=<n>: check every n-th output (Vulkan 1.3 environment), KEYS=<file>: only these keys, one per line
"""
import array, collections, json, multiprocessing as mp, os, re, struct, subprocess, sys, tempfile, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pokopia_spirv import Pack, read_map, split_record   # one reading of the package and the shader map for both steps: the c1 written here is the one the translator read
SET = {'vs': 0, 'ps': 1, 'gs': 2, 'hs': 3, 'ds': 4, 'cs': 0}
TEX_BASE, IMG_BASE, IMG_SLOTS, SEPTEX_BASE, SEPTEX_SLOT0, SAMP_BASE = 0x20, 0x120, 8, 0x168, 128, 0x568
SSBO = {'vs': 0x110, 'hs': 0x210, 'ds': 0x310, 'cs': 0x310, 'gs': 0x410, 'ps': 0x510}   # hull and domain: never measured, the gap the vertex and geometry tables leave (inference)
PUSH_BANK = 24                              # dmsh_abi.h DMSH_PUSH_BANK
DROP_CAPS = {10, 5378}                      # Float64, FragmentShaderPixelInterlockEXT
DROP_EXT = {'SPV_EXT_fragment_shader_interlock'}
OP_NAME, OP_EXTENSION, OP_CAPABILITY, OP_TYPE_IMAGE, OP_TYPE_SAMPLED_IMAGE, OP_TYPE_STRUCT, OP_TYPE_POINTER, OP_VARIABLE, OP_DECORATE, OP_FUNCTION = 5, 10, 17, 25, 27, 30, 32, 59, 71, 54
OP_TYPE_ARRAY, OP_TYPE_RTARRAY = 28, 29
OP_TYPE_INT, OP_TYPE_FLOAT, OP_TYPE_VECTOR = 21, 22, 23
DEC_LOCATION, DEC_BUILTIN = 30, 11
SC_INPUT, SC_OUTPUT = 1, 3
DIMC = {0: '1', 1: '2', 2: '3', 3: 'C', 4: 'R', 5: 'B', 6: 'S'}
DEC_BLOCK, DEC_BUFFER_BLOCK, DEC_BINDING, DEC_SET = 2, 3, 33, 34
SC_UNIFORM_CONSTANT, SC_UNIFORM, SC_STORAGE_BUFFER = 0, 2, 12
DIM_BUFFER = 5
MAX_BINDINGS, MAX_INPUTS = 40, 32   # nvn_vk.c shader_get: vbind b[40], in[32]


def decode_tex(name):
    m = re.match(r'[a-z]+_[ti]\w*?_(?:cb(\d+)|tcb)_([0-9A-F]+)', name)   # an image's name goes on after the handle: its format, empty when unknown
    if not m: return None
    slots = int(m.group(1)) if m.group(1) else -1; h = int(m.group(2), 16)
    tb = (slots & 0xffff) if slots >= 0 else 2; sb = ((slots >> 16) - 1) if slots >= 0 and (slots >> 16) else tb
    o0, o1, ty = h & 0x3fff, (h >> 14) & 0x3fff, h >> 28
    def slot(word, base, cb): return (word * 4 - base) // 8 if cb == 2 and word * 4 >= base and (word * 4 - base) % 8 == 0 else None
    words = [o0] + ([o1] if ty else [])
    if re.match(r'[a-z]+_i', name):
        return ('im', next((slot(x, IMG_BASE, tb) for x in words if IMG_BASE <= x * 4 < IMG_BASE + 8 * IMG_SLOTS and slot(x, IMG_BASE, tb) is not None), None))
    tw = [o0] if ty == 3 else words
    t = next((slot(x, TEX_BASE, tb) for x in tw if slot(x, TEX_BASE, tb) is not None and x * 4 < SEPTEX_BASE), None)
    if t is None: t = next((SEPTEX_SLOT0 + slot(x, SEPTEX_BASE, tb) for x in tw if slot(x, SEPTEX_BASE, tb) is not None and x * 4 < SAMP_BASE), None)
    if ty == 3: s = o1
    elif ty: s = next((slot(x, SAMP_BASE, sb) for x in words if x * 4 >= SAMP_BASE and slot(x, SAMP_BASE, sb) is not None), None)
    else: s = 0
    if t is None or s is None: return None
    return ('tx', t, s, ty)


def decode_ubo_tex(name):
    m = re.match(r'[a-z]+_t\w*?_cb(\d+)_([0-9A-F]+)', name)
    if not m: return None
    slots, h = int(m.group(1)), int(m.group(2), 16)
    tb = slots & 0xffff; sb = (slots >> 16) - 1 if slots >> 16 else tb
    o0, o1, ty = h & 0x3fff, (h >> 14) & 0x3fff, h >> 28
    if not 3 <= tb <= 16 or ty not in (0, 1, 3) or (ty == 1 and not 3 <= sb <= 16): return None
    if ty == 3: return ('tu', tb - 3, o0 * 4, -1, o1)
    if ty == 1: return ('tu', tb - 3, o0 * 4, sb - 3, o1 * 4)
    return ('tu', tb - 3, o0 * 4, tb - 3, o0 * 4)


def word_text(w, start, end):
    return w[start:end].tobytes().split(b'\0')[0].decode()


def parse(w):
    """The module's declarations (they all come before the first function): also where the first function starts, the capabilities and the extensions"""
    names, dec, blocks, types, ptrs, vars_, caps, exts = {}, collections.defaultdict(dict), {}, {}, {}, {}, [], []
    i = 5; n_w = len(w)
    while i < n_w:
        op, n = w[i] & 0xffff, w[i] >> 16
        if n == 0: raise ValueError('SPIR-V instruction length 0')
        if op == OP_FUNCTION: break
        if op == OP_NAME: names[w[i + 1]] = word_text(w, i + 2, i + n)
        elif op == OP_CAPABILITY: caps.append(w[i + 1])
        elif op == OP_EXTENSION: exts.append(word_text(w, i + 1, i + n))
        elif op == OP_DECORATE and n >= 3:
            t, d = w[i + 1], w[i + 2]
            if d in (DEC_BINDING, DEC_SET, DEC_LOCATION, DEC_BUILTIN): dec[t][d] = (w[i + 3] if n > 3 else 0, i + 3)
            elif d in (DEC_BLOCK, DEC_BUFFER_BLOCK): blocks[t] = d
        elif op == OP_TYPE_IMAGE: types[w[i + 1]] = ('image', w[i + 3], w[i + 7], w[i + 4], w[i + 5], w[i + 2])   # dim, sampled (1 sampled, 2 storage), depth, arrayed, sampled type
        elif op == OP_TYPE_INT: types[w[i + 1]] = ('int', w[i + 3])                                                 # signedness
        elif op == OP_TYPE_FLOAT: types[w[i + 1]] = ('float', w[i + 2])                                             # width
        elif op == OP_TYPE_VECTOR: types[w[i + 1]] = ('vec', w[i + 2], w[i + 3])
        elif op == OP_TYPE_SAMPLED_IMAGE: types[w[i + 1]] = ('sampled', w[i + 2])
        elif op == OP_TYPE_STRUCT: types[w[i + 1]] = ('struct',)
        elif op in (OP_TYPE_ARRAY, OP_TYPE_RTARRAY): types[w[i + 1]] = ('array', w[i + 2])
        elif op == OP_TYPE_POINTER: ptrs[w[i + 1]] = (w[i + 2], w[i + 3])
        elif op == OP_VARIABLE and w[i + 1] in ptrs: vars_[w[i + 2]] = ptrs[w[i + 1]]
        i += n
    return names, dec, blocks, types, vars_, i, caps, exts


def desc_type(var, vars_, types, blocks):
    sc, t = vars_[var]
    while types.get(t, ('',))[0] == 'array': t = types[t][1]
    k = types.get(t, ('',))
    if sc == SC_STORAGE_BUFFER or (sc == SC_UNIFORM and blocks.get(t) == DEC_BUFFER_BLOCK): return 'b'
    if sc == SC_UNIFORM: return 'u'
    if k[0] == 'sampled':
        im = types.get(k[1], ('', 0, 0, 0, 0, 0))
        if im[1] == DIM_BUFFER: return 't'
        st = types.get(im[5], ('float',)); ct = 'f' if st[0] == 'float' else 'i' if st[1] else 'u'
        return 'c' + DIMC.get(im[1], '?') + ('a' if im[4] else '') + ('s' if im[3] == 1 else '') + ct
    if k[0] == 'image':
        if k[1] == DIM_BUFFER: return 'j' if k[2] == 2 else 't'   # sampled 1: a texel buffer, declared as the image itself (no OpTypeSampledImage of Dim Buffer)
        if k[2] != 2: return 'c'
        st = types.get(k[5], ('float',)); ct = 'f' if st[0] == 'float' else 'i' if st[1] else 'u'
        return 'i' + DIMC.get(k[1], '?') + ('a' if k[4] else '') + ct
    return '?'


def kind_of(nm, stage, sb_sorted):
    if nm == 'support_buffer': return 'sup'
    m = re.fullmatch(r'[a-z]+_c(\d+)', nm)
    if m:
        n = int(m.group(1))
        if n == PUSH_BANK: return 'push'
        return f'ub:{n - 3}' if 3 <= n <= 16 else 'c1' if n == 1 else f'drv:{n}'
    m = re.fullmatch(r'[a-z]+_s(\d+)', nm)
    if m:
        k = int(m.group(1)); b = sb_sorted[k] if k < len(sb_sorted) else None
        if b and b['cb_slot'] >= 3: return f"sbu:{b['cb_slot'] - 3}:{b['cb_offset'] * 4}"
        off = b['cb_offset'] * 4 if b and b['cb_slot'] == 0 else None; base = SSBO.get(stage)
        if stage == 'cs' and off is not None and 0x210 <= off < 0x310 and (off - 0x210) % 16 == 0: return f'ubp:{(off - 0x210) // 16}'   # as dmsh_build.py reflect(): 'ubo_ptr'
        if off is not None and base is not None and 0 <= off - base < 0x100 and (off - base) % 16 == 0: return f'sb:{(off - base) // 16}'
        return '?'
    d = decode_ubo_tex(nm)
    if d: return ':'.join(map(str, d))
    d = decode_tex(nm)
    if d and None not in d: return ':'.join(map(str, d))
    return '?'


_packs = {}
def _record(pkg, name, idx):
    if name not in _packs: _packs[name] = Pack(os.path.join(pkg, name))
    return _packs[name][idx]


def one(args):
    """A shader -> (key, stage, row, c1, zlib, raw length, stats) or (key, stage, None, error)"""
    key, stg, name, idx, pkg, spv_dir = args
    p = os.path.join(spv_dir, key + '.spv')
    if not os.path.exists(p): return key, stg, None, 'no spv'
    raw = open(p, 'rb').read()
    w = array.array('I'); w.frombytes(raw)
    if sys.byteorder != 'little': w.byteswap()
    res = json.load(open(os.path.join(spv_dir, key + '.res.json'), encoding='utf-8'))
    names, dec, blocks, types, vars_, first_fn, caps, exts = parse(w)
    st = SET[stg]
    f1 = set(res.get('fetch1d', ()))   # texture slots the translator read as wrapped 2D rows
    for v, d in dec.items():
        if DEC_SET in d: w[d[DEC_SET][1]] = st
    sb_sorted = sorted(res.get('sbufs', []), key=lambda b: b['binding'])
    img_writes = {b['binding']: 1 if b.get('flags', 0) & 4 else 0 for b in res.get('images', [])}   # TextureUsageFlags.ImageStore
    binds = []
    for v, d in dec.items():
        if DEC_BINDING not in d or v not in vars_: continue
        ty = desc_type(v, vars_, types, blocks); nm = names.get(v, ''); k = kind_of(nm, stg, sb_sorted)
        if k.startswith('im:'): k += f':{img_writes.get(d[DEC_BINDING][0], 1)}'
        binds.append((d[DEC_BINDING][0], ty, k, nm))
    binds.sort()
    extras = [f'f1:{b}' for b, ty, k, _ in binds if f1 and k.startswith('tx:') and ty[:1] == 'c' and int(k.split(':')[1]) in f1]
    io = []
    for v, d in dec.items():
        if DEC_LOCATION not in d or v not in vars_ or vars_[v][0] not in (SC_INPUT, SC_OUTPUT): continue
        t = vars_[v][1]; n = 1
        if types.get(t, ('',))[0] == 'vec': n = types[t][2]; t = types[t][1]
        ct = 'f' if types.get(t, ('',))[0] == 'float' else 'i' if types.get(t, ('', 0))[1] else 'u'
        io.append(('i' if vars_[v][0] == SC_INPUT else 'o') + f'{d[DEC_LOCATION][0]}:{ct}{n}')
    io.sort()
    # drop unused capabilities / extension (whole instructions, all before the first function; SPIR-V has no absolute offsets)
    o = w[:5]; i = 5
    while i < first_fn:
        op, n = w[i] & 0xffff, w[i] >> 16
        if not ((op == OP_CAPABILITY and w[i + 1] in DROP_CAPS) or (op == OP_EXTENSION and word_text(w, i + 1, i + n) in DROP_EXT)): o.extend(w[i:i + n])
        i += n
    o.extend(w[first_fn:])
    spv = o.tobytes()
    c1 = split_record(_record(pkg, name, idx))[1] or b''
    row = f"{stg}\t{st}\t{' '.join(f'{b}:{t}:{k}' for b, t, k, _ in binds)}\t{' '.join(io)}\t{' '.join(extras)}".encode()
    kinds = collections.Counter(t[:1] + ' ' + k.split(':')[0] for b, t, k, _ in binds)
    unk = [re.sub(r'[0-9A-F]{2,}$', 'X', nm) for b, t, k, nm in binds if k == '?']
    stats = {'nb': len(binds), 'nin': sum(1 for x in io if x[0] == 'i'), 'kinds': kinds, 'unk': unk, 'caps': sorted(c for c in caps if c not in DROP_CAPS), 'exts': sorted(e for e in exts if e not in DROP_EXT),
             'f64': any(t[0] == 'float' and t[1] == 64 for t in types.values()), 'f1': bool(extras), 'arrays': sum(1 for v, d in dec.items() if DEC_BINDING in d and v in vars_ and types.get(vars_[v][1], ('',))[0] == 'array')}
    return key, stg, row, c1, zlib.compress(spv, 6), len(spv), stats, spv if os.environ.get('SPIRV_VAL') and int(key[:8], 16) % max(int(os.environ.get('VAL_SAMPLE', 0)) or 1, 1) == 0 and os.environ.get('VAL_SAMPLE') else None


def main(profile, pkg, spv_dir, out):
    os.makedirs(out, exist_ok=True)
    rows = read_map(profile)
    try: summary = json.load(open(os.path.join(spv_dir, 'summary.json'), encoding='utf-8'))
    except (OSError, ValueError): raise SystemExit(f'pokopia_vkshaders.py: no usable summary.json in {spv_dir}: run pokopia_spirv.py first')
    jobs = {}
    for name, idx, stg, key in rows:
        if key not in jobs: jobs[key] = (key, stg, name, idx, pkg, spv_dir)
    only = None
    if os.environ.get('KEYS'): only = {l.strip() for l in open(os.environ['KEYS'], encoding='utf-8') if l.strip()}
    todo = [j for k, j in sorted(jobs.items()) if summary.get(k, {}).get('st') == 'ok' and (only is None or k in only)]
    nospv = sorted(k for k in jobs if summary.get(k, {}).get('st') != 'ok')
    kinds, unk, caps, exts = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    most_b, most_i = (0, ''), (0, '')
    over, f64, val_bad = [], [], []
    missing = 0; nf1 = narr = 0; val_n = 0
    index = []; total_raw = 0
    val = os.environ.get('SPIRV_VAL')
    tmp = os.path.join(out, 'shaders.pack.new')
    with open(tmp, 'wb') as f, mp.Pool(int(os.environ.get('J', os.cpu_count() or 8))) as pool:
        f.write(b'PKSHPACK' + struct.pack('<IIQ', 1, 0, 0))
        for n_done, r in enumerate(pool.imap(one, todo, chunksize=16)):
            if not n_done % 8192 and n_done: print(f'progress: {n_done}/{len(todo)}', flush=True)
            key, stg = r[0], r[1]
            if r[2] is None: missing += 1; nospv.append(key); continue
            row, c1, z, raw_len, st, spv = r[2:]
            off = f.tell(); f.write(row); f.write(c1); f.write(z)
            index.append((int(key, 16), off, len(row), len(c1), len(z), raw_len)); total_raw += raw_len
            most_b, most_i = max(most_b, (st['nb'], key)), max(most_i, (st['nin'], key))
            if st['nb'] > MAX_BINDINGS or st['nin'] > MAX_INPUTS: over.append(f"{key} ({stg}): {st['nb']} bindings, {st['nin']} inputs")
            kinds.update(st['kinds']); unk.update(st['unk']); nf1 += st['f1']; narr += st['arrays']
            for c in st['caps']: caps[(stg, c)] += 1
            for e in st['exts']: exts[e] += 1
            if st['f64']: f64.append(key)
            if spv:
                val_n += 1
                with tempfile.NamedTemporaryFile(suffix='.spv', delete=False) as t: t.write(spv)
                v = subprocess.run([val, '--target-env', 'vulkan1.3', t.name], capture_output=True, text=True, errors='replace'); os.unlink(t.name)
                if v.returncode:
                    msg = (v.stderr or v.stdout).strip().splitlines()   # spirv-val can die with no message
                    val_bad.append((key, msg[0][:160] if msg else f'no message, exit code {v.returncode & 0xffffffff:#x}'))
        index.sort()
        at = f.tell()
        for e in index: f.write(struct.pack('<QQIIII', *e))
        f.seek(0); f.write(b'PKSHPACK' + struct.pack('<IIQ', 1, len(index), at))
    os.replace(tmp, os.path.join(out, 'shaders.pack'))
    size = os.path.getsize(os.path.join(out, 'shaders.pack'))
    print(f'shaders {len(index)}, without SPIR-V {len(nospv)}; SPIR-V {total_raw / 2**30:.2f} GiB, pack {size / 2**30:.2f} GiB; with a 1D integer fetch read as 2D rows (f1): {nf1}; arrayed bindings: {narr}')
    print('binding kinds:', ', '.join(f'{k} {v}' for k, v in kinds.most_common()))
    print(f'largest shader: {most_b[0]} bindings ({most_b[1]}), {most_i[0]} inputs ({most_i[1]}); the renderer keeps at most {MAX_BINDINGS} and {MAX_INPUTS}')
    if unk: print('undecoded names:', unk.most_common(10))
    print('capabilities:', ', '.join(f'{c} {sum(v for k, v in caps.items() if k[1] == c)}' for c in sorted({k[1] for k in caps})))
    print('extensions:', ', '.join(f'{k} {v}' for k, v in exts.most_common()))
    if val_n: print(f'spirv-val vulkan1.3: {val_n} checked, {len(val_bad)} failures'); [print('  ', k, e) for k, e in val_bad[:8]]
    if f64: print(f'[RED] vk: {len(f64)} shaders use 64-bit floats, which the Xclipse 920 lacks (for example {" ".join(f64[:4])})')
    if nospv: print(f'[RED] vk: {len(nospv)} of {len(jobs)} shaders have no valid SPIR-V and are left out (for example {" ".join(nospv[:4])})')
    if over: print(f'[RED] vk: {len(over)} shaders have more bindings or inputs than the renderer keeps ({MAX_BINDINGS} / {MAX_INPUTS}), for example ' + '; '.join(over[:3]))
    if not index: raise SystemExit('pokopia_vkshaders.py: no shader has valid SPIR-V')


if __name__ == '__main__':
    if len(sys.argv) != 5: raise SystemExit(__doc__)
    main(*sys.argv[1:])
