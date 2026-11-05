#!/usr/bin/env python3
"""Translate the shader package from nxnative's shader split into SPIR-V for the Android host's Vulkan replay (Pokopia: NVN2, Ampere machine code).

Jobs are made as nxnative's dm/gfx/tools/dmsh_build.py job_for() makes them for NVN2 (arch sm86: the instructions, control +0x38 long;
the c1 segment between control +0x40 and +0x44; compute's driver constant layout), translated by `dotnet dmsh_xlate.dll`, and each output is checked with the NDK's spirv-val (Vulkan 1.3 target).
Decisions that differ from the Metal build:
  * geometry shaders are translated like any other stage, without "geometry_topology": that field makes dmsh_xlate lower the shader to a compute kernel for Metal.
  * hull and domain shaders are translated too (the replay binds them on Vulkan); a domain shader's patch type, spacing and winding come from its control section (job "tess").
  * the translator keeps TranslationFlags.MetalImageAtomics (Program.cs hard-codes it): an image atomic gets a bounds guard, and a shader whose image atomic is not on a 32-bit R32
    buffer / 2D / 2D-array / 3D image, or uses TRAP or a predicate result, does not translate: counted as such below.

Usage: pokopia_spirv.py <profile dir> <shader pkg dir> <output dir>
Env: J=parallel jobs (default cpu count), SPIRV_VAL=spirv-val (optional), DOTNET=dotnet executable, XLATE_DLL=the translator (default .cache/tools/xlate), SAMPLE=<n>: only the n shaders of each stage
with the lowest keys (a measurement; keys are hashes, so this is an even sample), XLATE_DEBUG=0: the translator's debug mode off, so its AST optimizer runs (default: debug mode on, as nxnative's
dmsh_build.py; with the optimizer 591 of the first 1,657 of Pokopia's shaders fail inside the translator, so it stays off), XLATE_RESUME=1: continue from <out>.new, or from <out> when only that exists,
translating only the keys with no output, the stages listed in XLATE_REDO (comma separated, for example cs) and the keys listed in the file XLATE_REDO_KEYS (one per line), and checking only those plus any key the earlier summary.json does not call ok.
Output: <out>/<key>.spv, <key>.res.json, summary.json (status per key), and a line summary on stdout. Everything is translated into a fresh <out>.new that replaces <out> at the end, so no file of an
earlier run can pass as translated (with XLATE_RESUME, only the files of keys it does not translate again are kept). A shader the translator rejects with a [RED] line is only reported; exit code 1 when
no shader translated at all, 3 when a translator process crashed.
"""
import collections, concurrent.futures as cf, json, mmap, os, re, shutil, struct, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XLATE_DLL = os.environ.get('XLATE_DLL') or os.path.join(ROOT, '.cache', 'tools', 'xlate', 'dmsh_xlate.dll')
CTL_MAGIC = bytes.fromhex('34127698')
STAGE = {'vertex': 'vs', 'pixel': 'ps', 'compute': 'cs', 'geometry': 'gs', 'hull': 'hs', 'domain': 'ds'}   # shader_map.tsv column 3 (shader_split.py STAGE_NAME) -> the package file name prefix
CHUNK = 128   # shaders per translator process: a crash loses only these
TESS_TYPE = {4: 'triangles', 7: 'quads', 0x8e7a: 'isolines'}                                   # GL enums in a domain shader's control section
TESS_SPACING = {0x202: 'equal', 0x8e7b: 'odd', 0x8e7c: 'even'}
TESS_CW = {0x900: True, 0x901: False}


class Pack:
    """A package file's records by index ('BTCDBTCD' + 0x30 header + code area + control section, shader_split.py main), mapped: Pokopia's pixel package is about 4 GB"""
    def __init__(self, path):
        self.f = open(path, 'rb'); self.d = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.at = [m.start() for m in re.finditer(b'BTCDBTCD', self.d)] + [len(self.d)]

    def __getitem__(self, i): return self.d[self.at[i]:self.at[i + 1]]


def read_map(profile):
    """shader_map.tsv rows as (package file, record index, stage, key)"""
    rows = []
    for l in open(os.path.join(profile, 'shader_map.tsv'), encoding='utf-8'):
        r = l.rstrip('\n').split('\t')
        if not l.strip() or l.startswith('#') or len(r) < 4 or not r[0].endswith('.bin'): continue
        rows.append((r[0], int(r[1]), STAGE[r[2]], r[3]))
    return rows


def split_record(rec):
    """A package record -> (instructions, c1 segment, control section); pokopia_vkshaders.py takes the same c1 from here, so the file next to the SPIR-V is the one the translator read"""
    ci = rec.find(CTL_MAGIC); code, ctl = rec[0x38:ci], rec[ci:]
    a, e = struct.unpack_from('<2I', ctl, 0x40)
    return code[:struct.unpack_from('<I', ctl, 0x38)[0]], code[a:e] if a <= e <= len(code) else b'', ctl


def job_for(key, stage, rec, w, facts):
    code, c1, ctl = split_record(rec)
    cp = os.path.join(w, key + '.bin'); open(cp, 'wb').write(code)
    j = {'key': key, 'stage': stage, 'code': cp, 'arch': 'sm86', 'tex': {}, 'img': {}, 'attr': {}, 'fout': {}, 'c1': c1.hex(),
         'debug': os.environ.get('XLATE_DEBUG', '1') != '0', 'reduced': True, 'subgroup': 64}   # the phone's default subgroup size, and also right when a stage runs 32 lanes
    # compute lays out its driver constants in c[0] per shader (dmsh_build.py job_for, InstEmitAmpere.cs AMapC0): uniform, storage, texture entry counts, the size, the push constant bytes
    if stage == 'cs': j['nvn2_cs'] = [ctl[0xb1], ctl[0xb2], ctl[0xb3], struct.unpack_from('<H', ctl, 0xba)[0], ctl[0xa9] * 4]
    if stage == 'ds':
        ty, sp, cw = struct.unpack_from('<2I', ctl, 0x58) + struct.unpack_from('<I', ctl, 0x64)
        if ty not in TESS_TYPE or sp not in TESS_SPACING or cw not in TESS_CW: raise ValueError(f'domain shader with control +0x58 {ty:#x}, +0x5c {sp:#x}, +0x64 {cw:#x}')
        j['tess'] = [TESS_TYPE[ty], TESS_SPACING[sp], TESS_CW[cw]]
    if key in facts: j.update(facts[key])
    return j


def move_folder(a, b):
    """Windows refuses to rename a folder while anything is walking it or has a file in it open (seen with a du over this big folder), so try often"""
    for _ in range(3000):
        try: return os.replace(a, b)
        except PermissionError: time.sleep(0.1)
    os.replace(a, b)


def main(profile, pkg, final):
    t0 = time.monotonic()
    out = os.path.normpath(final) + '.new'
    resume = os.environ.get('XLATE_RESUME') == '1'
    in_place = resume and not os.path.isdir(out) and os.path.isdir(final)   # a run cut short is safe: each shader's files are checked, and summary.json is only rewritten at the end
    if in_place: out = final
    if os.path.isdir(out) and not resume: shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    w = os.path.join(out, '_jobs'); os.makedirs(w, exist_ok=True)
    facts_tsv = os.path.join(profile, 'shader_facts.tsv')
    facts = {k: json.loads(v) for k, v in (l.rstrip('\n').split('\t', 1) for l in open(facts_tsv, encoding='utf-8') if l.strip() and not l.startswith('#'))} if os.path.exists(facts_tsv) else {}
    rows = read_map(profile)
    sample = int(os.environ.get('SAMPLE', 0))
    if sample:
        by = collections.defaultdict(list)
        for r in rows: by[r[2]].append(r)
        rows = [r for rr in by.values() for r in sorted(rr, key=lambda r: r[3])[:sample]]
    packs, status = {}, {}
    def pack(name):
        if name not in packs: packs[name] = Pack(os.path.join(pkg, name))
        return packs[name]
    todo = {}
    for name, idx, stg, key in rows:
        if key not in todo: todo[key] = (name, idx, stg)
    keys = sorted(todo)
    prev, run = {}, keys
    if resume:
        sp = os.path.join(out, 'summary.json')
        if os.path.exists(sp): prev = json.load(open(sp))
        redo = set(os.environ.get('XLATE_REDO', '').split(','))
        redo_keys = set(open(os.environ['XLATE_REDO_KEYS']).read().split()) if os.environ.get('XLATE_REDO_KEYS') else set()
        files = lambda k: [os.path.join(out, k + e) for e in ('.spv', '.res.json')]
        run = [k for k in keys if todo[k][2] in redo or k in redo_keys or not all(map(os.path.exists, files(k)))]
        for k in run:
            for p in files(k):
                if os.path.exists(p): os.remove(p)
        print(f'resume: translating {len(run)} of {len(keys)}', flush=True)
    ran = set(run)
    chunks = [run[i:i + CHUNK] for i in range(0, len(run), CHUNK)]
    nj = int(os.environ.get('J', os.cpu_count() or 8))
    dotnet = os.environ.get('DOTNET', 'dotnet')

    def xl(ch):
        js = []; codes = []
        for k in ch:
            name, idx, stg = todo[k]
            try: j = job_for(k, stg, pack(name)[idx], w, facts)
            except ValueError as e: status[k] = ('job failed', str(e), stg); continue
            js.append(j); codes.append(j['code'])
        if not js: return None
        jp = os.path.join(w, f'job_{ch[0]}.json'); json.dump(js, open(jp, 'w'))
        for attempt in range(3):   # a translator process can crash on one run and not on the next (an access violation in DecodeShader, 1 process of a full run, its shaders fine the next time)
            r = subprocess.run([dotnet, XLATE_DLL, jp, out], capture_output=True, text=True, errors='replace')
            if not r.returncode or re.search(r'^\[RED\] \w+: ', r.stderr, re.M): break
            print(f'translator process crashed, exit {r.returncode}, attempt {attempt + 1} of 3', flush=True)
        for c in codes + [jp]: os.remove(c)
        return r
    done = 0; xerr, crashes = {}, []
    with cf.ThreadPoolExecutor(nj) as ex:
        for r in ex.map(xl, chunks):
            done += 1
            if not done % 64: print(f'progress: {done}/{len(chunks)} chunks, {time.monotonic() - t0:.0f} s', flush=True)
            if r is None: continue
            reds = [m for m in (re.match(r'\[RED\] (\w+): (.*)', ln) for ln in r.stderr.splitlines()) if m]
            for m in reds: xerr[m.group(1)] = m.group(2)[:200]
            if r.returncode and not reds: crashes.append(f'exit {r.returncode}: ' + r.stderr[-400:])
    os.rmdir(w)
    val = os.environ.get('SPIRV_VAL')

    def check(key):
        spv = os.path.join(out, key + '.spv'); stg = todo[key][2]
        if key in status: return key, status[key]
        if key in xerr: return key, ('translation failed', xerr[key], stg)
        if not os.path.exists(spv) or not os.path.exists(os.path.join(out, key + '.res.json')): return key, ('no output', 'no spv / res.json and no reason from the translator', stg)
        if key not in ran and prev.get(key, {}).get('st') == 'ok': return key, ('ok', '', stg)
        if val:
            v = subprocess.run([val, '--target-env', 'vulkan1.3', spv], capture_output=True, text=True, errors='replace')
            if v.returncode:
                msg = (v.stderr or v.stdout).strip().splitlines()   # spirv-val can die with no message
                return key, ('spirv-val failed', msg[0][:200] if msg else f'no message, exit code {v.returncode & 0xffffffff:#x}', stg)
        return key, ('ok', '', stg)
    with cf.ThreadPoolExecutor(nj) as ex: status.update(dict(ex.map(check, keys)))
    st = collections.Counter(s[0] for s in status.values())
    by_stage = collections.defaultdict(collections.Counter)
    size = collections.Counter(); code = collections.Counter()
    for k, s in status.items():
        by_stage[s[2]][s[0]] += 1
        if s[0] == 'ok': size[s[2]] += os.path.getsize(os.path.join(out, k + '.spv'))
    print(f'shaders {len(status)}: ' + ', '.join(f'{k} {v}' for k, v in st.most_common()) + f'; {time.monotonic() - t0:.0f} s')
    for stg, c in sorted(by_stage.items()): print(f'  {stg}: ' + ', '.join(f'{k} {v}' for k, v in c.most_common()) + f'; SPIR-V {size[stg] / 2**20:.1f} MiB')
    nat = sum(1 for s in status.values() if s[0] == 'translation failed' and 'image atomics' in s[1])   # TranslationFlags.MetalImageAtomics: see the header
    if nat: print(f'[RED] shaders: {nat} did not translate because of the image atomic rules the translator keeps from the Metal build (message "Metal image atomics ...")')
    for lvl in [k for k in st if k != 'ok']:
        for e, n in collections.Counter(s[1] for s in status.values() if s[0] == lvl).most_common(16): print(f'  [{lvl}] {n:5d}  {e}')
    for c in crashes: print('[translator crash] ' + c)
    with open(os.path.join(out, 'summary.json'), 'w') as f: json.dump({k: {'st': s[0], 'err': s[1], 'stage': s[2]} for k, s in sorted(status.items())}, f, indent=0)
    for p in packs.values(): p.d.close(); p.f.close()
    if not in_place:
        if os.path.isdir(final): shutil.rmtree(final)
        move_folder(out, final)   # a failed run is swapped in too: its summary.json says what failed
    bad = sorted(k for k, s in status.items() if s[0] != 'ok'); silent = [k for k, s in status.items() if s[0] == 'no output']
    if bad: print(f'[RED] shaders: {len(bad)} of {len(status)} did not translate (for example {" ".join(bad[:4])}); summary.json lists them all and why')
    if silent: print(f'[RED] shaders: {len(silent)} have no output and no reason (for example {" ".join(silent[:3])}): left out')
    if crashes or len(bad) == len(status):
        print('[RED] pokopia_spirv.py: ' + '; '.join(([f'{len(crashes)} translator process(es) crashed (the shaders in their chunks have no output)'] if crashes else []) + (['no shader translated'] if len(bad) == len(status) else [])))
        sys.exit(1 if len(bad) == len(status) else 3)


if __name__ == '__main__':
    if len(sys.argv) != 4: raise SystemExit(__doc__)
    main(*sys.argv[1:])
