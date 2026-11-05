#!/usr/bin/env python3
"""Build Pokemon Pokopia's Android app and its game files on Windows from your own dump, one step at a time.

  python android-host/build_android.py --dump <folder with updates/655360/{exefs,romfs}> [--exefs <folder with the code modules>] [--steps profile,split,xlate,translate,pack,rewrite,font,app] [--install] [--push]

Steps (each is skipped while its inputs are unchanged; --force runs it again):
  profile    the game profile for this build                           tools/make_profile.py           -> .cache/profile/<title>/game.tsv
  split      the shaders out of the RomFS and main                     nxnative's shader_split.py      -> .cache/shaders/pkg, shader_map.tsv, shader_facts.tsv
  xlate      the shader translator, from nxnative's sources            tools/build_xlate.sh            -> .cache/tools/xlate
  translate  Ampere machine code to SPIR-V (about an hour)             android-host/tools/pokopia_spirv.py -> .cache/shaders/spirv
  pack       SPIR-V patched for the replay, one compressed file        android-host/tools/pokopia_vkshaders.py -> output/android/files/vk/shaders.pack
  rewrite    the code modules, for native execution on Android         nxnative's patch_nso.py         -> output/android/files/mods
  font       shared-font stand-ins (Noto Sans CJK JP, SIL OFL)         tools/cjk_shared_font.py, nxnative's make_ext_font.py -> output/android/files/font
  app        the Android host, packed and signed                       android-host/build.sh, build_apk.sh -> .cache/android/apk/pokopia.apk
Everything made from the game stays under .cache/ and output/ (both git-ignored) and on your phone; nothing is uploaded. The dump is only read.
--install installs the APK over adb; --push copies mods/, font/, vk/ and the RomFS to the app's folder on the phone (adb push --sync: a push cut short resumes when run again).

Needs vendor/nx (tools/setup_vendor.py) and, in the environment: ANDROID_SDK, the .NET 10 SDK's dotnet (DOTNET, or on PATH), a JDK on PATH, Git Bash.
Run it with the Python that has requirements.txt. J sets the parallel jobs. KEYSTORE sets the signing key (default: Pokopia Android/signing/debug.keystore in %LOCALAPPDATA%).
"""
import argparse, hashlib, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
NX = REPO / 'vendor' / 'nx'
TITLE = '0400A3A01EE4C000'
UPDATE = '655360'
PACKAGE = 'io.github.madebymochi.pokopia'
PHONE_DIR = f'/sdcard/Android/data/{PACKAGE}/files'
MODULES = ('main', 'subsdk0', 'sdk')       # profile/<title>/game.tsv.in "modules"
MOD_FILES = ('patched.bin', 'fixups.tsv', 'gates.tsv')
REWRITER_FILES = ('patch_nso.py', 'nso.py', 'nso_code.py', 'nso_asm.py', 'nso_match.py')   # android-host/build.sh hashes the same files in the same order
SDK = Path(os.environ.get('ANDROID_SDK') or os.environ.get('ANDROID_HOME') or '')
DOTNET = os.environ.get('DOTNET') or shutil.which('dotnet') or ''
NDK = SDK / 'ndk' / '28.2.13676358'
SPIRV_VAL = NDK / 'shader-tools' / 'windows-x86_64' / 'spirv-val.exe'
ADB = SDK / 'platform-tools' / 'adb.exe'
CACHE = REPO / '.cache'
SHADERS = CACHE / 'shaders'
FILES = REPO / 'output' / 'android' / 'files'
STAMPS = CACHE / 'stamps'
LOGS = CACHE / 'logs'
PY = Path(sys.executable)
ISOLATED = (PY, '-I', '-Xutf8')             # -I ignores PYTHONUTF8, and the tools write text that the Windows default cannot
KEYSTORE = Path(os.environ.get('KEYSTORE') or Path(os.environ.get('LOCALAPPDATA', REPO / '.cache')) / 'Pokopia Android' / 'signing' / 'debug.keystore')   # outside the checkout: an APK signed with another key cannot update the app
FONT_URL = 'https://github.com/notofonts/noto-cjk/raw/Sans2.004/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf'
FONT_SHA256 = '68a3fc98800b2a27b371f2fb79991daf3633bd89309d4ffaa6946fd587f375b5'   # 16467736 bytes
FONT_TOOL = REPO / 'tools' / 'cjk_shared_font.py'


def say(msg): print(f'[build_android] {msg}', flush=True)


def die(msg): sys.exit(f'[build_android] error: {msg}')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def key_of(*parts):
    """What a step was made from: file contents (small files), or names, sizes and times (folders and big files)"""
    h = hashlib.sha256()
    for p in parts:
        p = Path(p) if not isinstance(p, str) else p
        if isinstance(p, str): h.update(p.encode()); continue
        if p.is_dir():
            for q in sorted(x for x in p.rglob('*') if x.is_file()):
                s = q.stat(); h.update(f'{q.relative_to(p)}:{s.st_size}:{s.st_mtime_ns}\n'.encode())
        elif p.stat().st_size < 1 << 24: h.update(p.read_bytes().replace(b'\r', b''))
        else: s = p.stat(); h.update(f'{p.name}:{s.st_size}:{s.st_mtime_ns}\n'.encode())
    return h.hexdigest()


def step(name, key, force, fn):
    """Runs fn unless the stamp says these inputs were built already"""
    STAMPS.mkdir(parents=True, exist_ok=True)
    st = STAMPS / f'{name}.key'
    if not force and st.is_file() and st.read_text().strip() == key: say(f'{name}: up to date'); return
    st.unlink(missing_ok=True)
    t = time.monotonic(); say(f'{name}: running'); fn()
    st.write_text(key + '\n'); say(f'{name}: done in {time.monotonic() - t:.0f} s')


def run(cmd, log, env=None):
    """A build command, its output in a log file under .cache/logs (the logs stay local: they can name paths and keys); stops at a failure"""
    LOGS.mkdir(parents=True, exist_ok=True)
    e = dict(os.environ, PYTHONUTF8='1', **(env or {}))
    with open(LOGS / log, 'w', encoding='utf-8', errors='replace') as f:
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=e)
    tail = (LOGS / log).read_text(encoding='utf-8', errors='replace').splitlines()[-6:]
    for l in tail: print('    ' + l[:200], flush=True)
    if r.returncode: die(f'{next((Path(str(c)).name for c in cmd[1:] if not str(c).startswith("-")), cmd[0])} failed with exit code {r.returncode}: see {LOGS / log}')


def bash_exe():
    git_bash = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Git' / 'bin' / 'bash.exe'   # first: PATH can hold Windows' WSL bash
    return git_bash if git_bash.is_file() else shutil.which('bash') or die('Git Bash not found: install Git for Windows')


def tool_env():
    return {'DM_GAMES_DIR': str(CACHE / 'profile'), 'DM_GAMES_FIRST': '1', 'DM_GAME': TITLE, 'J': os.environ.get('J', str(os.cpu_count() or 8))}


def do_profile(exefs):
    run([*ISOLATED, REPO / 'tools' / 'make_profile.py', REPO / 'profile' / TITLE / 'game.tsv.in', exefs, CACHE / 'profile'], 'profile.log')


def do_xlate():
    run([bash_exe(), REPO / 'tools' / 'build_xlate.sh'], 'xlate.log', {'DOTNET': DOTNET.replace(chr(92), '/')})


def do_split(romfs, exefs):
    SHADERS.mkdir(parents=True, exist_ok=True)
    run([*ISOLATED, NX / 'dm/gfx/tools/shader_split.py', romfs, exefs / 'main', SHADERS / 'pkg', SHADERS / 'shader_map.tsv', SHADERS / 'shader_facts.tsv'], 'split.log', tool_env())
    prof = SHADERS / 'profile'; prof.mkdir(exist_ok=True)
    for n in ('shader_map.tsv', 'shader_facts.tsv'): shutil.copyfile(SHADERS / n, prof / n)


def do_translate(force):
    # the same translator as the last run: only shaders new to the map are translated, not all of them again (--force: all)
    stamp, k = STAMPS / 'translator.key', translator_key()
    same = not force and stamp.is_file() and stamp.read_text().strip() == k and ((SHADERS / 'spirv' / 'summary.json').is_file() or (SHADERS / 'spirv.new').is_dir())
    stamp.write_text(k + '\n')   # before the run: one that fails or is cut short goes on from where it stopped
    run([*ISOLATED, HERE / 'tools' / 'pokopia_spirv.py', SHADERS / 'profile', SHADERS / 'pkg', SHADERS / 'spirv'], 'translate.log',
        dict(tool_env(), DOTNET=DOTNET, DOTNET_CLI_HOME=str(CACHE / 'dotnet-home'), DOTNET_NOLOGO='1', DOTNET_CLI_TELEMETRY_OPTOUT='1', SPIRV_VAL=str(SPIRV_VAL), **({'XLATE_RESUME': '1'} if same else {})))


def translator_key():
    return key_of(CACHE / 'tools' / 'xlate' / 'Ryujinx.Graphics.Shader.dll', CACHE / 'tools' / 'xlate' / 'dmsh_xlate.dll')


def do_pack():
    run([*ISOLATED, HERE / 'tools' / 'pokopia_vkshaders.py', SHADERS / 'profile', SHADERS / 'pkg', SHADERS / 'spirv', FILES / 'vk'], 'pack.log', dict(tool_env(), SPIRV_VAL=str(SPIRV_VAL), VAL_SAMPLE='64'))


def rewriter_id():
    return hashlib.sha256(b''.join((NX / 'dm/tools' / f).read_bytes().replace(b'\r', b'') for f in REWRITER_FILES)).hexdigest()[:16]


def do_rewrite(exefs):
    out = CACHE / 'patched_android'; shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    env = dict(tool_env(), DM_TARGET='android', DM_REWRITE_J='1')   # the rewriter's workers are forked: one process on Windows
    for m in MODULES: run([PY, NX / 'dm/tools/patch_nso.py', exefs / m, out, m], f'rewrite_{m}.log', env)
    mods = FILES / 'mods'; mods.mkdir(parents=True, exist_ok=True)
    for m in MODULES:
        for s in MOD_FILES: shutil.copy2(out / f'{m}.{s}', mods / f'{m}.{s}')
    (mods / 'rewriter.txt').write_text(rewriter_id() + '  ' + ' '.join(REWRITER_FILES) + '\n', newline='\n')


def do_font():
    out = FILES / 'font'; out.mkdir(parents=True, exist_ok=True)
    cjk, ext = out / 'NotoSansCJKjp.ttf', out / 'nintendo_ext.ttf'
    otf = CACHE / 'font' / 'NotoSansCJKjp-Regular.otf'
    if not otf.is_file() or sha256_file(otf) != FONT_SHA256:
        otf.parent.mkdir(parents=True, exist_ok=True); say(f'font: downloading {FONT_URL} (16 MB, SIL Open Font License)')
        urllib.request.urlretrieve(FONT_URL, otf)
        if sha256_file(otf) != FONT_SHA256: otf.unlink(); die(f'the font from {FONT_URL} does not have SHA-256 {FONT_SHA256}')
    run([*ISOLATED, FONT_TOOL, otf, str(cjk) + '.tmp'], 'font.log'); os.replace(str(cjk) + '.tmp', cjk)
    run([PY, NX / 'dm/tools/make_ext_font.py', ext, Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / 'arialbd.ttf'], 'ext_font.log')


def do_app():
    env = {'ANDROID_SDK': str(SDK).replace('\\', '/'), 'J': os.environ.get('J', str(os.cpu_count() or 8)), 'PY': str(PY).replace('\\', '/')}
    bash = bash_exe()
    run([bash, HERE / 'build.sh', 'all'], 'app_build.log', env)
    KEYSTORE.parent.mkdir(parents=True, exist_ok=True)
    run([bash, HERE / 'build_apk.sh'], 'app_apk.log', dict(env, KEYSTORE=str(KEYSTORE).replace('\\', '/')))


class Adb:
    def run(self, *a, check=True, capture=False):
        r = subprocess.run([str(ADB), *a], capture_output=capture, text=capture, encoding='utf-8' if capture else None, errors='replace' if capture else None)
        if check and r.returncode: die(f'adb {" ".join(a[:2])} failed with exit code {r.returncode}')
        return r

    def sh(self, cmd): return self.run('shell', cmd, check=False, capture=True).stdout.strip()


def install(adb):
    apk = CACHE / 'android' / 'apk' / 'pokopia.apk'
    if not apk.is_file(): die(f'no {apk}: run the app step')
    if adb.sh(f'pidof {PACKAGE}'): die('the game is running on the phone: installing would stop it, and what is not saved would be lost. Close it, then run this again')
    adb.run('install', '-r', str(apk))


def open_app_once(adb):
    """A phone that never ran the app has no Android/data/<package>/files: adb would make it as the shell user. The app starts once and makes its own"""
    has = lambda: adb.sh(f'[ -d {PHONE_DIR} ] && echo y') == 'y'
    if has(): return
    say('push: starting the app once, so that it makes its own folder on the phone')
    out = adb.sh(f'am start -W -n {PACKAGE}/android.app.NativeActivity')
    if 'Error' in out: die(f'the app did not start on the phone: {out}')
    for _ in range(30):
        if has(): break
        time.sleep(1)
    else: say('push: warning: the app did not make its folder on the phone in 30 seconds (is the phone locked?), so adb makes it')
    adb.sh(f'am force-stop {PACKAGE}')


def push(adb, romfs):
    """adb push --sync skips files already there with the same size and time, so a push cut short resumes when run again"""
    if not adb.sh(f'pm path {PACKAGE}').startswith('package:'): die(f'{PACKAGE} is not installed on the phone: run with --install')
    open_app_once(adb)
    adb.sh(f'mkdir -p {PHONE_DIR}/root/romfs')
    for d in ('mods', 'font', 'vk'):
        say(f'push: {d}/'); adb.run('push', '--sync', str(FILES / d), PHONE_DIR + '/')
    say('push: root/romfs (6.7 GB; cut short? run again: it resumes)')
    for d in sorted(romfs.iterdir()): adb.run('push', '--sync', str(d), PHONE_DIR + '/root/romfs/')   # one top-level entry at a time: a single push of a whole RomFS once sat idle
    # adb makes its folders drwxrws--- for its own user and the shared-storage group, and the app is neither: it could not open mods/ and closed at once (Galaxy S22, Android 16). Opening them for reading lets it in
    say('push: letting the app read the copied folders and files')
    tree = f'cd {PHONE_DIR} && find mods font vk root'
    adb.sh(f'{tree} -type d ! -perm -005 -exec chmod a+rx {{}} + ; {tree} -type f ! -perm -004 -exec chmod a+r {{}} +')
    closed = adb.sh(f'{tree} -type d ! -perm -005 | wc -l'), adb.sh(f'{tree} -type f ! -perm -004 ! -user $(stat -c %U .) | wc -l')   # a file the app wrote itself, such as vk/probe_at.txt, is its own to read
    if closed != ('0', '0'): die(f'{closed[0]} copied folders and {closed[1]} copied files in {PHONE_DIR} stay closed to the app, which then cannot start: run this again')
    check_sizes(adb, romfs)


def check_sizes(adb, romfs):
    """Every copied file must have its size on the phone"""
    want = {f'{d}/{p.relative_to(FILES / d).as_posix()}': p.stat().st_size for d in ('mods', 'font', 'vk') for p in (FILES / d).rglob('*') if p.is_file()}
    want.update({f'root/romfs/{p.relative_to(romfs).as_posix()}': p.stat().st_size for p in romfs.rglob('*') if p.is_file()})
    listed = adb.sh(f"cd {PHONE_DIR} && find mods font vk root -type f -exec stat -c '%s %n' {{}} +")
    have = {name: int(size) for size, name in (line.split(' ', 1) for line in listed.splitlines() if ' ' in line and line.split(' ', 1)[0].isdigit())}
    wrong = [name for name, size in want.items() if have.get(name) != size]
    if wrong: die(f'{len(wrong)} of {len(want)} copied files are missing or have the wrong size on the phone, first {wrong[0]}: run this again')
    say(f'push: all {len(want)} copied files have the right size on the phone')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # the tools' log lines hold arrows, which the Windows default cannot print
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', type=Path, required=True, help='the dump folder, holding updates/655360/exefs and romfs')
    ap.add_argument('--exefs', type=Path, help='the code modules, when they are not in the dump folder')
    ap.add_argument('--steps', default='profile,split,xlate,translate,pack,rewrite,font,app')
    ap.add_argument('--force', action='store_true', help='run the named steps even when their inputs are unchanged')
    ap.add_argument('--install', action='store_true'); ap.add_argument('--push', action='store_true')
    a = ap.parse_args()
    exefs, romfs = a.exefs or a.dump / 'updates' / UPDATE / 'exefs', a.dump / 'updates' / UPDATE / 'romfs'
    if not all((exefs / m).is_file() for m in MODULES) or not romfs.is_dir(): die(f'{a.dump} has no updates/{UPDATE}/exefs modules and romfs folder')
    steps = a.steps.split(',')
    if not NX.is_dir(): die('no vendor/nx: run tools/setup_vendor.py')
    if not (SDK / 'ndk' / '28.2.13676358').is_dir(): die('set ANDROID_SDK to an Android SDK with ndk;28.2.13676358, build-tools;35.0.0, platforms;android-35 and platform-tools')
    if not DOTNET and {'xlate', 'translate'} & set(steps): die("set DOTNET to the .NET 10 SDK's dotnet.exe, or put it on PATH")
    if 'profile' in steps: step('profile', key_of(REPO / 'profile' / TITLE / 'game.tsv.in', REPO / 'tools' / 'make_profile.py', *(exefs / m for m in MODULES)), a.force, lambda: do_profile(exefs))
    if 'split' in steps: step('split', key_of(NX / 'dm/gfx/tools/shader_split.py', exefs / 'main', romfs), a.force, lambda: do_split(romfs, exefs))
    if 'xlate' in steps: step('xlate', key_of(NX / 'dm/gfx/tools/dmsh_xlate', NX / 'dm/third_party/ryujinx', REPO / 'tools' / 'build_xlate.sh'), a.force, do_xlate)
    if 'translate' in steps: step('translate', key_of(SHADERS / 'shader_map.tsv', CACHE / 'tools' / 'xlate' / 'Ryujinx.Graphics.Shader.dll', CACHE / 'tools' / 'xlate' / 'dmsh_xlate.dll', HERE / 'tools' / 'pokopia_spirv.py'), a.force, lambda: do_translate(a.force))
    if 'pack' in steps: step('pack', key_of(SHADERS / 'spirv' / 'summary.json', HERE / 'tools' / 'pokopia_vkshaders.py'), a.force, do_pack)
    if 'rewrite' in steps: step('rewrite', key_of(*(exefs / m for m in MODULES), *(NX / 'dm/tools' / f for f in REWRITER_FILES), rewriter_id()), a.force, lambda: do_rewrite(exefs))
    if 'font' in steps: step('font', key_of(FONT_TOOL, NX / 'dm/tools/make_ext_font.py'), a.force, do_font)
    if 'app' in steps: do_app()
    if a.install or a.push:
        adb = Adb()
        if a.install: install(adb)
        if a.push: push(adb, romfs)


if __name__ == '__main__':
    main()
