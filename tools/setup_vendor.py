#!/usr/bin/env python3
"""Make vendor/nx: nxnative at the pinned commit with this project's patch applied. The Android build compiles it.

Usage: setup_vendor.py [--fresh]
  Without --fresh, a vendor/nx made from this pin and this patch is kept. One made from anything else is cloned again.
  The new clone is made beside the old one and takes its place only when it is whole.
  Line endings stay as they are in the repository, and the user's own git hooks, signing and attributes are switched off.
"""
import hashlib, os, shutil, stat, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
URL = 'https://github.com/seedds/nxnative.git'
PIN = 'e5620b1a05'
PATCH = REPO / 'patches' / 'nxnative' / f'nxnative-{PIN[:8]}-local.patch'
STAMP = 'pokopia-vendor-stamp'   # in vendor/nx/.git
NX = REPO / 'vendor' / 'nx'
NO_PATH = REPO / 'vendor' / '.none'   # does not exist: git finds no hooks and no attributes there


def stamp():
    return hashlib.sha256(PIN.encode() + b'\0' + PATCH.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def git(*a, cwd=None):
    subprocess.run(['git', '-c', 'core.autocrlf=false', '-c', f'core.hooksPath={NO_PATH}', '-c', f'core.attributesFile={NO_PATH}',
                    '-c', 'commit.gpgsign=false', '-c', 'tag.gpgsign=false', *a], cwd=cwd, check=True)


def rm(p):
    shutil.rmtree(p, onerror=lambda f, path, _: (os.chmod(path, stat.S_IWRITE), f(path)))   # git's objects are read-only on Windows


def main():
    f = NX / '.git' / STAMP
    if NX.exists() and '--fresh' not in sys.argv and f.is_file() and f.read_text(encoding='ascii', errors='replace').strip() == stamp():
        print(f'{NX} was made from this pin and patch: kept (--fresh clones it again)')
        return
    new, old = NX.with_name('nx.new'), NX.with_name('nx.old')
    for d in (new, old):
        if d.exists(): rm(d)   # left over from a run that was cut short
    NX.parent.mkdir(parents=True, exist_ok=True)
    print(f'cloning {URL} (about 170 MB)', flush=True)
    git('clone', '-q', '--no-recurse-submodules', URL, str(new))
    git('config', 'core.autocrlf', 'false', cwd=new)
    git('checkout', '-q', PIN, cwd=new)
    tmp = new / '.git' / PATCH.name
    tmp.write_bytes(PATCH.read_bytes().replace(b'\r\n', b'\n'))
    git('apply', '--whitespace=nowarn', str(tmp), cwd=new)
    git('add', '-A', cwd=new)
    git('-c', 'user.name=setup_vendor', '-c', 'user.email=setup_vendor@localhost', 'commit', '-q', '-m', 'Android patch', cwd=new)
    (new / '.git' / STAMP).write_text(stamp() + '\n', encoding='ascii')   # last: a setup cut short has no stamp and is done again
    if NX.exists(): os.replace(NX, old)
    os.replace(new, NX)
    if old.exists(): rm(old)
    print(f'{NX}: nxnative {PIN} with {PATCH.name}')


if __name__ == '__main__':
    main()
