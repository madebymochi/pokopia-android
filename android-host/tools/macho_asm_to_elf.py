#!/usr/bin/env python3
"""Rewrite nxnative's Mach-O style AArch64 assembly (dm/host/entry.S, dm/gfx/nvn/*.S) into ELF syntax for the Android build.

Mach-O -> ELF: C symbols lose their leading underscore, `sym@PAGE` -> `sym`, `sym@PAGEOFF` -> `:lo12:sym`,
`.private_extern` -> `.hidden`, `.cstring` -> a mergeable string section, `.zerofill` -> .bss. Comments (// ...) are left alone.
Usage: macho_asm_to_elf.py <in.S> <out.S>
"""
import re, sys

SYM = r'_([A-Za-z_][A-Za-z0-9_]*)'


def code_part(line):
    i = line.find('//')
    return (line, '') if i < 0 else (line[:i], line[i:])


def convert(line):
    code, cmt = code_part(line)
    m = re.match(r'\s*\.zerofill\s+__DATA,__bss,_(\w+),(\d+),(\d+)\s*$', code)
    if m: return f'    .bss\n    .p2align {m[3]}\n{m[1]}:\n    .space {m[2]}\n    .text' + cmt
    code = re.sub(SYM + r'@PAGEOFF', r':lo12:\1', code)
    code = re.sub(SYM + r'@PAGE\b', r'\1', code)
    code = re.sub(r'(?<![\w.$:])' + SYM, r'\1', code)
    code = re.sub(r'^\s*\.private_extern\b', '    .hidden', code)
    code = re.sub(r'^\s*\.cstring\b', '    .section .rodata.str1.1,"aMS",@progbits,1', code)
    return code + cmt


def main(src, dst):
    out = [convert(l.rstrip('\n')) for l in open(src, encoding='utf-8')]
    open(dst, 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n' + '    .section .note.GNU-stack,"",@progbits\n')


if __name__ == '__main__':
    if len(sys.argv) != 3: raise SystemExit(__doc__)
    main(*sys.argv[1:])
