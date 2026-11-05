"""Write the game profile the host build and the shader tools read.

Usage: make_profile.py <template game.tsv.in> <update exefs folder> <output folder>

Fills in the build row from the module IDs in the dump's own NSO headers, so build IDs stay out of Git.
Writes <output folder>/<title id>/game.tsv only when it changes.
"""
import os
import sys

VERSION, DISPLAY = "v655360", "2.0.1"


def module_id(path):
    with open(path, "rb") as f:
        head = f.read(0x60)
    if head[:4] != b"NSO0":
        sys.exit(f"{path} is not an NSO")
    return head[0x40:0x50].hex()


def main():
    template, exefs, out = sys.argv[1:4]
    with open(template, encoding="utf-8") as f:
        text = f.read()
    modules = next(line.split("\t")[1].split() for line in text.splitlines() if line.startswith("modules\t"))
    row = "\t".join(["build", VERSION, DISPLAY] + [f"{m}={module_id(os.path.join(exefs, m))}" for m in modules])
    title = os.path.basename(os.path.dirname(os.path.abspath(template)))
    text = text.replace("@BUILD@", row)
    dst = os.path.join(out, title, "game.tsv")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    old = None
    if os.path.exists(dst):
        with open(dst, encoding="utf-8") as f:
            old = f.read()
    if old != text:
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print(os.path.dirname(dst))


if __name__ == "__main__":
    main()
