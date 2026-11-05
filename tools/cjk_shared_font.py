#!/usr/bin/env python3
"""Shared-font stand-in: a TrueType (glyf) copy of Noto Sans CJK JP (SIL Open Font License), subset to Latin / punctuation / kana / CJK / Hangul / fullwidth.

The game's ScalableFont looks up CJK and Hangul glyphs in the system's shared font: with a Latin-only font it crashed in FSS_map_char / FSS_get_advance,
and the CFF (OTF) original crashed at boot, hence the conversion to glyf outlines. android-host/build_android.py downloads the OTF and runs this.
Usage: cjk_shared_font.py <NotoSansCJKjp-Regular.otf> <out.ttf>
"""
import sys
from fontTools.ttLib import TTFont, newTable
from fontTools.subset import Subsetter, Options
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

src, dst = sys.argv[1], sys.argv[2]
f = TTFont(src)
ranges = [(0x20, 0x24F), (0x2000, 0x206F), (0x20A0, 0x20CF), (0x2100, 0x214F), (0x2190, 0x21FF), (0x2460, 0x24FF), (0x2500, 0x25FF), (0x2600, 0x26FF),
          (0x3000, 0x30FF), (0x3130, 0x318F), (0x31F0, 0x31FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xAC00, 0xD7A3), (0xFF00, 0xFFEF)]
cmap = f.getBestCmap()
uni = [u for u in cmap if any(a <= u <= b for a, b in ranges)]
o = Options(); o.layout_features = []; o.hinting = False; o.notdef_outline = True; o.name_IDs = ['*']
s = Subsetter(o); s.populate(unicodes=uni); s.subset(f)
order = f.getGlyphOrder(); gs = f.getGlyphSet()
glyf = newTable('glyf'); glyf.glyphOrder = order; glyf.glyphs = {}
for name in order:
    pen = TTGlyphPen(gs)
    gs[name].draw(Cu2QuPen(pen, max_err=1.0, reverse_direction=True))
    glyf.glyphs[name] = pen.glyph()
f['glyf'] = glyf
f['loca'] = newTable('loca')
for t in ('CFF ', 'CFF2'):
    if t in f: del f[t]
if 'VORG' in f: del f['VORG']
f['maxp'] = newTable('maxp'); m = f['maxp']; m.tableVersion = 0x00010000
for k in ('maxZones', 'maxTwilightPoints', 'maxStorage', 'maxFunctionDefs', 'maxInstructionDefs', 'maxStackElements', 'maxSizeOfInstructions', 'maxComponentElements'):
    setattr(m, k, 0)
m.maxZones = 1; m.numGlyphs = len(order)
f['head'].glyphDataFormat = 0; f['head'].indexToLocFormat = 1
f['post'].formatType = 3.0
f.sfntVersion = '\x00\x01\x00\x00'
f.save(dst)
print('glyphs', len(order), 'unicodes', len(uni))
