// BC7 decoding on the CPU (the Xclipse 920 samples BC1-BC3 only): one level's linear blocks → RGBA8 texels.
// Ported to C from SuperMarioWonderRecomp runtime/src/nvn/nvn_bc.cpp (MIT, Copyright (c) 2026 BelmanteGu), whose partition
// and anchor tables were checked there against a reference decoder (Pillow). BC6H is below, written from the format's description.
#include <math.h>
#include <stdint.h>
#include <string.h>

// Bit i (2 subsets) or bits 2i..2i+1 (3 subsets): the subset of pixel i
static const uint16_t kPart2[64] = {
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, 0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
    0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce, 0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
    0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a, 0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c, 0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
};
static const uint32_t kPart3[64] = {
    0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050, 0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090,
    0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250, 0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
    0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200, 0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424,
    0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50, 0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
    0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000, 0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000,
    0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
};
static const uint8_t kAnchor2[64] = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15};
static const uint8_t kAnchor3a[64] = {3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
    8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3};
static const uint8_t kAnchor3b[64] = {15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
    15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8};

typedef struct { uint8_t subsets, partition_bits, rotation_bits, index_select_bits, color_bits, alpha_bits, endpoint_pbits, shared_pbits, index_bits, index2_bits; } bc7_mode;
static const bc7_mode kModes[8] = {{3, 4, 0, 0, 4, 0, 1, 0, 3, 0}, {2, 6, 0, 0, 6, 0, 0, 1, 3, 0}, {3, 6, 0, 0, 5, 0, 0, 0, 2, 0}, {2, 6, 0, 0, 7, 0, 1, 0, 2, 0},
                                   {1, 0, 2, 1, 5, 6, 0, 0, 2, 3}, {1, 0, 2, 0, 7, 8, 0, 0, 2, 2}, {1, 0, 0, 0, 7, 7, 1, 0, 4, 0}, {2, 6, 0, 0, 5, 5, 1, 0, 2, 0}};
static const uint8_t kW2[4] = {0, 21, 43, 64}, kW3[8] = {0, 9, 18, 27, 37, 46, 55, 64}, kW4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};
static const uint8_t *weights(uint32_t bits) { return bits == 2 ? kW2 : bits == 3 ? kW3 : kW4; }

// Reads 8 bytes at a time, so it needs 8 bytes past any bit it reads: the block decoders below copy a block to a longer buffer first
typedef struct { const uint8_t *b; uint32_t pos; } bitrd;
static inline uint32_t rd(bitrd *r, uint32_t n) { uint64_t w; memcpy(&w, r->b + (r->pos >> 3), 8); w >>= r->pos & 7; r->pos += n; return (uint32_t)(w & ((1ull << n) - 1)); }

// One 16-byte block → 4x4 RGBA8 texels at out (rows stride_px texels apart)
static void bc7_block(const uint8_t *block, uint8_t *out, uint32_t stride_px) {
    uint32_t mode = 0;
    while (mode < 8 && !(block[0] >> mode & 1)) mode++;
    if (mode == 8) { for (int y = 0; y < 4; y++) memset(out + y * stride_px * 4, 0, 16); return; }   // reserved: transparent black
    const bc7_mode *m = &kModes[mode];
    uint8_t blk[24] = {0}; memcpy(blk, block, 16);
    bitrd r = {blk, mode + 1};
    uint32_t part = rd(&r, m->partition_bits), rotation = rd(&r, m->rotation_bits), select = rd(&r, m->index_select_bits), ends = m->subsets * 2u;
    uint8_t ep[6][4] = {{0}}, pbit[6] = {0};
    for (uint32_t c = 0; c < 3; c++) for (uint32_t e = 0; e < ends; e++) ep[e][c] = (uint8_t)rd(&r, m->color_bits);
    for (uint32_t e = 0; e < ends; e++) ep[e][3] = (uint8_t)(m->alpha_bits ? rd(&r, m->alpha_bits) : 255);
    if (m->endpoint_pbits) for (uint32_t e = 0; e < ends; e++) pbit[e] = (uint8_t)rd(&r, 1);
    if (m->shared_pbits) for (uint32_t s = 0; s < m->subsets; s++) pbit[2 * s] = pbit[2 * s + 1] = (uint8_t)rd(&r, 1);
    int has_p = m->endpoint_pbits || m->shared_pbits;
    for (uint32_t e = 0; e < ends; e++) for (uint32_t c = 0; c < 4; c++) {
        uint32_t bits = c < 3 ? m->color_bits : m->alpha_bits;
        if (!bits) continue;   // alpha of an opaque mode: 255 already
        uint32_t v = ep[e][c];
        if (has_p) { v = (v << 1) | pbit[e]; bits++; }
        v <<= 8 - bits; ep[e][c] = (uint8_t)(v | (v >> bits));
    }
    uint32_t idx[16], idx2[16] = {0};
    for (uint32_t i = 0; i < 16; i++) {
        int anchor = i == 0 || (m->subsets == 2 && i == kAnchor2[part]) || (m->subsets == 3 && (i == kAnchor3a[part] || i == kAnchor3b[part]));
        idx[i] = rd(&r, m->index_bits - (anchor ? 1 : 0));
    }
    if (m->index2_bits) for (uint32_t i = 0; i < 16; i++) idx2[i] = rd(&r, m->index2_bits - (i == 0 ? 1 : 0));
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t s = m->subsets == 2 ? (kPart2[part] >> i & 1) : m->subsets == 3 ? (kPart3[part] >> (2 * i) & 3) : 0;
        const uint8_t *e0 = ep[2 * s], *e1 = ep[2 * s + 1];
        uint32_t ci = idx[i], ai = idx[i], cb = m->index_bits, ab = m->index_bits;
        if (m->index2_bits) {   // modes 4 and 5: a second index set, for alpha (or for colour when selected)
            ai = idx2[i]; ab = m->index2_bits;
            if (select) { uint32_t t = ci; ci = ai; ai = t; t = cb; cb = ab; ab = t; }
        }
        const uint8_t *wc = weights(cb), *wa = weights(ab);
        uint8_t *o = out + ((i >> 2) * stride_px + (i & 3)) * 4;
        for (uint32_t c = 0; c < 3; c++) o[c] = (uint8_t)(((64 - wc[ci]) * e0[c] + wc[ci] * e1[c] + 32) >> 6);
        o[3] = (uint8_t)(((64 - wa[ai]) * e0[3] + wa[ai] * e1[3] + 32) >> 6);
        if (rotation) { uint8_t t = o[3]; o[3] = o[rotation - 1]; o[rotation - 1] = t; }
    }
}

// bw x bh blocks (16 bytes each, row-major) → RGBA8 texels in rows of bw * 4 texels
void bc7_decode(const uint8_t *src, uint8_t *dst, uint32_t bw, uint32_t bh) {
    uint32_t pw = bw * 4;
    for (uint32_t by = 0; by < bh; by++) for (uint32_t bx = 0; bx < bw; bx++) bc7_block(src + ((uint64_t)by * bw + bx) * 16, dst + ((uint64_t)by * 4 * pw + bx * 4) * 4, pw);
}

// ---------- BC6H: 16-byte blocks of HDR colour, no alpha ----------
// The endpoint bits are scattered differently in each of the 14 modes. One entry reads n bits, lowest first, into component comp from its bit lsb, in the block's bit order
enum { C_RW, C_GW, C_BW, C_RX, C_GX, C_BX, C_RY, C_GY, C_BY, C_RZ, C_GZ, C_BZ };
typedef struct { uint8_t comp, lsb, n; } bc6_op;
#define BC6_BASE10 {C_RW, 0, 10}, {C_GW, 0, 10}, {C_BW, 0, 10}
#define BC6_HIGH(c) {c, 15, 1}, {c, 14, 1}, {c, 13, 1}, {c, 12, 1}, {c, 11, 1}, {c, 10, 1}
static const bc6_op kOp0[] = {{C_GY,4,1},{C_BY,4,1},{C_BZ,4,1}, BC6_BASE10, {C_RX,0,5},{C_GZ,4,1},{C_GY,0,4},{C_GX,0,5},{C_BZ,0,1},{C_GZ,0,4},{C_BX,0,5},{C_BZ,1,1},{C_BY,0,4},{C_RY,0,5},{C_BZ,2,1},{C_RZ,0,5},{C_BZ,3,1}};
static const bc6_op kOp1[] = {{C_GY,5,1},{C_GZ,4,1},{C_GZ,5,1},{C_RW,0,7},{C_BZ,0,1},{C_BZ,1,1},{C_BY,4,1},{C_GW,0,7},{C_BY,5,1},{C_BZ,2,1},{C_GY,4,1},{C_BW,0,7},{C_BZ,3,1},{C_BZ,5,1},{C_BZ,4,1},{C_RX,0,6},{C_GY,0,4},{C_GX,0,6},{C_GZ,0,4},{C_BX,0,6},{C_BY,0,4},{C_RY,0,6},{C_RZ,0,6}};
static const bc6_op kOp2[] = {BC6_BASE10, {C_RX,0,5},{C_RW,10,1},{C_GY,0,4},{C_GX,0,4},{C_GW,10,1},{C_BZ,0,1},{C_GZ,0,4},{C_BX,0,4},{C_BW,10,1},{C_BZ,1,1},{C_BY,0,4},{C_RY,0,5},{C_BZ,2,1},{C_RZ,0,5},{C_BZ,3,1}};
static const bc6_op kOp3[] = {BC6_BASE10, {C_RX,0,4},{C_RW,10,1},{C_GZ,4,1},{C_GY,0,4},{C_GX,0,5},{C_GW,10,1},{C_GZ,0,4},{C_BX,0,4},{C_BW,10,1},{C_BZ,1,1},{C_BY,0,4},{C_RY,0,4},{C_BZ,0,1},{C_BZ,2,1},{C_RZ,0,4},{C_GY,4,1},{C_BZ,3,1}};
static const bc6_op kOp4[] = {BC6_BASE10, {C_RX,0,4},{C_RW,10,1},{C_BY,4,1},{C_GY,0,4},{C_GX,0,4},{C_GW,10,1},{C_BZ,0,1},{C_GZ,0,4},{C_BX,0,5},{C_BW,10,1},{C_BY,0,4},{C_RY,0,4},{C_BZ,1,1},{C_BZ,2,1},{C_RZ,0,4},{C_BZ,4,1},{C_BZ,3,1}};
static const bc6_op kOp5[] = {{C_RW,0,9},{C_BY,4,1},{C_GW,0,9},{C_GY,4,1},{C_BW,0,9},{C_BZ,4,1},{C_RX,0,5},{C_GZ,4,1},{C_GY,0,4},{C_GX,0,5},{C_BZ,0,1},{C_GZ,0,4},{C_BX,0,5},{C_BZ,1,1},{C_BY,0,4},{C_RY,0,5},{C_BZ,2,1},{C_RZ,0,5},{C_BZ,3,1}};
static const bc6_op kOp6[] = {{C_RW,0,8},{C_GZ,4,1},{C_BY,4,1},{C_GW,0,8},{C_BZ,2,1},{C_GY,4,1},{C_BW,0,8},{C_BZ,3,1},{C_BZ,4,1},{C_RX,0,6},{C_GY,0,4},{C_GX,0,5},{C_BZ,0,1},{C_GZ,0,4},{C_BX,0,5},{C_BZ,1,1},{C_BY,0,4},{C_RY,0,6},{C_RZ,0,6}};
static const bc6_op kOp7[] = {{C_RW,0,8},{C_BZ,0,1},{C_BY,4,1},{C_GW,0,8},{C_GY,5,1},{C_GY,4,1},{C_BW,0,8},{C_GZ,5,1},{C_BZ,4,1},{C_RX,0,5},{C_GZ,4,1},{C_GY,0,4},{C_GX,0,6},{C_GZ,0,4},{C_BX,0,5},{C_BZ,1,1},{C_BY,0,4},{C_RY,0,5},{C_BZ,2,1},{C_RZ,0,5},{C_BZ,3,1}};
static const bc6_op kOp8[] = {{C_RW,0,8},{C_BZ,1,1},{C_BY,4,1},{C_GW,0,8},{C_BY,5,1},{C_GY,4,1},{C_BW,0,8},{C_BZ,5,1},{C_BZ,4,1},{C_RX,0,5},{C_GZ,4,1},{C_GY,0,4},{C_GX,0,5},{C_BZ,0,1},{C_GZ,0,4},{C_BX,0,6},{C_BY,0,4},{C_RY,0,5},{C_BZ,2,1},{C_RZ,0,5},{C_BZ,3,1}};
static const bc6_op kOp9[] = {{C_RW,0,6},{C_GZ,4,1},{C_BZ,0,1},{C_BZ,1,1},{C_BY,4,1},{C_GW,0,6},{C_GY,5,1},{C_BY,5,1},{C_BZ,2,1},{C_GY,4,1},{C_BW,0,6},{C_GZ,5,1},{C_BZ,3,1},{C_BZ,5,1},{C_BZ,4,1},{C_RX,0,6},{C_GY,0,4},{C_GX,0,6},{C_GZ,0,4},{C_BX,0,6},{C_BY,0,4},{C_RY,0,6},{C_RZ,0,6}};
static const bc6_op kOp10[] = {BC6_BASE10, {C_RX,0,10},{C_GX,0,10},{C_BX,0,10}};
static const bc6_op kOp11[] = {BC6_BASE10, {C_RX,0,9},{C_RW,10,1},{C_GX,0,9},{C_GW,10,1},{C_BX,0,9},{C_BW,10,1}};
static const bc6_op kOp12[] = {BC6_BASE10, {C_RX,0,8},{C_RW,11,1},{C_RW,10,1},{C_GX,0,8},{C_GW,11,1},{C_GW,10,1},{C_BX,0,8},{C_BW,11,1},{C_BW,10,1}};
static const bc6_op kOp13[] = {BC6_BASE10, {C_RX,0,4}, BC6_HIGH(C_RW), {C_GX,0,4}, BC6_HIGH(C_GW), {C_BX,0,4}, BC6_HIGH(C_BW)};
#define BC6_OPS(a) a, sizeof a / sizeof *a
typedef struct { uint8_t code, two, transformed, prec, dbits[3]; const bc6_op *ops; uint32_t nops; } bc6_mode;
static const bc6_mode kBc6[14] = {
    {0x00, 1, 1, 10, {5, 5, 5}, BC6_OPS(kOp0)}, {0x01, 1, 1, 7, {6, 6, 6}, BC6_OPS(kOp1)}, {0x02, 1, 1, 11, {5, 4, 4}, BC6_OPS(kOp2)}, {0x06, 1, 1, 11, {4, 5, 4}, BC6_OPS(kOp3)},
    {0x0a, 1, 1, 11, {4, 4, 5}, BC6_OPS(kOp4)}, {0x0e, 1, 1, 9, {5, 5, 5}, BC6_OPS(kOp5)}, {0x12, 1, 1, 8, {6, 5, 5}, BC6_OPS(kOp6)}, {0x16, 1, 1, 8, {5, 6, 5}, BC6_OPS(kOp7)},
    {0x1a, 1, 1, 8, {5, 5, 6}, BC6_OPS(kOp8)}, {0x1e, 1, 0, 6, {6, 6, 6}, BC6_OPS(kOp9)}, {0x03, 0, 0, 10, {10, 10, 10}, BC6_OPS(kOp10)}, {0x07, 0, 1, 11, {9, 9, 9}, BC6_OPS(kOp11)},
    {0x0b, 0, 1, 12, {8, 8, 8}, BC6_OPS(kOp12)}, {0x0f, 0, 1, 16, {4, 4, 4}, BC6_OPS(kOp13)},
};
static int32_t bc6_sext(int32_t v, uint32_t bits) { return (int32_t)((uint32_t)v << (32 - bits)) >> (32 - bits); }
static int32_t bc6_unquant(int32_t v, uint32_t prec, int is_signed) {
    if (!is_signed) {
        if (prec >= 15 || v == 0) return v;
        return v == (1 << prec) - 1 ? 0xffff : (int32_t)((((uint32_t)v << 16) + 0x8000) >> prec);
    }
    if (prec >= 16) return v;
    int32_t m = v < 0 ? -v : v, u = m == 0 ? 0 : m >= (1 << (prec - 1)) - 1 ? 0x7fff : (int32_t)((((uint32_t)m << 15) + 0x4000) >> (prec - 1));
    return v < 0 ? -u : u;
}
static uint16_t bc6_finish(int32_t v, int is_signed) {   // to half float bits: magnitude * 31 / 64 unsigned, * 31 / 32 signed
    if (!is_signed) return (uint16_t)((v * 31) >> 6);
    if (v >= 0) return (uint16_t)((v * 31) >> 5);
    int32_t m = (-v * 31) >> 5; return m ? (uint16_t)(0x8000 | m) : 0;   // not -0
}
// One 16-byte block → 4x4 RGBA16F texels at out (rows stride_px texels apart); alpha 1. A reserved mode decodes to opaque black
static void bc6_block(const uint8_t *block, uint16_t *out, uint32_t stride_px, int is_signed) {
    uint8_t blk[24] = {0}; memcpy(blk, block, 16);
    bitrd r = {blk, 0};
    uint32_t code = rd(&r, 2); if (code > 1) code |= rd(&r, 3) << 2;
    const bc6_mode *m = NULL; for (int i = 0; i < 14; i++) if (kBc6[i].code == code) m = &kBc6[i];
    if (!m) { for (uint32_t i = 0; i < 16; i++) { uint16_t *o = out + ((i >> 2) * stride_px + (i & 3)) * 4; o[0] = o[1] = o[2] = 0; o[3] = 0x3c00; } return; }
    int32_t c[12] = {0};
    for (uint32_t i = 0; i < m->nops; i++) c[m->ops[i].comp] |= (int32_t)(rd(&r, m->ops[i].n) << m->ops[i].lsb);
    uint32_t part = m->two ? rd(&r, 5) : 0, ends = m->two ? 4 : 2, prec = m->prec;
    for (uint32_t e = 0; e < ends; e++) for (uint32_t ch = 0; ch < 3; ch++) {
        int32_t *v = &c[e * 3 + ch];
        if (e == 0) { if (is_signed) *v = bc6_sext(*v, prec); }
        else {
            if (m->transformed || is_signed) *v = bc6_sext(*v, m->dbits[ch]);
            if (m->transformed) { *v = (*v + c[ch]) & ((1 << prec) - 1); if (is_signed) *v = bc6_sext(*v, prec); }
        }
    }
    for (uint32_t i = 0; i < ends * 3; i++) c[i] = bc6_unquant(c[i], prec, is_signed);
    const uint8_t *w = m->two ? kW3 : weights(4);
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t s = m->two ? (kPart2[part] >> i & 1) : 0, bits = m->two ? 3 : 4;
        if (i == 0 || (m->two && i == kAnchor2[part])) bits--;
        uint32_t x = rd(&r, bits);
        uint16_t *o = out + ((i >> 2) * stride_px + (i & 3)) * 4;
        for (uint32_t ch = 0; ch < 3; ch++) o[ch] = bc6_finish((c[s * 6 + ch] * (64 - w[x]) + c[s * 6 + 3 + ch] * w[x] + 32) >> 6, is_signed);
        o[3] = 0x3c00;
    }
}
// bw x bh blocks (16 bytes each, row-major) → RGBA16F texels in rows of bw * 4 texels
void bc6h_decode(const uint8_t *src, uint8_t *dst, uint32_t bw, uint32_t bh, int is_signed) {
    uint32_t pw = bw * 4;
    for (uint32_t by = 0; by < bh; by++) for (uint32_t bx = 0; bx < bw; bx++) bc6_block(src + ((uint64_t)by * bw + bx) * 16, (uint16_t *)(void *)(dst + ((uint64_t)by * 4 * pw + bx * 4) * 8), pw, is_signed);
}

// A half float that is not negative, inf or NaN, as an integer with 24 fractional bits: it fits exactly, up to bit 40
static uint64_t half_fix24(uint16_t h) { uint32_t e = h >> 10, m = h & 0x3ff; return e ? (uint64_t)(1024 + m) << (e - 1) : m; }
// Three of them as E5B9G9R9. A mantissa stands for mant * 2^(E - 24), so in these integers mant = v >> E, rounded. E is the smallest that fits the largest channel in 9 bits
uint32_t bc6_rgb9e5(uint16_t r, uint16_t g, uint16_t b) {
    uint64_t vr = half_fix24(r), vg = half_fix24(g), vb = half_fix24(b), top = vr > vg ? vr : vg;
    if (vb > top) top = vb;
    if (!top) return 0;
    uint32_t bits = 64u - (uint32_t)__builtin_clzll(top), e = bits > 9 ? bits - 9 : 0;
    uint64_t half = e ? 1ull << (e - 1) : 0;
    if (((top + half) >> e) > 511) { e++; half = 1ull << (e - 1); }
    if (e > 31) { e = 31; half = 1ull << 30; }   // the format's largest value is just under the largest half float's
    uint32_t mr = (uint32_t)((vr + half) >> e), mg = (uint32_t)((vg + half) >> e), mb = (uint32_t)((vb + half) >> e);
    if (mr > 511) mr = 511;
    if (mg > 511) mg = 511;
    if (mb > 511) mb = 511;
    return mr | mg << 9 | mb << 18 | e << 27;
}
// Unsigned BC6H blocks → E5B9G9R9 texels in rows of bw * 4 texels: half the memory of RGBA16F, and the format has no alpha to lose (BC6H's is always 1)
void bc6h_decode_e5(const uint8_t *src, uint8_t *dst, uint32_t bw, uint32_t bh) {
    uint32_t pw = bw * 4; uint16_t t[64];
    for (uint32_t by = 0; by < bh; by++) for (uint32_t bx = 0; bx < bw; bx++) {
        bc6_block(src + ((uint64_t)by * bw + bx) * 16, t, 4, 0);
        for (uint32_t y = 0; y < 4; y++) { uint8_t *row = dst + ((uint64_t)(by * 4 + y) * pw + bx * 4) * 4;
            for (uint32_t x = 0; x < 4; x++) { const uint16_t *h = t + (y * 4 + x) * 4; uint32_t px = bc6_rgb9e5(h[0], h[1], h[2]); memcpy(row + x * 4, &px, 4); } }
    }
}

// ---------- BC1 and BC3 encoding ----------
// This GPU samples BC1-BC3 but not BC7, and a BC7 texture decoded takes 4 bytes a texel. As BC3 it takes 1, and as BC1 half of that. Written from the formats' description: a line through a block's colours along their principal axis,
// then least squares on where its ends should be. Error is measured on the values as stored: an sRGB texture stays in its own space
static uint8_t g_om5[256][2], g_om6[256][2];   // for a block of one colour: the 5- and 6-bit ends whose mix of 2/3 and 1/3 comes nearest to an 8-bit value
static uint8_t g_h2s[0x7c00];   // a half float that is not negative, inf or NaN, as 8-bit sRGB: above 1 gives 255
static inline int ex5(uint32_t v) { return (int)(v << 3 | v >> 2); }
static inline int ex6(uint32_t v) { return (int)(v << 2 | v >> 4); }
void bc_enc_init(void) {
    static int done; if (done) return;
    done = 1;
    for (uint32_t h = 0; h < 0x7c00; h++) { uint32_t e = h >> 10, m = h & 0x3ff; double v = e ? ldexp(1024 + m, (int)e - 25) : ldexp(m, -24), s = v >= 1 ? 1 : v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1 / 2.4) - 0.055; g_h2s[h] = (uint8_t)(s * 255 + 0.5); }
    for (int v = 0; v < 256; v++) for (int six = 0; six < 2; six++) {
        int n = six ? 64 : 32, best = 1 << 30; uint8_t (*t)[2] = six ? g_om6 : g_om5;
        for (int a = 0; a < n; a++) for (int b = 0; b < n; b++) {
            int ea = six ? ex6((uint32_t)a) : ex5((uint32_t)a), eb = six ? ex6((uint32_t)b) : ex5((uint32_t)b);
            int mix = (2 * ea + eb + 1) / 3, d = mix > v ? mix - v : v - mix, gap = ea > eb ? ea - eb : eb - ea, e = d * 64 + gap;   // nearest first, then the ends closest together: GPUs round the mix differently
            if (e < best) { best = e; t[v][0] = (uint8_t)a; t[v][1] = (uint8_t)b; }
        }
    }
}
static void pal_of(uint32_t c0, uint32_t c1, int p[4][3]) {
    p[0][0] = ex5(c0 >> 11); p[0][1] = ex6(c0 >> 5 & 63); p[0][2] = ex5(c0 & 31);
    p[1][0] = ex5(c1 >> 11); p[1][1] = ex6(c1 >> 5 & 63); p[1][2] = ex5(c1 & 31);
    for (int k = 0; k < 3; k++) { p[2][k] = (2 * p[0][k] + p[1][k] + 1) / 3; p[3][k] = (p[0][k] + 2 * p[1][k] + 1) / 3; }
}
// The nearest of the four colours for each texel (RGBA8, 16 in a row). Returns the squared error
static uint32_t pick(const uint8_t *px, uint32_t c0, uint32_t c1, uint32_t *idx) {
    int p[4][3]; pal_of(c0, c1, p); uint32_t sse = 0, out = 0;
    for (int i = 0; i < 16; i++) { const uint8_t *t = px + i * 4; int best = 1 << 30, bk = 0;
        for (int k = 0; k < 4; k++) { int dr = t[0] - p[k][0], dg = t[1] - p[k][1], db = t[2] - p[k][2], e = dr * dr + dg * dg + db * db; if (e < best) { best = e; bk = k; } }
        sse += (uint32_t)best; out |= (uint32_t)bk << (2 * i); }
    *idx = out; return sse;
}
static inline int clamp8(float v) { return v <= 0 ? 0 : v >= 255 ? 255 : (int)(v + 0.5f); }
static inline uint32_t q565(int r, int g, int b) { return (uint32_t)((r * 31 + 127) / 255) << 11 | (uint32_t)((g * 63 + 127) / 255) << 5 | (uint32_t)((b * 31 + 127) / 255); }
// Least squares: the ends that fit these indices best. 0 if the indices leave them open
static int refine(const uint8_t *px, uint32_t idx, uint32_t *c0, uint32_t *c1) {
    static const int wa[4] = {3, 0, 2, 1}, wb[4] = {0, 3, 1, 2};   // how much of each end, in thirds
    int aa = 0, ab = 0, bb = 0, pa[3] = {0, 0, 0}, pb[3] = {0, 0, 0};
    for (int i = 0; i < 16; i++) { uint32_t k = idx >> (2 * i) & 3; int a = wa[k], b = wb[k]; aa += a * a; ab += a * b; bb += b * b;
        for (int c = 0; c < 3; c++) { pa[c] += a * px[i * 4 + c]; pb[c] += b * px[i * 4 + c]; } }
    int det = aa * bb - ab * ab; if (!det) return 0;
    float f = 3.0f / (float)det; int e0[3], e1[3];
    for (int c = 0; c < 3; c++) { e0[c] = clamp8((float)(pa[c] * bb - pb[c] * ab) * f); e1[c] = clamp8((float)(pb[c] * aa - pa[c] * ab) * f); }
    *c0 = q565(e0[0], e0[1], e0[2]); *c1 = q565(e1[0], e1[1], e1[2]);
    return 1;
}
static void enc_color(const uint8_t *px, uint8_t *out) {
    int mn[3] = {255, 255, 255}, mx[3] = {0, 0, 0}, sum[3] = {0, 0, 0};
    for (int i = 0; i < 16; i++) for (int k = 0; k < 3; k++) { int v = px[i * 4 + k]; if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v; sum[k] += v; }
    uint32_t c0, c1, idx;
    if (mn[0] == mx[0] && mn[1] == mx[1] && mn[2] == mx[2]) {
        c0 = (uint32_t)g_om5[mn[0]][0] << 11 | (uint32_t)g_om6[mn[1]][0] << 5 | g_om5[mn[2]][0];
        c1 = (uint32_t)g_om5[mn[0]][1] << 11 | (uint32_t)g_om6[mn[1]][1] << 5 | g_om5[mn[2]][1];
        idx = 0xaaaaaaaau;   // all at the mix of 2/3 and 1/3
    } else {
        float m[3] = {(float)sum[0] / 16, (float)sum[1] / 16, (float)sum[2] / 16}, cov[6] = {0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 16; i++) { float r = (float)px[i * 4] - m[0], g = (float)px[i * 4 + 1] - m[1], b = (float)px[i * 4 + 2] - m[2];
            cov[0] += r * r; cov[1] += r * g; cov[2] += r * b; cov[3] += g * g; cov[4] += g * b; cov[5] += b * b; }
        float ax[3] = {(float)(mx[0] - mn[0]), (float)(mx[1] - mn[1]), (float)(mx[2] - mn[2])};
        for (int it = 0; it < 4; it++) {   // power iteration towards the principal axis
            float x = ax[0] * cov[0] + ax[1] * cov[1] + ax[2] * cov[2], y = ax[0] * cov[1] + ax[1] * cov[3] + ax[2] * cov[4], z = ax[0] * cov[2] + ax[1] * cov[4] + ax[2] * cov[5];
            float l = x < 0 ? -x : x, t = y < 0 ? -y : y; if (t > l) l = t; t = z < 0 ? -z : z; if (t > l) l = t;
            if (l < 1e-3f) break;
            ax[0] = x / l; ax[1] = y / l; ax[2] = z / l;
        }
        float lo = 1e30f, hi = -1e30f; int il = 0, ih = 0;
        for (int i = 0; i < 16; i++) { float t = (float)px[i * 4] * ax[0] + (float)px[i * 4 + 1] * ax[1] + (float)px[i * 4 + 2] * ax[2]; if (t < lo) { lo = t; il = i; } if (t > hi) { hi = t; ih = i; } }
        const uint8_t *ph = px + ih * 4, *pl = px + il * 4;
        c0 = q565(ph[0], ph[1], ph[2]); c1 = q565(pl[0], pl[1], pl[2]);
        uint32_t best = pick(px, c0, c1, &idx);
        if (best) {   // rounded outward: colours less than a step apart would round to one end, and a gradient would lose its steps
            uint32_t a[3], b[3], ti;
            for (int k = 0; k < 3; k++) { uint32_t s = k == 1 ? 63 : 31, up = ax[k] >= 0;
                a[k] = ((uint32_t)ph[k] * s + (up ? 254u : 0u)) / 255; b[k] = ((uint32_t)pl[k] * s + (up ? 0u : 254u)) / 255; }
            uint32_t oc0 = a[0] << 11 | a[1] << 5 | a[2], oc1 = b[0] << 11 | b[1] << 5 | b[2], se = pick(px, oc0, oc1, &ti);
            if (se < best) { best = se; c0 = oc0; c1 = oc1; idx = ti; }
        }
        for (int pass = 0; pass < 2 && best; pass++) {
            uint32_t r0 = c0, r1 = c1, ti; if (!refine(px, idx, &r0, &r1) || (r0 == c0 && r1 == c1)) break;
            uint32_t se = pick(px, r0, r1, &ti); if (se >= best) break;
            best = se; c0 = r0; c1 = r1; idx = ti;
        }
    }
    if (c0 < c1) { uint32_t t = c0; c0 = c1; c1 = t; idx ^= 0x55555555u; }   // the first end larger: four colours, in BC1 too
    else if (c0 == c1) idx = 0;
    out[0] = (uint8_t)c0; out[1] = (uint8_t)(c0 >> 8); out[2] = (uint8_t)c1; out[3] = (uint8_t)(c1 >> 8); memcpy(out + 4, &idx, 4);
}
static void enc_alpha(const uint8_t *px, uint8_t *out) {
    int lo = 255, hi = 0; for (int i = 0; i < 16; i++) { int a = px[i * 4 + 3]; if (a < lo) lo = a; if (a > hi) hi = a; }
    out[0] = (uint8_t)hi; out[1] = (uint8_t)lo;
    uint64_t bits = 0;
    if (hi > lo) for (int i = 0; i < 16; i++) { int t = ((px[i * 4 + 3] - lo) * 14 + (hi - lo)) / (2 * (hi - lo));   // steps of 1/7 from lo to hi
        uint64_t code = t == 7 ? 0 : t == 0 ? 1 : (uint64_t)(8 - t); bits |= code << (3 * i); }
    memcpy(out + 2, &bits, 6);
}
// bw x bh BC7 blocks → BC3 blocks, or BC1 blocks for a texture with no alpha
void bc7_to_bc(const uint8_t *src, uint8_t *dst, uint32_t bw, uint32_t bh, int bc1) {
    uint8_t px[64]; uint32_t bs = bc1 ? 8 : 16;
    for (uint32_t by = 0; by < bh; by++) for (uint32_t bx = 0; bx < bw; bx++) {
        bc7_block(src + ((uint64_t)by * bw + bx) * 16, px, 4);
        uint8_t *o = dst + ((uint64_t)by * bw + bx) * bs;
        if (bc1) enc_color(px, o); else { enc_alpha(px, o); enc_color(px, o + 8); }
    }
}
// bw x bh unsigned BC6H blocks → BC1 blocks to be read as sRGB, so that sampling gives linear values again. For a texture whose values stay within 0 to 1 (bc6h_survey)
static void bc6_block_srgb(const uint8_t *block, uint8_t *px) {
    uint16_t t[64]; bc6_block(block, t, 4, 0);
    for (int i = 0; i < 16; i++) { for (int c = 0; c < 3; c++) px[i * 4 + c] = g_h2s[t[i * 4 + c]]; px[i * 4 + 3] = 255; }
}
void bc6h_to_bc1(const uint8_t *src, uint8_t *dst, uint32_t bw, uint32_t bh) {
    uint8_t px[64];
    for (uint32_t by = 0; by < bh; by++) for (uint32_t bx = 0; bx < bw; bx++) { bc6_block_srgb(src + ((uint64_t)by * bw + bx) * 16, px); enc_color(px, dst + ((uint64_t)by * bw + bx) * 8); }
}
// What values a BC6H texture holds, over up to 4,096 blocks spread over its memory: the largest channel, the share of texels with a channel above 1, and for an unsigned texture the root mean square error, of 255 in sRGB, that BC1 would have
void bc6h_survey(const uint8_t *mem, uint64_t size, int is_signed, float *top, float *over1, float *err) {
    uint64_t n = size / 16, step = n > 4096 ? n / 4096 : 1, cnt = 0, over = 0, sq = 0; uint16_t t[64]; uint32_t hi = 0; uint8_t px[64], c[8];
    for (uint64_t i = 0; i < n; i += step) {
        bc6_block(mem + i * 16, t, 4, is_signed);
        for (int k = 0; k < 16; k++) { uint32_t m = 0; for (int ch = 0; ch < 3; ch++) { uint32_t v = t[k * 4 + ch] & 0x7fff; if (v > m) m = v; }
            if (m > hi) hi = m;
            if (m > 0x3c00) over++;
            cnt++; }
        if (is_signed) continue;
        bc6_block_srgb(mem + i * 16, px); enc_color(px, c);
        uint32_t c0 = (uint32_t)c[0] | (uint32_t)c[1] << 8, c1 = (uint32_t)c[2] | (uint32_t)c[3] << 8, idx; memcpy(&idx, c + 4, 4); int p[4][3]; pal_of(c0, c1, p);
        for (int k = 0; k < 16; k++) { uint32_t q = idx >> (2 * k) & 3; for (int ch = 0; ch < 3; ch++) { int d = px[k * 4 + ch] - p[q][ch]; sq += (uint64_t)(d * d); } }
    }
    *top = hi ? ldexpf((float)(1024 + (hi & 0x3ff)), (int)(hi >> 10) - 25) : 0; *over1 = cnt ? (float)over / (float)cnt : 0; *err = cnt && !is_signed ? __builtin_sqrtf((float)sq / (float)(cnt * 3)) : 0;
}
// Whether a block can give anything but alpha 255. Modes 0 to 3 have no alpha; the others have it in their ends. A rotation swaps alpha with a colour channel: taken as not opaque. An all-zero block is padding between levels and counts as opaque
static int bc7_opaque(const uint8_t *b) {
    uint64_t lo, hi; memcpy(&lo, b, 8); memcpy(&hi, b + 8, 8);
    if (!(lo & 255)) return !lo && !hi;
    switch (__builtin_ctz((uint32_t)(lo & 255))) {
        case 4: return !(lo >> 5 & 3) && (lo >> 38 & 63) == 63 && (lo >> 44 & 63) == 63;
        case 5: return !(lo >> 6 & 3) && (lo >> 50 & 255) == 255 && ((lo >> 58 | hi << 6) & 255) == 255;
        case 6: return (lo >> 49 & 127) == 127 && (lo >> 56 & 127) == 127 && (lo >> 63) && (hi & 1);
        case 7: return (hi >> 10 & 0xfffff) == 0xfffff && (hi >> 30 & 15) == 15;
        default: return 1;
    }
}
// n blocks of a level, in a row: whether none gives alpha below 255
int bc7_blocks_opaque(const uint8_t *b, uint64_t n) { for (uint64_t i = 0; i < n; i++) if (!bc7_opaque(b + i * 16)) return 0; return 1; }
// What a BC7 texture's memory holds, read as blocks in any order, all levels and layers. err: root mean square error of 255 that BC1/BC3 would have, over up to 256 blocks spread over the memory. a_min: the smallest alpha in those blocks
typedef struct { float err_rgb, err_a; int a_min; } bc7_survey_t;
void bc7_survey(const uint8_t *mem, uint64_t size, bc7_survey_t *s) {
    uint64_t n = size / 16, step = n > 256 ? n / 256 : 1, rgb = 0, al = 0, cnt = 0;
    s->a_min = 255;
    for (uint64_t i = 0; i < n; i += step) {
        const uint8_t *b = mem + i * 16; uint64_t w0, w1; memcpy(&w0, b, 8); memcpy(&w1, b + 8, 8);
        if (!w0 && !w1) continue;
        uint8_t px[64], c[8], a[8]; bc7_block(b, px, 4); enc_color(px, c); enc_alpha(px, a);
        uint32_t c0 = (uint32_t)c[0] | (uint32_t)c[1] << 8, c1 = (uint32_t)c[2] | (uint32_t)c[3] << 8, idx; memcpy(&idx, c + 4, 4);
        int p[4][3], ap[8]; pal_of(c0, c1, p);
        ap[0] = a[0]; ap[1] = a[1]; for (int k = 2; k < 8; k++) ap[k] = ((8 - k) * a[0] + (k - 1) * a[1] + 3) / 7;
        uint64_t ab = 0; memcpy(&ab, a + 2, 6);
        for (int t = 0; t < 16; t++) { int k = idx >> (2 * t) & 3, ak = (int)(ab >> (3 * t) & 7);
            for (int ch = 0; ch < 3; ch++) { int d = px[t * 4 + ch] - p[k][ch]; rgb += (uint64_t)(d * d); }
            int d = px[t * 4 + 3] - ap[ak]; al += (uint64_t)(d * d); if (px[t * 4 + 3] < s->a_min) s->a_min = px[t * 4 + 3]; }
        cnt += 16;
    }
    s->err_rgb = cnt ? __builtin_sqrtf((float)rgb / (float)(cnt * 3)) : 0;
    s->err_a = cnt ? __builtin_sqrtf((float)al / (float)cnt) : 0;
}
