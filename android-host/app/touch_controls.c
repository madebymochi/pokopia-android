// On-screen Switch controls for the APK (front_android.c): nxnative's iOS controls (dm/ios/Sources/DmGamepad.m): same layouts, sizes, positions, touch rules and
// look, ported to C. Layout, touch tracking and a CPU-drawn layer (premultiplied RGBA8, half the window size) that the Vulkan present blends over the frame
// (dm_vk_overlay_image).
//   Two layouts; one tap on the layout button at the top centre (labelled "3D" / "2D") switches them, and the choice is kept in <state dir>/pad_layout.txt;
//   the first launch uses the game profile's pad_layout (DM_GAME_PAD_LAYOUT: 0 simple, 1 full, 2 touch; touch has no Android layout and starts in 2D).
//   2D: ZL above L and ZR above R in the top corners (teardrops, tip toward the screen corner), − and + at the top centre, the left stick in the bottom-left
//     corner (fixed ring, tilt measured from the ring centre, small dead zone), the A B X Y diamond in the bottom-right corner (sliding a finger switches keys).
//   3D: everything in 2D plus a D-pad (inward of and below the left stick, 8-way), a right stick that is not drawn (a touch landing in the right third of the
//     screen on no control holds it, centred where it lands and dragged along past its radius) and L3 / R3 (small round keys inward of ZL / ZR).
//   Touch-down ownership: locked keys first (shoulders, − / +, L3 / R3, the layout button), then whichever of the diamond, the D-pad and the left stick has the
//     smallest distance / grab radius, then in 3D the right stick for a touch in the right third. Touches no control takes go to the game's touch screen.
// Sizes are in u: 1 u = 1 dp on a 430 dp tall screen (at most 1.3 dp), as the iOS app's points. Android gives no safe area to a NativeActivity drawn under the
// camera cutout, so the sides keep K_SIDE dp and the bottom K_BOTTOM dp (the iOS app uses the system's insets).
// The look: DmGamepad.m's neutral glass: white at low opacity over a soft black shadow, a white outline, white labels turning black while held, the fill rising
// while held, a ring and an inner ring on the knob. On a bright picture white on white would vanish, so two things are added to see a control over anything
// while still seeing through it: a thin dark line outside each outline and a dark halo around each label. Everything is drawn here, no image files.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "touch_controls.h"
#include "dm_input.h"
#include "game_profile.h"

enum { B_A = 0, B_B, B_X, B_Y, B_LS, B_RS, B_L, B_R, B_ZL, B_ZR, B_PLUS, B_MINUS, B_LEFT, B_UP, B_RIGHT, B_DOWN, B_N };   // = the DM_BTN_ bit numbers
enum { C_NONE = 0, C_STICK, C_DPAD, C_FACE, C_BTN, C_SWAP, C_GAME };

// ---- Layout constants (in u; x measured from the outer edge, the right half mirrors): DmGamepad.m's ----
static const float
    K_ZL_X = 44, K_ZL_Y = 40, K_ZL_R = 26,          // ZL / ZR: centre from the outer edge and the top, radius
    K_L_X = 44, K_L_Y = 108, K_L_R = 24,            // L / R: same column as ZL / ZR
    K_DROP_TIP = 1.45f,                             // teardrop tip distance from the centre (times the radius)
    K_CORNER_IN = 0.5f,                             // how far the bottom-corner controls reach into the side margin
    K_FACE_X = 81, K_PAD_Y = 82,                    // diamond centre from the outer edge; diamond and stick centre from the bottom
    K_FACE_GAP = 50, K_FACE_R = 27,                 // diamond: key distance from the centre, key radius
    K_PILL_DX = 52, K_PILL_Y = 22, K_PILL_W = 50, K_PILL_H = 28,   // − / +: centre from the screen's midline and from the top, size
    K_SWAP_R = 15,                                  // layout button: top centre, level with − / +
    K_STICK_X = 81, K_STICK_R = 77, K_KNOB_R = 30,  // left stick: mirrors the diamond
    K_STICK_GRAB = 1.4f,                            // how near the ring centre a touch-down grabs a stick (times the ring radius)
    K_FULL = 0.8f,                                  // full tilt at this fraction of the ring radius (the knob still travels to the ring edge)
    K_DEAD = 0.1f,                                  // stick dead zone (times the ring radius)
    K_LOW_X = 231, K_LOW_Y = 64,                    // D-pad centre from the outer edge and the bottom (3D)
    K_DPAD_R = 48, K_ARM_W = 17, K_ARM_IN = 11,     // D-pad: arm outer edge from the centre, half-width, tip from the centre
    K_DPAD_GRAB = 1.3f,                             // D-pad grab radius (times the outer edge)
    K_RSTICK_R = 60,                                // right stick (not drawn): radius around its centre, the touch-down point
    K_RSWIPE = 1.0f / 3,                            // right stick area: this fraction of the screen width from the right edge
    K_CLICK_X = 102, K_CLICK_R = 17,                // L3 / R3: centre from the outer edge (level with ZL / ZR), radius
    K_SIDE = 44, K_BOTTOM = 14;                     // margins in dp standing in for the iOS safe area (side: camera cutout and rounded corners)
static const int k_btns[] = {B_ZL, B_L, B_ZR, B_R, B_MINUS, B_PLUS, B_LS, B_RS};   // keys locked at touch-down (L3 / R3 only in the 3D layout)
static int full_only(int b) { return b == B_LS || b == B_RS || b >= B_LEFT; }
static int is_drop(int b) { return b == B_ZL || b == B_L || b == B_ZR || b == B_R; }
static int is_click(int b) { return b == B_LS || b == B_RS; }

// ---- Shapes: signed distance (window pixels, negative inside) ----
enum { S_CIRCLE, S_PILL, S_DROP, S_POLY };
typedef struct {
    int kind; float cx, cy, r, w, h;   // circle: r; pill: w x h; drop: r, tip angle a, tip distance d, tip rounding rr; poly: n vertices already moved in by rr
    float a, d, rr; int n; float vx[5], vy[5];
    float x0, y0, x1, y1;              // bounding box (the outline)
} shp;
static float seg_dist(float px, float py, float ax, float ay, float bx, float by) {
    float vx = bx - ax, vy = by - ay, t = ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy); t = t < 0 ? 0 : t > 1 ? 1 : t;
    return hypotf(px - ax - t * vx, py - ay - t * vy);
}
static float sdf(const shp *s, float x, float y) {
    switch (s->kind) {
    case S_CIRCLE: return hypotf(x - s->cx, y - s->cy) - s->r;
    case S_PILL: { float half = s->w / 2 - s->h / 2, dx = fmaxf(0, fabsf(x - s->cx) - half); return hypotf(dx, y - s->cy) - s->h / 2; }
    case S_DROP: {   // a round cone (circle r at the centre, circle rr at h along the tip direction), Inigo Quilez's sdUnevenCapsule
        float ux = cosf(s->a), uy = sinf(s->a), px = x - s->cx, py = y - s->cy;
        float q = fabsf(px * -uy + py * ux), t = px * ux + py * uy, h = s->d - s->rr, b = (s->r - s->rr) / h, a = sqrtf(1 - b * b), k = -b * q + a * t;
        if (k < 0) return hypotf(q, t) - s->r;
        if (k > a * h) return hypotf(q, t - h) - s->rr;
        return a * q + b * t - s->r; }
    default: {       // convex polygon (exact distance), rounded by rr
        float d = 1e9f; int in = 1;
        for (int i = 0; i < s->n; i++) { int j = (i + 1) % s->n;
            d = fminf(d, seg_dist(x, y, s->vx[i], s->vy[i], s->vx[j], s->vy[j]));
            if ((s->vx[j] - s->vx[i]) * (y - s->vy[i]) - (s->vy[j] - s->vy[i]) * (x - s->vx[i]) < 0) in = 0; }   // clockwise on screen (y down)
        return (in ? -d : d) - s->rr; }
    }
}
static void bound(shp *s) {   // the outline's box, from the distance field (exact for these convex shapes to a fraction of a pixel)
    float m = s->kind == S_PILL ? s->w : s->kind == S_DROP ? 2 * s->d + 2 * s->r : s->kind == S_CIRCLE ? 2 * s->r : 400;
    s->x0 = s->cx + m; s->y0 = s->cy + m; s->x1 = s->cx - m; s->y1 = s->cy - m;
    for (int i = 0; i < 256; i++) {   // march out from the centre along 256 directions to the zero crossing
        float ux = cosf(i * 6.2831853f / 256), uy = sinf(i * 6.2831853f / 256), lo = 0, hi = m;
        for (int k = 0; k < 24; k++) { float mid = (lo + hi) / 2; if (sdf(s, s->cx + ux * mid, s->cy + uy * mid) < 0) lo = mid; else hi = mid; }
        float x = s->cx + ux * lo, y = s->cy + uy * lo;
        if (x < s->x0) s->x0 = x; if (y < s->y0) s->y0 = y; if (x > s->x1) s->x1 = x; if (y > s->y1) s->y1 = y;
    }
}
static shp circle_(float x, float y, float r) { shp s = {S_CIRCLE, x, y, r}; s.x0 = x - r; s.y0 = y - r; s.x1 = x + r; s.y1 = y + r; return s; }
static shp pill_(float x, float y, float w, float h) { shp s = {S_PILL, x, y, 0, w, h}; bound(&s); return s; }
static shp drop_(float x, float y, float r, float tip, float rr) { shp s = {S_DROP, x, y, r}; s.a = tip; s.d = r * K_DROP_TIP; s.rr = rr; bound(&s); return s; }
static shp arm_(float cx, float cy, float R, float w, float in, float ang, float rr) {   // one D-pad arm: an upward pentagon (tip toward the centre) rotated by ang
    float vx[5] = {-w, w, w, 0, -w}, vy[5] = {-R, -R, -(in + w), -in, -(in + w)};
    float cs = cosf(ang), sn = sinf(ang); shp s = {S_POLY, cx, cy}; s.n = 5; s.rr = rr;
    float px[5], py[5];
    for (int i = 0; i < 5; i++) { px[i] = cx + vx[i] * cs - vy[i] * sn; py[i] = cy + vx[i] * sn + vy[i] * cs; }
    for (int i = 0; i < 5; i++) {   // each vertex moved in by rr along both edge normals (the rounded outline is then the inner polygon grown by rr)
        int a = (i + 4) % 5, b = (i + 1) % 5;
        float e1x = px[i] - px[a], e1y = py[i] - py[a], e2x = px[b] - px[i], e2y = py[b] - py[i], l1 = hypotf(e1x, e1y), l2 = hypotf(e2x, e2y);
        float n1x = -e1y / l1, n1y = e1x / l1, n2x = -e2y / l2, n2y = e2x / l2, k = rr / (1 + n1x * n2x + n1y * n2y);   // inward normals (clockwise)
        s.vx[i] = px[i] + (n1x + n2x) * k; s.vy[i] = py[i] + (n1y + n2y) * k;
    }
    float mx = 0, my = 0; for (int i = 0; i < 5; i++) { mx += px[i] / 5; my += py[i] / 5; }
    s.cx = mx; s.cy = my; bound(&s); return s;
}

// ---- State ----
static struct {
    int ww, wh, linear; float U;       // window size, the present blends in linear light (tc_blend_linear), pixels per u
    float sx0, sx1, sb, cx0, cx1;      // safe area (left, right, bottom) and the bottom-corner controls' left / right
    int full; char path[600];
    shp key[B_N], swap, ring; int has_key[B_N];
    struct { int id, own, idx; uint64_t face; float x, y; } p[16];   // touches by slot
    int stick[2]; float snx[2], sny[2], rcx, rcy;   // the slot holding each stick (-1: none) and where it is now; the right stick's centre
    uint64_t face, dpad, btn; int swap_held, stick_shown;
    uint64_t buttons; int32_t ax[4];
    int dirty;
    uint8_t *img, *simg; int iw, ih, sok; uint64_t sstate;   // the layer, its static part, and the state the static part was drawn for
    int box[4], acc, clip[4];
} T = {.stick = {-1, -1}};

static float face_r(void) { return K_FACE_R * T.U; }
static void face_pos(int b, float *x, float *y) {
    float cx = T.cx1 - K_FACE_X * T.U, cy = T.sb - K_PAD_Y * T.U, g = K_FACE_GAP * T.U;
    *x = cx + (b == B_A ? g : b == B_Y ? -g : 0); *y = cy + (b == B_B ? g : b == B_X ? -g : 0);
}
static float face_cx(void) { return T.cx1 - K_FACE_X * T.U; }
static float home_x(void) { return T.cx0 + K_STICK_X * T.U; }
static float pad_y(void) { return T.sb - K_PAD_Y * T.U; }
static float dpad_x(void) { return T.cx0 + K_LOW_X * T.U; }
static float dpad_y(void) { return T.sb - K_LOW_Y * T.U; }
static float stick_r(int i) { return (i ? K_RSTICK_R : K_STICK_R) * T.U; }

static void save_layout(void) {
    if (!T.path[0]) return;
    FILE *f = fopen(T.path, "w"); if (f) { fprintf(f, "%d\n", T.full); fclose(f); }
}
void tc_init(const char *dir) {
    snprintf(T.path, sizeof T.path, "%s/pad_layout.txt", dir);
    T.full = DM_GAME_PAD_LAYOUT == 1;
    FILE *f = fopen(T.path, "r"); int v;
    if (f) { if (fscanf(f, "%d", &v) == 1) T.full = v != 0; fclose(f); }
}
void tc_layout(int w, int h, int density) {
    if (w <= 0 || h <= 0) return;
    float dp = (density > 0 ? density : 160) / 160.f, u = fminf(h / dp / 430.f, 1.3f), U = T.U = u * dp;
    T.ww = w; T.wh = h;
    T.sx0 = K_SIDE * dp; T.sx1 = w - K_SIDE * dp; T.sb = h - K_BOTTOM * dp;
    T.cx0 = K_SIDE * dp * K_CORNER_IN; T.cx1 = w - K_SIDE * dp * K_CORNER_IN;
    for (int b = 0; b < B_N; b++) T.has_key[b] = 1;
    for (int b = B_A; b <= B_Y; b++) { float x, y; face_pos(b, &x, &y); T.key[b] = circle_(x, y, face_r()); }
    for (int i = 0; i < 8; i++) { int b = k_btns[i], right = b == B_ZR || b == B_R || b == B_RS || b == B_PLUS;
        if (is_drop(b)) { int z = b == B_ZL || b == B_ZR;
            float x = right ? T.sx1 - (z ? K_ZL_X : K_L_X) * U : T.sx0 + (z ? K_ZL_X : K_L_X) * U, y = (z ? K_ZL_Y : K_L_Y) * U;
            T.key[b] = drop_(x, y, (z ? K_ZL_R : K_L_R) * U, right ? -0.78539816f : -2.3561945f, 4 * U * 0.6f); }   // tip toward the near top corner
        else if (is_click(b)) T.key[b] = circle_(right ? T.sx1 - K_CLICK_X * U : T.sx0 + K_CLICK_X * U, K_ZL_Y * U, K_CLICK_R * U);
        else T.key[b] = pill_((T.sx0 + T.sx1) / 2 + (right ? K_PILL_DX : -K_PILL_DX) * U, K_PILL_Y * U, K_PILL_W * U, K_PILL_H * U); }
    static const struct { int b; float a; } arms[] = {{B_UP, 0}, {B_RIGHT, 1.5707963f}, {B_DOWN, 3.1415927f}, {B_LEFT, -1.5707963f}};
    for (int i = 0; i < 4; i++) T.key[arms[i].b] = arm_(dpad_x(), dpad_y(), K_DPAD_R * U, K_ARM_W * U, K_ARM_IN * U, arms[i].a, 6 * U);
    T.swap = circle_((T.sx0 + T.sx1) / 2, K_PILL_Y * U, K_SWAP_R * U);
    T.ring = circle_(home_x(), pad_y(), stick_r(0));
    T.dirty = 1; T.sok = 0;
}

// ---- Touch ----
static float dist(float ax, float ay, float bx, float by) { return hypotf(ax - bx, ay - by); }
static int btn_at(float x, float y) {   // locked keys (tested on touch-down only, then locked)
    for (int i = 0; i < 8; i++) { int b = k_btns[i]; const shp *s = &T.key[b];
        if (full_only(b) && !T.full) continue;
        int hit = is_drop(b) ? dist(x, y, s->cx, s->cy) <= s->r * 1.35f : is_click(b) ? dist(x, y, s->cx, s->cy) <= K_CLICK_R * T.U * 1.35f
                : fabsf(x - s->cx) <= s->w / 2 + 8 * T.U && fabsf(y - s->cy) <= s->h / 2 + 8 * T.U;
        if (hit) return b; }
    return -1;
}
static int in_swap(float x, float y) { return dist(x, y, T.swap.cx, T.swap.cy) <= K_SWAP_R * T.U * 1.6f; }
static uint64_t face_at(float x, float y) {   // keys under the finger in the diamond (between two keys both count)
    uint64_t m = 0; float r = face_r() * 1.25f;
    for (int b = B_A; b <= B_Y; b++) { float fx, fy; face_pos(b, &fx, &fy); if (dist(x, y, fx, fy) <= r) m |= 1ull << b; }
    return m;
}
static uint64_t dpad_at(float x, float y) {   // 8 directions: 45° each, diagonals press two keys; a small centre circle counts as none
    float dx = x - dpad_x(), dy = y - dpad_y();
    if (hypotf(dx, dy) < K_DPAD_R * T.U * 0.25f) return 0;
    float a = atan2f(-dy, dx) * 57.29578f; uint64_t m = 0;
    if (a > -67.5f && a < 67.5f) m |= 1ull << B_RIGHT;
    if (a > 22.5f && a < 157.5f) m |= 1ull << B_UP;
    if (a > 112.5f || a < -112.5f) m |= 1ull << B_LEFT;
    if (a < -22.5f && a > -157.5f) m |= 1ull << B_DOWN;
    return m;
}
static void knob_off(int i, float *ox, float *oy) {   // knob offset from the stick centre (clamped to the stick radius)
    *ox = *oy = 0; if (T.stick[i] < 0) return;
    float hx = i ? T.rcx : home_x(), hy = i ? T.rcy : pad_y(), dx = T.snx[i] - hx, dy = T.sny[i] - hy, d = hypotf(dx, dy), R = stick_r(i);
    if (d > R) { dx *= R / d; dy *= R / d; }
    *ox = dx; *oy = dy;
}
static int slot_of(int id) { for (int k = 0; k < 16; k++) if (T.p[k].own && T.p[k].id == id) return k; return -1; }
static void begin(int id, float x, float y) {
    int k = -1; for (int i = 0; i < 16; i++) if (!T.p[i].own) { k = i; break; }
    if (k < 0) return;
    memset(&T.p[k], 0, sizeof T.p[k]); T.p[k].id = id; T.p[k].x = x; T.p[k].y = y;
    int hit = btn_at(x, y);
    if (hit >= 0) { T.p[k].own = C_BTN; T.p[k].idx = hit; return; }
    if (in_swap(x, y)) { T.p[k].own = C_SWAP; return; }
    // diamond, D-pad, left stick: the one with the smallest distance / grab radius, if it is <= 1
    float best = 1; int kind = C_NONE, idx = 0;
    float f = dist(x, y, face_cx(), pad_y()) / (K_FACE_GAP * T.U + face_r() * 1.6f); if (f <= best) { best = f; kind = C_FACE; }
    if (T.full) { float d = dist(x, y, dpad_x(), dpad_y()) / (K_DPAD_R * T.U * K_DPAD_GRAB); if (d < best) { best = d; kind = C_DPAD; } }
    if (T.stick[0] < 0) { float s = dist(x, y, home_x(), pad_y()) / (stick_r(0) * K_STICK_GRAB); if (s < best) { best = s; kind = C_STICK; } }
    if (kind == C_NONE && T.full && T.stick[1] < 0 && x >= T.ww - T.ww * K_RSWIPE) { kind = C_STICK; idx = 1; T.rcx = x; T.rcy = y; }   // centred where it lands
    T.p[k].own = kind ? kind : C_GAME; T.p[k].idx = idx;
    if (kind == C_FACE) T.p[k].face = face_at(x, y);
    if (kind == C_STICK) { T.stick[idx] = k; T.snx[idx] = x; T.sny[idx] = y; }
}
static void moved(int k, float x, float y) {
    T.p[k].x = x; T.p[k].y = y;
    if (T.p[k].own == C_FACE) T.p[k].face = face_at(x, y);
    else if (T.p[k].own == C_STICK) { int i = T.p[k].idx; T.snx[i] = x; T.sny[i] = y;
        if (i == 1) { float d = dist(x, y, T.rcx, T.rcy), R = stick_r(1);   // the right stick's centre follows a finger past its radius, so turning back responds at once
            if (d > R) { T.rcx = x - (x - T.rcx) * R / d; T.rcy = y - (y - T.rcy) * R / d; } } }
}
static void end(int k) { for (int i = 0; i < 2; i++) if (T.stick[i] == k) T.stick[i] = -1; T.p[k].own = C_NONE; }
static void reset_all(void) { for (int k = 0; k < 16; k++) T.p[k].own = C_NONE; T.stick[0] = T.stick[1] = -1; }
static void recompute(tc_result *out) {
    uint64_t face = 0, btn = 0, dpad = 0; int swap = 0;
    out->ntouch = 0;
    for (int k = 0; k < 16; k++) switch (T.p[k].own) {
        case C_FACE: face |= T.p[k].face; break;
        case C_BTN: btn |= 1ull << T.p[k].idx; break;
        case C_DPAD: dpad |= dpad_at(T.p[k].x, T.p[k].y); break;
        case C_SWAP: swap = 1; break;
        case C_GAME: if (out->ntouch < 16) { out->touch[out->ntouch].id = (uint32_t)k; out->touch[out->ntouch].x = T.p[k].x; out->touch[out->ntouch].y = T.p[k].y; out->ntouch++; } break;
    }
    for (int i = 0; i < 2; i++) {
        float ox, oy; knob_off(i, &ox, &oy); float R = stick_r(i), d = hypotf(ox, oy), dz = R * K_DEAD;
        T.ax[i * 2] = T.ax[i * 2 + 1] = 0;
        if (d <= dz) continue;
        float m = fminf((d - dz) / (R * K_FULL - dz), 1) / d;   // outside the dead zone: 0 rising to full at the full-tilt radius
        T.ax[i * 2] = (int32_t)lrintf(ox * m * DM_STICK_MAX); T.ax[i * 2 + 1] = (int32_t)lrintf(-oy * m * DM_STICK_MAX);   // host: up is positive
    }
    // A new layer only when something drawn changed: a key, or the left stick while it is held and once more on its release, when the knob goes home. The right
    // stick is not drawn, so turning the camera uploads no layer (DmGamepad.m's recompute has the same rule)
    int stick = T.stick[0] >= 0;
    if (face != T.face || btn != T.btn || dpad != T.dpad || swap != T.swap_held || stick || T.stick_shown) T.dirty = 1;
    T.stick_shown = stick;
    T.face = face; T.btn = btn; T.dpad = dpad; T.swap_held = swap;
    T.buttons = face | btn | dpad;
    out->buttons = T.buttons; out->lx = T.ax[0]; out->ly = T.ax[1]; out->rx = T.ax[2]; out->ry = T.ax[3];
}
int tc_event(int action, int id, const tc_pointer *ptr, int n, tc_result *out) {
    uint64_t was = T.buttons; int swas = T.swap_held, swap = 0;
    for (int i = 0; i < n; i++) { int k = slot_of(ptr[i].id); if (k >= 0) moved(k, ptr[i].x, ptr[i].y); }
    if (action == TC_DOWN) { for (int i = 0; i < n; i++) if (ptr[i].id == id && slot_of(id) < 0) begin(id, ptr[i].x, ptr[i].y); }
    else if (action == TC_UP) { int k = slot_of(id);
        if (k >= 0) { if (T.p[k].own == C_SWAP && in_swap(T.p[k].x, T.p[k].y)) swap = 1;   // the layout button counts only when released on it
                      end(k); } }
    else if (action == TC_CANCEL) reset_all();
    recompute(out);
    if (swap) {   // switch layouts: release every touch first (no key stays held), save, lay out again
        reset_all(); T.full = !T.full; save_layout(); T.sok = 0; T.dirty = 1; recompute(out);
    }
    return (T.buttons & ~was) != 0 || (T.swap_held && !swas);
}

// ---------- Drawing (premultiplied RGBA8, half resolution) ----------
// Sizes in u. A_ values are opacities; every control is white over a black shadow, as DmGamepad.m's, with no other colour.
static const float
    A_FILL = .14f, A_FILL_ON = .45f, A_FILL_RING = .06f,  // fill: a key at rest, a key held, the stick's ring
    A_LINE = .38f, K_LINE = 1.5f,                         // the outline, centred on the shape's edge
    A_SHADOW = .2f, K_SHADOW_R = 1.5f, K_SHADOW_Y = 1,    // soft shadow behind each control, showing through its fill
    A_RING_SHADE = .06f,                                  // added: the ring is by far the largest control, so inside it this replaces the shadow, to keep the picture clear
    A_EDGE = .4f, K_EDGE = 1.1f,                          // added: thin dark line just outside the outline
    A_TEXT = .94f, A_TEXT_ON = .7f,                       // label: white at rest (nxnative .6, here it has the halo to stand on), black while held
    A_HALO = .62f, K_HALO = .1f,                          // added: dark halo around a label at rest; its width, times the font size
    K_WEIGHT = .066f,                                     // label stroke half-width, times the font size
    K_TRACK = 2.3f,                                       // gap between two letters, in font cells (a capital is 7 cells tall)
    K_KNOB_IN = .62f;                                     // the knob's inner ring, times its radius
static const char *kLabel[B_N] = {"A", "B", "X", "Y", "L3", "R3", "L", "R", "ZL", "ZR", "+", "-"};
static float clamp01(float v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
static int imax_(int a, int b) { return a > b ? a : b; }
static int imin_(int a, int b) { return a < b ? a : b; }
// The opacities above are as on iOS, whose compositor blends display values. An sRGB swapchain (this game's: tc_blend_linear) blends in linear light instead, where white at .14 would show as .41 over
// black and black at .62 would take white only down to .80. So there each opacity becomes the one that gives the same result over black and over white
enum { BL_STEPS = 1024 };
static float g_bl[2][BL_STEPS + 1];   // by grey: black, white
void tc_blend_linear(int on) {
    if (on == T.linear) return;
    T.linear = on; T.sok = 0; T.dirty = 1;
    for (int i = 0; on && i <= BL_STEPS; i++) { float a = (float)i / BL_STEPS;
        g_bl[0][i] = 1 - powf(1 - a, 2.2f);
        g_bl[1][i] = a <= .04045f ? a / 12.92f : powf((a + .055f) / 1.055f, 2.4f); }
}
static void blend(int x, int y, float v, float a) {   // premultiplied "over", grey v (0 or 1)
    if (a <= 0) return;
    if (T.linear) a = g_bl[v > 0][(int)(a * BL_STEPS + .5f)];
    uint8_t *p = T.img + ((size_t)y * T.iw + x) * 4; float k = 1 - a, c = v * a * 255 + .5f;
    p[0] = (uint8_t)(c + p[0] * k); p[1] = (uint8_t)(c + p[1] * k); p[2] = (uint8_t)(c + p[2] * k); p[3] = (uint8_t)(a * 255 + .5f + p[3] * k);
}
static void grow(float x0, float y0, float x1, float y1) {   // while the knob is drawn (T.acc), the rectangle it covers grows to include this one (window pixels)
    if (!T.acc) return;
    int a = (int)floorf(x0 / 2) - 1, b = (int)floorf(y0 / 2) - 1, c = (int)ceilf(x1 / 2) + 1, d = (int)ceilf(y1 / 2) + 1;
    if (T.box[2] < T.box[0]) { T.box[0] = a; T.box[1] = b; T.box[2] = c; T.box[3] = d; return; }
    if (a < T.box[0]) T.box[0] = a; if (b < T.box[1]) T.box[1] = b; if (c > T.box[2]) T.box[2] = c; if (d > T.box[3]) T.box[3] = d;
}
// layer pixels whose window-pixel area meets [x0, x1] x [y0, y1], inside the clip; X, Y: the pixel's centre in window pixels
#define LOOP(X0, Y0, X1, Y1) for (int y = imax_((int)floorf((Y0) / 2), T.clip[1]), ye_ = imin_((int)ceilf((Y1) / 2), T.clip[3]); y <= ye_; y++) \
                             for (int x = imax_((int)floorf((X0) / 2), T.clip[0]), xe_ = imin_((int)ceilf((X1) / 2), T.clip[2]); x <= xe_; x++) \
                             for (float X = x * 2 + 1.f, Y = y * 2 + 1.f, once_ = 1; once_; once_ = 0)
static float cov(float d) { return clamp01(.5f - d * .5f); }   // coverage of a layer pixel d window pixels outside an edge
static float reach(void) { return (K_LINE / 2 + K_EDGE + K_SHADOW_R + K_SHADOW_Y) * T.U + 2; }   // how far a control's drawing goes past its outline (window pixels)
// a control as DmGamepad.m's CAShapeLayers draw it: soft shadow behind, fill, stroke centred on the outline; between them the added dark edge
static void draw_shape(const shp *s, float fill, float inner) {   // inner: the shadow's opacity well inside the outline
    float U = T.U, lw = K_LINE * U, ew = K_EDGE * U, sh = K_SHADOW_Y * U, blur = K_SHADOW_R * U, m = reach();
    grow(s->x0 - m, s->y0 - m, s->x1 + m, s->y1 + m);
    LOOP(s->x0 - m, s->y0 - m, s->x1 + m, s->y1 + m) {
        float d = sdf(s, X, Y), k = clamp01(-d / (2 * blur));
        blend(x, y, 0, (A_SHADOW + (inner - A_SHADOW) * k) * clamp01(.5f - sdf(s, X, Y - sh) / (2 * blur)));
        blend(x, y, 1, fill * cov(d));
        blend(x, y, 0, A_EDGE * cov(fabsf(d - (lw + ew) / 2) - ew / 2));
        blend(x, y, 1, A_LINE * cov(fabsf(d) - lw / 2));
    }
}
// ---- Labels: a small stroke font with rounded ends, standing in for the iOS system font's bold capitals. A capital is 7 cells tall, y down ----
typedef struct { char c; int n; float x0, x1, s[32][4]; } glyph;
static glyph kGlyph[12]; static int kGlyphs; static float pen_x, pen_y;
static glyph *new_glyph(char c) { glyph *g = &kGlyph[kGlyphs++]; g->c = c; g->x0 = 1e9f; g->x1 = -1e9f; return g; }
static void pen(float x, float y) { pen_x = x; pen_y = y; }
static void to(glyph *g, float x, float y) {
    float *s = g->s[g->n++]; s[0] = pen_x; s[1] = pen_y; s[2] = x; s[3] = y;
    g->x0 = fminf(g->x0, fminf(pen_x, x)); g->x1 = fmaxf(g->x1, fmaxf(pen_x, x)); pen(x, y);
}
static void arc(glyph *g, float cx, float cy, float rx, float ry, float a0, float a1) {   // degrees, 0 to the right, 90 down; from the pen's place on the arc
    for (int i = 1; i <= 8; i++) { float a = (a0 + (a1 - a0) * i / 8) * 3.1415927f / 180; to(g, cx + rx * cosf(a), cy + ry * sinf(a)); }
}
static void font_init(void) {
    if (kGlyphs) return;
    glyph *g;
    g = new_glyph('A'); pen(.2f, 7); to(g, 2.5f, 0); to(g, 4.8f, 7); pen(1.05f, 4.7f); to(g, 3.95f, 4.7f);
    g = new_glyph('B'); pen(.5f, 7); to(g, .5f, 0); to(g, 2.7f, 0); arc(g, 2.7f, 1.7f, 1.7f, 1.7f, -90, 90); to(g, .5f, 3.4f); pen(.5f, 3.4f); to(g, 2.9f, 3.4f); arc(g, 2.9f, 5.2f, 1.8f, 1.8f, -90, 90); to(g, .5f, 7);
    g = new_glyph('X'); pen(.3f, 0); to(g, 4.7f, 7); pen(4.7f, 0); to(g, .3f, 7);
    g = new_glyph('Y'); pen(.2f, 0); to(g, 2.5f, 3.6f); to(g, 4.8f, 0); pen(2.5f, 3.6f); to(g, 2.5f, 7);
    g = new_glyph('L'); pen(.6f, 0); to(g, .6f, 7); to(g, 4.2f, 7);
    g = new_glyph('R'); pen(.5f, 7); to(g, .5f, 0); to(g, 2.7f, 0); arc(g, 2.7f, 1.9f, 1.9f, 1.9f, -90, 90); to(g, .5f, 3.8f); pen(2.6f, 3.8f); to(g, 4.6f, 7);
    g = new_glyph('Z'); pen(.4f, 0); to(g, 4.6f, 0); to(g, .4f, 7); to(g, 4.6f, 7);
    g = new_glyph('D'); pen(.5f, 0); to(g, .5f, 7); to(g, 1.7f, 7); arc(g, 1.7f, 3.5f, 2.9f, 3.5f, 90, -90); to(g, .5f, 0);
    g = new_glyph('2'); pen(2.5f + 2.05f * cosf(-170 * 3.1415927f / 180), 2.1f + 2.1f * sinf(-170 * 3.1415927f / 180)); arc(g, 2.5f, 2.1f, 2.05f, 2.1f, -170, 25); to(g, .4f, 7); to(g, 4.7f, 7);
    g = new_glyph('3'); pen(2.45f + 1.75f * cosf(-160 * 3.1415927f / 180), 1.75f + 1.75f * sinf(-160 * 3.1415927f / 180)); arc(g, 2.45f, 1.75f, 1.75f, 1.75f, -160, 90); arc(g, 2.45f, 5.25f, 1.75f, 1.75f, -90, 160); pen(1.6f, 3.5f); to(g, 2.45f, 3.5f);
    g = new_glyph('+'); pen(2.5f, 1.2f); to(g, 2.5f, 5.8f); pen(.2f, 3.5f); to(g, 4.8f, 3.5f);
    g = new_glyph('-'); pen(.2f, 3.5f); to(g, 4.8f, 3.5f);
}
static const glyph *glyph_of(char c) { for (int i = 0; i < kGlyphs; i++) if (kGlyph[i].c == c) return &kGlyph[i]; return NULL; }
static float glyph_dist(const glyph *g, float u, float v) {   // in cells, to the nearest stroke
    float best = 1e9f;
    for (int i = 0; i < g->n; i++) { const float *s = g->s[i]; best = fminf(best, seg_dist(u, v, s[0], s[1], s[2], s[3])); }
    return best;
}
// fs: the iOS font size (window pixels); a capital is about 0.7 of it tall. The strings are at most two letters, all of them in the font above
static void draw_text(const char *t, float cx, float cy, float fs, int on) {
    const glyph *g[2]; int n = 0; float cell = fs * .7f / 7, tw = 0, hw = fs * K_WEIGHT, halo = fs * K_HALO;
    for (; *t && n < 2; t++) if ((g[n] = glyph_of(*t))) { tw += g[n]->x1 - g[n]->x0 + (n ? K_TRACK : 0); n++; }
    float x0 = cx - tw * cell / 2, y0 = cy - 3.5f * cell, m = hw + halo + 2;
    LOOP(x0 - m, y0 - m, x0 + tw * cell + m, y0 + 7 * cell + m) {
        float d = 1e9f, gx = 0;
        for (int i = 0; i < n; i++) { d = fminf(d, glyph_dist(g[i], (X - x0) / cell - gx + g[i]->x0, (Y - y0) / cell)); gx += g[i]->x1 - g[i]->x0 + K_TRACK; }
        d *= cell;
        if (on) blend(x, y, 0, A_TEXT_ON * cov(d - hw));
        else { blend(x, y, 0, A_HALO * clamp01((hw + halo - d) / (halo * .6f))); blend(x, y, 1, A_TEXT * cov(d - hw)); }   // full darkness for the first part of the halo, then fading out
    }
}
static float label_size(int b) { return (b <= B_Y ? 20 : is_drop(b) ? 16 : is_click(b) ? 13 : 18) * T.U; }
static int shown(int b) { return !full_only(b) || T.full; }
static uint64_t state_bits(void) { return T.buttons | (uint64_t)T.swap_held << B_N; }
static void draw_key(int b) {
    int on = (T.buttons >> b) & 1;
    draw_shape(&T.key[b], on ? A_FILL_ON : A_FILL, A_SHADOW);
    if (b < B_LEFT) draw_text(kLabel[b], T.key[b].cx, T.key[b].cy, label_size(b), on);
}
static void draw_swap(void) {
    draw_shape(&T.swap, T.swap_held ? A_FILL_ON : A_FILL, A_SHADOW);
    draw_text(T.full ? "3D" : "2D", T.swap.cx, T.swap.cy, 12 * T.U, T.swap_held);
}
static void draw_ring(void) { draw_shape(&T.ring, A_FILL_RING, A_RING_SHADE); }
static void draw_static(void) {   // everything but the knob: the ring, below the keys as on iOS, then the keys
    draw_ring();
    for (int b = 0; b < B_N; b++) if (shown(b)) draw_key(b);
    draw_swap();
}
static void box_of(const shp *s, int b[4]) {   // a control's layer box with its stroke, shadow and label
    float m = reach();
    b[0] = imax_((int)floorf((s->x0 - m) / 2) - 1, 0); b[1] = imax_((int)floorf((s->y0 - m) / 2) - 1, 0);
    b[2] = imin_((int)ceilf((s->x1 + m) / 2) + 1, T.iw - 1); b[3] = imin_((int)ceilf((s->y1 + m) / 2) + 1, T.ih - 1);
}
static int meets(const shp *s, const int b[4]) { int o[4]; box_of(s, o); return o[0] <= b[2] && b[0] <= o[2] && o[1] <= b[3] && b[1] <= o[3]; }
static void redraw_box(const shp *s0) {   // a key whose state changed: clear its box and draw every control that reaches into it, then keep that in the static part
    int b[4]; box_of(s0, b);
    if (b[2] < b[0] || b[3] < b[1]) return;
    memcpy(T.clip, b, sizeof b);
    for (int y = b[1]; y <= b[3]; y++) memset(T.img + ((size_t)y * T.iw + b[0]) * 4, 0, (size_t)(b[2] - b[0] + 1) * 4);
    if (meets(&T.ring, b)) draw_ring();
    for (int k = 0; k < B_N; k++) if (shown(k) && meets(&T.key[k], b)) draw_key(k);
    if (meets(&T.swap, b)) draw_swap();
    for (int y = b[1]; y <= b[3]; y++) memcpy(T.simg + ((size_t)y * T.iw + b[0]) * 4, T.img + ((size_t)y * T.iw + b[0]) * 4, (size_t)(b[2] - b[0] + 1) * 4);
    T.clip[0] = T.clip[1] = 0; T.clip[2] = T.iw - 1; T.clip[3] = T.ih - 1;
}
int tc_render(const uint8_t **rgba, int *w, int *h) {
    if (!T.dirty || !T.ww) return 0;
    T.dirty = 0;
    font_init();
    int iw = (T.ww + 1) / 2, ih = (T.wh + 1) / 2; size_t n = (size_t)iw * ih * 4;
    if (iw != T.iw || ih != T.ih) { free(T.img); free(T.simg); T.img = malloc(n); T.simg = malloc(n); T.iw = iw; T.ih = ih; T.sok = 0; }
    T.clip[0] = T.clip[1] = 0; T.clip[2] = iw - 1; T.clip[3] = ih - 1;
    // The keys change only on a press, the knob with every move: the static part is drawn once per state and copied; a press draws again only the keys
    // that changed (the whole layer every move was ~14 ms of CPU per frame with a thumb on the stick)
    uint64_t st = state_bits();
    if (T.sok) {   // the layer is the static part plus the knob of last time: put back what it covered
        int x0 = imax_(T.box[0], 0), y0 = imax_(T.box[1], 0), x1 = imin_(T.box[2], iw - 1), y1 = imin_(T.box[3], ih - 1);
        for (int y = y0; x0 <= x1 && y <= y1; y++) memcpy(T.img + ((size_t)y * iw + x0) * 4, T.simg + ((size_t)y * iw + x0) * 4, (size_t)(x1 - x0 + 1) * 4);
        for (int b = 0; b < B_N; b++) if (shown(b) && ((st ^ T.sstate) >> b & 1)) redraw_box(&T.key[b]);
        if ((st ^ T.sstate) >> B_N & 1) redraw_box(&T.swap);
    } else {
        memset(T.img, 0, n);
        draw_static();
        memcpy(T.simg, T.img, n); T.sok = 1;
    }
    T.sstate = st;
    // the knob, with its inner ring
    T.box[0] = T.box[1] = 0; T.box[2] = T.box[3] = -1; T.acc = 1;
    float ox, oy; knob_off(0, &ox, &oy);
    shp knob = circle_(home_x() + ox, pad_y() + oy, K_KNOB_R * T.U);
    draw_shape(&knob, T.stick[0] >= 0 ? A_FILL_ON : A_FILL, A_SHADOW);
    float ri = knob.r * K_KNOB_IN, lw = K_LINE * T.U;
    LOOP(knob.cx - ri - lw, knob.cy - ri - lw, knob.cx + ri + lw, knob.cy + ri + lw) blend(x, y, 1, A_LINE * cov(fabsf(hypotf(X - knob.cx, Y - knob.cy) - ri) - lw / 2));
    T.acc = 0;
    *rgba = T.img; *w = iw; *h = ih;
    return 1;
}
