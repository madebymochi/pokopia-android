// See pad_map.h
#include <math.h>
#include <string.h>
#include "pad_map.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "pad_hid.h"   // nxnative's round dead zone, pad_hid_stick
#pragma clang diagnostic pop

static int has(int source, int kind) { return (source & kind) == kind; }   // a whole source: the classes share low bits, so a plain AND would also match any keyboard

static uint64_t button_of(int source, int key) {
    int pad = has(source, PM_SOURCE_GAMEPAD) || has(source, PM_SOURCE_JOYSTICK);
    if (!pad && !has(source, PM_SOURCE_DPAD)) return 0;
    switch (key) {
        case PM_KEY_BUTTON_A: return DM_BTN_B;
        case PM_KEY_BUTTON_B: return DM_BTN_A;
        case PM_KEY_BUTTON_X: return DM_BTN_Y;
        case PM_KEY_BUTTON_Y: return DM_BTN_X;
        case PM_KEY_BUTTON_L1: return DM_BTN_L;
        case PM_KEY_BUTTON_R1: return DM_BTN_R;
        case PM_KEY_BUTTON_L2: return DM_BTN_ZL;
        case PM_KEY_BUTTON_R2: return DM_BTN_ZR;
        case PM_KEY_BUTTON_THUMBL: return DM_BTN_LS;
        case PM_KEY_BUTTON_THUMBR: return DM_BTN_RS;
        case PM_KEY_BUTTON_START: return DM_BTN_PLUS;
        case PM_KEY_BUTTON_SELECT: return DM_BTN_MINUS;
        case PM_KEY_DPAD_UP: return DM_BTN_UP;
        case PM_KEY_DPAD_DOWN: return DM_BTN_DOWN;
        case PM_KEY_DPAD_LEFT: return DM_BTN_LEFT;
        case PM_KEY_DPAD_RIGHT: return DM_BTN_RIGHT;
        // Some controllers send these for their small middle buttons. Left to Android, Back would close the game with what is not saved. The phone's own Back is not a controller's, so it stays Android's
        case PM_KEY_BACK: return pad ? DM_BTN_MINUS : 0;
        case PM_KEY_MENU: return pad ? DM_BTN_PLUS : 0;
    }
    return 0;
}

int pm_key(pm_state *s, int device, int source, int key, int action) {
    uint64_t b = button_of(source, key);
    if (!b) return PM_IGNORED;
    if (action == PM_DOWN) {
        if (!s->active || s->device != device) { memset(s, 0, sizeof *s); s->active = 1; s->device = device; }
        s->keys |= b;
        return PM_USED;
    }
    if (action == PM_UP && s->active && s->device == device) s->keys &= ~b;
    return PM_HANDLED;
}

static int hat_or_trigger(const pm_state *s) { return fabsf(s->hat_x) > .5f || fabsf(s->hat_y) > .5f || s->lt > .5f || s->rt > .5f; }
static int axes_held(const pm_state *s) { return s->stick[0] || s->stick[1] || s->stick[2] || s->stick[3] || hat_or_trigger(s); }
// Pushed on purpose. A worn stick that hovers at the dead zone's edge counts as held, so the game sees it, but it must not hide the on-screen pad or take over from another controller
static int axes_firm(const pm_state *s) {
    for (int i = 0; i < 4; i++) if (s->stick[i] > DM_STICK_MAX / 4 || s->stick[i] < -DM_STICK_MAX / 4) return 1;
    return hat_or_trigger(s);
}
static void clear_axes(pm_state *s) { memset(s->stick, 0, sizeof s->stick); s->hat_x = s->hat_y = s->lt = s->rt = 0; }
static void read_axes(pm_state *s, const pm_axes *a) {
    pad_hid_stick(a->x, -a->y, &s->stick[0], &s->stick[1]);
    pad_hid_stick(a->z, -a->rz, &s->stick[2], &s->stick[3]);
    s->hat_x = a->hat_x; s->hat_y = a->hat_y;
    s->lt = fmaxf(a->ltrigger, a->brake); s->rt = fmaxf(a->rtrigger, a->gas);
}

int pm_motion(pm_state *s, int device, const pm_axes *a, int cancel) {
    int same = s->active && s->device == device;
    pm_state t;
    if (same) t = *s; else memset(&t, 0, sizeof t);
    if (cancel) clear_axes(&t); else read_axes(&t, a);
    int firm = axes_firm(&t), held = axes_held(&t);
    if (!same) {   // with no controller in use any push claims it; taking over from another takes a firm one
        if (pm_held(s) || !(s->active ? firm : held)) return PM_HANDLED;
        t.active = 1; t.device = device;
    }
    int fresh = firm && !t.spent;
    t.spent = held && (t.spent || firm);
    *s = t;
    return fresh ? PM_USED : PM_HANDLED;
}

int pm_held(const pm_state *s) { return s->keys || axes_held(s); }
void pm_release(pm_state *s) { memset(s, 0, sizeof *s); }

void pm_output(const pm_state *s, uint64_t *buttons, int32_t stick[4]) {
    uint64_t b = s->keys;
    if (s->hat_x < -.5f) b |= DM_BTN_LEFT;
    if (s->hat_x > .5f) b |= DM_BTN_RIGHT;
    if (s->hat_y < -.5f) b |= DM_BTN_UP;
    if (s->hat_y > .5f) b |= DM_BTN_DOWN;
    if (s->lt > .5f) b |= DM_BTN_ZL;
    if (s->rt > .5f) b |= DM_BTN_ZR;
    *buttons = b;
    memcpy(stick, s->stick, sizeof s->stick);
}
