// A physical game controller's Android events, turned into Switch buttons and sticks. Pure C with no Android headers, so it is tested without a controller or a phone.
// Buttons by position, as on a Switch: the bottom face button (Android's A) is B. Sticks: X / Y left, Z / Rz right, Android's standard for a gamepad.
// One controller drives player 1. Another one takes over by pressing a button, or by a firm push while the first holds nothing, so two pads never fight over the state.
#pragma once
#include <stdint.h>

// Android's numbers (keycodes.h, input.h); front_android.c checks each one against the NDK when it compiles
enum {
    PM_KEY_BACK = 4, PM_KEY_DPAD_UP = 19, PM_KEY_DPAD_DOWN = 20, PM_KEY_DPAD_LEFT = 21, PM_KEY_DPAD_RIGHT = 22, PM_KEY_MENU = 82,
    PM_KEY_BUTTON_A = 96, PM_KEY_BUTTON_B = 97, PM_KEY_BUTTON_X = 99, PM_KEY_BUTTON_Y = 100, PM_KEY_BUTTON_L1 = 102, PM_KEY_BUTTON_R1 = 103,
    PM_KEY_BUTTON_L2 = 104, PM_KEY_BUTTON_R2 = 105, PM_KEY_BUTTON_THUMBL = 106, PM_KEY_BUTTON_THUMBR = 107, PM_KEY_BUTTON_START = 108, PM_KEY_BUTTON_SELECT = 109,
};
enum { PM_SOURCE_KEYBOARD = 0x101, PM_SOURCE_DPAD = 0x201, PM_SOURCE_GAMEPAD = 0x401, PM_SOURCE_JOYSTICK = 0x01000010 };

enum { PM_DOWN, PM_UP, PM_OTHER };                // what a key event does; a key's auto-repeat is PM_OTHER, or a held button on one controller would keep taking over from the other
enum { PM_IGNORED, PM_HANDLED, PM_USED };         // PM_IGNORED: not a controller's, Android keeps it. PM_USED: a fresh press or deflection, so the on-screen pad can go

typedef struct { float x, y, z, rz, hat_x, hat_y, ltrigger, rtrigger, brake, gas; } pm_axes;   // an axis the device lacks reads 0
typedef struct {
    int active, device;                // the controller in use, by Android device id
    uint64_t keys;
    int32_t stick[4];                  // lx, ly, rx, ry; up is positive, as on a Switch
    float hat_x, hat_y, lt, rt;
    int spent;                         // a firm push has counted as fresh use; the next one counts only after the axes are back at rest
} pm_state;

int pm_key(pm_state *s, int device, int source, int key, int action);
int pm_motion(pm_state *s, int device, const pm_axes *a, int cancel);   // returns PM_USED or PM_HANDLED. A gentle push reaches the game, but only a firm one from rest is fresh use: jitter, or a stick that stays held, does not end a touch
int pm_held(const pm_state *s);                                         // anything pressed or deflected
void pm_release(pm_state *s);                                           // nothing held, no controller in use: focus lost, or the controller went away
void pm_output(const pm_state *s, uint64_t *buttons, int32_t stick[4]);
