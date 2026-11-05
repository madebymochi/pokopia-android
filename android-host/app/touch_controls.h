// On-screen Switch controls of the APK (touch_controls.c). Coordinates are window pixels (landscape).
#pragma once
#include <stdint.h>
enum { TC_DOWN, TC_MOVE, TC_UP, TC_CANCEL };
typedef struct { int id; float x, y; } tc_pointer;
typedef struct { uint64_t buttons; int32_t lx, ly, rx, ry; int ntouch; struct { uint32_t id; float x, y; } touch[16]; } tc_result;
// state_dir: where the chosen layout (3D / 2D) is kept across runs
void tc_init(const char *state_dir);
// w, h: window pixels; density: the screen's dpi (AConfiguration_getDensity), so the controls keep their size in dp
void tc_layout(int w, int h, int density);
// One touch event: action for pointer id, with every pointer still on the screen (including id). Fills the controller state and the touches that go to
// the game (those no control takes). Returns 1 when a control was newly pressed (for a haptic tick)
int tc_event(int action, int id, const tc_pointer *ptr, int n, tc_result *out);
// The layer to show when it changed: premultiplied RGBA8 at half the window size. Returns 0 when nothing changed
int tc_render(const uint8_t **rgba, int *w, int *h);
// Whether the present blends the layer in linear light (an sRGB swapchain) rather than on display values; a change draws the layer again
void tc_blend_linear(int on);
