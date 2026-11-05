// Graphics debugging hooks that nxnative's Metal backend implements (nvn_mtl.m, nvn_capture.inc) and the common C calls: the NVN command capture
// (DM_NVN_CAPTURE, nvn_capture.c, which needs Apple's compression.h and is not built here) and the DM_CTL commands shot / texlist / texraw / texdump /
// passdump (hle_hid.c). Not available on Android: each logs once and does nothing.
#include "dm.h"
#include "nvn_capture.h"
static void unavailable(const char *what) { static int said; if (!said++) LOG("[RED] %s: graphics debugging hooks are not available on Android", what); }
void nvn_cap_submit(int n, const uint64_t *handles) { (void)n; (void)handles; unavailable("DM_NVN_CAPTURE"); }
void nvn_cap_present(const tex_t *t, const int crop[4]) { (void)t; (void)crop; unavailable("DM_NVN_CAPTURE"); }
void dm_gfx_shot_at(long k, const char *path) { (void)k; (void)path; unavailable("DM_CTL shot"); }
void dm_gfx_texlist(int target) { (void)target; unavailable("DM_CTL texlist"); }
void dm_gfx_texraw(uint64_t serial, const char *path) { (void)serial; (void)path; unavailable("DM_CTL texraw"); }
void dm_gfx_texdump_next(const char *path) { (void)path; unavailable("DM_CTL texdump"); }
void dm_gfx_passdump_next(long n) { (void)n; unavailable("DM_CTL passdump"); }
