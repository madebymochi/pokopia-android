// H.264 movie decoding is VideoToolbox on Apple (hle_movie_avc.c). Android: MediaCodec (AMediaCodec) later; until then every decode fails
// and hle_movie.c handles it as a decoder error.
#include "hle_movie_avc.h"
#include "dm.h"
mavc *mavc_new(void) { static int said; if (!said++) LOG("[RED] movie: H.264 decoding not available yet on Android (MediaCodec backend missing)"); return NULL; }
void mavc_free(mavc *a) { (void)a; }
void mavc_param_sets(mavc *a, const uint8_t *p, size_t n) { (void)a; (void)p; (void)n; }
void mavc_avcc(mavc *a, const uint8_t *p, size_t n) { (void)a; (void)p; (void)n; }
int mavc_decode(mavc *a, const uint8_t *p, int n, int64_t pts) { (void)a; (void)p; (void)n; (void)pts; return -1; }
int mavc_take(mavc *a, int flush, uint8_t *nv12, int w, int h, int64_t *pts) { (void)a; (void)flush; (void)nv12; (void)w; (void)h; (void)pts; return 0; }
int mavc_queued(const mavc *a) { (void)a; return 0; }
void mavc_set_rgba(mavc *a) { (void)a; }
void mavc_drop(mavc *a) { (void)a; }
