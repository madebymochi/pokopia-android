// compat: out-of-line parts of the Darwin / Mach stand-ins for nxnative's host on Android / Linux.
#define _GNU_SOURCE
#include "../darwin_compat.h"
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach-o/getsect.h>
#include <CommonCrypto/CommonDigest.h>
#include <sys/clonefile.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <ucontext.h>
#include <unistd.h>

const struct mach_header_64 _mh_execute_header = {0xfeedfacf};

uint64_t mach_absolute_time(void) { return clock_gettime_nsec_np(CLOCK_MONOTONIC); }

kern_return_t vm_read_overwrite(vm_map_t task, vm_address_t addr, vm_size_t size, vm_address_t data, vm_size_t *out_size) {
    struct iovec l = {(void *)data, size}, r = {(void *)addr, size};
    ssize_t n = process_vm_readv((pid_t)task, &l, 1, &r, 1, 0);
    if (n < 0) { if (out_size) *out_size = 0; return KERN_INVALID_ADDRESS; }
    if (out_size) *out_size = (vm_size_t)n;
    return (vm_size_t)n == size ? KERN_SUCCESS : KERN_INVALID_ADDRESS;
}

// /proc/self/task/<tid>/stat: field 3 state, 14 utime, 15 stime (clock ticks)
kern_return_t thread_info(thread_act_t th, thread_flavor_t flavor, thread_info_t out, mach_msg_type_number_t *count) {
    char p[64], b[1024]; snprintf(p, sizeof p, "/proc/self/task/%u/stat", th);
    int fd = open(p, O_RDONLY | O_CLOEXEC); if (fd < 0) return KERN_INVALID_ARGUMENT;
    ssize_t n = read(fd, b, sizeof b - 1); close(fd); if (n <= 0) return KERN_FAILURE; b[n] = 0;
    char *s = strrchr(b, ')'); if (!s) return KERN_FAILURE;
    char st = 0; unsigned long long ut = 0, sy = 0;
    if (sscanf(s + 2, "%c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &st, &ut, &sy) != 3) return KERN_FAILURE;
    long hz = sysconf(_SC_CLK_TCK); if (hz <= 0) hz = 100;
    uint64_t uus = ut * 1000000ull / (uint64_t)hz, sus = sy * 1000000ull / (uint64_t)hz;
    int rs = st == 'R' ? TH_STATE_RUNNING : st == 'D' ? TH_STATE_UNINTERRUPTIBLE : st == 'T' || st == 't' ? TH_STATE_STOPPED : TH_STATE_WAITING;
    if (flavor == THREAD_BASIC_INFO) {
        thread_basic_info_data_t *bi = (thread_basic_info_data_t *)out; memset(bi, 0, sizeof *bi);
        bi->user_time.seconds = (integer_t)(uus / 1000000); bi->user_time.microseconds = (integer_t)(uus % 1000000);
        bi->system_time.seconds = (integer_t)(sus / 1000000); bi->system_time.microseconds = (integer_t)(sus % 1000000);
        bi->run_state = rs; if (count) *count = THREAD_BASIC_INFO_COUNT; return KERN_SUCCESS;
    }
    if (flavor == THREAD_EXTENDED_INFO) {
        thread_extended_info_data_t *xi = (thread_extended_info_data_t *)out; memset(xi, 0, sizeof *xi);
        xi->pth_user_time = uus * 1000; xi->pth_system_time = sus * 1000; xi->pth_run_state = rs;
        if (count) *count = THREAD_EXTENDED_INFO_COUNT; return KERN_SUCCESS;
    }
    return KERN_NOT_SUPPORTED;
}

int mergesort(void *base, size_t n, size_t size, int (*cmp)(const void *, const void *)) {   // bottom-up, stable; 0, or -1 with errno when the buffer cannot be allocated, as BSD's
    if (n < 2) return 0;
    char *a = base, *b = malloc(n * size); if (!b) { errno = ENOMEM; return -1; }
    for (size_t w = 1; w < n; w *= 2) {
        for (size_t lo = 0; lo < n; lo += 2 * w) {
            size_t mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, k = lo;
            while (i < mid && j < hi) { if (cmp(a + j * size, a + i * size) < 0) memcpy(b + k++ * size, a + j++ * size, size); else memcpy(b + k++ * size, a + i++ * size, size); }
            if (i < mid) memcpy(b + k * size, a + i * size, (mid - i) * size), k += mid - i;
            if (j < hi) memcpy(b + k * size, a + j * size, (hi - j) * size);
        }
        char *t = a; a = b; b = t;
    }
    if (a != base) { memcpy(base, a, n * size); free(a); } else free(b);
    return 0;
}
int thread_selfcounts(int type, void *buf, size_t nbytes) { (void)type; memset(buf, 0, nbytes); return -1; }

void *pthread_get_stackaddr_np(pthread_t t) {
    pthread_attr_t a; void *lo = 0; size_t sz = 0;
    if (pthread_getattr_np(t, &a)) return 0;
    pthread_attr_getstack(&a, &lo, &sz); pthread_attr_destroy(&a);
    return (char *)lo + sz;
}
size_t pthread_get_stacksize_np(pthread_t t) {
    pthread_attr_t a; void *lo = 0; size_t sz = 0;
    if (pthread_getattr_np(t, &a)) return 0;
    pthread_attr_getstack(&a, &lo, &sz); pthread_attr_destroy(&a);
    return sz;
}

int dm_fcntl_getpath(int fd, char *out) {
    char p[64]; snprintf(p, sizeof p, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(p, out, PATH_MAX - 1); if (n < 0) return -1;
    out[n] = 0; return 0;
}

#ifndef RENAME_EXCHANGE
#define RENAME_EXCHANGE (1 << 1)
#endif
#ifndef RENAME_NOREPLACE
#define RENAME_NOREPLACE (1 << 0)
#endif
int renamex_np(const char *from, const char *to, unsigned flags) {
    unsigned lf = (flags & RENAME_SWAP ? RENAME_EXCHANGE : 0) | (flags & RENAME_EXCL ? RENAME_NOREPLACE : 0);
    return (int)syscall(SYS_renameat2, AT_FDCWD, from, AT_FDCWD, to, lf);
}

// mkdir (every caller: build.sh links with --wrap=mkdir): the app's shared folder (Android/data/<package>/files) keeps the save, and its group (adb,
// the PC's file transfer) needs write access to replace it, so new folders get the group's bits next to the owner's (the host asks for 0755)
int __real_mkdir(const char *path, mode_t mode);
int __wrap_mkdir(const char *path, mode_t mode) { return __real_mkdir(path, mode | ((mode & 0700) >> 3)); }

// clonefile: recursive copy that keeps modes; fails with EEXIST when dst exists (as on APFS). The commit swaps this copy in as the save and deletes the old one,
// so every file is fsync'd before close (a power cut right after an autosave left empty or short files in the new save); a write, fsync or close error fails the copy
static int sync_ok(int e) { return e == EINVAL || e == ENOTSUP || e == EROFS || e == ENOSYS; }   // storage that cannot flush at all: not an error of the copy (logged once)
static void sync_note(const char *what, int e) { static int once; if (!once) { once = 1; fprintf(stderr, "[dm] compat: %s is not supported here (errno %d): the save copy is not flushed\n", what, e); } }
static int copy_file(const char *s, const char *d, mode_t m) {
    int in = open(s, O_RDONLY | O_CLOEXEC); if (in < 0) return -1;
    int out = open(d, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, m & 07777); if (out < 0) { int e = errno; close(in); errno = e; return -1; }
    char buf[1 << 16]; ssize_t n; int e = 0;
    while (!e && (n = read(in, buf, sizeof buf)) != 0) {
        if (n < 0) { if (errno != EINTR) e = errno; continue; }
        for (char *p = buf; n > 0 && !e;) { ssize_t w = write(out, p, (size_t)n); if (w > 0) { p += w; n -= w; } else if (w == 0) e = ENOSPC; else if (errno != EINTR) e = errno; }
    }
    if (!e && fsync(out)) { if (sync_ok(errno)) sync_note("fsync", errno); else e = errno; }
    if (close(out) && !e) e = errno;
    close(in); errno = e;
    return e ? -1 : 0;
}
static void sync_dir(const char *p) {   // best effort: the entries of a folder just filled or created in (a failure is only logged)
    int fd = open(p, O_RDONLY | O_DIRECTORY | O_CLOEXEC); if (fd < 0) return;
    if (fsync(fd)) { if (sync_ok(errno)) sync_note("fsync of a folder", errno); else fprintf(stderr, "[dm] compat: [RED] fsync of folder %s failed errno=%d\n", p, errno); }
    close(fd);
}
static int copy_tree(const char *s, const char *d) {
    struct stat st; if (lstat(s, &st)) return -1;
    if (S_ISREG(st.st_mode)) return copy_file(s, d, st.st_mode);
    if (S_ISLNK(st.st_mode)) { char t[PATH_MAX]; ssize_t n = readlink(s, t, sizeof t - 1); if (n < 0) return -1; t[n] = 0; return symlink(t, d); }
    if (!S_ISDIR(st.st_mode)) { errno = ENOTSUP; return -1; }
    if (mkdir(d, st.st_mode & 07777)) return -1;
    DIR *dir = opendir(s); if (!dir) return -1;
    struct dirent *e; int r = 0;
    while (!r && (e = readdir(dir))) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        char a[PATH_MAX], b[PATH_MAX];
        if (snprintf(a, sizeof a, "%s/%s", s, e->d_name) >= (int)sizeof a || snprintf(b, sizeof b, "%s/%s", d, e->d_name) >= (int)sizeof b) { errno = ENAMETOOLONG; r = -1; break; }
        r = copy_tree(a, b);
    }
    int er = errno; closedir(dir); errno = er;
    if (!r) sync_dir(d);
    return r;
}
int clonefile(const char *src, const char *dst, uint32_t flags) {
    (void)flags; struct stat st;
    if (!lstat(dst, &st)) { errno = EEXIST; return -1; }
    int r = copy_tree(src, dst);
    if (!r) { char p[PATH_MAX]; snprintf(p, sizeof p, "%s", dst); char *sl = strrchr(p, '/'); if (sl && sl != p) { *sl = 0; sync_dir(p); } }   // the new top folder's entry in its parent
    return r;
}

// Signal context extras (dm.h UC_ESR, hle_thread.c): walk the records in uc_mcontext.__reserved
struct dm_ctx_head { uint32_t magic, size; };
static const struct dm_ctx_head *uc_find(const void *uc_, uint32_t magic) {
    const ucontext_t *uc = uc_; const uint8_t *p = (const uint8_t *)uc->uc_mcontext.__reserved, *end = p + sizeof uc->uc_mcontext.__reserved;
    while (p + sizeof(struct dm_ctx_head) <= end) {
        const struct dm_ctx_head *h = (const struct dm_ctx_head *)p;
        if (!h->magic || !h->size) return 0;
        if (h->magic == magic) return h;
        p += h->size;
    }
    return 0;
}
uint64_t dm_uc_esr(const void *uc) {
    const struct dm_ctx_head *h = uc_find(uc, 0x45535201 /* ESR_MAGIC */);
    return h ? *(const uint64_t *)(h + 1) : 0;
}
void dm_uc_fpsimd(const void *uc, void *v, uint32_t *fpcr, uint32_t *fpsr) {
    const struct dm_ctx_head *h = uc_find(uc, 0x46508001 /* FPSIMD_MAGIC */);
    if (!h) { memset(v, 0, 32 * 16); *fpcr = *fpsr = 0; return; }
    const uint32_t *f = (const uint32_t *)(h + 1);   // fpsr, fpcr, then __uint128_t vregs[32] (16-byte aligned at +16)
    *fpsr = f[0]; *fpcr = f[1];
    memcpy(v, (const uint8_t *)h + 16, 32 * 16);
}

// SHA-256 (FIPS 180-4), for CC_SHA256_*
static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define ROR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha_block(CC_SHA256_CTX *c, const uint8_t *b) {
    uint32_t w[64], s[8];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)b[4 * i] << 24 | (uint32_t)b[4 * i + 1] << 16 | (uint32_t)b[4 * i + 2] << 8 | b[4 * i + 3];
    for (int i = 16; i < 64; i++) { uint32_t s0 = ROR(w[i-15], 7) ^ ROR(w[i-15], 18) ^ (w[i-15] >> 3), s1 = ROR(w[i-2], 17) ^ ROR(w[i-2], 19) ^ (w[i-2] >> 10); w[i] = w[i-16] + s0 + w[i-7] + s1; }
    memcpy(s, c->h, sizeof s);
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROR(s[4], 6) ^ ROR(s[4], 11) ^ ROR(s[4], 25), ch = (s[4] & s[5]) ^ (~s[4] & s[6]), t1 = s[7] + S1 + ch + K[i] + w[i];
        uint32_t S0 = ROR(s[0], 2) ^ ROR(s[0], 13) ^ ROR(s[0], 22), mj = (s[0] & s[1]) ^ (s[0] & s[2]) ^ (s[1] & s[2]), t2 = S0 + mj;
        s[7] = s[6]; s[6] = s[5]; s[5] = s[4]; s[4] = s[3] + t1; s[3] = s[2]; s[2] = s[1]; s[1] = s[0]; s[0] = t1 + t2;
    }
    for (int i = 0; i < 8; i++) c->h[i] += s[i];
}
int CC_SHA256_Init(CC_SHA256_CTX *c) {
    static const uint32_t h0[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(c->h, h0, sizeof h0); c->len = 0; c->n = 0; return 1;
}
int CC_SHA256_Update(CC_SHA256_CTX *c, const void *d_, CC_LONG n) {
    const uint8_t *d = d_; c->len += n;
    while (n) { uint32_t k = 64 - c->n < n ? 64 - c->n : n; memcpy(c->buf + c->n, d, k); c->n += k; d += k; n -= k; if (c->n == 64) { sha_block(c, c->buf); c->n = 0; } }
    return 1;
}
int CC_SHA256_Final(unsigned char *md, CC_SHA256_CTX *c) {
    uint64_t bits = c->len * 8; uint8_t pad = 0x80; CC_SHA256_Update(c, &pad, 1);
    pad = 0; while (c->n != 56) CC_SHA256_Update(c, &pad, 1);
    uint8_t L[8]; for (int i = 0; i < 8; i++) L[i] = (uint8_t)(bits >> (56 - 8 * i));
    CC_SHA256_Update(c, L, 8);
    for (int i = 0; i < 8; i++) { md[4*i] = (uint8_t)(c->h[i] >> 24); md[4*i+1] = (uint8_t)(c->h[i] >> 16); md[4*i+2] = (uint8_t)(c->h[i] >> 8); md[4*i+3] = (uint8_t)c->h[i]; }
    return 1;
}
unsigned char *CC_SHA256(const void *data, CC_LONG len, unsigned char *md) {
    CC_SHA256_CTX c; CC_SHA256_Init(&c); CC_SHA256_Update(&c, data, len); CC_SHA256_Final(md, &c); return md;
}

// CFMutableDictionary (NULL callbacks): open addressing keyed by pointer value, mutex-free (callers hold their own lock)
#include <CoreFoundation/CoreFoundation.h>
struct dm_cfdict { const void **k, **v; size_t cap, n, tomb; };
static const void *const CF_TOMB = (const void *)1;
static size_t cf_hash(const void *p, size_t cap) { uint64_t x = (uint64_t)(uintptr_t)p; x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; return (size_t)x & (cap - 1); }
static void cf_grow(struct dm_cfdict *d) {
    size_t oc = d->cap; const void **ok = d->k, **ov = d->v;
    d->cap = oc ? oc * 2 : 64; d->k = calloc(d->cap, sizeof *d->k); d->v = calloc(d->cap, sizeof *d->v); d->n = d->tomb = 0;
    for (size_t i = 0; i < oc; i++) if (ok[i] && ok[i] != CF_TOMB) CFDictionarySetValue(d, ok[i], ov[i]);
    free(ok); free(ov);
}
CFMutableDictionaryRef CFDictionaryCreateMutable(CFAllocatorRef a, CFIndex cap, const void *kcb, const void *vcb) { (void)a; (void)cap; (void)kcb; (void)vcb; struct dm_cfdict *d = calloc(1, sizeof *d); cf_grow(d); return d; }
const void *CFDictionaryGetValue(CFDictionaryRef d, const void *k) {
    for (size_t i = cf_hash(k, d->cap), s = 0; s < d->cap; i = (i + 1) & (d->cap - 1), s++) { if (!d->k[i]) return 0; if (d->k[i] == k) return d->v[i]; }
    return 0;
}
void CFDictionarySetValue(CFMutableDictionaryRef d, const void *k, const void *v) {
    if ((d->n + d->tomb + 1) * 4 > d->cap * 3) cf_grow(d);
    size_t t = (size_t)-1;
    for (size_t i = cf_hash(k, d->cap);; i = (i + 1) & (d->cap - 1)) {
        if (d->k[i] == k) { d->v[i] = v; return; }
        if (d->k[i] == CF_TOMB) { if (t == (size_t)-1) t = i; continue; }
        if (!d->k[i]) { if (t != (size_t)-1) { i = t; d->tomb--; } d->k[i] = k; d->v[i] = v; d->n++; return; }
    }
}
void CFDictionaryRemoveValue(CFMutableDictionaryRef d, const void *k) {
    for (size_t i = cf_hash(k, d->cap), s = 0; s < d->cap; i = (i + 1) & (d->cap - 1), s++) { if (!d->k[i]) return; if (d->k[i] == k) { d->k[i] = CF_TOMB; d->v[i] = 0; d->n--; d->tomb++; return; } }
}
CFIndex CFDictionaryGetCount(CFDictionaryRef d) { return (CFIndex)d->n; }
