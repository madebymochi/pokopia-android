// compat: forced include (-include) for the Android / Linux build of nxnative's host: Darwin libc extensions that have no header of their own here.
#pragma once
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/param.h>
#include <string.h>
#include <pthread/qos.h>   // Darwin's pthread.h brings the QoS API in
#ifdef __cplusplus
extern "C" {
#endif
// Darwin clocks: UPTIME_RAW / MONOTONIC_RAW map to Linux's monotonic clocks
#ifndef CLOCK_UPTIME_RAW
#define CLOCK_UPTIME_RAW CLOCK_MONOTONIC
#endif
#ifndef CLOCK_MONOTONIC_RAW_APPROX
#define CLOCK_MONOTONIC_RAW_APPROX CLOCK_MONOTONIC
#endif
static inline uint64_t clock_gettime_nsec_np(clockid_t c) { struct timespec t; if (clock_gettime(c, &t)) return 0; return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static inline int getdtablesize(void) { return (int)sysconf(_SC_OPEN_MAX); }
// Darwin's struct stat names its time fields differently
#define st_mtimespec st_mtim
// BSD libc's stable sort
int mergesort(void *base, size_t n, size_t size, int (*cmp)(const void *, const void *));
// Darwin's stack address is the top (highest address) of the stack
void *pthread_get_stackaddr_np(pthread_t t);
size_t pthread_get_stacksize_np(pthread_t t);
// fcntl(F_GETPATH) has no Linux equivalent: darwin_compat.c reads /proc/self/fd
#define F_GETPATH 50
int dm_fcntl_getpath(int fd, char *out);
// renamex_np(RENAME_SWAP) = renameat2(RENAME_EXCHANGE)
#define RENAME_SWAP 0x2
#define RENAME_EXCL 0x4
int renamex_np(const char *from, const char *to, unsigned flags);
// Darwin's pthread_setname_np names the calling thread; Linux takes the thread and limits names to 15 bytes
static inline int dm_pthread_setname_self(const char *n) { char b[16]; strncpy(b, n, 15); b[15] = 0; return pthread_setname_np(pthread_self(), b); }
#define pthread_setname_np(n) dm_pthread_setname_self(n)
static inline int pthread_cond_timedwait_relative_np(pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *rel) {
    struct timespec t; clock_gettime(CLOCK_REALTIME, &t);
    t.tv_sec += rel->tv_sec; t.tv_nsec += rel->tv_nsec; if (t.tv_nsec >= 1000000000) { t.tv_sec++; t.tv_nsec -= 1000000000; }
    return pthread_cond_timedwait(c, m, &t);
}
#ifdef __cplusplus
}
#endif
