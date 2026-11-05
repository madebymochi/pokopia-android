// compat: dispatch semaphores and times over POSIX semaphores, and dispatch_after as a detached sleeping thread. No queues or sources: blocks need a runtime the NDK does not
// ship (no -fblocks, no libBlocksRuntime), so dispatch_after takes a plain function here (dispatch_after_f is the real API for it) and hle_lifecycle.c's block and
// dispatch_source code has to be #ifdef __APPLE__ with a pthread path beside it.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#ifndef NSEC_PER_MSEC
#define NSEC_PER_MSEC 1000000ull
#endif
#ifndef NSEC_PER_SEC
#define NSEC_PER_SEC 1000000000ull
#endif
#ifndef NSEC_PER_USEC
#define NSEC_PER_USEC 1000ull
#endif
typedef uint64_t dispatch_time_t;   // absolute CLOCK_REALTIME nanoseconds (sem_timedwait's clock); ~0 = forever
#define DISPATCH_TIME_NOW 0ull
#define DISPATCH_TIME_FOREVER (~0ull)
typedef struct dm_dsem { sem_t s; } *dispatch_semaphore_t;
static inline dispatch_time_t dispatch_time(dispatch_time_t when, int64_t delta) {
    if (when == DISPATCH_TIME_FOREVER) return when;
    if (when == DISPATCH_TIME_NOW) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); when = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
    return delta < 0 && (uint64_t)-delta > when ? 1 : when + (uint64_t)delta;
}
static inline dispatch_semaphore_t dispatch_semaphore_create(long v) { dispatch_semaphore_t d = calloc(1, sizeof *d); sem_init(&d->s, 0, (unsigned)v); return d; }
static inline long dispatch_semaphore_signal(dispatch_semaphore_t d) { sem_post(&d->s); return 0; }
static inline long dispatch_semaphore_wait(dispatch_semaphore_t d, dispatch_time_t t) {
    if (t == DISPATCH_TIME_FOREVER) { while (sem_wait(&d->s)) if (errno != EINTR) return -1; return 0; }
    struct timespec ts = {(time_t)(t / 1000000000ull), (long)(t % 1000000000ull)};
    while (sem_timedwait(&d->s, &ts)) { if (errno != EINTR) return -1; }
    return 0;
}
static inline void dispatch_release(void *o) { dispatch_semaphore_t d = o; if (d) { sem_destroy(&d->s); free(d); } }
// dispatch_after: one detached thread per call sleeps until the deadline (CLOCK_REALTIME, as dispatch_time counts), then runs the work. The queue is ignored. FOREVER never runs it.
typedef struct dm_dq *dispatch_queue_t;
typedef void (*dispatch_function_t)(void *);
static inline dispatch_queue_t dispatch_get_global_queue(long id, unsigned long flags) { static struct { int unused; } q; (void)id; (void)flags; return (dispatch_queue_t)&q; }
struct dm_after { dispatch_time_t when; void *ctx; dispatch_function_t fn; };
static inline void *dm_after_run(void *p) {
    struct dm_after a = *(struct dm_after *)p; free(p);
    struct timespec ts = {(time_t)(a.when / 1000000000ull), (long)(a.when % 1000000000ull)};
    while (clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
    a.fn(a.ctx); return NULL;
}
static inline void dispatch_after_f(dispatch_time_t when, dispatch_queue_t q, void *ctx, dispatch_function_t fn) {
    (void)q; if (when == DISPATCH_TIME_FOREVER || !fn) return;
    struct dm_after *a = malloc(sizeof *a); if (!a) return;
    a->when = when; a->ctx = ctx; a->fn = fn;
    pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
    pthread_t t; if (pthread_create(&t, &at, dm_after_run, a)) free(a);
    pthread_attr_destroy(&at);
}
#ifndef __BLOCKS__
typedef void (*dispatch_block_t)(void);
static inline void dm_after_call(void *f) { ((dispatch_block_t)f)(); }
static inline void dispatch_after(dispatch_time_t when, dispatch_queue_t q, dispatch_block_t work) { dispatch_after_f(when, q, (void *)work, dm_after_call); }
#endif
