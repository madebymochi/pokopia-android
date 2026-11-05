// compat: Mach absolute time on Linux = CLOCK_MONOTONIC nanoseconds (timebase 1/1).
#pragma once
#include <stdint.h>
#include <time.h>
#include <mach/mach.h>
typedef struct { uint32_t numer, denom; } mach_timebase_info_data_t, *mach_timebase_info_t;
uint64_t mach_absolute_time(void);   // a real function: nvn_entry.S calls it
static inline kern_return_t mach_timebase_info(mach_timebase_info_t i) { i->numer = 1; i->denom = 1; return KERN_SUCCESS; }
static inline kern_return_t mach_wait_until(uint64_t deadline) {
    struct timespec ts = {(time_t)(deadline / 1000000000ull), (long)(deadline % 1000000000ull)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == 4 /* EINTR */) {}
    return KERN_SUCCESS;
}
