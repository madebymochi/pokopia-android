// compat: no sysctl on Android; every query fails (callers fall back).
#pragma once
#include <stddef.h>
#include <sys/time.h>
#include <errno.h>
#define CTL_KERN 1
#define KERN_PROC 14
#define KERN_PROC_PID 1
struct kinfo_proc { struct { struct timeval p_starttime; } kp_proc; };
static inline int sysctl(int *mib, unsigned n, void *o, size_t *on, void *nv, size_t nn) { (void)mib; (void)n; (void)o; (void)on; (void)nv; (void)nn; errno = ENOSYS; return -1; }
static inline int sysctlbyname(const char *nm, void *o, size_t *on, void *nv, size_t nn) { (void)nm; (void)o; (void)on; (void)nv; (void)nn; errno = ENOSYS; return -1; }
