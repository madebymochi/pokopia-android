// compat: Darwin xattr calls (extra position / options arguments) over Linux's. Linux user attributes need the "user." prefix.
#pragma once
#include_next <sys/xattr.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#ifndef ENOATTR
#define ENOATTR ENODATA
#endif
#define XATTR_NOFOLLOW 0x0001
static inline const char *dm_xattr_name(const char *n, char *b, size_t cap) { if (!strncmp(n, "user.", 5)) return n; snprintf(b, cap, "user.%s", n); return b; }
static inline ssize_t dm_getxattr(const char *p, const char *n, void *v, size_t s, unsigned pos, int o) {
    char b[256]; (void)pos; n = dm_xattr_name(n, b, sizeof b); return (o & XATTR_NOFOLLOW) ? lgetxattr(p, n, v, s) : getxattr(p, n, v, s); }
static inline int dm_setxattr(const char *p, const char *n, const void *v, size_t s, unsigned pos, int o) {
    char b[256]; (void)pos; n = dm_xattr_name(n, b, sizeof b); return (o & XATTR_NOFOLLOW) ? lsetxattr(p, n, v, s, 0) : setxattr(p, n, v, s, 0); }
static inline int dm_removexattr(const char *p, const char *n, int o) {
    char b[256]; n = dm_xattr_name(n, b, sizeof b); return (o & XATTR_NOFOLLOW) ? lremovexattr(p, n) : removexattr(p, n); }
#define getxattr(p, n, v, s, pos, o) dm_getxattr(p, n, v, s, pos, o)
#define setxattr(p, n, v, s, pos, o) dm_setxattr(p, n, v, s, pos, o)
#define removexattr(p, n, o) dm_removexattr(p, n, o)
