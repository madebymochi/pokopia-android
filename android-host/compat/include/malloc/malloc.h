// compat: Darwin malloc introspection and zones over bionic malloc (a "zone" is just the process heap).
#pragma once
#include <malloc.h>
#include <stdlib.h>
static inline size_t malloc_size(const void *p) { return p ? malloc_usable_size((void *)p) : 0; }
static inline size_t malloc_good_size(size_t n) { return (n + 15) & ~(size_t)15; }
typedef struct { int unused; } malloc_zone_t;
static inline malloc_zone_t *malloc_create_zone(size_t start, unsigned flags) { (void)start; (void)flags; static malloc_zone_t z; return &z; }
static inline void malloc_set_zone_name(malloc_zone_t *z, const char *n) { (void)z; (void)n; }
static inline void *malloc_zone_malloc(malloc_zone_t *z, size_t n) { (void)z; return malloc(n); }
static inline void *malloc_zone_memalign(malloc_zone_t *z, size_t a, size_t n) { (void)z; return memalign(a, n); }
static inline void *malloc_zone_realloc(malloc_zone_t *z, void *p, size_t n) { (void)z; return realloc(p, n); }
static inline void malloc_zone_free(malloc_zone_t *z, void *p) { (void)z; free(p); }
