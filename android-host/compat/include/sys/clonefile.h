// compat: clonefile() = recursive copy (no copy-on-write on f2fs / ext4); the destination must not exist.
#pragma once
#include <stdint.h>
int clonefile(const char *src, const char *dst, uint32_t flags);
