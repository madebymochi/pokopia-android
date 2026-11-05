// compat: CFMutableDictionary with NULL callbacks only (a pointer -> pointer table), for hle_nvn.c's program key table.
#pragma once
#include <stddef.h>
typedef const void *CFAllocatorRef;
typedef long CFIndex;
typedef struct dm_cfdict *CFMutableDictionaryRef;
typedef const struct dm_cfdict *CFDictionaryRef;
CFMutableDictionaryRef CFDictionaryCreateMutable(CFAllocatorRef a, CFIndex cap, const void *kcb, const void *vcb);
const void *CFDictionaryGetValue(CFDictionaryRef d, const void *k);
void CFDictionarySetValue(CFMutableDictionaryRef d, const void *k, const void *v);
void CFDictionaryRemoveValue(CFMutableDictionaryRef d, const void *k);
CFIndex CFDictionaryGetCount(CFDictionaryRef d);
