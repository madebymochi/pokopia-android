// compat: getsectiondata() over ELF. The host's tables are ELF sections named like C identifiers (hle.h DM_SECTION), so the linker defines
// __start_<name> / __stop_<name>. Only the host executable's own sections are looked up; the segment name is ignored.
#pragma once
#include <stdint.h>
#include <string.h>
struct mach_header_64 { uint32_t magic; };
extern const struct mach_header_64 _mh_execute_header;   // compat/src/darwin_compat.c: an unused stand-in
#define DM_SECT_DECL(s) extern const uint8_t __start_##s[] __attribute__((weak)), __stop_##s[] __attribute__((weak));
DM_SECT_DECL(swhle) DM_SECT_DECL(swforce) DM_SECT_DECL(swnvn)
static inline const uint8_t *getsectiondata(const struct mach_header_64 *mh, const char *seg, const char *sect, unsigned long *size) {
    (void)mh; (void)seg; const uint8_t *a = 0, *b = 0;
    if (!strcmp(sect, "swhle")) { a = __start_swhle; b = __stop_swhle; }
    else if (!strcmp(sect, "swforce")) { a = __start_swforce; b = __stop_swforce; }
    else if (!strcmp(sect, "swnvn")) { a = __start_swnvn; b = __stop_swnvn; }
    *size = a ? (unsigned long)(b - a) : 0;
    return a;
}
