// compat: mach_vm_* for the host's write watch (probe.c, DM_PROBE_WATCH: a Mac debugging aid whose fault handler reads Darwin's mcontext). Not emulated: both fail like
// vm_region_64 / vm_remap in mach.h, so the watch never arms and the game runs as without it.
#pragma once
#include <mach/mach.h>
#ifdef __cplusplus
extern "C" {
#endif
static inline kern_return_t mach_vm_region(vm_map_t t, mach_vm_address_t *a, mach_vm_size_t *s, int f, vm_region_info_t i, mach_msg_type_number_t *n, mach_port_t *o) {
    (void)t; (void)a; (void)s; (void)f; (void)i; (void)n; (void)o; return KERN_NOT_SUPPORTED;
}
static inline kern_return_t mach_vm_remap(vm_map_t t, mach_vm_address_t *to, mach_vm_size_t size, mach_vm_address_t mask, int flags, vm_map_t st, mach_vm_address_t from, boolean_t copy, vm_prot_t *cur, vm_prot_t *max, vm_inherit_t inh) {
    (void)t; (void)to; (void)size; (void)mask; (void)flags; (void)st; (void)from; (void)copy; (void)cur; (void)max; (void)inh; return KERN_NOT_SUPPORTED;
}
#ifdef __cplusplus
}
#endif
