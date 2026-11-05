// compat: the subset of Mach the nxnative host uses, on Linux / Android.
// - mach_task_self / vm_read_overwrite: a memory read that fails instead of faulting (process_vm_readv on our own pid).
// - vm_allocate / vm_deallocate: anonymous mmap.
// - thread ports are Linux thread IDs; thread_info(THREAD_BASIC_INFO) reads /proc/self/task/<tid>/stat.
// - Remote thread suspension and register reads (crash / hang dumps), task_info, thread policies and vm_remap are not available:
//   they return KERN_FAILURE and the callers already treat that as "no information".
#pragma once
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef int kern_return_t;
typedef unsigned int natural_t;
typedef int integer_t;
typedef int boolean_t;
typedef unsigned int mach_port_t;
typedef mach_port_t thread_t, thread_act_t, task_t, vm_map_t;
typedef mach_port_t *thread_act_array_t;
typedef uintptr_t vm_address_t, vm_offset_t, vm_size_t;
typedef uint64_t mach_vm_address_t, mach_vm_size_t;
typedef natural_t mach_msg_type_number_t;
typedef int vm_prot_t;
typedef unsigned int vm_inherit_t;
typedef integer_t *thread_info_t, *task_info_t, *thread_policy_t;
typedef natural_t *thread_state_t;
typedef natural_t thread_flavor_t, task_flavor_t, thread_policy_flavor_t;
typedef int policy_t;
#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif
#define KERN_SUCCESS 0
#define KERN_INVALID_ADDRESS 1
#define KERN_PROTECTION_FAILURE 2
#define KERN_NO_SPACE 3
#define KERN_INVALID_ARGUMENT 4
#define KERN_FAILURE 5
#define KERN_NOT_SUPPORTED 46
#define VM_FLAGS_FIXED 0x0
#define VM_FLAGS_ANYWHERE 0x1
#define VM_FLAGS_OVERWRITE 0x4000
#define VM_INHERIT_NONE 2
#define VM_PROT_NONE 0
#define VM_PROT_READ 1
#define VM_PROT_WRITE 2
#define VM_PROT_EXECUTE 4

typedef struct { integer_t seconds, microseconds; } time_value_t;
typedef struct { time_value_t user_time, system_time; integer_t cpu_usage; policy_t policy; integer_t run_state, flags, suspend_count, sleep_time; } thread_basic_info_data_t;
#define THREAD_BASIC_INFO 3
#define THREAD_BASIC_INFO_COUNT ((mach_msg_type_number_t)(sizeof(thread_basic_info_data_t) / sizeof(natural_t)))
#define TH_STATE_RUNNING 1
#define TH_STATE_STOPPED 2
#define TH_STATE_WAITING 3
#define TH_STATE_UNINTERRUPTIBLE 4
#define TH_STATE_HALTED 5
typedef struct { uint64_t pth_user_time, pth_system_time; int32_t pth_cpu_usage, pth_policy, pth_run_state, pth_flags, pth_sleep_time, pth_curpri, pth_priority, pth_maxpriority; char pth_name[64]; } thread_extended_info_data_t;
#define THREAD_EXTENDED_INFO 5
#define THREAD_EXTENDED_INFO_COUNT ((mach_msg_type_number_t)(sizeof(thread_extended_info_data_t) / sizeof(natural_t)))
typedef struct { uint64_t __x[29], __fp, __lr, __sp, __pc; uint32_t __cpsr, __pad; } arm_thread_state64_t;
#define ARM_THREAD_STATE64 6
#define ARM_THREAD_STATE64_COUNT ((mach_msg_type_number_t)(sizeof(arm_thread_state64_t) / sizeof(natural_t)))
#define arm_thread_state64_get_pc(s) ((s).__pc)
#define __darwin_arm_thread_state64_get_pc(s) ((s).__pc)
#define __darwin_arm_thread_state64_get_lr(s) ((s).__lr)
#define __darwin_arm_thread_state64_get_sp(s) ((s).__sp)
#define __darwin_arm_thread_state64_get_fp(s) ((s).__fp)
#define arm_thread_state64_get_lr(s) ((s).__lr)
#define arm_thread_state64_get_sp(s) ((s).__sp)
#define arm_thread_state64_get_fp(s) ((s).__fp)
typedef struct { uint32_t period, computation, constraint; boolean_t preemptible; } thread_time_constraint_policy_data_t;
#define THREAD_TIME_CONSTRAINT_POLICY 2
#define THREAD_TIME_CONSTRAINT_POLICY_COUNT ((mach_msg_type_number_t)(sizeof(thread_time_constraint_policy_data_t) / sizeof(integer_t)))
typedef struct { uint64_t virtual_size; integer_t region_count, page_size; uint64_t resident_size, resident_size_peak, device, device_peak, internal, internal_peak, external, external_peak, reusable, reusable_peak, purgeable_volatile_pmap, purgeable_volatile_resident, purgeable_volatile_virtual, compressed, compressed_peak, compressed_lifetime, phys_footprint; } task_vm_info_data_t;
#define TASK_VM_INFO 22
#define TASK_VM_INFO_COUNT ((mach_msg_type_number_t)(sizeof(task_vm_info_data_t) / sizeof(natural_t)))
typedef struct { uint64_t total_user, total_system, task_interrupt_wakeups, task_platform_idle_wakeups, task_timer_wakeups_bin_1, task_timer_wakeups_bin_2; } task_power_info_data_t;
typedef struct { uint64_t task_gpu_utilisation, task_gpu_stat_reserved0, task_gpu_stat_reserved1, task_gpu_stat_reserved2; } gpu_energy_data;
typedef struct { task_power_info_data_t cpu_energy; gpu_energy_data gpu_energy; uint64_t task_ptime, task_pset_switches; uint64_t task_energy; } task_power_info_v2_data_t;
#define TASK_POWER_INFO_V2 26
#define TASK_POWER_INFO_V2_COUNT ((mach_msg_type_number_t)(sizeof(task_power_info_v2_data_t) / sizeof(natural_t)))

static inline mach_port_t mach_task_self(void) { return (mach_port_t)getpid(); }
static inline mach_port_t mach_thread_self(void) { return (mach_port_t)gettid(); }
static inline mach_port_t pthread_mach_thread_np(pthread_t t) { return (mach_port_t)pthread_gettid_np(t); }
static inline kern_return_t mach_port_deallocate(mach_port_t task, mach_port_t p) { (void)task; (void)p; return KERN_SUCCESS; }

kern_return_t vm_read_overwrite(vm_map_t task, vm_address_t addr, vm_size_t size, vm_address_t data, vm_size_t *out_size);
static inline kern_return_t vm_allocate(vm_map_t task, vm_address_t *addr, vm_size_t size, int flags) {
    (void)task; void *p = mmap((flags & VM_FLAGS_ANYWHERE) ? NULL : (void *)*addr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | ((flags & VM_FLAGS_ANYWHERE) ? 0 : MAP_FIXED), -1, 0);
    if (p == MAP_FAILED) return KERN_NO_SPACE;
    *addr = (vm_address_t)p; return KERN_SUCCESS;
}
#define vm_page_size ((vm_size_t)getpagesize())   // Mach exports it as a variable; here the kernel's page size (4 or 16 KiB)
static inline kern_return_t vm_deallocate(vm_map_t task, vm_address_t addr, vm_size_t size) { (void)task; return munmap((void *)addr, size) ? KERN_INVALID_ADDRESS : KERN_SUCCESS; }
static inline kern_return_t vm_remap(vm_map_t t, vm_address_t *to, vm_size_t size, vm_address_t mask, int flags, vm_map_t st, vm_address_t from, boolean_t copy, vm_prot_t *cur, vm_prot_t *max, vm_inherit_t inh) {
    (void)t; (void)to; (void)size; (void)mask; (void)flags; (void)st; (void)from; (void)copy; (void)cur; (void)max; (void)inh; return KERN_NOT_SUPPORTED;
}
#define MACH_PORT_NULL 0u
typedef struct { vm_prot_t protection, max_protection; vm_inherit_t inheritance; boolean_t shared, reserved; uint64_t offset; int behavior; unsigned short user_wired_count; } vm_region_basic_info_data_64_t;
typedef int *vm_region_info_t;
#define VM_REGION_BASIC_INFO_64 9
#define VM_REGION_BASIC_INFO_COUNT_64 ((mach_msg_type_number_t)(sizeof(vm_region_basic_info_data_64_t) / sizeof(int)))
static inline kern_return_t vm_region_64(vm_map_t t, vm_address_t *a, vm_size_t *s, int f, vm_region_info_t i, mach_msg_type_number_t *n, mach_port_t *o) {
    (void)t; (void)a; (void)s; (void)f; (void)i; (void)n; (void)o; return KERN_NOT_SUPPORTED;
}
kern_return_t thread_info(thread_act_t th, thread_flavor_t flavor, thread_info_t out, mach_msg_type_number_t *count);
static inline kern_return_t thread_suspend(thread_act_t th) { (void)th; return KERN_NOT_SUPPORTED; }
static inline kern_return_t thread_resume(thread_act_t th) { (void)th; return KERN_NOT_SUPPORTED; }
static inline kern_return_t thread_get_state(thread_act_t th, thread_flavor_t f, thread_state_t s, mach_msg_type_number_t *n) { (void)th; (void)f; (void)s; (void)n; return KERN_NOT_SUPPORTED; }
static inline kern_return_t task_threads(task_t t, thread_act_array_t *a, mach_msg_type_number_t *n) { (void)t; *a = 0; *n = 0; return KERN_NOT_SUPPORTED; }
static inline kern_return_t task_info(task_t t, task_flavor_t f, task_info_t o, mach_msg_type_number_t *n) { (void)t; (void)f; (void)o; (void)n; return KERN_NOT_SUPPORTED; }
static inline kern_return_t thread_policy_set(thread_act_t th, thread_policy_flavor_t f, thread_policy_t p, mach_msg_type_number_t n) { (void)th; (void)f; (void)p; (void)n; return KERN_NOT_SUPPORTED; }
static inline kern_return_t thread_policy_get(thread_act_t th, thread_policy_flavor_t f, thread_policy_t p, mach_msg_type_number_t *n, boolean_t *def) { (void)th; (void)f; (void)p; (void)n; if (def) *def = 1; return KERN_NOT_SUPPORTED; }
int thread_selfcounts(int type, void *buf, size_t nbytes);   // {instructions, cycles}: zeros (no PMU access for apps)
#ifdef __cplusplus
}
#endif
