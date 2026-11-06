// Vulkan replay of nxnative's NVN command records for the Android host: the interface hle_nvn.c expects from the Metal replay (dm/gfx/nvn/nvn_mtl.m), on Android's Vulkan 1.3.
// Written for the Galaxy S22's Xclipse 920 (features checked on the device): host-pointer import, dynamic rendering, extended dynamic state 1-3, vertex input dynamic state,
// push descriptors, depth clip control, null descriptors. No BC4-BC7 (decoded on the CPU), no D24S8 (D32S8 instead).
//
// Design (the Metal replay's TBDR optimizations are left out):
//   * memory pools are imported as VkDeviceMemory over the guest's own pages (VK_EXT_external_memory_host), one VkBuffer each: buffer addresses are guest pointers, so every UBO / SSBO /
//     vertex / index / copy address resolves to (pool buffer, offset) and GPU writes land in guest memory like on the Switch;
//   * textures are VkImages keyed by tex_t serial, all kept in VK_IMAGE_LAYOUT_GENERAL; sampled textures are deswizzled from guest memory (nvn_deswizzle.c) on first use unless the GPU
//     wrote them (render target, copy / clear destination);
//   * one primary command buffer per submission plus an upload command buffer submitted before it; a full memory barrier after every render pass and transfer;
//   * compute: OP_DISPATCH runs the stage-5 program as a compute pipeline (one descriptor set 0 of its own), outside render passes, see "Compute" below;
//   * shaders are the offline-patched SPIR-V of android-host/tools/pokopia_vkshaders.py, in one file <DM_VK_SHADERS>/shaders.pack (index, then per shader its binding table, c1 and compressed code);
//     sets: vertex 0, pixel 1 (the push descriptor set), geometry 2, hull 3, domain 4; pipelines keyed by (the five shaders, attachment formats, samples, topology class, patch size), everything else dynamic;
//   * CPU-visible effects of a submission (events, counter reports) run on a completion thread when its timeline value is reached.
// Present: with a window (dm_vk_set_window, the APK front) the window texture is blitted letterboxed into the swapchain; without one, DM_VK_SHOT=n[,n...] saves those presents as PPM files
// (DM_VK_SHOT_DIR, default DM_TEMP) so the adb executable can be checked offscreen.
#define VK_USE_PLATFORM_ANDROID_KHR 1
#include "nvn_obj.h"
#include "dmsh_abi.h"
#include "../../host/dm.h"
#include "../tex/nvn_deswizzle.h"
#include "nvn_primitive.h"   // nvn_expand_primitive: indexed quads with primitive restart (upstream 7a3a875)
#include <vulkan/vulkan.h>
#include <android/native_window.h>
#include <zlib.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/prctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 1u
#endif
#ifndef MADV_REMOVE
#define MADV_REMOVE 9
#endif
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <math.h>

#define RED_ONCE(id, ...) do { static int once_##id; if (!once_##id++) LOG("[RED] " __VA_ARGS__); } while (0)
#define VKCHK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) die("vk: %s failed: %d", #x, (int)r_); } while (0)
#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// ---------- Vulkan entry points beyond the NDK stub's 1.1 exports ----------
#define VK_FNS(X) \
    X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdPipelineBarrier2) X(vkCmdSetCullMode) X(vkCmdSetFrontFace) X(vkCmdSetPrimitiveTopology) \
    X(vkCmdSetDepthTestEnable) X(vkCmdSetDepthWriteEnable) X(vkCmdSetDepthCompareOp) X(vkCmdSetDepthBoundsTestEnable) X(vkCmdSetStencilTestEnable) X(vkCmdSetStencilOp) \
    X(vkCmdSetRasterizerDiscardEnable) X(vkCmdSetDepthBiasEnable) X(vkCmdSetPrimitiveRestartEnable) X(vkCmdBindVertexBuffers2) X(vkQueueSubmit2) \
    X(vkWaitSemaphores) X(vkGetSemaphoreCounterValue) X(vkCmdPushDescriptorSetKHR) X(vkCmdSetVertexInputEXT) X(vkCmdSetPolygonModeEXT) X(vkCmdSetDepthClampEnableEXT) \
    X(vkCmdSetColorBlendEnableEXT) X(vkCmdSetColorBlendEquationEXT) X(vkCmdSetColorWriteMaskEXT) X(vkCmdSetAlphaToCoverageEnableEXT) X(vkGetMemoryHostPointerPropertiesEXT) \
    X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkQueuePresentKHR)
#define VK_DECL(f) static PFN_##f p##f;
VK_FNS(VK_DECL)

// ---------- Device ----------
static VkInstance g_inst; static VkPhysicalDevice g_pd; static VkDevice g_dev; static VkQueue g_q; static uint32_t g_qf;
static VkPhysicalDeviceMemoryProperties g_mp; static VkPhysicalDeviceLimits g_lim;
static pthread_mutex_t g_qmu = PTHREAD_MUTEX_INITIALIZER;   // queue access (submissions on the submit thread, presents on the game's present thread)
static int g_has_divisor, g_has_u8;
static _Atomic int g_depth_zero, g_origin_ll;   // g_origin_ll: the device's window origin (viewport flip, present flip)
static _Atomic int g_queue_ll;                  // the active queue's raster flip, captured at queue creation (mtl_set_queue_origin_lower_left): front-face winding only (nvn_mtl.m g_queue_origin_ll)
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < g_mp.memoryTypeCount; i++)
        if ((bits >> i & 1) && (g_mp.memoryTypes[i].propertyFlags & want) == want && !(g_mp.memoryTypes[i].propertyFlags & avoid)) return i;
    for (uint32_t i = 0; i < g_mp.memoryTypeCount; i++) if ((bits >> i & 1) && (g_mp.memoryTypes[i].propertyFlags & want) == want) return i;
    die("vk: no memory type for bits %#x flags %#x", bits, want);
}
static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }

static void dev_init(void);
static void ensure_init(void) { pthread_once(&g_init_once, dev_init); }

// ---------- Timeline, completion thread, post-completion effects ----------
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; } post_op;   // kind 1 event, 2 counter report
static VkSemaphore g_tl;
static _Atomic uint64_t g_fence;            // last fence committed (the replay: the submit thread, or the replay thread when async replay is on, the default)
static _Atomic uint64_t g_done;             // last fence the GPU completed (completion thread)
static pthread_mutex_t g_dmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_dcv = PTHREAD_COND_INITIALIZER;
// Pass profiler (DM_VK_PASSPROF, section "Pass profiler" below): one record per timestamp mark; a submission's records travel with its pend to the completion thread (ppr, ppn marks, ppwant refused marks)
enum { PK_UPLOAD, PK_GAP, PK_PASS, PK_COPY, PK_CLEAR, PK_DISP, PK_FLUSH, PK_TAIL, PK_END, PK_N };
typedef struct { uint8_t kind, nc, samples, layered; uint32_t w, h, df, cf[4], draws, clears, d0; char vs[20], fs[20]; } pp_rec;
typedef struct pend { uint64_t fence; post_op *ops; int n; struct pend *next; int q; uint64_t frame, t_commit; pp_rec *ppr; int ppn, ppwant, late; uint32_t d_pool, d_rc; float ms_chg, ms_del; } pend;   // d_*, ms_*: stall diagnostics (pool changes and recycled images since the previous commit, time since the last pool change / finalize)
// GPU time: a timestamp pair per replay context (top of its upload buffer, end of its command buffer), read by the completion thread and summed per frame (presents so far at commit)
static VkQueryPool g_tsq; static double g_ts_ns;   // timestampPeriod (ns per tick); 0 = no timestamps
static struct { _Atomic uint64_t frame, ns, sub, done; } g_gf[64];
static _Atomic uint64_t g_gpu_ns_all, g_lat_ns_all, g_nlat;
static uint64_t g_presents;
static void passprof_done(const pend *p);   // "Pass profiler" below
static uint64_t g_prev_end, g_gapb[6];   // completion thread: end timestamp of the previous submission; gaps < 0.5, < 2, < 5, < 10, < 20, >= 20 ms
static void gpu_time_done(const pend *p) {
    if (!g_tsq || p->q < 0) return;
    uint64_t ts[2] = {0, 0};
    VkResult qr = vkGetQueryPoolResults(g_dev, g_tsq, 2 * p->q, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT);
    if (ts[0] && g_prev_end && ts[0] >= g_prev_end) {
        double gap = (ts[0] - g_prev_end) * g_ts_ns / 1e6; g_gapb[gap < 0.5 ? 0 : gap < 2 ? 1 : gap < 5 ? 2 : gap < 10 ? 3 : gap < 20 ? 4 : 5]++;
        if (p->late) LOG("[RED] vk: submission %llu: the GPU was idle %.3f ms before it started (gaps so far: <0.5 ms %llu, <2 %llu, <5 %llu, <10 %llu, <20 %llu, more %llu)", (unsigned long long)p->fence, gap,
            (unsigned long long)g_gapb[0], (unsigned long long)g_gapb[1], (unsigned long long)g_gapb[2], (unsigned long long)g_gapb[3], (unsigned long long)g_gapb[4], (unsigned long long)g_gapb[5]);
    }
    g_prev_end = qr == VK_SUCCESS && ts[1] >= ts[0] ? ts[1] : 0;
    if (qr == VK_SUCCESS && ts[1] >= ts[0]) {
        uint64_t ns = (uint64_t)((ts[1] - ts[0]) * g_ts_ns); g_gf[p->frame % 64].ns += ns; g_gpu_ns_all += ns;
        // a single submission that ran long on the GPU itself (start to end timestamps), against the time from commit to completion: tells a long-running shader from work that waited before it started
        if (ns > 200000000ull) LOG("[RED] vk: submission %llu (frame %llu) ran %.1f ms on the GPU (commit to completion %.1f ms)", (unsigned long long)p->fence, (unsigned long long)p->frame, ns / 1e6, (now_ns() - p->t_commit) / 1e6);
    } else LOG("vk: submission %llu: GPU timestamps unusable (%d, %llu .. %llu)", (unsigned long long)p->fence, (int)qr, (unsigned long long)ts[0], (unsigned long long)ts[1]);
    if (p->ppr) passprof_done(p);   // DM_VK_PASSPROF: the intervals between the marks of this submission (after the pair above, so its g_gpu_ns_all is the reference for the same submissions)
    g_lat_ns_all += now_ns() - p->t_commit; g_nlat++;
    g_gf[p->frame % 64].done++;
}
static pend *g_pend_head, *g_pend_tail; static pthread_mutex_t g_pmu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_pcv = PTHREAD_COND_INITIALIZER;
static uint64_t gpu_ticks(void) { uint64_t g = guest_ns(); if (!g) g = now_ns(); return dm_sdk_ounce() ? g / 32 : g * 12 / 625; }   // 19.2 MHz counter timebase, from the guest clock (as nvn_mtl.m)
static int pool_live(uint64_t addr, uint64_t len);   // below, with the pool table
static void *completion_main(void *arg) {
    (void)arg; pthread_setname_np("vk-complete");
    for (;;) {
        pthread_mutex_lock(&g_pmu);
        while (!g_pend_head) pthread_cond_wait(&g_pcv, &g_pmu);
        pend *p = g_pend_head; pthread_mutex_unlock(&g_pmu);
        VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_tl, .pValues = &p->fence};
        VkResult r = pvkWaitSemaphores(g_dev, &wi, 5000000000ull);
        if (r == VK_TIMEOUT) { LOG("[RED] vk: submission %llu has not completed after 5 s (GPU hang?); while it was recorded: %u memory pool changes, %u images recycled; last pool change %.1f ms, last pool finalize %.1f ms before its commit", (unsigned long long)p->fence, p->d_pool, p->d_rc, p->ms_chg, p->ms_del); p->late = 1; continue; }
        if (p->late) LOG("vk: submission %llu completed %.1f ms after commit (frame %llu)", (unsigned long long)p->fence, (now_ns() - p->t_commit) / 1e6, (unsigned long long)p->frame);
        if (r) { LOG("[RED] vk: waiting for submission %llu failed: %d (device lost?)", (unsigned long long)p->fence, (int)r); }
        if (!r) gpu_time_done(p);
        for (int i = 0; i < p->n; i++) {
            post_op *o = &p->ops[i];
            if (!pool_live(o->kind == 1 ? (uint64_t)(uintptr_t)o->val : o->addr, o->kind == 1 ? 4 : 16)) {   // events and reports live in memory pools: a target in no live pool (finalized, or a stale address) is guest heap now
                static _Atomic int nbad; int k = ++nbad;
                if (k <= 8 || !(k & (k - 1))) LOG("[RED] vk: %s write to %#llx after submission %llu is in no live memory pool: dropped (#%d)", o->kind == 1 ? "event" : "counter report",
                    (unsigned long long)(o->kind == 1 ? (uint64_t)(uintptr_t)o->val : o->addr), (unsigned long long)p->fence, k);
                continue;
            }
            if (o->kind == 1) { if (o->mode == 1) *o->val += o->v; else *o->val = o->v; }
            else { uint64_t t = gpu_ticks(), rec[2] = {(o->type == 0 || o->type == 0x10) ? (DM_GAME_RENDER_COND ? 0 : t) : o->type == 1 ? 1000000 : 0, t}; memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec); }   // samples passed: "visible" (no occlusion queries yet)
        }
        pthread_mutex_lock(&g_pmu); g_pend_head = p->next; if (!g_pend_head) g_pend_tail = NULL; pthread_mutex_unlock(&g_pmu);
        pthread_mutex_lock(&g_dmu); if (p->fence > g_done) atomic_store(&g_done, p->fence); pthread_cond_broadcast(&g_dcv); pthread_mutex_unlock(&g_dmu);
        free(p->ops); free(p->ppr); free(p);
    }
    return NULL;
}
int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    if (atomic_load(&g_done) >= fence) return 1;
    if (!timeout_ns) return 0;
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    uint64_t ns = (uint64_t)dl.tv_nsec + MIN(timeout_ns, 3600000000000ull); dl.tv_sec += ns / 1000000000ull; dl.tv_nsec = ns % 1000000000ull;
    pthread_mutex_lock(&g_dmu);
    while (atomic_load(&g_done) < fence) if (pthread_cond_timedwait(&g_dcv, &g_dmu, &dl) == ETIMEDOUT) break;
    int ok = atomic_load(&g_done) >= fence; pthread_mutex_unlock(&g_dmu); return ok;
}
uint64_t mtl_last_fence(void) { return g_fence; }
void mtl_finish(void) { mtl_wait(g_fence, ~0ull); }
void mtl_drain(void) { mtl_finish(); }

// ---------- Deferred destruction (objects the GPU may still use) ----------
typedef struct dead { uint64_t fence; VkImage img; VkDeviceMemory mem; VkBuffer buf; VkBufferView bview; VkImageView views[16]; int nv; struct dead *next; } dead;
static dead *g_dead; static pthread_mutex_t g_deadmu = PTHREAD_MUTEX_INITIALIZER;
static void dead_add(dead *d) { d->fence = g_fence + 1; pthread_mutex_lock(&g_deadmu); d->next = g_dead; g_dead = d; pthread_mutex_unlock(&g_deadmu); }
static void dead_reap(void) {
    uint64_t done = atomic_load(&g_done); pthread_mutex_lock(&g_deadmu);
    for (dead **pp = &g_dead; *pp;) {
        dead *d = *pp;
        if (d->fence <= done) { *pp = d->next; for (int i = 0; i < d->nv; i++) vkDestroyImageView(g_dev, d->views[i], NULL);
            if (d->bview) vkDestroyBufferView(g_dev, d->bview, NULL);
            if (d->img) vkDestroyImage(g_dev, d->img, NULL); if (d->buf) vkDestroyBuffer(g_dev, d->buf, NULL); if (d->mem) vkFreeMemory(g_dev, d->mem, NULL); free(d); }
        else pp = &d->next;
    }
    pthread_mutex_unlock(&g_deadmu);
}

// ---------- Memory pools: guest pages imported as device memory ----------
// Importing a pool pins every page of it at once, and pinned memory can't be swapped (measured on the phone: the kernel's pin counters rise by the pool's size). This game creates 4,988 MB of pools at the start and used a few MB of them
// in its first 14 s, so a large pool is imported piece by piece, as the GPU first needs each part (lazy_chunk). Textures need no import: they are read from guest memory on the CPU
static uint64_t g_nsub;   // submissions so far (with the contexts below)
// Where the submit and present paths wait, in ns since the last report (submit and present threads)
enum { WT_CTX, WT_RING, WT_SUBMIT, WT_PRESENT, WT_PFENCE, WT_ACQUIRE, WT_QPRESENT, WT_COMPOSITE, WT_N };
static _Atomic uint64_t g_wt[WT_N];
#define WT_ADD(k, ns) atomic_fetch_add_explicit(&g_wt[k], (uint64_t)(ns), memory_order_relaxed)
typedef struct pchunk { struct pchunk *next; uint64_t off, size, serial, last_sub; VkBuffer buf; VkDeviceMemory dm; } pchunk;   // last_sub: the submission that last looked it up
typedef struct { pthread_mutex_t mu; _Atomic(pchunk *) *tab; pchunk *all; int gone; } plazy;   // tab: per chunk of the pool, a piece that starts at or before it and reaches furthest
typedef struct { const uint8_t *mem; uint64_t size; VkDeviceMemory dm; VkBuffer buf; uint64_t serial; plazy *lz; } vpool;   // serial: never reused, unlike the buffer handle (tbuf_view's key)
static vpool *g_pool; static int g_npool, g_cpool; static pthread_mutex_t g_pomu = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t g_pool_serial;
static _Atomic uint32_t g_pool_gen = 1;   // bumped under g_pomu whenever g_pool changes (add / delete): diagnostics
static _Atomic uint64_t g_pool_t_chg, g_pool_t_del;   // now_ns() of the last pool change / finalize (stall diagnostics)
// Lookups read an immutable sorted copy of g_pool, published on every change (psnap_publish, under g_pomu): no lock and no per-thread state on the lookup path (the mutex and the
// per-thread cache's TLS calls were ~9% of the replay thread). A replaced copy is freed 2 s later (a reader holds one for well under a microsecond)
typedef struct { int n; vpool e[]; } vpsnap;
static _Atomic(vpsnap *) g_psnap;
enum { PRETIRE = 256 };
static struct { vpsnap *s; uint64_t t; } g_pretire[PRETIRE]; static int g_npretire;
static void psnap_publish(void) {
    vpsnap *ns = malloc(sizeof *ns + (size_t)g_npool * sizeof *g_pool); if (!ns) die("vk: pool snapshot allocation failed");
    ns->n = g_npool; memcpy(ns->e, g_pool, (size_t)g_npool * sizeof *g_pool);
    vpsnap *old = atomic_exchange_explicit(&g_psnap, ns, memory_order_acq_rel);
    uint64_t now = now_ns(); int k = 0;
    for (int i = 0; i < g_npretire; i++) { if (now - g_pretire[i].t > 2000000000ull) free(g_pretire[i].s); else g_pretire[k++] = g_pretire[i]; }
    g_npretire = k;
    if (old) { if (g_npretire == PRETIRE) { free(g_pretire[0].s); memmove(&g_pretire[0], &g_pretire[1], (PRETIRE - 1) * sizeof *g_pretire); g_npretire--; }   // > 128 changes a second: the oldest goes early
        g_pretire[g_npretire].s = old; g_pretire[g_npretire].t = now; g_npretire++; }
}
static const VkBufferUsageFlags BUF_ALL = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
// DM_POOL_STOP=<MB> ends the run before an import that would take what is imported past that size: a kill may leave a process's pins behind until the phone restarts, a clean exit does not
enum { POOL_LAZY_MIN = 32u << 20 };
static uint64_t g_chunk = 64u << 10;   // DM_POOL_CHUNK_KB. Measured in the world, pool memory pinned / pieces / replay CPU: 1 MB 312 MB / 197 / 10.9 ms, 256 KB 190 MB / 458 / 11.1 ms, 64 KB 150 MB / 1,395 / 11.2 ms, at the same frame time
enum { POOL_NOLEN = 4u << 20 };   // room from an address that comes with no usable length
static _Atomic uint64_t g_pool_total; static _Atomic int g_pool_pieces;
void dm_log_drain(void);
static uint64_t pool_charge(uint64_t size) {
    uint64_t total = atomic_fetch_add(&g_pool_total, size) + size;
    const char *e = getenv("DM_POOL_STOP"); if (e && total >> 20 > (uint64_t)atoll(e)) { LOG("vk: stopping the run before an import of %llu KB (DM_POOL_STOP %s MB)", (unsigned long long)(size >> 10), e); dm_log_drain(); _exit(0); }
    return total;
}
// GPU images by what fills them, in KB: on a phone they are system memory that can't be swapped. BC7 decoded to RGBA8 takes 4 times its size, BC6H decoded to RGBA16F 8 times, BC4 and BC5 twice. top: the largest level of large compressed textures
enum { IK_OTHER, IK_BC13, IK_BC7, IK_BC6, IK_BC45, IK_CONV, IK_N };
static _Atomic uint64_t g_img_kb[IK_N], g_img_n[IK_N], g_img_top_kb, g_img_side_kb[2][6];   // side: compressed images by largest side, [has mipmaps][4096 and up, 2048, 1024, 512, 256, smaller]
static int g_tex_drop_on;
static _Atomic uint32_t g_bc7_n[3];   // BC7 textures made BC1, BC3, RGBA8
static uint64_t g_ring_peak_seen(void), g_static_seen(void);
static void img_overlap_report(void);
void mtl_pool_report(void) {   // on the memory log's thread
    static uint64_t said; uint64_t t = atomic_load(&g_pool_total);
    if (t != said) { said = t; LOG("vk: %llu MB of memory pools imported, %d pieces", (unsigned long long)(t >> 20), atomic_load(&g_pool_pieces)); }
    static uint64_t said_img, said_img_t; uint64_t all = 0, img_now = now_ns(); for (int i = 0; i < IK_N; i++) all += atomic_load(&g_img_kb[i]);
    if (all > said_img + 32768 || all + 32768 < said_img || (all != said_img && img_now - said_img_t >= 30000000000ull)) { said_img = all; said_img_t = img_now;
        LOG("vk: images %llu MB: uncompressed %llu MB (%llu), BC1-3 %llu MB (%llu), BC7 %llu MB (%llu, of them to BC1 %u, to BC3 %u, stay RGBA8 %u), BC6H decoded %llu MB (%llu), BC4/5 as R8/RG8 %llu MB (%llu), others converted %llu MB (%llu); top levels of large compressed textures: %s %llu MB; static arena %llu MB; most ring in one submission, uploads aside: %llu KB", (unsigned long long)(all >> 10),
            (unsigned long long)(atomic_load(&g_img_kb[IK_OTHER]) >> 10), (unsigned long long)atomic_load(&g_img_n[IK_OTHER]), (unsigned long long)(atomic_load(&g_img_kb[IK_BC13]) >> 10), (unsigned long long)atomic_load(&g_img_n[IK_BC13]), (unsigned long long)(atomic_load(&g_img_kb[IK_BC7]) >> 10), (unsigned long long)atomic_load(&g_img_n[IK_BC7]), atomic_load(&g_bc7_n[0]), atomic_load(&g_bc7_n[1]), atomic_load(&g_bc7_n[2]),
            (unsigned long long)(atomic_load(&g_img_kb[IK_BC6]) >> 10), (unsigned long long)atomic_load(&g_img_n[IK_BC6]), (unsigned long long)(atomic_load(&g_img_kb[IK_BC45]) >> 10), (unsigned long long)atomic_load(&g_img_n[IK_BC45]), (unsigned long long)(atomic_load(&g_img_kb[IK_CONV]) >> 10), (unsigned long long)atomic_load(&g_img_n[IK_CONV]),
            g_tex_drop_on ? "left out, saving" : "kept, leaving out would save", (unsigned long long)(atomic_load(&g_img_top_kb) >> 10), (unsigned long long)(g_static_seen() >> 20), (unsigned long long)(g_ring_peak_seen() >> 10));
        #define SIDEMB(m, i) (unsigned long long)(atomic_load(&g_img_side_kb[m][i]) >> 10)
        LOG("vk: compressed images in MB by the guest's largest side, with mipmaps / without: 4096 %llu / %llu, 2048 %llu / %llu, 1024 %llu / %llu, 512 %llu / %llu, 256 %llu / %llu, smaller %llu / %llu",
            SIDEMB(1, 0), SIDEMB(0, 0), SIDEMB(1, 1), SIDEMB(0, 1), SIDEMB(1, 2), SIDEMB(0, 2), SIDEMB(1, 3), SIDEMB(0, 3), SIDEMB(1, 4), SIDEMB(0, 4), SIDEMB(1, 5), SIDEMB(0, 5));
        #undef SIDEMB
    }
    // Every 30 s: how much of what is imported the GPU still uses. Pinned pages can't be swapped, so pieces it stopped using are what freeing idle pieces would give back
    static uint64_t last_t, ticks[5]; uint64_t now = now_ns();
    if (now - last_t >= 30000000000ull) { last_t = now;
        int n = 0, n30 = 0, n120 = 0; uint64_t b = 0, b30 = 0, b120 = 0, uniq = 0;
        pthread_mutex_lock(&g_pomu);
        for (int i = 0; i < g_npool; i++) { plazy *z = g_pool[i].lz; if (!z) continue;
            pthread_mutex_lock(&z->mu);
            for (pchunk *c = z->all; c; c = c->next) { uint64_t ls = __atomic_load_n(&c->last_sub, __ATOMIC_RELAXED); n++; b += c->size;
                if (ls <= ticks[0]) { n30++; b30 += c->size; }
                if (ls <= ticks[3]) { n120++; b120 += c->size; } }
            for (uint64_t j = 0; j <= g_pool[i].size / g_chunk; j++) if (atomic_load_explicit(&z->tab[j], memory_order_relaxed)) uniq += g_chunk;
            pthread_mutex_unlock(&z->mu); }
        pthread_mutex_unlock(&g_pomu);
        memmove(ticks + 1, ticks, 4 * sizeof *ticks); ticks[0] = g_nsub;
        LOG("vk: imported pieces: %d, %llu MB, %llu MB of pool covered; not used for 30 s: %d, %llu MB; for 120 s: %d, %llu MB", n, (unsigned long long)(b >> 20), (unsigned long long)(uniq >> 20), n30, (unsigned long long)(b30 >> 20), n120, (unsigned long long)(b120 >> 20));
        img_overlap_report();
    }
}
static VkResult import_range(const void *mem, uint64_t size, VkBuffer *buf, VkDeviceMemory *dm) {
    *buf = 0; *dm = 0;
    VkMemoryHostPointerPropertiesEXT hp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult r = pvkGetMemoryHostPointerPropertiesEXT(g_dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, mem, &hp);
    VkExternalMemoryBufferCreateInfo eb = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &eb, .size = size, .usage = BUF_ALL};
    if (!r) r = vkCreateBuffer(g_dev, &bi, NULL, buf);
    if (!r) {
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev, *buf, &mr);
        VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = (void *)mem};
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, size, mem_type(hp.memoryTypeBits & mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
        r = vkAllocateMemory(g_dev, &ai, NULL, dm);
        if (!r) r = vkBindBufferMemory(g_dev, *buf, *dm, 0);
    }
    if (r) { if (*buf) vkDestroyBuffer(g_dev, *buf, NULL); if (*dm) vkFreeMemory(g_dev, *dm, NULL); *buf = 0; *dm = 0; }
    return r;
}
// A piece holds [off, off + len) of a large pool. A new one is this chunk and the next, so that anything up to a chunk long from an address in this one fits (an address may come with no usable length), and more only if the length asks for it.
// Pieces overlap, and stay until the pool is finalized
static pchunk *lazy_chunk(const vpool *p, uint64_t off, uint64_t len) {
    plazy *z = p->lz; uint64_t i = off / g_chunk;
    pchunk *c = atomic_load_explicit(&z->tab[i], memory_order_acquire);
    if (c && off + len <= c->off + c->size) return c;
    pthread_mutex_lock(&z->mu);
    c = atomic_load_explicit(&z->tab[i], memory_order_acquire);
    if (z->gone) c = NULL;
    else if (!c || off + len > c->off + c->size) {
        // Far into the driver's limit on allocations, new pieces take more at a time: more memory pinned rather than a failed import
        // This driver says it allows 4,294,967,295, so at most 8,192 count here: every allocation has a cost at each submission, in the kernel
        uint64_t limit = MIN((uint64_t)g_lim.maxMemoryAllocationCount, (uint64_t)8192), npc = (uint64_t)atomic_load(&g_pool_pieces), span = npc > limit / 4 * 3 ? 256 * g_chunk : npc > limit / 2 ? 64 * g_chunk : 2 * g_chunk;
        uint64_t lo = i * g_chunk, hi = MAX(lo + span, (off + len + g_chunk - 1) & ~(g_chunk - 1)), t0 = now_ns(); hi = MIN(hi, p->size);
        pool_charge(hi - lo);
        dm_guest_write_prepare((void *)(p->mem + lo), hi - lo);   // the driver pins for writing, and the texture tracker may have set some of these pages read-only
        pchunk *n = calloc(1, sizeof *n); VkResult r = n ? import_range(p->mem + lo, hi - lo, &n->buf, &n->dm) : VK_ERROR_OUT_OF_HOST_MEMORY;
        if (r) { static int nl; if (nl++ < 8) LOG("[RED] vk: importing %llu KB at +%#llx of memory pool %p failed: %d", (unsigned long long)((hi - lo) >> 10), (unsigned long long)lo, (const void *)p->mem, (int)r); atomic_fetch_sub(&g_pool_total, hi - lo); free(n); c = NULL; }
        else {
            n->off = lo; n->size = hi - lo; n->serial = atomic_fetch_add(&g_pool_serial, 1) + 1; n->next = z->all; z->all = n; atomic_fetch_add(&g_pool_pieces, 1);
            for (uint64_t j = i; j <= (hi - 1) / g_chunk; j++) { pchunk *e = atomic_load_explicit(&z->tab[j], memory_order_relaxed); if (!e || e->off + e->size < hi) atomic_store_explicit(&z->tab[j], n, memory_order_release); }
            if (hi - lo > 2 * g_chunk) LOG("vk: memory pool %p: %llu KB at +%#llx imported in %.1f ms, for %llu bytes", (const void *)p->mem, (unsigned long long)((hi - lo) >> 10), (unsigned long long)lo, (now_ns() - t0) / 1e6, (unsigned long long)len);
            c = n;
        }
    }
    pthread_mutex_unlock(&z->mu);
    return c;
}
// A large pool's memory becomes shared memory (DM_POOL_SHARED=0 keeps it private). Samsung's memory guard counts a process's RssAnon + VmSwap and ended a run at 4 GiB of it. This game's pools hold more than half of that:
// loaded data, most of it idle and swapped out. Shared pages are swapped the same way and the rest of Android's memory handling sees them as before, but they are in neither count. Emulators map guest memory this way for their own reasons.
// What the game put in the range before making it a pool is kept. Pages in swap at this moment can't be found from here, so a range with any stays private.
// Another guest thread writing to the range during the copy would lose that write: "pages kept" in the log is how much there was to copy, 0 so far in every run
static struct { const uint8_t *mem; uint64_t size, released; uint64_t *gone; } g_shared[32]; static int g_nshared; static pthread_mutex_t g_shared_mu = PTHREAD_MUTEX_INITIALIZER;   // gone: a bit per page given back
static uint64_t range_swap_kb(const uint8_t *mem, uint64_t size) {   // the Swap lines of every mapping over the range (the dirty tracker splits mappings); -1: can't be read
    FILE *f = fopen("/proc/self/smaps", "re"); if (!f) return (uint64_t)-1;
    char l[256], perm[8]; int in = 0; uint64_t kb = 0; unsigned long long a, b, v;
    while (fgets(l, sizeof l, f)) {
        if (sscanf(l, "%llx-%llx %4s", &a, &b, perm) == 3) in = a < (uintptr_t)mem + size && (uintptr_t)mem < b;
        else if (in && sscanf(l, "Swap: %llu kB", &v) == 1) kb += v;
    }
    fclose(f); return kb;
}
static void pool_make_shared(const void *mem, uint64_t size) {
    static const char *e; static int init; if (!init) { init = 1; e = getenv("DM_POOL_SHARED"); }
    if (e && !atoi(e)) return;
    const uint8_t *m = mem;
    pthread_mutex_lock(&g_shared_mu);
    int skip = g_nshared == 32; for (int i = 0; i < g_nshared; i++) if (m < g_shared[i].mem + g_shared[i].size && g_shared[i].mem < m + size) skip = 1;   // a pool made again over the same range keeps its pages
    pthread_mutex_unlock(&g_shared_mu);
    if (skip) return;
    uint64_t t0 = now_ns(), swp = range_swap_kb(m, size), npg = size >> 12, kept = 0;
    if (swp) { LOG("[RED] vk: memory pool %p stays private: its mapping has %lld KB in swap", mem, (long long)swp); return; }
    unsigned char *vec = malloc(npg); int fd = syscall(SYS_memfd_create, "dmpool", MFD_CLOEXEC);
    if (!vec || fd < 0 || ftruncate(fd, (off_t)size) || mincore((void *)mem, size, vec)) { LOG("[RED] vk: memory pool %p stays private: errno %d", mem, errno); free(vec); if (fd >= 0) close(fd); return; }
    for (uint64_t i = 0; i < npg; i++) if (vec[i] & 1) { if (pwrite(fd, m + (i << 12), 4096, (off_t)(i << 12)) != 4096) { LOG("[RED] vk: memory pool %p stays private: errno %d", mem, errno); free(vec); close(fd); return; } kept++; }
    free(vec);
    if (mmap((void *)mem, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == MAP_FAILED) die("vk: mapping shared memory over memory pool %p failed: errno %d", mem, errno);
    close(fd);
    uint64_t *gone = calloc((npg + 63) / 64, 8);   // none (out of memory): this pool's pages are not punched out when given back, and they stay until no part of it is mapped
    pthread_mutex_lock(&g_shared_mu); g_shared[g_nshared].mem = m; g_shared[g_nshared].size = size; g_shared[g_nshared].released = 0; g_shared[g_nshared].gone = gone; g_nshared++; pthread_mutex_unlock(&g_shared_mu);
    LOG("vk: memory pool %p (%llu MB) is shared memory now, %llu pages kept, %.1f ms", mem, (unsigned long long)(size >> 20), (unsigned long long)kept, (now_ns() - t0) / 1e6);
}
// Guest memory given back (guest_release in hle_svc.c, which then maps fresh private pages over the range). In a shared pool the file would keep those pages until every last page of the pool was gone, so they are punched out first.
// A piece the GPU still has imported stays pinned and alive for that work, as it does for private memory. Once all of a pool's range was given back, a new pool there can be shared again
void dm_pool_shared_release(uintptr_t lo, uintptr_t hi) {
    pthread_mutex_lock(&g_shared_mu);
    for (int i = 0; i < g_nshared; i++) {
        uintptr_t a = (uintptr_t)g_shared[i].mem, b = a + g_shared[i].size, l = lo > a ? lo : a, h = hi < b ? hi : b;
        if (l >= h || !g_shared[i].gone) continue;
        // Only pages not given back before. A page given back has private memory over it now (guest_release), where MADV_REMOVE fails, and counting it twice would drop the pool from this table while parts of it are still shared
        uint64_t *gone = g_shared[i].gone, pe = (h - a + 4095) >> 12;
        for (uint64_t p = (l - a) >> 12; p < pe;) {
            if (gone[p >> 6] >> (p & 63) & 1) { p++; continue; }
            uint64_t q = p; while (q < pe && !(gone[q >> 6] >> (q & 63) & 1)) { gone[q >> 6] |= 1ull << (q & 63); q++; }
            if (madvise((void *)(a + (p << 12)), (q - p) << 12, MADV_REMOVE)) { static int nred; if (nred++ < 8) LOG("[RED] vk: shared memory %#lx+%#lx not freed (errno %d)", (unsigned long)(a + (p << 12)), (unsigned long)((q - p) << 12), errno); }
            g_shared[i].released += (q - p) << 12; p = q;
        }
        if (g_shared[i].released >= g_shared[i].size) { LOG("vk: shared memory pool %#lx given back", (unsigned long)a); free(gone); g_shared[i--] = g_shared[--g_nshared]; }
    }
    pthread_mutex_unlock(&g_shared_mu);
}
void mtl_pool_add(const void *mem, uint64_t size) {
    ensure_init();
    { static int once; if (!once) { once = 1; const char *e = getenv("DM_POOL_CHUNK_KB"); if (e && atoi(e) >= 4) g_chunk = (uint64_t)atoi(e) << 10 & ~4095ull; } }
    vpool p = {mem, size, 0, 0, atomic_fetch_add(&g_pool_serial, 1) + 1, NULL};
    static _Atomic int nseen; int say = atomic_fetch_add(&nseen, 1) < 16 || size >= (16u << 20);
    if (((uintptr_t)mem | size) & 4095) LOG("[RED] vk: memory pool %p size %#llx is not 4 KB aligned: not importable, its resources are skipped", mem, (unsigned long long)size);
    else if (size >= POOL_LAZY_MIN) {
        p.lz = calloc(1, sizeof *p.lz); if (p.lz) p.lz->tab = calloc(size / g_chunk + 1, sizeof *p.lz->tab);
        if (!p.lz || !p.lz->tab) die("vk: memory pool table allocation failed");
        pthread_mutex_init(&p.lz->mu, NULL);
        { char nm[32]; snprintf(nm, sizeof nm, "dmpool%llu", (unsigned long long)(size >> 20)); prctl(0x53564d41 /* PR_SET_VMA */, 0 /* PR_SET_VMA_ANON_NAME */, mem, size, strdup(nm)); }   // for DM_MEMMAP (front_android.c) and pool_make_shared. The kernel may keep the pointer, so the name stays
        pool_make_shared(mem, size);
        if (say) LOG("vk: memory pool %p (%llu KB): imported piece by piece, as the GPU needs it", mem, (unsigned long long)(size >> 10));
    } else {
        uint64_t t0 = now_ns(); pool_charge(size);
        VkResult r = import_range(mem, size, &p.buf, &p.dm);
        if (r) { LOG("[RED] vk: importing memory pool %p size %#llx failed: %d", mem, (unsigned long long)size, (int)r); atomic_fetch_sub(&g_pool_total, size); }
        else if (say) LOG("vk: memory pool %p (%llu KB) imported as device memory in %.1f ms", mem, (unsigned long long)(size >> 10), (now_ns() - t0) / 1e6);
    }
    pthread_mutex_lock(&g_pomu);
    if (g_npool == g_cpool) { g_cpool = g_cpool ? 2 * g_cpool : 64; g_pool = realloc(g_pool, g_cpool * sizeof *g_pool); }
    int i = g_npool++; while (i > 0 && g_pool[i - 1].mem > p.mem) { g_pool[i] = g_pool[i - 1]; i--; }   // sorted by address
    atomic_fetch_add(&g_pool_gen, 1); atomic_store(&g_pool_t_chg, now_ns());
    g_pool[i] = p; psnap_publish();
    pthread_mutex_unlock(&g_pomu);
}
void mtl_pool_del(const void *mem) {
    pthread_mutex_lock(&g_pomu);
    for (int i = 0; i < g_npool; i++) if (g_pool[i].mem == mem) {
        dead *d = calloc(1, sizeof *d); d->buf = g_pool[i].buf; d->mem = g_pool[i].dm;
        if (g_pool[i].buf) atomic_fetch_sub(&g_pool_total, g_pool[i].size);
        plazy *z = g_pool[i].lz;
        memmove(&g_pool[i], &g_pool[i + 1], (g_npool - i - 1) * sizeof *g_pool); g_npool--; atomic_fetch_add(&g_pool_gen, 1); { uint64_t t = now_ns(); atomic_store(&g_pool_t_chg, t); atomic_store(&g_pool_t_del, t); } psnap_publish();
        pthread_mutex_unlock(&g_pomu);
        if (z) {   // z itself stays: a lookup may have just read it. A few KB per large pool, and the game creates those once
            pthread_mutex_lock(&z->mu); z->gone = 1;
            int any = 0; for (pchunk *c = z->all, *nx; c; c = nx) { nx = c->next; dead *dc = calloc(1, sizeof *dc); dc->buf = c->buf; dc->mem = c->dm; dead_add(dc); atomic_fetch_sub(&g_pool_total, c->size); atomic_fetch_sub(&g_pool_pieces, 1); any = 1; }
            z->all = NULL; pthread_mutex_unlock(&z->mu);
            if (any) { if (!mtl_wait(g_fence, 2000000000ull)) LOG("[RED] vk: memory pool %p finalized while submission %llu is still running after 2 s", mem, (unsigned long long)g_fence); dead_reap(); }
            free(d); return;
        }
        if (d->buf || d->mem) { dead_add(d); if (!mtl_wait(g_fence, 2000000000ull)) LOG("[RED] vk: memory pool %p finalized while submission %llu is still running after 2 s", mem, (unsigned long long)g_fence); dead_reap(); } else free(d);   // the guest may reuse the pages right after Finalize: release the import first
        return;
    }
    pthread_mutex_unlock(&g_pomu);
}
// The pool holding [addr, addr + len), copied into *out (0 if none). A copy, not a pointer into g_pool: game threads add and finalize pools (mtl_pool_add / mtl_pool_del: loading,
// hle_nvn's mapped-page runs) while the replay thread records; a pointer read after the unlock mixed two pools' buffer and base (the 5 s GPU hang). Lock-free on g_psnap,
// with shared hints of the entry found last per 1 MB region of addresses (several lookups per draw; a hint is checked against the range, so a stale one only costs the search)
static _Atomic int g_pohint[64];
static int pool_get(uint64_t addr, uint64_t len, vpool *out) {
    const vpsnap *sn = atomic_load_explicit(&g_psnap, memory_order_acquire); if (!sn) return 0;
    _Atomic int *hp = &g_pohint[(addr >> 20) & 63];
    { int h = atomic_load_explicit(hp, memory_order_relaxed);
      if (h < sn->n) { const vpool *e = &sn->e[h]; uint64_t b = (uint64_t)(uintptr_t)e->mem; if (addr >= b && addr + len <= b + e->size) { *out = *e; return 1; } } }
    int lo = 0, hi = sn->n - 1;
    while (lo <= hi) { int m = (lo + hi) / 2; const vpool *e = &sn->e[m]; uint64_t b = (uint64_t)(uintptr_t)e->mem;
        if (addr < b) hi = m - 1; else if (addr >= b + e->size) lo = m + 1;
        else { if (addr + len > b + e->size) return 0;
            *out = *e; atomic_store_explicit(hp, m, memory_order_relaxed); return 1; } }
    return 0;
}
static int pool_live(uint64_t addr, uint64_t len) { vpool p; return pool_get(addr, len, &p); }
int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    vpool p; if (!pool_get(addr, len, &p)) return 0;
    *buf = (void *)p.mem; *off = addr - (uint64_t)(uintptr_t)p.mem; return 1;
}
// The buffer holding [addr, addr + len). clamp: len only says how much is wanted, and may run past the pool's end. *psize: that buffer's size, and *off the place in it, so *psize - *off bytes can be bound from there
static int pool_buf(uint64_t addr, uint64_t len, int clamp, VkBuffer *b, VkDeviceSize *off, uint64_t *psize, uint64_t *serial) {
    if (!len) { len = POOL_NOLEN; clamp = 1; }
    vpool p; if (!pool_get(addr, clamp ? 1 : len, &p)) return 0;
    uint64_t o = addr - (uint64_t)(uintptr_t)p.mem; if (clamp) len = MIN(len, p.size - o);
    if (p.lz) { pchunk *c = lazy_chunk(&p, o, len); if (!c) return 0; __atomic_store_n(&c->last_sub, g_nsub, __ATOMIC_RELAXED); *b = c->buf; *off = o - c->off; if (psize) *psize = c->size; if (serial) *serial = c->serial; return 1; }
    if (!p.buf) return 0;
    *b = p.buf; *off = o; if (psize) *psize = p.size; if (serial) *serial = p.serial; return 1;
}
static int buf_at_w(uint64_t addr, uint64_t want, VkBuffer *b, VkDeviceSize *off, uint64_t *psize) { return pool_buf(addr, want, 1, b, off, psize, NULL); }   // want: bytes the caller expects to use
static int buf_at(uint64_t addr, uint64_t len, VkBuffer *b, VkDeviceSize *off) { return pool_buf(addr, len, 0, b, off, NULL, NULL); }
void mtl_alias_drop_pool(const void *pool) { (void)pool; }

// ---------- Host-visible arenas: per-submission ring (uploads, driver banks, converted indices) and a static arena (c1 segments, support buffer, quad indices) ----------
typedef struct { VkBuffer buf; VkDeviceMemory mem; uint8_t *map; uint64_t size; } hbuf;
static hbuf hbuf_new(uint64_t size, VkBufferUsageFlags usage) {
    hbuf h = {0}; h.size = size;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
    VKCHK(vkCreateBuffer(g_dev, &bi, NULL, &h.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_dev, h.buf, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0)};
    VKCHK(vkAllocateMemory(g_dev, &ai, NULL, &h.mem)); VKCHK(vkBindBufferMemory(g_dev, h.buf, h.mem, 0));
    VKCHK(vkMapMemory(g_dev, h.mem, 0, VK_WHOLE_SIZE, 0, (void **)&h.map));
    return h;
}
// 16 MB, as a few frames of ordinary use take KBs, once large texture uploads go elsewhere (up_alloc): this was 128 MB of GPU memory, always taken. An allocation over a chunk gets a buffer of its own
enum { RING_CHUNK = 4 << 20, RING_N = 4, UP_RING_LIMIT = 8 << 20 };
static struct { hbuf h[RING_N]; uint64_t fence[RING_N]; int cur; uint64_t head, limit, extra, sub, peak; } g_ring;   // limit: RING_CHUNK (DM_VK_RING_LIMIT=<KB> lowers it, to test the path below); extra: what this submission took beyond the ring; sub, peak: bytes this submission, the most outside texture uploads
static uint64_t g_ring_peak_seen(void) { return g_ring.peak; }
static hbuf g_static; static uint64_t g_static_head, g_static_total; static pthread_mutex_t g_stmu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_static_seen(void) { return g_static_total; }
typedef struct { VkBuffer buf; uint64_t off; uint8_t *p; } halloc;
static halloc ring_alloc(uint64_t n, uint64_t align) {   // submit thread only; memory valid until the current submission completes
    if (n > RING_CHUNK) { hbuf own = hbuf_new(n, BUF_ALL); dead *dd = calloc(1, sizeof *dd); if (dd) { dd->buf = own.buf; dd->mem = own.mem; dead_add(dd); } g_ring.extra += n; g_ring.sub += n; return (halloc){own.buf, 0, own.map}; }
    uint64_t o = (g_ring.head + align - 1) & ~(align - 1);
    if (o + n > g_ring.limit) {
        int next = (g_ring.cur + 1) % RING_N; o = 0;
        // The submission being recorded holds every chunk. It is not on the GPU yet, so a wait for it can only time out, and reusing the chunk then overwrites what its commands still read: 10 s per chunk, the black screen after
        // character creation. That chunk stays with the submission, as a buffer freed when it completes, and a new one takes its place. Nothing else waits on this path, and an entry cached for the old chunk stays right
        if (g_ring.fence[next] > g_fence) {
            dead *dd = calloc(1, sizeof *dd); if (dd) { dd->buf = g_ring.h[next].buf; dd->mem = g_ring.h[next].mem; dead_add(dd); }
            g_ring.h[next] = hbuf_new(RING_CHUNK, BUF_ALL); g_ring.extra += RING_CHUNK;
        } else { uint64_t w0 = now_ns(); if (!mtl_wait(g_ring.fence[next], 10000000000ull)) LOG("[RED] vk: ring chunk %d still in use after 10 s", next); WT_ADD(WT_RING, now_ns() - w0); }
        g_ring.cur = next;
    }
    g_ring.sub += n;
    g_ring.head = o + n; g_ring.fence[g_ring.cur] = g_fence + 1;
    return (halloc){g_ring.h[g_ring.cur].buf, o, g_ring.h[g_ring.cur].map + o};
}
static int g_nc1_blocks;   // constant blocks taken from the static arena by shader_get
// Nothing here is freed. It grows by 16 MB as shaders load: a fixed 64 MB was GPU memory taken from the start, and when full it ended the run
static halloc static_alloc(uint64_t n, uint64_t align) {
    pthread_mutex_lock(&g_stmu);
    uint64_t o = (g_static_head + align - 1) & ~(align - 1);
    if (!g_static.buf || o + n > g_static.size) { g_static = hbuf_new(MAX(n, (uint64_t)16u << 20), BUF_ALL); g_static_total += g_static.size; o = 0; }
    g_static_head = o + n;
    pthread_mutex_unlock(&g_stmu);
    return (halloc){g_static.buf, o, g_static.map + o};
}

// ---------- Formats ----------
enum { CV_NONE, CV_RGBX_A1, CV_BC4U, CV_BC4S, CV_BC5U, CV_BC5S, CV_BC7, CV_BC6, CV_RGB_EXPAND8, CV_RGB_EXPAND16, CV_RGB_EXPAND32 };
typedef struct { VkFormat vk; uint8_t conv, alpha_one, depth, stencil; } fmt_t;
static fmt_t fmt_of(int f) {
    static const VkFormat T[0x9d] = {
        [0x01] = VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT,
        VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R16G16_SFLOAT,
        VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
        VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16_SNORM,
        VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT,
        VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM,
        VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
        VK_FORMAT_S8_UINT, VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_A4B4G4R4_UNORM_PACK16, VK_FORMAT_A1R5G5B5_UNORM_PACK16, VK_FORMAT_A1R5G5B5_UNORM_PACK16, VK_FORMAT_B5G6R5_UNORM_PACK16,
        VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
        VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC2_UNORM_BLOCK, VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_BC1_RGB_SRGB_BLOCK, VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
        VK_FORMAT_BC2_SRGB_BLOCK, VK_FORMAT_BC3_SRGB_BLOCK, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB,
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R8_USCALED, VK_FORMAT_R8_SSCALED, VK_FORMAT_R16_USCALED, VK_FORMAT_R16_SSCALED, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8G8_USCALED, VK_FORMAT_R8G8_SSCALED,
        VK_FORMAT_R16G16_USCALED, VK_FORMAT_R16G16_SSCALED, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R8G8B8_USCALED, VK_FORMAT_R8G8B8_SSCALED, VK_FORMAT_R16G16B16_USCALED,
        VK_FORMAT_R16G16B16_SSCALED, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R8G8B8A8_USCALED, VK_FORMAT_R8G8B8A8_SSCALED, VK_FORMAT_R16G16B16A16_USCALED,
        VK_FORMAT_R16G16B16A16_SSCALED, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
        VK_FORMAT_A2B10G10R10_SNORM_PACK32, VK_FORMAT_A2B10G10R10_SINT_PACK32, VK_FORMAT_A2B10G10R10_USCALED_PACK32, VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
        VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM,
        VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
        [0x95] = VK_FORMAT_R5G6B5_UNORM_PACK16, VK_FORMAT_A1R5G5B5_UNORM_PACK16, VK_FORMAT_A1R5G5B5_UNORM_PACK16, VK_FORMAT_R5G5B5A1_UNORM_PACK16, VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB};
    fmt_t r = {0};
    if (f >= 0x79 && f <= 0x86) { r.vk = (VkFormat)(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * (f - 0x79)); return r; }
    if (f >= 0x87 && f <= 0x94) { r.vk = (VkFormat)(VK_FORMAT_ASTC_4x4_SRGB_BLOCK + 2 * (f - 0x87)); return r; }
    if (f <= 0 || f > 0x9c) return r;
    r.vk = T[f];
    switch (f) {
        case 0x37: case 0x3a: case 0x6d: case 0x6e: case 0x6f: case 0x70: case 0x71: case 0x72: case 0x73: case 0x74: case 0x75: case 0x76: case 0x77: case 0x78: case 0x96: case 0x99: case 0x9b: r.alpha_one = 1; break;
        case 0x49: r.conv = CV_BC4U; break; case 0x4a: r.conv = CV_BC4S; break; case 0x4b: r.conv = CV_BC5U; break; case 0x4c: r.conv = CV_BC5S; break;
        case 0x4d: case 0x4e: r.conv = CV_BC7; break; case 0x4f: case 0x50: r.conv = CV_BC6; break;
        case 0x31: r.stencil = 1; break; case 0x32: case 0x33: case 0x34: r.depth = 1; break; case 0x35: case 0x36: r.depth = r.stencil = 1; break;
    }
    if (f >= 0x19 && f <= 0x24) r.conv = f <= 0x1c ? CV_RGB_EXPAND8 : f <= 0x21 ? CV_RGB_EXPAND16 : CV_RGB_EXPAND32;   // 3-component textures: expanded to 4 (the vertex formats keep 3)
    return r;
}
static VkFormat tex_vk_format(int f) {   // the image format a texture is created with (conversions land in their destination format)
    fmt_t m = fmt_of(f);
    switch (m.conv) {
        case CV_RGB_EXPAND8: return (VkFormat)(m.vk + (VK_FORMAT_R8G8B8A8_UNORM - VK_FORMAT_R8G8B8_UNORM) * 0 + 0) == m.vk ? (f == 0x19 ? VK_FORMAT_R8G8B8A8_UNORM : f == 0x1a ? VK_FORMAT_R8G8B8A8_SNORM : f == 0x1b ? VK_FORMAT_R8G8B8A8_UINT : VK_FORMAT_R8G8B8A8_SINT) : m.vk;
        case CV_RGB_EXPAND16: return f == 0x1d ? VK_FORMAT_R16G16B16A16_SFLOAT : f == 0x1e ? VK_FORMAT_R16G16B16A16_UNORM : f == 0x1f ? VK_FORMAT_R16G16B16A16_SNORM : f == 0x20 ? VK_FORMAT_R16G16B16A16_UINT : VK_FORMAT_R16G16B16A16_SINT;
        case CV_RGB_EXPAND32: return f == 0x22 ? VK_FORMAT_R32G32B32A32_SFLOAT : f == 0x23 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SINT;
    }
    return m.vk;
}
static VkFormatProperties fmt_props(VkFormat f) {
    static VkFormatProperties cache[256]; static uint8_t have[256];
    if ((unsigned)f < 256) { if (!have[f]) { vkGetPhysicalDeviceFormatProperties(g_pd, f, &cache[f]); have[f] = 1; } return cache[f]; }
    VkFormatProperties p; vkGetPhysicalDeviceFormatProperties(g_pd, f, &p); return p;
}
static int vk_is_depth(VkFormat f) { return f == VK_FORMAT_D16_UNORM || f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_X8_D24_UNORM_PACK32; }
static VkImageAspectFlags vk_aspect(VkFormat f) {
    if (f == VK_FORMAT_S8_UINT) return VK_IMAGE_ASPECT_STENCIL_BIT;
    if (f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT) return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    if (vk_is_depth(f)) return VK_IMAGE_ASPECT_DEPTH_BIT;
    return VK_IMAGE_ASPECT_COLOR_BIT;
}
static uint32_t vk_texel_bytes(VkFormat f) {   // bytes per texel of the uncompressed formats this file reads back (screenshots / ReadTexels)
    switch (f) {   // every uncompressed format of fmt_of's table; the block formats (BC, ASTC) have no texel size and fall to the default
        case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8_USCALED: case VK_FORMAT_R8_SSCALED: case VK_FORMAT_S8_UINT: return 1;
        case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8_USCALED: case VK_FORMAT_R8G8_SSCALED:
        case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SNORM: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16_USCALED: case VK_FORMAT_R16_SSCALED: case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_B5G6R5_UNORM_PACK16: case VK_FORMAT_R5G6B5_UNORM_PACK16: case VK_FORMAT_A4B4G4R4_UNORM_PACK16: case VK_FORMAT_A1R5G5B5_UNORM_PACK16: case VK_FORMAT_R5G5B5A1_UNORM_PACK16: return 2;
        case VK_FORMAT_R8G8B8_UNORM: case VK_FORMAT_R8G8B8_SNORM: case VK_FORMAT_R8G8B8_UINT: case VK_FORMAT_R8G8B8_SINT: case VK_FORMAT_R8G8B8_USCALED: case VK_FORMAT_R8G8B8_SSCALED: return 3;
        case VK_FORMAT_R16G16B16_SFLOAT: case VK_FORMAT_R16G16B16_UNORM: case VK_FORMAT_R16G16B16_SNORM: case VK_FORMAT_R16G16B16_UINT: case VK_FORMAT_R16G16B16_SINT: case VK_FORMAT_R16G16B16_USCALED: case VK_FORMAT_R16G16B16_SSCALED: return 6;
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R16G16B16A16_USCALED: case VK_FORMAT_R16G16B16A16_SSCALED:
        case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_D32_SFLOAT_S8_UINT: return 8;
        case VK_FORMAT_R32G32B32_SFLOAT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32_SINT: return 12;
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: return 16;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_R8G8B8A8_USCALED: case VK_FORMAT_R8G8B8A8_SSCALED:
        case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT:
        case VK_FORMAT_R16G16_USCALED: case VK_FORMAT_R16G16_SSCALED: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2B10G10R10_SNORM_PACK32: case VK_FORMAT_A2B10G10R10_SINT_PACK32: case VK_FORMAT_A2B10G10R10_USCALED_PACK32: case VK_FORMAT_A2B10G10R10_SSCALED_PACK32:
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_X8_D24_UNORM_PACK32: return 4;
        default: return 4;
    }
}

// ---------- BC4 / BC5 decoding (the Xclipse 920 samples BC1-BC3 only) ----------
static void bc4_block(const uint8_t *b, uint8_t *out, int stride_px, int step, int sn) {   // 4x4 texels, one channel; out advances step bytes per texel
    int e[8]; if (sn) { e[0] = (int8_t)b[0]; e[1] = (int8_t)b[1]; } else { e[0] = b[0]; e[1] = b[1]; }
    if (e[0] > e[1]) for (int i = 1; i < 7; i++) e[i + 1] = ((7 - i) * e[0] + i * e[1]) / 7;
    else { for (int i = 1; i < 5; i++) e[i + 1] = ((5 - i) * e[0] + i * e[1]) / 5; e[6] = sn ? -127 : 0; e[7] = sn ? 127 : 255; }
    uint64_t bits = 0; for (int i = 0; i < 6; i++) bits |= (uint64_t)b[2 + i] << (8 * i);
    for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) out[(y * stride_px + x) * step] = (uint8_t)e[(bits >> (3 * (4 * y + x))) & 7];
}
// One level's linear BCn blocks (bw x bh blocks) → texels (bytes per texel cpt: 1 BC4, 2 BC5), rows of w texels padded to whole blocks: the copy uses the block-padded extent clipped to the level
static void bc45_decode(const uint8_t *src, uint8_t *dst, uint32_t bw, uint32_t bh, int two, int sn) {
    uint32_t pw = bw * 4; int cpt = two ? 2 : 1, bpb = two ? 16 : 8;
    for (uint32_t by = 0; by < bh; by++) for (uint32_t bx = 0; bx < bw; bx++) {
        const uint8_t *b = src + (by * bw + bx) * bpb; uint8_t *o = dst + ((by * 4) * pw + bx * 4) * cpt;
        bc4_block(b, o, pw, cpt, sn); if (two) bc4_block(b + 8, o + 1, pw, cpt, sn);
    }
}

void bc7_decode(const uint8_t *src, uint8_t *dst, uint32_t bw, uint32_t bh);   // bc7.c
void bc6h_decode(const uint8_t *src, uint8_t *dst, uint32_t bw, uint32_t bh, int is_signed);   // bc7.c
void bc6h_decode_e5(const uint8_t *src, uint8_t *dst, uint32_t bw, uint32_t bh);
void bc7_to_bc(const uint8_t *src, uint8_t *dst, uint32_t bw, uint32_t bh, int bc1);
typedef struct { float err_rgb, err_a; int a_min; } bc7_survey_t;
void bc7_survey(const uint8_t *mem, uint64_t size, bc7_survey_t *s);
int bc7_blocks_opaque(const uint8_t *b, uint64_t n);
void bc6h_survey(const uint8_t *mem, uint64_t size, int is_signed, float *top, float *over1, float *err);
void bc6h_to_bc1(const uint8_t *src, uint8_t *dst, uint32_t bw, uint32_t bh);
void bc_enc_init(void);

// ---------- Images ----------
typedef struct { VkImageViewType type; VkFormat fmt; uint16_t l0, nl, s0, ns; uint8_t sw[4]; VkImageAspectFlags aspect; VkImageView view; } vview;
typedef struct vimg {
    uint64_t serial; tex_t t; VkImage img; VkDeviceMemory mem; VkFormat fmt; VkImageAspectFlags aspect; VkImageType itype;
    uint32_t w, h, d, layers, levels, samples; int cube_ok, mutable_, fl;   // fl: created with a VkImageFormatListCreateInfo (DM_VK_IMGFLAGS), views limited to fmt_family(fmt)
    VkImageUsageFlags usage; VkImageCreateFlags cflags;   // as created (the recycling key, with the fields above)
    int gpu_owned, uploaded, cpu_dirty, inited, cpuw_noted;   // cpuw_noted: "CPU write into a GPU-written texture" was logged for it
    vview v[16]; int nv;
    uint8_t *lin_copy; uint64_t lin_sub;   // linear CPU texture: copy of its guest bytes at the last upload, and the submission that last compared them (lin_changed)
    uint32_t lin_mru[4], lin_roll;   // large ones: the 4 blocks that changed last, most recent first (block + 1, 0 = none), the next block of the rolling scan
    uint64_t lin_miss;   // DM_VK_LIN_CHECK: the first submission whose block compare missed a change (0 = none pending)
    uint32_t kb, top_kb; int kind, top_able, side;   // its memory in KB, as g_img_kb counts it; what leaving out its largest levels saved, or would save
    int drop;   // the guest's largest levels this image leaves out: image level n is the guest's level n + drop, and w, h, levels are the image's
} vimg;
static uint64_t g_nsub;   // submissions so far (with the contexts below)
static int vimg_linear_cpu(const struct vimg *v); static void lin_snap(struct vimg *v);
static struct { uint64_t key; vimg *v; } *g_ih; static uint32_t g_ihcap, g_ihn; static pthread_mutex_t g_imu = PTHREAD_MUTEX_INITIALIZER;
static uint32_t hmix(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; return (uint32_t)k; }
static vimg *ih_get(uint64_t k) {
    if (!g_ihcap) return NULL;
    for (uint32_t i = hmix(k) & (g_ihcap - 1);; i = (i + 1) & (g_ihcap - 1)) { if (!g_ih[i].key) return NULL; if (g_ih[i].key == k) return g_ih[i].v; }
}
static void ih_put(uint64_t k, vimg *v);
static void ih_grow(void) {
    uint32_t oc = g_ihcap; typeof(g_ih) o = g_ih;
    g_ihcap = oc ? 2 * oc : 4096; g_ih = calloc(g_ihcap, sizeof *g_ih); g_ihn = 0;
    for (uint32_t i = 0; i < oc; i++) if (o[i].key && o[i].v) ih_put(o[i].key, o[i].v);
    free(o);
}
static void ih_put(uint64_t k, vimg *v) {
    if ((g_ihn + 1) * 2 > g_ihcap) ih_grow();
    for (uint32_t i = hmix(k) & (g_ihcap - 1);; i = (i + 1) & (g_ihcap - 1)) {
        if (g_ih[i].key == k) { g_ih[i].v = v; return; }
        if (!g_ih[i].key) { g_ih[i].key = k; g_ih[i].v = v; g_ihn++; return; }
    }
}
static void ih_del(uint64_t k) {   // backward-shift deletion
    if (!g_ihcap) return;
    uint32_t i = hmix(k) & (g_ihcap - 1);
    for (;; i = (i + 1) & (g_ihcap - 1)) { if (!g_ih[i].key) return; if (g_ih[i].key == k) break; }
    g_ih[i].key = 0; g_ih[i].v = NULL; g_ihn--;
    for (uint32_t j = (i + 1) & (g_ihcap - 1); g_ih[j].key; j = (j + 1) & (g_ihcap - 1)) {
        uint64_t kk = g_ih[j].key; vimg *vv = g_ih[j].v; g_ih[j].key = 0; g_ih[j].v = NULL; g_ihn--; ih_put(kk, vv);
    }
}
static uint64_t g_stat_up_bytes, g_stat_up_n, g_stat_draws, g_stat_passes, g_stat_drop[16];
static int g_fmt_seen[0x100];
// Every 30 s: how much image memory stands for the same guest memory. The game can place textures over each other in a pool, as render targets used at different moments or as two objects for one texture. Here each gets an image of its own, so that memory is spent twice
typedef struct { uintptr_t lo, hi; uint32_t kb; int rt; } ovr;
static int ovr_cmp(const void *a, const void *b) { const ovr *x = a, *y = b; return x->lo < y->lo ? -1 : x->lo > y->lo ? 1 : x->hi > y->hi ? -1 : x->hi < y->hi ? 1 : 0; }
static void img_overlap_report(void) {
    pthread_mutex_lock(&g_imu);
    ovr *o = g_ihn ? malloc(g_ihn * sizeof *o) : NULL; int n = 0;
    if (o) for (uint32_t i = 0; i < g_ihcap; i++) { const vimg *v = g_ih[i].v;
        if (g_ih[i].key && v && v->t.mem && v->t.L.size) { o[n].lo = (uintptr_t)v->t.mem; o[n].hi = o[n].lo + v->t.L.size; o[n].kb = v->kb; o[n].rt = (v->t.d.flags & 8) != 0; n++; } }
    pthread_mutex_unlock(&g_imu);
    if (!o) return;
    qsort(o, (size_t)n, sizeof *o, ovr_cmp);
    for (int k = 0; k < 2; k++) {   // k 1: render targets. An image whose guest memory an earlier one covers whole is spent twice
        uint64_t kb = 0, guest = 0, uniq = 0, twice_kb = 0; int cnt = 0, twice = 0; uintptr_t top = 0;
        for (int i = 0; i < n; i++) { if (o[i].rt != k) continue;
            cnt++; kb += o[i].kb; guest += o[i].hi - o[i].lo;
            if (o[i].hi <= top) { twice++; twice_kb += o[i].kb; continue; }
            uniq += o[i].hi - (o[i].lo > top ? o[i].lo : top); top = o[i].hi; }
        LOG("vk: %s: %d images, %llu MB here; their guest memory %llu MB, %llu MB without overlaps; inside another's guest memory: %d images, %llu MB here", k ? "render targets" : "other textures", cnt, (unsigned long long)(kb >> 10), (unsigned long long)(guest >> 20), (unsigned long long)(uniq >> 20), twice, (unsigned long long)(twice_kb >> 10));
    }
    free(o);
}

static void vimg_free(vimg *v) {
    if (v->kb) { atomic_fetch_sub(&g_img_kb[v->kind], v->kb); atomic_fetch_sub(&g_img_n[v->kind], 1); atomic_fetch_sub(&g_img_top_kb, v->top_kb); if (v->side) atomic_fetch_sub(&g_img_side_kb[(v->side - 1) / 6][(v->side - 1) % 6], v->kb); }
    dead *d = calloc(1, sizeof *d); d->img = v->img; d->mem = v->mem;
    for (int i = 0; i < v->nv; i++) d->views[d->nv++] = v->v[i].view;
    dead_add(d); free(v->lin_copy); free(v);
}
// Image recycling (DM_VK_IMG_RECYCLE, default on, 0 = off): a game can create and finalize its transient render-target textures every frame, and each was a vkCreateImage + dedicated vkAllocateMemory
// (a driver BO and GPU VA range: ioctls) then a free: ~25% of the replay thread while walking (simpleperf call graph, rt_begin -> vimg_get -> amdgpu_bo_alloc). A released image goes here with its memory
// and views; vimg_get takes one with the same creation parameters once the submission that last could use it has completed. Entries unused for RC_AGE submissions, or past RC_N, are freed
enum { RC_N = 192, RC_AGE = 600 };
static struct { vimg *v; uint64_t fence, sub; } g_rc[RC_N]; static int g_nrc, g_rc_on = -1; static pthread_mutex_t g_rcmu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_stat_img_new, g_stat_img_reuse;
static int rc_on(void) { if (g_rc_on < 0) { const char *e = getenv("DM_VK_IMG_RECYCLE"); g_rc_on = !e || atoi(e); LOG("vk: image recycling %s (DM_VK_IMG_RECYCLE)", g_rc_on ? "on" : "off"); } return g_rc_on; }

static void vimg_destroy(vimg *v) {
    if (!rc_on() || !v->img || !v->mem || !v->inited) { vimg_free(v); return; }
    vimg *old = NULL;
    pthread_mutex_lock(&g_rcmu);
    if (g_nrc == RC_N) { old = g_rc[0].v; memmove(&g_rc[0], &g_rc[1], (RC_N - 1) * sizeof *g_rc); g_nrc--; }   // full: the oldest goes
    g_rc[g_nrc].v = v; g_rc[g_nrc].fence = g_fence + 1; g_rc[g_nrc].sub = g_nsub; g_nrc++;
    pthread_mutex_unlock(&g_rcmu);
    if (old) vimg_free(old);
}
static vimg *rc_take(const vimg *k) {   // a released image created like k, idle on the GPU; NULL if none
    uint64_t done = atomic_load(&g_done); vimg *r = NULL;
    pthread_mutex_lock(&g_rcmu);
    for (int i = g_nrc - 1; i >= 0; i--) { const vimg *c = g_rc[i].v;
        if (g_rc[i].fence <= done && c->fmt == k->fmt && c->w == k->w && c->h == k->h && c->d == k->d && c->layers == k->layers && c->levels == k->levels && c->samples == k->samples &&
            c->itype == k->itype && c->usage == k->usage && c->cflags == k->cflags && c->fl == k->fl) { r = g_rc[i].v; memmove(&g_rc[i], &g_rc[i + 1], (g_nrc - i - 1) * sizeof *g_rc); g_nrc--; break; } }
    pthread_mutex_unlock(&g_rcmu);
    return r;
}
static void rc_age(void) {   // mtl_sub_begin: free entries released RC_AGE submissions ago
    vimg *f[RC_N]; int nf = 0;
    pthread_mutex_lock(&g_rcmu);
    while (g_nrc > 0 && g_rc[0].sub + RC_AGE < g_nsub) { f[nf++] = g_rc[0].v; memmove(&g_rc[0], &g_rc[1], (g_nrc - 1) * sizeof *g_rc); g_nrc--; }
    pthread_mutex_unlock(&g_rcmu);
    for (int i = 0; i < nf; i++) vimg_free(f[i]);
}
void mtl_tex_init(const tex_t *t) { (void)t; }   // images are created at first use (most textures the game initializes are never drawn with in a given scene)
static _Atomic uint32_t g_ih_gen = 1;   // bumped under g_imu when an image leaves g_ih (mtl_tex_release): validates vimg_get's cache
void mtl_tex_release(tex_t *t) {
    if (!t->serial) return;
    pthread_mutex_lock(&g_imu); vimg *v = ih_get(t->serial); if (v) { ih_del(t->serial); atomic_fetch_add_explicit(&g_ih_gen, 1, memory_order_release); } pthread_mutex_unlock(&g_imu);
    if (v) vimg_destroy(v);
}
void mtl_tex_cpu_written(const tex_t *t) {
    int note = 0; pthread_mutex_lock(&g_imu); vimg *v = ih_get(t->serial); if (v) { v->cpu_dirty = 1; if (v->gpu_owned && !v->cpuw_noted) { v->cpuw_noted = 1; note = 1; } } pthread_mutex_unlock(&g_imu);
    if (note) { static int n; if (n++ < 16) LOG("[RED] vk: CPU write into a GPU-written texture %#llx: not uploaded (an upload is skipped for a texture the GPU wrote; the picture keeps the GPU's content)", (unsigned long long)t->serial); }   // once per texture, the first 16 textures
}
int mtl_tex_gpu_owned(uint64_t serial) { pthread_mutex_lock(&g_imu); vimg *v = ih_get(serial); int r = v && v->gpu_owned; pthread_mutex_unlock(&g_imu); return r; }

typedef struct ctx ctx;
static ctx *g_cur;   // the submission being recorded (submit thread)
static VkCommandBuffer cur_up(void);
static void img_barrier_init(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect) {
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_NONE, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
        .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    pvkCmdPipelineBarrier2(cb, &di);
}
// DM_VK_IMGFLAGS (mask; 1 = 2, the default; 0 off): 2 gives every mutable-format colour image a VkImageFormatListCreateInfo of fmt_family() instead of "all compatible formats" (GPU 26.1 -> 24.4 ms a frame with DM_VK_BARRIER=1 in another game), 4 leaves STORAGE out of the usage
// (diagnostic: Pokopia's compute shaders write images, img_binding then binds null ones). AMD's driver drops DCC for a mutable-format image without a list that is DCC-compatible (xgl vk_image.cpp -> PAL Device::ComputeDccFormatEncoding on GFX10)
static int g_imgflags = -1;
static int imgflags(void) { if (g_imgflags < 0) { const char *e = getenv("DM_VK_IMGFLAGS"); int m = e ? atoi(e) : 1; g_imgflags = m == 1 ? 2 : m & 6; LOG("vk: DM_VK_IMGFLAGS=%s: format lists %s, STORAGE image usage %s", e ? e : "1 (default)", g_imgflags & 2 ? "on" : "off", g_imgflags & 4 ? "dropped" : "kept"); } return g_imgflags; }
static int fmt_family(VkFormat f, VkFormat out[8]) {   // f first, then the formats it may be viewed as: same channel layout and swizzle, all unsigned (UNORM / UINT / SRGB) or signed (SNORM / SINT)
    static const VkFormat G[][6] = {
        {VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT}, {VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT},
        {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_SRGB}, {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB},
        {VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT}, {VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT},
        {VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT}, {VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT},
        {VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT}, {VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT}, {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UINT_PACK32}};
    int n = 0; out[n++] = f;
    for (unsigned g = 0; g < sizeof G / sizeof *G; g++) {
        int has = 0; for (int i = 0; i < 6 && G[g][i]; i++) has |= G[g][i] == f;
        if (has) { for (int i = 0; i < 6 && G[g][i]; i++) if (G[g][i] != f) out[n++] = G[g][i]; break; }
    }
    return n;   // floats (R16 / RG16 / RGBA16 / R32 SFLOAT, B10G11R11, ...) are alone: a float / integer mix is not DCC-compatible either
}
// What a BC6H texture decodes to. Unsigned ones (NVN 0x50) take half the memory in the shared exponent format, which Vulkan requires every device to sample and filter: in the world, 6 textures of this kind took 335 MB as RGBA16F. DM_TEX_E5=0 keeps RGBA16F.
// Those whose values all stay within 0 to 1, as they do in this game's, need no float format at all: as BC1 read as sRGB they take an eighth of that again, and sampling gives linear values as before. This goes by a sample of the blocks (a rare texel above 1 would stop at 1),
// and a texture too far from its original stays as it was. DM_TEX_BC6=0 keeps them float, DM_TEX_BC6_ERR: the largest error allowed, of 255 in sRGB
static VkFormat bc6_fmt(const tex_t *t) {
    int nvn = t->d.format, is_signed = nvn == 0x4f;
    static int on = -1, bc1 = -1; static float maxe; VkFormatFeatureFlags want = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if (on < 0) { const char *e = getenv("DM_TEX_E5"); on = (!e || atoi(e)) && (fmt_props(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32).optimalTilingFeatures & want) == want;
        const char *e2 = getenv("DM_TEX_BC6"), *e3 = getenv("DM_TEX_BC6_ERR"); bc1 = (!e2 || atoi(e2)) && (fmt_props(VK_FORMAT_BC1_RGB_SRGB_BLOCK).optimalTilingFeatures & want) == want; maxe = e3 ? (float)atof(e3) : 6.0f;
        if (bc1) bc_enc_init();
        LOG("vk: unsigned BC6H textures become %s (DM_TEX_E5)%s, largest error allowed %.1f of 255 (DM_TEX_BC6_ERR)", on ? "E5B9G9R9" : "RGBA16F", bc1 ? ", or BC1 when their values stay within 0 to 1" : "", (double)maxe); }
    VkFormat keep = !is_signed && on ? VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 : VK_FORMAT_R16G16B16A16_SFLOAT, f = keep;
    if (t->mem && t->L.size) { float top, over, err = 0; uint64_t t0 = now_ns(); bc6h_survey(t->mem, t->L.size, is_signed, &top, &over, &err);
        if (bc1 && !is_signed && top <= 1.0005f && err <= maxe) f = VK_FORMAT_BC1_RGB_SRGB_BLOCK;
        static int nl; if (nl++ < 32) LOG("vk: BC6H texture %ux%u x%u lv%u target %d: largest value over sampled blocks %.2f, %.1f%% of texels above 1, error as BC1 %.1f of 255: %s (%.1f ms)", t->d.width, t->d.height, t->d.target == 4 ? t->d.depth : 1, t->d.levels, t->d.target, (double)top, (double)over * 100, (double)err, f == VK_FORMAT_BC1_RGB_SRGB_BLOCK ? "BC1" : "stays float", (now_ns() - t0) / 1e6); }
    return f;
}
// The other format of a BC1-BC3 pair, which differ only in being sRGB or not; 0 for anything else. These images may be viewed as the other one: the game does (a BC1 texture viewed as sRGB, in the world)
static VkFormat bc_twin(VkFormat f) {
    switch (f) {
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK: return VK_FORMAT_BC1_RGB_SRGB_BLOCK; case VK_FORMAT_BC1_RGB_SRGB_BLOCK: return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK; case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case VK_FORMAT_BC2_UNORM_BLOCK: return VK_FORMAT_BC2_SRGB_BLOCK; case VK_FORMAT_BC2_SRGB_BLOCK: return VK_FORMAT_BC2_UNORM_BLOCK;
        case VK_FORMAT_BC3_UNORM_BLOCK: return VK_FORMAT_BC3_SRGB_BLOCK; case VK_FORMAT_BC3_SRGB_BLOCK: return VK_FORMAT_BC3_UNORM_BLOCK;
        default: return VK_FORMAT_UNDEFINED;
    }
}
static int bc_twin_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_TEX_BC_TWIN"); on = !e || atoi(e); LOG("vk: BC1-BC3 images %s be viewed as sRGB or not, as the game asks (DM_TEX_BC_TWIN)", on ? "can" : "can't"); } return on; }
static int fmt_is_srgb(VkFormat f) {
    switch (f) { case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_BC1_RGB_SRGB_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK: return 1; default: return 0; }
}
// What a BC7 texture becomes. Decoded to RGBA8 it takes 4 bytes a texel, 215 MB in the world. Encoded again as BC3 it takes 1, and as BC1, when no block has alpha, half of that. The price is some of BC7's quality and more work at loading, which runs on several threads (cv_push).
// A texture too far from its original stays RGBA8, by the error over a sample of its blocks. DM_TEX_BC7=0 keeps them all RGBA8. DM_TEX_BC7_ERR: the largest error allowed, of 255
static VkFormat bc7_fmt(const tex_t *t, uint32_t w, uint32_t h, uint32_t layers, uint32_t levels) {
    int srgb = t->d.format == 0x4e; VkFormat keep = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    static int on = -1; static float maxe;
    if (on < 0) { const char *e = getenv("DM_TEX_BC7"), *e2 = getenv("DM_TEX_BC7_ERR"); VkFormatFeatureFlags want = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        maxe = e2 ? (float)atof(e2) : 8.0f;   // in the world the worst of 86 was 3.4
        on = (!e || atoi(e)) && (fmt_props(VK_FORMAT_BC1_RGB_UNORM_BLOCK).optimalTilingFeatures & want) == want && (fmt_props(VK_FORMAT_BC3_UNORM_BLOCK).optimalTilingFeatures & want) == want;
        if (on) bc_enc_init();
        LOG("vk: BC7 textures %s (DM_TEX_BC7), largest error allowed %.1f of 255 (DM_TEX_BC7_ERR)", on ? "become BC1 or BC3" : "decode to RGBA8", maxe); }
    if (!on || !t->mem || !t->L.size) { atomic_fetch_add(&g_bc7_n[2], 1); return keep; }
    bc7_survey_t s; uint64_t t0 = now_ns(); bc7_survey(t->mem, t->L.size, &s);
    // Alpha, level by level and laid out: the memory between a level's blocks can hold anything. A 3D texture counts as having alpha, without a look
    int opaque = s.a_min == 255 && t->d.target != 2 && !(t->d.flags & (0x10 | 0x40));
    for (uint32_t lv = 0; opaque && lv < MAX(levels, 1u); lv++) {
        uint32_t lw = MAX(w >> lv, 1u), lh = t->d.target == 3 ? 1 : MAX(h >> lv, 1u), bx = (lw + 3) / 4, by = (lh + 3) / 4; uint64_t pitch = (uint64_t)bx * 16;
        uint8_t *lin = malloc(pitch * by); if (!lin) { opaque = 0; break; }
        for (uint32_t ly = 0; opaque && ly < MAX(layers, 1u); ly++) if (nvn_deswizzle(&t->d, lv, ly, t->mem, lin, pitch) || !bc7_blocks_opaque(lin, (uint64_t)bx * by)) opaque = 0;
        free(lin);
    }
    int to = s.err_rgb > maxe || s.err_a > maxe ? 2 : opaque ? 0 : 1; atomic_fetch_add(&g_bc7_n[to], 1);
    { static int nl; if (MAX(w, h) >= 256 && nl++ < 128) LOG("vk: BC7 texture %ux%u x%u lv%u: %s, error of 255 over sampled blocks: colour %.1f, alpha %.1f, smallest alpha %d; looked at in %.1f ms", w, h, layers, levels, to == 2 ? "stays RGBA8" : to == 0 ? "BC1" : "BC3", (double)s.err_rgb, (double)s.err_a, s.a_min, (now_ns() - t0) / 1e6); }
    return to == 2 ? keep : to == 0 ? (srgb ? VK_FORMAT_BC1_RGB_SRGB_BLOCK : VK_FORMAT_BC1_RGB_UNORM_BLOCK) : (srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK);
}
// The host image of a texture (created on first use; submit thread, inside a submission)
static vimg *vimg_get(const tex_t *t) {
    if (!t || !t->serial) return NULL;
    // direct-mapped cache of serial -> image, used only here (the submit thread): no g_imu while no image was released since the entry was filled (g_ih_gen; the game releases a few
    // transient render targets a frame). The lock per lookup was ~3% of the replay thread
    static struct { uint64_t serial; vimg *v; uint32_t gen; } vc[512];
    uint32_t gen = atomic_load_explicit(&g_ih_gen, memory_order_acquire), slot = hmix(t->serial) & 511; vimg *v;
    if (vc[slot].serial == t->serial && vc[slot].gen == gen) v = vc[slot].v;
    else { pthread_mutex_lock(&g_imu); gen = atomic_load_explicit(&g_ih_gen, memory_order_relaxed); v = ih_get(t->serial); pthread_mutex_unlock(&g_imu);
        if (v) { vc[slot].serial = t->serial; vc[slot].v = v; vc[slot].gen = gen; } }
    if (v) {
        if (v->t.mem != t->mem || memcmp(&v->t.d, &t->d, sizeof t->d)) {   // diagnostic: one serial, two different textures (an object copied after TextureInitialize?)
            static int n; if (n++ < 12) LOG("[RED] vk: texture serial %#llx seen with different storage: mem %p → %p, %ux%u fmt %#x → %ux%u fmt %#x", (unsigned long long)t->serial,
                (const void *)v->t.mem, (const void *)t->mem, v->t.d.width, v->t.d.height, v->t.d.format, t->d.width, t->d.height, t->d.format);
        }
        return v;
    }
    const nvn_texdesc *d = &t->d;
    if (d->target == 10) { RED_ONCE(1, "a texture buffer (target BUFFER) at a sampled image binding: a texel buffer binding reads it (tbuf_view), this one reads the dummy"); return NULL; }
    VkFormat f = tex_vk_format(d->format);
    if (!f) { RED_ONCE(2, "texture format %#x has no Vulkan counterpart", d->format); return NULL; }
    fmt_t m = fmt_of(d->format);
    if (m.conv == CV_BC4U || m.conv == CV_BC4S) f = m.conv == CV_BC4U ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8_SNORM;
    else if (m.conv == CV_BC5U || m.conv == CV_BC5S) f = m.conv == CV_BC5U ? VK_FORMAT_R8G8_UNORM : VK_FORMAT_R8G8_SNORM;
    else if (m.conv == CV_BC7) f = bc7_fmt(t, d->width, d->height, d->target == 4 ? d->depth : d->target == 8 ? 6 : d->target == 9 ? 6 * d->depth : d->target == 3 ? d->height : 1, d->levels);
    else if (m.conv == CV_BC6) f = bc6_fmt(t);
    if (f == VK_FORMAT_S8_UINT && !(fmt_props(f).optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) f = VK_FORMAT_D32_SFLOAT_S8_UINT;
    v = calloc(1, sizeof *v); v->serial = t->serial; v->t = *t; v->fmt = f; v->aspect = vk_aspect(f);
    v->w = MAX(d->width, 1u); v->h = MAX(d->height, 1u); v->d = 1; v->layers = 1; v->levels = MAX(d->levels, 1u); v->samples = d->samples > 1 ? d->samples : 1; v->itype = VK_IMAGE_TYPE_2D;
    switch (d->target) {
        case 0: v->h = 1; break;
        case 2: v->itype = VK_IMAGE_TYPE_3D; v->d = MAX(d->depth, 1u); break;
        case 3: v->layers = MAX(d->height, 1u); v->h = 1; break;
        case 4: case 6: v->layers = MAX(d->depth, 1u); break;
        case 8: v->layers = 6; break;
        case 9: v->layers = 6 * MAX(d->depth, 1u); break;
    }
    // A large compressed texture with mipmaps can leave out its largest level, three quarters of its memory and of the time to decode it: the game's textures are made for a large screen, and this one is 6 inches. These can't be render targets, and the GPU can't write a
    // compressed format as an image, so nothing writes to them but uploads (the IMAGE flag alone says nothing here: RegisterImage sets it). A shader that asks the size sees the smaller one. In the world this is 296 MB of the 1,308 MB of images, and it shortens loading too.
    // The price is softer textures seen from very close. DM_TEX_DROP: how many levels (default 1, 0 = off). DM_TEX_DROP_MIN: from this width or height (default 1024). Cube maps are left alone: their levels stand for roughness
    { static int maxd = -1, mins; if (maxd < 0) { const char *e = getenv("DM_TEX_DROP"), *e2 = getenv("DM_TEX_DROP_MIN"); mins = e2 ? atoi(e2) : 1024; maxd = e ? atoi(e) : 1; g_tex_drop_on = maxd > 0; LOG("vk: large compressed textures leave out %d levels, from %d texels (DM_TEX_DROP, DM_TEX_DROP_MIN)", maxd, mins); }
      int cf = (m.conv >= CV_BC4U && m.conv <= CV_BC6) || (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_BC7_SRGB_BLOCK) || (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
      int able = cf && (d->target == 1 || d->target == 4) && v->samples == 1 && !(d->flags & (0x10 | 0x20 | 0x40)) && (int)v->levels >= MAX(maxd, 1) + 2 && (int)MAX(v->w, v->h) >= mins;
      if (cf && (int)MAX(v->w, v->h) >= mins) { static int nl; if (nl++ < 24) LOG("vk: compressed texture %ux%u x%u lv%u target %d NVN format %#x flags %#x: top level %s", v->w, v->h, v->layers, v->levels, d->target, d->format, d->flags, able ? (maxd > 0 ? "left out" : "could be left out") : "stays"); }
      v->top_able = able;
      if (able && maxd > 0) { v->drop = maxd; v->w = MAX(v->w >> maxd, 1u); v->h = MAX(v->h >> maxd, 1u); v->levels -= (uint32_t)maxd; } }
    v->cube_ok = v->itype == VK_IMAGE_TYPE_2D && v->w == v->h && v->layers >= 6 && v->samples == 1;
    VkFormatProperties fp = fmt_props(f); int imf = imgflags();
    VkImageUsageFlags u = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) u |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    int compressed = (f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_BC7_SRGB_BLOCK) || (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
    v->mutable_ = !compressed && !vk_is_depth(f) && f != VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    if ((d->flags & 4) && !(imf & 4)) {   // for image slots: storage views in its own format or, as a mutable extended-usage image, in another of its family (sRGB formats have no storage support, UNORM ones do)
        VkFormat fam[8]; int nf = v->mutable_ ? fmt_family(f, fam) : 0, can = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
        for (int i = 0; i < nf && !can; i++) can = (fmt_props(fam[i]).optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
        if (can) u |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    VkFormat twin = bc_twin_on() ? bc_twin(f) : VK_FORMAT_UNDEFINED;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = (v->cube_ok ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0) | (v->mutable_ ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0) | (twin ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0) |
            (v->itype == VK_IMAGE_TYPE_3D && (u & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ? VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT : 0),
        .imageType = v->itype, .format = f, .extent = {v->w, v->h, v->d}, .mipLevels = v->levels, .arrayLayers = v->layers, .samples = (VkSampleCountFlagBits)v->samples,
        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = u, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    if (v->samples > 1) ci.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
    VkFormat fl[8]; VkImageFormatListCreateInfo fli = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
    if (v->mutable_ && (imf & 2)) { fli.viewFormatCount = (uint32_t)fmt_family(f, fl); fli.pViewFormats = fl; ci.pNext = &fli; v->fl = 1; }
    if (twin) { fl[0] = f; fl[1] = twin; fli.viewFormatCount = 2; fli.pViewFormats = fl; ci.pNext = &fli; v->fl = 1; }
    v->usage = ci.usage; v->cflags = ci.flags;
    if (rc_on()) { vimg *o = rc_take(v);
        if (o) {   // same parameters: its image, memory and views; contents start undefined as for a new image
            v->img = o->img; v->mem = o->mem; memcpy(v->v, o->v, sizeof v->v); v->nv = o->nv; v->kb = o->kb; v->top_kb = o->top_kb; v->kind = o->kind; v->side = o->side; free(o->lin_copy); free(o); g_stat_img_reuse++;
            img_barrier_init(cur_up(), v->img, v->aspect); v->inited = 1;
            pthread_mutex_lock(&g_imu); ih_put(t->serial, v); pthread_mutex_unlock(&g_imu);
            return v;
        } }
    g_stat_img_new++;
    VkResult r = vkCreateImage(g_dev, &ci, NULL, &v->img);
    if (r && v->fl) { RED_ONCE(47, "vk: DM_VK_IMGFLAGS: creating an image of Vulkan format %d with a format list failed (%d): created without", (int)f, (int)r); ci.pNext = NULL; v->fl = 0; r = vkCreateImage(g_dev, &ci, NULL, &v->img); }
    if (r) { LOG("[RED] vk: creating a %ux%ux%u x%u lv%u image of NVN format %#x (Vulkan %d) failed: %d", v->w, v->h, v->d, v->layers, v->levels, d->format, (int)f, (int)r); free(v); return NULL; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev, v->img, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)};
    r = vkAllocateMemory(g_dev, &ai, NULL, &v->mem);
    if (r) { LOG("[RED] vk: out of memory for a %llu KB image: %d", (unsigned long long)(mr.size >> 10), (int)r); vkDestroyImage(g_dev, v->img, NULL); free(v); return NULL; }
    VKCHK(vkBindImageMemory(g_dev, v->img, v->mem, 0));
    img_barrier_init(cur_up(), v->img, v->aspect); v->inited = 1;
    v->kb = (uint32_t)((mr.size + 1023) >> 10); v->kind = m.conv == CV_BC7 ? IK_BC7 : m.conv == CV_BC6 ? IK_BC6 : (m.conv >= CV_BC4U && m.conv <= CV_BC5S) ? IK_BC45 : m.conv ? IK_CONV : compressed ? IK_BC13 : IK_OTHER;
    if (v->drop) v->top_kb = v->kb * ((1u << (2 * v->drop)) - 1);
    else if (v->top_able) v->top_kb = v->kb - v->kb / 4;
    atomic_fetch_add(&g_img_kb[v->kind], v->kb); atomic_fetch_add(&g_img_n[v->kind], 1); atomic_fetch_add(&g_img_top_kb, v->top_kb);
    if (v->kind != IK_OTHER && v->kind != IK_CONV) { uint32_t gs = MAX(d->width, d->height); int sb = gs >= 4096 ? 0 : gs >= 2048 ? 1 : gs >= 1024 ? 2 : gs >= 512 ? 3 : gs >= 256 ? 4 : 5, hm = MAX(d->levels, 1u) > 1;
        v->side = hm * 6 + sb + 1; atomic_fetch_add(&g_img_side_kb[hm][sb], v->kb); }
    if (v->kb >= 8192) { static int nb; if (nb++ < 64) LOG("vk: large image %ux%u x%u lv%u target %d NVN format %#x Vulkan %d flags %#x: %u MB", v->w, v->h, v->layers, v->levels, d->target, d->format, (int)f, d->flags, v->kb >> 10); }
    if (!g_fmt_seen[d->format & 0xff]++) LOG("vk: first texture of NVN format %#x: %ux%u x%u lv%u target %d → Vulkan format %d%s", d->format, v->w, v->h, v->layers, v->levels, d->target, (int)f,
        m.conv ? " (decoded on the CPU)" : "");
    if (v->itype == VK_IMAGE_TYPE_3D) { static int n3; if (n3++ < 16) LOG("vk: 3D texture serial %#llx %ux%ux%u lv%u NVN format %#x, storage %p flags %#x", (unsigned long long)t->serial, v->w, v->h, v->d, v->levels, d->format, (const void *)t->mem, d->flags); }
    pthread_mutex_lock(&g_imu); ih_put(t->serial, v); pthread_mutex_unlock(&g_imu);
    return v;
}
static VkImageView vimg_view(vimg *v, VkImageViewType type, VkFormat fmt, unsigned l0, unsigned nl, unsigned s0, unsigned ns, const uint8_t sw[4], VkImageAspectFlags aspect) {
    if (v->drop) { unsigned dr = (unsigned)v->drop; if (l0 >= dr) l0 -= dr; else { if (nl) nl = nl > dr - l0 ? nl - (dr - l0) : 1; l0 = 0; } }   // the caller's levels are the guest's
    if (l0 >= v->levels) l0 = v->levels - 1; if (!nl || l0 + nl > v->levels) nl = v->levels - l0;
    unsigned nlay = v->itype == VK_IMAGE_TYPE_3D && type != VK_IMAGE_VIEW_TYPE_3D ? MAX(v->d >> l0, 1u) : v->layers;   // 2D views of a 3D image: its depth planes
    if (s0 >= nlay) s0 = nlay - 1; if (!ns || s0 + ns > nlay) ns = nlay - s0;
    for (int i = 0; i < v->nv; i++) { vview *w = &v->v[i];
        if (w->type == type && w->fmt == fmt && w->l0 == l0 && w->nl == nl && w->s0 == s0 && w->ns == ns && !memcmp(w->sw, sw, 4) && w->aspect == aspect) return w->view; }
    VkImageViewUsageCreateInfo uci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
    VkFormatProperties fp = fmt_props(fmt);
    uci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) uci.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) uci.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) uci.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if ((v->usage & VK_IMAGE_USAGE_STORAGE_BIT) && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && !(sw[0] | sw[1] | sw[2] | sw[3])) uci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;   // storage descriptors need the identity swizzle
    VkImageViewCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, v->mutable_ ? &uci : NULL, .image = v->img, .viewType = type, .format = fmt,
        .components = {(VkComponentSwizzle)sw[0], (VkComponentSwizzle)sw[1], (VkComponentSwizzle)sw[2], (VkComponentSwizzle)sw[3]}, .subresourceRange = {aspect, l0, nl, s0, ns}};
    VkImageView view;
    VkResult r = vkCreateImageView(g_dev, &ci, NULL, &view);
    if (r) { RED_ONCE(3, "vk: creating an image view failed (%d): type %d format %d image format %d levels %u+%u layers %u+%u", (int)r, (int)type, (int)fmt, (int)v->fmt, l0, nl, s0, ns); return VK_NULL_HANDLE; }
    if (v->kb >= 8192) { static int nviewlog; if (nviewlog++ < 64) LOG("vk: view of a %u MB image %ux%u x%u lv%u: type %d, format %d, levels %u+%u, layers %u+%u", v->kb >> 10, v->w, v->h, v->layers, v->levels, (int)type, (int)fmt, l0, nl, s0, ns); }
    if (v->nv == 16) { dead *d = calloc(1, sizeof *d); d->views[d->nv++] = v->v[0].view; dead_add(d); memmove(&v->v[0], &v->v[1], 15 * sizeof v->v[0]); v->nv = 15; }
    v->v[v->nv++] = (vview){type, fmt, (uint16_t)l0, (uint16_t)nl, (uint16_t)s0, (uint16_t)ns, {sw[0], sw[1], sw[2], sw[3]}, aspect, view};
    return view;
}
static const uint8_t SW_ID[4] = {0, 0, 0, 0};

// Staging for texture uploads. They go through the ring, whose memory stays taken until the submission ends, so a submission that loads a whole scene would hold as much staging as the scene's textures, on top of the images themselves.
// Past DM_VK_UP_LIMIT MB (default 8) in one submission, or for a level over a ring chunk, they go through an arena instead (DM_VK_UP_ARENA MB, default 32): when it is full, the upload command buffer runs at once and is waited for (up_flush), and the arena starts over.
// A level larger than the arena gets a buffer of its own, gone once its copy has run. So staging stays within the limit, the arena and the largest level, whatever the scene loads. Normal frames never get here, and the arena exists only while it is used
static struct { hbuf h; uint64_t head, size, limit, sub_bytes, ring_bytes, last_sub, wait_ns; int flushes, inited; } g_ups;
static void up_flush(void);
static void hbuf_free_now(const hbuf *h) { vkDestroyBuffer(g_dev, h->buf, NULL); vkFreeMemory(g_dev, h->mem, NULL); }   // no command may still use it
static void ups_init(void) {
    if (g_ups.inited) return;
    g_ups.inited = 1; g_ups.size = 32u << 20; g_ups.limit = UP_RING_LIMIT;
    const char *e = getenv("DM_VK_UP_ARENA"); if (e && atoi(e) > 0) g_ups.size = (uint64_t)atoi(e) << 20;
    e = getenv("DM_VK_UP_LIMIT"); if (e) g_ups.limit = (uint64_t)atoi(e) << 20;
}
static halloc up_alloc(uint64_t n) {
    g_ups.sub_bytes += n;
    if (n <= RING_CHUNK && g_ups.sub_bytes <= g_ups.limit) { g_ups.ring_bytes += n; return ring_alloc(n, 16); }
    if (!g_ups.h.buf) { g_ups.h = hbuf_new(g_ups.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT); g_ups.head = 0; }
    uint64_t o = (g_ups.head + 15) & ~15ull;
    if (o + n > g_ups.size) { up_flush(); o = 0; }
    g_ups.head = o + n; g_ups.last_sub = g_nsub;
    return (halloc){g_ups.h.buf, o, g_ups.h.map + o};
}
// Texture conversions on several threads. The submit thread deswizzles a level and queues the conversion into its staging memory, in slices of block rows; helper threads and the submit thread itself work through the queue. Nothing converted is submitted before
// cv_wait (up_flush, and the end of vimg_upload), so this stays inside one upload. Small levels run at once on the submit thread. DM_TEX_THREADS: helper threads, default 3 (0 = all on the submit thread)
enum { CVJ_BC45, CVJ_BC7, CVJ_BC7_BC, CVJ_BC6, CVJ_BC6_E5, CVJ_BC6_BC1 };
typedef struct { _Atomic int left; uint8_t *lin, *dst; uint64_t pitch, dstrow; uint32_t bx, rows; uint8_t kind, two, sn, bc1; } cvjob;   // one level of one layer. pitch, dstrow: bytes from one row of blocks to the next in lin and dst
enum { CV_Q = 256, CV_SLICE_ROWS = 32, CV_SMALL_BLOCKS = 1024 };
static struct { pthread_mutex_t mu; pthread_cond_t work, idle; struct { cvjob *j; uint32_t y0, n; } q[CV_Q]; int head, n, busy, nth; } g_cv = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
static void cv_slice(cvjob *j, uint32_t y0, uint32_t n) {
    const uint8_t *s = j->lin + y0 * j->pitch; uint8_t *d = j->dst + y0 * j->dstrow;
    switch (j->kind) {
        case CVJ_BC45: bc45_decode(s, d, j->bx, n, j->two, j->sn); break;
        case CVJ_BC7: bc7_decode(s, d, j->bx, n); break;
        case CVJ_BC7_BC: bc7_to_bc(s, d, j->bx, n, j->bc1); break;
        case CVJ_BC6: bc6h_decode(s, d, j->bx, n, j->sn); break;
        case CVJ_BC6_E5: bc6h_decode_e5(s, d, j->bx, n); break;
        case CVJ_BC6_BC1: bc6h_to_bc1(s, d, j->bx, n); break;
    }
    if (atomic_fetch_sub(&j->left, 1) == 1) { free(j->lin); free(j); }
}
static void *cv_main(void *arg) {
    (void)arg; pthread_setname_np("tex-convert");
    pthread_mutex_lock(&g_cv.mu);
    for (;;) {
        while (!g_cv.n) pthread_cond_wait(&g_cv.work, &g_cv.mu);
        cvjob *j = g_cv.q[g_cv.head].j; uint32_t y0 = g_cv.q[g_cv.head].y0, n = g_cv.q[g_cv.head].n; g_cv.head = (g_cv.head + 1) % CV_Q; g_cv.n--; g_cv.busy++;
        pthread_mutex_unlock(&g_cv.mu);
        cv_slice(j, y0, n);
        pthread_mutex_lock(&g_cv.mu);
        g_cv.busy--; if (!g_cv.n && !g_cv.busy) pthread_cond_broadcast(&g_cv.idle);
    }
    return NULL;
}
// Takes lin (malloc'ed), which is freed once converted
static void cv_push(int kind, uint8_t *lin, uint8_t *dst, uint64_t pitch, uint64_t dstrow, uint32_t bx, uint32_t rows, int two, int sn, int bc1) {
    static int nth = -1;
    if (nth < 0) { const char *e = getenv("DM_TEX_THREADS"); nth = e ? atoi(e) : 3; if (nth < 0) nth = 0; if (nth > 7) nth = 7; int ok = 0;
        for (int i = 0; i < nth; i++) { pthread_t th; if (!pthread_create(&th, NULL, cv_main, NULL)) { pthread_detach(th); ok++; } }
        g_cv.nth = ok; LOG("vk: textures convert on %d helper threads and the submit thread (DM_TEX_THREADS)", ok); }
    cvjob *j = malloc(sizeof *j); if (!j) die("vk: out of memory for a texture conversion");
    j->lin = lin; j->dst = dst; j->pitch = pitch; j->dstrow = dstrow; j->bx = bx; j->rows = rows; j->kind = (uint8_t)kind; j->two = (uint8_t)two; j->sn = (uint8_t)sn; j->bc1 = (uint8_t)bc1;
    uint32_t nsl = (rows + CV_SLICE_ROWS - 1) / CV_SLICE_ROWS;
    if (!g_cv.nth || (uint64_t)bx * rows < CV_SMALL_BLOCKS) { atomic_init(&j->left, 1); cv_slice(j, 0, rows); return; }
    atomic_init(&j->left, (int)nsl);
    for (uint32_t i = 0; i < nsl; i++) { uint32_t y0 = i * CV_SLICE_ROWS, n = MIN(rows - y0, (uint32_t)CV_SLICE_ROWS);
        pthread_mutex_lock(&g_cv.mu);
        if (g_cv.n == CV_Q) { pthread_mutex_unlock(&g_cv.mu); cv_slice(j, y0, n); continue; }   // queue full: this thread helps
        g_cv.q[(g_cv.head + g_cv.n) % CV_Q].j = j; g_cv.q[(g_cv.head + g_cv.n) % CV_Q].y0 = y0; g_cv.q[(g_cv.head + g_cv.n) % CV_Q].n = n; g_cv.n++;
        pthread_cond_signal(&g_cv.work); pthread_mutex_unlock(&g_cv.mu); }
}
// Everything queued is converted when this returns. The calling thread works through what is left
static void cv_wait(void) {
    pthread_mutex_lock(&g_cv.mu);
    while (g_cv.n) {
        cvjob *j = g_cv.q[g_cv.head].j; uint32_t y0 = g_cv.q[g_cv.head].y0, n = g_cv.q[g_cv.head].n; g_cv.head = (g_cv.head + 1) % CV_Q; g_cv.n--; g_cv.busy++;
        pthread_mutex_unlock(&g_cv.mu);
        cv_slice(j, y0, n);
        pthread_mutex_lock(&g_cv.mu);
        g_cv.busy--;
    }
    while (g_cv.busy) pthread_cond_wait(&g_cv.idle, &g_cv.mu);
    pthread_mutex_unlock(&g_cv.mu);
}
// Upload a texture's content from guest memory (deswizzled) into its image; recorded into the upload command buffer, which runs before this submission's commands
static void vimg_upload(vimg *v) {
    const tex_t *t = &v->t; const nvn_texdesc *d = &t->d;
    if (!t->mem) { v->uploaded = 1; return; }
    fmt_t m = fmt_of(d->format);
    int bw, bh, bpb = nvn_fmt_bpb(d->format, &bw, &bh);
    if (!bpb) { RED_ONCE(4, "vk: texture format %#x has no block size: not uploaded", d->format); v->uploaded = 1; return; }
    if (d->samples > 1) { v->uploaded = 1; return; }
    int linear = (d->flags & (0x10 | 0x40)) != 0;
    ups_init();
    uint32_t nlayer = v->itype == VK_IMAGE_TYPE_3D ? 1 : v->layers;
    for (uint32_t lv = 0; lv < v->levels; lv++) {
        uint32_t lw = MAX(v->w >> lv, 1u), lh = MAX(v->h >> lv, 1u), lz = v->itype == VK_IMAGE_TYPE_3D ? MAX(v->d >> lv, 1u) : 1;
        uint32_t bx = (lw + bw - 1) / bw, by = (lh + bh - 1) / bh;
        uint64_t pitch = (uint64_t)bx * bpb, lsize = pitch * by * lz;
        for (uint32_t ly = 0; ly < nlayer; ly++) {
            uint64_t outsz = lsize; uint32_t ocpt = 0;   // converted size / bytes per texel of the conversion output
            if (m.conv == CV_BC4U || m.conv == CV_BC4S) { ocpt = 1; outsz = (uint64_t)bx * 4 * by * 4 * lz; }
            else if (m.conv == CV_BC5U || m.conv == CV_BC5S) { ocpt = 2; outsz = (uint64_t)bx * 4 * by * 4 * lz * 2; }
            else if (m.conv == CV_BC7 && bc_twin(v->fmt)) outsz = (uint64_t)bx * by * lz * (v->fmt == VK_FORMAT_BC1_RGB_UNORM_BLOCK || v->fmt == VK_FORMAT_BC1_RGB_SRGB_BLOCK ? 8 : 16);
            else if (m.conv == CV_BC7) { ocpt = 4; outsz = (uint64_t)bx * 4 * by * 4 * lz * 4; }
            else if (m.conv == CV_BC6 && bc_twin(v->fmt)) outsz = (uint64_t)bx * by * lz * 8;
            else if (m.conv == CV_BC6) { ocpt = v->fmt == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 ? 4 : 8; outsz = (uint64_t)bx * 4 * by * 4 * lz * ocpt; }
            else if (m.conv >= CV_RGB_EXPAND8) { ocpt = bpb / 3 * 4; outsz = (uint64_t)lw * lh * lz * ocpt; }
            // A level larger than the arena gets a buffer of its own: a 4096x4096 BC7 texture decodes to 64 MB
            hbuf big = {0}; if (outsz > g_ups.size) big = hbuf_new(outsz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            halloc st = big.buf ? (halloc){big.buf, 0, big.map} : up_alloc(outsz);
            uint8_t *lin = m.conv ? malloc(lsize) : st.p;
            if (linear) {
                uint64_t stride = d->stride ? d->stride : pitch; const uint8_t *src = t->mem + (lv + v->drop ? nvn_tex_level_offset(d, lv + v->drop) : 0) + (uint64_t)ly * t->L.layer_size;
                for (uint32_t z = 0; z < lz; z++) for (uint32_t y = 0; y < by; y++) memcpy(lin + ((uint64_t)z * by + y) * pitch, src + ((uint64_t)z * by + y) * stride, pitch);
            } else if (nvn_deswizzle(d, lv + v->drop, ly, t->mem, lin, pitch)) { RED_ONCE(6, "vk: deswizzling a %ux%u texture of format %#x target %d failed", lw, lh, d->format, d->target); if (m.conv) free(lin); if (big.buf) hbuf_free_now(&big); continue; }
            uint32_t rowlen = bx * bw, imgh = by * bh;   // buffer rows in texels: whole blocks (Vulkan clips the copy to the level)
            // The rows of blocks of a 3D level's slices follow each other, in lin and in staging, so one queue entry covers them all
            if (m.conv == CV_BC4U || m.conv == CV_BC4S || m.conv == CV_BC5U || m.conv == CV_BC5S) {
                cv_push(CVJ_BC45, lin, st.p, pitch, (uint64_t)bx * 4 * 4 * ocpt, bx, by * lz, m.conv >= CV_BC5U, m.conv == CV_BC4S || m.conv == CV_BC5S, 0); lin = NULL;
                rowlen = bx * 4; imgh = by * 4;
            } else if (m.conv == CV_BC7 && bc_twin(v->fmt)) {
                int bc1 = v->fmt == VK_FORMAT_BC1_RGB_UNORM_BLOCK || v->fmt == VK_FORMAT_BC1_RGB_SRGB_BLOCK;
                cv_push(CVJ_BC7_BC, lin, st.p, pitch, (uint64_t)bx * (bc1 ? 8 : 16), bx, by * lz, 0, 0, bc1); lin = NULL;
                rowlen = bx * 4; imgh = by * 4;
            } else if (m.conv == CV_BC7) {
                cv_push(CVJ_BC7, lin, st.p, pitch, (uint64_t)bx * 4 * 4 * 4, bx, by * lz, 0, 0, 0); lin = NULL;
                rowlen = bx * 4; imgh = by * 4;
            } else if (m.conv == CV_BC6 && bc_twin(v->fmt)) {
                cv_push(CVJ_BC6_BC1, lin, st.p, pitch, (uint64_t)bx * 8, bx, by * lz, 0, 0, 1); lin = NULL;
                rowlen = bx * 4; imgh = by * 4;
            } else if (m.conv == CV_BC6) {
                cv_push(ocpt == 4 ? CVJ_BC6_E5 : CVJ_BC6, lin, st.p, pitch, (uint64_t)bx * 4 * 4 * ocpt, bx, by * lz, 0, d->format == 0x4f, 0); lin = NULL;
                rowlen = bx * 4; imgh = by * 4;
            } else if (m.conv >= CV_RGB_EXPAND8) {
                uint32_t c = bpb / 3; const uint8_t one8 = 0xff; const uint16_t one16 = m.vk == VK_FORMAT_R16G16B16_SFLOAT ? 0x3c00 : 0xffff; const uint32_t one32 = m.vk == VK_FORMAT_R32G32B32_SFLOAT ? 0x3f800000u : 1u;
                for (uint64_t i = 0; i < (uint64_t)lw * lh * lz; i++) { memcpy(st.p + i * 4 * c, lin + i * 3 * c, 3 * c);
                    if (c == 1) memcpy(st.p + i * 4 + 3, &one8, 1); else if (c == 2) memcpy(st.p + i * 8 + 6, &one16, 2); else memcpy(st.p + i * 16 + 12, &one32, 4); }
                rowlen = lw; imgh = lh;
            }
            if (m.conv) free(lin);   // NULL if a conversion took it
            VkBufferImageCopy bic = {.bufferOffset = st.off, .bufferRowLength = rowlen, .bufferImageHeight = imgh,
                .imageSubresource = {v->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : v->aspect, lv, v->itype == VK_IMAGE_TYPE_3D ? 0 : ly, 1}, .imageExtent = {lw, lh, lz}};
            if (v->aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) { if (big.buf) hbuf_free_now(&big); continue; }   // depth-stencil content is never uploaded (D24S8 has no layout in common with D32S8)
            vkCmdCopyBufferToImage(cur_up(), st.buf, v->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bic);
            g_stat_up_bytes += outsz;
            if (big.buf) { up_flush(); hbuf_free_now(&big); }
        }
    }
    cv_wait();
    g_stat_up_n++;
    v->uploaded = 1; v->cpu_dirty = 0;
    if (vimg_linear_cpu(v)) { lin_snap(v); v->lin_sub = g_nsub; }
}

// Linear textures the CPU writes in place: the Metal replay makes them on the pool buffer, so CPU writes show at once; here the image is a copy, made again when the
// guest bytes change (compared at most once per submission). A ScalableFont glyph cache (nn::font TextureCache) plots each new glyph into such a texture: glyphs
// first needed after the first upload stayed blank (letters missing from the game's text)
static int vimg_linear_cpu(const vimg *v) { return !v->gpu_owned && v->t.mem && (v->t.d.flags & 0x10) && !(v->t.d.flags & 0x40) && v->t.L.size && v->t.L.size <= (64u << 20); }
// The guest bytes are kept as a copy and compared with memcmp: a hash of them every submission was ~2 ms a frame of the replay thread (a serial multiply chain; simpleperf, push_stage -> lin_hash)
enum { LIN_BLK = 256 << 10 };
static void lin_snap(vimg *v) { uint64_t n = v->t.L.size; if (!v->lin_copy) v->lin_copy = malloc(n); if (v->lin_copy) memcpy(v->lin_copy, v->t.mem, n); }
static uint64_t g_lin_cmp_n, g_lin_cmp_bytes, g_lin_max, g_lin_up; static uint32_t g_lin_max_w, g_lin_max_h, g_lin_max_fmt;   // compares, bytes compared, the largest texture compared, changes seen (uploaded again)
// DM_VK_LIN_CHECK=1: also compare large textures whole when the blocks saw no change, and count the changes the blocks missed and how many submissions late they were seen
static int g_lin_check = -1; static uint64_t g_lin_miss_sub, g_lin_lat_n, g_lin_lat_sum, g_lin_lat_max;
static int lin_changed(vimg *v) {
    if (g_lin_check < 0) { const char *e = getenv("DM_VK_LIN_CHECK"); g_lin_check = e && atoi(e); if (g_lin_check) LOG("vk: DM_VK_LIN_CHECK=1: block compares of linear textures are checked against whole compares"); }
    if (!vimg_linear_cpu(v) || !v->uploaded || v->lin_sub == g_nsub) return 0;
    v->lin_sub = g_nsub;
    g_lin_cmp_n++; if (v->t.L.size > g_lin_max) { g_lin_max = v->t.L.size; g_lin_max_w = v->w; g_lin_max_h = v->h; g_lin_max_fmt = v->t.d.format; }
    if (!v->lin_copy) return 1;   // no copy (out of memory): uploaded every submission
    uint64_t sz = v->t.L.size;
    if (sz <= 4 * LIN_BLK) { g_lin_cmp_bytes += sz; if (!memcmp(v->lin_copy, v->t.mem, sz)) return 0; }
    else {   // large (a glyph cache of 2048x4096 R8 is 8 MB: a whole compare was ~1.7 ms a frame): the 4 blocks that changed last (the glyph cache writes to a few regions, one per
        // font size: blocks 0, 22, 27 in one game's menu), the one after the latest (new glyphs are packed next to the last ones) and 2 blocks of a rolling scan, so a change in a new
        // region is seen within nb / 2 submissions (DM_VK_LIN_CHECK measures it). Any change uploads the whole texture and copies it again (lin_snap)
        uint32_t nb = (uint32_t)((sz + LIN_BLK - 1) / LIN_BLK), cand[7]; int nc = 0, hit = -1;
        for (int k = 0; k < 4; k++) if (v->lin_mru[k]) cand[nc++] = (v->lin_mru[k] - 1) % nb;
        cand[nc++] = v->lin_mru[0] ? v->lin_mru[0] % nb : 0; cand[nc++] = v->lin_roll % nb; cand[nc++] = (v->lin_roll + 1) % nb;
        v->lin_roll = (v->lin_roll + 2) % nb;
        for (int k = 0; k < nc && hit < 0; k++) { int dup = 0; for (int j = 0; j < k; j++) dup |= cand[j] == cand[k]; if (dup) continue;
            uint64_t o = (uint64_t)cand[k] * LIN_BLK, len = MIN((uint64_t)LIN_BLK, sz - o); g_lin_cmp_bytes += len; if (memcmp(v->lin_copy + o, v->t.mem + o, len)) hit = (int)cand[k]; }
        if (hit < 0) {
            if (g_lin_check && memcmp(v->lin_copy, v->t.mem, sz)) { g_lin_miss_sub++; if (!v->lin_miss) v->lin_miss = g_nsub; }
            return 0;
        }
        if (v->lin_miss) { uint64_t lat = g_nsub - v->lin_miss; g_lin_lat_n++; g_lin_lat_sum += lat; if (lat > g_lin_lat_max) g_lin_lat_max = lat;
            static int nl; if (nl++ < 20) LOG("vk: DM_VK_LIN_CHECK: %ux%u change seen %llu submissions late (block %d; recent blocks %d %d %d %d)", v->w, v->h, (unsigned long long)lat, hit,
                (int)v->lin_mru[0] - 1, (int)v->lin_mru[1] - 1, (int)v->lin_mru[2] - 1, (int)v->lin_mru[3] - 1);
            v->lin_miss = 0; }
        { uint32_t b = (uint32_t)hit + 1; int k = 0; while (k < 3 && v->lin_mru[k] != b) k++; for (; k > 0; k--) v->lin_mru[k] = v->lin_mru[k - 1]; v->lin_mru[0] = b; }   // move to front
    }
    g_lin_up++; static int n; if (n++ < 4) LOG("vk: linear texture serial %#llx %ux%u fmt %#x changed in guest memory: uploaded again", (unsigned long long)v->serial, v->w, v->h, v->t.d.format);
    return 1;
}

// ---------- Descriptor tables (texture / sampler pools) ----------
typedef struct { tex_t t; view_t v; uint8_t ok, has_v; } treg;
typedef struct { samp_par p; uint8_t ok; } sreg;
// Descriptor registrations by id (game threads write, the replay reads each draw): chunks of REG_CH entries that are never moved or freed, and a sequence number per entry
// (odd while written) so the reader copies an entry without a lock and retries if a write overlapped (the g_rmu lock per binding was in push_stage's ~2% of mutex time)
enum { REG_CH = 1024, REG_NCH = (1 << 20) / REG_CH };
typedef struct { _Atomic uint32_t seq; treg r; } tslot;
typedef struct { _Atomic uint32_t seq; sreg r; } sslot;
static _Atomic(tslot *) g_tch[REG_NCH]; static _Atomic(sslot *) g_sch[REG_NCH]; static pthread_mutex_t g_rmu = PTHREAD_MUTEX_INITIALIZER;   // g_rmu: writers only
void mtl_set_tex_pool_n(int n) { (void)n; }
void mtl_reg_tex(int id, const tex_t *t, const view_t *v) {
    if (id < 0 || id >= (1 << 20)) return;
    pthread_mutex_lock(&g_rmu);
    tslot *c = atomic_load_explicit(&g_tch[id / REG_CH], memory_order_relaxed);
    if (!c) { if (!(c = calloc(REG_CH, sizeof *c))) die("vk: texture registry allocation failed"); atomic_store_explicit(&g_tch[id / REG_CH], c, memory_order_release); }
    tslot *e = &c[id % REG_CH]; uint32_t q = atomic_load_explicit(&e->seq, memory_order_relaxed);
    atomic_store_explicit(&e->seq, q + 1, memory_order_relaxed); atomic_thread_fence(memory_order_release);
    e->r.t = *t; e->r.ok = 1; e->r.has_v = v != NULL; if (v) e->r.v = *v; else memset(&e->r.v, 0, sizeof e->r.v);
    atomic_store_explicit(&e->seq, q + 2, memory_order_release);
    pthread_mutex_unlock(&g_rmu);
}
void mtl_reg_samp(int id, const samp_par *p) {
    if (id < 0 || id >= (1 << 20)) return;
    pthread_mutex_lock(&g_rmu);
    sslot *c = atomic_load_explicit(&g_sch[id / REG_CH], memory_order_relaxed);
    if (!c) { if (!(c = calloc(REG_CH, sizeof *c))) die("vk: sampler registry allocation failed"); atomic_store_explicit(&g_sch[id / REG_CH], c, memory_order_release); }
    sslot *e = &c[id % REG_CH]; uint32_t q = atomic_load_explicit(&e->seq, memory_order_relaxed);
    atomic_store_explicit(&e->seq, q + 1, memory_order_relaxed); atomic_thread_fence(memory_order_release);
    e->r.p = *p; e->r.ok = 1;
    atomic_store_explicit(&e->seq, q + 2, memory_order_release);
    pthread_mutex_unlock(&g_rmu);
}
static int reg_tex_get(int id, treg *out) {   // 1 if registered: a consistent copy in *out
    if (id <= 0 || id >= (1 << 20)) return 0;
    tslot *c = atomic_load_explicit(&g_tch[id / REG_CH], memory_order_acquire); if (!c) return 0;
    tslot *e = &c[id % REG_CH];
    for (;;) { uint32_t q = atomic_load_explicit(&e->seq, memory_order_acquire); if (q & 1) continue;
        *out = e->r; atomic_thread_fence(memory_order_acquire);
        if (atomic_load_explicit(&e->seq, memory_order_relaxed) == q) return out->ok; }
}
static uint32_t reg_tex_seq(int id) {   // the entry's version (odd while written; 0 if never registered): with reg_samp_seq, the key of tex_binding's cache
    if (id <= 0 || id >= (1 << 20)) return 0;
    tslot *c = atomic_load_explicit(&g_tch[id / REG_CH], memory_order_acquire); return c ? atomic_load_explicit(&c[id % REG_CH].seq, memory_order_acquire) : 0;
}
static uint32_t reg_samp_seq(int id) {
    if (id <= 0 || id >= (1 << 20)) return 0;
    sslot *c = atomic_load_explicit(&g_sch[id / REG_CH], memory_order_acquire); return c ? atomic_load_explicit(&c[id % REG_CH].seq, memory_order_acquire) : 0;
}
static int reg_samp_get(int id, samp_par *out) {
    if (id <= 0 || id >= (1 << 20)) return 0;
    sslot *c = atomic_load_explicit(&g_sch[id / REG_CH], memory_order_acquire); if (!c) return 0;
    sslot *e = &c[id % REG_CH]; sreg r;
    for (;;) { uint32_t q = atomic_load_explicit(&e->seq, memory_order_acquire); if (q & 1) continue;
        r = e->r; atomic_thread_fence(memory_order_acquire);
        if (atomic_load_explicit(&e->seq, memory_order_relaxed) == q) { if (r.ok) *out = r.p; return r.ok; } }
}
// Samplers by parameters
static struct { samp_par p; int dref; VkSampler s; } *g_samp; static int g_nsamp, g_csamp;
static VkSamplerAddressMode wrapm(int w) {
    switch (w) { case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT; case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; case 2: case 3: case 4: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
                 case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; }
}
// found by hash (open addressing over g_samp indices + 1): the linear search over every sampler made so far was 3.6% of the replay thread (~1000 samplers while walking)
static int *g_samph; static uint32_t g_samphcap;
static uint32_t samp_hash(const samp_par *p, int dref) {
    uint64_t h = 0x9e3779b97f4a7c15ull ^ (uint64_t)dref, w; const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i + 8 <= sizeof *p; i += 8) { memcpy(&w, b + i, 8); h = (h ^ w) * 0xff51afd7ed558ccdull; h ^= h >> 32; }
    return (uint32_t)h;
}
static void samph_put(int idx) {   // idx into g_samp
    if (((uint32_t)idx + 1) * 2 > g_samphcap) {
        free(g_samph); g_samphcap = g_samphcap ? 2 * g_samphcap : 1024; g_samph = calloc(g_samphcap, sizeof *g_samph);
        for (int i = 0; i < idx; i++) samph_put(i);
    }
    for (uint32_t k = samp_hash(&g_samp[idx].p, g_samp[idx].dref) & (g_samphcap - 1);; k = (k + 1) & (g_samphcap - 1)) if (!g_samph[k]) { g_samph[k] = idx + 1; return; }
}
_Static_assert(sizeof(samp_par) % 8 == 0, "samp_hash reads samp_par in 8-byte words");
static VkSampler sampler_get(const samp_par *p, int dref) {
    if (g_samphcap) for (uint32_t k = samp_hash(p, dref) & (g_samphcap - 1); g_samph[k]; k = (k + 1) & (g_samphcap - 1)) {
        const int i = g_samph[k] - 1; if (g_samp[i].dref == dref && !memcmp(&g_samp[i].p, p, sizeof *p)) return g_samp[i].s; }
    VkSamplerCustomBorderColorCreateInfoEXT bc = {VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT, .format = VK_FORMAT_UNDEFINED};
    memcpy(bc.customBorderColor.float32, p->border, 16);
    VkSamplerReductionModeCreateInfo rd = {VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, NULL, p->reduction == 1 ? VK_SAMPLER_REDUCTION_MODE_MIN : p->reduction == 2 ? VK_SAMPLER_REDUCTION_MODE_MAX : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE};
    int mip = p->minf >= 2, border = p->ws == 5 || p->wt == 5 || p->wr == 5;
    VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, border ? (void *)&bc : NULL,
        .magFilter = p->magf ? VK_FILTER_LINEAR : VK_FILTER_NEAREST, .minFilter = (p->minf == 1 || p->minf == 3 || p->minf == 5) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = (p->minf == 4 || p->minf == 5) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = wrapm(p->ws), .addressModeV = wrapm(p->wt), .addressModeW = wrapm(p->wr), .mipLodBias = p->lod_bias,
        .anisotropyEnable = p->aniso > 1.f, .maxAnisotropy = MIN(MAX(p->aniso, 1.f), g_lim.maxSamplerAnisotropy),
        .compareEnable = dref, .compareOp = dref ? (VkCompareOp)((p->cmp_func >= 1 && p->cmp_func <= 8 ? p->cmp_func : 4) - 1) : VK_COMPARE_OP_NEVER,
        .minLod = mip ? p->lod_min : 0, .maxLod = mip ? MAX(p->lod_max, p->lod_min) : 0.25f, .borderColor = border ? VK_BORDER_COLOR_FLOAT_CUSTOM_EXT : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK};
    if (p->reduction) { rd.pNext = (void *)ci.pNext; ci.pNext = &rd; }
    VkSampler s; VkResult r = vkCreateSampler(g_dev, &ci, NULL, &s);
    if (r) { RED_ONCE(7, "vk: creating a sampler failed: %d", (int)r); return VK_NULL_HANDLE; }
    if (g_nsamp == g_csamp) { g_csamp = g_csamp ? 2 * g_csamp : 256; g_samp = realloc(g_samp, g_csamp * sizeof *g_samp); }
    g_samp[g_nsamp].p = *p; g_samp[g_nsamp].dref = dref; g_samp[g_nsamp].s = s; samph_put(g_nsamp); g_nsamp++;
    if (g_nsamp == 3000) LOG("[RED] vk: 3000 distinct samplers");
    return s;
}

// ---------- Shaders and their binding tables ----------
enum { K_UNK, K_SUP, K_C1, K_DRV0, K_DRV2, K_UB, K_SB, K_SBU, K_TX, K_TU, K_IM, K_UBP, K_PUSH };
typedef struct { uint8_t binding, kind; char type, dim; uint8_t arr, dref, ct, f1; int a, b, c, d; } vbind;   // f1: vk_extra.tsv f1:<binding>, a 1D integer fetch read as a 2D texelFetch at (x & 0x3FFF, x >> 14) (tex_binding_f1)
typedef struct vshader {
    char key[20]; int stage, set; VkShaderModule mod; int nb; vbind b[40]; int nin; struct { uint8_t loc; char ct; uint8_t n; } in[32];
    VkDescriptorSetLayout dsl; VkBuffer c1buf; uint64_t c1off, c1size; int has_c1; int failed; int layer_out;
} vshader;
static VkBuffer g_zero_buf;   // the first 64 KB block of the static arena, all zeros (set in dev_init)
// <DM_VK_SHADERS>/shaders.pack (android-host/tools/pokopia_vkshaders.py): a header, an index sorted by key, then per shader its row (stage, set, bindings, inputs / outputs, extras), its c1 segment and its
// zlib-compressed SPIR-V. Pokopia has about 99,000 shaders, of which a session uses a few thousand, and their code is about 33 GB uncompressed: one file, only the index in memory, a record read when a program first needs it
typedef struct { uint64_t key, off; uint32_t row_len, c1_len, spv_len, spv_raw; } pack_ent;
static pack_ent *g_refl; static vshader **g_refl_sh; static uint32_t g_nrefl; static int g_pack_fd = -1;
static int pack_read(void *d, size_t n, uint64_t off) {
    for (uint8_t *p = d; n; ) { ssize_t r = pread(g_pack_fd, p, n, (off_t)off); if (r < 0 && errno == EINTR) continue; if (r <= 0) return 0; p += r; n -= (size_t)r; off += (uint64_t)r; }
    return 1;
}
static void refl_load(void) {
    const char *d = getenv("DM_VK_SHADERS"); char p[600]; snprintf(p, sizeof p, "%s/shaders.pack", d ? d : "vk");
    g_pack_fd = open(p, O_RDONLY | O_CLOEXEC);
    if (g_pack_fd < 0) { LOG("[RED] vk: no shader pack %s (DM_VK_SHADERS names the folder of android-host/tools/pokopia_vkshaders.py output): nothing will draw (%s)", p, strerror(errno)); return; }
    struct { char magic[8]; uint32_t ver, n; uint64_t index; } h;
    if (!pack_read(&h, sizeof h, 0) || memcmp(h.magic, "PKSHPACK", 8) || h.ver != 1) { LOG("[RED] vk: %s is not a shader pack of this version: the copy may be damaged, copy it again; nothing will draw", p); return; }
    pack_ent *e = malloc(h.n * sizeof *e); vshader **sh = calloc(h.n, sizeof *sh);
    if (!e || !sh || !pack_read(e, h.n * sizeof *e, h.index)) { LOG("[RED] vk: cannot read the index of %s (%u shaders): the copy may have been cut short, copy it again; nothing will draw", p, h.n); free(e); free(sh); return; }
    g_refl = e; g_refl_sh = sh; g_nrefl = h.n;
    LOG("vk: shader pack %s: %u shaders", p, g_nrefl);
}
static void *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    if (s < 0) { fclose(f); return NULL; }
    void *b = malloc(s ? s : 1); if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, s, f) != (size_t)s) { free(b); fclose(f); return NULL; }
    fclose(f); *n = s; return b;
}
static void vk_parse_bind(vbind *o, char *tok) {   // binding:type:kind[:a[:b[:c[:d]]]]
    char *f[8] = {0}; int n = 0; for (char *s = strtok(tok, ":"); s && n < 8; s = strtok(NULL, ":")) f[n++] = s;
    memset(o, 0, sizeof *o); if (n < 3) return;
    o->binding = (uint8_t)atoi(f[0]); o->type = f[1][0];
    if (o->type == 'c' || o->type == 'i') { o->dim = f[1][1]; const char *q = f[1] + 2; if (*q == 'a') { o->arr = 1; q++; } if (*q == 's') { o->dref = 1; q++; } o->ct = *q; }
    const char *k = f[2]; int x[4] = {0}; for (int i = 0; i < 4 && 3 + i < n; i++) x[i] = atoi(f[3 + i]);
    o->a = x[0]; o->b = x[1]; o->c = x[2]; o->d = x[3];
    if (!strcmp(k, "sup")) o->kind = K_SUP; else if (!strcmp(k, "c1")) o->kind = K_C1; else if (!strcmp(k, "drv")) o->kind = x[0] == 2 ? K_DRV2 : K_DRV0; else if (!strcmp(k, "push")) o->kind = K_PUSH;
    else if (!strcmp(k, "ub")) o->kind = K_UB; else if (!strcmp(k, "sb")) o->kind = K_SB; else if (!strcmp(k, "sbu")) o->kind = K_SBU; else if (!strcmp(k, "ubp")) o->kind = K_UBP;
    else if (!strcmp(k, "tx")) o->kind = K_TX; else if (!strcmp(k, "tu")) o->kind = K_TU; else if (!strcmp(k, "im")) o->kind = K_IM; else o->kind = K_UNK;
}
static VkDescriptorType bind_dtype(const vbind *b) {
    switch (b->type) { case 'u': return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; case 'b': return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; case 'c': return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                       case 't': return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER; case 'i': return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; default: return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER; }
}
static vshader *shader_get2(const uint8_t key8[8], int quiet) {
    char k[17]; uint64_t kv = 0; for (int i = 0; i < 8; i++) { sprintf(k + 2 * i, "%02x", key8[i]); kv = kv << 8 | key8[i]; }
    uint32_t lo = 0, hi = g_nrefl; while (lo < hi) { uint32_t m = lo + (hi - lo) / 2; if (g_refl[m].key < kv) lo = m + 1; else hi = m; }
    if (lo == g_nrefl || g_refl[lo].key != kv) { static int n; if (!quiet && n++ < 16) LOG("[RED] vk: shader %s is not in the shader pack", k); return NULL; }
    if (g_refl_sh[lo]) return g_refl_sh[lo]->failed ? NULL : g_refl_sh[lo];
    const pack_ent *pe = &g_refl[lo];
    vshader *s = calloc(1, sizeof *s); g_refl_sh[lo] = s; memcpy(s->key, k, 17);
    size_t rec_n = (size_t)pe->row_len + pe->c1_len + pe->spv_len; char *rec = malloc(rec_n + 1);
    if (!rec || !pack_read(rec, rec_n, pe->off)) { LOG("[RED] vk: cannot read shader %s from the shader pack: the copy may have been cut short, copy it again", k); free(rec); s->failed = 1; return NULL; }
    char *row = malloc((size_t)pe->row_len + 1); memcpy(row, rec, pe->row_len); row[pe->row_len] = 0;
    const uint8_t *c1p = (const uint8_t *)rec + pe->row_len, *spvp = c1p + pe->c1_len;
    // split on tabs keeping empty fields
    char *rp = row, *stage = strsep(&rp, "\t"), *set = strsep(&rp, "\t"), *binds = strsep(&rp, "\t"), *io = strsep(&rp, "\t"), *extras = strsep(&rp, "\t");
    s->stage = !stage ? -1 : !strcmp(stage, "vs") ? 0 : !strcmp(stage, "ps") ? 1 : !strcmp(stage, "gs") ? 2 : !strcmp(stage, "hs") ? 3 : !strcmp(stage, "ds") ? 4 : !strcmp(stage, "cs") ? 5 : -1; s->set = set ? atoi(set) : 0;
    int nbx = 0, nix = 0;   // bindings / vertex inputs past the 40 / 32 the tables hold
    if (binds) { char *sv2 = NULL; for (char *t = strtok_r(binds, " ", &sv2); t; t = strtok_r(NULL, " ", &sv2)) { if (s->nb >= 40) { nbx++; continue; } char tk[96]; snprintf(tk, sizeof tk, "%s", t); vk_parse_bind(&s->b[s->nb++], tk); } }
    if (io) { char *sv3 = NULL; for (char *t = strtok_r(io, " ", &sv3); t; t = strtok_r(NULL, " ", &sv3))
        if (t[0] == 'i') { if (s->nin >= 32) { nix++; continue; } int loc = atoi(t + 1); char *c = strchr(t, ':'); if (c) { s->in[s->nin].loc = (uint8_t)loc; s->in[s->nin].ct = c[1]; s->in[s->nin].n = (uint8_t)atoi(c + 2); s->nin++; } } }
    // extras: f1:<binding> marks a combined image sampler (tex_binding_f1); tf<b>:... (transform feedback) is read and ignored
    if (extras) { char *sv4 = NULL; for (char *t = strtok_r(extras, " ", &sv4); t; t = strtok_r(NULL, " ", &sv4))
        if (!strncmp(t, "f1:", 3)) { int bn = atoi(t + 3); for (int i = 0; i < s->nb; i++) if (s->b[i].binding == bn && s->b[i].type == 'c') s->b[i].f1 = 1; } }
    free(row);
    if (nbx || nix) LOG("[RED] vk: shader %s has %d bindings and %d vertex inputs, over the limit: only the first 40 bindings / 32 inputs are used (the layout may miss resources the shader reads)", k, s->nb + nbx, s->nin + nix);   // once per shader (shader_get runs once per shader); the shader is used as before
    uLongf n = pe->spv_raw; void *code = malloc(n ? n : 1);
    if (!code || uncompress(code, &n, spvp, pe->spv_len) != Z_OK || n != pe->spv_raw) { LOG("[RED] vk: shader %s in the shader pack does not decompress: the copy may be damaged, copy it again", k); free(code); free(rec); s->failed = 1; return NULL; }
    if (n < 20 || (n & 3) || *(const uint32_t *)code != 0x07230203u) { LOG("[RED] vk: shader %s is damaged (%zu bytes, first word %#x)", k, (size_t)n, n >= 4 ? *(const uint32_t *)code : 0u); free(code); free(rec); s->failed = 1; return NULL; }   // header is 5 words, magic 0x07230203
    {   // writes the layer (OpDecorate / OpMemberDecorate ... BuiltIn Layer): its draws need a layered pass
        const uint32_t *w = code; size_t nw = n / 4;
        for (size_t i = 5; i < nw; ) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16; if (!wc || i + wc > nw) break;
            if ((op == 71 && wc >= 4 && w[i + 2] == 11 && w[i + 3] == 9) || (op == 72 && wc >= 5 && w[i + 3] == 11 && w[i + 4] == 9)) s->layer_out = 1;
            i += wc; }
    }
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = n, .pCode = code};
    VkResult r = vkCreateShaderModule(g_dev, &mi, NULL, &s->mod); free(code);
    if (r) { LOG("[RED] vk: shader module %s: %d", k, (int)r); free(rec); s->failed = 1; return NULL; }
    VkDescriptorSetLayoutBinding lb[40];
    VkShaderStageFlags sf = s->stage == 0 ? VK_SHADER_STAGE_VERTEX_BIT : s->stage == 1 ? VK_SHADER_STAGE_FRAGMENT_BIT : s->stage == 2 ? VK_SHADER_STAGE_GEOMETRY_BIT : s->stage == 3 ? VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT : s->stage == 4 ? VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT : VK_SHADER_STAGE_COMPUTE_BIT;
    for (int i = 0; i < s->nb; i++) lb[i] = (VkDescriptorSetLayoutBinding){s->b[i].binding, bind_dtype(&s->b[i]), 1, sf, NULL};
    // Vulkan allows one push descriptor set per pipeline layout: the pixel stage (most textures) pushes, the other stages allocate sets from the submission's pool
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = s->stage == 1 ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0, .bindingCount = (uint32_t)s->nb, .pBindings = lb};
    VKCHK(vkCreateDescriptorSetLayout(g_dev, &li, NULL, &s->dsl));
    for (int i = 0; i < s->nb; i++) if (s->b[i].kind == K_C1) {   // the shader's own constant segment (cbuf slot 1): 64 KB declared, only its own bytes bound (Pokopia keeps thousands of shaders alive)
        if (!pe->c1_len) { s->c1buf = g_zero_buf; s->c1off = 0; s->c1size = 65536; s->has_c1 = 1; break; }   // no bytes: the zero buffer (first block of the static arena, never written), not a new block each
        s->c1size = (pe->c1_len + 255ull) & ~255ull; halloc h = static_alloc(s->c1size, 256); g_nc1_blocks++; memset(h.p, 0, s->c1size); memcpy(h.p, c1p, pe->c1_len);
        s->c1buf = h.buf; s->c1off = h.off; s->has_c1 = 1; break;
    }
    free(rec);
    return s;
}
// A program's shaders by stage (cached per program serial). An NVN2 program is keyed at its first use because its code area is not final when it is set (hle_nvn.c), and it may not be final then either: a GPU copy recorded
// just before may not have run, and the key of unfinished code is in no pack. So a stage whose key finds nothing is keyed again later, and only then reported. DM_VK_MISSING=<folder> saves what stays missing, to compare on the PC
typedef struct { uint64_t serial; vshader *sh[6]; uint8_t missing; uint32_t tries; uint64_t t0, next; } vprog;
static struct { uint64_t serial; vprog *p; } *g_ph; static uint32_t g_phcap, g_phn;
void prog_key_forget(const prog_t *p, unsigned st);
int prog_code(const prog_t *p, unsigned st, const uint8_t **ctl, uint32_t *nctl, const uint8_t **code, uint32_t *ncode);
enum { PROG_REPORT_TRIES = 30 };
static void prog_missing_dump(const prog_t *p, unsigned st, const uint8_t k[8]) {
    const char *dir = getenv("DM_VK_MISSING"); const uint8_t *ctl, *code; uint32_t nctl, ncode; if (!dir || !prog_code(p, st, &ctl, &nctl, &code, &ncode)) return;
    mkdir(dir, 0777);
    char path[700]; int n = snprintf(path, sizeof path, "%s/", dir); for (int i = 0; i < 8; i++) n += snprintf(path + n, sizeof path - n, "%02x", k[i]);
    FILE *f = fopen(path, "wb"); if (!f) return;
    uint32_t h[2] = {nctl, ncode}; fwrite(h, 4, 2, f); fwrite(ctl, 1, nctl, f); if (ncode < (1u << 24)) fwrite(code, 1, ncode, f); fclose(f);
}
static void prog_fill(const prog_t *p, vprog *o) {
    int again = o->tries++ > 0; uint8_t missing = 0;
    for (unsigned i = 0; i < p->nstage && i < 6; i++) {
        if (again && !(o->missing >> i & 1)) continue;
        if (again) prog_key_forget(p, i);
        uint8_t k[8]; if (!prog_key(p, i, k)) continue;
        vshader *s = shader_get2(k, o->tries < PROG_REPORT_TRIES);
        if (s && s->stage >= 0) { o->sh[s->stage] = s; if (again) { static int n; if (n++ < 16) LOG("vk: program %llu stage %u: its key found a shader on try %u, %.2f s after its first use", (unsigned long long)p->serial, i, o->tries, (now_ns() - o->t0) / 1e9); } }
        else { missing |= (uint8_t)(1u << i); if (o->tries == PROG_REPORT_TRIES) prog_missing_dump(p, i, k); }
    }
    o->missing = missing; o->next = now_ns() + (o->tries < PROG_REPORT_TRIES ? 100000000ull : 2000000000ull);
}
static vprog *prog_get(const prog_t *p) {
    if (!p || !p->serial) return NULL;
    if (g_phcap) for (uint32_t i = hmix(p->serial) & (g_phcap - 1); g_ph[i].serial; i = (i + 1) & (g_phcap - 1)) if (g_ph[i].serial == p->serial) { vprog *o = g_ph[i].p; if (o->missing && now_ns() >= o->next) prog_fill(p, o); return o; }
    vprog *o = calloc(1, sizeof *o); o->serial = p->serial; o->t0 = now_ns();
    prog_fill(p, o);
    if ((g_phn + 1) * 2 > g_phcap) {
        uint32_t oc = g_phcap; typeof(g_ph) old = g_ph; g_phcap = oc ? 2 * oc : 1024; g_ph = calloc(g_phcap, sizeof *g_ph); g_phn = 0;
        for (uint32_t i = 0; i < oc; i++) if (old[i].serial) { uint32_t j = hmix(old[i].serial) & (g_phcap - 1); while (g_ph[j].serial) j = (j + 1) & (g_phcap - 1); g_ph[j] = old[i]; g_phn++; }
        free(old);
    }
    uint32_t j = hmix(p->serial) & (g_phcap - 1); while (g_ph[j].serial) j = (j + 1) & (g_phcap - 1); g_ph[j].serial = p->serial; g_ph[j].p = o; g_phn++;
    return o;
}
void mtl_prog_release(prog_t *p) { (void)p; }   // program serials are never reused (nvn_new_serial); the cache entry just goes stale

// ---------- State shadowing (draw cost) ----------
// draw() used to record ~35 dynamic-state commands, the pipeline, the vertex input and the vertex buffers for every draw. The command buffer keeps all of them, so draw() now records a value only when
// it differs from the last one recorded here (the shadow D below, per command buffer: shadow_reset() forgets it). DM_VK_SHADOW=0 turns it all off (the recorded stream is then as before);
// DM_VK_SHADOW_MASK picks parts (default 7): 1 draw state (dynamic state, pipeline, vertex input, vertex buffers), 2 reuse of unchanged driver-constant blocks, 4 pipeline lookup by hash,
// 8 re-record the whole draw state after every pipeline change (diagnostic: a driver that drops dynamic state at pipeline binds)
static int g_sh = -1;
static uint64_t g_sh_n[10];   // dynamic state groups recorded / skipped, pipeline binds recorded / skipped, vertex inputs recorded / skipped, vertex buffer binds recorded / skipped, driver blocks reused / built
static int shadow(int bit) {
    if (g_sh < 0) { const char *e = getenv("DM_VK_SHADOW"), *m = getenv("DM_VK_SHADOW_MASK"); g_sh = e && !atoi(e) ? 0 : m ? (int)strtol(m, NULL, 0) & 15 : 7; LOG("vk: state shadowing mask %d (DM_VK_SHADOW=0 turns it off)", g_sh); }
    return g_sh & bit;
}
typedef struct { uint32_t a[4]; } u4; typedef struct { float a[4]; } f4;
static struct {   // what the command buffer holds; ok / vi_ok / vbv (bit per binding) / nbl (leading blend attachments) / pl say which parts are known, k_* are the constant states
    int ok, vi_ok; VkPipeline pl; uint32_t vbv, nvb, nva, nbl;
    VkViewport view; VkRect2D scis; f4 bcol, bias_v; u4 sof, sob;
    uint32_t k_lw, k_pr, k_rd, k_db, cull, front, poly, topo, bias_en, dclamp, dt, dw, dcmp, st, a2c, vmask[2], wmask[2], sref[2];
    VkBool32 be[8]; VkColorBlendEquationEXT eq[8]; VkColorComponentFlags wm[8];
    u4 kb[17], ka[32]; struct { VkBuffer b; VkDeviceSize off, sz; } vb[17];
} D;
// Forgets the shadow: the next draw records everything. Called when the command buffer starts (mtl_sub_begin) and after every vkCmdBeginRendering (rt_begin) and attachment clear, so also after any
// copy / fill / update / barrier (they end the pass first) and anything between passes; call it after binding a graphics pipeline with another dynamic state set, or anything else that may disturb it, into g_cur->cb.
// The present path and the dumps record into command buffers of their own.
static void shadow_reset(void) { memset(&D, 0, sizeof D); }
// Driver-constant blocks (push_stage K_DRV0 / K_DRV2, 4 KB ring allocations): the same inputs within one submission reuse the previous block of that stage and kind
typedef union { struct { uint64_t a[16]; uint32_t s[16]; int32_t binst, pad; } d0; uint32_t d2[168]; } drv_in;
static struct { int valid, cur; VkBuffer buf; uint64_t off; drv_in k; } g_drvc[6][2]; static drv_in g_drvn;
static void drvc_reset(void) { for (int s = 0; s < 6; s++) for (int k = 0; k < 2; k++) g_drvc[s][k].valid = 0; }

// ---------- Pipelines ----------
typedef struct { vshader *vs, *fs, *gs, *hs, *ds; uint32_t cf[8]; uint32_t df, sf; uint8_t nc, samples, topo, dz, patch; } pkey;
typedef struct { pkey k; VkPipeline pl; VkPipelineLayout lay; } pent;
static pent *g_pl; static int g_npl, g_cpl;
static VkPipelineCache g_pcache;
// Pipeline cache on disk (DM_VK_PCACHE, set by the Android front end to <internal>/vk_pcache.bin): pipelines are compiled the first time a draw needs them, which stalled the game
// for 100-230 ms when new ones came into view. The cache is loaded at init (the driver rejects data of another device or driver version by its header) and written back from a
// detached thread every 600 presents when pipelines were created since the last write (tmp file + rename, so a kill mid-write leaves the old file)
static _Atomic int g_pc_new, g_pc_saving;
static void *pcache_save_main(void *a) {
    const char *path = a; size_t n = 0; static size_t last;   // pipelines created from cache hits add nothing: same size, skip the 2-4 MB write
    if (vkGetPipelineCacheData(g_dev, g_pcache, &n, NULL) == VK_SUCCESS && n && n != last) {
        void *d = malloc(n);
        if (d && vkGetPipelineCacheData(g_dev, g_pcache, &n, d) == VK_SUCCESS) {
            char tmp[1100]; snprintf(tmp, sizeof tmp, "%s.tmp", path); FILE *f = fopen(tmp, "wb");
            if (f && fwrite(d, 1, n, f) == n && !fclose(f)) { rename(tmp, path); last = n; LOG("vk: pipeline cache saved (%zu KB)", n / 1024); } else { if (f) fclose(f); LOG("[RED] vk: writing the pipeline cache %s failed", tmp); }
        }
        free(d);
    }
    g_pc_saving = 0; return NULL;
}
static void pcache_maybe_save(void) {
    const char *path = getenv("DM_VK_PCACHE");
    if (!path || !*path || !g_pc_new || g_pc_saving) return;
    g_pc_new = 0; g_pc_saving = 1; pthread_t th;
    if (pthread_create(&th, NULL, pcache_save_main, (void *)path)) g_pc_saving = 0; else pthread_detach(th);
}
static uint64_t g_stat_pso;
static int pkey_eq(const pkey *a, const pkey *b) {   // field by field: the struct has trailing padding
    return a->vs == b->vs && a->fs == b->fs && a->gs == b->gs && a->hs == b->hs && a->ds == b->ds && !memcmp(a->cf, b->cf, sizeof a->cf) && a->df == b->df && a->sf == b->sf && a->nc == b->nc && a->samples == b->samples && a->topo == b->topo && a->dz == b->dz && a->patch == b->patch;
}
static uint32_t pkey_hash(const pkey *k) {
    uint64_t h = (uint64_t)(uintptr_t)k->vs * 0x9e3779b97f4a7c15ull ^ (uint64_t)(uintptr_t)k->fs;
    if (k->gs || k->hs || k->ds) h = hmix(h ^ (uint64_t)(uintptr_t)k->gs ^ (uint64_t)(uintptr_t)k->hs << 1 ^ (uint64_t)(uintptr_t)k->ds << 2 ^ (uint64_t)k->patch << 56);
    for (int i = 0; i < 8; i += 2) h = hmix(h ^ ((uint64_t)k->cf[i] | (uint64_t)k->cf[i + 1] << 32));
    h = hmix(h ^ ((uint64_t)k->df | (uint64_t)k->sf << 32));
    return hmix(h ^ ((uint64_t)k->nc | (uint64_t)k->samples << 8 | (uint64_t)k->topo << 16 | (uint64_t)k->dz << 24));
}
static int32_t *g_plh; static uint32_t g_plhcap; static int g_pl_last = -1;   // open-addressed table of g_pl indices + 1 (g_pl is realloc'd: never pointers), the last pipeline returned
static void plh_put(int idx) { uint32_t i = pkey_hash(&g_pl[idx].k) & (g_plhcap - 1); while (g_plh[i]) i = (i + 1) & (g_plhcap - 1); g_plh[i] = idx + 1; }
static void plh_add(int idx) {   // g_pl[idx] is stored
    if ((uint32_t)(idx + 1) * 2 > g_plhcap) { free(g_plh); g_plhcap = g_plhcap ? 2 * g_plhcap : 1024; g_plh = calloc(g_plhcap, sizeof *g_plh); for (int i = 0; i <= idx; i++) plh_put(i); }
    else plh_put(idx);
}
static pent *pipeline_get(const pkey *k) {
    if (shadow(4)) {   // pipelines whose creation failed (pl 0) are in the table too
        if (g_pl_last >= 0 && pkey_eq(&g_pl[g_pl_last].k, k)) return &g_pl[g_pl_last];
        if (g_plhcap) for (uint32_t i = pkey_hash(k) & (g_plhcap - 1); g_plh[i]; i = (i + 1) & (g_plhcap - 1)) if (pkey_eq(&g_pl[g_plh[i] - 1].k, k)) return &g_pl[g_pl_last = g_plh[i] - 1];
    } else for (int i = 0; i < g_npl; i++) if (!memcmp(&g_pl[i].k, k, sizeof *k)) return &g_pl[i];
    static VkDescriptorSetLayout empty; if (!empty) { VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; VKCHK(vkCreateDescriptorSetLayout(g_dev, &li, NULL, &empty)); }
    VkDescriptorSetLayout sl[5] = {k->vs->dsl, k->fs ? k->fs->dsl : empty, k->gs ? k->gs->dsl : empty, k->hs ? k->hs->dsl : empty, k->ds ? k->ds->dsl : empty};
    VkPipelineLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = k->hs || k->ds ? 5 : k->gs ? 3 : 2, .pSetLayouts = sl};
    VkPipelineLayout lay; VKCHK(vkCreatePipelineLayout(g_dev, &lci, NULL, &lay));
    VkPipelineShaderStageCreateInfo st[5]; uint32_t nst = 0;
#define PSTAGE(sh_, bit_) do { if (sh_) st[nst++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = bit_, .module = (sh_)->mod, .pName = "main"}; } while (0)
    PSTAGE(k->vs, VK_SHADER_STAGE_VERTEX_BIT); PSTAGE(k->hs, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT); PSTAGE(k->ds, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT); PSTAGE(k->gs, VK_SHADER_STAGE_GEOMETRY_BIT); PSTAGE(k->fs, VK_SHADER_STAGE_FRAGMENT_BIT);
#undef PSTAGE
    static const VkPrimitiveTopology topo[4] = {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = topo[k->topo]};
    // The winding GLSLC recorded is the OpenGL one, relative to a domain with its origin at the lower left (inference: not yet seen on the device; DM_VK_TESS_UL=1 is Vulkan's default, for a test)
    static int tess_ul = -1; if (tess_ul < 0) { const char *e = getenv("DM_VK_TESS_UL"); tess_ul = e && atoi(e); }
    VkPipelineTessellationDomainOriginStateCreateInfo tdo = {VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_DOMAIN_ORIGIN_STATE_CREATE_INFO, .domainOrigin = tess_ul ? VK_TESSELLATION_DOMAIN_ORIGIN_UPPER_LEFT : VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT};
    VkPipelineTessellationStateCreateInfo tess = {VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO, &tdo, .patchControlPoints = k->patch};
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, .negativeOneToOne = !k->dz};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, &dcc, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = (VkSampleCountFlagBits)k->samples};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8] = {{0}}; for (int i = 0; i < 8; i++) cba[i].colorWriteMask = 0xf;
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = k->nc, .pAttachments = cba};
    static const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE, VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
        VK_DYNAMIC_STATE_VERTEX_INPUT_EXT, VK_DYNAMIC_STATE_POLYGON_MODE_EXT, VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT, VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
        VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT, VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT, VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = sizeof dyn / sizeof *dyn, .pDynamicStates = dyn};
    VkFormat cf[8]; for (int i = 0; i < 8; i++) cf[i] = (VkFormat)k->cf[i];
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = k->nc, .pColorAttachmentFormats = cf, .depthAttachmentFormat = (VkFormat)k->df, .stencilAttachmentFormat = (VkFormat)k->sf};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, .stageCount = nst, .pStages = st, .pInputAssemblyState = &ia, .pTessellationState = k->hs ? &tess : NULL,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb, .pDynamicState = &dy, .layout = lay};
    VkPipeline pl; uint64_t t0 = now_ns();
    VkResult r = vkCreateGraphicsPipelines(g_dev, g_pcache, 1, &gi, NULL, &pl); g_pc_new++;
    if (r) { LOG("[RED] vk: pipeline vs %s fs %s gs %s hs %s ds %s: %d", k->vs->key, k->fs ? k->fs->key : "-", k->gs ? k->gs->key : "-", k->hs ? k->hs->key : "-", k->ds ? k->ds->key : "-", (int)r); pl = VK_NULL_HANDLE; }
    g_stat_pso++; { uint64_t ms_ = (now_ns() - t0) / 1000000; if (ms_ > 200) LOG("vk: pipeline vs %s fs %s took %llu ms", k->vs->key, k->fs ? k->fs->key : "-", (unsigned long long)ms_); }
    if (g_npl == g_cpl) { g_cpl = g_cpl ? 2 * g_cpl : 512; g_pl = realloc(g_pl, g_cpl * sizeof *g_pl); }
    g_pl[g_npl] = (pent){*k, pl, lay}; plh_add(g_npl); g_pl_last = g_npl;
    return &g_pl[g_npl++];
}

// ---------- Submissions ----------
struct ctx { VkCommandPool pool; VkCommandBuffer cb, up; uint64_t fence; int up_used; post_op *posts; int npost, cpost; VkDescriptorPool dp[64]; int ndp, curdp;
             VkCommandBuffer xcb[32], xup[32]; int nx, ichunk; };   // command buffer pairs for the chunks of one submission (sub_flush); cb / up = the current chunk
// How many submissions can be in flight: a new one waits for the one its context last ran. Some games submit about once a frame, Pokopia about 11 times, so with 4 the replay could not get ahead of the GPU:
// the render thread sat 33 ms a frame in that wait and the GPU idled a third of the time. Measured at the same spot, 4 / 16: frame 48 to 61 / 25 to 30 ms, GPU busy 68 / 91%, GPU clock and temperature not held equal
// (the runs at 16 got hotter), so the frame times are not a sustained rate. Costs about one more frame of latency, as a submission waits longer to reach the GPU. DM_VK_NCTX: 1 to NCTX
enum { NCTX = 32 };
static ctx g_ctx[NCTX]; static uint64_t g_nsub; static int g_nctx = 16;
static VkCommandBuffer cur_up(void) { g_cur->up_used = 1; return g_cur->up; }
static VkDescriptorPool dpool_new(void) {
    // A quarter of the first size: every context keeps at least one, and a large submission takes more (a pool held about 2 MB of system memory)
    VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2048},
                                 {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 64}};
    VkDescriptorPoolCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1024, .poolSizeCount = sizeof ps / sizeof *ps, .pPoolSizes = ps};
    VkDescriptorPool p; VKCHK(vkCreateDescriptorPool(g_dev, &ci, NULL, &p)); return p;
}
static VkDescriptorSet dset_alloc(VkDescriptorSetLayout l) {   // from the current submission's pools (reset when its context is reused)
    ctx *c = g_cur;
    for (;;) {
        if (c->curdp == c->ndp) { if (c->ndp == 64) die("vk: more than 64 descriptor pools in one submission"); c->dp[c->ndp++] = dpool_new(); }
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = c->dp[c->curdp], .descriptorSetCount = 1, .pSetLayouts = &l};
        VkDescriptorSet s; VkResult r = vkAllocateDescriptorSets(g_dev, &ai, &s);
        if (!r) return s;
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) die("vk: descriptor set allocation failed: %d", (int)r);
        c->curdp++;
    }
}
static void post(post_op o) { ctx *c = g_cur; if (c->npost == c->cpost) { c->cpost = c->cpost ? 2 * c->cpost : 64; c->posts = realloc(c->posts, c->cpost * sizeof *c->posts); } c->posts[c->npost++] = o; }
// Barriers between passes / copies / dispatches (default 1). DM_VK_BARRIER=0: all of them are the full ALL_COMMANDS -> ALL_COMMANDS memory barrier of barrier_full (on AMD a full drain plus cache flush /
// invalidate each). 1: barrier_k, a barrier per kind of producer (pass end, compute dispatch, transfer) whose source is that producer's stages and write accesses; 2: as 1, and a pass end assumes no
// vertex / fragment shader writes a storage buffer (SHADER_WRITE left out of its source access; unverified for other games' shaders). The destination of all of them is everything this file records later
// (any later command can read or write what an earlier one wrote: the next pass, a copy, a dispatch, the next submission, the present composite), so the consumers are not narrowed: only the producers
enum { BK_XFER, BK_PASS, BK_COMPUTE };
static int g_barmode = -1;
static void barrier_full(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    pvkCmdPipelineBarrier2(cb, &di);
}
static void barrier_k(VkCommandBuffer cb, int kind) {
    if (g_barmode < 0) { const char *e = getenv("DM_VK_BARRIER"); g_barmode = e ? atoi(e) : 1; LOG("vk: DM_VK_BARRIER=%d (0 full barriers, 1 per-producer barriers, 2 also no shader writes at pass ends)", g_barmode); }
    if (!g_barmode) { barrier_full(cb); return; }
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = kind == BK_PASS ? VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT : kind == BK_COMPUTE ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
        .srcAccessMask = kind == BK_PASS ? (VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | (g_barmode == 1 ? VK_ACCESS_2_SHADER_WRITE_BIT : 0)) :   // clears in a pass are attachment writes
                         kind == BK_COMPUTE ? VK_ACCESS_2_SHADER_WRITE_BIT : VK_ACCESS_2_TRANSFER_WRITE_BIT,   // a dispatch writes storage buffers and images; every image stays in GENERAL, so this global barrier covers them
        .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                         VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                         VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    pvkCmdPipelineBarrier2(cb, &di);
}
// The call sites other than rp_end / dispatch follow copies, clears, fills, updates and the upload command buffer: transfer commands
static void barrier_all(VkCommandBuffer cb) { barrier_k(cb, BK_XFER); }

// Replay state (as nvn_mtl.m's S / R)
#define NTEX 256
static struct {
    const prog_t *bst[6]; blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    uint32_t attr[16]; int nattr; vs_st stream[16]; int nstream; NVNbufferRange vbuf[16];
    p_bind_buf ubo[6][16], ssbo[6][16]; uint64_t tex[6][NTEX], samp[6][128], img[6][32];
    uint32_t st_vmask[2], st_wmask[2], st_ref[2]; p_poly_offset po; float vp[16][4]; int has_vp; float dr[2]; int depth_clamp; float blend_color[4];
    int64_t copy_stride, copy_img_stride; int rast_discard; p_scissor sc; int has_sc;
    uint8_t vp_sw[16][4]; p_prim_restart restart;   // OP_VP_SWIZZLE (per viewport, XOR the identity: 0 = unchanged) and OP_PRIM_RESTART
    uint8_t push[2][DMSH_PUSH_SIZE]; int patch;   // OP_PUSH_CONST: NVN2 push constants of the graphics stages and of compute, bound as c[24]; OP_PATCH_SIZE
} S;
static struct { p_set_rt rt; int has, active; uint32_t w, h; uint32_t cf[8]; uint32_t df, sf; int nc; uint8_t samples; int layered; } R;
static int g_rp_layered;   // the pass about to open is for a vertex shader that writes the layer (as in nvn_mtl.m: a colour grading LUT can be built so)
static int g_rp_empty;     // the pass about to open may have no attachments: a draw under rasterizer discard, for its vertex stage's stores

// ---------- Pass profiler: DM_VK_PASSPROF=1 (=2 also logs every frame), DM_VK_PASSPROF_MAX=n marks per submission (default 512) ----------
// Where a frame's GPU time goes, per render pass. One timestamp ("mark") is written at each boundary, with ALL_COMMANDS (vkCmdWriteTimestamp2; BOTTOM_OF_PIPE if it is missing): it is taken once everything recorded
// before it has finished. The interval from mark i to mark i+1 takes the kind of mark i, so the intervals tile the submission's GPU time with no holes. Marks go into whatever g_cur->cb is current (sub_flush
// swaps it), in recording order = execution order: a chunk's upload buffer runs before its command buffer, so its mark is written when the chunk begins (pp_chunk_begin), never when it ends.
//   PK_PASS    mark just before vkCmdBeginRendering (outside the instance, so the pass is not split) -> the mark rp_end writes right after vkCmdEndRendering: the pass's own time, its draws
//   PK_GAP     that end-of-pass mark, and the one before the barrier_all that follows a copy / clear / fill / dispatch: the barrier (drain + cache flush) and whatever runs up to the next mark show as gap time
//   PK_COPY / PK_CLEAR / PK_DISP: a mark right before the copy / update / clear / fill / dispatch command (dispatch: right after its rp_end, so its bind, indirect-count copy and small barrier count as dispatch)
//   PK_UPLOAD  start of a chunk's upload buffer (texture uploads, image init) -> start of the chunk's command buffer;  PK_FLUSH  end of a chunk's command buffer (sub_flush) -> start of the next chunk's upload
//   buffer: the GPU is idle there when it outruns the replay, which the existing per-submission pair counts as busy time;  PK_END the last mark (one slot is always kept for it);  PK_TAIL the last interval
//   of a submission that ran out of marks (not attributed to anything)
// The records (pass description from rt_begin, draw / clear counts from draw / clear_*) are copied into the submission's pend at commit and read with the timestamps on the completion thread
// (passprof_done, from gpu_time_done after the fence): per frame (the submissions of one present), and over PP_WIN presents with the top passes by average GPU time. Off: one predictable branch per site
enum { PP_WIN = 300, PP_PASSES = 512, PP_TOP = 15, PP_TOPGAP = 8 };   // presents per report, pass slots (index of the pass in its frame), passes / gaps listed
static struct { int on, lvl, cap, q, n, want, pass, sub; uint64_t mask; PFN_vkCmdWriteTimestamp2 stamp2; VkQueryPool qp[NCTX]; pp_rec *rec[NCTX]; uint64_t *ts, *dur; } g_pp;   // submit thread: ctx q, marks n of cap, refused marks want, record of the open pass (-1: none)
static void pp_init(uint32_t valid_bits) {   // dev_init, after the g_tsq pair
    const char *e = getenv("DM_VK_PASSPROF"), *m = getenv("DM_VK_PASSPROF_MAX"); int lvl = e ? atoi(e) : 0, cap = m ? atoi(m) : 512, ok = 1;
    if (lvl <= 0) return;
    if (!g_tsq || !valid_bits) { LOG("[RED] vk: passprof off: no GPU timestamps (DM_VK_NO_GPU_TIME, or the queue has no timestamp bits)"); return; }
    cap = MIN(MAX(cap, 16), 16384);
    for (int i = 0; i < g_nctx && ok; i++) {
        VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = (uint32_t)cap};
        g_pp.rec[i] = calloc((size_t)cap, sizeof *g_pp.rec[i]); ok = g_pp.rec[i] && vkCreateQueryPool(g_dev, &qi, NULL, &g_pp.qp[i]) == VK_SUCCESS;
    }
    g_pp.ts = malloc((size_t)cap * sizeof *g_pp.ts); g_pp.dur = malloc((size_t)cap * sizeof *g_pp.dur);
    if (!ok || !g_pp.ts || !g_pp.dur) { LOG("[RED] vk: passprof off: no memory for %d marks per submission", cap); return; }
    g_pp.stamp2 = (PFN_vkCmdWriteTimestamp2)vkGetDeviceProcAddr(g_dev, "vkCmdWriteTimestamp2");   // not in VK_FNS (that loader dies on a miss); the sync2 feature is required anyway, so it is there
    g_pp.mask = valid_bits >= 64 ? ~0ull : (1ull << valid_bits) - 1; g_pp.cap = cap; g_pp.lvl = lvl; g_pp.on = 1;
    { const char *sb = getenv("DM_VK_PASSPROF_SUB"); g_pp.sub = sb ? atoi(sb) : 0; if (g_pp.sub > 0) LOG("vk: passprof: DM_VK_PASSPROF_SUB=%d: a mark after every %d draws inside a pass (locates a stalled draw; the report then counts each segment as a pass)", g_pp.sub, g_pp.sub); }
    LOG("vk: passprof on (DM_VK_PASSPROF=%d): %d marks per submission, %s timestamps, %.2f ns per tick, %u valid bits, a report every %d presents", lvl, cap, g_pp.stamp2 ? "ALL_COMMANDS (sync2)" : "BOTTOM_OF_PIPE", g_ts_ns, valid_bits, (int)PP_WIN);
}
static void pp_stamp(VkCommandBuffer cb, uint32_t i) {
    if (g_pp.stamp2) g_pp.stamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, g_pp.qp[g_pp.q], i);
    else vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_pp.qp[g_pp.q], i);
}
static int pp_mark(VkCommandBuffer cb, int kind) {   // a mark of the given kind into cb; its record index, or -1 when the submission is out of marks (counted: the report says how many it needed)
    if (g_pp.n >= g_pp.cap - 1) { g_pp.want++; return -1; }   // the last slot is kept for pp_end
    int i = g_pp.n++; pp_rec *r = &g_pp.rec[g_pp.q][i]; memset(r, 0, sizeof *r); r->kind = (uint8_t)kind; pp_stamp(cb, (uint32_t)i); return i;
}
#define PP(kind) do { if (g_pp.on) pp_mark(g_cur->cb, kind); } while (0)
#define PP_CLEAR() do { if (g_pp.on && g_pp.pass >= 0) g_pp.rec[g_pp.q][g_pp.pass].clears++; } while (0)
static void pp_chunk_begin(ctx *c, int first) {   // mtl_sub_begin (first: also resets the pool and the record state) and sub_flush: c->cb / c->up are the fresh chunk's, begun
    if (first) { g_pp.q = (int)(c - g_ctx); g_pp.n = g_pp.want = 0; g_pp.pass = -1; vkCmdResetQueryPool(c->up, g_pp.qp[g_pp.q], 0, (uint32_t)g_pp.cap); }
    pp_mark(c->up, PK_UPLOAD); pp_mark(c->cb, PK_GAP);   // up runs first: its interval is the uploads; the command buffer's mark covers its start up to the first pass / op
}
static void pp_pass_begin(void) {   // rt_begin: R describes the pass about to begin
    int i = pp_mark(g_cur->cb, PK_PASS); g_pp.pass = i;
    if (i < 0) return;
    pp_rec *r = &g_pp.rec[g_pp.q][i]; r->w = R.w; r->h = R.h; r->nc = (uint8_t)R.nc; r->samples = R.samples; r->layered = (uint8_t)R.layered; r->df = R.df ? R.df : R.sf;
    for (int k = 0; k < 4; k++) r->cf[k] = R.cf[k];
}
static void pp_pass_end(void) { pp_mark(g_cur->cb, PK_GAP); g_pp.pass = -1; }   // rp_end, between vkCmdEndRendering and barrier_all: the barrier is gap time, not pass time
static void pp_pass_draw(const vshader *vs, const vshader *fs) {   // draw(), where a draw is recorded: the pass's draw count, and the shader keys of its first draw
    if (g_pp.pass < 0) return;
    pp_rec *r = &g_pp.rec[g_pp.q][g_pp.pass];
    if (!r->draws++) { memcpy(r->vs, vs->key, sizeof r->vs); memcpy(r->fs, fs ? fs->key : "-", fs ? sizeof r->fs : 2); }
    if (g_pp.sub > 0 && r->draws >= (uint32_t)g_pp.sub) {   // DM_VK_PASSPROF_SUB: the pass continues in a new segment (same description, d0 = its first draw's index in the pass)
        int i = pp_mark(g_cur->cb, PK_PASS); if (i < 0) return;
        pp_rec *o = &g_pp.rec[g_pp.q][g_pp.pass], *n = &g_pp.rec[g_pp.q][i];
        n->w = o->w; n->h = o->h; n->nc = o->nc; n->samples = o->samples; n->layered = o->layered; n->df = o->df; memcpy(n->cf, o->cf, sizeof n->cf); n->d0 = o->d0 + o->draws;
        g_pp.pass = i;
    }
}
static void pp_end(void) {   // mtl_sub_commit, after the last rp_end and the upload barrier: the END mark. Refused marks mean the interval before it spans the rest of the submission: unattributed
    int i = g_pp.n; pp_rec *r = g_pp.rec[g_pp.q];
    if (g_pp.want && i > 0) r[i - 1].kind = PK_TAIL;
    memset(&r[i], 0, sizeof r[i]); r[i].kind = PK_END; pp_stamp(g_cur->cb, (uint32_t)i); g_pp.n = i + 1; g_pp.pass = -1;
}
// Aggregation (completion thread only, no locks): the frame in progress, and the window
typedef struct { uint64_t n, ns, max, gap, draws, clears, sig, chg; pp_rec last; } pp_slot;   // one pass index of the frame: occurrences, GPU ns total / worst, ns of the gap before it, draws, clears, layout signature and its changes, latest record
static struct { int open; uint64_t frame, ord, draws, clears, ns[PK_N], cnt[PK_N]; } g_ppf;
static struct { uint64_t start, frames, subs, span_max, trunc, maxn, maxneed, bad, nfail, chg, draws, clears, ref0, ns[PK_N], cnt[PK_N]; pp_slot slot[PP_PASSES]; } g_ppw;
static const char *pp_fmt(uint32_t f, char *b) {   // short names for the formats the game's targets use; "f<n>" (the VkFormat number) otherwise
    switch (f) {
        case VK_FORMAT_R8G8B8A8_UNORM: return "RGBA8"; case VK_FORMAT_R8G8B8A8_SRGB: return "RGBA8s"; case VK_FORMAT_B8G8R8A8_UNORM: return "BGRA8"; case VK_FORMAT_B8G8R8A8_SRGB: return "BGRA8s";
        case VK_FORMAT_R16G16B16A16_SFLOAT: return "RGBA16F"; case VK_FORMAT_R32G32B32A32_SFLOAT: return "RGBA32F"; case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return "R11G11B10F";
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return "RGB10A2"; case VK_FORMAT_R8_UNORM: return "R8"; case VK_FORMAT_R8G8_UNORM: return "RG8"; case VK_FORMAT_R16_SFLOAT: return "R16F";
        case VK_FORMAT_R16G16_SFLOAT: return "RG16F"; case VK_FORMAT_R32_SFLOAT: return "R32F"; case VK_FORMAT_R32G32_SFLOAT: return "RG32F"; case VK_FORMAT_R32_UINT: return "R32U";
        case VK_FORMAT_D32_SFLOAT: return "D32"; case VK_FORMAT_D32_SFLOAT_S8_UINT: return "D32S8"; case VK_FORMAT_D16_UNORM: return "D16"; case VK_FORMAT_S8_UINT: return "S8";
    }
    snprintf(b, 12, "f%u", f); return b;
}
static void pp_desc(const pp_rec *r, char *o, size_t n) {   // "1280x720 1x c[RGBA16F,R11G11B10F] d[D32S8]"
    char t[12]; size_t k = (size_t)snprintf(o, n, "%ux%u%s %ux c[", r->w, r->h, r->layered ? " layered" : "", r->samples);
    for (int i = 0; i < r->nc && i < 4 && k < n; i++) k += (size_t)snprintf(o + k, n - k, "%s%s", i ? "," : "", r->cf[i] ? pp_fmt(r->cf[i], t) : "-");
    if (r->nc > 4 && k < n) k += (size_t)snprintf(o + k, n - k, "+%d", r->nc - 4);
    if (k < n) snprintf(o + k, n - k, "] d[%s]", r->df ? pp_fmt(r->df, t) : "-");
}
static uint64_t pp_sig(const pp_rec *r) {   // what makes two frames' pass N "the same pass": size, samples, formats (not the shaders, which vary with the scene)
    uint32_t v[9] = {r->w, r->h, r->nc, r->samples, r->df, r->cf[0], r->cf[1], r->cf[2], r->cf[3]}; uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 9; i++) h = (h ^ v[i]) * 1099511628211ull;
    return h;
}
static int pp_cmp_avg(const void *a, const void *b) { const pp_slot *x = &g_ppw.slot[*(const int *)a], *y = &g_ppw.slot[*(const int *)b]; double u = (double)x->ns / x->n, v = (double)y->ns / y->n; return u < v ? 1 : u > v ? -1 : 0; }
static int pp_cmp_gap(const void *a, const void *b) { const pp_slot *x = &g_ppw.slot[*(const int *)a], *y = &g_ppw.slot[*(const int *)b]; double u = (double)x->gap / x->n, v = (double)y->gap / y->n; return u < v ? 1 : u > v ? -1 : 0; }
#define PPMS(a, k) ((a).ns[k] / 1e6)
static void pp_frame_close(void) {   // the frame in g_ppf has all its submissions: into the window
    uint64_t span = 0;
    for (int k = 0; k < PK_N; k++) { span += g_ppf.ns[k]; g_ppw.ns[k] += g_ppf.ns[k]; g_ppw.cnt[k] += g_ppf.cnt[k]; }
    g_ppw.frames++; g_ppw.draws += g_ppf.draws; g_ppw.clears += g_ppf.clears; if (span > g_ppw.span_max) g_ppw.span_max = span;
    if (g_pp.lvl >= 2) LOG("vk: passprof frame %llu: GPU %.2f ms = passes %.2f (%llu, %llu draws) + gaps %.2f + upload %.2f + flush-idle %.2f + copy %.2f (%llu) + clear/fill %.2f (%llu) + dispatch %.2f (%llu) + untracked tail %.2f",
        (unsigned long long)g_ppf.frame, span / 1e6, PPMS(g_ppf, PK_PASS), (unsigned long long)g_ppf.cnt[PK_PASS], (unsigned long long)g_ppf.draws, PPMS(g_ppf, PK_GAP), PPMS(g_ppf, PK_UPLOAD), PPMS(g_ppf, PK_FLUSH),
        PPMS(g_ppf, PK_COPY), (unsigned long long)g_ppf.cnt[PK_COPY], PPMS(g_ppf, PK_CLEAR), (unsigned long long)g_ppf.cnt[PK_CLEAR], PPMS(g_ppf, PK_DISP), (unsigned long long)g_ppf.cnt[PK_DISP], PPMS(g_ppf, PK_TAIL));
}
static void pp_shader_report(double F);   // below
static void pp_report(uint64_t next) {   // the window is complete: log it, start the next one at present `next`
    const double F = g_ppw.frames ? (double)g_ppw.frames : 1.0; uint64_t span = 0;
    for (int k = 0; k < PK_N; k++) span += g_ppw.ns[k];
    if (g_ppw.frames) {
        double ref = ((double)atomic_load(&g_gpu_ns_all) - (double)g_ppw.ref0) / 1e6 / F;   // the existing pair: same submissions up to one frame at the window edges; it counts flush-idle time as busy
        LOG("vk: passprof window presents %llu..%llu: %llu frames, %llu submissions: GPU %.2f ms/frame (worst frame %.2f), passes %.2f ms (%.1f%%), gaps %.2f ms (%.1f%%), other %.2f ms (%.1f%%); existing timestamp pair %.2f ms/frame",
            (unsigned long long)g_ppw.start, (unsigned long long)next - 1, (unsigned long long)g_ppw.frames, (unsigned long long)g_ppw.subs, span / 1e6 / F, g_ppw.span_max / 1e6, PPMS(g_ppw, PK_PASS) / F, 100.0 * g_ppw.ns[PK_PASS] / (span ? span : 1),
            PPMS(g_ppw, PK_GAP) / F, 100.0 * g_ppw.ns[PK_GAP] / (span ? span : 1), (span - g_ppw.ns[PK_PASS] - g_ppw.ns[PK_GAP]) / 1e6 / F, 100.0 * (span - g_ppw.ns[PK_PASS] - g_ppw.ns[PK_GAP]) / (span ? span : 1), ref);
        LOG("vk: passprof   per frame: %.1f passes, %.0f draws, %.1f clears in passes; gaps %.1f; ms: upload %.2f, flush-idle %.2f, copy %.2f (%.1f ops), clear/fill %.2f (%.1f), dispatch %.2f (%.1f), untracked tail %.2f",
            g_ppw.cnt[PK_PASS] / F, g_ppw.draws / F, g_ppw.clears / F, g_ppw.cnt[PK_GAP] / F, PPMS(g_ppw, PK_UPLOAD) / F, PPMS(g_ppw, PK_FLUSH) / F, PPMS(g_ppw, PK_COPY) / F, g_ppw.cnt[PK_COPY] / F,
            PPMS(g_ppw, PK_CLEAR) / F, g_ppw.cnt[PK_CLEAR] / F, PPMS(g_ppw, PK_DISP) / F, g_ppw.cnt[PK_DISP] / F, PPMS(g_ppw, PK_TAIL) / F);
        LOG("vk: passprof   marks: most used %llu of %d per submission; %llu submissions ran out (needed up to %llu: DM_VK_PASSPROF_MAX), %llu skipped for marks out of order, %llu failed readbacks, %llu pass layout changes between frames",
            (unsigned long long)g_ppw.maxn, g_pp.cap, (unsigned long long)g_ppw.trunc, (unsigned long long)g_ppw.maxneed, (unsigned long long)g_ppw.bad, (unsigned long long)g_ppw.nfail, (unsigned long long)g_ppw.chg);
        int idx[PP_PASSES], nu = 0;
        for (int i = 0; i < PP_PASSES; i++) if (g_ppw.slot[i].n) idx[nu++] = i;
        qsort(idx, (size_t)nu, sizeof *idx, pp_cmp_avg);
        static int ntop = -1; if (ntop < 0) { const char *e = getenv("DM_VK_PASSPROF_TOP"); ntop = e ? atoi(e) : PP_TOP; }
        for (int j = 0; j < nu && j < ntop; j++) {
            const pp_slot *s = &g_ppw.slot[idx[j]]; char d[160], chg[40] = ""; pp_desc(&s->last, d, sizeof d);
            if (s->chg) snprintf(chg, sizeof chg, " [layout changed %llux]", (unsigned long long)s->chg);
            LOG("vk: passprof   top %2d: pass %3d  %.3f ms avg (worst %.3f, in %llu of %llu frames), gap before %.3f ms, %.1f draws %.1f clears, %s, first draw vs %s fs %s%s", j + 1, idx[j], s->ns / 1e6 / s->n, s->max / 1e6,
                (unsigned long long)s->n, (unsigned long long)g_ppw.frames, s->gap / 1e6 / s->n, (double)s->draws / s->n, (double)s->clears / s->n, d, s->last.vs[0] ? s->last.vs : "-", s->last.fs[0] ? s->last.fs : "-", chg);
        }
        nu = 0; for (int i = 0; i < PP_PASSES; i++) if (g_ppw.slot[i].n && g_ppw.slot[i].gap) idx[nu++] = i;
        qsort(idx, (size_t)nu, sizeof *idx, pp_cmp_gap);
        for (int j = 0; j < nu && j < PP_TOPGAP; j++) {
            const pp_slot *s = &g_ppw.slot[idx[j]]; char d[160]; pp_desc(&s->last, d, sizeof d);
            LOG("vk: passprof   gap %d: before pass %3d: %.3f ms avg (barrier / drain after the previous pass or op, and what ran in between), the pass is %s", j + 1, idx[j], s->gap / 1e6 / s->n, d);
        }
    }
    pp_shader_report(F);
    memset(&g_ppw, 0, sizeof g_ppw); g_ppw.start = next; g_ppw.ref0 = atomic_load(&g_gpu_ns_all);
}
// Per shader pair: pass time by the vs / fs of a pass's first draw and the target size, over the window (pass slots move between frames when draw counts change; shaders do not).
// Exact per draw with DM_VK_PASSPROF_SUB=1 (every draw its own segment, at the cost of a drain per draw); with whole passes the pass is charged to its first draw's shaders
enum { PP_SH = 1024, PP_SHTOP = 20 };
typedef struct { char vs[20], fs[20]; uint32_t w, h; uint64_t ns, draws, n; pp_rec last; } pp_sh;
static pp_sh g_ppsh[PP_SH]; static int g_ppsh_n; static uint64_t g_ppsh_full;
static void pp_shader_add(const pp_rec *r, uint64_t ns) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 20 && r->vs[i]; i++) h = (h ^ (uint8_t)r->vs[i]) * 1099511628211ull;
    for (int i = 0; i < 20 && r->fs[i]; i++) h = (h ^ (uint8_t)r->fs[i]) * 1099511628211ull;
    h = (h ^ r->w) * 1099511628211ull; h = (h ^ r->h) * 1099511628211ull;
    for (unsigned i = 0, b = (unsigned)(h % PP_SH); i < PP_SH; i++, b = (b + 1) % PP_SH) {
        pp_sh *e = &g_ppsh[b];
        if (!e->n) { if (g_ppsh_n >= PP_SH * 3 / 4) { g_ppsh_full++; return; } memcpy(e->vs, r->vs, 20); memcpy(e->fs, r->fs, 20); e->w = r->w; e->h = r->h; g_ppsh_n++; }
        else if (e->w != r->w || e->h != r->h || memcmp(e->vs, r->vs, 20) || memcmp(e->fs, r->fs, 20)) continue;
        e->ns += ns; e->draws += r->draws; e->n++; e->last = *r; return;
    }
}
static int pp_cmp_sh(const void *a, const void *b) { uint64_t u = g_ppsh[*(const int *)a].ns, v = g_ppsh[*(const int *)b].ns; return u < v ? 1 : u > v ? -1 : 0; }
static void pp_shader_report(double F) {   // pp_report: the top shader pairs of the window, then a fresh table
    int ix[PP_SH], m = 0; uint64_t tot = 0;
    for (int i = 0; i < PP_SH; i++) if (g_ppsh[i].n) { ix[m++] = i; tot += g_ppsh[i].ns; }
    qsort(ix, (size_t)m, sizeof *ix, pp_cmp_sh);
    for (int j = 0; j < m && j < PP_SHTOP; j++) { const pp_sh *e = &g_ppsh[ix[j]]; char d[160]; pp_desc(&e->last, d, sizeof d);
        LOG("vk: passprof   shader %2d: vs %.16s fs %.16s  %.3f ms/frame (%.1f%% of pass time), %.1f draws in %.1f segments per frame, %s", j + 1, e->vs[0] ? e->vs : "-", e->fs[0] ? e->fs : "-",
            e->ns / 1e6 / F, 100.0 * e->ns / (tot ? tot : 1), e->draws / F, e->n / F, d); }
    if (g_ppsh_full) LOG("[RED] vk: passprof: shader table full, %llu segments not counted", (unsigned long long)g_ppsh_full);
    memset(g_ppsh, 0, sizeof g_ppsh); g_ppsh_n = 0; g_ppsh_full = 0;
}
static void passprof_done(const pend *p) {   // completion thread, after p's fence: marks 0..n-1 are all written; interval i = ts[i+1] - ts[i] belongs to rec[i].kind
    int n = p->ppn; const pp_rec *r = p->ppr; uint64_t *ts = g_pp.ts, *dur = g_pp.dur;
    if (!g_pp.on || n < 2 || n > g_pp.cap || p->q < 0) return;
    static const char *kn[PK_N] = {"upload", "gap", "pass", "copy", "clear", "dispatch", "flush", "tail", "end"};
    if (vkGetQueryPoolResults(g_dev, g_pp.qp[p->q], 0, (uint32_t)n, (size_t)n * 8, ts, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) {
        RED_ONCE(45, "vk: passprof: reading the marks of a submission failed"); g_ppw.nfail++;
        if (p->late) {   // a stalled submission: find the first mark the GPU never wrote
            int last = -1; for (int i = 0; i < n; i++) { uint64_t v[2] = {0, 0}; if (vkGetQueryPoolResults(g_dev, g_pp.qp[p->q], (uint32_t)i, 1, 16, v, 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) >= 0 && v[1]) { last = i; ts[i] = v[0]; } else break; }
            char d[96] = "-"; if (last >= 0 && last < n) pp_desc(&r[last], d, sizeof d);
            { int cs = -1, nch = 0; uint32_t dr = 0; for (int i = 0; i <= last && i < n; i++) { if (r[i].kind == PK_UPLOAD) { cs = i; nch++; dr = 0; } else if (i < last) dr += r[i].draws; }
              char kinds[64] = ""; int kw = 0; for (int i = MAX(0, last - 6); i <= last && i < n && kw < 56; i++) kw += snprintf(kinds + kw, sizeof kinds - kw, "%s%c", i == MAX(0, last - 6) ? "" : ",", "UGPCXDFTE"[r[i].kind < PK_N ? r[i].kind : 0]);
              double idle = cs > 0 && r[cs - 1].kind == PK_FLUSH ? (double)((ts[cs] - ts[cs - 1]) & g_pp.mask) * g_ts_ns / 1e6 : -1, run = cs >= 0 ? (double)((ts[last] - ts[cs]) & g_pp.mask) * g_ts_ns / 1e6 : -1;
              LOG("[RED] vk: passprof: stalled submission %llu: chunk %d starts at mark %d, the stalled interval is %d marks and %u draws after it (mark kinds %d..%d: %s; U upload G gap P pass C copy X clear D dispatch F flush); GPU idle before the chunk %.3f ms (-1: first chunk), chunk start to the stalled interval %.3f ms",
                  (unsigned long long)p->fence, nch, cs, last - cs, dr, MAX(0, last - 6), last, kinds, idle, run); }
            LOG("[RED] vk: passprof: stalled submission %llu wrote marks 0..%d of %d; the GPU was in interval %d: %s %s (vs/cs %.16s fs %.16s, %u draws from draw %u of the pass)", (unsigned long long)p->fence, last, n, last,
                last >= 0 && r[last].kind < PK_N ? kn[r[last].kind] : "?", d, last >= 0 ? r[last].vs : "-", last >= 0 ? r[last].fs : "-", last >= 0 ? r[last].draws : 0u, last >= 0 ? r[last].d0 : 0u);
        }
        return; }
    for (int i = 0; i + 1 < n; i++) {
        uint64_t t = (ts[i + 1] - ts[i]) & g_pp.mask;
        if (t > g_pp.mask >> 1) { RED_ONCE(46, "vk: passprof: mark %d of %d is before mark %d (%llu < %llu): submission skipped, recording order and execution order disagree", i + 1, n, i, (unsigned long long)ts[i + 1], (unsigned long long)ts[i]); g_ppw.bad++; return; }
        dur[i] = (uint64_t)(t * g_ts_ns);
        if (dur[i] > 500000000ull && r[i].kind < PK_N) { char d[96]; pp_desc(&r[i], d, sizeof d);
            LOG("[RED] vk: passprof: submission %llu interval %d of %d took %.1f ms: %s %s (vs/cs %.16s fs %.16s, %u draws)", (unsigned long long)p->fence, i, n, dur[i] / 1e6, kn[r[i].kind], d, r[i].vs, r[i].fs, r[i].draws); }
    }
    if (!g_ppf.open || p->frame != g_ppf.frame) {   // submissions complete in order: a new frame number closes the previous frame
        if (g_ppf.open) { pp_frame_close(); if (p->frame - g_ppw.start >= PP_WIN) pp_report(p->frame); }
        else g_ppw.start = p->frame;
        memset(&g_ppf, 0, sizeof g_ppf); g_ppf.open = 1; g_ppf.frame = p->frame;
    }
    g_ppw.subs++; if ((uint64_t)n > g_ppw.maxn) g_ppw.maxn = (uint64_t)n;
    if (p->ppwant) { g_ppw.trunc++; if ((uint64_t)(n + p->ppwant) > g_ppw.maxneed) g_ppw.maxneed = (uint64_t)(n + p->ppwant); }
    for (int i = 0; i + 1 < n; i++) {
        int k = r[i].kind; if (k >= PK_N) continue;
        g_ppf.ns[k] += dur[i]; g_ppf.cnt[k]++;
        if (k != PK_PASS) continue;
        uint64_t o = g_ppf.ord++, sig = pp_sig(&r[i]); pp_slot *s = &g_ppw.slot[o < PP_PASSES ? o : PP_PASSES - 1];
        if (s->n && s->sig != sig) { s->chg++; g_ppw.chg++; }
        s->sig = sig; s->last = r[i]; s->n++; s->ns += dur[i]; if (dur[i] > s->max) s->max = dur[i];
        if (i > 0 && r[i - 1].kind == PK_GAP) s->gap += dur[i - 1];
        s->draws += r[i].draws; s->clears += r[i].clears; g_ppf.draws += r[i].draws; g_ppf.clears += r[i].clears;
        pp_shader_add(&r[i], dur[i]);
    }
}
// ---------- Conditional rendering (profile key render_cond), as nvn_mtl.m ----------
// SetRenderEnableConditional(mode, rec) renders if the 64-bit counters at rec and rec + 16 are equal (mode 0) / differ (mode 1), until SetRenderEnable(true). Some games, like Tears of the
// Kingdom, clears a target under such a condition only when something was drawn into it since its last clear (rec set to 1 by the CPU, a timestamp report after each clear writes 0).
// Evaluated here on the CPU: a report replayed earlier in this submission has not been written yet, so a timestamp one counts with its known counter 0; a condition on a
// samples-passed report is unknown and keeps drawing. Inside a block evaluated false, draws and clears are dropped
static int g_rc_skip;
#define RC_PEND 256
static struct { uint64_t addr; int kind; } g_rc_pend[RC_PEND]; static int g_rc_npend;   // reports replayed in this submission: kind 1 timestamp (counter 0), 2 another counter (unknown)
static uint64_t g_rc_occ[4096];   // addresses ever written by a non-timestamp report: conditions on them keep drawing
static void rc_occ_add(uint64_t a) { for (uint32_t i = (uint32_t)(a >> 4) & 4095, k = 0; k < 4096; k++, i = (i + 1) & 4095) { if (g_rc_occ[i] == a) return; if (!g_rc_occ[i]) { g_rc_occ[i] = a; return; } } }
static int rc_occ_has(uint64_t a) { for (uint32_t i = (uint32_t)(a >> 4) & 4095, k = 0; k < 4096; k++, i = (i + 1) & 4095) { if (g_rc_occ[i] == a) return 1; if (!g_rc_occ[i]) return 0; } return 0; }
static void rc_note(uint64_t a, int kind) {
    if (kind == 2) rc_occ_add(a);
    for (int i = 0; i < g_rc_npend; i++) if (g_rc_pend[i].addr == a) { g_rc_pend[i].kind = kind; return; }
    if (g_rc_npend < RC_PEND) { g_rc_pend[g_rc_npend].addr = a; g_rc_pend[g_rc_npend++].kind = kind; }
}
static int rc_value(uint64_t a, uint64_t *v) {   // 0 = unknown at replay time
    if (rc_occ_has(a)) return 0;
    for (int i = g_rc_npend - 1; i >= 0; i--) if (g_rc_pend[i].addr == a) { if (g_rc_pend[i].kind != 1) return 0; *v = 0; return 1; }
    *v = *(const volatile uint64_t *)(uintptr_t)a; return 1;
}
static uint64_t g_stat_rc_skipped;
// Diagnostics. DM_VK_TRACE=1 with DM_VK_SHOT: log every pass of the frame before each shot (targets, draws, clears).
// DM_VK_DBG bits: 1 depth / stencil test off, 2 culling off, 4 no viewport y flip, 8 blending off with all channels written, 16 colour clears skipped, 32 depth-stencil clears skipped
static int g_trace, g_dbg = -1; static unsigned g_tr_pass, g_tr_draws, g_tr_clears, g_tr_skip;
// DM_VK_TRACE_FULL=1: in the traced frame, every draw of a 1280x720 R11G11B10F + D32 pass (the depth prepass and the forward colour pass) is logged, not only the first 6
static int g_tr_full;
// DM_VK_TRACE_FS=<fs key>: in the traced frame, the uniform buffers of the first 2 draws with this fragment shader are logged (both stages, words as the CPU sees them at record time)
static int g_trfs_now, g_trfs_cnt;
static uint64_t g_dyn_dump[32]; static int g_ndyn;   // with DM_VK_DUMP: the images those draws sample are dumped at the shot too (raw .bin as well)
static int dbg(int bit) { if (g_dbg < 0) { const char *e = getenv("DM_VK_DBG"); g_dbg = e ? atoi(e) : 0; if (g_dbg) LOG("vk: DM_VK_DBG=%d", g_dbg); } return g_dbg & bit; }
static void tr_end(void) { if (g_trace && g_tr_pass) LOG("vk: T   pass %u: %u draws, %u clears, %u skipped (empty scissor)", g_tr_pass, g_tr_draws, g_tr_clears, g_tr_skip); g_tr_draws = g_tr_clears = g_tr_skip = 0; }
static void state_reset(void) {
    memset(&S, 0, sizeof S);
    for (int i = 0; i < 8; i++) S.blend[i] = (blend_st){i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    S.color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; S.chmask.mask = 0xffffffffu;
    S.poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    S.ds = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS, 1, 1, 1, 1, 1, 1};
    S.st_vmask[0] = S.st_vmask[1] = S.st_wmask[0] = S.st_wmask[1] = 0xff; S.dr[1] = 1; S.patch = 3;
}

// GPU bindings belong to a queue, including across submissions (as nvn_mtl.m, 7a3a875): one saved S / R per queue serial. Fire Emblem X-2 destroys its launcher's queue before creating the game's,
// and Three Houses alternates submissions on two queues. Both calls come from nvnQueueSubmitCommands / nvnQueueFinalize under the submission lock, on the thread that replays (mtl_sub_begin follows
// mtl_queue_activate); no render pass is open then (mtl_sub_commit ended it), so R.active is 0 in every saved copy
typedef struct queue_state { uint64_t serial; __typeof__(S) s; __typeof__(R) r; struct queue_state *next; } queue_state;
static queue_state *g_queue_states, *g_active_queue; static int g_state_inited;
void mtl_queue_activate(uint64_t serial) {
    if (g_active_queue && g_active_queue->serial == serial) return;
    if (g_active_queue) { g_active_queue->s = S; g_active_queue->r = R; }
    queue_state *q = g_queue_states;
    while (q && q->serial != serial) q = q->next;
    if (q) { S = q->s; R = q->r; R.active = 0; }
    else {
        q = calloc(1, sizeof *q); if (!q) die("vk: queue state allocation failed");
        q->serial = serial; q->next = g_queue_states; g_queue_states = q;
        state_reset(); memset(&R, 0, sizeof R); g_state_inited = 1;
    }
    g_active_queue = q;
}
void mtl_queue_forget(uint64_t serial) {   // QueueFinalize: the queue's state, with the guest object pointers in it, goes
    queue_state **p = &g_queue_states;
    while (*p && (*p)->serial != serial) p = &(*p)->next;
    if (!*p) return;
    queue_state *q = *p; *p = q->next;
    if (g_active_queue == q) { g_active_queue = NULL; state_reset(); memset(&R, 0, sizeof R); g_state_inited = 1; }
    free(q);
}
void mtl_set_queue_origin_lower_left(int on) { atomic_store(&g_queue_ll, on != 0); }   // the raster flip captured at queue creation; selected before each submission, under the submission lock

static int g_since_flush; static uint64_t g_nflush;   // draws since the last chunk was submitted, chunks submitted (sub_flush)
// The uploads recorded so far run now, ahead of the submission being recorded, and are waited for. Its other commands are not submitted, and run after them as they always did: uploads come first anyway
static void up_flush(void) {
    ctx *c = g_cur; static VkFence fe; uint64_t t0 = now_ns();
    cv_wait();   // what these copies read
    if (!fe) { VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VKCHK(vkCreateFence(g_dev, &fci, NULL, &fe)); }
    if (c->up_used) barrier_all(c->up);
    VKCHK(vkEndCommandBuffer(c->up));
    VkCommandBufferSubmitInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = c->up};
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi};
    pthread_mutex_lock(&g_qmu); VkResult r = pvkQueueSubmit2(g_q, 1, &si, fe); pthread_mutex_unlock(&g_qmu);
    if (r) die("vk: queue submit (uploads) failed: %d (device lost?)", (int)r);
    for (int w = 1; (r = vkWaitForFences(g_dev, 1, &fe, VK_TRUE, 10000000000ull)) == VK_TIMEOUT; w++) LOG("[RED] vk: GPU hang? uploads still running after %d s", 10 * w);
    if (r) die("vk: waiting for uploads failed: %d (device lost?)", (int)r);
    VKCHK(vkResetFences(g_dev, 1, &fe));
    vkFreeCommandBuffers(g_dev, c->pool, 1, &c->up);   // its pool can't reset one alone: a fresh one takes its place, and the pool's reset covers it from then on
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = c->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VKCHK(vkAllocateCommandBuffers(g_dev, &ai, &c->up)); c->xup[c->ichunk] = c->up;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VKCHK(vkBeginCommandBuffer(c->up, &bi));
    c->up_used = 0; g_ups.head = 0; g_ups.flushes++; g_ups.wait_ns += now_ns() - t0;
}
void mtl_sub_begin(void) {
    ensure_init();
    if (!g_state_inited) { g_state_inited = 1; state_reset(); }
    ctx *c = &g_ctx[g_nsub++ % g_nctx];
    uint64_t w0 = now_ns();
    if (c->fence) for (int w = 1; !mtl_wait(c->fence, 10000000000ull); w++) LOG("[RED] vk: GPU hang? submission %llu still running after %d s: its command pool and descriptor pools are not reset while it runs, waiting on", (unsigned long long)c->fence, 10 * w);   // was one line, then a reset under the running submission
    WT_ADD(WT_CTX, now_ns() - w0);
    dead_reap(); rc_age();
    g_ups.sub_bytes = g_ups.ring_bytes = g_ups.wait_ns = 0; g_ups.flushes = 0; g_ring.extra = 0; g_ring.sub = 0;
    if (g_ups.h.buf && g_nsub - g_ups.last_sub > 3000) { dead *dd = calloc(1, sizeof *dd); if (dd) { dd->buf = g_ups.h.buf; dd->mem = g_ups.h.mem; dead_add(dd); } memset(&g_ups.h, 0, sizeof g_ups.h); g_ups.head = 0; }   // no loading for a while: the arena goes
    VKCHK(vkResetCommandPool(g_dev, c->pool, 0));
    c->ichunk = 0; c->cb = c->xcb[0]; c->up = c->xup[0];
    for (int i = 0; i < c->ndp; i++) VKCHK(vkResetDescriptorPool(g_dev, c->dp[i], 0));
    c->curdp = 0;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VKCHK(vkBeginCommandBuffer(c->cb, &bi)); VKCHK(vkBeginCommandBuffer(c->up, &bi));
    if (g_tsq) { int q = (int)(c - g_ctx); vkCmdResetQueryPool(c->up, g_tsq, 2 * q, 2); vkCmdWriteTimestamp(c->up, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_tsq, 2 * q); }
    c->up_used = 0; c->npost = 0; g_cur = c; R.active = 0; g_rc_npend = 0; g_rc_skip = 0; g_since_flush = 0;
    if (g_pp.on) pp_chunk_begin(c, 1);
    shadow_reset(); drvc_reset();   // a new command buffer holds no state; ring blocks of earlier submissions are not reused
}
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_exec_rest(const uint64_t *handles, int n) { (void)handles; (void)n; }
static void rp_end(void) { if (R.active) { pvkCmdEndRendering(g_cur->cb); R.active = 0; if (g_pp.on) pp_pass_end(); barrier_k(g_cur->cb, BK_PASS); tr_end(); } }
uint64_t mtl_sub_commit(void) {
    ctx *c = g_cur; rp_end();
    if (c->up_used) barrier_all(c->up);
    if (g_pp.on) pp_end();
    if (g_tsq) vkCmdWriteTimestamp(c->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_tsq, 2 * (int)(c - g_ctx) + 1);
    VKCHK(vkEndCommandBuffer(c->up)); VKCHK(vkEndCommandBuffer(c->cb));
    pthread_mutex_lock(&g_qmu);
    uint64_t fence = ++g_fence; c->fence = fence;
    VkCommandBufferSubmitInfo cbs[2] = {{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = c->up}, {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = c->cb}};
    VkSemaphoreSubmitInfo sig = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_tl, .value = fence, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 2, .pCommandBufferInfos = cbs, .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &sig};
    uint64_t w0 = now_ns(); VkResult r = pvkQueueSubmit2(g_q, 1, &si, VK_NULL_HANDLE); WT_ADD(WT_SUBMIT, now_ns() - w0);
    if (r) die("vk: queue submit failed: %d (device lost?)", (int)r);
    pend *p = calloc(1, sizeof *p); p->fence = fence; p->n = c->npost; p->q = g_tsq ? (int)(c - g_ctx) : -1; p->frame = g_presents; p->t_commit = now_ns();
    { static uint32_t g0; static uint64_t r0; uint32_t g = atomic_load(&g_pool_gen); p->d_pool = g - g0; g0 = g; p->d_rc = (uint32_t)(g_stat_img_reuse - r0); r0 = g_stat_img_reuse;
      uint64_t tc = atomic_load(&g_pool_t_chg), td = atomic_load(&g_pool_t_del); p->ms_chg = tc ? (p->t_commit - tc) / 1e6f : -1; p->ms_del = td ? (p->t_commit - td) / 1e6f : -1; }
    { uint64_t f = g_presents; if (g_gf[f % 64].frame != f) { g_gf[f % 64].ns = 0; g_gf[f % 64].sub = 0; g_gf[f % 64].done = 0; g_gf[f % 64].frame = f; } g_gf[f % 64].sub++; }
    if (c->npost) { p->ops = malloc(c->npost * sizeof *p->ops); memcpy(p->ops, c->posts, c->npost * sizeof *p->ops); }
    if (g_pp.on && (p->ppr = malloc((size_t)g_pp.n * sizeof *p->ppr))) { memcpy(p->ppr, g_pp.rec[g_pp.q], (size_t)g_pp.n * sizeof *p->ppr); p->ppn = g_pp.n; p->ppwant = g_pp.want; }   // the context's records are reused by its next submission
    pthread_mutex_lock(&g_pmu); if (g_pend_tail) g_pend_tail->next = p; else g_pend_head = p; g_pend_tail = p; pthread_cond_signal(&g_pcv); pthread_mutex_unlock(&g_pmu);
    pthread_mutex_unlock(&g_qmu);
    { uint64_t rest = g_ring.sub - MIN(g_ring.sub, g_ups.ring_bytes); if (rest > g_ring.peak) g_ring.peak = rest; }
    if (g_ups.flushes || g_ring.extra || g_ups.sub_bytes > (64u << 20)) { static int nl; if (nl++ < 64) LOG("vk: submission %llu loaded %.1f MB of textures: ran its uploads early %d times, %.1f ms in all; ring beyond its %d MB: %llu MB", (unsigned long long)fence, g_ups.sub_bytes / 1048576.0, g_ups.flushes, g_ups.wait_ns / 1e6, RING_N * (RING_CHUNK >> 20), (unsigned long long)(g_ring.extra >> 20)); }
    g_cur = NULL;
    return fence;
}

// ---------- Render targets ----------
static VkFormat view_format(const vimg *v, const view_t *vw, int has) {
    if (!has || !vw->set_format) return v->fmt;
    if (v->t.d.format != vw->format && fmt_of(v->t.d.format).conv == CV_BC6 && fmt_of(vw->format).conv != CV_BC6) { VkFormat of = tex_vk_format(vw->format); if (of && of != v->fmt) RED_ONCE(134, "vk: view format %d of a BC6H texture: its own format is used", (int)of); return v->fmt; }
    if (fmt_of(vw->format).conv == CV_BC6 && (v->fmt == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 || v->fmt == VK_FORMAT_BC1_RGB_SRGB_BLOCK)) return v->fmt;   // a view of the texture's own BC6H format: see bc6_fmt
    VkFormat f = tex_vk_format(vw->format);
    if (f && bc_twin(v->fmt)) return !bc_twin_on() || fmt_is_srgb(f) == fmt_is_srgb(v->fmt) ? v->fmt : bc_twin(v->fmt);   // sRGB or not; BC7 and native BC1-3 alike
    if (!f || !v->mutable_) { if (f && f != v->fmt) RED_ONCE(51, "vk: view format %d of an image of format %d, which cannot be viewed in another format (depth or compressed): the image's own format is used", (int)f, (int)v->fmt); return v->fmt; }
    if (v->fl) {   // an image created with a format list may only be viewed in the formats of the list
        VkFormat fam[8]; int n = fmt_family(v->fmt, fam), in = 0; for (int i = 0; i < n; i++) in |= fam[i] == f;
        if (!in) { RED_ONCE(46, "vk: DM_VK_IMGFLAGS: view format %d of an image of format %d is outside its format list: the image's own format is used (colours may be reinterpreted wrongly)", (int)f, (int)v->fmt); return v->fmt; }
    }
    if (vk_texel_bytes(f) != vk_texel_bytes(v->fmt)) { RED_ONCE(50, "vk: view format %d (%u bytes a texel) of an image of format %d (%u bytes) differs in texel size: the image's own format is used (colours may be reinterpreted wrongly)", (int)f, vk_texel_bytes(f), (int)v->fmt, vk_texel_bytes(v->fmt)); return v->fmt; }
    return f;
}
// Chunked submission: the game waits for frame N's GPU work before it submits frame N+1, and one submission is replayed whole before the GPU sees any of it, so replay (CPU) and GPU
// ran one after the other (phone: ~26 + ~27 ms a frame). Between passes, after DM_VK_FLUSH draws (default 500, 0 = off), the commands so far go to the queue and replay continues
// in a fresh command buffer pair of the same context: the GPU starts on the frame while it is still being replayed. Queue order is kept; every pass already ends with barrier_all,
// and the context's timeline fence, post-ops, ring and descriptor memory are signalled / released only with the last chunk (mtl_sub_commit)
static int g_flush_draws = -1;
static void sub_flush(void) {
    ctx *c = g_cur;
    if (c->ichunk + 1 >= 32) return;
    if (c->ichunk + 1 >= c->nx) {
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = c->pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 2};
        VkCommandBuffer cbs[2]; if (vkAllocateCommandBuffers(g_dev, &ai, cbs)) return;
        c->xcb[c->nx] = cbs[0]; c->xup[c->nx] = cbs[1]; c->nx++;
    }
    PP(PK_FLUSH);   // the old command buffer's last mark (c->cb is still the old one): up to the next chunk's first mark the GPU waits for the replay
    if (c->up_used) barrier_all(c->up);
    VKCHK(vkEndCommandBuffer(c->up)); VKCHK(vkEndCommandBuffer(c->cb));
    VkCommandBufferSubmitInfo cbs[2] = {{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = c->up}, {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = c->cb}};
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 2, .pCommandBufferInfos = cbs};
    uint64_t w0 = now_ns(); pthread_mutex_lock(&g_qmu); VkResult r = pvkQueueSubmit2(g_q, 1, &si, VK_NULL_HANDLE); pthread_mutex_unlock(&g_qmu); WT_ADD(WT_SUBMIT, now_ns() - w0);
    if (r) die("vk: queue submit (chunk) failed: %d (device lost?)", (int)r);
    c->ichunk++; c->cb = c->xcb[c->ichunk]; c->up = c->xup[c->ichunk]; c->up_used = 0;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VKCHK(vkBeginCommandBuffer(c->cb, &bi)); VKCHK(vkBeginCommandBuffer(c->up, &bi));
    shadow_reset(); g_since_flush = 0; g_nflush++;
    if (g_pp.on) pp_chunk_begin(c, 0);
}
static int rt_begin(void) {
    if (R.active) return 1;
    if (g_flush_draws < 0) { const char *e = getenv("DM_VK_FLUSH"); g_flush_draws = e ? atoi(e) : 500; LOG("vk: chunked submission every %d draws (DM_VK_FLUSH, 0 = off)", g_flush_draws); }
    if (g_flush_draws > 0 && g_since_flush >= g_flush_draws) sub_flush();
    int none = !R.has;   // R.rt is stale then
    if (none && !g_rp_empty) return 0;
    VkRenderingAttachmentInfo ca[8]; memset(ca, 0, sizeof ca);
    VkRenderingAttachmentInfo da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, sa = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    uint32_t w = ~0u, h = ~0u, layers = ~0u; int nc = 0; uint8_t samples = 1;
    memset(R.cf, 0, sizeof R.cf); R.df = R.sf = 0;
    for (int i = 0; !none && i < R.rt.n && i < 8; i++) {
        ca[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        const tex_t *t = R.rt.color[i]; if (!t) continue;
        vimg *v = vimg_get(t); if (!v) continue;
        const view_t *vw = &R.rt.cview[i]; int has = R.rt.has_cview[i];
        unsigned lv = has && vw->set_levels ? vw->base_level : 0, ly = has && vw->set_layers ? vw->min_layer : 0;
        VkFormat f = view_format(v, vw, has);
        unsigned have = (v->itype == VK_IMAGE_TYPE_3D ? MAX(v->d >> lv, 1u) : v->layers), nl = 1;   // 3D: the layers are the depth planes (2D_ARRAY_COMPATIBLE image)
        if (ly >= have) ly = have - 1;
        if (g_rp_layered) { nl = have - ly; layers = MIN(layers, nl); }
        VkImageView iv = vimg_view(v, nl > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, f, lv, 1, ly, nl, SW_ID, VK_IMAGE_ASPECT_COLOR_BIT);
        if (!iv) continue;
        v->gpu_owned = 1; v->uploaded = 1;
        ca[i].imageView = iv; ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL; ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        R.cf[i] = f; nc = i + 1; samples = (uint8_t)v->samples;
        w = MIN(w, MAX(v->w >> lv, 1u)); h = MIN(h, MAX(v->h >> lv, 1u));
    }
    if (!none && R.rt.depth) {
        vimg *v = vimg_get(R.rt.depth);
        if (v && (v->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) {
            const view_t *vw = &R.rt.dview; int has = R.rt.has_dview;
            unsigned lv = has && vw->set_levels ? vw->base_level : 0, ly = has && vw->set_layers ? vw->min_layer : 0, nl = 1;
            if (ly >= v->layers) ly = v->layers - 1;
            if (g_rp_layered) { nl = v->layers - ly; layers = MIN(layers, nl); }
            VkImageView iv = vimg_view(v, nl > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, v->fmt, lv, 1, ly, nl, SW_ID, v->aspect);
            if (iv) {
                v->gpu_owned = 1; v->uploaded = 1;
                if (v->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) { da.imageView = iv; da.imageLayout = VK_IMAGE_LAYOUT_GENERAL; da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; da.storeOp = VK_ATTACHMENT_STORE_OP_STORE; R.df = v->fmt; }
                if (v->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) { sa.imageView = iv; sa.imageLayout = VK_IMAGE_LAYOUT_GENERAL; sa.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; sa.storeOp = VK_ATTACHMENT_STORE_OP_STORE; R.sf = v->fmt; }
                w = MIN(w, MAX(v->w >> lv, 1u)); h = MIN(h, MAX(v->h >> lv, 1u)); if (!nc) samples = (uint8_t)v->samples;
            }
        }
    }
    if (w == ~0u) { if (!g_rp_empty) return 0; w = h = 1; }
    R.w = w; R.h = h; R.nc = nc; R.samples = samples; R.layered = g_rp_layered;
    if (layers == ~0u || !g_rp_layered) layers = 1;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = layers, .colorAttachmentCount = (uint32_t)nc, .pColorAttachments = ca,
        .pDepthAttachment = R.df ? &da : NULL, .pStencilAttachment = R.sf ? &sa : NULL};
    if (g_pp.on) pp_pass_begin();
    pvkCmdBeginRendering(g_cur->cb, &ri);
    R.active = 1; g_stat_passes++; shadow_reset();   // dynamic state persists across rendering instances; forgotten anyway (conservative)
    if (g_trace) {
        char b[512]; int o = snprintf(b, sizeof b, "vk: T pass %u %ux%u%s:", ++g_tr_pass, w, h, layers > 1 ? " layered" : "");
        for (int i = 0; i < nc; i++) if (R.cf[i]) o += snprintf(b + o, sizeof b - o, " c%d=%#llx/f%u", i, (unsigned long long)R.rt.color[i]->serial, R.cf[i]);
        if (R.rt.depth && (R.df || R.sf)) o += snprintf(b + o, sizeof b - o, " d=%#llx/f%u", (unsigned long long)R.rt.depth->serial, R.df ? R.df : R.sf);
        LOG("%s", b);
        { static int full = -1; if (full < 0) { const char *e = getenv("DM_VK_TRACE_FULL"); full = e && atoi(e); }
          g_tr_full = full && w == 1280 && h == 720 && nc >= 1 && R.cf[0] == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && R.df == VK_FORMAT_D32_SFLOAT; if (g_tr_full) LOG("vk: T   (full draw list for this pass)"); }
    }
    return 1;
}
static VkRect2D clip_rect(void) {
    VkRect2D r = {{0, 0}, {R.w, R.h}};
    if (S.has_sc) {
        int64_t x0 = MAX(S.sc.x, 0), y0 = MAX(S.sc.y, 0), x1 = MIN((int64_t)S.sc.x + S.sc.w, (int64_t)R.w), y1 = MIN((int64_t)S.sc.y + S.sc.h, (int64_t)R.h);
        if (x1 <= x0 || y1 <= y0) { r.extent.width = r.extent.height = 0; return r; }
        r = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
    }
    return r;
}
static void clear_color(const p_clear_color *c) {
    if (dbg(16)) return;
    if (c->index < 0 || c->index >= R.rt.n || !R.rt.color[c->index]) { RED_ONCE(8, "color clear target %d invalid", c->index); return; }
    if (!rt_begin()) return;
    if (!R.cf[c->index]) return;
    VkRect2D rc = clip_rect(); if (!rc.extent.width) return;
    if (!(c->mask & 0xf)) return;   // no channels: nothing to clear (as nvn_mtl.m clear_color_tri)
    if ((c->mask & 0xf) != 0xf) RED_ONCE(9, "color clear with channel mask %#x: cleared whole", c->mask);
    VkClearAttachment ca = {VK_IMAGE_ASPECT_COLOR_BIT, (uint32_t)c->index};
    VkFormat f = (VkFormat)R.cf[c->index];
    int is_uint = f == VK_FORMAT_R8_UINT || f == VK_FORMAT_R16_UINT || f == VK_FORMAT_R32_UINT || f == VK_FORMAT_R8G8_UINT || f == VK_FORMAT_R16G16_UINT || f == VK_FORMAT_R32G32_UINT || f == VK_FORMAT_R8G8B8A8_UINT || f == VK_FORMAT_R16G16B16A16_UINT || f == VK_FORMAT_R32G32B32A32_UINT || f == VK_FORMAT_A2B10G10R10_UINT_PACK32;
    int is_sint = f == VK_FORMAT_R8_SINT || f == VK_FORMAT_R16_SINT || f == VK_FORMAT_R32_SINT || f == VK_FORMAT_R8G8_SINT || f == VK_FORMAT_R16G16_SINT || f == VK_FORMAT_R32G32_SINT || f == VK_FORMAT_R8G8B8A8_SINT || f == VK_FORMAT_R16G16B16A16_SINT || f == VK_FORMAT_R32G32B32A32_SINT;
    if (is_uint || is_sint) memcpy(ca.clearValue.color.uint32, c->rgba, 16);   // NVN passes integer clear values in the same 16 bytes
    else memcpy(ca.clearValue.color.float32, c->rgba, 16);
    VkClearRect cr = {rc, 0, 1};
    vkCmdClearAttachments(g_cur->cb, 1, &ca, 1, &cr); shadow_reset(); PP_CLEAR();   // an implementation may clear with a pipeline of its own
    if (g_trace) { g_tr_clears++; LOG("vk: T   clear c%d mask %#x %.3f %.3f %.3f %.3f rect %d,%d %ux%u", c->index, c->mask, c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3], rc.offset.x, rc.offset.y, rc.extent.width, rc.extent.height); }
}
static void clear_ds(const p_clear_ds *c) {
    if (dbg(32)) return;
    if (!rt_begin()) return;
    VkImageAspectFlags a = 0;
    if (c->depth_mask && R.df) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if ((c->stencil_mask & 0xff) && R.sf) { a |= VK_IMAGE_ASPECT_STENCIL_BIT; if ((c->stencil_mask & 0xff) != 0xff) RED_ONCE(52, "vk: stencil clear with mask %#x: cleared whole (vkCmdClearAttachments ignores write masks)", c->stencil_mask); }
    if (!a) return;
    VkRect2D rc = clip_rect(); if (!rc.extent.width) return;
    VkClearAttachment ca = {a, 0, {.depthStencil = {c->depth, (uint32_t)c->stencil}}};
    VkClearRect cr = {rc, 0, 1};
    vkCmdClearAttachments(g_cur->cb, 1, &ca, 1, &cr); shadow_reset(); PP_CLEAR();
    if (g_trace) { g_tr_clears++; LOG("vk: T   clear ds depth %.3f dmask %d stencil %d smask %#x", c->depth, c->depth_mask, c->stencil, c->stencil_mask); }
}

// ---------- Draws ----------
static halloc g_support, g_quad_ib;
static halloc g_sb_scratch;   // what an unbound / undecoded storage buffer points at: compute writes land here, never in g_zero_buf (every draw reads that) or the static arena
static VkImageView g_dummy_view[8]; static VkImageView g_dummy_depth_view[2]; static VkSampler g_dummy_samp;
static VkFormat vertex_format(int f, char ct) {   // NVN vertex attribute format → Vulkan, matched to the shader input's numeric class
    fmt_t m = fmt_of(f); VkFormat v = m.vk;
    if (!v) return VK_FORMAT_UNDEFINED;
    if (ct == 'f') return v;
    struct { VkFormat fl, i, u; } map[] = {
        {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_SINT, VK_FORMAT_R32_UINT}, {VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_SINT, VK_FORMAT_R32G32_UINT},
        {VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_SINT, VK_FORMAT_R32G32B32_UINT}, {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SINT, VK_FORMAT_R32G32B32A32_UINT},
        {VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SINT, VK_FORMAT_R8_UINT}, {VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SINT, VK_FORMAT_R8G8_UINT}, {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_UINT},
        {VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SINT, VK_FORMAT_R16_UINT}, {VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SINT, VK_FORMAT_R16G16_UINT}, {VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R16G16B16A16_UINT},
        {VK_FORMAT_R8_USCALED, VK_FORMAT_R8_SINT, VK_FORMAT_R8_UINT}, {VK_FORMAT_R16_USCALED, VK_FORMAT_R16_SINT, VK_FORMAT_R16_UINT}, {VK_FORMAT_R8G8B8A8_USCALED, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_UINT}};
    for (unsigned i = 0; i < sizeof map / sizeof *map; i++) if (v == map[i].fl) return ct == 'i' ? map[i].i : map[i].u;
    return v;
}
static VkBlendFactor bfac(int f) {
    switch (f) {
        case 1: return VK_BLEND_FACTOR_ZERO; case 2: return VK_BLEND_FACTOR_ONE; case 3: return VK_BLEND_FACTOR_SRC_COLOR; case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 5: return VK_BLEND_FACTOR_SRC_ALPHA; case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; case 7: return VK_BLEND_FACTOR_DST_ALPHA; case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 9: return VK_BLEND_FACTOR_DST_COLOR; case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR; case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR; case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR; case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA; case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
        case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR; case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR; case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA; case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    }
    return VK_BLEND_FACTOR_ONE;
}
static int g_has_dsb;   // the device has dualSrcBlend (dev_init): SRC1 factors are invalid without it
static int g_cs_sub32;  // a compute pipeline can require a subgroup size of 32 (dev_init)
static VkBlendFactor bfac_d(int f, VkBlendFactor fallback) {   // bfac, with the SRC1 factors replaced by the given one (ONE for a source, ZERO for a destination factor) on a device without dualSrcBlend
    VkBlendFactor v = bfac(f);
    if (!g_has_dsb && v >= VK_BLEND_FACTOR_SRC1_COLOR && v <= VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA) { RED_ONCE(56, "vk: blend factor %#x needs dualSrcBlend, which this GPU lacks: %s used instead", f, fallback == VK_BLEND_FACTOR_ONE ? "ONE" : "ZERO"); return fallback; }
    return v;
}
// An advanced blend mode (blend_st.adv, nvn_obj.h) replaces the blend functions; as nvn_mtl.m adv_blend. The Porter-Duff modes with uncorrelated overlap are sums of the source and destination
// weighted by 1, 0, As, Ad and their complements (NV_blend_equation_advanced, premultiplied colours), which the fixed-function factors give exactly; the result goes through ADD. A source that is
// not premultiplied is multiplied by As first, which only a source factor of ONE can take (SRC_ALPHA). The other modes (multiply, screen, overlay, HSL, ...) need the destination colour in the
// shader: they keep the state's own functions (the nearest standard blend), with a red line once per mode. VK_EXT_blend_operation_advanced is not used
static blend_st adv_blend(blend_st b) {
    if (!b.adv) return b;
    enum { Z = NVN_BLEND_FUNC_ZERO, O = NVN_BLEND_FUNC_ONE, SA = NVN_BLEND_FUNC_SRC_ALPHA, ISA = NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA, DA = NVN_BLEND_FUNC_DST_ALPHA, IDA = NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA };
    static const uint8_t pd[][4] = {   // {source factor, destination factor} for colour, then for alpha
        [NVN_BLEND_ADVANCED_MODE_ZERO] = {Z, Z, Z, Z},        [NVN_BLEND_ADVANCED_MODE_SRC] = {O, Z, O, Z},           [NVN_BLEND_ADVANCED_MODE_DST] = {Z, O, Z, O},
        [NVN_BLEND_ADVANCED_MODE_SRC_OVER] = {O, ISA, O, ISA}, [NVN_BLEND_ADVANCED_MODE_DST_OVER] = {IDA, O, IDA, O},
        [NVN_BLEND_ADVANCED_MODE_SRC_IN] = {DA, Z, DA, Z},     [NVN_BLEND_ADVANCED_MODE_DST_IN] = {Z, SA, Z, SA},
        [NVN_BLEND_ADVANCED_MODE_SRC_OUT] = {IDA, Z, IDA, Z},  [NVN_BLEND_ADVANCED_MODE_DST_OUT] = {Z, ISA, Z, ISA},
        [NVN_BLEND_ADVANCED_MODE_SRC_ATOP] = {DA, ISA, Z, O},  [NVN_BLEND_ADVANCED_MODE_DST_ATOP] = {IDA, SA, O, Z},
        [NVN_BLEND_ADVANCED_MODE_XOR] = {IDA, ISA, IDA, ISA},  [NVN_BLEND_ADVANCED_MODE_PLUS] = {O, O, O, O},
        [NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED] = {O, O, O, O}, [NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED_ALPHA] = {O, O, O, O},
    };
    const uint8_t *f = b.adv < sizeof pd / sizeof *pd && pd[b.adv][0] ? pd[b.adv] : NULL;
    if (!f || b.adv_ovl || (b.adv_npm && f[0] != O && f[0] != Z)) {
        static uint64_t seen; if (!(seen >> (b.adv & 63) & 1)) { seen |= 1ull << (b.adv & 63);
            LOG("[RED] vk: advanced blend mode %#x (overlap %d, %spremultiplied source) has no fixed-function equivalent: the state's blend functions are used", (unsigned)b.adv, (int)b.adv_ovl, b.adv_npm ? "not " : ""); }
        return b;
    }
    b.src_rgb = b.adv_npm && f[0] == O ? SA : f[0]; b.dst_rgb = f[1]; b.src_a = f[2]; b.dst_a = f[3];
    b.eq_rgb = b.eq_a = NVN_BLEND_EQUATION_ADD;
    return b;
}
static VkBlendOp bop(int e) { switch (e) { case 2: return VK_BLEND_OP_SUBTRACT; case 3: return VK_BLEND_OP_REVERSE_SUBTRACT; case 4: return VK_BLEND_OP_MIN; case 5: return VK_BLEND_OP_MAX; } return VK_BLEND_OP_ADD; }
static VkCompareOp cmpop(int f) { return (VkCompareOp)((f >= 1 && f <= 8 ? f : 8) - 1); }
static VkStencilOp stop(int o) { return (VkStencilOp)((o >= 1 && o <= 8 ? o : 1) - 1); }
static VkImageViewType view_type_for(char dim, int arr) {
    switch (dim) { case '1': return arr ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D; case '3': return VK_IMAGE_VIEW_TYPE_3D; case 'C': return arr ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE; }
    return arr ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
}
static int dummy_index(VkImageViewType t) { switch (t) { case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return 1; case VK_IMAGE_VIEW_TYPE_CUBE: return 2; case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return 3; case VK_IMAGE_VIEW_TYPE_3D: return 4; default: return 0; } }
static uint64_t ubo_word(int stage, int reg, int off) {   // a bindless handle the game wrote into its own uniform buffer
    const p_bind_buf *u = reg >= 0 && reg < 16 ? &S.ubo[stage][reg] : NULL; uint32_t v = 0;
    if (u && u->addr && (uint64_t)off + 4 <= u->size) memcpy(&v, (const void *)(uintptr_t)(u->addr + off), 4);
    return v;
}
// f1 bindings (vk_extra.tsv): the shader reads a 1D texture or texture buffer by integer coordinate x as a 2D texelFetch at (x & 0x3FFF, x >> 14). A texture buffer (target 10) is presented as a 2D
// image of 16384-texel rows and ceil(n / 16384) rows: a copy of its guest texels made in the submission's upload command buffer when they differ from the last copy (compared once per
// submission). Texels the GPU writes inside the same submission (compute) are not seen; a plain 1D texture (target 0) is already a 2D image one row high, so it takes the normal path
typedef struct { uint64_t serial; uint32_t h; VkFormat fmt; VkImage img; VkDeviceMemory dm; VkImageView view; uint8_t *copy; uint64_t bytes, sub; } f1img;
static f1img g_f1[32]; static int g_nf1, g_f1_next;
static void f1_retire(f1img *e) {
    if (e->img || e->dm || e->view) { dead *d = calloc(1, sizeof *d); if (d) { d->img = e->img; d->mem = e->dm; if (e->view) d->views[d->nv++] = e->view; dead_add(d); } }
    free(e->copy); memset(e, 0, sizeof *e);
}
static int tex_binding_f1(int tid, int sid, const vbind *b, VkDescriptorImageInfo *ii) {
    treg tr; if (!reg_tex_get(tid, &tr)) return 0;
    const tex_t *t = &tr.t; const nvn_texdesc *d = &t->d;
    if (d->target != 10 || !t->mem || !d->width) return 0;
    VkFormat f = tex_vk_format(d->format); fmt_t m = fmt_of(d->format); int bw = 1, bh = 1, bpb = nvn_fmt_bpb(d->format, &bw, &bh);
    if (!f || m.conv || bw != 1 || bh != 1 || !bpb) { RED_ONCE(72, "vk: texture buffer format %#x cannot be presented as a 2D image (f1 binding): normal path", d->format); return 0; }
    const uint32_t W = 16384; uint64_t n = d->width, bytes = n * (uint64_t)bpb; uint32_t h = (uint32_t)((n + W - 1) / W);
    if (bytes > (16u << 20) || h > g_lim.maxImageDimension2D || W > g_lim.maxImageDimension2D) { RED_ONCE(73, "vk: texture buffer of %llu texels is too large for an f1 binding: normal path", (unsigned long long)n); return 0; }
    if (!pool_live((uint64_t)(uintptr_t)t->mem, bytes)) { RED_ONCE(74, "vk: texture buffer storage %p is not inside one memory pool (f1 binding): normal path", (const void *)t->mem); return 0; }
    f1img *e = NULL; for (int i = 0; i < g_nf1; i++) if (g_f1[i].serial == t->serial) { e = &g_f1[i]; break; }
    if (e && (e->h != h || e->fmt != f)) { f1_retire(e); e = NULL; }
    if (!e) {
        e = g_nf1 < 32 ? &g_f1[g_nf1++] : &g_f1[g_f1_next++ % 32]; if (e->img) f1_retire(e);
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f, .extent = {W, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VkImage img = VK_NULL_HANDLE; VkDeviceMemory dm = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; VkResult r = vkCreateImage(g_dev, &ci, NULL, &img);
        if (!r) { VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev, img, &mr);
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
            r = vkAllocateMemory(g_dev, &ai, NULL, &dm); if (!r) r = vkBindImageMemory(g_dev, img, dm, 0); }
        if (!r) { VkImageViewCreateInfo vc = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}; r = vkCreateImageView(g_dev, &vc, NULL, &view); }
        if (r) { RED_ONCE(75, "vk: creating the 2D image of an f1 texture buffer failed (%d): normal path", (int)r); if (view) vkDestroyImageView(g_dev, view, NULL); if (img) vkDestroyImage(g_dev, img, NULL); if (dm) vkFreeMemory(g_dev, dm, NULL); memset(e, 0, sizeof *e); return 0; }
        memset(e, 0, sizeof *e); e->serial = t->serial; e->h = h; e->fmt = f; e->img = img; e->dm = dm; e->view = view; e->bytes = bytes;
        img_barrier_init(cur_up(), img, VK_IMAGE_ASPECT_COLOR_BIT);
        { static int nl; if (nl++ < 8) LOG("vk: f1 binding: texture buffer %#llx (%llu texels, format %#x) presented as a %ux%u 2D image", (unsigned long long)t->serial, (unsigned long long)n, d->format, W, h); }
    }
    if (e->sub != g_nsub) {
        e->sub = g_nsub;
        if (!e->copy || e->bytes != bytes || memcmp(e->copy, t->mem, bytes)) {
            uint8_t *nc = realloc(e->copy, bytes); if (nc) { e->copy = nc; memcpy(nc, t->mem, bytes); } e->bytes = bytes;
            halloc st = ring_alloc(bytes, 16); memcpy(st.p, t->mem, bytes);
            uint32_t last = (uint32_t)(n - (uint64_t)(h - 1) * W);
            VkBufferImageCopy bic[2]; memset(bic, 0, sizeof bic); int nb = 0;
            if (h > 1) bic[nb++] = (VkBufferImageCopy){.bufferOffset = st.off, .bufferRowLength = W, .bufferImageHeight = 1, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, h - 1, 1}};
            bic[nb++] = (VkBufferImageCopy){.bufferOffset = st.off + (uint64_t)(h - 1) * W * (uint64_t)bpb, .bufferRowLength = W, .bufferImageHeight = 1, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageOffset = {0, (int32_t)(h - 1), 0}, .imageExtent = {last, 1, 1}};
            vkCmdCopyBufferToImage(cur_up(), st.buf, e->img, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)nb, bic); g_stat_up_bytes += bytes;
        }
    }
    ii->imageView = e->view;
    samp_par sp; ii->sampler = reg_samp_get(sid, &sp) ? sampler_get(&sp, b->dref) : VK_NULL_HANDLE; if (!ii->sampler) ii->sampler = g_dummy_samp;
    (void)tid; return 1;
}
// The image view and sampler of one combined texture binding
static uint64_t g_bc_hit, g_bc_miss;   // tex_binding's cache
static void tex_binding(int stage, const vbind *b, VkDescriptorImageInfo *ii) {
    uint64_t h = 0; int sid = 0;
    if (b->kind == K_TX) { h = b->a < NTEX ? S.tex[stage][b->a] : 0; sid = b->c == 0 ? (int)(h >> 20) : b->c == 1 ? (b->b >= 0 && b->b < 128 ? (int)(S.samp[stage][b->b] >> 20) : 0) : b->b; }   // tx:t:s:ty: ty is the token's 4th number, stored in c (it was read from d, always 0: every sampler came from the texture handle)
    else if (b->kind == K_TU) { h = ubo_word(stage, b->a, b->b); sid = b->c < 0 ? b->d : (int)(ubo_word(stage, b->c, b->d) >> 20); }
    int tid = (int)(h & 0xfffff);
    VkImageViewType vt = view_type_for(b->dim, b->arr);
    ii->imageLayout = VK_IMAGE_LAYOUT_GENERAL; ii->imageView = VK_NULL_HANDLE; ii->sampler = VK_NULL_HANDLE;
    if (b->f1 && tex_binding_f1(tid, sid, b, ii)) return;
    // Cache of the result for this submission (submit thread only): the same texture / sampler pairs are bound by most draws, and the registry copies, image and view lookups and sampler
    // hash were most of push_stage's own time. Keyed by the ids, their registry versions, the binding's view shape and the image table generation; an image the CPU wrote since goes the long way
    static struct { int tid, sid; uint32_t tq, sq, gen; uint8_t dim, arr, dref; uint64_t sub; vimg *v; VkImageView view; VkSampler samp; } bc[1024];
    uint32_t tq = reg_tex_seq(tid), sq = reg_samp_seq(sid), gen = atomic_load_explicit(&g_ih_gen, memory_order_acquire);
    static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_BINDCACHE"); on = !e || atoi(e); LOG("vk: texture binding cache %s (DM_VK_BINDCACHE)", on ? "on" : "off"); }
    int cacheable = on && !((tq | sq) & 1) && !g_trace && !g_trfs_now;
    uint32_t slot = (hmix(((uint64_t)tid << 32) ^ (uint64_t)sid ^ ((uint64_t)b->dim << 21) ^ ((uint64_t)b->arr << 25) ^ ((uint64_t)b->dref << 26))) & 1023;
    typeof(bc[0]) *ce = &bc[slot];
    if (cacheable && ce->sub == g_nsub && ce->tid == tid && ce->sid == sid && ce->tq == tq && ce->sq == sq && ce->gen == gen && ce->dim == b->dim && ce->arr == b->arr && ce->dref == b->dref &&
        (!ce->v || ce->v->gpu_owned || !ce->v->cpu_dirty) && atomic_load_explicit(&g_ih_gen, memory_order_acquire) == gen) {   // gen again: no release freed v while its flags were read
        ii->imageView = ce->view; ii->sampler = ce->samp; g_bc_hit++; return; }
    g_bc_miss++;
    vimg *vv = NULL;
    treg tr; samp_par sp;
    int ok = reg_tex_get(tid, &tr), sok = reg_samp_get(sid, &sp);
    if (g_trace && (g_tr_draws < 6 || g_tr_full)) LOG("vk: T     st%d binding %d tex id %d → %s%#llx %ux%u fmt %#x, sampler %d", stage, b->binding, tid, ok ? "serial " : "unregistered ", ok ? (unsigned long long)tr.t.serial : 0ull,
        ok ? tr.t.d.width : 0, ok ? tr.t.d.height : 0, ok ? tr.t.d.format : 0, sid);
    if (ok) {
        vimg *v = vimg_get(&tr.t); vv = v;
        if (v && g_trfs_now) { int k = 0; while (k < g_ndyn && g_dyn_dump[k] != v->serial) k++; if (k == g_ndyn && g_ndyn < 32) g_dyn_dump[g_ndyn++] = v->serial; }
        if (v && g_trace && g_tr_full) LOG("vk: T       image type %d %ux%ux%u layers %u levels %u target %d view %d dim %d arr %d gpu %d uploaded %d dirty %d", (int)v->itype, v->w, v->h, v->d, v->layers, tr.t.d.levels,
            (int)tr.t.d.target, (int)vt, (int)b->dim, (int)b->arr, v->gpu_owned, v->uploaded, v->cpu_dirty);
        if (v) {
            if (!v->gpu_owned && (!v->uploaded || v->cpu_dirty || lin_changed(v))) vimg_upload(v);
            const view_t *vw = &tr.v; int has = tr.has_v;
            unsigned l0 = has && vw->set_levels ? vw->base_level : 0, nl = has && vw->set_levels ? vw->nlevels : 0, s0 = has && vw->set_layers ? vw->min_layer : 0, ns = has && vw->set_layers ? vw->nlayers : 0;
            int okt = 1;
            if (vt == VK_IMAGE_VIEW_TYPE_3D) okt = v->itype == VK_IMAGE_TYPE_3D;
            else if (v->itype == VK_IMAGE_TYPE_3D) okt = 0;
            else if (vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) { okt = v->cube_ok; if (vt == VK_IMAGE_VIEW_TYPE_CUBE) { if (!ns) ns = 6; if (ns != 6) okt = 0; } else { if (!ns) ns = v->layers - s0; ns -= ns % 6; if (!ns) okt = 0; } }
            else if (vt == VK_IMAGE_VIEW_TYPE_2D || vt == VK_IMAGE_VIEW_TYPE_1D) ns = 1;
            if (v->samples > 1) okt = 0;
            if (okt) {
                fmt_t m = fmt_of(tr.t.d.format);
                uint8_t sw[4]; unsigned tsw = tr.t.set_swz ? tr.t.swz : (2u | 3u << 3 | 4u << 6 | 5u << 9);
                for (int i = 0; i < 4; i++) { unsigned s = has && vw->set_swizzle ? vw->swz[i] : (tsw >> (3 * i)) & 7;
                    if (m.alpha_one && s == 5) s = 1;
                    sw[i] = s == 0 ? VK_COMPONENT_SWIZZLE_ZERO : s == 1 ? VK_COMPONENT_SWIZZLE_ONE : (uint8_t)(VK_COMPONENT_SWIZZLE_R + (s - 2)); }
                VkImageAspectFlags asp = v->aspect;
                if (asp == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) asp = has && vw->set_dsmode && vw->dsmode == 1 ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
                if (asp & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) memset(sw, 0, 4);
                ii->imageView = vimg_view(v, vt, asp == VK_IMAGE_ASPECT_COLOR_BIT ? view_format(v, vw, has) : v->fmt, l0, nl, s0, ns, sw, asp);
            }
        }
    }
    if (!ii->imageView) { ii->imageView = b->dref && (vt == VK_IMAGE_VIEW_TYPE_2D || vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY) ? g_dummy_depth_view[vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY] : g_dummy_view[dummy_index(vt)]; }
    ii->sampler = sok ? sampler_get(&sp, b->dref) : VK_NULL_HANDLE;
    if (!ii->sampler) { static samp_par dp = {1, 1, 7, 7, 7, 0, 4, 0, 0, 1000, 0, 1, {0, 0, 0, 0}}; ii->sampler = b->dref ? sampler_get(&dp, 1) : g_dummy_samp; }
    if (cacheable) { ce->tid = tid; ce->sid = sid; ce->tq = tq; ce->sq = sq; ce->gen = gen; ce->dim = (uint8_t)b->dim; ce->arr = (uint8_t)b->arr; ce->dref = (uint8_t)b->dref; ce->sub = g_nsub;
        ce->v = vv; ce->view = ii->imageView; ce->samp = ii->sampler; }
}
// One texel buffer binding: the NVN texture buffer (target 10) registered under the handle, as a view on its memory pool's buffer, so with no copy it reads what the CPU and compute shaders wrote. The format is the
// buffer's own: 3-component ones stay 3-component (the memory is tightly packed; the Xclipse 920 takes R32G32B32 as a texel buffer, but no 8- or 16-bit one, which log a RED). With nothing usable: a null descriptor (robustness2
// nullDescriptor, which device creation requires), reads give zero. A texture of another target at such a binding (the translator cannot tell a 1D fetch from a buffer fetch) logs a RED too. Views stay for reuse, keyed
// by the pool's serial, so a finalized pool's views only age out of the cache; an evicted one is destroyed when the submission that may use it has finished. Replay thread only, like tex_binding's cache
typedef struct { uint64_t pool, off, bytes; VkFormat fmt; VkBufferView view; } tbview;
enum { TBV_SETS = 1024, TBV_WAYS = 4 };
static tbview g_tbv[TBV_SETS][TBV_WAYS]; static uint8_t g_tbv_next[TBV_SETS];
static VkBufferView tbuf_view(int stage, const vbind *b) {
    uint64_t h = 0;
    if (b->kind == K_TX) h = b->a < NTEX ? S.tex[stage][b->a] : 0;
    else if (b->kind == K_TU) h = ubo_word(stage, b->a, b->b);
    int tid = (int)(h & 0xfffff);
    treg tr; if (!reg_tex_get(tid, &tr)) { RED_ONCE(80, "vk: texel buffer binding %d of stage %d holds id %d, which is not registered: a null descriptor", b->binding, stage, tid); return VK_NULL_HANDLE; }
    const tex_t *t = &tr.t; const nvn_texdesc *d = &t->d;
    if (d->target != 10 || !t->mem || !d->width) { RED_ONCE(81, "vk: texel buffer binding %d of stage %d holds a texture of target %d, width %u: a null descriptor", b->binding, stage, (int)d->target, d->width); return VK_NULL_HANDLE; }
    int nf = tr.has_v && tr.v.set_format ? tr.v.format : d->format;
    fmt_t m = fmt_of(nf); int bw = 1, bh = 1, bpb = nvn_fmt_bpb(nf, &bw, &bh);
    if (!m.vk || (m.conv && m.conv < CV_RGB_EXPAND8) || bw != 1 || bh != 1 || !bpb || !(fmt_props(m.vk).bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT)) {
        RED_ONCE(82, "vk: texture buffer of NVN format %#x (Vulkan %d) cannot be a texel buffer on this device: a null descriptor", nf, (int)m.vk); return VK_NULL_HANDLE; }
    for (int i = 0; i < 4; i++) { unsigned s = tr.has_v && tr.v.set_swizzle ? tr.v.swz[i] : t->set_swz ? (unsigned)(t->swz >> (3 * i)) & 7 : 2u + (unsigned)i;
        if (s != 2u + (unsigned)i) { RED_ONCE(83, "vk: a texture buffer has a swizzle, which a texel buffer cannot have: ignored"); break; } }
    uint64_t addr = (uint64_t)(uintptr_t)t->mem, bytes = (uint64_t)d->width * (uint64_t)bpb;
    VkBuffer tbuf; VkDeviceSize off; uint64_t tser; if (!pool_buf(addr, bytes, 0, &tbuf, &off, NULL, &tser)) { RED_ONCE(84, "vk: texture buffer storage %p, %llu bytes, is not inside one imported memory pool: a null descriptor", (const void *)t->mem, (unsigned long long)bytes); return VK_NULL_HANDLE; }
    if (off % g_lim.minTexelBufferOffsetAlignment || d->width > g_lim.maxTexelBufferElements) { RED_ONCE(85, "vk: texture buffer at pool offset %#llx of %u texels is outside the device's limits: a null descriptor", (unsigned long long)off, d->width); return VK_NULL_HANDLE; }
    tbview *set = g_tbv[hmix(tser * 0x9e3779b97f4a7c15ull ^ off ^ bytes << 24 ^ (uint64_t)m.vk << 52) % TBV_SETS];
    for (int i = 0; i < TBV_WAYS; i++) if (set[i].view && set[i].pool == tser && set[i].off == off && set[i].bytes == bytes && set[i].fmt == m.vk) return set[i].view;
    VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = tbuf, .format = m.vk, .offset = off, .range = bytes};
    VkBufferView view = VK_NULL_HANDLE; VkResult r = vkCreateBufferView(g_dev, &ci, NULL, &view);
    if (r) { RED_ONCE(86, "vk: creating a texel buffer view of Vulkan format %d, %llu bytes, failed (%d): a null descriptor", (int)m.vk, (unsigned long long)bytes, (int)r); return VK_NULL_HANDLE; }
    uint32_t si = (uint32_t)(set - g_tbv[0]) / TBV_WAYS; tbview *e = &set[g_tbv_next[si]++ % TBV_WAYS];
    if (e->view) { dead *dd = calloc(1, sizeof *dd); dd->bview = e->view; dead_add(dd); }
    *e = (tbview){tser, off, bytes, m.vk, view};
    return view;
}
// One storage image binding: the image registered under the handle in image slot a, one level, identity swizzle (Vulkan's rules for storage views). The translated shaders declare no format, so the view's format decides how
// texels convert. With nothing usable: a null descriptor (robustness2 nullDescriptor, which device creation requires), reads give zero and writes are dropped. b: the shader writes it (res.json flags & 4)
static void img_binding(int stage, const vbind *b, VkDescriptorImageInfo *ii) {
    ii->imageLayout = VK_IMAGE_LAYOUT_GENERAL; ii->imageView = VK_NULL_HANDLE; ii->sampler = VK_NULL_HANDLE;
    int tid = b->a >= 0 && b->a < 8 ? (int)(S.img[stage][b->a] & 0xfffff) : 0;
    treg tr;
    if (!reg_tex_get(tid, &tr)) { RED_ONCE(76, "vk: image slot %d of stage %d holds id %d, which is not registered: a null image", b->a, stage, tid); return; }
    tr.t.d.flags |= NVN_TEXTURE_FLAGS_IMAGE;
    vimg *v = vimg_get(&tr.t); if (!v) return;
    if (!(v->usage & VK_IMAGE_USAGE_STORAGE_BIT)) { RED_ONCE(77, "vk: a texture of NVN format %#x (Vulkan %d, %u samples) in an image slot has no storage usage (first used without the image flag, or no storage format): a null image", tr.t.d.format, (int)v->fmt, v->samples); return; }
    VkImageViewType vt = view_type_for(b->dim, b->arr);
    if ((vt == VK_IMAGE_VIEW_TYPE_3D) != (v->itype == VK_IMAGE_TYPE_3D)) { RED_ONCE(78, "vk: image slot view type %d on an image of type %d: a null image", (int)vt, (int)v->itype); return; }
    const view_t *vw = &tr.v; int has = tr.has_v;
    unsigned l0 = has && vw->set_levels ? vw->base_level : 0, s0 = has && vw->set_layers ? vw->min_layer : 0, ns = has && vw->set_layers ? vw->nlayers : 0;
    if (vt == VK_IMAGE_VIEW_TYPE_2D || vt == VK_IMAGE_VIEW_TYPE_1D) ns = 1;
    VkFormat f = view_format(v, vw, has);
    if (!(fmt_props(f).optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
        VkFormat fam[8], g = VK_FORMAT_UNDEFINED; int nf = v->mutable_ ? fmt_family(v->fmt, fam) : 0;
        for (int i = 0; i < nf && !g; i++) if (fmt_props(fam[i]).optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) g = fam[i];
        if (!g) { RED_ONCE(79, "vk: view format %d of an image of format %d has no storage support, nor has its family: a null image", (int)f, (int)v->fmt); return; }
        f = g;
    }
    if (!v->gpu_owned && (!v->uploaded || v->cpu_dirty || lin_changed(v))) vimg_upload(v);   // what the CPU put there first, as in tex_binding: from now on, if written, the GPU's copy counts
    if (b->b) { v->gpu_owned = 1; v->uploaded = 1; }
    ii->imageView = vimg_view(v, vt, f, l0, 1, s0, ns, SW_ID, VK_IMAGE_ASPECT_COLOR_BIT);
}
static int g_draw_binst;
// The inputs of a driver-constant block (what push_stage writes into it) against the last block of this stage and kind built in this submission: on a match its ring allocation is bound again
static int drv_reuse(int stage, int k2, VkDescriptorBufferInfo *out) {
    if (stage < 0 || stage > 4) return 0;   // not compute: its c0 block also holds the group counts and uniform table (dispatch), which the key does not cover
    drv_in *k = &g_drvn; size_t sz;
    if (!k2) { sz = sizeof k->d0; memset(&k->d0, 0, sz); for (int r = 0; r < 16; r++) { k->d0.a[r] = S.ssbo[stage][r].addr; k->d0.s[r] = (uint32_t)S.ssbo[stage][r].size; } k->d0.binst = g_draw_binst; }
    else { sz = sizeof k->d2; for (int r = 0; r < 8; r++) k->d2[r] = (uint32_t)S.img[stage][r]; for (int r = 0; r < 32; r++) k->d2[8 + r] = (uint32_t)S.tex[stage][r]; for (int r = 0; r < 128; r++) k->d2[40 + r] = (uint32_t)S.tex[stage][128 + r]; }
    typeof(g_drvc[0][0]) *e = &g_drvc[stage][k2];
    if (shadow(2) && e->valid && e->cur == g_ring.cur && !memcmp(k, &e->k, sz)) { *out = (VkDescriptorBufferInfo){e->buf, e->off, 4096}; g_sh_n[8]++; return 1; }   // same ring chunk: not recycled since
    g_sh_n[9]++; return 0;
}
static void drv_remember(int stage, int k2, const VkDescriptorBufferInfo *bi) {   // right after drv_reuse missed and the block was built
    if (stage < 0 || stage > 4) return;
    typeof(g_drvc[0][0]) *e = &g_drvc[stage][k2];
    e->valid = 1; e->cur = g_ring.cur; e->buf = bi->buffer; e->off = bi->offset; e->k = g_drvn;
}
static uint32_t g_disp_dim[3]; static halloc g_c0_dim; static int g_push_fallback;   // the current dispatch's group counts (c0 + 0x200), the c0 bank it was written into, storage bindings that fell back to g_sb_scratch
static int push_stage(VkPipelineLayout lay, vshader *sh, int stage) {
    VkWriteDescriptorSet w[40]; VkDescriptorBufferInfo bi[40]; VkDescriptorImageInfo ii[40]; VkBufferView tv[40]; int n = 0;
    for (int i = 0; i < sh->nb; i++) {
        const vbind *b = &sh->b[i];
        VkWriteDescriptorSet *o = &w[n]; memset(o, 0, sizeof *o);
        o->sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; o->dstBinding = b->binding; o->descriptorCount = 1; o->descriptorType = bind_dtype(b);
        if (b->type == 'u' || b->type == 'b') {
            bi[n] = b->type == 'b' ? (VkDescriptorBufferInfo){g_sb_scratch.buf, g_sb_scratch.off, 65536} : (VkDescriptorBufferInfo){g_zero_buf, 0, 256};
            switch (b->kind) {
                case K_SUP: bi[n] = (VkDescriptorBufferInfo){g_support.buf, g_support.off, 1392}; break;
                case K_C1: if (sh->has_c1) bi[n] = (VkDescriptorBufferInfo){sh->c1buf, sh->c1off, sh->c1size}; break;
                case K_PUSH: { halloc h = ring_alloc(DMSH_PUSH_SIZE, 256); memcpy(h.p, S.push[stage == 5 ? 1 : 0], DMSH_PUSH_SIZE); bi[n] = (VkDescriptorBufferInfo){h.buf, h.off, DMSH_PUSH_SIZE}; break; }   // NVN2 push constants (c[24]): the stage group's current block
                case K_UB: case K_SB: case K_UBP: {
                    const p_bind_buf *u = b->a >= 0 && b->a < 16 ? (b->kind == K_SB ? &S.ssbo[stage][b->a] : &S.ubo[stage][b->a]) : NULL;
                    VkBuffer vb; VkDeviceSize off;
                    uint64_t psz = 0, sz = u && u->size ? u->size : 16; if (b->kind == K_UB) sz = MIN(sz, (uint64_t)65536); else sz = MIN(sz, (uint64_t)g_lim.maxStorageBufferRange);
                    if (u && u->addr && buf_at_w(u->addr, sz, &vb, &off, &psz)) {
                        { VkDeviceSize al = b->kind == K_UB ? g_lim.minUniformBufferOffsetAlignment : g_lim.minStorageBufferOffsetAlignment;   // not enforced before: reported
                          if (al && off % al) RED_ONCE(38, "vk: shader %s stage %d slot %d: buffer offset %#llx in its pool is not a multiple of the GPU's descriptor offset alignment (%llu)", sh->key, stage, b->a, (unsigned long long)off, (unsigned long long)al); }
                        sz = MIN(sz, psz - off);
                        bi[n] = (VkDescriptorBufferInfo){vb, off, sz};
                        if (g_trfs_now && b->kind == K_UB) {
                            const uint32_t *wd = (const uint32_t *)(uintptr_t)u->addr; unsigned nw = (unsigned)MIN(u->size ? u->size : 16, (uint64_t)1024) / 4;
                            LOG("vk: T       ub st%d binding %d slot %d addr %#llx size %#llx", stage, b->binding, b->a, (unsigned long long)u->addr, (unsigned long long)u->size);
                            for (unsigned q = 0; q < nw; q += 8) { char t[200]; int o = snprintf(t, sizeof t, "vk: T         +%03x:", q * 4); for (unsigned r = q; r < q + 8 && r < nw; r++) o += snprintf(t + o, sizeof t - o, " %08x", wd[r]); LOG("%s", t); }
                        }
                    } else { g_push_fallback++; static int nl; if (nl++ < 8) LOG("[RED] vk: shader %s stage %d %s slot %d has no buffer bound (addr %#llx)", sh->key, stage, b->kind == K_UB ? "uniform" : "storage", b->a, u ? (unsigned long long)u->addr : 0ull); }
                    break; }
                case K_SBU: { uint64_t a = 0; const p_bind_buf *u = b->a >= 0 && b->a < 16 ? &S.ubo[stage][b->a] : NULL;
                    if (u && u->addr && (uint64_t)b->b + 8 <= u->size) memcpy(&a, (const void *)(uintptr_t)(u->addr + b->b), 8);
                    // Laid out like the driver bank's storage buffers, address then size (K_DRV0): inference. An unusable size still gets a whole chunk from the address, and a wrong one costs at most 256 MB pinned
                    // The size being inferred, a small one still gets 1 MB, what these bindings always had room for: pieces are much smaller now
                    uint64_t want = POOL_NOLEN; if (u && u->addr && (uint64_t)b->b + 12 <= u->size) { uint32_t sw; memcpy(&sw, (const void *)(uintptr_t)(u->addr + b->b + 8), 4); if (sw) want = MAX(MIN((uint64_t)sw, (uint64_t)256u << 20), (uint64_t)1u << 20); }
                    VkBuffer vb; VkDeviceSize off; uint64_t psz;
                    if (a && buf_at_w(a, want, &vb, &off, &psz)) bi[n] = (VkDescriptorBufferInfo){vb, off, psz - off};
                    break; }
                case K_DRV0: case K_DRV2: {
                    if (drv_reuse(stage, b->kind == K_DRV2, &bi[n])) break;
                    halloc h = ring_alloc(4096, 256); memset(h.p, 0, 4096);
                    if (b->kind == K_DRV0) {
                        int base = stage == 0 ? DMSH_C0_SSBO_VS : stage == 1 ? DMSH_C0_SSBO_PS : stage == 2 ? DMSH_C0_SSBO_GS : stage == 3 ? 0x210 : stage == 4 ? 0x310 : stage == 5 ? DMSH_C0_SSBO_CS : -1;   // hull 0x210, domain 0x310: never measured, the gap the vertex and geometry tables leave (inference)
                        if (base >= 0) for (int r = 0; r < 16; r++) { uint64_t a = S.ssbo[stage][r].addr; uint32_t s = (uint32_t)S.ssbo[stage][r].size; memcpy(h.p + base + 16 * r, &a, 8); memcpy(h.p + base + 16 * r + 8, &s, 4); }
                        if (stage != 5) { float mp = 2047.f; memcpy(h.p + 0x610, &mp, 4); memcpy(h.p + 0x644, &g_draw_binst, 4); }
                        else {   // compute (nvn_mtl.m encode_args_): the dispatch size at 0x200, the uniform buffer table at 0x210 + 16 r (address 64 bits, size 32 bits)
                            memcpy(h.p + 0x200, g_disp_dim, 12); g_c0_dim = h;
                            for (int r = 0; r < 16; r++) { uint64_t a = S.ubo[5][r].addr; uint32_t sz = (uint32_t)S.ubo[5][r].size; memcpy(h.p + 0x210 + 16 * r, &a, 8); memcpy(h.p + 0x210 + 16 * r + 8, &sz, 4); }
                        }
                    } else {
                        for (int r = 0; r < 8; r++) { uint32_t v = (uint32_t)S.img[stage][r]; memcpy(h.p + 0x120 + 8 * r, &v, 4); }
                        for (int r = 0; r < 32; r++) { uint32_t v = (uint32_t)S.tex[stage][r]; memcpy(h.p + 0x20 + 8 * r, &v, 4); }
                        for (int r = 0; r < 128 && 0x168 + 8 * r + 4 <= 4096; r++) { uint32_t v = (uint32_t)S.tex[stage][128 + r]; memcpy(h.p + 0x168 + 8 * r, &v, 4); }
                    }
                    bi[n] = (VkDescriptorBufferInfo){h.buf, h.off, 4096}; drv_remember(stage, b->kind == K_DRV2, &bi[n]); break; }
                default: if (b->type == 'b') g_push_fallback++; break;
            }
            o->pBufferInfo = &bi[n];
        } else if (b->type == 'c') { tex_binding(stage, b, &ii[n]); o->pImageInfo = &ii[n]; }
        else if (b->type == 'i') { img_binding(stage, b, &ii[n]); o->pImageInfo = &ii[n]; }
        else if (b->type == 't') { tv[n] = tbuf_view(stage, b); o->pTexelBufferView = &tv[n]; }
        else { RED_ONCE(10,"vk: binding type %c (shader %s) not implemented: left out", b->type, sh->key); continue; }
        n++;
    }
    if (sh->stage == 1) { if (n) pvkCmdPushDescriptorSetKHR(g_cur->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, lay, (uint32_t)sh->set, (uint32_t)n, w); }
    else {
        VkDescriptorSet ds = dset_alloc(sh->dsl);
        for (int i = 0; i < n; i++) w[i].dstSet = ds;
        if (n) vkUpdateDescriptorSets(g_dev, (uint32_t)n, w, 0, NULL);
        vkCmdBindDescriptorSets(g_cur->cb, sh->stage == 5 ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS, lay, (uint32_t)sh->set, 1, &ds, 0, NULL);
    }
    return 1;
}
static void drop(int why) { g_stat_drop[why & 15]++; }
// Records the dynamic state (call, with the new value n_) unless the shadow (valid when use) already holds it: scalar / bytewise
#define DY(fld, val, call) do { __typeof__(D.fld) n_ = (val); if (!use || n_ != D.fld) { D.fld = n_; call; g_sh_n[0]++; } else g_sh_n[1]++; } while (0)
#define DYM(fld, val, call) do { __typeof__(D.fld) n_ = (val); if (!use || memcmp(&n_, &D.fld, sizeof n_)) { D.fld = n_; call; g_sh_n[0]++; } else g_sh_n[1]++; } while (0)
static int vb_same(int slot, VkBuffer b, VkDeviceSize off, VkDeviceSize sz, int sh) {   // 1: the command buffer already has this vertex buffer binding (else it is noted as bound: the caller binds it)
    int same = sh && (D.vbv >> slot & 1) && D.vb[slot].b == b && D.vb[slot].off == off && D.vb[slot].sz == sz;
    D.vb[slot].b = b; D.vb[slot].off = off; D.vb[slot].sz = sz; D.vbv |= 1u << slot; g_sh_n[same ? 7 : 6]++; return same;
}
static int vi_same(const VkVertexInputBindingDescription2EXT *vb, int nvb, const VkVertexInputAttributeDescription2EXT *va, int nva, int sh) {   // as vb_same, for the vertex input state
    u4 kb[17], ka[32];   // compact keys: the Vulkan descriptions have padding
    for (int i = 0; i < nvb; i++) kb[i] = (u4){{vb[i].binding, vb[i].stride, (uint32_t)vb[i].inputRate, vb[i].divisor}};
    for (int i = 0; i < nva; i++) ka[i] = (u4){{va[i].location, va[i].binding, (uint32_t)va[i].format, va[i].offset}};
    int same = sh && D.vi_ok && D.nvb == (uint32_t)nvb && D.nva == (uint32_t)nva && !memcmp(kb, D.kb, nvb * sizeof *kb) && !memcmp(ka, D.ka, nva * sizeof *ka);
    if (!same) { memcpy(D.kb, kb, nvb * sizeof *kb); memcpy(D.ka, ka, nva * sizeof *ka); D.nvb = nvb; D.nva = nva; D.vi_ok = 1; }
    g_sh_n[same ? 5 : 4]++; return same;
}
// DM_VK_DRAWCHECK=1: CPU-side sanity check of every draw's arguments against the memory it will read (diagnostic for GPU hangs: a draw with garbage counts or indices runs
// until the driver's watchdog). Logs draws with huge counts, an index range past its pool, indices past a vertex stream's bound size, indirect arguments (as the CPU sees them
// at record time; the GPU may write them later). Indices are scanned on the CPU: slow. sub = the submission number the stall lines print
// =2: only the index scan (reads the index data, no checks). It and the checker made the stalls rare as a side effect, as does the pacing below
static int drawcheck_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_DRAWCHECK"); on = e ? atoi(e) : 0;
    if (on) LOG("vk: DM_VK_DRAWCHECK=%d: %s (diagnostic, slow)", on, on == 2 ? "index data scanned on the CPU, no checks" : "draw arguments are checked on the CPU"); } return on; }
// Draw pacing (DM_VK_DRAW_PACE_NS, default 0 = off): a busy wait of that many ns before each draw is recorded. It was the mitigation of the intermittent ~5 s GPU hang (1000 ns: ~1 per 15k frames
// instead of ~1 per 2.4k) until the cause was found: the pool lookup race (pool_get). It only narrowed the race window. With the fix, no pacing: 0 hangs in 9.3k and 7.5k frames, hot
static int g_pace_ns = -1;
static volatile uint32_t g_scan_sink;
static void drawcheck_scan(const p_draw_idx *di) {
    if (!di || di->ind || di->count <= 0 || di->count > (1 << 22)) return;
    int isz = di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4; const uint8_t *src = (const uint8_t *)(uintptr_t)di->ibuf; uint32_t mx = 0, n = (uint32_t)di->count;
    for (uint32_t i = 0; i < n; i++) { uint32_t v = isz == 1 ? src[i] : isz == 2 ? ((const uint16_t *)src)[i] : ((const uint32_t *)src)[i]; if (v > mx) mx = v; }
    g_scan_sink += mx;
}
static void draw_pace(void) {
    if (g_pace_ns < 0) { const char *e = getenv("DM_VK_DRAW_PACE_NS"); g_pace_ns = e ? atoi(e) : 0; LOG("vk: draw pacing %d ns per draw (DM_VK_DRAW_PACE_NS, 0 = off)", g_pace_ns); }
    if (g_pace_ns > 0) { uint64_t t = now_ns() + (uint64_t)g_pace_ns; while (now_ns() < t) { } }
}
static void drawcheck(const p_draw *dw, const p_draw_idx *di, uint32_t inst, uint32_t used, const vshader *vs, const vshader *fs) {
    static uint64_t pass_seen; static uint32_t nd; static int nlog; static uint64_t nbad, nchk;
    if (pass_seen != g_stat_passes) { pass_seen = g_stat_passes; nd = 0; }
    uint32_t d = nd++; nchk++;
    char why[200]; int w = 0; why[0] = 0;
    int64_t lo = 0, hi = -1;   // vertex index range the draw fetches (before instancing)
    uint32_t ind[5] = {0}; int has_ind = 0;
    if (di) {
        int isz = di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4;
        if (di->ind) { if (pool_live(di->ind, 20)) { memcpy(ind, (const void *)(uintptr_t)di->ind, 20); has_ind = 1; } else w += snprintf(why + w, sizeof why - w, " indirect-args-not-in-pool"); }
        uint32_t count = has_ind ? ind[0] : (uint32_t)di->count, first = has_ind ? ind[2] : 0; int32_t bv = has_ind ? (int32_t)ind[3] : di->base_vertex;
        if (has_ind) inst = ind[1] ? ind[1] : 0;
        if (count > (1u << 22)) w += snprintf(why + w, sizeof why - w, " count=%u", count);
        else if (!pool_live(di->ibuf + (uint64_t)first * isz, (uint64_t)count * isz + 1)) w += snprintf(why + w, sizeof why - w, " index-range-past-pool(first %u count %u size %d)", first, count, isz);
        else if (count) {
            const uint8_t *src = (const uint8_t *)(uintptr_t)di->ibuf + (uint64_t)first * isz; uint32_t mx = 0, mn = ~0u;
            for (uint32_t i = 0; i < count; i++) { uint32_t v = isz == 1 ? src[i] : isz == 2 ? ((const uint16_t *)src)[i] : ((const uint32_t *)src)[i]; if (v > mx) mx = v; if (v < mn) mn = v; }
            lo = (int64_t)mn + bv; hi = (int64_t)mx + bv;
        }
    } else if (dw) {
        if (dw->ind) { if (pool_live(dw->ind, 16)) { memcpy(ind, (const void *)(uintptr_t)dw->ind, 16); has_ind = 1; } else w += snprintf(why + w, sizeof why - w, " indirect-args-not-in-pool"); }
        uint32_t count = has_ind ? ind[0] : (uint32_t)dw->count, first = has_ind ? ind[2] : (uint32_t)dw->first;
        if (has_ind) inst = ind[1];
        if (count > (1u << 22)) w += snprintf(why + w, sizeof why - w, " count=%u", count);
        else if (count) { lo = first; hi = (int64_t)first + count - 1; }
    }
    if (inst > (1u << 20)) w += snprintf(why + w, sizeof why - w, " instances=%u", inst);
    if (lo < 0 && hi >= 0) w += snprintf(why + w, sizeof why - w, " negative-vertex(%lld)", (long long)lo);
    for (int st = 0; st < 16 && hi >= 0; st++) if (used >> st & 1) {
        uint32_t stride = S.stream[st].stride, div = S.stream[st].divisor; uint64_t sz = S.vbuf[st].size;
        if (!stride || !sz) continue;
        uint64_t nv = sz / stride, need = div ? (uint64_t)g_draw_binst + (inst ? (inst - 1) / div : 0) : (uint64_t)hi;
        if (need >= nv && w < 150) w += snprintf(why + w, sizeof why - w, " stream%d-%s-%llu-of-%llu", st, div ? "instance" : "vertex", (unsigned long long)need, (unsigned long long)nv);
    }
    if (!w) return;
    nbad++;
    { static char seen[64][20]; static int nseen; int inst_only = strstr(why, "-instance-") && !strstr(why, "count=") && !strstr(why, "past-pool") && !strstr(why, "-vertex-") && !strstr(why, "negative") && !strstr(why, "indirect-args");
      if (inst_only) { for (int i = 0; i < nseen; i++) if (!memcmp(seen[i], vs->key, 16)) return; if (nseen < 64) memcpy(seen[nseen++], vs->key, 16); } }
    if (nlog++ < 2000) LOG("[RED] vk: drawcheck: sub %llu pass %llu %ux%u draw %u (vs %.16s fs %.16s)%s%s; %s count %d inst %u bv %d range %lld..%lld (%llu bad of %llu)",
        (unsigned long long)(g_fence + 1), (unsigned long long)g_stat_passes, R.w, R.h, d, vs->key, fs ? fs->key : "-", has_ind ? " indirect" : "", why,
        di ? "indexed" : "arrays", di ? di->count : dw->count, inst, di ? di->base_vertex : 0, (long long)lo, (long long)hi, (unsigned long long)nbad, (unsigned long long)nchk);
}
static int shader_may_store(const vshader *sh) { for (int i = 0; sh && i < sh->nb; i++) if (sh->b[i].type == 'b' || sh->b[i].type == 'i') return 1; return 0; }
static void draw(const p_draw *dw, const p_draw_idx *di) {
    g_draw_binst = dw ? dw->base_instance : di->base_instance;
    if ((dw && dw->mdi_cnt) || (di && di->mdi_cnt)) { RED_ONCE(11, "vk: indirect-count multi-draws are not implemented"); drop(1); return; }
    vprog *pv = prog_get(S.bst[0]);
    vshader *vs = pv ? pv->sh[0] : NULL;
    // Nothing reaches the rasterizer: only the stores of the vertex stage are left to do (30 of Pokopia's vertex shaders use storage buffers). Transform feedback capture is not implemented
    int discard = S.rast_discard || S.poly.cull == NVN_FACE_FRONT_AND_BACK;
    if (discard && !shader_may_store(vs)) { if (S.rast_discard) drop(2); return; }
    vprog *pf = S.bst[1] && !discard ? prog_get(S.bst[1]) : NULL;
    vshader *fs = pf ? pf->sh[1] : NULL;
    if (!vs) { drop(3); return; }
    vshader *other[5] = {0};   // geometry, hull, domain (NVN stages 2, 3, 4)
    for (int st = 2; st < 5; st++) if (S.bst[st]) { vprog *po = S.bst[st] == S.bst[0] ? pv : prog_get(S.bst[st]); other[st] = po ? po->sh[st] : NULL; }
    vshader *gs = other[2], *hs = other[3], *ds = other[4];
    int prim = dw ? dw->prim : di->prim;
    if (prim == 14 && (!hs || !ds)) { RED_ONCE(73, "vk: a patch draw without both a hull and a domain shader (hull %s domain %s): skipped", hs ? hs->key : "-", ds ? ds->key : "-"); drop(5); return; }
    if (prim != 14 && (hs || ds)) { RED_ONCE(74, "vk: primitive %d with a hull or domain shader bound (hull %s domain %s): drawn without them", prim, hs ? hs->key : "-", ds ? ds->key : "-"); hs = ds = NULL; }
    if (prim == 14 && (S.patch < 1 || (uint32_t)S.patch > g_lim.maxTessellationPatchSize)) { RED_ONCE(75, "vk: patch size %d is over this GPU's limit %u: skipped", S.patch, g_lim.maxTessellationPatchSize); drop(5); return; }
    int layer_out = vs->layer_out || (gs && gs->layer_out) || (ds && ds->layer_out);
    if (S.color.logic_op != NVN_LOGIC_OP_COPY) RED_ONCE(53, "vk: logic op %d is not applied (only COPY is drawn)", (int)S.color.logic_op);
    if (S.color.alpha_func != NVN_ALPHA_FUNC_ALWAYS) RED_ONCE(54, "vk: alpha test function %d is not applied (the host has no alpha test)", (int)S.color.alpha_func);
    if (S.has_vp && !discard && (!(S.vp[0][2] > 0.f) || S.vp[0][3] == 0.f)) { RED_ONCE(55,"vk: draw skipped: viewport %g x %g (Vulkan needs a width above 0 and a height other than 0)", S.vp[0][2], S.vp[0][3]); drop(15); return; }
    if (R.active && R.layered != layer_out) rp_end();
    g_rp_layered = layer_out; g_rp_empty = discard;
    int ok_rt = rt_begin(); g_rp_layered = g_rp_empty = 0;
    if (!ok_rt) { drop(4); return; }
    VkRect2D sc = clip_rect(); if (!sc.extent.width) { if (!discard) { g_tr_skip++; return; } sc = (VkRect2D){{0, 0}, {1, 1}}; }
    VkPrimitiveTopology tp; int quads = 0;
    switch (prim) {
        case 0: tp = VK_PRIMITIVE_TOPOLOGY_POINT_LIST; break; case 1: tp = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; break;
        case 2: RED_ONCE(12, "vk: line loops drawn as line strips"); tp = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break; case 3: tp = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP; break;
        case 4: tp = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; break; case 5: case 8: tp = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP; break; case 6: case 9: tp = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN; break;
        case 7: tp = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; quads = 1; break;
        case 10: tp = VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY; break; case 11: tp = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY; break;
        case 12: tp = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY; break; case 13: tp = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY; break;
        case 14: tp = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST; break;   // NVN's numbers are OpenGL's (GL_PATCHES)
        default: RED_ONCE(13, "vk: primitive type %d not implemented", prim); drop(5); return;
    }
    int tclass = tp == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST ? 3 : tp == VK_PRIMITIVE_TOPOLOGY_POINT_LIST ? 0 : tp <= VK_PRIMITIVE_TOPOLOGY_LINE_STRIP || tp == VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY || tp == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY ? 1 : 2;
    pkey k; memset(&k, 0, sizeof k);
    k.vs = vs; k.fs = fs; k.gs = gs; k.hs = hs; k.ds = ds; k.patch = hs ? (uint8_t)S.patch : 0; memcpy(k.cf, R.cf, sizeof k.cf); k.df = R.df; k.sf = R.sf; k.nc = (uint8_t)R.nc; k.samples = R.samples ? R.samples : 1; k.topo = (uint8_t)tclass; k.dz = (uint8_t)atomic_load(&g_depth_zero);
    pent *pe = pipeline_get(&k);
    if (!pe->pl) { drop(6); return; }
    VkCommandBuffer cb = g_cur->cb; g_since_flush++;
    int sh = shadow(1) != 0;   // record only what differs from the shadow D (see State shadowing)
    if (!sh || pe->pl != D.pl) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pe->pl); D.pl = pe->pl; g_sh_n[2]++; if (sh && (g_sh & 8)) { D.ok = D.vi_ok = 0; D.vbv = 0; } } else g_sh_n[3]++;
    int use = sh && D.ok;   // the shadow holds the whole dynamic state
    // dynamic state
    float vx = 0, vy = 0, vw = (float)R.w, vh = (float)R.h;
    if (S.has_vp && !discard) { vx = S.vp[0][0]; vy = S.vp[0][1]; vw = S.vp[0][2]; vh = S.vp[0][3]; }
    // Viewport swizzle (OP_VP_SWIZZLE, viewport 0): NEGATIVE_Y negates clip y, the same as the viewport flipped about its centre, and facing is decided after the viewport transform in Vulkan
    // as in Metal, so the front-face winding stays (nvn_mtl.m). Other swizzles (axis exchanges, negated x / z / w) are not expressible without VK_NV_viewport_swizzle (not used): drawn unswizzled
    int swy = 0;
    { uint32_t sw; memcpy(&sw, S.vp_sw[0], 4); swy = sw == 0x100u;
      if (sw && !swy) RED_ONCE(70, "vk: viewport swizzle %d,%d,%d,%d is not implemented (drawn unswizzled; only NEGATIVE_Y is)", S.vp_sw[0][0], S.vp_sw[0][1] ^ 2, S.vp_sw[0][2] ^ 4, S.vp_sw[0][3] ^ 6); }
    VkViewport vp = {vx, vy + vh, vw, -vh, S.dr[0], S.dr[1]};
    if ((atomic_load(&g_origin_ll) != 0) != swy || dbg(4)) vp = (VkViewport){vx, vy, vw, vh, S.dr[0], S.dr[1]};
    if (vp.minDepth < 0 || vp.minDepth > 1 || vp.maxDepth < 0 || vp.maxDepth > 1) { vp.minDepth = MIN(MAX(vp.minDepth, 0.f), 1.f); vp.maxDepth = MIN(MAX(vp.maxDepth, 0.f), 1.f); }
    f4 bc = {{S.blend_color[0], S.blend_color[1], S.blend_color[2], S.blend_color[3]}};
    DYM(view, vp, vkCmdSetViewport(cb, 0, 1, &n_)); DYM(scis, sc, vkCmdSetScissor(cb, 0, 1, &n_)); DY(k_lw, 1u, vkCmdSetLineWidth(cb, 1.0f));
    DYM(bcol, bc, vkCmdSetBlendConstants(cb, n_.a));
    DY(cull, dbg(2) ? VK_CULL_MODE_NONE : S.poly.cull == NVN_FACE_FRONT ? VK_CULL_MODE_FRONT_BIT : S.poly.cull == NVN_FACE_BACK ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE, pvkCmdSetCullMode(cb, n_));
    DY(front, (S.poly.front == NVN_FRONT_FACE_CCW) != (atomic_load(&g_queue_ll) != 0) ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE, pvkCmdSetFrontFace(cb, (VkFrontFace)n_));   // the queue's raster flip, not the device's current origin (nvn_mtl.m, Xenoblade 2)
    DY(poly, S.poly.mode == NVN_POLYGON_MODE_LINE ? VK_POLYGON_MODE_LINE : S.poly.mode == NVN_POLYGON_MODE_POINT ? VK_POLYGON_MODE_POINT : VK_POLYGON_MODE_FILL, pvkCmdSetPolygonModeEXT(cb, (VkPolygonMode)n_));
    DY(topo, tp, pvkCmdSetPrimitiveTopology(cb, (VkPrimitiveTopology)n_));
    // Primitive restart (OP_PRIM_RESTART): Vulkan's restart index is fixed at the all-ones value of the index type, and only strips and fans (without VK_EXT_primitive_topology_list_restart) restart;
    // any other index / topology is not applied (logged once). Indexed quads are expanded on the CPU below, which does honour the index
    int pr_on = 0;
    if (di && S.restart.on && !quads) {
        uint32_t full = di->itype == 0 ? 0xffu : di->itype == 1 ? 0xffffu : 0xffffffffu;
        if (S.restart.index == full && (tp == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP || tp == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || tp == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN)) pr_on = 1;
        else RED_ONCE(71, "vk: primitive restart with index %#x on primitive %d (index type %d) is not applied: Vulkan restarts only on the all-ones index of strips and fans", S.restart.index, prim, di->itype);
    }
    DY(k_pr, pr_on ? 2u : 1u, pvkCmdSetPrimitiveRestartEnable(cb, n_ == 2u ? VK_TRUE : VK_FALSE));
    DY(k_rd, discard ? 2u : 1u, pvkCmdSetRasterizerDiscardEnable(cb, n_ == 2u ? VK_TRUE : VK_FALSE));
    int bias = (S.poly.offset_en & 4) != 0; f4 bv = {{bias ? S.po.units : 0, bias ? S.po.clamp : 0, bias ? S.po.factor : 0, 0}};
    DY(bias_en, bias, pvkCmdSetDepthBiasEnable(cb, n_)); DYM(bias_v, bv, vkCmdSetDepthBias(cb, n_.a[0], n_.a[1], n_.a[2]));
    DY(dclamp, S.depth_clamp != 0, pvkCmdSetDepthClampEnableEXT(cb, n_));
    DY(dt, R.df && S.ds.dt && !dbg(1), pvkCmdSetDepthTestEnable(cb, n_)); DY(dw, R.df && S.ds.dw, pvkCmdSetDepthWriteEnable(cb, n_)); DY(dcmp, cmpop(S.ds.df), pvkCmdSetDepthCompareOp(cb, (VkCompareOp)n_));
    DY(k_db, 1u, pvkCmdSetDepthBoundsTestEnable(cb, VK_FALSE));
    DY(st, R.sf && S.ds.st && !dbg(1), pvkCmdSetStencilTestEnable(cb, n_));
    u4 sof = {{stop(S.ds.fail_f), stop(S.ds.zpass_f), stop(S.ds.zfail_f), cmpop(S.ds.sf_f)}}, sob = {{stop(S.ds.fail_b), stop(S.ds.zpass_b), stop(S.ds.zfail_b), cmpop(S.ds.sf_b)}};
    DYM(sof, sof, pvkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, (VkStencilOp)n_.a[0], (VkStencilOp)n_.a[1], (VkStencilOp)n_.a[2], (VkCompareOp)n_.a[3]));
    DYM(sob, sob, pvkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, (VkStencilOp)n_.a[0], (VkStencilOp)n_.a[1], (VkStencilOp)n_.a[2], (VkCompareOp)n_.a[3]));
    DY(vmask[0], S.st_vmask[0], vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_BIT, n_)); DY(vmask[1], S.st_vmask[1], vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_BACK_BIT, n_));
    DY(wmask[0], S.st_wmask[0], vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_BIT, n_)); DY(wmask[1], S.st_wmask[1], vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_BACK_BIT, n_));
    DY(sref[0], S.st_ref[0], vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_BIT, n_)); DY(sref[1], S.st_ref[1], vkCmdSetStencilReference(cb, VK_STENCIL_FACE_BACK_BIT, n_));
    DY(a2c, S.ms.a2c != 0, pvkCmdSetAlphaToCoverageEnableEXT(cb, n_));
    if (R.nc) {
        VkBool32 be[8]; VkColorBlendEquationEXT eq[8]; VkColorComponentFlags wm[8];
        for (int i = 0; i < R.nc; i++) {
            const blend_st bs_ = adv_blend(S.blend[i]), *b = &bs_;   // an advanced blend mode replaces the functions (VK_EXT_blend_operation_advanced is not used)
            be[i] = (S.color.blend_en >> i) & 1;
            eq[i] = (VkColorBlendEquationEXT){bfac_d(b->src_rgb, VK_BLEND_FACTOR_ONE), bfac_d(b->dst_rgb, VK_BLEND_FACTOR_ZERO), bop(b->eq_rgb), bfac_d(b->src_a, VK_BLEND_FACTOR_ONE), bfac_d(b->dst_a, VK_BLEND_FACTOR_ZERO), bop(b->eq_a)};
            wm[i] = (S.chmask.mask >> (4 * i)) & 0xf;
            if (dbg(8)) { be[i] = 0; wm[i] = 0xf; }
        }
        int chg = !use || (uint32_t)R.nc > D.nbl;   // the shadow knows the first nbl attachments
        for (int i = 0; i < R.nc && !chg; i++) chg = D.be[i] != be[i] || D.wm[i] != wm[i] || memcmp(&D.eq[i], &eq[i], sizeof eq[i]);
        if (chg) {
            pvkCmdSetColorBlendEnableEXT(cb, 0, R.nc, be); pvkCmdSetColorBlendEquationEXT(cb, 0, R.nc, eq); pvkCmdSetColorWriteMaskEXT(cb, 0, R.nc, wm);
            for (int i = 0; i < R.nc; i++) { D.be[i] = be[i]; D.eq[i] = eq[i]; D.wm[i] = wm[i]; }
            D.nbl = MAX(D.nbl, (uint32_t)R.nc); g_sh_n[0]++;
        } else g_sh_n[1]++;
    }
    D.ok = sh;
#undef DY
#undef DYM
    { static char want[40]; static int init; if (!init) { init = 1; const char *e = getenv("DM_VK_TRACE_FS"); if (e) snprintf(want, sizeof want, "%s", e); }
      g_trfs_now = g_trace && want[0] && fs && g_trfs_cnt < 2 && !strcmp(fs->key, want); if (g_trfs_now) { g_trfs_cnt++; LOG("vk: T   vertex inputs and uniform buffers of the next draw (vs %s fs %s)", vs->key, fs->key); } }
    // vertex input
    VkVertexInputBindingDescription2EXT vb[17]; VkVertexInputAttributeDescription2EXT va[32]; int nvb = 0, nva = 0; uint32_t used = 0;
    for (int i = 0; i < vs->nin; i++) {
        int loc = vs->in[i].loc; char ct = vs->in[i].ct;
        va_st a; uint32_t raw = loc < 16 ? S.attr[loc] : 0; memcpy(&a, &raw, 4);
        VkFormat f = a.fmt ? vertex_format(a.fmt, ct) : VK_FORMAT_UNDEFINED;
        int s = a.stream; VkBuffer b; VkDeviceSize off;
        if (g_trfs_now) {
            uint64_t ad = s < 16 ? S.vbuf[s].address : 0; uint32_t st = s < 16 ? S.stream[s].stride : 0, w[8] = {0};
            if (ad) for (int j = 0; j < 2; j++) memcpy(&w[4 * j], (const void *)(uintptr_t)(ad + (uint64_t)j * st + a.offset), 16);
            LOG("vk: T     in loc %d (of %d bound) type %c nvn fmt %#x vk %d stream %d offset %u stride %u addr %#llx size %#llx divisor %u; vertex 0: %08x %08x %08x %08x; vertex 1: %08x %08x %08x %08x",
                loc, S.nattr, ct, a.fmt, (int)f, s, (unsigned)a.offset, st, (unsigned long long)ad, s < 16 ? (unsigned long long)S.vbuf[s].size : 0ull, s < 16 ? S.stream[s].divisor : 0u, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        }
        if (f && !(fmt_props(f).bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) { RED_ONCE(14, "vk: vertex format %d (NVN %#x) is not a vertex format on this GPU", (int)f, a.fmt); f = VK_FORMAT_UNDEFINED; }
        uint64_t psz = 0;
        if (!f || s >= 16 || !S.vbuf[s].address || !buf_at_w(S.vbuf[s].address, S.vbuf[s].size ? S.vbuf[s].size : POOL_NOLEN, &b, &off, &psz)) {
            va[nva++] = (VkVertexInputAttributeDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, NULL, (uint32_t)loc, 16, ct == 'f' ? VK_FORMAT_R32G32B32A32_SFLOAT : ct == 'i' ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_UINT, 0};
            used |= 1u << 16; continue;
        }
        va[nva++] = (VkVertexInputAttributeDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, NULL, (uint32_t)loc, (uint32_t)s, f, a.offset};
        if (!(used >> s & 1)) {
            used |= 1u << s;
            uint32_t div = S.stream[s].divisor;
            vb[nvb++] = (VkVertexInputBindingDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, NULL, (uint32_t)s, S.stream[s].stride, div ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX, div > 1 && g_has_divisor ? div : 1};
            if (div > 1 && !g_has_divisor) RED_ONCE(15, "vk: instance divisor %u without VK_EXT_vertex_attribute_divisor", div);
            VkDeviceSize sz = S.vbuf[s].size ? S.vbuf[s].size : VK_WHOLE_SIZE; if (sz != VK_WHOLE_SIZE) sz = MIN(sz, psz - off);
            if (!vb_same(s, b, off, sz, sh)) pvkCmdBindVertexBuffers2(cb, (uint32_t)s, 1, &b, &off, &sz, NULL);
        }
    }
    if (used >> 16 & 1) { vb[nvb++] = (VkVertexInputBindingDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, NULL, 16, 0, VK_VERTEX_INPUT_RATE_VERTEX, 1};
        VkDeviceSize zo = 0; if (!vb_same(16, g_zero_buf, 0, VK_WHOLE_SIZE, sh)) pvkCmdBindVertexBuffers2(cb, 16, 1, &g_zero_buf, &zo, NULL, NULL); }   // sizes NULL: the whole buffer
    if (!vi_same(vb, nvb, va, nva, sh)) pvkCmdSetVertexInputEXT(cb, (uint32_t)nvb, vb, (uint32_t)nva, va);
    // resources
    push_stage(pe->lay, vs, 0);
    if (fs) push_stage(pe->lay, fs, 1);
    if (gs) push_stage(pe->lay, gs, 2);
    if (hs) push_stage(pe->lay, hs, 3);
    if (ds) push_stage(pe->lay, ds, 4);
    g_trfs_now = 0;
    { static char skip[512]; static int init; if (!init) { init = 1; const char *e = getenv("DM_VK_SKIPFS"); if (e) { snprintf(skip, sizeof skip, ",%s,", e); LOG("vk: DM_VK_SKIPFS=%s: draws with these fragment shaders are skipped (diagnostic)", e); } }
      if (skip[0] && fs) { char k[64]; snprintf(k, sizeof k, ",%s,", fs->key); if (strstr(skip, k)) return; } }
    { static int sw = -1, shh; if (sw < 0) { const char *e = getenv("DM_VK_SKIPPASS"); sw = 0; if (e && sscanf(e, "%dx%d", &sw, &shh) == 2) LOG("vk: DM_VK_SKIPPASS=%dx%d: draws into passes of this size are skipped (diagnostic)", sw, shh); else sw = 0; }
      if (sw && (int)R.w == sw && (int)R.h == shh) return; }
    // the draw
    uint32_t inst = (uint32_t)((dw ? dw->instances : di->instances) ? (dw ? dw->instances : di->instances) : 1);
    { int dc = drawcheck_on(); if (dc == 1) drawcheck(dw, di, inst, used, vs, fs); else if (dc == 2) drawcheck_scan(di); }
    draw_pace();
    if (dw) {
        if (dw->ind) { VkBuffer b; VkDeviceSize off; if (!buf_at(dw->ind, 16, &b, &off)) { drop(7); return; }
            if (off & 3) { RED_ONCE(57, "vk: indirect draw skipped: argument offset %#llx is not a multiple of 4", (unsigned long long)off); drop(7); return; }
            if (quads) { RED_ONCE(16, "vk: indirect quads"); drop(8); return; } vkCmdDrawIndirect(cb, b, off, 1, 16); }
        else if (quads) {
            uint32_t nq = (uint32_t)dw->count / 4; if (nq > 65536) { RED_ONCE(17, "vk: more than 65536 quads in one draw"); nq = 65536; }
            vkCmdBindIndexBuffer(cb, g_quad_ib.buf, g_quad_ib.off, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cb, nq * 6, inst, 0, dw->first, (uint32_t)dw->base_instance);
        } else vkCmdDraw(cb, (uint32_t)dw->count, inst, (uint32_t)dw->first, (uint32_t)dw->base_instance);
    } else {
        int isz = di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4;
        VkIndexType it = di->itype == 0 ? VK_INDEX_TYPE_UINT8_EXT : di->itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        VkBuffer ib; VkDeviceSize ioff;
        if (!buf_at_w(di->ibuf, di->ind ? (uint64_t)32u << 20 : (uint64_t)(uint32_t)di->count * isz, &ib, &ioff, NULL)) { drop(9); return; }   // indirect: the count is the GPU's, so room for far more than any index buffer seen
        if (quads) {   // indexed quads: converted on the CPU into a triangle list
            if (di->ind) { RED_ONCE(18, "vk: indirect indexed quads"); drop(8); return; }
            uint32_t nq = (uint32_t)di->count / 4;
            int rst = S.restart.on != 0;   // primitive restart: the restart index ends a quad and the next starts after it (nvn_expand_primitive), so every index is read, not nq * 4
            uint64_t nread = rst ? (uint64_t)(uint32_t)di->count : (uint64_t)nq * 4;
            if (!pool_live(di->ibuf, nread * isz)) { RED_ONCE(58, "vk: indexed quads skipped: %llu indices of %d bytes from %#llx are not inside one memory pool", (unsigned long long)nread, isz, (unsigned long long)di->ibuf); drop(9); return; }   // the CPU conversion below reads them all
            halloc h = ring_alloc((uint64_t)nq * 6 * 4 + 4, 4); uint32_t *o = (uint32_t *)h.p; const uint8_t *src = (const uint8_t *)(uintptr_t)di->ibuf;
            uint32_t ni = nq * 6;
            if (rst) ni = nvn_expand_primitive(o, src, (uint32_t)di->count, (unsigned)isz, 7u, 1, S.restart.index);
            else for (uint32_t q = 0; q < nq; q++) { uint32_t x[4]; for (int j = 0; j < 4; j++) { uint32_t v = 0; memcpy(&v, src + (q * 4 + j) * isz, isz); x[j] = v; }
                o[6 * q] = x[0]; o[6 * q + 1] = x[1]; o[6 * q + 2] = x[2]; o[6 * q + 3] = x[0]; o[6 * q + 4] = x[2]; o[6 * q + 5] = x[3]; }
            vkCmdBindIndexBuffer(cb, h.buf, h.off, VK_INDEX_TYPE_UINT32);
            if (ni) vkCmdDrawIndexed(cb, ni, inst, 0, di->base_vertex, (uint32_t)di->base_instance);
        } else {
            if (it == VK_INDEX_TYPE_UINT8_EXT && !g_has_u8) { RED_ONCE(19, "vk: 8-bit indices without VK_EXT_index_type_uint8"); drop(10); return; }
            if (ioff % isz) { RED_ONCE(59, "vk: indexed draw skipped: index buffer offset %#llx is not a multiple of the index size %d", (unsigned long long)ioff, isz); drop(9); return; }   // vkCmdBindIndexBuffer
            vkCmdBindIndexBuffer(cb, ib, ioff, it);
            if (di->ind) { VkBuffer b; VkDeviceSize off; if (!buf_at(di->ind, 20, &b, &off)) { drop(7); return; }
                if (off & 3) { RED_ONCE(60, "vk: indirect indexed draw skipped: argument offset %#llx is not a multiple of 4", (unsigned long long)off); drop(7); return; }
                vkCmdDrawIndexedIndirect(cb, b, off, 1, 20); }
            else vkCmdDrawIndexed(cb, (uint32_t)di->count, inst, 0, di->base_vertex, (uint32_t)di->base_instance);
        }
    }
    g_stat_draws++; if (g_pp.on) pp_pass_draw(vs, fs);
    if (discard) rp_end();   // its barrier makes the stores visible to what reads them next, often the vertex stage of the following draw
    if (g_trace && (++g_tr_draws <= 6 || g_tr_full)) {
        const p_draw_idx *q = di; LOG("vk: T   draw #%u vs %s fs %s prim %d count %d inst %u%s vp %.0f,%.0f %.0fx%.0f dr %.2f-%.2f sc %d,%d %ux%u cull %d front %d dt %d dw %d df %d blend %#x mask %#x",
            g_tr_draws, vs->key, fs ? fs->key : "-", prim, dw ? dw->count : q->count, inst, dw ? "" : " indexed", vx, vy, vw, vh, S.dr[0], S.dr[1], sc.offset.x, sc.offset.y, sc.extent.width, sc.extent.height,
            (int)S.poly.cull, (int)S.poly.front, (int)S.ds.dt, (int)S.ds.dw, (int)S.ds.df, (unsigned)S.color.blend_en, S.chmask.mask);
    }
}

// ---------- Compute ----------
// OP_DISPATCH (hle_nvn.c nvnCommandBufferDispatchCompute / Indirect: groups x y z, or ind = the address of three u32 counts the GPU reads) runs the program bound to stage 5 (S.bst[5], BindProgram stages & 0x20).
// As nvn_mtl.m dispatch(): the compute shader reads its group counts and uniform buffer addresses from the driver segment c0 (+0x200, +0x210 + 16 r; push_stage K_DRV0), and its storage
// buffers from c0 + 0x310 + 16 r (sb) or on the uniform slots (ubp, android-host/tools/pokopia_vkshaders.py). Outside a render pass: the pass ends first; a full barrier follows (compute -> graphics / compute / copies)
typedef struct { vshader *cs; VkPipeline pl; VkPipelineLayout lay; } cspent;
static cspent *g_cspl; static int g_ncspl, g_ccspl;
static uint64_t g_stat_disp, g_stat_disp_ind;
static cspent *cpipeline_get(vshader *cs) {
    for (int i = 0; i < g_ncspl; i++) if (g_cspl[i].cs == cs) return &g_cspl[i];
    VkPipelineLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &cs->dsl};
    VkPipelineLayout lay; VKCHK(vkCreatePipelineLayout(g_dev, &lci, NULL, &lay));
    // Compute runs a warp's 32 lanes where the device lets the pipeline choose: the translator's loops over lanes (REDUX) name lanes 0 to 31
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sub32 = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = 32};
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, g_cs_sub32 ? &sub32 : NULL, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs->mod, .pName = "main"}, .layout = lay};
    VkPipeline pl; uint64_t t0 = now_ns();
    VkResult r = vkCreateComputePipelines(g_dev, g_pcache, 1, &ci, NULL, &pl); g_pc_new++;
    if (r) { LOG("[RED] vk: compute pipeline %s: %d (the shader declares GroupNonUniform capabilities: check the device's subgroup operations)", cs->key, (int)r); pl = VK_NULL_HANDLE; }
    { uint64_t ms_ = (now_ns() - t0) / 1000000; if (ms_ > 200) LOG("vk: compute pipeline %s took %llu ms", cs->key, (unsigned long long)ms_); }
    g_stat_pso++;
    if (g_ncspl == g_ccspl) { g_ccspl = g_ccspl ? 2 * g_ccspl : 32; g_cspl = realloc(g_cspl, g_ccspl * sizeof *g_cspl); }
    g_cspl[g_ncspl] = (cspent){cs, pl, lay};
    return &g_cspl[g_ncspl++];
}
static void dispatch(const p_dispatch *dp) {
    { static int off = -1; if (off < 0) { const char *e = getenv("DM_VK_NO_COMPUTE"); off = e && atoi(e); if (off) LOG("vk: DM_VK_NO_COMPUTE: compute dispatches are skipped (diagnostic)"); } if (off) return; }
    if (!S.bst[5]) { RED_ONCE(35, "vk: no program bound to the compute stage before the dispatch"); drop(11); return; }
    vprog *pc = prog_get(S.bst[5]); vshader *cs = pc ? pc->sh[5] : NULL;
    if (!cs) { RED_ONCE(39, "vk: the dispatched program has no compute shader in the shader table (or it is not mapped)"); drop(12); return; }
    VkBuffer ib = VK_NULL_HANDLE; VkDeviceSize io = 0;
    if (dp->ind) {   // DispatchComputeIndirect: three u32 group counts at ind, usually written by the GPU earlier in the frame
        if (!buf_at(dp->ind, 12, &ib, &io)) { RED_ONCE(40, "vk: indirect dispatch skipped: argument address %#llx is in no imported pool", (unsigned long long)dp->ind); drop(13); return; }
        if (io & 3) { RED_ONCE(41, "vk: indirect dispatch skipped: argument offset %#llx is not 4-byte aligned", (unsigned long long)io); drop(13); return; }
    } else {
        if (dp->x <= 0 || dp->y <= 0 || dp->z <= 0) return;
        if ((uint32_t)dp->x > g_lim.maxComputeWorkGroupCount[0] || (uint32_t)dp->y > g_lim.maxComputeWorkGroupCount[1] || (uint32_t)dp->z > g_lim.maxComputeWorkGroupCount[2]) {
            RED_ONCE(42, "vk: dispatch of %d x %d x %d groups is over the device limit: skipped", dp->x, dp->y, dp->z); drop(13); return; }
    }
    cspent *pe = cpipeline_get(cs);
    if (!pe->pl) { drop(14); return; }
    rp_end(); PP(PK_DISP);
    if (g_pp.on && g_pp.n > 0) { pp_rec *r = &g_pp.rec[g_pp.q][g_pp.n - 1]; memcpy(r->vs, cs->key, sizeof r->vs); snprintf(r->fs, sizeof r->fs, "%s", dp->ind ? "indirect" : "direct"); }
    VkCommandBuffer cb = g_cur->cb;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pe->pl);
    g_disp_dim[0] = dp->ind ? 0 : (uint32_t)dp->x; g_disp_dim[1] = dp->ind ? 0 : (uint32_t)dp->y; g_disp_dim[2] = dp->ind ? 0 : (uint32_t)dp->z;
    memset(&g_c0_dim, 0, sizeof g_c0_dim); g_push_fallback = 0;
    push_stage(pe->lay, cs, 5);
    { static char seen[64][20]; static int ns; int i; for (i = 0; i < ns; i++) if (!strcmp(seen[i], cs->key)) break;
      if (i == ns && ns < 64) {   // first dispatch of each compute shader: for checking on the phone
          snprintf(seen[ns++], 20, "%s", cs->key); module_t *m = module_of((void *)dp->caller); int unk = 0; for (int j = 0; j < cs->nb; j++) if (cs->b[j].type == 'b' && cs->b[j].kind == K_UNK) unk++;
          LOG("vk: first dispatch of compute shader %s: %s %d,%d,%d groups, call site %s+%#llx, %d storage bindings unresolved or unbound (scratch buffer)%s", cs->key, dp->ind ? "indirect" : "direct", dp->x, dp->y, dp->z, m ? m->name : "?",
              m ? (unsigned long long)(dp->caller - (uint64_t)m->base) : (unsigned long long)dp->caller, g_push_fallback, unk ? " [RED] undecoded bindings: pack the shaders again with the current android-host/tools/pokopia_vkshaders.py" : ""); } }
    if (dp->ind) {   // the shader reads the counts at c0 + 0x200: copy the GPU-written ones there first, ordered before the dispatch (as nvn_mtl.m)
        if (g_c0_dim.buf) {
            VkBufferCopy bc = {io, g_c0_dim.off + 0x200, 12}; vkCmdCopyBuffer(cb, ib, g_c0_dim.buf, 1, &bc);
            VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT};
            VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb}; pvkCmdPipelineBarrier2(cb, &di);
        }
        vkCmdDispatchIndirect(cb, ib, io); g_stat_disp_ind++;
    } else vkCmdDispatch(cb, (uint32_t)dp->x, (uint32_t)dp->y, (uint32_t)dp->z);
    PP(PK_GAP); barrier_k(cb, BK_COMPUTE);
    g_stat_disp++;
    if (g_trace) LOG("vk: T dispatch cs %s %s %d,%d,%d", cs->key, dp->ind ? "indirect" : "direct", dp->x, dp->y, dp->z);
}

// ---------- Copies ----------
// is the region inside the image's level and layers (3D: its depth at that level)? Compressed formats are measured against the extent rounded up to whole blocks, so no copy that was
// recorded before this check is refused for a partial block at the edge; 1D arrays (NVN keeps their layer count in the height) are not checked in y
static int copy_in_image(const vimg *v, unsigned level, unsigned layer0, const NVNcopyRegion *r, int nz, int bw, int bh) {
    if (level >= v->levels || r->xoffset < 0 || r->yoffset < 0 || r->zoffset < 0 || r->width < 0 || r->height < 0 || nz < 0) return 0;
    int is3d = v->itype == VK_IMAGE_TYPE_3D; if (bw < 1) bw = 1; if (bh < 1) bh = 1;
    uint64_t lw = MAX(v->w >> level, 1u), lh = MAX(v->h >> level, 1u), ld = is3d ? MAX(v->d >> level, 1u) : v->layers;
    lw = (lw + bw - 1) / bw * bw; lh = (lh + bh - 1) / bh * bh;
    uint64_t z0 = (is3d ? 0 : layer0) + (uint64_t)r->zoffset;
    return (uint64_t)r->xoffset + r->width <= lw && (v->t.d.target == 3 || (uint64_t)r->yoffset + r->height <= lh) && z0 + nz <= ld;
}
static void copy_b2t(const p_copy_b2t *c) {
    vimg *v = vimg_get(c->tex);
    if (g_trace) LOG("vk: T copy buffer %#llx → texture %#llx %dx%d", (unsigned long long)c->src, c->tex ? (unsigned long long)c->tex->serial : 0ull, c->r.width, c->r.height);
    if (!v) return;
    rp_end();
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh); if (!bpb) return;
    fmt_t m = fmt_of(c->tex->d.format);
    if (m.conv) { RED_ONCE(20, "vk: buffer→texture copy into a converted format %#x: skipped", c->tex->d.format); return; }
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    if (level < (unsigned)v->drop) { v->gpu_owned = 1; v->uploaded = 1; return; }   // a level this image left out
    level -= (unsigned)v->drop;
    uint64_t row = S.copy_stride ? (uint64_t)S.copy_stride : (uint64_t)((c->r.width + bw - 1) / bw) * bpb, rows = (c->r.height + bh - 1) / bh, img = S.copy_img_stride ? (uint64_t)S.copy_img_stride : row * rows;
    int is3d = v->itype == VK_IMAGE_TYPE_3D, nz = c->r.depth ? c->r.depth : 1;
    VkBuffer b; VkDeviceSize off;
    if (!buf_at(c->src, img * nz, &b, &off)) { RED_ONCE(21, "vk: buffer→texture copy source is in no imported pool"); return; }
    if (row % bpb || img % row) { RED_ONCE(22, "vk: buffer→texture copy strides %llu / %llu are not whole texels", (unsigned long long)row, (unsigned long long)img); }
    VkBufferImageCopy bic = {.bufferOffset = off, .bufferRowLength = (uint32_t)(row / bpb * bw), .bufferImageHeight = (uint32_t)(img / row * bh),
        .imageSubresource = {v->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, level, is3d ? 0 : layer0 + c->r.zoffset, is3d ? 1 : (uint32_t)nz},
        .imageOffset = {c->r.xoffset, c->r.yoffset, is3d ? c->r.zoffset : 0}, .imageExtent = {(uint32_t)c->r.width, (uint32_t)c->r.height, is3d ? (uint32_t)nz : 1}};
    if (v->aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) { RED_ONCE(23, "vk: buffer→texture copy into depth-stencil: skipped"); return; }
    if (off % 4) { RED_ONCE(24, "vk: buffer→texture copy source offset %#llx is not 4-byte aligned", (unsigned long long)off); return; }
    if (!copy_in_image(v, level, layer0, &c->r, nz, bw, bh)) { RED_ONCE(68, "vk: buffer→texture copy skipped: region %d,%d,%d %dx%dx%d level %u layer %u is outside the %ux%u level of a %u-level, %u-layer texture", c->r.xoffset, c->r.yoffset, c->r.zoffset, c->r.width, c->r.height, nz, level, layer0, MAX(v->w >> level, 1u), MAX(v->h >> level, 1u), v->levels, v->layers); return; }
    PP(PK_COPY); vkCmdCopyBufferToImage(g_cur->cb, b, v->img, VK_IMAGE_LAYOUT_GENERAL, 1, &bic);
    PP(PK_GAP); barrier_all(g_cur->cb);
    v->gpu_owned = 1; v->uploaded = 1;
}
static void copy_t2b(const p_copy_b2t *c) {
    vimg *v = vimg_get(c->tex);
    if (g_trace) LOG("vk: T copy texture %#llx %dx%d → buffer %#llx", c->tex ? (unsigned long long)c->tex->serial : 0ull, c->r.width, c->r.height, (unsigned long long)c->src);
    if (!v) return;
    rp_end();
    if (!v->gpu_owned && (!v->uploaded || v->cpu_dirty)) vimg_upload(v);
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh); if (!bpb) return;
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    if (level < (unsigned)v->drop) { RED_ONCE(131, "vk: texture→buffer copy from a level this image left out: skipped (DM_TEX_DROP=0 keeps it)"); return; }
    level -= (unsigned)v->drop;
    uint64_t row = S.copy_stride ? (uint64_t)S.copy_stride : (uint64_t)((c->r.width + bw - 1) / bw) * bpb, rows = (c->r.height + bh - 1) / bh, img = S.copy_img_stride ? (uint64_t)S.copy_img_stride : row * rows;
    int is3d = v->itype == VK_IMAGE_TYPE_3D, nz = c->r.depth ? c->r.depth : 1;
    VkBuffer b; VkDeviceSize off;
    if (!buf_at(c->src, img * nz, &b, &off)) { RED_ONCE(25, "vk: texture→buffer copy destination is in no imported pool"); return; }
    if ((vk_texel_bytes(v->fmt) != (uint32_t)bpb && bw == 1) || fmt_of(c->tex->d.format).conv) { RED_ONCE(26, "vk: texture→buffer copy from a substituted format (NVN %#x): skipped", c->tex->d.format); return; }
    if (off % 4) { RED_ONCE(69, "vk: texture→buffer copy destination offset %#llx is not 4-byte aligned: skipped", (unsigned long long)off); return; }   // copy_b2t checks it too
    if (!copy_in_image(v, level, layer0, &c->r, nz, bw, bh)) { RED_ONCE(70, "vk: texture→buffer copy skipped: region %d,%d,%d %dx%dx%d level %u layer %u is outside the %ux%u level of a %u-level, %u-layer texture", c->r.xoffset, c->r.yoffset, c->r.zoffset, c->r.width, c->r.height, nz, level, layer0, MAX(v->w >> level, 1u), MAX(v->h >> level, 1u), v->levels, v->layers); return; }
    VkBufferImageCopy bic = {.bufferOffset = off, .bufferRowLength = (uint32_t)(row / bpb * bw), .bufferImageHeight = (uint32_t)(img / row * bh),
        .imageSubresource = {v->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, level, is3d ? 0 : layer0 + c->r.zoffset, is3d ? 1 : (uint32_t)nz},
        .imageOffset = {c->r.xoffset, c->r.yoffset, is3d ? c->r.zoffset : 0}, .imageExtent = {(uint32_t)c->r.width, (uint32_t)c->r.height, is3d ? (uint32_t)nz : 1}};
    PP(PK_COPY); vkCmdCopyImageToBuffer(g_cur->cb, v->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &bic);
    PP(PK_GAP); barrier_all(g_cur->cb);
}
static void copy_t2t(const p_copy_t2t *c) {
    vimg *s = vimg_get(c->src), *d = vimg_get(c->dst);
    if (g_trace) LOG("vk: T copy texture %#llx (%d,%d %dx%d) → %#llx (%d,%d %dx%d) flags %#x%s", c->src ? (unsigned long long)c->src->serial : 0ull, c->sr.xoffset, c->sr.yoffset, c->sr.width, c->sr.height,
        c->dst ? (unsigned long long)c->dst->serial : 0ull, c->dr.xoffset, c->dr.yoffset, c->dr.width, c->dr.height, c->flags, !s || !d ? " (no image: skipped)" : "");
    if (!s || !d) return;
    rp_end();
    if (!s->gpu_owned && (!s->uploaded || s->cpu_dirty)) vimg_upload(s);
    unsigned sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, ss = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0;
    unsigned dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0, ds = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
    if (sl < (unsigned)s->drop || dl < (unsigned)d->drop) { RED_ONCE(132, "vk: texture copy from or to a level the image left out: skipped (DM_TEX_DROP=0 keeps it)"); d->gpu_owned = 1; d->uploaded = 1; return; }
    sl -= (unsigned)s->drop; dl -= (unsigned)d->drop;
    int nz = MAX(c->sr.depth, 1);
    VkImageAspectFlags sa = s->aspect, da = d->aspect;
    {   // both regions inside their images (a resolve takes its destination size from the source region)
        NVNcopyRegion dr = c->dr; if (c->flags & DM_COPY_RESOLVE) { dr.width = c->sr.width; dr.height = c->sr.height; }
        int sbw = 1, sbh = 1, dbw = 1, dbh = 1; nvn_fmt_bpb(c->src->d.format, &sbw, &sbh); nvn_fmt_bpb(c->dst->d.format, &dbw, &dbh);
        if (!copy_in_image(s, sl, ss, &c->sr, nz, sbw, sbh) || !copy_in_image(d, dl, ds, &dr, nz, dbw, dbh)) {
            RED_ONCE(71, "vk: texture copy skipped: source %d,%d,%d %dx%d level %u layer %u / destination %d,%d,%d %dx%d level %u layer %u (depth %d) is outside its texture (%ux%u, %u levels, %u layers / %ux%u, %u levels, %u layers)",
                c->sr.xoffset, c->sr.yoffset, c->sr.zoffset, c->sr.width, c->sr.height, sl, ss, dr.xoffset, dr.yoffset, dr.zoffset, dr.width, dr.height, dl, ds, nz, s->w, s->h, s->levels, s->layers, d->w, d->h, d->levels, d->layers); return; } }
    PP(PK_COPY);   // one of resolve / copy / blit below (or nothing: the RED_ONCE branch); the mark pair is then ~0
    if (c->flags & DM_COPY_RESOLVE) {
        VkImageResolve rr = {{sa, sl, ss, 1}, {c->sr.xoffset, c->sr.yoffset, 0}, {da, dl, ds, 1}, {c->dr.xoffset, c->dr.yoffset, 0}, {(uint32_t)c->sr.width, (uint32_t)c->sr.height, 1}};
        if (s->samples > 1 && d->samples == 1 && sa == VK_IMAGE_ASPECT_COLOR_BIT) vkCmdResolveImage(g_cur->cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &rr);
        else RED_ONCE(27, "vk: resolve of a non-multisampled or depth texture: skipped");
    } else if (c->sr.width == c->dr.width && c->sr.height == c->dr.height && s->fmt == d->fmt && !(c->flags & (4 | 8)) && s->samples == d->samples) {
        int s3 = s->itype == VK_IMAGE_TYPE_3D, d3 = d->itype == VK_IMAGE_TYPE_3D;
        VkImageCopy ic = {{sa, sl, s3 ? 0 : ss + c->sr.zoffset, s3 ? 1 : (uint32_t)nz}, {c->sr.xoffset, c->sr.yoffset, s3 ? c->sr.zoffset : 0},
                          {da, dl, d3 ? 0 : ds + c->dr.zoffset, d3 ? 1 : (uint32_t)nz}, {c->dr.xoffset, c->dr.yoffset, d3 ? c->dr.zoffset : 0}, {(uint32_t)c->sr.width, (uint32_t)c->sr.height, s3 || d3 ? (uint32_t)nz : 1}};
        vkCmdCopyImage(g_cur->cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &ic);
    } else if (bc_twin(s->fmt) || bc_twin(d->fmt)) {
        RED_ONCE(133, "vk: texture copy with a compressed image, Vulkan formats %d and %d, that a plain copy can't do (one BC7 texture may have become BC1, the other BC3; DM_TEX_BC7=0 keeps them alike): skipped", (int)s->fmt, (int)d->fmt);
    } else if (sa == da && s->samples == 1 && d->samples == 1) {
        int x0 = c->sr.xoffset, x1 = c->sr.xoffset + c->sr.width, y0 = c->sr.yoffset, y1 = c->sr.yoffset + c->sr.height;
        if (c->flags & 4) { int t = x0; x0 = x1; x1 = t; } if (c->flags & 8) { int t = y0; y0 = y1; y1 = t; }
        VkImageBlit ib = {{sa, sl, ss + c->sr.zoffset, (uint32_t)nz}, {{x0, y0, 0}, {x1, y1, 1}}, {da, dl, ds + c->dr.zoffset, (uint32_t)nz}, {{c->dr.xoffset, c->dr.yoffset, 0}, {c->dr.xoffset + c->dr.width, c->dr.yoffset + c->dr.height, 1}}};
        int lin = (c->flags & 1) && sa == VK_IMAGE_ASPECT_COLOR_BIT && (fmt_props(s->fmt).optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
        vkCmdBlitImage(g_cur->cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &ib, lin ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    } else RED_ONCE(28, "vk: texture copy between different aspects / sample counts: skipped");
    PP(PK_GAP); barrier_all(g_cur->cb);
    d->gpu_owned = 1; d->uploaded = 1;
}
static void clear_tex(const p_clear_tex *c) {
    vimg *v = vimg_get(c->tex);
    if (g_trace) LOG("vk: T clear texture %#llx mask %#x", c->tex ? (unsigned long long)c->tex->serial : 0ull, c->mask);
    if (!v) return;
    rp_end();
    unsigned lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, ly = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    VkImageSubresourceRange r = {v->aspect, lv, 1, ly + (v->itype == VK_IMAGE_TYPE_3D ? 0 : c->r.zoffset), v->itype == VK_IMAGE_TYPE_3D ? 1 : (uint32_t)MAX(c->r.depth, 1)};
    PP(PK_CLEAR);
    if (v->aspect & VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue cv; memcpy(&cv, c->c.u, 16); vkCmdClearColorImage(g_cur->cb, v->img, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &r); }
    else { VkClearDepthStencilValue dv = {c->c.f[0], c->c.u[1]}; vkCmdClearDepthStencilImage(g_cur->cb, v->img, VK_IMAGE_LAYOUT_GENERAL, &dv, 1, &r); }
    PP(PK_GAP); barrier_all(g_cur->cb);
    v->gpu_owned = 1; v->uploaded = 1;
}
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) {
    VkBuffer b; VkDeviceSize off; if (!buf_at(dst, n, &b, &off)) { RED_ONCE(29, "vk: buffer fill outside imported pools"); return; }
    if (off & 3) { RED_ONCE(72, "vk: buffer fill skipped: address %#llx is not a multiple of 4 (the fill would land on other bytes)", (unsigned long long)dst); return; }
    if (n & 3) RED_ONCE(73, "vk: buffer fill of %llu bytes is not a multiple of 4: rounded up (the extra bytes after it are written too)", (unsigned long long)n);
    rp_end(); PP(PK_CLEAR); vkCmdFillBuffer(g_cur->cb, b, off & ~3ull, (n + 3) & ~3ull, value); PP(PK_GAP); barrier_all(g_cur->cb);
}
static void sub_copy(uint64_t s, uint64_t d, uint64_t n) {
    VkBuffer sb, db; VkDeviceSize so, dof;
    if (!n) return;
    if (!buf_at(s, n, &sb, &so) || !buf_at(d, n, &db, &dof)) { RED_ONCE(30, "vk: buffer copy outside imported pools"); return; }
    rp_end(); VkBufferCopy bc = {so, dof, n}; PP(PK_COPY); vkCmdCopyBuffer(g_cur->cb, sb, db, 1, &bc); PP(PK_GAP); barrier_all(g_cur->cb);
}
static void sub_update(const void *data, uint64_t d, uint64_t n) {
    VkBuffer db; VkDeviceSize dof;
    if (!n) return;
    if (!buf_at(d, n, &db, &dof)) { RED_ONCE(31, "vk: inline buffer update outside imported pools"); return; }
    rp_end(); PP(PK_COPY);
    if ((dof | n) & 3 || n > 65536) {   // not expressible as vkCmdUpdateBuffer: through the ring
        halloc h = ring_alloc(n, 4); memcpy(h.p, data, n); VkBufferCopy bc = {h.off, dof, n}; vkCmdCopyBuffer(g_cur->cb, h.buf, db, 1, &bc);
    } else vkCmdUpdateBuffer(g_cur->cb, db, dof, n, data);
    PP(PK_GAP); barrier_all(g_cur->cb);
}
static uint64_t g_stat_ub;

// ---------- The record walker ----------
static void state_op(uint32_t op, const void *a) {
    switch (op) {
        case OP_SCISSOR: S.sc = *(const p_scissor *)a; S.has_sc = 1; if (g_trace) LOG("vk: T scissor %d,%d %dx%d", S.sc.x, S.sc.y, S.sc.w, S.sc.h); break;
        case OP_BIND_PROGRAM: { const p_bind_program *bp = a; for (int st = 0; st < 6; st++) if (bp->stages >> st & 1) S.bst[st] = bp->prog; break; }
        case OP_BIND_STATE: { const p_bind_state *b = a; blend_st bs;
            switch (b->kind) {
                case 0: memcpy(&bs, b->bytes, sizeof bs); S.blend[bs.target & 7] = bs; break;
                case 1: memcpy(&S.color, b->bytes, sizeof S.color); break; case 2: memcpy(&S.chmask, b->bytes, sizeof S.chmask); break;
                case 3: memcpy(&S.ms, b->bytes, sizeof S.ms); break; case 4: memcpy(&S.poly, b->bytes, sizeof S.poly); break; case 5: memcpy(&S.ds, b->bytes, sizeof S.ds); break;
            } break; }
        // nvnCommandBufferBindVertexAttribState(n, ...) disables every attribute from n up: an older draw's attribute left in place made a ground shader read another mesh's
        // all-white vertex colours as its location 12 (added to the albedo: the ground drew white). A disabled attribute reads the zero buffer (draw)
        case OP_BIND_VATTR: { const p_bind_vattr *b = a; if (b->first >= 0 && b->first + b->n <= 16) { memcpy(&S.attr[b->first], b->st, 4 * b->n); memset(&S.attr[b->first + b->n], 0, 4 * (16 - b->first - b->n)); S.nattr = b->first + b->n; } break; }
        case OP_BIND_VSTREAM: { const p_bind_vstream *b = a; if (b->first >= 0 && b->first + b->n <= 16) { memcpy(&S.stream[b->first], b->st, 8 * b->n); S.nstream = MAX(S.nstream, b->first + b->n); } break; }
        // OP_BIND_VBUF1 / OP_BIND_UBO_SMALL / OP_BIND_TEX1 are compact (24-byte) records of the multi-bind ops' state; vbuf_rec / bind_buf_rec / bind_rec (nvn_obj.h) expand them to the same payload
        case OP_BIND_VBUF: case OP_BIND_VBUF1: { p_bind_vbuf tmp; const p_bind_vbuf *b = vbuf_rec(op, a, &tmp); if (b->first >= 0 && b->n >= 0 && b->first + b->n <= 16) memcpy(&S.vbuf[b->first], b->r, sizeof *b->r * b->n); break; }
        case OP_BIND_UBO: case OP_BIND_UBO_SMALL: { p_bind_buf tmp; const p_bind_buf *b = bind_buf_rec(op, a, &tmp); if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) S.ubo[b->stage][b->index] = *b; break; }
        case OP_BIND_SSBO: { const p_bind_buf *b = a; if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) S.ssbo[b->stage][b->index] = *b; break; }
        case OP_BIND_TEX: case OP_BIND_TEX1: case OP_BIND_SAMP: case OP_BIND_IMG: { p_bind_handles t1; const p_bind_handles *b = op == OP_BIND_TEX1 ? bind_rec(op, a, &t1) : a;   // TEX1: one combined handle (nvnCommandBufferBindTexture), the pool bases added
            int lim = op == OP_BIND_TEX || op == OP_BIND_TEX1 ? NTEX : op == OP_BIND_SAMP ? 128 : 32;
            if (b->stage < 0 || b->stage >= 6 || b->first < 0 || b->n < 0 || b->first + b->n > lim || b->n > 32) { RED_ONCE(32, "vk: handle bind out of range (op %u stage %d first %d n %d)", op, b->stage, b->first, b->n); break; }
            uint64_t h[32]; memcpy(h, b->h, 8 * (size_t)b->n);   // the record holds n handles, not 32 (RECN): a copy of the whole array read past a record at the end of a command block
            if (b->defer && b->src) for (int i = 0; i < b->n; i++) memcpy(&h[i], (const uint8_t *)(uintptr_t)b->src + 8 * i, 8);   // BindTexturesDeferred: the handles are read at execution time
            uint64_t *dst = op == OP_BIND_TEX || op == OP_BIND_TEX1 ? S.tex[b->stage] : op == OP_BIND_SAMP ? S.samp[b->stage] : S.img[b->stage];
            memcpy(dst + b->first, h, 8 * (size_t)b->n); break; }
        case OP_STENCIL: { const p_stencil *b = a; for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) S.st_vmask[f] = b->value_mask; else if (b->which == 1) S.st_wmask[f] = b->mask; else S.st_ref[f] = b->ref; } break; }
        case OP_POLY_OFFSET: S.po = *(const p_poly_offset *)a; break;
        case OP_VIEWPORTS: { const p_viewports *b = a; for (int i = 0; i < b->n && b->first + i < 16; i++) memcpy(S.vp[b->first + i], b->v[i], 16); S.has_vp = 1; break; }
        case OP_DEPTH_RANGE: { const p_depth_range *b = a; S.dr[0] = b->n; S.dr[1] = b->f; break; }
        case OP_DEPTH_CLAMP: S.depth_clamp = ((const p_depth_clamp *)a)->on; break;
        case OP_VP_SWIZZLE: { const p_vp_swizzle *b = a; if (b->first >= 0 && b->n >= 0 && b->first + b->n <= 16) memcpy(S.vp_sw[b->first], b->sw, 4 * (size_t)b->n); break; }
        case OP_PRIM_RESTART: S.restart = *(const p_prim_restart *)a; break;
        case OP_BLEND_COLOR: memcpy(S.blend_color, ((const p_blend_color *)a)->rgba, 16); break;
        case OP_COPY_STRIDE: { const p_copy_stride *cs = a; if (cs->image) S.copy_img_stride = cs->stride; else S.copy_stride = cs->stride; break; }
        case OP_RAST_DISCARD: S.rast_discard = ((const p_rast_discard *)a)->on; break;
        case OP_PUSH_CONST: { const p_push_const *b = a; memcpy(S.push[b->compute ? 1 : 0] + b->offset, b->data, b->size); break; }   // hle_nvn.c push_const checks the range
        case OP_PATCH_SIZE: S.patch = ((const p_patch_size *)a)->n; break;
        case OP_DRAW_TEX: RED_ONCE(72, "vk: nvnCommandBufferDrawTexture is not implemented: not drawn"); break;
        case OP_TF_BIND: case OP_TF_STATE: RED_ONCE(33, "vk: transform feedback is not implemented"); break;
    }
}
static uint64_t g_stat_ops[OP_COUNT_ + 1];
static int g_exec_depth, g_exec_bad;   // g_exec_bad: a guard below stopped the walk of this handle (set on the way out of every nested CALL, cleared at the top level)
void mtl_exec(const uint8_t *p) {
    if (!g_cur) { RED_ONCE(34, "vk: commands replayed outside a submission"); return; }
    if (g_trace) LOG("vk: T %s %p", g_exec_depth ? "call" : "handle", (const void *)p);
    if (!g_exec_depth) g_exec_bad = 0;
    g_exec_depth++;
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        if (h.op != OP_END && (h.op >= OP_COUNT_ || h.size < 8)) {   // a stale or overwritten command handle: a size of 0 would spin here for ever, a random one walks off into random memory
            RED_ONCE(48, "vk: bad command record at %p (op %u, size %u): the rest of this command buffer is not replayed", (const void *)p, h.op, h.size); g_exec_bad = 1; g_exec_depth--; return; }
        const void *a = p + 8;
        g_stat_ops[h.op < OP_COUNT_ ? h.op : OP_COUNT_]++;
        switch (h.op) {
            case OP_END: g_exec_depth--; return;
            case OP_BEGIN: break;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8);
                if (g_exec_depth > 64) { RED_ONCE(49, "vk: command buffers called more than 64 deep (a CALL loop?) at %p: the rest of this command buffer is not replayed", (const void *)p); g_exec_bad = 1; g_exec_depth--; return; }
                mtl_exec((const uint8_t *)(uintptr_t)t); if (g_exec_bad) { g_exec_depth--; return; } break; }
            case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, (const uint8_t *)a + 8, 4); memcpy(&v, (const uint8_t *)a + 12, 4); post((post_op){1, val, m, v, 0, 0}); break; }
            case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, (const uint8_t *)a + 8, 8); post((post_op){2, NULL, 0, 0, ad, (uint32_t)t});
                if (DM_GAME_RENDER_COND) rc_note(ad, t == 0 || t == 0x10 ? 1 : 2); break; }
            case OP_RENDER_COND: { const p_render_cond *rc = a;   // recorded only with the profile key render_cond (hle_nvn.c)
                if (rc->mode >= 2) g_rc_skip = rc->mode == 3;
                else { uint64_t x = 0, y = 0; int known = rc_value(rc->addr, &x) & rc_value(rc->addr + 16, &y); g_rc_skip = known && (rc->mode ? x == y : x != y); }
                if (g_trace) LOG("vk: T render condition mode %d → %s", rc->mode, g_rc_skip ? "skip" : "draw");
                break; }
            case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); if (s->magic != MAGIC('S', 'Y', 'N', 'C')) { RED_ONCE(44, "vk: FenceSync on %p, which is no initialized sync object now: skipped", (void *)s); break; } if (!__atomic_load_n(&s->fenced, __ATOMIC_ACQUIRE) || s->fence < g_fence + 1) { s->fence = g_fence + 1; __atomic_store_n(&s->fenced, 1, __ATOMIC_RELEASE); } break; }   // fence values only grow: a QueueFenceSync issued after this submit (a later ticket) is not lowered by the late replay; the fence is published before fenced (read by the guest thread)
            case OP_RESET_COUNTER: break;
            case OP_CLEAR_BUF: { const p_clear_buf *cb = a; mtl_sub_fill(cb->addr, cb->size, cb->value); break; }
            case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, (const uint8_t *)a + 8, 8); memcpy(&n, (const uint8_t *)a + 16, 8); sub_copy(s_, d, n); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, (const uint8_t *)a + 8, 8); sub_update((const uint8_t *)a + 16, d, n); g_stat_ub++; break; }
            case OP_SET_RT: { const p_set_rt *r = a; if (g_trace) LOG("vk: T set targets n %d c0 %#llx d %#llx at %p", r->n, r->n && r->color[0] ? (unsigned long long)r->color[0]->serial : 0ull, r->depth ? (unsigned long long)r->depth->serial : 0ull, (const void *)p); if (!R.active || memcmp(&R.rt, r, sizeof *r)) { rp_end(); R.rt = *r; R.has = 1; } break; }
            case OP_CLEAR_COLOR: if (!g_rc_skip) clear_color(a); else g_stat_rc_skipped++; break;
            case OP_CLEAR_DS: if (!g_rc_skip) clear_ds(a); else g_stat_rc_skipped++; break;
            case OP_DRAW: { if (g_rc_skip) { g_stat_rc_skipped++; break; } p_draw t; draw(draw_rec(a, h.size, &t), NULL); break; }
            case OP_DRAW_IDX: { if (g_rc_skip) { g_stat_rc_skipped++; break; } p_draw_idx t; draw(NULL, draw_idx_rec(a, h.size, &t)); break; }
            case OP_COPY_B2T: copy_b2t(a); break;
            case OP_COPY_T2B: copy_t2b(a); break;
            case OP_COPY_T2T: copy_t2t(a); break;
            case OP_CLEAR_TEX: clear_tex(a); break;
            case OP_DISPATCH: { p_dispatch t; memset(&t, 0, sizeof t); memcpy(&t, a, MIN((size_t)h.size - 8, sizeof t)); dispatch(&t); break; }   // not under g_rc_skip (as nvn_mtl.m)
            case OP_BARRIER: if (R.active) { rp_end(); } break;
            case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: break;
            default: state_op(h.op, a); break;
        }
        p += h.size;
    }
}

// ---------- Present ----------
static ANativeWindow *g_win; static int g_win_changed; static pthread_mutex_t g_wmu = PTHREAD_MUTEX_INITIALIZER;
static VkQueryPool g_ctsq; static int g_cts_used[8];   // DM_VK_PASSPROF: the composite's GPU time, a pair of timestamps for each frame slot
static VkSurfaceKHR g_surf; static VkSwapchainKHR g_sc; static VkImage g_sci[8]; static VkImageView g_scv[8]; static uint32_t g_nsci; static VkFormat g_scf; static VkExtent2D g_sce;
static int g_rot; static VkSurfaceTransformFlagBitsKHR g_sc_transform; static VkExtent2D g_logical;   // the swapchain's pre-transform in quarter turns, and the window's own (landscape) size the frame is laid out in
static VkSemaphore g_acq[8], g_rel[8]; static uint32_t g_frame_i;
static VkCommandPool g_ppool; static VkCommandBuffer g_pcb[8]; static VkFence g_pfence[8]; static int g_pfence_used[8];
static uint64_t g_presents;
// The APK front's on-screen controls: an RGBA8 layer (premultiplied alpha, sRGB) the front draws on the CPU at any size; the present stretches it over the window
// The input thread reads and writes these while a present can be blocked in vkQueuePresentKHR (about 10 ms a frame when the GPU is behind) holding g_wmu, so they stay out of g_wmu: touches must not wait for the GPU
static _Atomic int g_frame_rect[4], g_present_linear; static pthread_mutex_t g_ovmu = PTHREAD_MUTEX_INITIALIZER;   // g_ovmu: g_ov
static struct { hbuf stage; VkImage img; VkDeviceMemory mem; VkImageView view; uint32_t w, h, cw, ch; int dirty, visible; float opacity; } g_ov;
void dm_vk_overlay_image(const uint8_t *rgba, int w, int h, float opacity) {   // rgba NULL or w 0: no layer
    pthread_mutex_lock(&g_ovmu);
    if (!rgba || w <= 0 || h <= 0) g_ov.visible = 0;
    else {
        uint64_t sz = (uint64_t)w * h * 4;
        if (g_ov.stage.size < sz) {   // an earlier present may still be copying from the old buffer: freed through the deferred list, once the next submission (queued after every present made so far) completed
            if (g_ov.stage.buf) { dead *d = calloc(1, sizeof *d); if (d) { d->buf = g_ov.stage.buf; d->mem = g_ov.stage.mem; dead_add(d); } }
            g_ov.stage = hbuf_new(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT); }
        memcpy(g_ov.stage.map, rgba, sz); g_ov.w = (uint32_t)w; g_ov.h = (uint32_t)h; g_ov.dirty = 1; g_ov.visible = 1; g_ov.opacity = opacity;
    }
    pthread_mutex_unlock(&g_ovmu);
}
// Where the last present put the game frame in the window (logical pixels, landscape); 0 x 0 before the first one
void dm_vk_frame_rect(int out[4]) { for (int i = 0; i < 4; i++) out[i] = atomic_load_explicit(&g_frame_rect[i], memory_order_relaxed); }   // changes only with the window's layout
// Whether the present's blending happens in linear light: an sRGB swapchain, taken when the game's window texture is sRGB, as this game's is
int dm_vk_present_linear(void) { return atomic_load_explicit(&g_present_linear, memory_order_relaxed); }
#include "present_spv.h"
static struct { VkPipelineLayout lay; VkDescriptorSetLayout dsl; VkSampler samp; VkFormat fmt; VkPipeline pl; } g_cp;
static int composite_init(VkFormat fmt) {
    if (g_cp.pl && g_cp.fmt == fmt) return 1;
    if (!g_cp.lay) {
        VkDescriptorSetLayoutBinding lb = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL};
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 1, .pBindings = &lb};
        VKCHK(vkCreateDescriptorSetLayout(g_dev, &li, NULL, &g_cp.dsl));
        VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 48};
        VkPipelineLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &g_cp.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
        VKCHK(vkCreatePipelineLayout(g_dev, &lci, NULL, &g_cp.lay));
        VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
        VKCHK(vkCreateSampler(g_dev, &sci, NULL, &g_cp.samp));
    }
    if (g_cp.pl) { vkDestroyPipeline(g_dev, g_cp.pl, NULL); g_cp.pl = VK_NULL_HANDLE; }
    VkShaderModule m[2];
    VkShaderModuleCreateInfo mi[2] = {{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof present_vert_spv, .pCode = present_vert_spv},
                                      {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof present_frag_spv, .pCode = present_frag_spv}};
    VKCHK(vkCreateShaderModule(g_dev, &mi[0], NULL, &m[0])); VKCHK(vkCreateShaderModule(g_dev, &mi[1], NULL, &m[1]));
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = m[0], .pName = "main"},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = m[1], .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba = {1, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD, 0xf};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &cba};
    static const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, .stageCount = 2, .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb, .pDynamicState = &dy, .layout = g_cp.lay};
    VkResult r = vkCreateGraphicsPipelines(g_dev, VK_NULL_HANDLE, 1, &gi, NULL, &g_cp.pl);
    vkDestroyShaderModule(g_dev, m[0], NULL); vkDestroyShaderModule(g_dev, m[1], NULL);
    if (r) { LOG("[RED] vk: present composite pipeline: %d", (int)r); g_cp.pl = VK_NULL_HANDLE; return 0; }
    g_cp.fmt = fmt; return 1;
}
// The controls layer's image follows the front's size; its copy from the staging buffer is recorded before the composite pass
static void overlay_upload(VkCommandBuffer cb) {
    pthread_mutex_lock(&g_ovmu);
    if (!g_ov.visible || !g_ov.dirty) { pthread_mutex_unlock(&g_ovmu); return; }
    if (g_ov.cw != g_ov.w || g_ov.ch != g_ov.h) {
        if (g_ov.img) { pthread_mutex_lock(&g_qmu); vkQueueWaitIdle(g_q); pthread_mutex_unlock(&g_qmu); vkDestroyImageView(g_dev, g_ov.view, NULL); vkDestroyImage(g_dev, g_ov.img, NULL); vkFreeMemory(g_dev, g_ov.mem, NULL); }
        // UNORM: the layer's values go to the blend as they are. The front draws it for the space the swapchain blends in (dm_vk_present_linear)
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {g_ov.w, g_ov.h, 1}, .mipLevels = 1, .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VKCHK(vkCreateImage(g_dev, &ci, NULL, &g_ov.img));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev, g_ov.img, &mr);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
        VKCHK(vkAllocateMemory(g_dev, &ai, NULL, &g_ov.mem)); VKCHK(vkBindImageMemory(g_dev, g_ov.img, g_ov.mem, 0));
        VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_ov.img, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VKCHK(vkCreateImageView(g_dev, &vci, NULL, &g_ov.view));
        g_ov.cw = g_ov.w; g_ov.ch = g_ov.h;
    }
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_ov.img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b}; pvkCmdPipelineBarrier2(cb, &di);
    VkBufferImageCopy bic = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {g_ov.w, g_ov.h, 1}};
    vkCmdCopyBufferToImage(cb, g_ov.stage.buf, g_ov.img, VK_IMAGE_LAYOUT_GENERAL, 1, &bic);
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT; b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT; b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT; b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL; pvkCmdPipelineBarrier2(cb, &di);
    g_ov.dirty = 0;
    pthread_mutex_unlock(&g_ovmu);
}
static void composite_quad(VkCommandBuffer cb, VkImageView view, const float dst[4], const float src[4], float rot, float opaque, float opacity) {
    VkDescriptorImageInfo ii = {g_cp.samp, view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    pvkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_cp.lay, 0, 1, &w);
    float pc[12] = {dst[0], dst[1], dst[2], dst[3], src[0], src[1], src[2], src[3], rot, opaque, opacity, 0};
    vkCmdPushConstants(cb, g_cp.lay, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, pc);
    vkCmdDraw(cb, 4, 1, 0, 0);
}
// The surface is destroyed only on the front's thread (dm_vk_set_window): its last reference tears down the window's BLASTBufferQueue, which attaches the
// calling thread to the Java VM, and ART aborts on a guest thread ("Check failed: FindStackTop", seen on going to the background). The present thread
// rebuilds only the swapchain (out of date)
static void swapchain_destroy(int surface_too) {
    if (!g_sc && !(surface_too && g_surf)) return;
    pthread_mutex_lock(&g_qmu); vkQueueWaitIdle(g_q); pthread_mutex_unlock(&g_qmu);
    for (uint32_t i = 0; i < g_nsci; i++) if (g_scv[i]) { vkDestroyImageView(g_dev, g_scv[i], NULL); g_scv[i] = VK_NULL_HANDLE; }
    if (g_sc) pvkDestroySwapchainKHR(g_dev, g_sc, NULL); g_sc = VK_NULL_HANDLE; g_nsci = 0;
    if (surface_too && g_surf) { vkDestroySurfaceKHR(g_inst, g_surf, NULL); g_surf = VK_NULL_HANDLE; }
}
void dm_vk_set_window(void *w) {   // the APK front: the activity's ANativeWindow (NULL when its surface goes away); done with the old one on return
    pthread_mutex_lock(&g_wmu);
    if (g_dev) swapchain_destroy(1);
    if (w) ANativeWindow_acquire((ANativeWindow *)w);
    if (g_win) ANativeWindow_release(g_win);
    g_win = (ANativeWindow *)w; g_win_changed = 0;
    pthread_mutex_unlock(&g_wmu);
}
static int swapchain_make(VkFormat want) {
    if (!g_win) return 0;
    VkAndroidSurfaceCreateInfoKHR si = {VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR, .window = g_win};
    if (!g_surf && vkCreateAndroidSurfaceKHR(g_inst, &si, NULL, &g_surf)) { LOG("[RED] vk: cannot create the window surface"); g_surf = VK_NULL_HANDLE; return 0; }
    VkBool32 ok = 0; VkResult sr = vkGetPhysicalDeviceSurfaceSupportKHR(g_pd, g_qf, g_surf, &ok);
    if (sr || !ok) RED_ONCE(61, "vk: surface support query for queue family %u: result %d, supported %u (a swapchain on an unsupported queue is invalid; trying anyway)", g_qf, (int)sr, (unsigned)ok);
    VkSurfaceCapabilitiesKHR cap; if ((sr = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_pd, g_surf, &cap))) { RED_ONCE(62, "vk: surface capabilities query failed: %d: no swapchain", (int)sr); return 0; }
    uint32_t nf = 0; VkSurfaceFormatKHR sf[32];
    if ((sr = vkGetPhysicalDeviceSurfaceFormatsKHR(g_pd, g_surf, &nf, NULL)) || !nf) { RED_ONCE(63, "vk: the surface lists no formats (result %d, count %u): no swapchain", (int)sr, nf); return 0; }
    nf = MIN(nf, 32u); sr = vkGetPhysicalDeviceSurfaceFormatsKHR(g_pd, g_surf, &nf, sf);   // VK_INCOMPLETE (more than 32): the first 32 are enough
    if (sr < 0 || !nf) { RED_ONCE(64, "vk: reading the surface formats failed: %d: no swapchain", (int)sr); return 0; }
    g_scf = sf[0].format; VkColorSpaceKHR scs = sf[0].colorSpace;   // the colour space of the chosen entry, not always SRGB_NONLINEAR
    for (uint32_t i = 0; i < nf; i++) if (sf[i].format == want) { g_scf = want; scs = sf[i].colorSpace; if (scs == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) break; }
    atomic_store_explicit(&g_present_linear, g_scf == VK_FORMAT_R8G8B8A8_SRGB || g_scf == VK_FORMAT_B8G8R8A8_SRGB, memory_order_relaxed);
    g_logical = cap.currentExtent.width != ~0u ? cap.currentExtent : (VkExtent2D){(uint32_t)ANativeWindow_getWidth(g_win), (uint32_t)ANativeWindow_getHeight(g_win)};
    // Pre-rotation (Android): the swapchain takes the panel's native orientation and the current transform, and the composite turns the frame itself, so the
    // compositor does not rotate every frame. currentExtent is in the current orientation: swapped for a quarter turn. DM_VK_NOPREROTATE=1: the compositor rotates
    g_rot = cap.currentTransform == VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ? 1 : cap.currentTransform == VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR ? 2 : cap.currentTransform == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR ? 3 : 0;
    if (getenv("DM_VK_NOPREROTATE") || !(cap.supportedTransforms & cap.currentTransform)) g_rot = 0;
    g_sce = g_rot & 1 ? (VkExtent2D){g_logical.height, g_logical.width} : g_logical;
    VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = g_surf, .minImageCount = MAX(cap.minImageCount, 3u), .imageFormat = g_scf,
        .imageColorSpace = scs, .imageExtent = g_sce, .imageArrayLayers = 1, .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform = g_rot ? cap.currentTransform : VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, .compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR, .presentMode = VK_PRESENT_MODE_FIFO_KHR, .clipped = 1};
    if (!g_rot && !(cap.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)) ci.preTransform = cap.currentTransform;
    g_sc_transform = cap.currentTransform;
    if (!(cap.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)) ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (cap.maxImageCount && ci.minImageCount > cap.maxImageCount) ci.minImageCount = cap.maxImageCount;
    VkResult r = pvkCreateSwapchainKHR(g_dev, &ci, NULL, &g_sc);
    if (r) { LOG("[RED] vk: swapchain creation failed: %d", (int)r); g_sc = VK_NULL_HANDLE; return 0; }
    { uint32_t nim = 0; VkResult ir = pvkGetSwapchainImagesKHR(g_dev, g_sc, &nim, NULL);   // the arrays hold 8: a swapchain with more would be indexed past them by the acquire
      if (ir || nim == 0 || nim > 8) { RED_ONCE(65, "vk: the swapchain has %u images (result %d), at most 8 are supported: no swapchain", nim, (int)ir); pvkDestroySwapchainKHR(g_dev, g_sc, NULL); g_sc = VK_NULL_HANDLE; g_nsci = 0; return 0; } }
    g_nsci = 8; { VkResult gr = pvkGetSwapchainImagesKHR(g_dev, g_sc, &g_nsci, g_sci);
      if (gr) { RED_ONCE(66, "vk: reading the swapchain images failed: %d: no swapchain", (int)gr); pvkDestroySwapchainKHR(g_dev, g_sc, NULL); g_sc = VK_NULL_HANDLE; g_nsci = 0; return 0; } }
    for (uint32_t i = 0; i < g_nsci; i++) {
        VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_sci[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = g_scf, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VKCHK(vkCreateImageView(g_dev, &vci, NULL, &g_scv[i]));
    }
    composite_init(g_scf);
    LOG("vk: swapchain %ux%u format %d, %u images (window %dx%d, current transform %#x, pre-rotated %d quarter turns, layout %ux%u)", g_sce.width, g_sce.height, (int)g_scf, g_nsci,
        ANativeWindow_getWidth(g_win), ANativeWindow_getHeight(g_win), cap.currentTransform, g_rot, g_logical.width, g_logical.height);
    return 1;
}
static float f11(uint32_t v, int mb) { int e = (int)(v >> mb) & 31; float m = (float)(v & ((1u << mb) - 1)) / (float)(1u << mb); return e ? ldexpf(1 + m, e - 15) : ldexpf(m, -14); }
static uint8_t hdr8(float v) { v = v < 0 ? 0 : v; v = v / (1 + v); return (uint8_t)(powf(v, 1 / 2.2f) * 255 + .5f); }   // dumps of float targets: Reinhard, gamma 2.2 (for looking only)
static void write_ppm(const char *path, const uint8_t *px, uint32_t w, uint32_t h, VkFormat f) {
    FILE *o = fopen(path, "wb"); if (!o) { LOG("[RED] vk: cannot write %s", path); return; }
    fprintf(o, "P6\n%u %u\n255\n", w, h);
    uint8_t *row = malloc(3 * w);
    for (uint32_t y = 0; y < h; y++) {
        for (uint32_t x = 0; x < w; x++) {
            const uint8_t *s = px + ((uint64_t)y * w + x) * vk_texel_bytes(f); uint8_t *d = row + 3 * x;
            if (f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB) { d[0] = s[2]; d[1] = s[1]; d[2] = s[0]; }
            else if (f == VK_FORMAT_A2B10G10R10_UNORM_PACK32) { uint32_t v; memcpy(&v, s, 4); d[0] = (v & 1023) >> 2; d[1] = ((v >> 10) & 1023) >> 2; d[2] = ((v >> 20) & 1023) >> 2; }
            else if (f == VK_FORMAT_B10G11R11_UFLOAT_PACK32) { uint32_t v; memcpy(&v, s, 4); d[0] = hdr8(f11(v & 2047, 6)); d[1] = hdr8(f11((v >> 11) & 2047, 6)); d[2] = hdr8(f11(v >> 22, 5)); }
            else if (f == VK_FORMAT_R16G16B16A16_SFLOAT) { for (int c = 0; c < 3; c++) { uint16_t hv; memcpy(&hv, s + 2 * c, 2); int e = (hv >> 10) & 31; float m = (hv & 1023) / 1024.f; float v = e ? (1 + m) * (float)(1 << e) / 32768.f : m / 16384.f; v = v < 0 ? 0 : v > 1 ? 1 : v; d[c] = (uint8_t)(v * 255 + .5f); } }
            else { d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; }
        }
        fwrite(row, 1, 3 * w, o);
    }
    free(row); fclose(o); LOG("vk: wrote %s (%ux%u)", path, w, h);
}
static int shot_wanted(uint64_t n) {
    static uint64_t want[64]; static int nw = -1;
    if (nw < 0) { nw = 0; for (const char *q = getenv("DM_VK_SHOT"); q && *q && nw < 64; ) { want[nw++] = strtoull(q, (char **)&q, 10); if (*q == ',') q++; else break; } }
    for (int i = 0; i < nw; i++) if (want[i] == n) return 1;
    return 0;
}
// DM_VK_DUMP=1 with DM_VK_SHOT: at each shot also save every GPU-written 2D image (colour as RGB, depth as grey) with its mean, to see which passes produce content
static int dump_serial_wanted(uint64_t serial) {
    static uint64_t want[32]; static int nw = -1;
    if (nw < 0) { nw = 0; for (const char *q = getenv("DM_VK_DUMP_SERIALS"); q && *q && nw < 32; ) { want[nw++] = strtoull(q, (char **)&q, 16); if (*q == ',') q++; else break; } }
    for (int i = 0; i < nw; i++) if (want[i] == serial) return 1;
    for (int i = 0; i < g_ndyn; i++) if (g_dyn_dump[i] == serial) return 1;
    return 0;
}
static void dump_targets(uint64_t n) {
    enum { MAXD = 128 }; vimg *list[MAXD]; int nl = 0;
    pthread_mutex_lock(&g_qmu); vkQueueWaitIdle(g_q); pthread_mutex_unlock(&g_qmu);
    pthread_mutex_lock(&g_imu);
    for (uint32_t i = 0; i < g_ihcap && nl < MAXD; i++) { vimg *v = g_ih[i].v;
        if (g_ih[i].key && v && v->samples == 1 && ((v->gpu_owned && v->itype == VK_IMAGE_TYPE_2D && v->w >= 16 && v->h >= 16) || v->itype == VK_IMAGE_TYPE_3D || (v->itype == VK_IMAGE_TYPE_2D && dump_serial_wanted(v->serial))) &&
            (v->aspect == VK_IMAGE_ASPECT_COLOR_BIT || v->fmt == VK_FORMAT_D32_SFLOAT || v->fmt == VK_FORMAT_D32_SFLOAT_S8_UINT)) list[nl++] = v; }
    pthread_mutex_unlock(&g_imu);
    const char *dir = getenv("DM_VK_SHOT_DIR"); if (!dir) dir = getenv("DM_TEMP"); if (!dir) dir = ".";
    pthread_mutex_lock(&g_wmu);
    for (int k = 0; k < nl; k++) {
        vimg *v = list[k]; int depth = v->aspect != VK_IMAGE_ASPECT_COLOR_BIT; uint32_t tb = depth ? 4 : vk_texel_bytes(v->fmt);
        uint32_t vd = v->itype == VK_IMAGE_TYPE_3D ? v->d : 1;   // 3D: the slices one under another
        uint64_t sz = (uint64_t)v->w * v->h * vd * tb;
        hbuf hb = hbuf_new(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_ppool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VkCommandBuffer cb; VKCHK(vkAllocateCommandBuffers(g_dev, &ai, &cb));
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; VKCHK(vkBeginCommandBuffer(cb, &bi));
        VkBufferImageCopy bic = {.imageSubresource = {depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {v->w, v->h, vd}};
        vkCmdCopyImageToBuffer(cb, v->img, VK_IMAGE_LAYOUT_GENERAL, hb.buf, 1, &bic);
        VKCHK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
        pthread_mutex_lock(&g_qmu); VKCHK(vkQueueSubmit(g_q, 1, &si, VK_NULL_HANDLE)); vkQueueWaitIdle(g_q); pthread_mutex_unlock(&g_qmu);
        vkFreeCommandBuffers(g_dev, g_ppool, 1, &cb);
        double mean[3] = {0};
        uint8_t *rgb = malloc((uint64_t)v->w * v->h * 3);
        if (depth) { for (uint64_t i = 0; i < (uint64_t)v->w * v->h; i++) { float f; memcpy(&f, hb.map + 4 * i, 4); uint8_t g = (uint8_t)(MIN(MAX(f, 0.f), 1.f) * 255); rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = g; } }
        char path[600]; snprintf(path, sizeof path, "%s/dump_%llu_%llx_%ux%ux%u_f%d%s.ppm", dir, (unsigned long long)n, (unsigned long long)v->serial, v->w, v->h, vd, (int)v->fmt, v->gpu_owned ? "" : "_cpu");
        if (depth) { FILE *o = fopen(path, "wb"); if (o) { fprintf(o, "P6\n%u %u\n255\n", v->w, v->h); fwrite(rgb, 1, (uint64_t)v->w * v->h * 3, o); fclose(o); } }
        else write_ppm(path, hb.map, v->w, v->h * vd, v->fmt);
        if (dump_serial_wanted(v->serial)) { char bp[620]; snprintf(bp, sizeof bp, "%s.bin", path); FILE *o = fopen(bp, "wb"); if (o) { fwrite(hb.map, 1, sz, o); fclose(o); LOG("vk: wrote %s (%llu bytes, format %d)", bp, (unsigned long long)sz, (int)v->fmt); } }
        for (uint64_t i = 0; i < (uint64_t)v->w * v->h; i += 13) { const uint8_t *q = hb.map + i * tb; for (int c = 0; c < 3 && c < (int)tb; c++) mean[c] += q[c]; }
        uint64_t cnt = ((uint64_t)v->w * v->h + 12) / 13;
        LOG("vk: dump %s mean bytes %.1f %.1f %.1f", path, mean[0] / cnt, mean[1] / cnt, mean[2] / cnt);
        free(rgb); vkDestroyBuffer(g_dev, hb.buf, NULL); vkFreeMemory(g_dev, hb.mem, NULL);
    }
    pthread_mutex_unlock(&g_wmu);
}
static uint64_t present_inner(const tex_t *t, const int crop[4]);
uint64_t mtl_present(const tex_t *t, double period_s, const int crop[4]) {
    (void)period_s;
    uint64_t w0 = now_ns(), n = present_inner(t, crop); WT_ADD(WT_PRESENT, now_ns() - w0);
    return n;
}
static uint64_t present_inner(const tex_t *t, const int crop[4]) {
    uint64_t n = ++g_presents;
    if (n % 600 == 0) pcache_maybe_save();
    if (g_tsq && (n == 120 || n % 600 == 0)) { static uint64_t t0, g0, l0, k0; uint64_t t = now_ns(), g = g_gpu_ns_all, l = g_lat_ns_all, k = g_nlat;
        if (t0) LOG("vk: GPU busy %.1f%% of wall time over the last %llu presents (%.2f ms GPU per present); submission commit to completion %.2f ms on average", 100.0 * (g - g0) / (t - t0),
            (unsigned long long)(n == 600 ? 480 : 600), (g - g0) / 1e6 / (n == 600 ? 480 : 600), k > k0 ? (l - l0) / 1e6 / (k - k0) : 0.0);
        t0 = t; g0 = g; l0 = l; k0 = k; }
    if (n % 300 == 0) { double f = 1e6 * 300;   // per present, over the last 300
        uint64_t v[WT_N]; for (int i = 0; i < WT_N; i++) v[i] = atomic_exchange(&g_wt[i], 0);
        LOG("vk: waits per present up to %llu, ms: for a free context %.2f, for a ring chunk %.2f, vkQueueSubmit2 %.2f; present call %.2f, of it slot fence %.2f, acquire %.2f, vkQueuePresentKHR %.2f; composite on the GPU %.2f", (unsigned long long)n,
            v[WT_CTX] / f, v[WT_RING] / f, v[WT_SUBMIT] / f, v[WT_PRESENT] / f, v[WT_PFENCE] / f, v[WT_ACQUIRE] / f, v[WT_QPRESENT] / f, v[WT_COMPOSITE] / f); }
    if (g_trace) { LOG("vk: T end of frame %llu", (unsigned long long)n); g_trace = 0; }
    if (getenv("DM_VK_TRACE") && shot_wanted(n + 1)) { g_trace = 1; g_tr_pass = 0; g_trfs_cnt = 0; LOG("vk: T frame %llu begins", (unsigned long long)n + 1); }
    if (n == 1 || n % 600 == 0) LOG("vk: present #%llu: %llu draws, %llu passes, %llu pipelines, %llu uploads (%llu MB), %llu inline updates, %llu clears / draws under a false render condition; dropped draws %llu/%llu/%llu/%llu/%llu/%llu",
        (unsigned long long)n, (unsigned long long)g_stat_draws, (unsigned long long)g_stat_passes, (unsigned long long)g_stat_pso, (unsigned long long)g_stat_up_n, (unsigned long long)(g_stat_up_bytes >> 20), (unsigned long long)g_stat_ub, (unsigned long long)g_stat_rc_skipped,
        (unsigned long long)g_stat_drop[1], (unsigned long long)g_stat_drop[3], (unsigned long long)g_stat_drop[4], (unsigned long long)g_stat_drop[6], (unsigned long long)g_stat_drop[7], (unsigned long long)g_stat_drop[9]);
    if (n == 1 || n % 600 == 0) LOG("vk: compute: %llu dispatches (%llu indirect), %llu compute pipelines; dropped %llu/%llu/%llu/%llu", (unsigned long long)g_stat_disp, (unsigned long long)g_stat_disp_ind, (unsigned long long)g_ncspl,
        (unsigned long long)g_stat_drop[11], (unsigned long long)g_stat_drop[12], (unsigned long long)g_stat_drop[13], (unsigned long long)g_stat_drop[14]);
    if (n == 1 || n % 600 == 0) LOG("vk: shadow mask %d, recorded / skipped: dynamic state groups %llu / %llu, pipeline binds %llu / %llu, vertex inputs %llu / %llu, vertex buffer binds %llu / %llu; driver blocks reused %llu, built %llu", g_sh,
        (unsigned long long)g_sh_n[0], (unsigned long long)g_sh_n[1], (unsigned long long)g_sh_n[2], (unsigned long long)g_sh_n[3], (unsigned long long)g_sh_n[4], (unsigned long long)g_sh_n[5], (unsigned long long)g_sh_n[6], (unsigned long long)g_sh_n[7], (unsigned long long)g_sh_n[8], (unsigned long long)g_sh_n[9]);
    if (n == 1 || n % 600 == 0) LOG("vk: images: %llu created, %llu recycled, %d waiting in the recycle pool; memory pool changes %u; linear CPU textures compared %llu times, %.1f MB, %llu changed (the largest %ux%u fmt %#x, %llu KB); block compare misses %llu submissions, %llu changes late, mean %.1f / max %llu submissions; texture bindings cached %llu / looked up %llu", (unsigned long long)g_stat_img_new, (unsigned long long)g_stat_img_reuse, g_nrc, (unsigned)atomic_load(&g_pool_gen),
        (unsigned long long)g_lin_cmp_n, g_lin_cmp_bytes / 1048576.0, (unsigned long long)g_lin_up, g_lin_max_w, g_lin_max_h, g_lin_max_fmt, (unsigned long long)(g_lin_max >> 10),
        (unsigned long long)g_lin_miss_sub, (unsigned long long)g_lin_lat_n, g_lin_lat_n ? (double)g_lin_lat_sum / g_lin_lat_n : 0.0, (unsigned long long)g_lin_lat_max, (unsigned long long)g_bc_hit, (unsigned long long)g_bc_miss);
    pthread_mutex_lock(&g_imu); vimg *v = t ? ih_get(t->serial) : NULL; pthread_mutex_unlock(&g_imu);
    if (!v) { if (n < 4) LOG("vk: present #%llu: the window texture was never drawn", (unsigned long long)n); return n; }
    int sx = 0, sy = 0, sw = (int)v->w, sh = (int)v->h;
    if (crop && crop[2] > 0 && crop[3] > 0) { sx = crop[0]; sy = crop[1]; sw = crop[2]; sh = crop[3]; }
    int shot = shot_wanted(n);
    pthread_mutex_lock(&g_wmu);
    if (g_win_changed) { g_win_changed = 0; swapchain_destroy(0); }
    if (g_win && !g_sc) swapchain_make(v->fmt == VK_FORMAT_R8G8B8A8_SRGB || v->fmt == VK_FORMAT_B8G8R8A8_SRGB ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM);
    int have_sc = g_sc != VK_NULL_HANDLE;
    if (!have_sc && !shot) { pthread_mutex_unlock(&g_wmu); return n; }
    // DM_VK_PRESENT_WAIT=1: the present (blit submission + vkQueuePresentKHR) waits on the CPU until the frame's GPU work has completed, so it never reaches the queue while the frame runs (diagnostic for GPU hangs)
    { static int pw = -1; if (pw < 0) { const char *e = getenv("DM_VK_PRESENT_WAIT"); pw = e && atoi(e); if (pw) LOG("vk: DM_VK_PRESENT_WAIT: presents wait for the frame's GPU work (diagnostic)"); }
      if (pw && g_fence && !mtl_wait(g_fence, 10000000000ull)) LOG("[RED] vk: present: submission %llu still running after 10 s", (unsigned long long)g_fence); }
    uint32_t fi = g_frame_i++ % 8;
    if (g_pfence_used[fi]) {
        uint64_t w0 = now_ns(); VkResult fr = vkWaitForFences(g_dev, 1, &g_pfence[fi], VK_TRUE, 2000000000ull); WT_ADD(WT_PFENCE, now_ns() - w0);
        if (fr == VK_TIMEOUT) {   // resetting its fence and command buffer while the GPU still runs them is invalid: this present is dropped instead
            static int nto; if (nto++ < 8) LOG("[RED] vk: present: frame slot %u's previous present has not finished after 2 s (GPU hang?): this present is skipped (#%d)", fi, nto);
            pthread_mutex_unlock(&g_wmu); return n; }
        vkResetFences(g_dev, 1, &g_pfence[fi]); g_pfence_used[fi] = 0;
        if (g_cts_used[fi]) { uint64_t ts[2]; if (vkGetQueryPoolResults(g_dev, g_ctsq, 2 * fi, 2, sizeof ts, ts, sizeof *ts, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && ts[1] > ts[0]) WT_ADD(WT_COMPOSITE, (double)(ts[1] - ts[0]) * g_ts_ns); g_cts_used[fi] = 0; } }
    VkCommandBuffer cb = g_pcb[fi];
    VKCHK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VKCHK(vkBeginCommandBuffer(cb, &bi));
    if (g_ctsq) { vkCmdResetQueryPool(cb, g_ctsq, 2 * fi, 2); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_ctsq, 2 * fi); }
    uint32_t idx = 0; int acquired = 0;
    if (have_sc) {
        uint64_t w0 = now_ns(); VkResult r = pvkAcquireNextImageKHR(g_dev, g_sc, 1000000000ull, g_acq[fi], VK_NULL_HANDLE, &idx); WT_ADD(WT_ACQUIRE, now_ns() - w0);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) { g_win_changed = 1; }
        else if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) { acquired = 1; if (idx >= g_nsci) { RED_ONCE(67, "vk: acquired swapchain image %u of %u: present skipped", idx, g_nsci); acquired = 0; } }   // g_sci / g_scv hold g_nsci entries
        else LOG("[RED] vk: acquire failed: %d", (int)r);
    }
    halloc rb = {0};
    if (shot) {   // readback of the window texture (whole level 0)
        uint64_t sz = (uint64_t)v->w * v->h * vk_texel_bytes(v->fmt);
        static hbuf shotbuf; if (shotbuf.size < sz) { if (shotbuf.buf) { vkDestroyBuffer(g_dev, shotbuf.buf, NULL); vkFreeMemory(g_dev, shotbuf.mem, NULL); } shotbuf = hbuf_new(sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT); }
        rb = (halloc){shotbuf.buf, 0, shotbuf.map};
        VkBufferImageCopy bic = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {v->w, v->h, 1}};
        vkCmdCopyImageToBuffer(cb, v->img, VK_IMAGE_LAYOUT_GENERAL, rb.buf, 1, &bic);
    }
    if (acquired && g_cp.pl) {
        overlay_upload(cb);
        VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_sci[idx], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b}; pvkCmdPipelineBarrier2(cb, &di);
        VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = g_scv[idx], .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue = {.color = {{0, 0, 0, 1}}}};
        VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, g_sce}, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
        pvkCmdBeginRendering(cb, &ri);
        VkViewport vpt = {0, 0, (float)g_sce.width, (float)g_sce.height, 0, 1}; VkRect2D scr = {{0, 0}, g_sce};
        vkCmdSetViewport(cb, 0, 1, &vpt); vkCmdSetScissor(cb, 0, 1, &scr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_cp.pl);
        // letterbox in the window's own orientation: the source aspect fitted in, centred
        double sa = (double)sw / sh, da = (double)g_logical.width / g_logical.height; int dw, dh;
        if (da > sa) { dh = (int)g_logical.height; dw = (int)(dh * sa + .5); } else { dw = (int)g_logical.width; dh = (int)(dw / sa + .5); }
        int dx = ((int)g_logical.width - dw) / 2, dy = ((int)g_logical.height - dh) / 2;
        { const int fr[4] = {dx, dy, dw, dh}; for (int i = 0; i < 4; i++) atomic_store_explicit(&g_frame_rect[i], fr[i], memory_order_relaxed); }
        float lw = (float)g_logical.width, lh = (float)g_logical.height;
        float dst[4] = {2.f * dx / lw - 1, 2.f * dy / lh - 1, 2.f * (dx + dw) / lw - 1, 2.f * (dy + dh) / lh - 1};
        float src[4] = {(float)sx / v->w, (float)sy / v->h, (float)(sx + sw) / v->w, (float)(sy + sh) / v->h};
        if (atomic_load(&g_origin_ll)) { float t = src[1]; src[1] = src[3]; src[3] = t; }
        VkImageView gv = vimg_view(v, VK_IMAGE_VIEW_TYPE_2D, v->fmt, 0, 1, 0, 1, SW_ID, VK_IMAGE_ASPECT_COLOR_BIT);
        if (gv) composite_quad(cb, gv, dst, src, (float)g_rot, 1, 1);
        { pthread_mutex_lock(&g_ovmu); int vis = g_ov.visible; float opa = g_ov.opacity; VkImageView ovv = g_ov.view; pthread_mutex_unlock(&g_ovmu);   // the view changes only on this thread
          if (vis && ovv) { const float full[4] = {-1, -1, 1, 1}, all[4] = {0, 0, 1, 1}; composite_quad(cb, ovv, full, all, (float)g_rot, 0, opa); } }
        pvkCmdEndRendering(cb);
        b.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT; b.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT; b.dstStageMask = VK_PIPELINE_STAGE_2_NONE; b.dstAccessMask = 0;
        b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; pvkCmdPipelineBarrier2(cb, &di);
    } else if (acquired) {   // no composite pipeline: present the image cleared rather than undefined
        VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g_sci[idx], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b}; pvkCmdPipelineBarrier2(cb, &di);
        VkClearColorValue black = {{0, 0, 0, 1}}; VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, g_sci[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rr);
        b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT; b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT; b.dstStageMask = VK_PIPELINE_STAGE_2_NONE; b.dstAccessMask = 0;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; pvkCmdPipelineBarrier2(cb, &di);
    }
    if (g_ctsq) { vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_ctsq, 2 * fi + 1); g_cts_used[fi] = 1; }
    VKCHK(vkEndCommandBuffer(cb));
    pthread_mutex_lock(&g_qmu);   // queue order after the submissions already made: the window texture is complete when this runs
    VkCommandBufferSubmitInfo cbs = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSemaphoreSubmitInfo wait = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_acq[fi], .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSemaphoreSubmitInfo sig = {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_rel[fi], .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = acquired, .pWaitSemaphoreInfos = &wait, .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbs,
        .signalSemaphoreInfoCount = acquired ? 1 : 0, .pSignalSemaphoreInfos = &sig};
    VkResult r = pvkQueueSubmit2(g_q, 1, &si, g_pfence[fi]); g_pfence_used[fi] = 1;
    if (r) { pthread_mutex_unlock(&g_qmu); pthread_mutex_unlock(&g_wmu); die("vk: present submit failed: %d", (int)r); }
    if (acquired) {
        VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1, .pWaitSemaphores = &g_rel[fi], .swapchainCount = 1, .pSwapchains = &g_sc, .pImageIndices = &idx};
        uint64_t w0 = now_ns(); VkResult pr = pvkQueuePresentKHR(g_q, &pi); WT_ADD(WT_QPRESENT, now_ns() - w0);
        if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_ERROR_SURFACE_LOST_KHR) g_win_changed = 1;   // not SUBOPTIMAL: a rotated panel reports it again after every rebuild
        else if (pr == VK_SUBOPTIMAL_KHR && g_rot && g_surf) {   // ... except when the panel turned (sensorLandscape: the phone flipped 180 degrees): rebuild with the new transform
            VkSurfaceCapabilitiesKHR cap; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_pd, g_surf, &cap);
            if (cap.currentTransform != g_sc_transform) g_win_changed = 1;
        }
    }
    pthread_mutex_unlock(&g_qmu);
    if (shot) vkWaitForFences(g_dev, 1, &g_pfence[fi], VK_TRUE, 5000000000ull);
    pthread_mutex_unlock(&g_wmu);
    if (shot) {
        const char *dir = getenv("DM_VK_SHOT_DIR"); if (!dir) dir = getenv("DM_TEMP"); if (!dir) dir = ".";
        char path[600]; snprintf(path, sizeof path, "%s/shot_%llu.ppm", dir, (unsigned long long)n);
        write_ppm(path, rb.p, v->w, v->h, v->fmt);
        LOG("vk: shot %llu: window texture serial %#llx format %d", (unsigned long long)n, (unsigned long long)v->serial, (int)v->fmt);
        if (getenv("DM_VK_DUMP")) dump_targets(n);
    }
    return n;
}
void mtl_present_wait(uint64_t ticket) { (void)ticket; }
uint32_t mtl_present_qos(void) { return 0; }
double mtl_presented_time(uint64_t n) { (void)n; return 0; }
void mtl_ub_present(double iv_ms) { (void)iv_ms; }
void mtl_set_frame(uint64_t f) { (void)f; }
int mtl_has_window(void) { return g_win != NULL; }
int mtl_display_hz(void) { return g_win ? 60 : 0; }
int mtl_fps_target(void) { return 60; }
int mtl_link_active(void) { return 0; }
int mtl_link_wait_tick(int vsyncs, uint64_t timeout_ns) { (void)vsyncs; (void)timeout_ns; return 0; }
void mtl_xv_window_tex(uint64_t serial) { (void)serial; }
void mtl_set_depth_zero(int on) { if (atomic_exchange(&g_depth_zero, on) != on) LOG("vk: depth mode %s", on ? "near=0 (Vulkan's own)" : "near=-w (GL: depth clip control negativeOneToOne)"); }
void mtl_set_origin_lower_left(int on) { atomic_store(&g_origin_ll, on); }

// CPU texture read (ReadTexels): packaged / CPU-written textures from guest memory, GPU-written ones read back
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    size_t wb = (size_t)((r->width + bw - 1) / bw) * bpb, hb = (r->height + bh - 1) / bh, x0 = (size_t)(r->xoffset / bw) * bpb, y0 = r->yoffset / bh;
    int nz = r->depth ? r->depth : 1, is3d = t->d.target == 2;
    pthread_mutex_lock(&g_imu); vimg *v = ih_get(t->serial); int gpu = v && v->gpu_owned; pthread_mutex_unlock(&g_imu);
    if (!gpu) {
        if (!t->mem || !bpb) { memset(dst, 0, img * nz); return; }
        uint32_t lw = MAX(t->d.width >> level, 1u), lh = MAX(t->d.height >> level, 1u), lz = is3d ? MAX(t->d.depth >> level, 1u) : 1;
        size_t lp = (size_t)((lw + bw - 1) / bw) * bpb, lr = (lh + bh - 1) / bh;
        uint8_t *lin = malloc(lp * lr * lz);
        for (int z = 0; z < nz; z++) {
            unsigned layer = is3d ? 0 : layer0 + r->zoffset + z;
            if ((is3d && z == 0) || !is3d) if (nvn_deswizzle(&t->d, level, layer, t->mem, lin, lp)) { RED_ONCE(36, "vk: ReadTexels deswizzle failed"); free(lin); memset(dst, 0, img * nz); return; }
            const uint8_t *src = lin + (is3d ? (size_t)(r->zoffset + z) * lp * lr : 0);
            for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, src + (y0 + y) * lp + x0, wb);
        }
        free(lin); return;
    }
    mtl_finish();
    if (vk_texel_bytes(v->fmt) != (uint32_t)bpb || bw != 1) { RED_ONCE(37, "vk: ReadTexels of a GPU-written texture in a substituted / compressed format: zeros"); memset(dst, 0, img * nz); return; }
    hbuf hb2 = hbuf_new(wb * hb * nz + 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_ppool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    pthread_mutex_lock(&g_wmu);   // g_ppool belongs to the present path
    VkCommandBuffer cb; VKCHK(vkAllocateCommandBuffers(g_dev, &ai, &cb));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; VKCHK(vkBeginCommandBuffer(cb, &bi));
    VkBufferImageCopy bic = {.bufferRowLength = (uint32_t)(wb / bpb), .imageSubresource = {v->aspect & VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : VK_IMAGE_ASPECT_DEPTH_BIT, level, is3d ? 0 : layer0 + r->zoffset, is3d ? 1 : (uint32_t)nz},
        .imageOffset = {r->xoffset, r->yoffset, is3d ? r->zoffset : 0}, .imageExtent = {(uint32_t)r->width, (uint32_t)r->height, is3d ? (uint32_t)nz : 1}};
    vkCmdCopyImageToBuffer(cb, v->img, VK_IMAGE_LAYOUT_GENERAL, hb2.buf, 1, &bic);
    VKCHK(vkEndCommandBuffer(cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; VKCHK(vkCreateFence(g_dev, &fci, NULL, &fe));
    pthread_mutex_lock(&g_qmu);
    VkCommandBufferSubmitInfo cbs = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbs};
    VKCHK(pvkQueueSubmit2(g_q, 1, &si, fe));
    pthread_mutex_unlock(&g_qmu);
    if (vkWaitForFences(g_dev, 1, &fe, VK_TRUE, 5000000000ull) == VK_TIMEOUT) {   // the GPU may still write the buffer and use the fence and command buffer: they stay alive (a small leak) and the game gets zeros
        LOG("[RED] vk: ReadTexels: the GPU copy has not finished after 5 s (GPU hang?): the result is zeros, its fence, command buffer and buffer are left alive");
        pthread_mutex_unlock(&g_wmu); memset(dst, 0, img * nz); return; }
    vkDestroyFence(g_dev, fe, NULL);
    vkFreeCommandBuffers(g_dev, g_ppool, 1, &cb);
    pthread_mutex_unlock(&g_wmu);
    for (int z = 0; z < nz; z++) for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, hb2.map + (z * hb + y) * wb, wb);
    vkDestroyBuffer(g_dev, hb2.buf, NULL); vkFreeMemory(g_dev, hb2.mem, NULL);
    (void)x0; (void)y0;
}

// ---------- Device setup ----------
static VkImageView dummy_image(VkImageViewType vt, VkFormat f, VkImageAspectFlags asp) {
    int cube = vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, is3 = vt == VK_IMAGE_VIEW_TYPE_3D;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0, .imageType = is3 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D, .format = f,
        .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = cube ? 6 : 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VkImage im; VKCHK(vkCreateImage(g_dev, &ci, NULL, &im));
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_dev, im, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)};
    VkDeviceMemory m; VKCHK(vkAllocateMemory(g_dev, &ai, NULL, &m)); VKCHK(vkBindImageMemory(g_dev, im, m, 0));
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im, .viewType = vt, .format = f, .subresourceRange = {asp, 0, 1, 0, cube ? 6 : 1}};
    VkImageView v; VKCHK(vkCreateImageView(g_dev, &vi, NULL, &v));
    // cleared to zero in a one-off command buffer
    VkCommandPool pool; VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = g_qf}; VKCHK(vkCreateCommandPool(g_dev, &pi, NULL, &pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb; VKCHK(vkAllocateCommandBuffers(g_dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}; VKCHK(vkBeginCommandBuffer(cb, &bi));
    img_barrier_init(cb, im, asp);
    VkImageSubresourceRange rr = {asp, 0, 1, 0, cube ? 6 : 1};
    if (asp == VK_IMAGE_ASPECT_COLOR_BIT) { VkClearColorValue z = {{0, 0, 0, 0}}; vkCmdClearColorImage(cb, im, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); }
    else { VkClearDepthStencilValue z = {1, 0}; vkCmdClearDepthStencilImage(cb, im, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr); }
    barrier_all(cb);
    VKCHK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    VKCHK(vkQueueSubmit(g_q, 1, &si, VK_NULL_HANDLE)); VKCHK(vkQueueWaitIdle(g_q));
    vkDestroyCommandPool(g_dev, pool, NULL);
    return v;
}
static void dev_init(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "dmhost", .apiVersion = VK_API_VERSION_1_3};
    const char *iext[] = {"VK_KHR_surface", "VK_KHR_android_surface"};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app, .enabledExtensionCount = 2, .ppEnabledExtensionNames = iext};
    VKCHK(vkCreateInstance(&ici, NULL, &g_inst));
    uint32_t n = 1; vkEnumeratePhysicalDevices(g_inst, &n, &g_pd); if (!n) die("vk: no GPU");
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(g_pd, &pp); g_lim = pp.limits;
    LOG("vk: the driver allows %u memory allocations", g_lim.maxMemoryAllocationCount);
    vkGetPhysicalDeviceMemoryProperties(g_pd, &g_mp);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, NULL); VkQueueFamilyProperties qp[16]; nq = MIN(nq, 16u); vkGetPhysicalDeviceQueueFamilyProperties(g_pd, &nq, qp);
    g_qf = 0; for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { g_qf = i; break; }
    uint32_t ne = 0; vkEnumerateDeviceExtensionProperties(g_pd, NULL, &ne, NULL); VkExtensionProperties *ep = calloc(ne, sizeof *ep); vkEnumerateDeviceExtensionProperties(g_pd, NULL, &ne, ep);
    const char *need[] = {"VK_KHR_swapchain", "VK_KHR_push_descriptor", "VK_EXT_external_memory_host", "VK_EXT_extended_dynamic_state3", "VK_EXT_vertex_input_dynamic_state",
                          "VK_EXT_custom_border_color", "VK_EXT_depth_clip_control", "VK_EXT_robustness2"};
    const char *dext[16]; int nd = 0;
    for (unsigned i = 0; i < sizeof need / sizeof *need; i++) { if (!has_ext(ep, ne, need[i])) die("vk: the GPU lacks %s", need[i]); dext[nd++] = need[i]; }
    if ((g_has_u8 = has_ext(ep, ne, "VK_EXT_index_type_uint8"))) dext[nd++] = "VK_EXT_index_type_uint8";
    if ((g_has_divisor = has_ext(ep, ne, "VK_EXT_vertex_attribute_divisor"))) dext[nd++] = "VK_EXT_vertex_attribute_divisor";
    free(ep);
    // features: everything the device has, minus robust buffer / image access (cost; null descriptors come from robustness2)
    VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT dv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT};
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT u8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT, g_has_divisor ? &dv : NULL};
    VkPhysicalDeviceRobustness2FeaturesEXT rb = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, g_has_u8 ? (void *)&u8 : (g_has_divisor ? (void *)&dv : NULL)};
    VkPhysicalDeviceDepthClipControlFeaturesEXT dcc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT, &rb};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT, &dcc};
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vid = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT, &cbc};
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT e3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT, &vid};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &e3};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    vkGetPhysicalDeviceFeatures2(g_pd, &f2);
    g_has_dsb = f2.features.dualSrcBlend != 0;   // enabled below with everything else the device has
    {
        VkPhysicalDeviceVulkan13Properties p13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
        VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p13};
        vkGetPhysicalDeviceProperties2(g_pd, &p2);
        g_cs_sub32 = f13.subgroupSizeControl && (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) && p13.minSubgroupSize <= 32 && p13.maxSubgroupSize >= 32;
        LOG("vk: subgroup sizes %u to %u; compute fixed at 32: %d", p13.minSubgroupSize, p13.maxSubgroupSize, g_cs_sub32);
    }
    if (!f12.shaderOutputLayer) LOG("[RED] vk: no shaderOutputLayer: layered passes (colour grading LUT) draw into their first layer only");
    // DM_VK_ROBUST=1 keeps robust buffer/image access (as the driver reports it): out-of-bounds reads return zero instead of faulting (diagnostic for GPU hangs)
    { const char *e = getenv("DM_VK_ROBUST"); if (e && atoi(e)) LOG("vk: DM_VK_ROBUST: robustBufferAccess %u robustImageAccess %u robustBufferAccess2 %u robustImageAccess2 %u (diagnostic)", f2.features.robustBufferAccess, f13.robustImageAccess, rb.robustBufferAccess2, rb.robustImageAccess2);
      else { f2.features.robustBufferAccess = 0; f13.robustImageAccess = 0; rb.robustBufferAccess2 = 0; rb.robustImageAccess2 = 0; } }
    if (!f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore || !vid.vertexInputDynamicState || !rb.nullDescriptor || !dcc.depthClipControl)
        die("vk: missing features (dynamicRendering %u synchronization2 %u timelineSemaphore %u vertexInputDynamicState %u nullDescriptor %u depthClipControl %u)",
            f13.dynamicRendering, f13.synchronization2, f12.timelineSemaphore, vid.vertexInputDynamicState, rb.nullDescriptor, dcc.depthClipControl);
    float prio = 1; VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = g_qf, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi, .enabledExtensionCount = (uint32_t)nd, .ppEnabledExtensionNames = dext};
    VKCHK(vkCreateDevice(g_pd, &dci, NULL, &g_dev));
    vkGetDeviceQueue(g_dev, g_qf, 0, &g_q);
#define VK_LOAD(f) if (!(p##f = (PFN_##f)vkGetDeviceProcAddr(g_dev, #f))) die("vk: no %s", #f);
    VK_FNS(VK_LOAD)
    LOG("vk: %s, Vulkan %u.%u.%u, driver %#x; extensions: uint8 indices %d, instance divisors %d", pp.deviceName, VK_VERSION_MAJOR(pp.apiVersion), VK_VERSION_MINOR(pp.apiVersion), VK_VERSION_PATCH(pp.apiVersion), pp.driverVersion, g_has_u8, g_has_divisor);
    VkSemaphoreTypeCreateInfo st = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &st}; VKCHK(vkCreateSemaphore(g_dev, &sci, NULL, &g_tl));
    VkSemaphoreCreateInfo bsci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (int i = 0; i < 8; i++) { VKCHK(vkCreateSemaphore(g_dev, &bsci, NULL, &g_acq[i])); VKCHK(vkCreateSemaphore(g_dev, &bsci, NULL, &g_rel[i])); }
    if (g_lim.timestampPeriod > 0 && !getenv("DM_VK_NO_GPU_TIME")) {
        VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 * NCTX};
        if (vkCreateQueryPool(g_dev, &qpi, NULL, &g_tsq) == VK_SUCCESS) { g_ts_ns = g_lim.timestampPeriod; LOG("vk: GPU timestamps on (%.2f ns per tick)", g_ts_ns); } else g_tsq = VK_NULL_HANDLE;
    }
    { const char *e = getenv("DM_VK_NCTX"); int n = e ? atoi(e) : g_nctx; g_nctx = n < 1 ? 1 : n > NCTX ? NCTX : n; LOG("vk: %d submissions in flight at most (DM_VK_NCTX)", g_nctx); }
    pp_init(qp[g_qf].timestampValidBits);   // DM_VK_PASSPROF: per-pass timing (needs g_tsq's timestampPeriod)
    if (g_pp.on) { VkQueryPoolCreateInfo qi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 16}; if (vkCreateQueryPool(g_dev, &qi, NULL, &g_ctsq) != VK_SUCCESS) g_ctsq = VK_NULL_HANDLE; }
    for (int i = 0; i < g_nctx; i++) {
        VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, .queueFamilyIndex = g_qf};
        VKCHK(vkCreateCommandPool(g_dev, &pi, NULL, &g_ctx[i].pool));
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_ctx[i].pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 2};
        VkCommandBuffer cbs[2]; VKCHK(vkAllocateCommandBuffers(g_dev, &ai, cbs)); g_ctx[i].cb = g_ctx[i].xcb[0] = cbs[0]; g_ctx[i].up = g_ctx[i].xup[0] = cbs[1]; g_ctx[i].nx = 1;
    }
    VkCommandPoolCreateInfo ppi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_qf};
    VKCHK(vkCreateCommandPool(g_dev, &ppi, NULL, &g_ppool));
    VkCommandBufferAllocateInfo pai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_ppool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 8};
    VKCHK(vkAllocateCommandBuffers(g_dev, &pai, g_pcb));
    for (int i = 0; i < 8; i++) { VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VKCHK(vkCreateFence(g_dev, &fci, NULL, &g_pfence[i])); }
    for (int i = 0; i < RING_N; i++) g_ring.h[i] = hbuf_new(RING_CHUNK, BUF_ALL);
    g_ring.limit = RING_CHUNK; { const char *e = getenv("DM_VK_RING_LIMIT"); if (e && atoi(e) > 0) g_ring.limit = MIN((uint64_t)atoi(e) << 10, (uint64_t)RING_CHUNK); }
    {   // zero buffer (unbound vertex attributes, unbound uniform slots), Ryujinx support buffer (render scale 1), shared quad indices
        halloc z = static_alloc(65536, 256); memset(z.p, 0, 65536); g_zero_buf = z.buf;
        if (z.off) die("vk: the zero buffer must start the static arena");
        g_support = static_alloc(1392, 256); memset(g_support.p, 0, 1392);
        g_sb_scratch = static_alloc(65536, 256); memset(g_sb_scratch.p, 0, 65536);
        for (int i = 0; i < 73; i++) { float one = 1.f; memcpy(g_support.p + 192 + 16 * i, &one, 4); }
        float vp[4] = {1, 1, 1, 1}; memcpy(g_support.p + 144, vp, 16); memcpy(g_support.p + 160, vp, 16);
        g_quad_ib = static_alloc(65536 * 6 * 4, 4); uint32_t *ix = (uint32_t *)g_quad_ib.p;
        for (uint32_t q = 0; q < 65536; q++) { uint32_t b = 4 * q; uint32_t v[6] = {b, b + 1, b + 2, b, b + 2, b + 3}; memcpy(ix + 6 * q, v, 24); }
    }
    static const VkImageViewType vts[5] = {VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, VK_IMAGE_VIEW_TYPE_3D};
    for (int i = 0; i < 5; i++) g_dummy_view[i] = dummy_image(vts[i], VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    g_dummy_depth_view[0] = dummy_image(VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
    g_dummy_depth_view[1] = dummy_image(VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT);
    { samp_par dp = {1, 1, 7, 7, 7, 0, 4, 0, 0, 1000, 0, 1, {0, 0, 0, 0}}; g_dummy_samp = sampler_get(&dp, 0); }
    {   VkPipelineCacheCreateInfo pci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO}; const char *pc = getenv("DM_VK_PCACHE"); size_t n = 0; void *d = pc && *pc ? read_file(pc, &n) : NULL;
        pci.initialDataSize = d ? n : 0; pci.pInitialData = d;
        if (d && vkCreatePipelineCache(g_dev, &pci, NULL, &g_pcache) == VK_SUCCESS) LOG("vk: pipeline cache loaded from %s (%zu KB)", pc, n / 1024);
        else { if (d) LOG("[RED] vk: the pipeline cache %s was refused: starting empty", pc); pci.initialDataSize = 0; pci.pInitialData = NULL; VKCHK(vkCreatePipelineCache(g_dev, &pci, NULL, &g_pcache)); }
        free(d); }
    refl_load();
    pthread_t th; pthread_create(&th, NULL, completion_main, NULL); pthread_detach(th);
}

// ---------- Shader pack probe ----------
// DM_VK_PROBE=<n> in env.txt (front_android.c): no game is started. Every compute shader, every domain shader (with a hull shader, by turns), every geometry shader and n vertex + pixel pairs in the shader pack go through
// shader_get / cpipeline_get / pipeline_get, and the driver's answers and times are logged, so a shader pack can meet a device's driver before any game code runs. Shaders are paired by their place in the pack, not as
// the game pairs them: this shows what the driver accepts and what a compile costs, not that anything draws. Nothing is submitted. DM_VK_PROBE_FROM=<i>: skip the first i of each list (after a driver crash: probe_at.txt
// beside the pack names the list and place); DM_VK_PROBE_MINUTES=<m>: a list stops after m minutes (default 30); DM_VK_PCACHE= (empty) in env.txt: times without the pipeline cache kept from an earlier run
typedef struct { const char *name; uint32_t *ms; int n, bad, total; uint64_t t0, end; } probe_run;
static int g_probe_at = -1;
static void probe_begin(probe_run *r, const char *name, int total, int minutes) {
    memset(r, 0, sizeof *r); r->name = name; r->total = total; r->ms = calloc((size_t)total + 1, sizeof *r->ms); r->t0 = now_ns(); r->end = r->t0 + (uint64_t)minutes * 60000000000ull;
}
static int probe_more(probe_run *r, int i, uint32_t e) {   // false: the list is out of time. The place goes to a file at once: the log thread may lose the last lines when a driver crash ends the process
    if (now_ns() > r->end) { LOG("probe: %s: stopped by the time limit at %d of %d", r->name, i, r->total); return 0; }
    if (r->n && !(r->n % 50)) LOG("probe: %s: %d of %d, %llu s", r->name, i, r->total, (unsigned long long)((now_ns() - r->t0) / 1000000000ull));
    char b[64]; int n = snprintf(b, sizeof b, "%-12s %6d %016llx\n", r->name, i, (unsigned long long)g_refl[e].key);
    if (g_probe_at >= 0) (void)!pwrite(g_probe_at, b, (size_t)n, 0);
    return 1;
}
static int probe_cmp(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return x < y ? -1 : x > y; }
static void probe_report(probe_run *r) {
    if (!r->n) { LOG("probe: %s: nothing created", r->name); free(r->ms); return; }
    qsort(r->ms, (size_t)r->n, sizeof *r->ms, probe_cmp);
    uint64_t sum = 0; for (int i = 0; i < r->n; i++) sum += r->ms[i];
    LOG("probe: %s: %d tried, %d refused; ms: median %u, 90%% %u, 99%% %u, worst %u; %llu s in all", r->name, r->n, r->bad, r->ms[r->n / 2], r->ms[r->n * 9 / 10], r->ms[r->n * 99 / 100], r->ms[r->n - 1], (unsigned long long)(sum / 1000));
    free(r->ms);
}
static vshader *probe_get(int64_t e) {
    if (e < 0) return NULL;
    uint8_t k[8]; for (int j = 0; j < 8; j++) k[j] = (uint8_t)(g_refl[e].key >> (56 - 8 * j));
    return shader_get2(k, 0);
}
static void probe_put(int64_t e) {   // the module goes and the entry is forgotten; the vshader stays allocated, because the pipeline tables compare by its address
    if (e < 0 || !g_refl_sh[e]) return;
    if (g_refl_sh[e]->mod) vkDestroyShaderModule(g_dev, g_refl_sh[e]->mod, NULL);
    g_refl_sh[e]->mod = VK_NULL_HANDLE; g_refl_sh[e] = NULL;
}
static void probe_compute(probe_run *r, int64_t e) {
    uint64_t t0 = now_ns(), head = g_static_head; int ok = 0;
    vshader *s = probe_get(e);
    if (s) { cspent *c = cpipeline_get(s); if (c->pl) { ok = 1; vkDestroyPipeline(g_dev, c->pl, NULL); c->pl = VK_NULL_HANDLE; } }
    probe_put(e); g_static_head = head;   // the shader's c1 block is not used again
    r->ms[r->n++] = (uint32_t)((now_ns() - t0) / 1000000); r->bad += !ok;
}
static void probe_graphics(probe_run *r, int64_t ve, int64_t fe, int64_t ge, int64_t he, int64_t de) {
    uint64_t t0 = now_ns(), head = g_static_head; int ok = 0;
    vshader *vs = probe_get(ve), *fs = probe_get(fe), *gs = probe_get(ge), *hs = probe_get(he), *ds = probe_get(de);
    if (vs && (fe < 0 || fs) && (ge < 0 || gs) && (he < 0 || hs) && (de < 0 || ds)) {
        pkey k; memset(&k, 0, sizeof k);
        k.vs = vs; k.fs = fs; k.gs = gs; k.hs = hs; k.ds = ds; k.patch = hs ? 3 : 0; k.nc = 4; k.samples = 1; k.topo = hs ? 3 : 2; k.dz = 1; k.df = VK_FORMAT_D32_SFLOAT;
        for (int i = 0; i < 4; i++) k.cf[i] = VK_FORMAT_R8G8B8A8_UNORM;
        pent *p = pipeline_get(&k);
        if (p->pl) { ok = 1; vkDestroyPipeline(g_dev, p->pl, NULL); p->pl = VK_NULL_HANDLE; }
    }
    probe_put(ve); probe_put(fe); probe_put(ge); probe_put(he); probe_put(de); g_static_head = head;
    r->ms[r->n++] = (uint32_t)((now_ns() - t0) / 1000000); r->bad += !ok;
}
void dm_vk_probe(int pairs) {
    ensure_init();
    if (!g_nrefl) { LOG("[RED] probe: no shader pack"); return; }
    { const char *d = getenv("DM_VK_SHADERS"); char p[600]; snprintf(p, sizeof p, "%s/probe_at.txt", d ? d : "vk"); g_probe_at = open(p, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644); }
    const char *ef = getenv("DM_VK_PROBE_FROM"), *em = getenv("DM_VK_PROBE_MINUTES"); int from = ef ? atoi(ef) : 0, minutes = em && atoi(em) > 0 ? atoi(em) : 30;
    uint32_t *l[6]; int nl[6] = {0}; for (int s = 0; s < 6; s++) l[s] = malloc(g_nrefl * sizeof *l[s]);
    for (uint32_t i = 0; i < g_nrefl; i++) {
        char b[2]; if (!pack_read(b, 2, g_refl[i].off)) { LOG("[RED] probe: cannot read shader %u from the pack", i); return; }
        int s = !memcmp(b, "vs", 2) ? 0 : !memcmp(b, "ps", 2) ? 1 : !memcmp(b, "gs", 2) ? 2 : !memcmp(b, "hs", 2) ? 3 : !memcmp(b, "ds", 2) ? 4 : !memcmp(b, "cs", 2) ? 5 : -1;
        if (s >= 0) l[s][nl[s]++] = i;
    }
    VkPhysicalDevicePushDescriptorPropertiesKHR pd = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR}; VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &pd}; vkGetPhysicalDeviceProperties2(g_pd, &p2);
    LOG("probe: pack has vs %d, ps %d, gs %d, hs %d, ds %d, cs %d; %d minutes a list, from place %d; limits: compute shared memory %u, workgroup invocations %u, push descriptors %u", nl[0], nl[1], nl[2], nl[3], nl[4], nl[5], minutes, from, g_lim.maxComputeSharedMemorySize, g_lim.maxComputeWorkGroupInvocations, pd.maxPushDescriptors);
    probe_run r;
    probe_begin(&r, "compute", nl[5], minutes);
    for (int i = from; i < nl[5] && probe_more(&r, i, l[5][i]); i++) probe_compute(&r, l[5][i]);
    probe_report(&r);
    if (nl[0]) {
        int nv = 0;   // a new vertex shader each time, wrapping
        #define PROBE_VS l[0][nv++ % nl[0]]
        probe_begin(&r, "geometry", nl[2], minutes);
        for (int i = from; i < nl[2] && probe_more(&r, i, l[2][i]); i++) probe_graphics(&r, PROBE_VS, -1, l[2][i], -1, -1);
        probe_report(&r);
        probe_begin(&r, "tessellation", nl[3] ? nl[4] : 0, minutes);
        for (int i = from; nl[3] && i < nl[4] && probe_more(&r, i, l[4][i]); i++) probe_graphics(&r, PROBE_VS, -1, -1, l[3][i % nl[3]], l[4][i]);
        probe_report(&r);
        if (pairs > nl[1]) pairs = nl[1];
        probe_begin(&r, "vertex+pixel", pairs, minutes);
        for (int i = from; i < pairs && probe_more(&r, i, l[1][i * (int64_t)nl[1] / pairs]); i++) probe_graphics(&r, PROBE_VS, l[1][i * (int64_t)nl[1] / pairs], -1, -1, -1);
        probe_report(&r);
        #undef PROBE_VS
    }
    FILE *f = fopen("/proc/self/status", "r"); char line[128];
    if (f) { while (fgets(line, sizeof line, f)) if (!strncmp(line, "VmHWM", 5) || !strncmp(line, "VmRSS", 5)) LOG("probe: %.*s", (int)strcspn(line, "\n"), line); fclose(f); }
    for (int s = 0; s < 6; s++) free(l[s]);
    if (g_probe_at >= 0) { close(g_probe_at); g_probe_at = -1; }
    LOG("probe: done");
}

// ---------- Statistics the NVN layer reads ----------
double mtl_gpu_ms(uint64_t f) {   // GPU time of the submissions made while frame f was current; -1 until all of them completed (or no timestamps)
    if (!g_tsq || g_gf[f % 64].frame != f || g_gf[f % 64].done < g_gf[f % 64].sub) return -1;
    return g_gf[f % 64].ns / 1e6;
}
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; (void)t0; (void)t1; return 0; }
void mtl_stats(uint64_t *passes, uint64_t *draws) { *passes = g_stat_passes; *draws = g_stat_draws; }
void mtl_bw(uint64_t *load, uint64_t *store, uint64_t *full) { *load = *store = *full = 0; }
void mtl_args_stats(uint64_t *enc, uint64_t *reuse) { *enc = *reuse = 0; }
void mtl_alloc_stats(uint64_t *a, uint64_t *b, int *n) { *a = *b = 0; *n = 0; }
void mtl_create_ns(uint64_t *lib, uint64_t *pso) { *lib = *pso = 0; }
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { *lib = 0; *pso = g_stat_pso; }
void mtl_pkg_stats(uint64_t *up, uint64_t *re) { *up = g_stat_up_n; *re = 0; }
void mtl_use_stats(uint64_t *call, uint64_t *skip) { *call = *skip = 0; }
void mtl_vbuf_stats(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d, uint64_t *e, uint64_t *f) { *a = *b = *c = *d = *e = *f = 0; }
// The waits of a sync WindowAcquireTexture handed out (sync_t.fenced 2): the same wait as mtl_wait, only a line of its own in nvn_mtl.m's wait ledger (which this layer does not keep)
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { return mtl_wait(fence, timeout_ns); }
// RegisterImage declares image use even when the builder left IMAGE out: the snapshot gets the IMAGE flag (nvn_mtl.m); the replay creates storage views only on demand
void mtl_reg_image(int id, const tex_t *t, const view_t *v) {
    if (!t) { mtl_reg_tex(id, NULL, v); return; }
    tex_t image = *t; image.d.flags |= NVN_TEXTURE_FLAGS_IMAGE; mtl_reg_tex(id, &image, v);
}
// The Metal replay's prewarm and background library load: shaders and pipelines load at first use here
void mtl_prepare(void) {}
void mtl_prog_prefetch(const prog_t *p) { (void)p; }
// Statistics and tuning that nvn_mtl.m keeps and this layer does not: zero (hle_nvn.c's per-second report prints them, nothing branches on them except the display link, which is never active here)
void mtl_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) { (void)f; (void)gpu_ms; (void)frame_ms; }
uint64_t mtl_copy_cut_count(void) { return 0; }
unsigned mtl_cpu_tex_count(void) { return 0; }
void mtl_cshoist_stats(uint64_t *m, uint64_t *d, uint64_t s[8]) { *m = *d = 0; memset(s, 0, 8 * sizeof *s); }
void mtl_geohoist_stats(uint64_t *h, uint64_t *r, uint64_t s[6]) { *h = *r = 0; memset(s, 0, 6 * sizeof *s); }
void mtl_geomesh_stats(uint64_t *n, uint64_t *w) { *n = *w = 0; }
void mtl_hybrid_stats(uint64_t *draws, uint64_t *made, uint64_t *none) { *draws = *made = *none = 0; }
int mtl_link_rate(void) { return 60; }   // no display link here (mtl_link_active() is 0), so nothing paces on it
int mtl_link_stalled(void) { return 0; }
int mtl_link_tick(uint64_t n, double *target, double *lead_ms, int *ticks, double *gap_ms) { (void)n; (void)target; (void)lead_ms; (void)ticks; (void)gap_ms; return 0; }
int mtl_link_wait_ticks(int ticks, uint64_t timeout_ns) { (void)ticks; (void)timeout_ns; return 0; }
int mtl_link_wait_vsync(uint64_t *seen, uint64_t timeout_ns) { (void)seen; (void)timeout_ns; return 0; }   // hle_sdk16.c has a weak one; the vsync thread then paces itself
uint64_t mtl_pkg_edge_skips(void) { return 0; }
void mtl_pso_line(char *out, size_t n) { if (n) snprintf(out, n, "vk pipelines %llu", (unsigned long long)g_stat_pso); }
int mtl_pso_prebuild_tool(void) { return 0; }   // DM_PSO_PREBUILD=tool is a Metal pipeline-archive tool: never taken here
void mtl_sm_stats(uint64_t *a, uint64_t *b, uint64_t *c) { *a = *b = *c = 0; }
int mtl_snap_on(void) { return 0; }   // present snapshots (DM_PRESENT_SNAP) are a Metal display-link feature
uint64_t mtl_snap_acquire_wait(void) { return 0; }
// hle_nvn.c's Mac-only power report (task_info) also names these; weak so a port that keeps the report links, and one that defines them elsewhere wins
__attribute__((weak)) void mtl_res_stats(char *out, size_t n) { if (n) out[0] = 0; }
__attribute__((weak)) void mtl_barrier_stats(uint64_t *skip, uint64_t *full) { *skip = *full = 0; }
__attribute__((weak)) uint64_t mtl_cut_count(void) { return 0; }
__attribute__((weak)) uint64_t mtl_blit_splits(void) { return 0; }
__attribute__((weak)) void mtl_ubd_stats(uint64_t *a, uint64_t *b) { *a = *b = 0; }
__attribute__((weak)) void mtl_ubp_stats(uint64_t *a, uint64_t *b, uint64_t *c) { *a = *b = *c = 0; }
__attribute__((weak)) void mtl_lc_stats(uint64_t *a, uint64_t *b) { *a = *b = 0; }
__attribute__((weak)) void mtl_ubp_why(uint64_t w[5]) { memset(w, 0, 5 * sizeof *w); }
__attribute__((weak)) void mtl_mem_stat(unsigned long long frame, unsigned long long mb) { (void)frame; (void)mb; }
uint64_t mtl_pkg_miss(void) { return 0; }
uint64_t mtl_upload_bytes(void) { return g_stat_up_bytes; }
uint64_t mtl_upload_ns_take(void) { return 0; }
uint64_t mtl_upload_gpu_ns_take(void) { return 0; }
uint64_t mtl_drawable_ns_take(void) { return 0; }
uint64_t mtl_udc_saved_bytes(void) { return 0; }
uint64_t mtl_ds_omit(void) { return 0; }
