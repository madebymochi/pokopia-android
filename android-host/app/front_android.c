// NativeActivity front end (the APK): the window goes to the Vulkan replay (dm_vk_set_window), touches and game controllers go to nn::hid.
// android_main (native_app_glue's thread) owns the activity's events; the host's main() runs on its own thread, where front_run runs the guest.
// Data: the app's folder on shared storage (Android/data/<package>/files, which a PC sees over USB), or DM_DATA from env.txt there; it holds mods/, root/romfs,
// vk/ and font/ (android-host/build_android.py makes them from the player's dump). The save is there too (save/), so it can be backed up from a PC; a save kept
// inside the app by an older development build (another package) is not read. The log goes to logcat (tag pokopia) and to
// log.txt in that folder (log.prev.txt: the run before).
// env.txt lines VAR=value set further environment (DM_LANG, DM_VK_SHOT, ...).
// The window is immersive (system bars hidden, swipe to show them, drawn under the camera cutout); sound is on (DM_AUDIO=1, AAudio); the on-screen pad
// is touch_controls.c, with a haptic tick on each press (DM_HAPTICS=0: off).
// native_app_glue is built into this file so its ANativeActivity_onCreate can be wrapped: the immersive setup needs the UI thread, which only the
// activity's callbacks run on
#define ANativeActivity_onCreate glue_ANativeActivity_onCreate
#include <android_native_app_glue.c>
#undef ANativeActivity_onCreate
#include <android/log.h>
#include <android/window.h>
#include <jni.h>
#include <pthread.h>
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <dirent.h>
#include <dlfcn.h>
#include <signal.h>
#include <stdatomic.h>
#include "front.h"
#include "dm.h"
#include "dm_input.h"
#include "touch_controls.h"
#include "pad_map.h"

int main(int argc, char **argv);   // the host (dm/host/main.c), here a function of this library
void dm_vk_set_window(void *w);
void dm_vk_probe(int pairs);
void dm_vk_overlay_image(const uint8_t *rgba, int w, int h, float opacity);
void dm_vk_frame_rect(int out[4]);
int dm_vk_present_linear(void);
void dm_focus_set(int st);                         // hle.h
void dm_audio_suspend(int on);
void guest_clock_freeze_for(int reason, int on);

static struct android_app *g_app;
static char g_data[512], g_internal[512];
static int g_started;

// ---------- Log: stdout / stderr → logcat and last.log ----------
static int g_logfd = -1, g_logrd = -1;
static void *log_thread(void *a) {
    int fd = (int)(intptr_t)a; char buf[2048]; size_t n = 0;
    for (;;) {
        ssize_t r = read(fd, buf + n, sizeof buf - 1 - n);
        if (r <= 0) break;
        n += (size_t)r;
        char *s = buf, *nl;
        while ((nl = memchr(s, '\n', n - (size_t)(s - buf)))) {
            *nl = 0; __android_log_write(ANDROID_LOG_INFO, "pokopia", s);
            if (g_logfd >= 0) { *nl = '\n'; write(g_logfd, s, (size_t)(nl - s + 1)); }
            s = nl + 1;
        }
        n -= (size_t)(s - buf); memmove(buf, s, n);
        if (n == sizeof buf - 1) { buf[n] = 0; __android_log_write(ANDROID_LOG_INFO, "pokopia", buf); if (g_logfd >= 0) write(g_logfd, buf, n); n = 0; }   // a line this long: logcat gets it in chunks, log.txt in full (the rest follows, no newline added)
    }
    return NULL;
}
static void log_init(const char *dir) {
    char p[600], q[600]; snprintf(p, sizeof p, "%s/log.txt", dir); snprintf(q, sizeof q, "%s/log.prev.txt", dir);
    rename(p, q);
    g_logfd = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    int fds[2]; if (pipe(fds)) return;
    setvbuf(stdout, NULL, _IOLBF, 0); setvbuf(stderr, NULL, _IONBF, 0);
    dup2(fds[1], 1); dup2(fds[1], 2);
    g_logrd = fds[0];
    pthread_t th; pthread_create(&th, NULL, log_thread, (void *)(intptr_t)fds[0]); pthread_detach(th);
}
// The host is about to _exit (an SDK Break, a crash): give the log thread up to 0.5 s to empty the pipe, then a moment to write its last chunk (the Break line and its
// backtrace never reached logcat or last.log before). Only async-signal-safe calls: crash.c calls it from its signal handler
void dm_log_drain(void) {
    for (int i = 0; i < 50; i++) { int n = 0; if (g_logrd < 0 || ioctl(g_logrd, FIONREAD, &n) || n <= 0) break; usleep(10000); }
    usleep(30000);
}

// ---------- Paths and environment ----------
static void mkdirs(const char *p) {
    char t[600]; snprintf(t, sizeof t, "%s", p);
    for (char *s = t + 1; *s; s++) if (*s == '/') { *s = 0; mkdir(t, 0755); *s = '/'; }
    mkdir(t, 0755);
}
static void env_file(const char *path) {   // VAR=value lines; # comments
    FILE *f = fopen(path, "r"); if (!f) return;
    char line[1024]; int first = 1;
    while (fgets(line, sizeof line, f)) {
        char *l = line; if (first && !strncmp(l, "\xEF\xBB\xBF", 3)) l += 3;   // a byte order mark (Notepad) would end up in the first name
        first = 0;
        l[strcspn(l, "\r\n")] = 0;
        while (*l == ' ' || *l == '\t') l++;
        char *eq = strchr(l, '=');
        if (l[0] == '#' || !eq) continue;
        char *nm = l, *v = eq + 1, *ne = eq; *eq = 0;
        while (ne > nm && (ne[-1] == ' ' || ne[-1] == '\t')) *--ne = 0;
        while (*v == ' ' || *v == '\t') v++;
        char *ve = v + strlen(v); while (ve > v && (ve[-1] == ' ' || ve[-1] == '\t')) *--ve = 0;
        if (!*nm) continue;
        setenv(nm, v, 1);
        LOG("front: env %s=%s (%s)%s", nm, v, path, strncmp(nm, "DM_", 3) ? " [not a DM_ name: a typo?]" : "");
    }
    fclose(f);
}
static int readable_layout(const char *d) {
    char p[600]; snprintf(p, sizeof p, "%s/mods", d);
    if (access(p, R_OK | X_OK)) return 0;
    snprintf(p, sizeof p, "%s/vk/shaders.pack", d);
    int fd = open(p, O_RDONLY); if (fd < 0) return 0;
    char c; int ok = read(fd, &c, 1) == 1; close(fd); return ok;
}
static void paths_init(ANativeActivity *act) {
    snprintf(g_internal, sizeof g_internal, "%s", act->internalDataPath ? act->internalDataPath : "/data/data/io.github.madebymochi.pokopia/files");
    const char *ext = act->externalDataPath ? act->externalDataPath : g_internal;
    umask(007);   // the shared folder's group (adb, the PC's file transfer) may replace files there, a save among them; folders get the group's bits in compat's mkdir
    mkdirs(g_internal); mkdirs(ext);
    log_init(ext);
    char p[600]; snprintf(p, sizeof p, "%s/env.txt", ext); env_file(p);
    const char *d = getenv("DM_DATA");
    snprintf(g_data, sizeof g_data, "%s", d && readable_layout(d) ? d : ext);
    LOG("front: data %s%s, internal %s", g_data, readable_layout(g_data) ? "" : " ([RED] no mods/ or vk/shaders.pack there: copy the game folder made by android-host/build_android.py)", g_internal);
    {   // the rewritten modules must come from the patch_nso.py this host was built with (its gate numbers): build_android.py writes the hash build.sh baked in here
        char rw[600], id[40] = ""; snprintf(rw, sizeof rw, "%s/mods/rewriter.txt", g_data);
        FILE *f = fopen(rw, "r"); if (f) { if (fscanf(f, "%39s", id) != 1) id[0] = 0; fclose(f); }
        if (!id[0]) LOG("front: no mods/rewriter.txt: rewriter not checked");
        else if (strcmp(id, DM_REWRITER_ID)) LOG("front: [RED] mods/ were rewritten by patch_nso.py %s but this APK was built with %s: build the APK and the game folder again from one checkout", id, DM_REWRITER_ID);
        else LOG("front: rewriter %s matches", id);
    }
    char s[600];
    snprintf(s, sizeof s, "%s/root/", g_data); setenv("DM_ROOT", s, 0);
    snprintf(s, sizeof s, "%s/aoc", g_data); setenv("DM_AOC_DIR", s, 0);   // the DLC RomFS folders aoc/<index>/ (build_android.py); none there = no DLC
    // Device save data. Builds on nxnative 76a2976 kept it in <DM_SAVE>_device = save/user_device; nxnative 7a3a875 mounts device
    // save data at DM_SAVE itself when main mounts no account save (a main that imports only MountDeviceSaveData), so DM_SAVE names that same folder:
    // a save from an older build is found where it is, and the README's backup path stays right
    snprintf(s, sizeof s, "%s/save/user_device", ext);
    mkdirs(s); setenv("DM_SAVE", s, 0);
    snprintf(s, sizeof s, "%s/save/temp", g_internal); mkdirs(s); setenv("DM_TEMP", s, 0);
    snprintf(s, sizeof s, "%s/font", g_data); setenv("DM_FONT_DIR", s, 0);
    {   // Shared-font stand-in. First choice: font/NotoSansCJKjp.ttf, a TrueType (glyf) copy of the phone's own Noto Sans CJK made by tools/cjk_shared_font.py: a game's ScalableFont
        // asks the shared font for Hangul / CJK glyphs, and with Roboto (Latin only) it crashed twice in FSS_map_char / FSS_get_advance during character creation; the CFF original
        // crashed at boot. Without it: the phone's Roboto (Latin only)
        char st[600]; snprintf(st, sizeof st, "%s/NotoSansCJKjp.ttf", s);
        int own = getenv("DM_SHARED_FONT") != NULL, cjk = !access(st, R_OK);   // own: env.txt chose the font
        if (cjk) setenv("DM_SHARED_FONT", st, 0);
        snprintf(st, sizeof st, "%s/standard.ttf", s);
        const char *sys = "/system/fonts/RobotoStatic-Regular.ttf";
        int rob = !own && !cjk && access(st, R_OK) && !access(sys, R_OK);
        if (rob) setenv("DM_SHARED_FONT", sys, 0);
        if (!own && !cjk) LOG("front: [RED] font/NotoSansCJKjp.ttf is missing, %s: character creation can crash (copy the game folder made by android-host/build_android.py again)", rob ? "using Roboto" : "no stand-in font set");
    }
    snprintf(s, sizeof s, "%s/vk", g_data); setenv("DM_VK_SHADERS", s, 0);
    snprintf(s, sizeof s, "%s/vk_pcache.bin", g_internal); setenv("DM_VK_PCACHE", s, 0);   // Vulkan pipeline cache kept across runs (nvn_vk.c); DM_VK_PCACHE= (empty) in env.txt turns it off
    setenv("DM_LANG", "en-GB", 0);
    setenv("DM_AUDIO", "1", 0);   // real output (compat/src/audio_aaudio.c); DM_AUDIO=0 in env.txt for the silent sink
    setenv("DM_OPMODE", "0", 0);  // handheld: 720p, as a phone is held (the profile's default is docked); DM_OPMODE=1 in env.txt for docked
    // Same-core preemption (vendor hle_thread.c, DM_PREEMPT=1 in env.txt) stays off: with it four runs crashed in four places (boot, font engine, a submitted handle 0x1, a
    // command buffer overwritten in play); without it character creation ran crash-free twice. The movie-end crash it seemed to fix is caught by hle_nvn.c cmd_bad
    chdir(g_internal);
}

// ---------- What fills memory: DM_MEMLOG=<ms>, DM_MEM_STOP=<MB> ----------
// Android's low memory killer ended the first run 6 s in, at 3.9 GB in RAM, after closing other apps. DM_MEMLOG logs this process's memory and page faults at that interval, and where its running threads are.
// DM_MEM_STOP ends the run once RAM + swap reach that many MB, before the killer starts on the phone's other apps
typedef struct { uint64_t pc, lr; int tid; } mem_sample;
static mem_sample g_ms[64];
static _Atomic int g_ms_n;
static void on_mem_sample(int sig, siginfo_t *si, void *uc_) {
    (void)sig; (void)si;
    ucontext_t *uc = uc_; int i = atomic_fetch_add(&g_ms_n, 1);
    if (i < 64) { g_ms[i].pc = UC_PC(uc); g_ms[i].lr = UC_LR(uc); g_ms[i].tid = (int)syscall(SYS_gettid); }
}
static long proc_field(const char *text, const char *name) {   // a "Name:  123 kB" line of /proc/self/status or /proc/meminfo
    const char *p = strstr(text, name); return p ? atol(p + strlen(name)) : 0;
}
static size_t read_text(const char *path, char *out, size_t cap) {
    int fd = open(path, O_RDONLY | O_CLOEXEC); if (fd < 0) { out[0] = 0; return 0; }
    size_t have = 0; for (ssize_t n; have < cap - 1 && (n = read(fd, out + have, cap - 1 - have)) > 0;) have += (size_t)n;
    close(fd); out[have] = 0; return have;
}
static int stat_of(const char *path, char *state, long *minflt, long *majflt) {   // after the ")" of the name: state, then ppid, pgrp, session, tty, tpgid, flags, minflt, cminflt, majflt
    char b[1024]; if (!read_text(path, b, sizeof b)) return 0;
    const char *p = strrchr(b, ')'); if (!p) return 0;
    long skip, cskip; return sscanf(p + 1, " %c %ld %ld %ld %ld %ld %ld %ld %ld %ld", state, &skip, &skip, &skip, &skip, &skip, &skip, minflt, &cskip, majflt) == 10;
}
static size_t where(char *out, size_t cap, uint64_t a) {
    module_t *m = module_of((void *)a); Dl_info di;
    if (m) return (size_t)snprintf(out, cap, "%s+%#llx", m->name, (unsigned long long)(a - (uint64_t)m->base));
    if (dladdr((void *)a, &di) && di.dli_sname) return (size_t)snprintf(out, cap, "%s+%#lx", di.dli_sname, (unsigned long)(a - (uint64_t)di.dli_saddr));
    if (dladdr((void *)a, &di) && di.dli_fname) { const char *n = strrchr(di.dli_fname, '/'); return (size_t)snprintf(out, cap, "%s+%#lx", n ? n + 1 : di.dli_fname, (unsigned long)(a - (uint64_t)di.dli_fbase)); }
    return (size_t)snprintf(out, cap, "%#llx", (unsigned long long)a);
}
// ---------- What holds it: DM_MEMMAP=<s> ----------
// Samsung's memory guard ("Heimdall" in the system log) ended a run at 157 s, in the foreground: its usage is RssAnon + VmSwap (1,848,964 + 2,605,692 = 4,454,656 KB, as it logged) against a limit of 4,194,304 KB.
// An app may not read /proc/self/pagemap (errno 13 on the phone), but /proc/self/smaps gives RAM and swap per mapping, and each large memory pool is a mapping of its own by name ("dmpool<MB>", mtl_pool_add). Sizes only, no contents
int dm_guest_as_contains(uintptr_t a);
static void mem_map_report(double at) {
    uint64_t hb = 0; size_t hn = 0; dm_guest_heap_range(&hb, &hn);
    enum { NAMES = 4096 }; static struct { char name[72]; uint64_t anon, swp; } t[NAMES]; int nt = 0;
    FILE *f = fopen("/proc/self/smaps", "re"); if (!f) { static int said; if (!said++) LOG("[RED] memmap: /proc/self/smaps: errno %d", errno); return; }
    static char l[512]; int cur = -1; uint64_t vmas = 0;
    while (fgets(l, sizeof l, f)) {
        unsigned long long a, b, kb; int off = 0;
        if (sscanf(l, "%llx-%llx %*s %*s %*s %*s %n", &a, &b, &off) == 2 && off > 0) {
            char name[72]; snprintf(name, sizeof name, "%s", l + off); name[strcspn(name, "\n")] = 0; vmas++;
            if (!name[0]) strcpy(name, a >= hb && a < hb + hn ? "(guest heap outside large pools)" : dm_guest_as_contains((uintptr_t)a) ? "(guest block, not heap)" : "(unnamed)");
            cur = -1; for (int i = 0; i < nt; i++) if (!strcmp(t[i].name, name)) cur = i;
            if (cur < 0 && nt < NAMES) { cur = nt++; strcpy(t[cur].name, name); t[cur].anon = t[cur].swp = 0; }
            continue;
        }
        if (cur < 0) continue;
        if (sscanf(l, "Anonymous: %llu kB", &kb) == 1 || (!strncmp(t[cur].name, "/memfd:", 7) && sscanf(l, "Rss: %llu kB", &kb) == 1)) t[cur].anon += kb;
        else if (sscanf(l, "Swap: %llu kB", &kb) == 1) t[cur].swp += kb;
    }
    fclose(f);
    LOG("memmap: %.0f s: heap opened %llu MB, %llu mappings", at, (unsigned long long)(hn >> 20), (unsigned long long)vmas);
    for (int i = 0; i < nt; i++) if ((t[i].anon + t[i].swp) >> 10 >= 4) LOG("memmap: %s: %llu MB in RAM, %llu MB in swap", t[i].name, (unsigned long long)(t[i].anon >> 10), (unsigned long long)(t[i].swp >> 10));
}
static void *mem_thread(void *arg) {
    (void)arg;
    const char *em = getenv("DM_MEMLOG"), *es = getenv("DM_MEM_STOP"), *emm = getenv("DM_MEMMAP"); int ms = em && atoi(em) > 0 ? atoi(em) : 1000; long stop = es ? atol(es) : 0; double mm_every = emm ? atof(emm) : 0, mm_next = mm_every;
    struct sigaction sa = {.sa_sigaction = on_mem_sample, .sa_flags = SA_SIGINFO | SA_ONSTACK | SA_RESTART}; sigemptyset(&sa.sa_mask); sigaction(SIGPROF, &sa, NULL);
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    long minflt0 = 0, majflt0 = 0, shm_swap = 0; int self = (int)syscall(SYS_gettid);
    static char st[8192], mi[8192], vm[32768]; char line[1800], path[96], c;
    for (;;) {
        read_text("/proc/self/status", st, sizeof st); read_text("/proc/meminfo", mi, sizeof mi);
        long pinned = read_text("/proc/vmstat", vm, sizeof vm) ? (proc_field(vm, "nr_foll_pin_acquired ") - proc_field(vm, "nr_foll_pin_released ")) * 4 / 1024 : -1;   // -1: Android may not let an app read it
        long rss = proc_field(st, "VmRSS:") / 1024, swap = proc_field(st, "VmSwap:") / 1024, minflt = 0, majflt = 0;
        // VmSwap leaves out shared memory in swap, which is where the large pools live now (pool_make_shared). smaps_rollup counts it, but walks every page table, so only on every 10th sample
        { static int nroll; if (!(nroll++ % 10) && read_text("/proc/self/smaps_rollup", vm, sizeof vm)) { long v = proc_field(vm, "Swap:") / 1024 - swap; shm_swap = v > 0 ? v : 0; } }
        stat_of("/proc/self/stat", &c, &minflt, &majflt);
        struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
        size_t q = (size_t)snprintf(line, sizeof line, "mem: %.1f s RAM %ld MB (anon %ld file %ld shared %ld) swap %ld, shared swap %ld, Samsung's count %ld; faults +%ld minor +%ld major; phone available %ld MB, swap free %ld MB, pinned %ld MB; input samples %llu; threads", (double)(t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) / 1e9,
            rss, proc_field(st, "RssAnon:") / 1024, proc_field(st, "RssFile:") / 1024, proc_field(st, "RssShmem:") / 1024, swap, shm_swap, proc_field(st, "RssAnon:") / 1024 + swap, minflt - minflt0, majflt - majflt0, proc_field(mi, "MemAvailable:") / 1024, proc_field(mi, "SwapFree:") / 1024, pinned, (unsigned long long)dm_input_samples());
        minflt0 = minflt; majflt0 = majflt;
        atomic_store(&g_ms_n, 0);
        DIR *d = opendir("/proc/self/task"); int disk = 0;
        for (struct dirent *e; d && (e = readdir(d));) {
            int tid = atoi(e->d_name); if (tid <= 0 || tid == self) continue;
            long a, b; snprintf(path, sizeof path, "/proc/self/task/%d/stat", tid);
            if (stat_of(path, &c, &a, &b)) { if (c == 'R') syscall(SYS_tgkill, getpid(), tid, SIGPROF); else if (c == 'D') disk++; }
        }
        if (d) closedir(d);
        usleep(2000);
        int n = atomic_load(&g_ms_n); if (n > 64) n = 64;
        q += (size_t)snprintf(line + q, sizeof line - q, " in disk wait %d, running %d:", disk, n);
        for (int i = 0; i < n && q < sizeof line - 240; i++) {
            char name[32]; snprintf(path, sizeof path, "/proc/self/task/%d/comm", g_ms[i].tid); read_text(path, name, sizeof name); name[strcspn(name, "\n")] = 0;
            q += (size_t)snprintf(line + q, sizeof line - q, " [%s ", name); q += where(line + q, sizeof line - q, g_ms[i].pc);
            q += (size_t)snprintf(line + q, sizeof line - q, " from "); q += where(line + q, sizeof line - q, g_ms[i].lr);
            q += (size_t)snprintf(line + q, sizeof line - q, "]");
        }
        LOG("%s", line);
        { void mtl_pool_report(void); mtl_pool_report(); }
        // The allocator keeps freed memory for reuse. Every 30 s it gives back what it holds, and the log says what that costs
        { static long last; if (t.tv_sec - last >= 30) { last = t.tv_sec; struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
            int r = mallopt(-104 /* M_PURGE_ALL */, 0); if (r != 1) r = mallopt(-101 /* M_PURGE */, 0);
            clock_gettime(CLOCK_MONOTONIC, &b); static int nl; if (nl++ < 6) LOG("mem: allocator gave memory back (%d) in %.1f ms", r, (double)(b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) / 1e6); } }
        if (mm_every > 0) { double at = (double)(t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) / 1e9; if (at >= mm_next) { mem_map_report(at); mm_next = at + mm_every; } }
        if (stop && rss + swap + shm_swap >= stop) { LOG("mem: stopping the run: RAM + swap reached %ld MB (DM_MEM_STOP %ld)", rss + swap + shm_swap, stop); dm_log_drain(); _exit(0); }
        usleep((useconds_t)ms * 1000);
    }
    return NULL;
}

// ---------- Host thread ----------
static void *host_thread(void *a) {
    (void)a;
    { const char *e = getenv("DM_VK_PROBE"); if (e) { dm_vk_probe(atoi(e)); dm_log_drain(); exit(0); } }   // nvn_vk.c: the shader pack against the driver; no game
    char mods[600]; snprintf(mods, sizeof mods, "%s/mods", g_data);
    char *argv[] = {"dmhost", mods, NULL};
    int r = main(2, argv);
    LOG("front: host returned %d", r);
    exit(r);
    return NULL;
}
void front_run(void (*guest)(void)) { guest(); }   // on the host thread: android_main keeps the activity's events
void *front_metal_layer(void) { return NULL; }
int front_screen_max_fps(void) { return 0; }

// ---------- Java: immersive window (UI thread) and haptics ----------
static int jfail(JNIEnv *e, const char *what) {
    if (!(*e)->ExceptionCheck(e)) return 0;
    (*e)->ExceptionClear(e); LOG("[RED] front: %s threw a Java exception", what); return 1;
}
// Window.setDecorFitsSystemWindows(false), cutout mode SHORT_EDGES, the system bars hidden with BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE (API 30+)
static void immersive(ANativeActivity *a) {
    JNIEnv *e = a->env;
    (*e)->PushLocalFrame(e, 16);
    jclass ac = (*e)->GetObjectClass(e, a->clazz), wc = (*e)->FindClass(e, "android/view/Window");
    jobject win = (*e)->CallObjectMethod(e, a->clazz, (*e)->GetMethodID(e, ac, "getWindow", "()Landroid/view/Window;"));
    if (jfail(e, "getWindow") || !win) goto out;
    jclass lpc = (*e)->FindClass(e, "android/view/WindowManager$LayoutParams");
    jobject lp = (*e)->CallObjectMethod(e, win, (*e)->GetMethodID(e, wc, "getAttributes", "()Landroid/view/WindowManager$LayoutParams;"));
    if (!jfail(e, "getAttributes") && lp) {
        (*e)->SetIntField(e, lp, (*e)->GetFieldID(e, lpc, "layoutInDisplayCutoutMode", "I"), 1);   // LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        (*e)->CallVoidMethod(e, win, (*e)->GetMethodID(e, wc, "setAttributes", "(Landroid/view/WindowManager$LayoutParams;)V"), lp);
        jfail(e, "setAttributes");
    }
    (*e)->CallVoidMethod(e, win, (*e)->GetMethodID(e, wc, "setDecorFitsSystemWindows", "(Z)V"), JNI_FALSE);
    jfail(e, "setDecorFitsSystemWindows");
    jobject ic = (*e)->CallObjectMethod(e, win, (*e)->GetMethodID(e, wc, "getInsetsController", "()Landroid/view/WindowInsetsController;"));
    if (jfail(e, "getInsetsController") || !ic) goto out;
    jclass tc = (*e)->FindClass(e, "android/view/WindowInsets$Type"), icc = (*e)->FindClass(e, "android/view/WindowInsetsController");
    jint bars = (*e)->CallStaticIntMethod(e, tc, (*e)->GetStaticMethodID(e, tc, "systemBars", "()I"));
    (*e)->CallVoidMethod(e, ic, (*e)->GetMethodID(e, icc, "setSystemBarsBehavior", "(I)V"), 2);
    (*e)->CallVoidMethod(e, ic, (*e)->GetMethodID(e, icc, "hide", "(I)V"), bars);
    if (!jfail(e, "WindowInsetsController.hide")) LOG("front: immersive (system bars hidden, cutout short edges)");
out:
    (*e)->PopLocalFrame(e, NULL);
}
static void (*g_glue_focus)(ANativeActivity *, int);
static void on_focus(ANativeActivity *a, int has) {   // the bars come back after a swipe or a dialog: hide them again
    if (g_glue_focus) g_glue_focus(a, has);
    if (has) immersive(a);
}

// ---------- A notice for the player: a Toast, shown on the UI thread ----------
// The APK has no Java code, so nothing can implement Runnable for runOnUiThread; instead the UI thread's own ALooper watches a pipe: any thread writes an int, the looper calls
// note_cb on the UI thread (the only one a->env belongs to), which shows the text with Toast.makeText(activity, text, LENGTH_LONG). A write to the non-blocking pipe cannot stall
// the game thread (a full pipe drops the value: the player has had the notice by then)
static struct { ANativeActivity *act; int rd, wr; } g_note = {.rd = -1, .wr = -1};
static int note_cb(int fd, int events, void *data) {
    (void)events; (void)data;
    int v, err = 0, n = 0; while (read(fd, &v, sizeof v) == (ssize_t)sizeof v) { err = v; n++; }   // until EAGAIN; the last one wins
    if (!n) return 1;
    char m[200];
    if (err == FRONT_SAVE_REPLACED) snprintf(m, sizeof m, "Your save was replaced while the game was running: close the game fully and open it again. The game will not save until then.");
    else if (err == ENOSPC) snprintf(m, sizeof m, "Storage is full, so progress was not saved. Free up space and save again.");
    else snprintf(m, sizeof m, "Saving failed (%s); progress was not saved.", strerror(err));
    JNIEnv *e = g_note.act->env;
    (*e)->PushLocalFrame(e, 8);
    jclass tc = (*e)->FindClass(e, "android/widget/Toast"); if (jfail(e, "FindClass Toast") || !tc) goto out;
    jmethodID mk = (*e)->GetStaticMethodID(e, tc, "makeText", "(Landroid/content/Context;Ljava/lang/CharSequence;I)Landroid/widget/Toast;"); if (jfail(e, "Toast.makeText id") || !mk) goto out;
    jmethodID sh = (*e)->GetMethodID(e, tc, "show", "()V"); if (jfail(e, "Toast.show id") || !sh) goto out;
    jstring js = (*e)->NewStringUTF(e, m); if (!js) goto out;
    jobject t = (*e)->CallStaticObjectMethod(e, tc, mk, g_note.act->clazz, js, 1);   // 1: LENGTH_LONG
    if (jfail(e, "Toast.makeText") || !t) goto out;
    (*e)->CallVoidMethod(e, t, sh); jfail(e, "Toast.show");
out:
    (*e)->PopLocalFrame(e, NULL);
    return 1;
}
// ---------- The game's text box (nn::swkbd, hle_swkbd.c): an AlertDialog built over JNI on the UI thread ----------
// With no Java code nothing here can implement a click listener. But AlertDialog also takes a Message per button, posted when that button is pressed, and a Message can carry a Runnable:
// a FutureTask around a do-nothing Thread is a Runnable the framework already has, and its isDone() afterwards says which button it was. The dialog closes itself after a press.
// The game thread that asked waits (as it does on the console, where the keyboard applet blocks only that thread) and has the UI thread look 10 times a second
enum { TB_IDLE, TB_OPENING, TB_SHOWN, TB_DONE, TB_FAILED };
static struct {
    int rd, wr; _Atomic int state;
    const char *title, *sub, *guide, *initial; int max_chars, numeric, ok; uint16_t text[1003]; int ntext;
    jobject dlg, edit, task_ok;
} g_tb = {.rd = -1, .wr = -1};
extern int g_swkbd_min_chars, g_swkbd_numeric_max;
size_t swkbd_u8_to_u16(const char *s, size_t max_units, uint16_t *out, size_t cap);
size_t swkbd_u16_to_u8(const uint16_t *s, size_t n, char *out, size_t cap);
static jstring tb_utf16(JNIEnv *e, const char *s) {   // from UTF-8 by way of UTF-16: NewStringUTF wants modified UTF-8, and text with an emoji in it is not that
    static uint16_t u[1003]; size_t n = swkbd_u8_to_u16(s ? s : "", 1002, u, 1003); return (*e)->NewString(e, u, (jsize)n);
}
static jobject tb_button(JNIEnv *e, jobject handler, jobject dlg, jclass dc, int which, const char *label) {   // returns the FutureTask that says whether this button was pressed
    jclass tc = (*e)->FindClass(e, "java/lang/Thread"), fc = (*e)->FindClass(e, "java/util/concurrent/FutureTask"), mc = (*e)->FindClass(e, "android/os/Message");
    jobject idle = (*e)->NewObject(e, tc, (*e)->GetMethodID(e, tc, "<init>", "()V"));
    jobject task = (*e)->NewObject(e, fc, (*e)->GetMethodID(e, fc, "<init>", "(Ljava/lang/Runnable;Ljava/lang/Object;)V"), idle, idle);
    jobject msg = (*e)->CallStaticObjectMethod(e, mc, (*e)->GetStaticMethodID(e, mc, "obtain", "(Landroid/os/Handler;Ljava/lang/Runnable;)Landroid/os/Message;"), handler, task);
    (*e)->CallVoidMethod(e, dlg, (*e)->GetMethodID(e, dc, "setButton", "(ILjava/lang/CharSequence;Landroid/os/Message;)V"), which, (*e)->NewStringUTF(e, label), msg);
    return task;
}
static void tb_open(void) {
    JNIEnv *e = g_note.act->env; jobject act = g_note.act->clazz;
    (*e)->PushLocalFrame(e, 64);
    jclass etc = (*e)->FindClass(e, "android/widget/EditText"), bc = (*e)->FindClass(e, "android/app/AlertDialog$Builder"), dc = (*e)->FindClass(e, "android/app/AlertDialog"), wc = (*e)->FindClass(e, "android/view/Window");
    jclass hc = (*e)->FindClass(e, "android/os/Handler"), lc = (*e)->FindClass(e, "android/os/Looper"), ifc = (*e)->FindClass(e, "android/text/InputFilter"), lfc = (*e)->FindClass(e, "android/text/InputFilter$LengthFilter");
    if (jfail(e, "text box classes")) goto bad;
    jobject et = (*e)->NewObject(e, etc, (*e)->GetMethodID(e, etc, "<init>", "(Landroid/content/Context;)V"), act);
    (*e)->CallVoidMethod(e, et, (*e)->GetMethodID(e, etc, "setSingleLine", "()V"));
    if (g_tb.numeric) (*e)->CallVoidMethod(e, et, (*e)->GetMethodID(e, etc, "setInputType", "(I)V"), 2);   // TYPE_CLASS_NUMBER
    if (g_tb.max_chars > 0) {
        jobject lf = (*e)->NewObject(e, lfc, (*e)->GetMethodID(e, lfc, "<init>", "(I)V"), g_tb.max_chars);
        jobjectArray fs = (*e)->NewObjectArray(e, 1, ifc, lf);
        (*e)->CallVoidMethod(e, et, (*e)->GetMethodID(e, etc, "setFilters", "([Landroid/text/InputFilter;)V"), fs);
    }
    if (g_tb.guide && *g_tb.guide) (*e)->CallVoidMethod(e, et, (*e)->GetMethodID(e, etc, "setHint", "(Ljava/lang/CharSequence;)V"), tb_utf16(e, g_tb.guide));
    if (g_tb.initial && *g_tb.initial) {
        (*e)->CallVoidMethod(e, et, (*e)->GetMethodID(e, etc, "setText", "(Ljava/lang/CharSequence;)V"), tb_utf16(e, g_tb.initial));
        (*e)->CallVoidMethod(e, et, (*e)->GetMethodID(e, etc, "selectAll", "()V"));
    }
    if (jfail(e, "text box field")) goto bad;
    jobject b = (*e)->NewObject(e, bc, (*e)->GetMethodID(e, bc, "<init>", "(Landroid/content/Context;)V"), act);
    if (g_tb.title && *g_tb.title) (*e)->CallObjectMethod(e, b, (*e)->GetMethodID(e, bc, "setTitle", "(Ljava/lang/CharSequence;)Landroid/app/AlertDialog$Builder;"), tb_utf16(e, g_tb.title));
    if (g_tb.sub && *g_tb.sub) (*e)->CallObjectMethod(e, b, (*e)->GetMethodID(e, bc, "setMessage", "(Ljava/lang/CharSequence;)Landroid/app/AlertDialog$Builder;"), tb_utf16(e, g_tb.sub));
    (*e)->CallObjectMethod(e, b, (*e)->GetMethodID(e, bc, "setView", "(Landroid/view/View;)Landroid/app/AlertDialog$Builder;"), et);
    (*e)->CallObjectMethod(e, b, (*e)->GetMethodID(e, bc, "setCancelable", "(Z)Landroid/app/AlertDialog$Builder;"), JNI_FALSE);
    jobject d = (*e)->CallObjectMethod(e, b, (*e)->GetMethodID(e, bc, "create", "()Landroid/app/AlertDialog;"));
    if (jfail(e, "text box dialog") || !d) goto bad;
    jobject looper = (*e)->CallStaticObjectMethod(e, lc, (*e)->GetStaticMethodID(e, lc, "getMainLooper", "()Landroid/os/Looper;"));
    jobject h = (*e)->NewObject(e, hc, (*e)->GetMethodID(e, hc, "<init>", "(Landroid/os/Looper;)V"), looper);
    jobject tok = tb_button(e, h, d, dc, -1, "OK");   // BUTTON_POSITIVE
    (void)tb_button(e, h, d, dc, -2, "Cancel");       // BUTTON_NEGATIVE
    if (jfail(e, "text box buttons") || !tok) goto bad;
    jobject w = (*e)->CallObjectMethod(e, d, (*e)->GetMethodID(e, dc, "getWindow", "()Landroid/view/Window;"));
    if (w) (*e)->CallVoidMethod(e, w, (*e)->GetMethodID(e, wc, "setSoftInputMode", "(I)V"), 5);   // SOFT_INPUT_STATE_ALWAYS_VISIBLE
    (*e)->CallVoidMethod(e, d, (*e)->GetMethodID(e, dc, "show", "()V"));
    if (jfail(e, "text box show")) goto bad;
    (*e)->CallBooleanMethod(e, et, (*e)->GetMethodID(e, etc, "requestFocus", "()Z")); jfail(e, "text box focus");
    g_tb.dlg = (*e)->NewGlobalRef(e, d); g_tb.edit = (*e)->NewGlobalRef(e, et); g_tb.task_ok = (*e)->NewGlobalRef(e, tok);
    atomic_store(&g_tb.state, TB_SHOWN);
    (*e)->PopLocalFrame(e, NULL);
    return;
bad:
    jfail(e, "text box");
    atomic_store(&g_tb.state, TB_FAILED);
    (*e)->PopLocalFrame(e, NULL);
}
static void tb_poll(void) {
    if (atomic_load(&g_tb.state) != TB_SHOWN) return;
    JNIEnv *e = g_note.act->env;
    (*e)->PushLocalFrame(e, 16);
    jclass dc = (*e)->FindClass(e, "android/app/AlertDialog"), fc = (*e)->FindClass(e, "java/util/concurrent/FutureTask"), tc = (*e)->FindClass(e, "android/widget/EditText"), oc = (*e)->FindClass(e, "java/lang/Object");
    if (!(*e)->CallBooleanMethod(e, g_tb.dlg, (*e)->GetMethodID(e, dc, "isShowing", "()Z"))) {
        g_tb.ok = (*e)->CallBooleanMethod(e, g_tb.task_ok, (*e)->GetMethodID(e, fc, "isDone", "()Z"));
        jobject ed = (*e)->CallObjectMethod(e, g_tb.edit, (*e)->GetMethodID(e, tc, "getText", "()Landroid/text/Editable;"));
        jstring js = ed ? (*e)->CallObjectMethod(e, ed, (*e)->GetMethodID(e, oc, "toString", "()Ljava/lang/String;")) : NULL;
        g_tb.ntext = 0;
        if (js) { jsize n = (*e)->GetStringLength(e, js); if (n > 1002) n = 1002; (*e)->GetStringRegion(e, js, 0, n, g_tb.text); g_tb.ntext = n; }
        if (jfail(e, "text box result")) g_tb.ok = 0;
        (*e)->DeleteGlobalRef(e, g_tb.dlg); (*e)->DeleteGlobalRef(e, g_tb.edit); (*e)->DeleteGlobalRef(e, g_tb.task_ok);
        g_tb.dlg = g_tb.edit = g_tb.task_ok = NULL;
        atomic_store(&g_tb.state, TB_DONE);
    }
    jfail(e, "text box look");
    (*e)->PopLocalFrame(e, NULL);
}
static int tb_cb(int fd, int events, void *data) {
    (void)events; (void)data;
    int v; while (read(fd, &v, sizeof v) == (ssize_t)sizeof v) { if (v == 1) tb_open(); else tb_poll(); }
    return 1;
}
// Returns 1 with the text, 0 when the player pressed Cancel, -1 when no box could be shown (hle_swkbd.c then answers as a cancel). The text is not logged: it can be private
int front_text_input(const char *title, const char *sub, const char *guide, const char *initial, int max_chars, char *out, int out_cap) {
    if (g_tb.wr < 0) return -1;
    static pthread_mutex_t one = PTHREAD_MUTEX_INITIALIZER; pthread_mutex_lock(&one);
    const int minc = g_swkbd_min_chars, numeric = g_swkbd_numeric_max >= 0;
    static char again[1003 * 3 + 1]; int r;
    LOG("front: text box up, limit %d characters, at least %d", max_chars, minc);
    for (const char *start = initial;;) {
        g_tb.title = title; g_tb.sub = sub; g_tb.guide = guide; g_tb.initial = start; g_tb.max_chars = max_chars; g_tb.numeric = numeric;
        atomic_store(&g_tb.state, TB_OPENING);
        int v = 1; if (write(g_tb.wr, &v, sizeof v) != (ssize_t)sizeof v) { LOG("[RED] front: text box pipe full or closed"); r = -1; break; }
        while (atomic_load(&g_tb.state) == TB_OPENING || atomic_load(&g_tb.state) == TB_SHOWN) {
            usleep(100000);
            v = 2; if (atomic_load(&g_tb.state) == TB_SHOWN && write(g_tb.wr, &v, sizeof v) != (ssize_t)sizeof v) { /* the next look does it */ }
        }
        if (atomic_load(&g_tb.state) == TB_FAILED) { LOG("[RED] front: the text box could not be shown"); r = -1; break; }
        if (!g_tb.ok) { r = 0; break; }
        g_tb.text[g_tb.ntext] = 0;
        swkbd_u16_to_u8(g_tb.text, (size_t)g_tb.ntext, again, sizeof again);
        if (minc > 0 && g_tb.ntext < minc) { start = again; continue; }   // the applet keeps OK disabled until the text is long enough; here the box comes back
        snprintf(out, (size_t)out_cap, "%s", again); r = 1; break;
    }
    atomic_store(&g_tb.state, TB_IDLE);
    LOG("front: text box closed: %s", r > 0 ? "OK" : r == 0 ? "cancel" : "no box");
    pthread_mutex_unlock(&one);
    return r;
}
// DM_TEST_TEXTBOX=<seconds>: asks for text that long after the start, the way the game would, without needing the game to reach its naming screen
static void *tb_test_thread(void *arg) {
    sleep((unsigned)(uintptr_t)arg);
    char out[256] = ""; int r = front_text_input("Test title", "Test subtitle", "Test hint", "Abc", 12, out, sizeof out);
    LOG("front: text box test: result %d, %zu bytes back", r, strlen(out));
    return NULL;
}
static void note_init(ANativeActivity *a) {   // on the UI thread (ANativeActivity_onCreate)
    ALooper *lp = ALooper_forThread(); int fds[2];
    if (!lp || pipe2(fds, O_CLOEXEC | O_NONBLOCK)) { LOG("[RED] front: no way to show notices (looper %p): save failures reach log.txt only", (void *)lp); return; }
    if (ALooper_addFd(lp, fds[0], ALOOPER_POLL_CALLBACK, ALOOPER_EVENT_INPUT, note_cb, NULL) != 1) { close(fds[0]); close(fds[1]); LOG("[RED] front: cannot watch the notice pipe"); return; }
    g_note.act = a; g_note.rd = fds[0]; g_note.wr = fds[1];
    if (pipe2(fds, O_CLOEXEC | O_NONBLOCK)) { LOG("[RED] front: no text box pipe: the game's text boxes will count as cancelled"); return; }
    if (ALooper_addFd(lp, fds[0], ALOOPER_POLL_CALLBACK, ALOOPER_EVENT_INPUT, tb_cb, NULL) != 1) { close(fds[0]); close(fds[1]); LOG("[RED] front: cannot watch the text box pipe"); return; }
    g_tb.rd = fds[0]; g_tb.wr = fds[1];
}
uint64_t dm_fs_commit_count(void);   // hle_fs.c: successful save commits so far
// hle_fs.c calls this on the game thread when a save transaction is refused, a commit's copy or swap fails, or the save mounted read-only: log it and show the player a Toast.
// A failure streak (every refused commit after the first, until one commit succeeds) gets one notice: the streak ends when dm_fs_commit_count() has moved
void front_save_failed(int err) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static uint64_t shown_at = ~0ull; static int replaced_shown;
    uint64_t c = dm_fs_commit_count(); int fresh;
    pthread_mutex_lock(&mu);   // FRONT_SAVE_REPLACED (a restored save found while running: every later commit is refused) is its own notice, once, and does not use up the write-failure streak
    if (err == FRONT_SAVE_REPLACED) { fresh = !replaced_shown; replaced_shown = 1; } else { fresh = c != shown_at; shown_at = c; }
    pthread_mutex_unlock(&mu);
    LOG("[RED] front: save failed, %s: progress since the last good save is not on storage%s", err == FRONT_SAVE_REPLACED ? "the save folder was replaced from outside" : strerror(err), fresh ? "; telling the player" : " (same failure streak or already told: no new notice)");
    if (fresh && g_note.wr >= 0) { int v = err; if (write(g_note.wr, &v, sizeof v) != (ssize_t)sizeof v) LOG("front: notice pipe full or closed"); }
}
JNIEXPORT void ANativeActivity_onCreate(ANativeActivity *a, void *saved, size_t saved_size) {
    glue_ANativeActivity_onCreate(a, saved, saved_size);
    g_glue_focus = a->callbacks->onWindowFocusChanged; a->callbacks->onWindowFocusChanged = on_focus;
    immersive(a);
    note_init(a);
}
// Vibrator.vibrate(VibrationEffect.createPredefined(EFFECT_TICK)) from android_main's thread (attached to the VM); needs android.permission.VIBRATE
static struct { JNIEnv *env; jobject vib, fx; jmethodID vibrate; int on; } g_hap;
static void haptics_init(ANativeActivity *a) {
    const char *h = getenv("DM_HAPTICS"); if (h && !atoi(h)) return;
    JNIEnv *e; if ((*a->vm)->AttachCurrentThread(a->vm, &e, NULL)) return;
    g_hap.env = e;
    (*e)->PushLocalFrame(e, 16);
    jclass ac = (*e)->GetObjectClass(e, a->clazz), vc = (*e)->FindClass(e, "android/os/Vibrator"), fc = (*e)->FindClass(e, "android/os/VibrationEffect");
    jobject vib = (*e)->CallObjectMethod(e, a->clazz, (*e)->GetMethodID(e, ac, "getSystemService", "(Ljava/lang/String;)Ljava/lang/Object;"), (*e)->NewStringUTF(e, "vibrator"));
    jobject fx = jfail(e, "getSystemService") ? NULL : (*e)->CallStaticObjectMethod(e, fc, (*e)->GetStaticMethodID(e, fc, "createPredefined", "(I)Landroid/os/VibrationEffect;"), 2);
    if (!jfail(e, "createPredefined") && vib && fx) {
        g_hap.vib = (*e)->NewGlobalRef(e, vib); g_hap.fx = (*e)->NewGlobalRef(e, fx);
        g_hap.vibrate = (*e)->GetMethodID(e, vc, "vibrate", "(Landroid/os/VibrationEffect;)V"); g_hap.on = 1;
    }
    (*e)->PopLocalFrame(e, NULL);
    LOG("front: haptics %s", g_hap.on ? "on" : "unavailable");
}
static void haptic_tick(void) {
    if (!g_hap.on) return;
    (*g_hap.env)->CallVoidMethod(g_hap.env, g_hap.vib, g_hap.vibrate, g_hap.fx);
    if (jfail(g_hap.env, "vibrate")) g_hap.on = 0;
}

// ---------- Input ----------
// Touches go to touch_controls.c (the on-screen pad, shown until a game controller is used); a controller's state is merged with the pad's
static int g_ww, g_wh, g_dpi, g_overlay = 1;
static float g_opacity = 1;
static tc_result g_tc;
static pm_state g_pad;
_Static_assert(PM_KEY_BACK == AKEYCODE_BACK && PM_KEY_MENU == AKEYCODE_MENU && PM_KEY_DPAD_UP == AKEYCODE_DPAD_UP && PM_KEY_DPAD_DOWN == AKEYCODE_DPAD_DOWN && PM_KEY_DPAD_LEFT == AKEYCODE_DPAD_LEFT && PM_KEY_DPAD_RIGHT == AKEYCODE_DPAD_RIGHT
    && PM_KEY_BUTTON_A == AKEYCODE_BUTTON_A && PM_KEY_BUTTON_B == AKEYCODE_BUTTON_B && PM_KEY_BUTTON_X == AKEYCODE_BUTTON_X && PM_KEY_BUTTON_Y == AKEYCODE_BUTTON_Y && PM_KEY_BUTTON_L1 == AKEYCODE_BUTTON_L1 && PM_KEY_BUTTON_R1 == AKEYCODE_BUTTON_R1
    && PM_KEY_BUTTON_L2 == AKEYCODE_BUTTON_L2 && PM_KEY_BUTTON_R2 == AKEYCODE_BUTTON_R2 && PM_KEY_BUTTON_THUMBL == AKEYCODE_BUTTON_THUMBL && PM_KEY_BUTTON_THUMBR == AKEYCODE_BUTTON_THUMBR && PM_KEY_BUTTON_START == AKEYCODE_BUTTON_START && PM_KEY_BUTTON_SELECT == AKEYCODE_BUTTON_SELECT
    && PM_SOURCE_KEYBOARD == AINPUT_SOURCE_KEYBOARD && PM_SOURCE_DPAD == AINPUT_SOURCE_DPAD && PM_SOURCE_GAMEPAD == AINPUT_SOURCE_GAMEPAD && PM_SOURCE_JOYSTICK == AINPUT_SOURCE_JOYSTICK, "pad_map.h no longer matches the NDK");

static void overlay_push(void) {   // after the first present (the device exists, the frame rect is known)
    int fr[4]; dm_vk_frame_rect(fr);
    if (fr[2] <= 0 || !g_overlay) return;
    // at most ~30 layers a second (the game presents at 30): a touch screen sends 120-240 moves a second, each one used to draw and upload the layer again; the main loop below picks up the last one
    static uint64_t last; struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); uint64_t now = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    if (now - last < 33000000ull) return;
    tc_blend_linear(dm_vk_present_linear());
    const uint8_t *p; int w, h;
    if (tc_render(&p, &w, &h)) { dm_vk_overlay_image(p, w, h, g_opacity); last = now; }
}
static void input_publish(void) {
    uint64_t b; int32_t st[4]; pm_output(&g_pad, &b, st);
    b |= g_tc.buttons;
    int32_t lx = st[0] ? st[0] : g_tc.lx, ly = st[1] ? st[1] : g_tc.ly, rx = st[2] ? st[2] : g_tc.rx, ry = st[3] ? st[3] : g_tc.ry;
    static int log = -1; if (log < 0) { const char *e = getenv("DM_INPUT_LOG"); log = e && atoi(e); }   // diagnostic: every state change
    static uint64_t pb; static int32_t p[4];
    if (log && (b != pb || lx != p[0] || ly != p[1] || rx != p[2] || ry != p[3])) LOG("front: pad buttons %#llx left %d,%d right %d,%d", (unsigned long long)b, lx, ly, rx, ry);
    pb = b; p[0] = lx; p[1] = ly; p[2] = rx; p[3] = ry;
    dm_pad_set(b, lx, ly, rx, ry);
}
static void pad_used(void) {
    if (!g_overlay) return;
    g_overlay = 0; dm_vk_overlay_image(NULL, 0, 0, 0);
    tc_pointer none; tc_event(TC_CANCEL, 0, &none, 0, &g_tc); dm_touch_set(NULL, 0);
}
static int32_t on_input(struct android_app *app, AInputEvent *e) {
    (void)app;
    int src = AInputEvent_getSource(e), type = AInputEvent_getType(e);
    if (type == AINPUT_EVENT_TYPE_MOTION && (src & AINPUT_SOURCE_TOUCHSCREEN) == AINPUT_SOURCE_TOUCHSCREEN) {
        if (!g_overlay) { g_overlay = 1; tc_layout(g_ww, g_wh, g_dpi); }
        int a = AMotionEvent_getAction(e), act = a & AMOTION_EVENT_ACTION_MASK, pi = (a & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >> AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
        tc_pointer p[16]; int n = 0;
        for (size_t i = 0; i < AMotionEvent_getPointerCount(e) && n < 16; i++, n++) p[n] = (tc_pointer){AMotionEvent_getPointerId(e, i), AMotionEvent_getX(e, i), AMotionEvent_getY(e, i)};
        int ta = act == AMOTION_EVENT_ACTION_DOWN || act == AMOTION_EVENT_ACTION_POINTER_DOWN ? TC_DOWN : act == AMOTION_EVENT_ACTION_UP || act == AMOTION_EVENT_ACTION_POINTER_UP ? TC_UP
               : act == AMOTION_EVENT_ACTION_CANCEL ? TC_CANCEL : TC_MOVE;
        if (tc_event(ta, AMotionEvent_getPointerId(e, pi), p, n, &g_tc)) haptic_tick();
        { static int log = -1; if (log < 0) { const char *v = getenv("DM_INPUT_LOG"); log = v && atoi(v); }
          if (log && ta != TC_MOVE) LOG("front: touch %s id %d at %.0f,%.0f (%d down, %d to the game)", ta == TC_DOWN ? "down" : ta == TC_UP ? "up" : "cancel", AMotionEvent_getPointerId(e, pi), AMotionEvent_getX(e, pi), AMotionEvent_getY(e, pi), n, g_tc.ntouch); }
        int fr[4]; dm_vk_frame_rect(fr);
        dm_touch t[16]; int nt = 0;
        for (int i = 0; i < g_tc.ntouch && fr[2] > 0; i++) {
            float u = (g_tc.touch[i].x - fr[0]) / fr[2], v = (g_tc.touch[i].y - fr[1]) / fr[3];
            if (u >= 0 && u < 1 && v >= 0 && v < 1) t[nt++] = (dm_touch){g_tc.touch[i].id, u, v};
        }
        dm_touch_set(t, nt); input_publish(); overlay_push();
        return 1;
    }
    if (type == AINPUT_EVENT_TYPE_MOTION && (src & AINPUT_SOURCE_JOYSTICK) == AINPUT_SOURCE_JOYSTICK) {
#define AX(n) AMotionEvent_getAxisValue(e, AMOTION_EVENT_AXIS_##n, 0)
        pm_axes a = {.x = AX(X), .y = AX(Y), .z = AX(Z), .rz = AX(RZ), .hat_x = AX(HAT_X), .hat_y = AX(HAT_Y), .ltrigger = AX(LTRIGGER), .rtrigger = AX(RTRIGGER), .brake = AX(BRAKE), .gas = AX(GAS)};
#undef AX
        if (pm_motion(&g_pad, AInputEvent_getDeviceId(e), &a, (AMotionEvent_getAction(e) & AMOTION_EVENT_ACTION_MASK) == AMOTION_EVENT_ACTION_CANCEL) == PM_USED) pad_used();
        input_publish(); return 1;
    }
    if (type == AINPUT_EVENT_TYPE_KEY) {
        int a = AKeyEvent_getAction(e), r = pm_key(&g_pad, AInputEvent_getDeviceId(e), src, AKeyEvent_getKeyCode(e), a == AKEY_EVENT_ACTION_DOWN && !AKeyEvent_getRepeatCount(e) ? PM_DOWN : a == AKEY_EVENT_ACTION_UP ? PM_UP : PM_OTHER);
        if (r == PM_IGNORED) return 0;
        if (r == PM_USED) pad_used();
        input_publish(); return 1;
    }
    return 0;
}

// ---------- Activity events ----------
// Lifecycle as front_ios.m / hle_oe.c script_bg: pause = focus lost; stop = background (audio to the silent sink, guest clock frozen, graphics drained)
static int g_bg;
// A save commit copies the save to <save>.new, swaps it in and deletes the old one (hle_fs.c commit_mount_locked), all under the file system's locks on the game thread: .new exists
// exactly while one runs, and dm_fs_commit_count() only says a commit has ended. Leaving or exiting then would cut the commit off, so wait for it, up to 2 s (not
// at all after a wait ran out: a .new left by a failed copy would cost 2 s each time, until a commit succeeds)
static char g_savenew[2][600];   // the user save's and the device save's .new
static void save_paths(void) {   // the committed folders are named as in hle_fs.c (nxnative 7a3a875): the account save is DM_SAVE; device save data is DM_SAVE itself (the same folder, one .new) unless main also mounts account save data, then <DM_SAVE>_device. DM_DEVICE_SAVE is no longer read; no trailing slashes
    const char *u = getenv("DM_SAVE"); char b[560];
    if (u) { snprintf(b, sizeof b, "%s", u); for (size_t l = strlen(b); l > 1 && b[l - 1] == '/'; ) b[--l] = 0;
        snprintf(g_savenew[0], sizeof g_savenew[0], "%s.new", b); snprintf(g_savenew[1], sizeof g_savenew[1], "%s_device.new", b); }
}
static int commit_running(void) { for (int i = 0; i < 2; i++) if (g_savenew[i][0] && !access(g_savenew[i], F_OK)) return 1; return 0; }
static void save_wait(const char *why) {
    static uint64_t gave_up = ~0ull; uint64_t c0 = dm_fs_commit_count(); int ms = 0;
    if (c0 == gave_up || !commit_running()) return;
    while (ms < 2000 && commit_running()) { usleep(20000); ms += 20; }
    uint64_t c1 = dm_fs_commit_count(); if (commit_running()) gave_up = c0;
    LOG("front: %s: waited %d ms for the save commit (%s; commits %llu -> %llu)", why, ms, commit_running() ? "[RED] still running" : "done", (unsigned long long)c0, (unsigned long long)c1);
}
// Focus lost (notification shade, call screen, Recents): Android cancels held keys, but a stick's return to centre goes to the window that has focus now, so a held stick, hat
// or trigger would stay held on coming back (nxnative's iOS app resigns active the same way, DmGamepad.m). The pad stays shown; its touches are cancelled
static void input_release(void) {
    pm_release(&g_pad);
    tc_pointer none; tc_event(TC_CANCEL, 0, &none, 0, &g_tc); dm_touch_set(NULL, 0); input_publish();
}
// A controller that is switched off or unplugged may never report letting go, so what it held would stay held. While something is held, ask Android twice a second whether the controller still exists.
// Touches are left alone. On android_main's thread, which stays attached until _exit
static void pad_watch(ANativeActivity *a) {
    static JNIEnv *env; static jclass cls; static jmethodID get; static int off; static uint64_t next;
    if (off || !pm_held(&g_pad)) return;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); uint64_t now = (uint64_t)ts.tv_sec * 1000 + (uint64_t)ts.tv_nsec / 1000000;
    if (now < next) return;
    next = now + 500;
    if (!env) {
        JNIEnv *e; off = 1;   // for good if any of this fails: the game plays on as before
        if ((*a->vm)->AttachCurrentThread(a->vm, &e, NULL)) return;
        jclass c = (*e)->FindClass(e, "android/view/InputDevice");
        if (jfail(e, "FindClass InputDevice") || !c) return;
        get = (*e)->GetStaticMethodID(e, c, "getDevice", "(I)Landroid/view/InputDevice;");
        if (jfail(e, "GetStaticMethodID getDevice") || !get) return;
        cls = (*e)->NewGlobalRef(e, c); (*e)->DeleteLocalRef(e, c);
        if (!cls) return;
        env = e; off = 0;
    }
    jobject d = (*env)->CallStaticObjectMethod(env, cls, get, (jint)g_pad.device);
    if (jfail(env, "InputDevice.getDevice")) { off = 1; return; }
    if (d) { (*env)->DeleteLocalRef(env, d); return; }
    LOG("front: the controller went away: its buttons and sticks are released");
    pm_release(&g_pad); input_publish();
}
// Tell Android the game is paced at the present rate (DM_PRESENT_HZ, 60): on a 120 Hz panel it can then run the display at 60 (or a multiple) instead of guessing, and each paced
// frame stops waiting for the next 8.3 ms vblank. COMPATIBILITY_DEFAULT is the NDK's value for games (FIXED_SOURCE is for video). API 30, minSdk is 31. Not done for a 30
// fps step cap: only nvn_game_fps() exposes it, and it also reports 30 for ordinary interval-2 play, so it is not a cap signal
static void frame_rate_hint(struct android_app *app) {
    if (!app->window) return;
    const char *e = getenv("DM_PRESENT_HZ"); int hz = e ? atoi(e) : 60; if (hz <= 0) return;   // 0: uncapped (profiling)
    int32_t r = ANativeWindow_setFrameRate(app->window, (float)hz, ANATIVEWINDOW_FRAME_RATE_COMPATIBILITY_DEFAULT);
    LOG("front: frame rate hint %d fps%s", hz, r ? " [RED] refused" : "");
}
static void background(int on) {
    if (on) { dm_focus_set(2); dm_focus_set(3); dm_audio_suspend(1); guest_clock_freeze_for(1, 1); dm_gfx_suspend(1); }
    else { dm_gfx_suspend(0); guest_clock_freeze_for(1, 0); dm_audio_suspend(0); dm_focus_set(2); }
    g_bg = on;
}
static void window_size(struct android_app *app) {
    g_ww = ANativeWindow_getWidth(app->window); g_wh = ANativeWindow_getHeight(app->window); g_dpi = AConfiguration_getDensity(app->config);
    tc_layout(g_ww, g_wh, g_dpi);
}
static void on_cmd(struct android_app *app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            if (!app->window) break;
            window_size(app);
            LOG("front: window %dx%d", g_ww, g_wh);
            dm_vk_set_window(app->window);
            frame_rate_hint(app);
            if (!g_started) {
                g_started = 1;
                pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setstacksize(&at, 64 << 20);
                pthread_t th; if (pthread_create(&th, &at, host_thread, NULL)) LOG("[RED] front: cannot start the host thread");
            }
            break;
        case APP_CMD_WINDOW_RESIZED: case APP_CMD_CONFIG_CHANGED:
            if (app->window) window_size(app);
            break;
        case APP_CMD_TERM_WINDOW: dm_vk_set_window(NULL); break;
        case APP_CMD_LOST_FOCUS: input_release(); break;
        case APP_CMD_PAUSE: input_release(); if (g_started) dm_focus_set(2); break;
        case APP_CMD_STOP: if (g_started && !g_bg) { save_wait("stop"); background(1); } break;
        case APP_CMD_START: if (g_bg) background(0); break;
        case APP_CMD_RESUME: if (g_started) dm_focus_set(1); break;
        case APP_CMD_DESTROY: LOG("front: activity destroyed: exiting"); save_wait("destroy"); _exit(0);
    }
}
void android_main(struct android_app *app) {
    g_app = app;
    app->onAppCmd = on_cmd; app->onInputEvent = on_input;
    ANativeActivity_setWindowFlags(app->activity, AWINDOW_FLAG_KEEP_SCREEN_ON | AWINDOW_FLAG_FULLSCREEN, 0);
    paths_init(app->activity); save_paths();
    if (getenv("DM_MEMLOG") || getenv("DM_MEM_STOP")) { pthread_t th; if (!pthread_create(&th, NULL, mem_thread, NULL)) pthread_detach(th); }
    if (getenv("DM_TEST_TEXTBOX")) { pthread_t th; if (!pthread_create(&th, NULL, tb_test_thread, (void *)(uintptr_t)atoi(getenv("DM_TEST_TEXTBOX")))) pthread_detach(th); }
    tc_init(g_internal);
    haptics_init(app->activity);
    { const char *o = getenv("DM_PAD_OPACITY"); if (o) g_opacity = (float)atof(o); }
    for (;;) {
        int events; struct android_poll_source *src;
        while (ALooper_pollOnce(g_bg ? -1 : 17, NULL, &events, (void **)&src) >= 0) {   // the timeout: the pad's first layer waits for the first present, and a layer held back by overlay_push's rate limit; in the background there is nothing to draw: wait for an event
            if (src) src->process(app, src);
            if (app->destroyRequested) { LOG("front: destroy requested: exiting"); save_wait("destroy"); _exit(0); }
            pad_watch(app->activity);   // here too: a steady stream of touches never reaches the timeout below
        }
        pad_watch(app->activity);
        overlay_push();
    }
}
