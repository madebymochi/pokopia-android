// Host scheduling of the guest threads on Android. The Darwin QoS calls the host makes are no-ops here (compat/include/pthread/qos.h), so every guest thread ran
// at nice 0 whatever its Horizon priority: the guest main thread lost ~0.2 s a second off the CPU right after waking workers (futex wake, preempted) while
// it decides the frame rate (measured on a Galaxy S22).
// DM_THREAD_NICE="name=nice,...": nice value of guest threads by guest name ("main thread" is the guest main thread), set on the thread when it starts.
// DM_ADPF=1 or DM_ADPF="name,name,...": a performance hint session (APerformanceHint, API 33, loaded with dlsym: the build targets 31) over those threads
// (1: main thread, Presentation Thread, ActorCalcSub0, ActorCalcSub1), targeting the present period and fed each frame's work time (hle_nvn.c dm_host_frame).
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>
#include "dm.h"

typedef struct APerformanceHintManager APerformanceHintManager;
typedef struct APerformanceHintSession APerformanceHintSession;
static struct {
    APerformanceHintManager *(*get_manager)(void);
    APerformanceHintSession *(*create)(APerformanceHintManager *, const int32_t *, size_t, int64_t);
    int (*update_target)(APerformanceHintSession *, int64_t);
    int (*report)(APerformanceHintSession *, int64_t);
    int (*set_threads)(APerformanceHintSession *, const pid_t *, size_t);   // API 34
    void (*close)(APerformanceHintSession *);
    int64_t (*rate)(APerformanceHintManager *);   // -1: the device does not support hint sessions
} H;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_init, g_adpf;
static char g_nice[512], g_adpf_names[512];
enum { NT = 16 };
static int32_t g_tids[NT]; static int g_nt, g_nt_session;
static APerformanceHintSession *g_sess; static int64_t g_target;
static uint64_t g_nrep, g_nover;
// Scheduler accounting of the two threads on the frame's path (/proc/self/task/<tid>/schedstat: ns on CPU, ns runnable waiting for a CPU, slices), logged every 300 frames
static struct { pid_t tid; uint64_t run, wait, n; } g_st[2];
static uint64_t g_st_t, g_st_frames;
static int st_read(pid_t tid, uint64_t *run, uint64_t *wait, uint64_t *n) {
    char path[64], b[96]; snprintf(path, sizeof path, "/proc/self/task/%d/schedstat", (int)tid);
    FILE *f = fopen(path, "r"); if (!f) return 0;
    int ok = fgets(b, sizeof b, f) && sscanf(b, "%llu %llu %llu", (unsigned long long *)run, (unsigned long long *)wait, (unsigned long long *)n) == 3; fclose(f); return ok;
}
static uint64_t mono_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }
static void st_note(void) {   // present thread, once per frame
    if (++g_st_frames % 300) return;
    uint64_t now = mono_ns(); double dt = g_st_t ? (now - g_st_t) / 1e9 : 0; char line[256]; int o = 0;
    for (int i = 0; i < 2; i++) { uint64_t r, w, n; if (!g_st[i].tid || !st_read(g_st[i].tid, &r, &w, &n)) continue;
        if (dt > 0) o += snprintf(line + o, sizeof line - o, " %s cpu %.0f wait %.0f ms/s, %.0f slices/s;", i ? "Presentation Thread" : "main thread", (r - g_st[i].run) / 1e6 / dt, (w - g_st[i].wait) / 1e6 / dt, (n - g_st[i].n) / dt);
        g_st[i].run = r; g_st[i].wait = w; g_st[i].n = n; }
    if (dt > 0 && o) LOG("perf: sched over %.1f s (300 frames, %.1f fps):%s", dt, 300 / dt, line);
    g_st_t = now;
}

static void init_locked(void) {
    if (g_init) return;
    g_init = 1;
    const char *n = getenv("DM_THREAD_NICE"); if (n) snprintf(g_nice, sizeof g_nice, ",%s,", n);
    const char *a = getenv("DM_ADPF");
    if (a && *a && strcmp(a, "0")) {
        snprintf(g_adpf_names, sizeof g_adpf_names, ",%s,", !strcmp(a, "1") ? "main thread,Presentation Thread,ActorCalcSub0,ActorCalcSub1" : a);
        void *lib = dlopen("libandroid.so", RTLD_NOW);
        if (lib) {
            H.get_manager = (APerformanceHintManager *(*)(void))dlsym(lib, "APerformanceHint_getManager");
            H.create = (APerformanceHintSession *(*)(APerformanceHintManager *, const int32_t *, size_t, int64_t))dlsym(lib, "APerformanceHint_createSession");
            H.update_target = (int (*)(APerformanceHintSession *, int64_t))dlsym(lib, "APerformanceHint_updateTargetWorkDuration");
            H.report = (int (*)(APerformanceHintSession *, int64_t))dlsym(lib, "APerformanceHint_reportActualWorkDuration");
            H.set_threads = (int (*)(APerformanceHintSession *, const pid_t *, size_t))dlsym(lib, "APerformanceHint_setThreads");
            H.close = (void (*)(APerformanceHintSession *))dlsym(lib, "APerformanceHint_closeSession");
            H.rate = (int64_t (*)(APerformanceHintManager *))dlsym(lib, "APerformanceHint_getPreferredUpdateRateNanos");
        }
        g_adpf = H.get_manager && H.create && H.update_target && H.report && H.close;
        LOG("perf: DM_ADPF threads %s: %s", g_adpf_names, g_adpf ? "performance hint API found" : "[RED] APerformanceHint not available");
    }
    if (n) LOG("perf: DM_THREAD_NICE %s", n);
}

static int listed(const char *list, const char *name, const char **after) {   // ",name" followed by '=' or ',' in a ",a,b=c," list
    size_t l = strlen(name);
    for (const char *p = list; (p = strstr(p, name)); p++)
        if (p > list && p[-1] == ',' && (p[l] == '=' || p[l] == ',')) { if (after) *after = p + l; return 1; }
    return 0;
}

// hle_thread.c: on each guest thread once it has its guest name (the guest main thread as "main thread")
void dm_host_thread_started(const char *name, int prio) {
    pthread_mutex_lock(&g_mu); init_locked();
    const char *v;
    if (g_nice[0] && listed(g_nice, name, &v) && *v == '=') {
        int nice = atoi(v + 1), r = setpriority(PRIO_PROCESS, (id_t)gettid(), nice);
        LOG("perf: thread '%s' (priority %d, tid %d) nice %d%s", name, prio, (int)gettid(), nice, r ? " [RED] refused" : "");
    }
    if (g_adpf && listed(g_adpf_names, name, NULL) && g_nt < NT) g_tids[g_nt++] = (int32_t)gettid();
    if (!strcmp(name, "main thread")) g_st[0].tid = gettid(); else if (!strcmp(name, "Presentation Thread")) g_st[1].tid = gettid();
    pthread_mutex_unlock(&g_mu);
}

// hle_nvn.c, once per presented frame: the frame's work time (present call to present call, minus the present itself) and the present period
void dm_host_frame(uint64_t work_ns, uint64_t period_ns) {
    st_note();
    if (!g_adpf || !period_ns) return;
    pthread_mutex_lock(&g_mu);
    if (g_nt && g_nt != g_nt_session) {   // threads added since the session was made
        if (g_sess && H.set_threads && !H.set_threads(g_sess, g_tids, (size_t)g_nt)) g_nt_session = g_nt;
        else {
            if (g_sess) H.close(g_sess);
            APerformanceHintManager *m = H.get_manager();
            g_sess = m ? H.create(m, g_tids, (size_t)g_nt, (int64_t)period_ns) : NULL; g_target = (int64_t)period_ns; g_nt_session = g_nt;
            LOG("perf: hint session over %d threads, target %.2f ms, preferred update rate %lld ns%s", g_nt, period_ns / 1e6, m && H.rate ? (long long)H.rate(m) : -2ll,
                g_sess ? "" : " [RED] createSession failed: hint sessions off");   // the S22 (Android 16, Exynos 2200) returns none
            if (!g_sess) g_adpf = 0;
        }
    }
    if (g_sess) {
        if ((int64_t)period_ns != g_target) { H.update_target(g_sess, (int64_t)period_ns); g_target = (int64_t)period_ns; }
        H.report(g_sess, (int64_t)work_ns); g_nrep++; if (work_ns > period_ns) g_nover++;
        if (g_nrep % 1800 == 0) LOG("perf: hint session: %llu frames reported, %llu over the target", (unsigned long long)g_nrep, (unsigned long long)g_nover);
    }
    pthread_mutex_unlock(&g_mu);
}
