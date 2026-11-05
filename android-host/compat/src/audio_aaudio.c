// compat: CoreAudio's output AudioUnit on AAudio, for hle_audio.c (AudioToolbox.h). One AAudio output stream per unit: 16-bit interleaved PCM at the
// unit's stream format, low-latency shared mode, usage GAME; the data callback calls the unit's render callback with one interleaved buffer.
// A disconnected stream (headphones unplugged, route change) is reopened on its own thread and restarted if it was playing. A stream that cannot be opened (at start, or
// after an error) does not stop the game: AudioUnitInitialize still succeeds, AudioOutputUnitStart tries to open one and fails while it cannot, and hle_audio.c then runs the
// silent sink and its watchdog calls Start again after 1, 2, 4, 8 s (so an error other than a disconnect is recovered there, with that back-off).
#include <AudioToolbox/AudioToolbox.h>
#include <aaudio/AAudio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include "dm.h"

struct dm_ac { int unused; };
struct dm_au {
    AudioStreamBasicDescription fmt; AURenderCallbackStruct cb; int have_fmt, have_cb;
    AAudioStream *st; int playing; pthread_mutex_t mu;
};
static struct dm_ac g_output;

AudioComponent AudioComponentFindNext(AudioComponent c, const AudioComponentDescription *d) {
    return !c && d && d->componentType == kAudioUnitType_Output ? &g_output : NULL;
}
OSStatus AudioComponentInstanceNew(AudioComponent c, AudioComponentInstance *o) {
    if (c != &g_output) { *o = NULL; return -1; }
    struct dm_au *u = calloc(1, sizeof *u); pthread_mutex_init(&u->mu, NULL); *o = u; return noErr;
}
OSStatus AudioUnitSetProperty(AudioUnit u, AudioUnitPropertyID p, AudioUnitScope s, AudioUnitElement e, const void *d, UInt32 n) {
    (void)s; (void)e;
    if (!u) return -1;
    if (p == kAudioUnitProperty_StreamFormat && n >= sizeof u->fmt) {
        const AudioStreamBasicDescription *f = d;
        if (f->mFormatID != kAudioFormatLinearPCM || f->mBitsPerChannel != 16 || (f->mFormatFlags & (kLinearPCMFormatFlagIsFloat | kLinearPCMFormatFlagIsNonInterleaved))) return -1;
        u->fmt = *f; u->have_fmt = 1; return noErr;
    }
    if (p == kAudioUnitProperty_SetRenderCallback && n >= sizeof u->cb) { u->cb = *(const AURenderCallbackStruct *)d; u->have_cb = 1; return noErr; }
    return -1;   // the device buffer size (a Mac diagnostic) and the rest: not supported
}

// hle_audio.c reads the Mac output device and its buffer-size range here (au_buffer, compiled because TARGET_OS_IPHONE is 0): CurrentDevice fails, which ends that query at once,
// as AAudio sizes its own buffer in open_stream. The stream format is answered as set. The device-change listeners are accepted and never called: a disconnect is handled by on_error.
OSStatus AudioUnitGetProperty(AudioUnit u, AudioUnitPropertyID p, AudioUnitScope s, AudioUnitElement e, void *d, UInt32 *n) {
    (void)s; (void)e;
    if (!u || !d || !n) return -1;
    if (p == kAudioUnitProperty_StreamFormat && u->have_fmt && *n >= sizeof u->fmt) { memcpy(d, &u->fmt, sizeof u->fmt); *n = sizeof u->fmt; return noErr; }
    return -1;
}
OSStatus AudioUnitAddPropertyListener(AudioUnit u, AudioUnitPropertyID p, AudioUnitPropertyListenerProc f, void *ud) { (void)u; (void)p; (void)f; (void)ud; return noErr; }
OSStatus AudioUnitRemovePropertyListenerWithUserData(AudioUnit u, AudioUnitPropertyID p, AudioUnitPropertyListenerProc f, void *ud) { (void)u; (void)p; (void)f; (void)ud; return noErr; }

static aaudio_data_callback_result_t on_data(AAudioStream *st, void *ctx, void *data, int32_t frames) {
    (void)st;
    struct dm_au *u = ctx;
    AudioBufferList bl = {1, {{u->fmt.mChannelsPerFrame, (UInt32)frames * u->fmt.mBytesPerFrame, data}}};
    AudioUnitRenderActionFlags fl = 0; AudioTimeStamp ts; memset(&ts, 0, sizeof ts);
    if (u->cb.inputProc(u->cb.inputProcRefCon, &fl, &ts, 0, (UInt32)frames, &bl)) memset(data, 0, (size_t)frames * u->fmt.mBytesPerFrame);
    return AAUDIO_CALLBACK_RESULT_CONTINUE;
}
static void on_error(AAudioStream *st, void *ctx, aaudio_result_t err);
static aaudio_result_t open_stream(struct dm_au *u) {
    AAudioStreamBuilder *b; aaudio_result_t r = AAudio_createStreamBuilder(&b);
    if (r != AAUDIO_OK) return r;
    AAudioStreamBuilder_setDirection(b, AAUDIO_DIRECTION_OUTPUT);
    AAudioStreamBuilder_setFormat(b, AAUDIO_FORMAT_PCM_I16);
    AAudioStreamBuilder_setChannelCount(b, (int32_t)u->fmt.mChannelsPerFrame);
    AAudioStreamBuilder_setSampleRate(b, (int32_t)u->fmt.mSampleRate);
    AAudioStreamBuilder_setPerformanceMode(b, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    AAudioStreamBuilder_setSharingMode(b, AAUDIO_SHARING_MODE_SHARED);
    AAudioStreamBuilder_setUsage(b, AAUDIO_USAGE_GAME);
    AAudioStreamBuilder_setContentType(b, AAUDIO_CONTENT_TYPE_MUSIC);
    AAudioStreamBuilder_setDataCallback(b, on_data, u);
    AAudioStreamBuilder_setErrorCallback(b, on_error, u);
    r = AAudioStreamBuilder_openStream(b, &u->st);
    AAudioStreamBuilder_delete(b);
    if (r == AAUDIO_OK) {   // on_data fills frames * the game's bytes per frame: a stream of another shape would be overrun (channels, format) or play at the wrong pitch and pace (rate)
        int32_t ch = AAudioStream_getChannelCount(u->st), hz = AAudioStream_getSampleRate(u->st); aaudio_format_t f = AAudioStream_getFormat(u->st);
        if (hz != (int32_t)u->fmt.mSampleRate) { static int warned; if (!warned++) LOG("[RED] aaudio: the device opened %d Hz instead of %d Hz: sound plays at the wrong pitch", hz, (int)u->fmt.mSampleRate); }   // kept open: wrong pitch beats no sound
        if (ch != (int32_t)u->fmt.mChannelsPerFrame || f != AAUDIO_FORMAT_PCM_I16) {
            LOG("[RED] aaudio: the device opened %d Hz %d ch format %d instead of %d Hz %d ch 16-bit PCM: stream closed, no sound", hz, ch, (int)f, (int)u->fmt.mSampleRate, (int)u->fmt.mChannelsPerFrame);
            AAudioStream_close(u->st); u->st = NULL; return AAUDIO_ERROR_INVALID_FORMAT;
        }
        // a little above the minimum: the game's renderer fills 5 ms blocks on its own thread
        AAudioStream_setBufferSizeInFrames(u->st, AAudioStream_getFramesPerBurst(u->st) * 4);
        LOG("aaudio: output %d Hz %d ch, burst %d frames, buffer %d frames, %s", AAudioStream_getSampleRate(u->st), AAudioStream_getChannelCount(u->st),
            AAudioStream_getFramesPerBurst(u->st), AAudioStream_getBufferSizeInFrames(u->st),
            AAudioStream_getPerformanceMode(u->st) == AAUDIO_PERFORMANCE_MODE_LOW_LATENCY ? "low latency" : "normal latency");
    } else {   // retried by the watchdog every 1-8 s: log the first few, then every 8th
        static int nfail; if (nfail++ < 4 || !(nfail & 7)) LOG("[RED] aaudio: cannot open an output stream: %s (attempt %d)", AAudio_convertResultToText(r), nfail);
        u->st = NULL;
    }
    return r;
}
static void *reopen(void *p) {   // not on the callback thread: AAudio forbids closing a stream from its own callbacks
    struct dm_au *u = p;
    pthread_mutex_lock(&u->mu);
    if (u->st) { AAudioStream_close(u->st); u->st = NULL; }
    aaudio_result_t r = open_stream(u);
    if (r == AAUDIO_OK && u->playing) AAudioStream_requestStart(u->st);
    if (r == AAUDIO_OK) LOG("aaudio: output reopened after a disconnect%s", u->playing ? " and restarted" : "");
    else LOG("[RED] aaudio: reopening after a disconnect failed: the watchdog retries");
    pthread_mutex_unlock(&u->mu);
    return NULL;
}
static void on_error(AAudioStream *st, void *ctx, aaudio_result_t err) {
    (void)st;
    if (err != AAUDIO_ERROR_DISCONNECTED) { LOG("[RED] aaudio: stream error %s: the callbacks stop and the watchdog reopens the stream", AAudio_convertResultToText(err)); return; }
    pthread_t t; if (!pthread_create(&t, NULL, reopen, ctx)) pthread_detach(t);
}
OSStatus AudioUnitInitialize(AudioUnit u) {
    if (!u || !u->have_fmt || !u->have_cb) return -1;
    pthread_mutex_lock(&u->mu); aaudio_result_t r = u->st ? AAUDIO_OK : open_stream(u); pthread_mutex_unlock(&u->mu);
    if (r != AAUDIO_OK) LOG("[RED] aaudio: no output stream yet: the game runs without sound (silent sink) and the stream is tried again");   // not a failure: the game used to quit here
    return noErr;
}
OSStatus AudioOutputUnitStart(AudioUnit u) {
    if (!u) return -1;
    pthread_mutex_lock(&u->mu);
    aaudio_result_t r = u->st ? AAudioStream_requestStart(u->st) : AAUDIO_ERROR_INVALID_STATE;
    if (r != AAUDIO_OK) {   // no stream, or one that is in an error state (it cannot be started again): a fresh one, started once
        if (u->st) { AAudioStream_close(u->st); u->st = NULL; }
        r = open_stream(u); if (r == AAUDIO_OK) r = AAudioStream_requestStart(u->st);
    }
    if (r == AAUDIO_OK) u->playing = 1;
    pthread_mutex_unlock(&u->mu);
    return r == AAUDIO_OK ? noErr : -1;
}
OSStatus AudioOutputUnitStop(AudioUnit u) {
    if (!u) return noErr;
    pthread_mutex_lock(&u->mu); u->playing = 0; if (u->st) AAudioStream_requestStop(u->st); pthread_mutex_unlock(&u->mu);
    return noErr;
}
OSStatus AudioUnitUninitialize(AudioUnit u) {
    if (!u) return noErr;
    pthread_mutex_lock(&u->mu); if (u->st) { AAudioStream_close(u->st); u->st = NULL; } u->playing = 0; pthread_mutex_unlock(&u->mu);
    return noErr;
}
OSStatus AudioComponentInstanceDispose(AudioComponentInstance u) {
    if (!u) return noErr;
    AudioUnitUninitialize(u); pthread_mutex_destroy(&u->mu); free(u);
    return noErr;
}
