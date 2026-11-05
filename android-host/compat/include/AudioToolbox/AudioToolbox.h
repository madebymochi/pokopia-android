// compat: the AudioToolbox surface hle_audio.c / hle_movie.c use. The output AudioUnit is AAudio (compat/src/audio_aaudio.c: 16-bit PCM, a data callback
// that calls the render callback); hle_audio.c uses it with DM_AUDIO=1 (the APK front sets it), else its silent sink. No converter is found: movie audio stays silent.
#pragma once
#include <stdint.h>
#include <stddef.h>
typedef uint32_t UInt32; typedef int32_t SInt32; typedef int64_t SInt64; typedef double Float64; typedef int32_t OSStatus;
#define noErr 0
typedef struct dm_au *AudioUnit; typedef AudioUnit AudioComponentInstance; typedef struct dm_ac *AudioComponent; typedef struct dm_conv *AudioConverterRef;
typedef UInt32 AudioUnitRenderActionFlags, AudioUnitPropertyID, AudioUnitScope, AudioUnitElement, AudioConverterPropertyID, AudioFormatID, AudioFormatFlags;
typedef struct { Float64 mSampleTime; uint64_t mHostTime; Float64 mRateScalar; uint64_t mWordClockTime; char mSMPTETime[24]; UInt32 mFlags, mReserved; } AudioTimeStamp;
typedef struct { UInt32 mNumberChannels, mDataByteSize; void *mData; } AudioBuffer;
typedef struct { UInt32 mNumberBuffers; AudioBuffer mBuffers[1]; } AudioBufferList;
typedef struct { UInt32 componentType, componentSubType, componentManufacturer, componentFlags, componentFlagsMask; } AudioComponentDescription;
typedef struct { Float64 mSampleRate; AudioFormatID mFormatID; AudioFormatFlags mFormatFlags; UInt32 mBytesPerPacket, mFramesPerPacket, mBytesPerFrame, mChannelsPerFrame, mBitsPerChannel, mReserved; } AudioStreamBasicDescription;
typedef struct { SInt64 mStartOffset; UInt32 mVariableFramesInPacket, mDataByteSize; } AudioStreamPacketDescription;
typedef UInt32 AudioDeviceID;   // hle_audio.c's Mac-only buffer-size query (au_buffer) names it
typedef struct { Float64 mMinimum, mMaximum; } AudioValueRange;
typedef void (*AudioUnitPropertyListenerProc)(void *, AudioUnit, AudioUnitPropertyID, AudioUnitScope, AudioUnitElement);
typedef OSStatus (*AURenderCallback)(void *, AudioUnitRenderActionFlags *, const AudioTimeStamp *, UInt32, UInt32, AudioBufferList *);
typedef struct { AURenderCallback inputProc; void *inputProcRefCon; } AURenderCallbackStruct;
typedef OSStatus (*AudioConverterComplexInputDataProc)(AudioConverterRef, UInt32 *, AudioBufferList *, AudioStreamPacketDescription **, void *);
enum { kAudioUnitType_Output = 0x61756f75, kAudioUnitSubType_DefaultOutput = 0x64656620, kAudioUnitSubType_RemoteIO = 0x72696f63, kAudioUnitManufacturer_Apple = 0x6170706c,
       kAudioUnitProperty_StreamFormat = 8, kAudioUnitProperty_SetRenderCallback = 23, kAudioDevicePropertyBufferFrameSize = 0x6673697a, kAudioDevicePropertyBufferFrameSizeRange = 0x66737a23, kAudioOutputUnitProperty_CurrentDevice = 2000, kAudioObjectUnknown = 0,
       kAudioUnitScope_Global = 0, kAudioUnitScope_Input = 1, kAudioUnitScope_Output = 2,
       kAudioFormatLinearPCM = 0x6c70636d, kAudioFormatMPEG4AAC = 0x61616320, kAudioFormatMPEGLayer3 = 0x2e6d7033,
       kLinearPCMFormatFlagIsFloat = 1, kLinearPCMFormatFlagIsSignedInteger = 4, kLinearPCMFormatFlagIsPacked = 8, kLinearPCMFormatFlagIsNonInterleaved = 32,
       kAudioConverterDecompressionMagicCookie = 0x646d6763 };
AudioComponent AudioComponentFindNext(AudioComponent c, const AudioComponentDescription *d);
OSStatus AudioComponentInstanceNew(AudioComponent c, AudioComponentInstance *o);
OSStatus AudioComponentInstanceDispose(AudioComponentInstance u);
OSStatus AudioUnitSetProperty(AudioUnit u, AudioUnitPropertyID p, AudioUnitScope s, AudioUnitElement e, const void *d, UInt32 n);
OSStatus AudioUnitGetProperty(AudioUnit u, AudioUnitPropertyID p, AudioUnitScope s, AudioUnitElement e, void *d, UInt32 *n);
OSStatus AudioUnitAddPropertyListener(AudioUnit u, AudioUnitPropertyID p, AudioUnitPropertyListenerProc f, void *ud);
OSStatus AudioUnitRemovePropertyListenerWithUserData(AudioUnit u, AudioUnitPropertyID p, AudioUnitPropertyListenerProc f, void *ud);
OSStatus AudioUnitInitialize(AudioUnit u);
OSStatus AudioUnitUninitialize(AudioUnit u);
OSStatus AudioOutputUnitStart(AudioUnit u);
OSStatus AudioOutputUnitStop(AudioUnit u);
static inline OSStatus AudioConverterNew(const AudioStreamBasicDescription *i, const AudioStreamBasicDescription *o, AudioConverterRef *c) { (void)i; (void)o; *c = NULL; return -1; }
static inline OSStatus AudioConverterDispose(AudioConverterRef c) { (void)c; return noErr; }
static inline OSStatus AudioConverterSetProperty(AudioConverterRef c, AudioConverterPropertyID p, UInt32 n, const void *d) { (void)c; (void)p; (void)n; (void)d; return -1; }
static inline OSStatus AudioConverterFillComplexBuffer(AudioConverterRef c, AudioConverterComplexInputDataProc f, void *u, UInt32 *n, AudioBufferList *o, AudioStreamPacketDescription *pd) { (void)c; (void)f; (void)u; (void)o; (void)pd; *n = 0; return -1; }
