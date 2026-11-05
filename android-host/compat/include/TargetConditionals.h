// compat: Apple's TargetConditionals.h for the Android / Linux host build (no Apple target is active).
#pragma once
#define TARGET_OS_MAC 0
#define TARGET_OS_OSX 0
#define TARGET_OS_IPHONE 0
#define TARGET_OS_IOS 0
#define TARGET_OS_SIMULATOR 0
#define TARGET_OS_ANDROID 1
