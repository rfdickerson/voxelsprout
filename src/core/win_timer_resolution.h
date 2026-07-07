#pragma once

// Windows-only: requests/releases 1 ms scheduler timer resolution for the
// process lifetime. Header-only, and windows.h is fully contained inside the
// #if block below (with NOMINMAX/WIN32_LEAN_AND_MEAN) so its macro soup
// (min/max/near/far) never leaks into callers — safe to include from large
// translation units like app.cc that use std::min/std::max freely.
//
// Why this matters: Windows' default scheduler tick is ~15.6 ms. Any blocking
// wait inside a render loop (vkWaitForFences, vkAcquireNextImageKHR, GLFW's
// internal event wait) can round up to that granularity, producing a periodic
// ~16 ms hitch riding on top of an otherwise much faster frame time. Call
// requestHighResTimer() once at startup and releaseHighResTimer() once at
// shutdown (matched pair).

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#endif

namespace odai::core {

inline void requestHighResTimer() {
#if defined(_WIN32)
    timeBeginPeriod(1);
#endif
}

inline void releaseHighResTimer() {
#if defined(_WIN32)
    timeEndPeriod(1);
#endif
}

}  // namespace odai::core
