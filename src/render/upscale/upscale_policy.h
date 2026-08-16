#pragma once

// Upscaler backend selection, Vulkan-free. See upscaler.cc for why this is
// separate from the renderer, and for the contract that resolveUpscaler() always
// returns a backend that will actually run.

#include "render/renderer_types.h"

#include <string_view>

namespace odai::render {

// True when this binary was built with the backend's implementation. A compile
// -time fact: XeSS and DLSS are vendor SDKs, so this answers "was the SDK there
// when this was built", not "will it work on this machine".
[[nodiscard]] bool upscalerBackendCompiledIn(UpscalerBackend backend);

[[nodiscard]] const char* upscalerBackendName(UpscalerBackend backend);
[[nodiscard]] const char* upscalerQualityName(UpscalerQuality quality);

// Lower-case names, as a config file or a command line spells them. Return false
// on an unrecognised string rather than silently defaulting -- a typo'd backend
// name should be reported, not quietly become something else.
bool parseUpscalerBackend(std::string_view text, UpscalerBackend& outBackend);
bool parseUpscalerQuality(std::string_view text, UpscalerQuality& outQuality);

// Decides what will actually run, given what was asked for and whether the XeSS
// runtime accepted this device. Never fails: an unavailable backend falls back
// to Temporal with `reason` set to why.
[[nodiscard]] UpscalerStatus resolveUpscaler(
    const UpscalerSettings& settings, bool runtimeSupportsXess);

}  // namespace odai::render
