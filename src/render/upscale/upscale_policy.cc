// Upscaler backend selection.
//
// Deliberately Vulkan-free and in src/render/ rather than
// src/render/backend/vulkan/: deciding WHICH backend runs is a question about
// what was compiled in and what the caller asked for, not about the device, and
// keeping it separable is what lets it be reasoned about (and later tested)
// without standing up a renderer.
//
// The contract the rest of the engine relies on: resolveUpscaler() ALWAYS
// returns a usable backend. Asking for XeSS on a build without the SDK, or on a
// machine whose driver cannot run it, is not an error -- it reports the reason
// and hands back Temporal. That is what lets one command line, one config file
// and one CI invocation work across machines with different GPUs and different
// SDKs installed, which is the whole point of putting a vendor upscaler behind
// an option rather than a hard dependency.

#include "render/upscale/upscale_policy.h"

namespace odai::render {

bool upscalerBackendCompiledIn(UpscalerBackend backend) {
    switch (backend) {
    case UpscalerBackend::Off:
    case UpscalerBackend::Temporal:
        // Built from the jittered TAA and the skinned motion vectors already in
        // this tree, so it needs no SDK and is always present.
        return true;
    case UpscalerBackend::Xess:
#if defined(ODAI_HAS_XESS)
        return true;
#else
        return false;
#endif
    case UpscalerBackend::Fsr:
    case UpscalerBackend::Dlss:
        // Declared in the enum so the selection path, the config parsing and the
        // status reporting are all exercised for them -- and so that asking for
        // one gives a specific answer instead of "unknown backend". Neither has
        // an implementation; see the CMake options.
        return false;
    }
    return false;
}

const char* upscalerBackendName(UpscalerBackend backend) {
    switch (backend) {
    case UpscalerBackend::Off: return "off";
    case UpscalerBackend::Temporal: return "temporal";
    case UpscalerBackend::Xess: return "xess";
    case UpscalerBackend::Fsr: return "fsr";
    case UpscalerBackend::Dlss: return "dlss";
    }
    return "unknown";
}

const char* upscalerQualityName(UpscalerQuality quality) {
    switch (quality) {
    case UpscalerQuality::UltraQuality: return "ultraquality";
    case UpscalerQuality::Quality: return "quality";
    case UpscalerQuality::Balanced: return "balanced";
    case UpscalerQuality::Performance: return "performance";
    case UpscalerQuality::UltraPerformance: return "ultraperformance";
    case UpscalerQuality::Native: return "native";
    }
    return "unknown";
}

bool parseUpscalerBackend(std::string_view text, UpscalerBackend& outBackend) {
    if (text == "off" || text == "none" || text == "native") {
        outBackend = UpscalerBackend::Off;
    } else if (text == "temporal" || text == "internal" || text == "taau") {
        outBackend = UpscalerBackend::Temporal;
    } else if (text == "xess") {
        outBackend = UpscalerBackend::Xess;
    } else if (text == "fsr") {
        outBackend = UpscalerBackend::Fsr;
    } else if (text == "dlss") {
        outBackend = UpscalerBackend::Dlss;
    } else {
        return false;
    }
    return true;
}

bool parseUpscalerQuality(std::string_view text, UpscalerQuality& outQuality) {
    if (text == "ultraquality" || text == "uq") {
        outQuality = UpscalerQuality::UltraQuality;
    } else if (text == "quality" || text == "q") {
        outQuality = UpscalerQuality::Quality;
    } else if (text == "balanced" || text == "b") {
        outQuality = UpscalerQuality::Balanced;
    } else if (text == "performance" || text == "p") {
        outQuality = UpscalerQuality::Performance;
    } else if (text == "ultraperformance" || text == "up") {
        outQuality = UpscalerQuality::UltraPerformance;
    } else if (text == "native" || text == "n") {
        outQuality = UpscalerQuality::Native;
    } else {
        return false;
    }
    return true;
}

UpscalerStatus resolveUpscaler(const UpscalerSettings& settings, bool runtimeSupportsXess) {
    UpscalerStatus status{};
    status.requested = settings.backend;
    status.active = settings.backend;
    status.compiledIn = upscalerBackendCompiledIn(settings.backend);
    status.runtimeAvailable = status.compiledIn;
    status.renderScale = 1.0f;
    status.reason = "";

    if (!status.compiledIn) {
        switch (settings.backend) {
        case UpscalerBackend::Xess:
            status.reason = "XeSS was not compiled in (configure with -DODAI_ENABLE_XESS=ON "
                            "and -DODAI_XESS_SDK_DIR=<sdk>)";
            break;
        case UpscalerBackend::Fsr:
            status.reason = "the FSR backend is declared but not implemented";
            break;
        case UpscalerBackend::Dlss:
            status.reason = "the DLSS backend is declared but not implemented";
            break;
        default:
            status.reason = "backend unavailable";
            break;
        }
        status.active = UpscalerBackend::Temporal;
        status.runtimeAvailable = true;
    } else if (settings.backend == UpscalerBackend::Xess && !runtimeSupportsXess) {
        // Compiled in but the device or driver said no. A distinct case from the
        // one above and worth its own message: the fix is a driver, not a
        // rebuild.
        status.reason = "the XeSS runtime reported this device as unsupported";
        status.active = UpscalerBackend::Temporal;
        status.runtimeAvailable = false;
    }

    // Render scale follows the preset for every reconstructing backend. Off
    // means native, which is the one case where the preset is meaningless --
    // honouring it there would render at 1/1.5 and then not reconstruct, which
    // is just a blurry frame with no upscaler to blame.
    if (status.active != UpscalerBackend::Off) {
        status.renderScale = upscalerQualityScale(settings.quality);
    }
    return status;
}

}  // namespace odai::render
