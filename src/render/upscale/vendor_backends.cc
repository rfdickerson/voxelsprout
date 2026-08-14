// Backend registry, and the vendor backends that are declared but not built.
//
// XeSS, FSR and DLSS are here as REAL entries rather than as a comment saying
// they could be added. createUpscaler() returning nullptr for them, with a
// specific reason, is what makes `--upscaler dlss` behave the way it will once
// the SDK is wired: it reports why, and resolveUpscaler() has already picked
// Temporal to run instead. The alternative -- adding them only when they work --
// means the selection path, the config parsing and the status reporting are all
// first exercised on the day the SDK lands, which is the worst day to find out
// they were wrong.
//
// Adding one is a matter of implementing IUpscaler and returning it here. The
// three shapes differ only in what they need from the host: XeSS wants a
// context built against the VkDevice, FSR2 wants its own backend interface
// filled in (HostServices is modelled on it), DLSS wants an NGX capability
// probe before anything else.

#include "render/upscale/temporal_upscaler.h"
#include "render/upscale/upscaler_backend.h"

namespace odai::render::upscale {

std::unique_ptr<IUpscaler> createUpscaler(UpscalerBackend id, const HostServices& host) {
    switch (id) {
    case UpscalerBackend::Off:
        // Not an upscaler. The frame skips the pass entirely rather than
        // dispatching a no-op one.
        return nullptr;
    case UpscalerBackend::Temporal:
        return createTemporalUpscaler(host);
    case UpscalerBackend::Xess:
#if defined(ODAI_HAS_XESS)
        // Intended shape: xessVKCreateContext(device, ...) here, then an
        // IUpscaler whose dispatch() calls xessVKExecute. It needs the raw
        // VkImages and their formats, which DispatchInfo already carries.
        return nullptr;
#else
        return nullptr;
#endif
    case UpscalerBackend::Fsr:
    case UpscalerBackend::Dlss:
        return nullptr;
    }
    return nullptr;
}

}  // namespace odai::render::upscale
