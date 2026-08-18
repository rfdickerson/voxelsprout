// Backend registry, and the vendor backends that are declared but not built.
//
// XeSS is optional; temporal reconstruction is always available as fallback.

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
    }
    return nullptr;
}

}  // namespace odai::render::upscale
