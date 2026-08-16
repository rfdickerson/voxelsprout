#include "render/upscale/upscale_contract.h"

#include <algorithm>
#include <cmath>

namespace odai::render::upscale {

Extent2D renderExtentFor(Extent2D displayExtent, UpscalerQuality quality) {
    const float scale = upscalerQualityScale(quality);
    const auto apply = [scale](std::uint32_t value) {
        const float scaled = static_cast<float>(value) * scale;
        return std::max(1u, static_cast<std::uint32_t>(scaled + 0.5f));
    };
    return Extent2D{apply(displayExtent.width), apply(displayExtent.height)};
}

std::uint32_t jitterPhaseCount(Extent2D renderExtent, Extent2D displayExtent) {
    const float ratioX = static_cast<float>(displayExtent.width) /
                         static_cast<float>(std::max(1u, renderExtent.width));
    const float ratioY = static_cast<float>(displayExtent.height) /
                         static_cast<float>(std::max(1u, renderExtent.height));
    // Clamped at 1 so a host that renders LARGER than it displays (supersampling)
    // gets the 8-phase base sequence rather than a fractional one.
    const float ratioSquared = std::max(1.0f, ratioX * ratioY);
    // Capped at 128: past that the history window needed to resolve the sequence
    // is longer than anything survives without ghosting.
    return std::min(128u, static_cast<std::uint32_t>((8.0f * ratioSquared) + 0.5f));
}

float haltonRadicalInverse(std::uint32_t index, std::uint32_t base) {
    if (base < 2u) {
        return 0.0f;
    }
    float result = 0.0f;
    float invBase = 1.0f / static_cast<float>(base);
    float fraction = invBase;
    while (index > 0u) {
        result += static_cast<float>(index % base) * fraction;
        index /= base;
        fraction *= invBase;
    }
    return result;
}

JitterOffset jitterOffsetPixels(std::uint32_t phase) {
    JitterOffset offset{};
    offset.x = haltonRadicalInverse(phase, 2u) - 0.5f;
    offset.y = haltonRadicalInverse(phase, 3u) - 0.5f;
    return offset;
}

JitterOffset jitterOffsetNdc(std::uint32_t phase, Extent2D renderExtent) {
    const JitterOffset pixels = jitterOffsetPixels(phase);
    JitterOffset ndc{};
    ndc.x = (pixels.x * 2.0f) / static_cast<float>(std::max(1u, renderExtent.width));
    ndc.y = (pixels.y * 2.0f) / static_cast<float>(std::max(1u, renderExtent.height));
    return ndc;
}

float recommendedMipLodBias(Extent2D renderExtent, Extent2D displayExtent) {
    const float ratio = static_cast<float>(std::max(1u, renderExtent.width)) /
                        static_cast<float>(std::max(1u, displayExtent.width));
    return std::log2(ratio) - 1.0f;
}

}  // namespace odai::render::upscale
