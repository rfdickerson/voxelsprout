#pragma once

// The temporal-reconstruction contract: what a host has to do so that ANY
// upscaler -- this engine's own, XeSS, FSR, DLSS -- can reconstruct from its
// frames. Vulkan-free on purpose, so it can be unit-tested and so a host that
// is not this renderer can adopt it without pulling a device in.
//
// Every vendor SDK specifies these same four things, in the same terms, because
// they are properties of the ALGORITHM rather than of the implementation:
//
//   1. the render extent a quality preset asks for,
//   2. how many distinct sub-pixel jitter phases the sequence must have,
//   3. where in the pixel each phase samples,
//   4. how far the texture LOD has to be biased to compensate.
//
// Getting any of them wrong does not fail loudly. It produces a reconstruction
// that never converges, which reads as "the upscaler is soft" rather than as a
// bug in the host -- so they are stated here once, in one place, rather than
// left inline in whichever pass happened to need them.

#include "render/renderer_types.h"

#include <cstdint>

namespace odai::render::upscale {

// Render extent a preset asks for, rounded rather than truncated: truncating
// 1920/1.3 gives 1476 and rounding gives 1477, and the half-pixel that costs
// shows up as a persistent shimmer along one screen edge where the
// reconstruction has no sample.
struct Extent2D {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

[[nodiscard]] Extent2D renderExtentFor(Extent2D displayExtent, UpscalerQuality quality);

// THE SEQUENCE LENGTH SCALES WITH THE UPSCALE RATIO, and this is the single
// most commonly missed part of a temporal-upscale integration.
//
// Eight phases is right when input and output share a grid: the samples only
// have to cover one pixel. Upscaling, each input pixel covers ratio^2 output
// pixels, and a length-8 sequence spread over that area leaves most output
// pixels with no sample near them. Intel's XeSS guide states the same rule.
//
// Derived from the EXTENTS rather than from the quality preset, so a host that
// sets its render scale by hand -- as this engine's ODAI_RENDER_SCALE does --
// still gets a sequence long enough for the ratio it actually chose.
[[nodiscard]] std::uint32_t jitterPhaseCount(Extent2D renderExtent, Extent2D displayExtent);

// Radical inverse in `base`: Halton's definition. Low-discrepancy, so a short
// window of frames covers the pixel evenly instead of clumping the way white
// noise does.
[[nodiscard]] float haltonRadicalInverse(std::uint32_t index, std::uint32_t base);

// Sub-pixel offset for a phase, in PIXELS, centred on the pixel so the offsets
// straddle its centre rather than all falling in one corner. Phase is taken
// 1-based: Halton(0) is exactly 0 for every base, so a 0-based sequence spends
// its first frame not jittering at all.
struct JitterOffset {
    float x = 0.0f;
    float y = 0.0f;
};

[[nodiscard]] JitterOffset jitterOffsetPixels(std::uint32_t phase);

// The same offset expressed in NDC for the RENDER extent -- the grid the scene
// is rasterized on, not the display's. NDC spans 2 units across the extent, so
// one pixel is 2/extent.
[[nodiscard]] JitterOffset jitterOffsetNdc(std::uint32_t phase, Extent2D renderExtent);

// Texture LOD bias a host should add to its material samplers while upscaling.
//
// Mip selection is computed from screen-space derivatives at the RENDER extent,
// so rendering at half resolution picks a mip one level blurrier than the
// display resolution can show, and the upscaler then has no high-frequency
// detail to reconstruct FROM. log2(render/display) - 1 is the value FSR2, XeSS
// and DLSS all publish.
//
// This engine does not currently apply it -- its imported-material samplers are
// created once with no bias and the value would have to reach them at sampler
// creation, not per draw. Stated here because it is part of the contract a
// vendoring host has to satisfy, and because "the upscaled image is soft" is
// otherwise very hard to attribute.
[[nodiscard]] float recommendedMipLodBias(Extent2D renderExtent, Extent2D displayExtent);

}  // namespace odai::render::upscale
