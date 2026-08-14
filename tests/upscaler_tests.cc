// Upscaler backend selection.
//
// Worth testing precisely because the interesting behaviour is what happens when
// a backend is NOT available, and that path is invisible on the machine that
// builds it: a developer with the XeSS SDK installed never exercises the
// fallback, and a developer without it never exercises the success case. Both
// are pinned here against the compile-time flag rather than against whatever
// this particular machine happens to have.
//
// Pure CPU; no Vulkan, matching every other test in this project.

#include <algorithm>
#include <cmath>
#include "render/upscale/upscale_contract.h"
#include "render/upscale/upscale_policy.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string_view>

namespace {

using odai::render::UpscalerBackend;
using odai::render::UpscalerQuality;
using odai::render::UpscalerSettings;
using odai::render::UpscalerStatus;

bool nearlyEqual(float a, float b) { return std::fabs(a - b) < 1e-5f; }

// The engine's own path has no SDK and must always be selectable -- it is the
// fallback every other backend resolves to, so if it could ever be unavailable
// there would be nothing to fall back TO.
void testTemporalIsAlwaysAvailable() {
    assert(odai::render::upscalerBackendCompiledIn(UpscalerBackend::Temporal));
    assert(odai::render::upscalerBackendCompiledIn(UpscalerBackend::Off));

    UpscalerSettings settings{};
    settings.backend = UpscalerBackend::Temporal;
    const UpscalerStatus status = odai::render::resolveUpscaler(settings, false);
    assert(status.active == UpscalerBackend::Temporal);
    assert(status.requested == UpscalerBackend::Temporal);
    assert(status.runtimeAvailable);
}

// FSR and DLSS are declared in the enum but have no implementation. Asking for
// one must land on Temporal with a reason that says so, not silently render at
// the requested backend's name.
void testUnimplementedBackendsFallBack() {
    for (const UpscalerBackend backend : {UpscalerBackend::Fsr, UpscalerBackend::Dlss}) {
        assert(!odai::render::upscalerBackendCompiledIn(backend));
        UpscalerSettings settings{};
        settings.backend = backend;
        const UpscalerStatus status = odai::render::resolveUpscaler(settings, false);
        assert(status.requested == backend);
        assert(status.active == UpscalerBackend::Temporal);
        assert(!status.compiledIn);
        assert(std::string_view(status.reason).find("not implemented") != std::string_view::npos);
    }
}

// XeSS behaves differently depending on how this binary was built, and BOTH
// branches are asserted so the test is meaningful either way.
void testXessFollowsBuildConfiguration() {
    UpscalerSettings settings{};
    settings.backend = UpscalerBackend::Xess;

#if defined(ODAI_HAS_XESS)
    assert(odai::render::upscalerBackendCompiledIn(UpscalerBackend::Xess));
    // Compiled in and the device accepted it: XeSS runs.
    const UpscalerStatus supported = odai::render::resolveUpscaler(settings, true);
    assert(supported.active == UpscalerBackend::Xess);
    // Compiled in but the device refused: a DIFFERENT reason from "not built",
    // because the fix is a driver rather than a rebuild.
    const UpscalerStatus unsupported = odai::render::resolveUpscaler(settings, false);
    assert(unsupported.active == UpscalerBackend::Temporal);
    assert(std::string_view(unsupported.reason).find("runtime") != std::string_view::npos);
#else
    assert(!odai::render::upscalerBackendCompiledIn(UpscalerBackend::Xess));
    // Not built in: the runtime flag must not matter. Claiming device support
    // for a backend that was never compiled must still fall back, or a caller
    // probing capabilities could select a path that does not exist.
    const UpscalerStatus status = odai::render::resolveUpscaler(settings, true);
    assert(status.active == UpscalerBackend::Temporal);
    assert(!status.compiledIn);
    assert(std::string_view(status.reason).find("ODAI_ENABLE_XESS") != std::string_view::npos);
#endif
}

// The presets are XeSS's published ratios, and they have to stay that way: a
// preset is only comparable across backends if it means the same internal
// resolution on each.
void testQualityPresetsDriveRenderScale() {
    assert(nearlyEqual(upscalerQualityScale(UpscalerQuality::Quality), 1.0f / 1.5f));
    assert(nearlyEqual(upscalerQualityScale(UpscalerQuality::Performance), 0.5f));
    assert(nearlyEqual(upscalerQualityScale(UpscalerQuality::UltraPerformance), 1.0f / 3.0f));

    UpscalerSettings settings{};
    settings.backend = UpscalerBackend::Temporal;
    settings.quality = UpscalerQuality::Performance;
    assert(nearlyEqual(odai::render::resolveUpscaler(settings, false).renderScale, 0.5f));

    // Off means native. Honouring the preset there would render at a fraction
    // and then not reconstruct -- a blurry frame with no upscaler to blame.
    settings.backend = UpscalerBackend::Off;
    const UpscalerStatus offStatus = odai::render::resolveUpscaler(settings, false);
    assert(offStatus.active == UpscalerBackend::Off);
    assert(nearlyEqual(offStatus.renderScale, 1.0f));
}

// An unrecognised name must be reported, never silently defaulted: a typo'd
// backend that quietly becomes something else is indistinguishable from the
// requested one having been unavailable.
void testParsingRejectsUnknownNames() {
    UpscalerBackend backend = UpscalerBackend::Off;
    assert(odai::render::parseUpscalerBackend("xess", backend));
    assert(backend == UpscalerBackend::Xess);
    assert(odai::render::parseUpscalerBackend("native", backend));
    assert(backend == UpscalerBackend::Off);
    assert(!odai::render::parseUpscalerBackend("xes", backend));
    assert(!odai::render::parseUpscalerBackend("", backend));
    // A rejected parse must leave the value untouched.
    assert(backend == UpscalerBackend::Off);

    UpscalerQuality quality = UpscalerQuality::Quality;
    assert(odai::render::parseUpscalerQuality("balanced", quality));
    assert(quality == UpscalerQuality::Balanced);
    assert(!odai::render::parseUpscalerQuality("ultra", quality));
    assert(quality == UpscalerQuality::Balanced);
}

// The reconstruction contract. These are the numbers a host has to honour for
// ANY backend -- getting one wrong produces a reconstruction that never
// converges rather than anything that looks like a bug, so they are pinned.
void testJitterPhaseCountScalesWithRatio() {
    using namespace odai::render::upscale;
    // Native: the samples only have to cover one pixel, so the base sequence.
    assert(jitterPhaseCount({1920u, 1080u}, {1920u, 1080u}) == 8u);
    // Quality (1.5x per axis) covers 2.25 output pixels per input pixel.
    assert(jitterPhaseCount({1280u, 720u}, {1920u, 1080u}) == 18u);
    // Performance (2x per axis) covers 4.
    assert(jitterPhaseCount({960u, 540u}, {1920u, 1080u}) == 32u);
    // Supersampling must not produce a sequence shorter than the base one.
    assert(jitterPhaseCount({3840u, 2160u}, {1920u, 1080u}) == 8u);
    // And the cap holds, or the history window needed outlives the history.
    assert(jitterPhaseCount({16u, 16u}, {1920u, 1080u}) == 128u);
}

void testJitterStraddlesThePixelCentre() {
    using namespace odai::render::upscale;
    // Phase is 1-based because Halton(0) is 0 in every base: a 0-based sequence
    // would spend its first frame not jittering at all.
    const JitterOffset first = jitterOffsetPixels(1u);
    assert(first.x != 0.0f || first.y != 0.0f);
    // Centred, so offsets fall either side of the pixel centre rather than all
    // in one corner -- and every one stays inside the pixel.
    float minX = 1.0f;
    float maxX = -1.0f;
    for (std::uint32_t phase = 1u; phase <= 64u; ++phase) {
        const JitterOffset offset = jitterOffsetPixels(phase);
        assert(offset.x >= -0.5f && offset.x < 0.5f);
        assert(offset.y >= -0.5f && offset.y < 0.5f);
        minX = std::min(minX, offset.x);
        maxX = std::max(maxX, offset.x);
    }
    assert(minX < 0.0f && maxX > 0.0f);
}

void testRenderExtentRoundsRatherThanTruncates() {
    using namespace odai::render::upscale;
    // 1920/1.5 is exact; the preset that exposes rounding is UltraQuality,
    // where truncating 1920/1.3 gives 1476 and the lost half-pixel shows up as
    // a permanent shimmer along one screen edge.
    assert(renderExtentFor({1920u, 1080u}, UpscalerQuality::Quality).width == 1280u);
    assert(renderExtentFor({1920u, 1080u}, UpscalerQuality::UltraQuality).width == 1477u);
    assert(renderExtentFor({1920u, 1080u}, UpscalerQuality::Performance).width == 960u);
    // Never degenerate, however small the target.
    assert(renderExtentFor({1u, 1u}, UpscalerQuality::UltraPerformance).width == 1u);
}

void testMipBiasMatchesThePublishedRule() {
    using namespace odai::render::upscale;
    // log2(render/display) - 1, which is what FSR2, XeSS and DLSS all publish.
    // Native still biases by -1: mip selection is computed at the render
    // extent either way, and the upscaler needs detail to reconstruct from.
    assert(std::abs(recommendedMipLodBias({1920u, 1080u}, {1920u, 1080u}) + 1.0f) < 1e-5f);
    assert(std::abs(recommendedMipLodBias({960u, 540u}, {1920u, 1080u}) + 2.0f) < 1e-5f);
}

}  // namespace

int main() {
    testTemporalIsAlwaysAvailable();
    testUnimplementedBackendsFallBack();
    testXessFollowsBuildConfiguration();
    testQualityPresetsDriveRenderScale();
    testParsingRejectsUnknownNames();
    testJitterPhaseCountScalesWithRatio();
    testJitterStraddlesThePixelCentre();
    testRenderExtentRoundsRatherThanTruncates();
    testMipBiasMatchesThePublishedRule();
    std::printf("upscaler tests passed\n");
    return 0;
}
