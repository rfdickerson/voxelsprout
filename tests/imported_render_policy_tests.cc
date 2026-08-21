#include <cmath>
#include <cstdlib>
#include <iostream>

#include "render/backend/vulkan/frame_math.h"
#include "render/renderer_types.h"

namespace {

void expect(bool value, const char* message) {
    if (!value) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    using namespace odai::render;

    ImportedInteriorLighting exterior{};
    expect(shouldRenderImportedDirectionalShadows(exterior), "exterior uses directional shadows");
    expect(shouldRenderImportedSky(exterior), "exterior renders sky");
    expect(shouldUseImportedSkyLighting(exterior), "exterior uses sky lighting");
    expect(!shouldUseImportedScreenSpaceGi(exterior), "exterior does not force interior SSGI");

    ImportedInteriorLighting interior{};
    interior.enabled = true;
    interior.hasAuthoredLighting = true;
    expect(!shouldRenderImportedDirectionalShadows(interior), "authored interior suppresses sun shadows");
    expect(!shouldRenderImportedSky(interior), "authored interior suppresses sky by default");
    interior.indirectLightingMode = ImportedInteriorLighting::IndirectLightingMode::ScreenSpaceDiffuse;
    expect(shouldUseImportedScreenSpaceGi(interior), "authored interior may request SSGI");
    interior.localShadowMode = ImportedInteriorLighting::LocalShadowMode::ShadowMapsWithContact;
    expect(shouldUseImportedPointShadowMaps(interior), "interior shadow maps are selected");
    expect(shouldUseImportedContactShadows(interior), "interior contact shadows are selected");

    expect(screenSpaceGiQuarterExtent(5) == 2, "SSGI quarter extent rounds up");
    expect(screenSpaceGiHistorySampleAccepted(1000.0f, 1010.0f, 0.8f),
           "SSGI accepts stable history");
    expect(!screenSpaceGiHistorySampleAccepted(900.0f, 1010.0f, 0.8f),
           "SSGI rejects disocclusion");
    expect(std::fabs(screenSpaceGiClampedLuminance(4.0f, 1.0f) - 0.3675f) < 1e-5f,
           "SSGI energy clamp stays receiver-relative");

    std::cout << "imported render policy tests passed\n";
    return 0;
}
