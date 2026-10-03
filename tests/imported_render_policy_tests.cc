#include "render/imported_gi.h"
#include "render/weather_wind_policy.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "render/backend/vulkan/frame_math.h"
#include "render/renderer_types.h"
#include "render/water_depth_policy.h"
#include "render/image_space_contrast.h"

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
    {
        using namespace odai::importer;
        ImportedSceneTexture t;
        t.width = t.height = 4;
        t.format = TextureFormat::BC1;
        t.rgba8 = {0, 248, 31, 0, 0xE4, 0, 0, 0}; // red, blue, both interpolants
        ImportedScenePackedVertex v{};
        v.textureIndex = 0;
        auto sample = [&](float u) { v.uv[0] = u; return sampleImportedGiAlbedo({t}, v); };
        expect(sample(0)[0] == 1 && sample(0)[2] == 0, "GI decodes BC1 red endpoint");
        expect(sample(.25f)[2] == 1 && sample(.25f)[0] == 0, "GI decodes BC1 blue endpoint");
        expect(sample(.5f)[0] > .39f && sample(.5f)[0] < .41f,
            "GI decodes interpolated BC1 in linear light");
        expect(sample(-.75f) == sample(.25f), "GI wraps negative texture coordinates");
        for (auto format : {TextureFormat::BC2, TextureFormat::BC3}) {
            t.format = format;
            t.rgba8 = {0,0,0,0,0,0,0,0, 0,248,31,0, 0xE4,0,0,0};
            expect(sample(.25f)[2] == 1, "GI skips BC2/BC3 alpha block");
        }
        t.format = TextureFormat::RGBA8;
        t.width = t.height = 1;
        t.rgba8 = {128,64,32,255};
        expect(std::abs(sample(0)[0] - 128.f/255) < 1e-6f, "GI keeps linear RGBA linear");
        t.format = TextureFormat::RGBA8Srgb;
        expect(sample(0)[0] > .21f && sample(0)[0] < .22f, "GI decodes sRGB RGBA");
        t.rgba8.clear();
        expect(sample(0)[0] == 1, "truncated texture falls back safely");
        v.flags = kImportedSceneMaterialFlagVertexColorTint;
        v.color[0] = .25f;
        expect(sample(0)[0] == .25f, "GI honors authored vertex tint");
        const float a[3] = {0,0,0}, b[3] = {1024,0,0}, c[3] = {0,1024,0};
        expect(importedGiTriangleOverlapsCell(a,b,c,{256,256,0},16), "large wall remains solid");
        expect(!importedGiTriangleOverlapsCell(a,b,c,{900,900,0},16), "triangle AABB does not fill empty space");
        expect(!importedGiTriangleOverlapsCell(a,b,c,{256,256,33},16), "wall does not expand into adjacent air");
        expect(importedGiTriangleOverlapsCell(a,b,c,{256,256,16},16), "boundary contact conservatively blocks light");
        expect(computeVoxelGiStableOriginY(224,0,true,256) == 0, "GI anchor suppresses small camera motion");
        expect(computeVoxelGiStableOriginY(256,0,true,256) == 256, "GI anchor follows a moving camera");
        const std::array<odai::math::Vector3,9> sh{};
        const std::array<std::array<float,3>,9> oldSh{};
        const auto flags = [&](bool dirty, bool initialized, float origin, float light) {
            return computeVoxelGiFlags(sh,oldSh,{origin,0,0},{0,0,0},true,dirty,initialized,
                {0,-1,0},{0,-1,0},{light,1,1},{1,1,1},1,1,.5f,.5f,.001f,.001f,.001f);
        };
        expect(!flags(false,true,0,1).needsComputeUpdate, "stationary GI skips dispatch");
        expect(flags(true,true,0,1).needsOccupancyUpload, "residency invalidates GI occupancy");
        expect(flags(false,true,256,1).needsOccupancyUpload, "volume motion invalidates GI occupancy");
        expect(flags(false,true,0,2).needsComputeUpdate && !flags(false,true,0,2).needsOccupancyUpload,
            "lighting updates GI without voxelizing geometry again");
        expect(flags(false,false,0,1).needsOccupancyUpload, "failed occupancy upload retries");
    }
    const auto north = sampleAuthoredWeatherWind(0, 0, 0.5f, 4);
    const auto east = sampleAuthoredWeatherWind(90, 0, 1, 4);
    expect(std::abs(north[0]) < 1e-6f && north[1] == -1 && north[2] == 0.5f, "Authored north wind maps to engine -Z");
    expect(std::abs(east[0] - 1) < 1e-6f && std::abs(east[1]) < 1e-6f, "Authored east wind maps to engine +X");
    for (int t = 0; t < 1000; ++t) {
        const auto gust = sampleAuthoredWeatherWind(0, 30, 0, t);
        expect(std::abs(std::atan2(gust[0], -gust[1])) <= 0.524f && gust[2] == 0, "Gusts stay inside the authored range and calm stays calm");
    }
    CameraPose viewCamera{};
    viewCamera.yawDegrees = -37.0f;
    viewCamera.pitchDegrees = -2.0f;
    const auto originView = computeCameraView(viewCamera);
    for (const float coordinate : {13980.0f, 48800.0f, -61440.0f, 180000.0f}) {
        viewCamera.x = coordinate;
        viewCamera.y = 800.0f;
        viewCamera.z = coordinate + 0.125f;
        const auto movedView = computeCameraView(viewCamera);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                expect(movedView(row, column) == originView(row, column),
                    "world translation cannot perturb camera orientation");
            }
            expect(std::fabs(movedView(row, 0) * viewCamera.x +
                movedView(row, 1) * viewCamera.y + movedView(row, 2) * viewCamera.z +
                movedView(row, 3)) < 0.01f, "camera eye maps to view origin");
        }
    }
    expect(celestialNightVisibility(30) == 0, "white WTHR stars cannot appear in daylight");
    expect(celestialNightVisibility(0) == 0, "celestial fade begins below horizon");
    expect(celestialNightVisibility(-3) == 0.5f, "celestial twilight fade is continuous");
    expect(celestialNightVisibility(-6) == 1, "authored night stars retain full intensity");
    expect(authoredNightAmbientWeight(false, 1, -30) == 0, "other games retain night policy");
    expect(authoredNightAmbientWeight(true, 0, -30) == 0, "weather override disables authored fill");
    expect(authoredNightAmbientWeight(true, 1, 6) == 0, "daytime lighting unchanged");
    expect(authoredNightAmbientWeight(true, 1, -30) == 1, "night uses authored ambient once");
    expect(authoredNightAmbientWeight(true, 1, 3) == 0.5f, "twilight blends smoothly");
    for (float contrast : {0.5f, 1.0f, 1.45f, 4.0f}) {
        expect(imageSpaceContrast(0, contrast) == 0 && imageSpaceContrast(1, contrast) == 1,
               "contrast preserves black and white endpoints");
        expect(imageSpaceContrast(0.5f, contrast) == 0.5f, "contrast preserves pivot");
        float previous = 0;
        for (int i = 1; i < 1000; ++i) {
            const float x = i / 1000.0f;
            const float y = imageSpaceContrast(x, contrast);
            expect(y > previous && y < 1, "contrast retains distinct dark and bright samples");
            if (contrast == 1) expect(std::fabs(y - x) < 1e-6f, "neutral contrast is identity");
            previous = y;
        }
    }


    expect(waterShoreCoverage(1000, 0) == 1, "missing bottom retains deep water");
    expect(waterShoreCoverage(1000, 900) == 0, "foreground bank remains dry");
    expect(waterShoreCoverage(1000, 1000) == 0, "contact has no opaque water seam");
    expect(std::fabs(waterShoreCoverage(1000, 1001) - 0.5f) < 1e-6f,
           "near shoreline blends continuously");
    expect(waterShoreCoverage(1000, 1100) == 1, "submerged bottom retains full water");
    expect(std::fabs(waterShoreCoverage(10000, 10005) - 0.5f) < 1e-6f,
           "distant shore tolerance follows FP16 precision");

    auto projection = odai::math::Matrix4::identity();
    projection(2, 2) = 0; projection(2, 3) = 1;
    projection(3, 2) = -1; projection(3, 3) = 0;
    const auto visible = [&](float x0, float y0, float z0,
                             float x1, float y1, float z1) {
        const float low[] = {x0, y0, z0}, high[] = {x1, y1, z1};
        return importedBoundsIntersectClip(low, high, projection, 0.0f);
    };
    expect(visible(-1, -1, -10, 1, 1, -2), "front box visible in reverse-Z frustum");
    expect(!visible(-1, -1, 2, 1, 1, 10), "entire behind-camera box rejected");
    expect(visible(-1, -1, -2, 1, 1, 2), "eye-crossing box conservatively retained");
    expect(!visible(20, -1, -10, 30, 1, -2), "side-plane box rejected");
    expect(!visible(-0.1f, -0.1f, -0.5f, 0.1f, 0.1f, -0.2f), "near-plane box rejected");
    expect(visible(2, 0, 0, 1, 1, 1), "invalid bounds retained");
    expect(visible(std::numeric_limits<float>::quiet_NaN(), 0, 0, 1, 1, 1),
           "non-finite bounds retained");
    projection = odai::math::Matrix4::identity();
    expect(visible(-0.5f, -0.5f, 0.1f, 0.5f, 0.5f, 0.9f), "orthographic box retained");
    expect(!visible(-0.5f, -0.5f, 2, 0.5f, 0.5f, 3), "orthographic far box rejected");

    ImportedInteriorLighting exterior{};
    expect(shouldRenderImportedDirectionalShadows(exterior), "exterior uses directional shadows");
    expect(shouldRenderImportedSky(exterior), "exterior renders sky");
    expect(shouldUseImportedSkyLighting(exterior), "exterior uses sky lighting");
    expect(!shouldUseImportedScreenSpaceGi(exterior), "exterior does not force interior SSGI");
    ImportedExteriorLighting outdoor{};
    outdoor.screenSpaceGi = true;

    constexpr ImportedExteriorLighting morrowind = morrowindExteriorLighting();
    expect(morrowind.screenSpaceGi && morrowind.worldSpaceGi, "Morrowind exterior enables offscreen diffuse bounce");
    expect(morrowind.bounceStrength > 0.0f && morrowind.bounceStrength <= 1.0f,
           "Morrowind exterior bounce stays bounded");

    constexpr ImportedExteriorLighting skyrim = skyrimSeExteriorLighting();
    expect(skyrim.screenSpaceGi, "Skyrim exterior enables diffuse bounce");
    expect(skyrim.diffuseWrap < 0.15f && skyrim.ambientScale < 0.7f,
           "Skyrim exterior preserves directional relief");
    expect(skyrim.sunlightScale > 1.0f && skyrim.bounceStrength < 0.4f,
           "Skyrim exterior keeps sunlight dominant over bounce");
    expect(shouldUseImportedScreenSpaceGi(exterior, outdoor), "exterior explicitly enables diffuse bounce");
    expect(shouldRenderImportedDirectionalShadows(exterior) && shouldRenderImportedSky(exterior),
           "outdoor bounce retains sun shadows and sky");
    outdoor.screenSpaceGi = false;
    expect(!shouldUseImportedScreenSpaceGi(exterior, outdoor), "exterior GI can be disabled independently");
    outdoor.screenSpaceGi = true;

    ImportedInteriorLighting interior{};
    interior.enabled = true;
    interior.hasAuthoredLighting = true;
    expect(!shouldUseImportedScreenSpaceGi(interior, outdoor), "exterior GI cannot override interior policy");
    expect(!shouldRenderImportedDirectionalShadows(interior), "authored interior suppresses sun shadows");
    expect(!shouldRenderImportedSky(interior), "authored interior suppresses sky by default");
    interior.indirectLightingMode = ImportedInteriorLighting::IndirectLightingMode::ScreenSpaceDiffuse;
    expect(shouldUseImportedScreenSpaceGi(interior), "authored interior may request SSGI");
    interior.hasAuthoredLighting = false;
    expect(!shouldUseImportedScreenSpaceGi(interior, outdoor), "legacy interior does not inherit exterior GI");
    interior.hasAuthoredLighting = true;
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
