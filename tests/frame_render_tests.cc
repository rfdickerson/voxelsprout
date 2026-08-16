#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "render/backend/vulkan/shadow_culling_utils.h"
#include "render/backend/vulkan/frame_math.h"
#include "render/renderer_types.h"
#include "world/chunk.h"

namespace {

TEST(FrameRenderTest, BuildShadowCandidateMaskReturnsEmptyWhenDisabled) {
    const std::vector<odai::world::Chunk> chunks = {
        odai::world::Chunk(0, 0, 0),
        odai::world::Chunk(1, 0, 0),
        odai::world::Chunk(0, 1, 0)
    };

    const std::vector<std::size_t> visibleChunkIndices = {0u};
    const std::vector<std::uint8_t> candidates =
        odai::render::buildShadowCandidateMask(chunks, visibleChunkIndices, false);

    EXPECT_TRUE(candidates.empty());
}

TEST(FrameRenderTest, BuildShadowCandidateMaskMarksNeighborChunks) {
    const std::vector<odai::world::Chunk> chunks = {
        odai::world::Chunk(0, 0, 0),
        odai::world::Chunk(1, 0, 0),
        odai::world::Chunk(2, 0, 0),
        odai::world::Chunk(0, 1, 0)
    };

    const std::vector<std::size_t> visibleChunkIndices = {0u, 3u};
    const std::vector<std::uint8_t> candidates =
        odai::render::buildShadowCandidateMask(chunks, visibleChunkIndices, true);

    ASSERT_EQ(candidates.size(), 4u);
    EXPECT_EQ(candidates[0u], 1u);
    EXPECT_EQ(candidates[1u], 1u);
    EXPECT_EQ(candidates[2u], 0u);
    EXPECT_EQ(candidates[3u], 1u);
}

TEST(FrameRenderTest, BuildShadowCandidateMaskSkipsInvalidVisibleIndices) {
    const std::vector<odai::world::Chunk> chunks = {
        odai::world::Chunk(0, 0, 0),
        odai::world::Chunk(0, 0, 1)
    };
    const std::vector<std::size_t> visibleChunkIndices = {0u, 99u};

    const std::vector<std::uint8_t> candidates =
        odai::render::buildShadowCandidateMask(chunks, visibleChunkIndices, true);

    ASSERT_EQ(candidates.size(), 2u);
    EXPECT_EQ(candidates[0u], 1u);
    EXPECT_EQ(candidates[1u], 1u);
}

TEST(FrameRenderTest, ImportedInteriorPolicyCoversExteriorAndOrdinaryInterior) {
    odai::render::ImportedInteriorLighting exterior{};
    EXPECT_TRUE(odai::render::shouldRenderImportedDirectionalShadows(exterior));
    EXPECT_TRUE(odai::render::shouldRenderImportedSky(exterior));
    EXPECT_TRUE(odai::render::shouldUseImportedSkyLighting(exterior));
    EXPECT_FALSE(odai::render::shouldUseImportedPointShadowMaps(exterior));
    EXPECT_FALSE(odai::render::shouldUseImportedContactShadows(exterior));
    EXPECT_FALSE(odai::render::shouldUseImportedRayTracedLocalShadows(exterior));
    EXPECT_FALSE(odai::render::shouldUseImportedScreenSpaceGi(exterior));

    odai::render::ImportedInteriorLighting interior{};
    interior.enabled = true;
    interior.hasAuthoredLighting = true;
    EXPECT_FALSE(odai::render::shouldRenderImportedDirectionalShadows(interior));
    EXPECT_FALSE(odai::render::shouldRenderImportedSky(interior));
    EXPECT_FALSE(odai::render::shouldUseImportedSkyLighting(interior));
    EXPECT_FALSE(odai::render::shouldUseImportedPointShadowMaps(interior));
    EXPECT_FALSE(odai::render::shouldUseImportedContactShadows(interior));
    EXPECT_FALSE(odai::render::shouldUseImportedRayTracedLocalShadows(interior));
    EXPECT_FALSE(odai::render::shouldUseImportedScreenSpaceGi(interior));

    interior.indirectLightingMode =
        odai::render::ImportedInteriorLighting::IndirectLightingMode::ScreenSpaceDiffuse;
    EXPECT_TRUE(odai::render::shouldUseImportedScreenSpaceGi(interior));

    interior.localShadowMode =
        odai::render::ImportedInteriorLighting::LocalShadowMode::ShadowMaps;
    EXPECT_TRUE(odai::render::shouldUseImportedPointShadowMaps(interior));
    EXPECT_FALSE(odai::render::shouldUseImportedContactShadows(interior));
    EXPECT_FALSE(odai::render::shouldUseImportedRayTracedLocalShadows(interior));

    interior.localShadowMode =
        odai::render::ImportedInteriorLighting::LocalShadowMode::ShadowMapsWithContact;
    EXPECT_TRUE(odai::render::shouldUseImportedPointShadowMaps(interior));
    EXPECT_TRUE(odai::render::shouldUseImportedContactShadows(interior));
    EXPECT_FALSE(odai::render::shouldUseImportedRayTracedLocalShadows(interior));

    interior.localShadowMode =
        odai::render::ImportedInteriorLighting::LocalShadowMode::RayTraced;
    EXPECT_FALSE(odai::render::shouldUseImportedPointShadowMaps(interior));
    EXPECT_FALSE(odai::render::shouldUseImportedContactShadows(interior));
    EXPECT_TRUE(odai::render::shouldUseImportedRayTracedLocalShadows(interior));
}

TEST(FrameRenderTest, ImportedInteriorPolicyKeepsSkyFlagsIndependent) {
    odai::render::ImportedInteriorLighting showSky{};
    showSky.enabled = true;
    showSky.hasAuthoredLighting = true;
    showSky.showSky = true;
    EXPECT_TRUE(odai::render::shouldRenderImportedSky(showSky));
    EXPECT_FALSE(odai::render::shouldUseImportedSkyLighting(showSky));
    EXPECT_FALSE(odai::render::shouldRenderImportedDirectionalShadows(showSky));

    odai::render::ImportedInteriorLighting useSkyLighting{};
    useSkyLighting.enabled = true;
    useSkyLighting.hasAuthoredLighting = true;
    useSkyLighting.useSkyLighting = true;
    EXPECT_FALSE(odai::render::shouldRenderImportedSky(useSkyLighting));
    EXPECT_TRUE(odai::render::shouldUseImportedSkyLighting(useSkyLighting));
    EXPECT_FALSE(odai::render::shouldRenderImportedDirectionalShadows(useSkyLighting));
}

TEST(FrameRenderTest, LegacyInteriorModeRetainsCompatibilityPolicy) {
    odai::render::ImportedInteriorLighting legacyInterior{};
    legacyInterior.enabled = true;
    EXPECT_TRUE(odai::render::shouldRenderImportedDirectionalShadows(legacyInterior));
    EXPECT_TRUE(odai::render::shouldRenderImportedSky(legacyInterior));
    EXPECT_TRUE(odai::render::shouldUseImportedSkyLighting(legacyInterior));
    legacyInterior.localShadowMode =
        odai::render::ImportedInteriorLighting::LocalShadowMode::ShadowMaps;
    EXPECT_FALSE(odai::render::shouldUseImportedPointShadowMaps(legacyInterior));
    legacyInterior.localShadowMode =
        odai::render::ImportedInteriorLighting::LocalShadowMode::ShadowMapsWithContact;
    EXPECT_FALSE(odai::render::shouldUseImportedPointShadowMaps(legacyInterior));
    EXPECT_FALSE(odai::render::shouldUseImportedContactShadows(legacyInterior));
    legacyInterior.indirectLightingMode =
        odai::render::ImportedInteriorLighting::IndirectLightingMode::ScreenSpaceDiffuse;
    EXPECT_FALSE(odai::render::shouldUseImportedScreenSpaceGi(legacyInterior));
}

TEST(FrameRenderTest, ScreenSpaceGiQuarterExtentRoundsUp) {
    EXPECT_EQ(odai::render::screenSpaceGiQuarterExtent(2400u), 600u);
    EXPECT_EQ(odai::render::screenSpaceGiQuarterExtent(1500u), 375u);
    EXPECT_EQ(odai::render::screenSpaceGiQuarterExtent(1u), 1u);
    EXPECT_EQ(odai::render::screenSpaceGiQuarterExtent(5u), 2u);
}

TEST(FrameRenderTest, ScreenSpaceGiHistoryRejectsDisocclusionAndNormalMismatch) {
    EXPECT_TRUE(odai::render::screenSpaceGiHistorySampleAccepted(1000.0f, 1010.0f, 0.8f));
    EXPECT_FALSE(odai::render::screenSpaceGiHistorySampleAccepted(900.0f, 1010.0f, 0.8f));
    EXPECT_FALSE(odai::render::screenSpaceGiHistorySampleAccepted(1000.0f, 1010.0f, 0.49f));
    EXPECT_FALSE(odai::render::screenSpaceGiHistorySampleAccepted(0.0f, 1010.0f, 1.0f));
}

TEST(FrameRenderTest, ScreenSpaceGiEnergyClampStaysReceiverRelative) {
    EXPECT_FLOAT_EQ(odai::render::screenSpaceGiClampedLuminance(0.01f, 1.0f), 0.01f);
    EXPECT_FLOAT_EQ(odai::render::screenSpaceGiClampedLuminance(4.0f, 1.0f), 0.3675f);
    EXPECT_FLOAT_EQ(odai::render::screenSpaceGiClampedLuminance(4.0f, 0.0f), 0.0175f);
}

}  // namespace
