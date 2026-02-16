#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "render/backend/vulkan/shadow_culling_utils.h"
#include "world/chunk.h"

namespace {

TEST(FrameRenderTest, BuildShadowCandidateMaskReturnsEmptyWhenDisabled) {
    const std::vector<voxelsprout::world::Chunk> chunks = {
        voxelsprout::world::Chunk(0, 0, 0),
        voxelsprout::world::Chunk(1, 0, 0),
        voxelsprout::world::Chunk(0, 1, 0)
    };

    const std::vector<std::size_t> visibleChunkIndices = {0u};
    const std::vector<std::uint8_t> candidates =
        voxelsprout::render::buildShadowCandidateMask(chunks, visibleChunkIndices, false);

    EXPECT_TRUE(candidates.empty());
}

TEST(FrameRenderTest, BuildShadowCandidateMaskMarksNeighborChunks) {
    const std::vector<voxelsprout::world::Chunk> chunks = {
        voxelsprout::world::Chunk(0, 0, 0),
        voxelsprout::world::Chunk(1, 0, 0),
        voxelsprout::world::Chunk(2, 0, 0),
        voxelsprout::world::Chunk(0, 1, 0)
    };

    const std::vector<std::size_t> visibleChunkIndices = {0u, 3u};
    const std::vector<std::uint8_t> candidates =
        voxelsprout::render::buildShadowCandidateMask(chunks, visibleChunkIndices, true);

    ASSERT_EQ(candidates.size(), 4u);
    EXPECT_EQ(candidates[0u], 1u);
    EXPECT_EQ(candidates[1u], 1u);
    EXPECT_EQ(candidates[2u], 0u);
    EXPECT_EQ(candidates[3u], 1u);
}

TEST(FrameRenderTest, BuildShadowCandidateMaskSkipsInvalidVisibleIndices) {
    const std::vector<voxelsprout::world::Chunk> chunks = {
        voxelsprout::world::Chunk(0, 0, 0),
        voxelsprout::world::Chunk(0, 0, 1)
    };
    const std::vector<std::size_t> visibleChunkIndices = {0u, 99u};

    const std::vector<std::uint8_t> candidates =
        voxelsprout::render::buildShadowCandidateMask(chunks, visibleChunkIndices, true);

    ASSERT_EQ(candidates.size(), 2u);
    EXPECT_EQ(candidates[0u], 1u);
    EXPECT_EQ(candidates[1u], 1u);
}

}  // namespace
