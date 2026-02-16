#include <gtest/gtest.h>

#include <array>

#include "render/backend/vulkan/frame_math.h"

namespace {

void ExpectNear(const voxelsprout::math::Vector3& lhs, const voxelsprout::math::Vector3& rhs) {
    EXPECT_NEAR(lhs.x, rhs.x, 1e-5f);
    EXPECT_NEAR(lhs.y, rhs.y, 1e-5f);
    EXPECT_NEAR(lhs.z, rhs.z, 1e-5f);
}

} // namespace

TEST(FrameMathTest, ComputeCameraFrameCalculatesForwardAndChunkIndices) {
    const voxelsprout::render::CameraPose camera{
        .x = 32.5f,
        .y = -15.2f,
        .z = 15.9f,
        .yawDegrees = 90.0f,
        .pitchDegrees = 0.0f,
        .fovDegrees = 70.0f
    };

    const voxelsprout::render::CameraFrameDerived frame = voxelsprout::render::computeCameraFrame(camera);
    ExpectNear(frame.forward, voxelsprout::math::Vector3{0.0f, 0.0f, 1.0f});
    EXPECT_EQ(frame.chunkX, 1);
    EXPECT_EQ(frame.chunkY, -1);
    EXPECT_EQ(frame.chunkZ, 0);
}

TEST(FrameMathTest, ComputeVoxelGiAxisOriginAndVerticalStability) {
    constexpr float halfSpan = 32.0f;
    constexpr float cellSize = 1.0f;

    EXPECT_FLOAT_EQ(voxelsprout::render::computeVoxelGiAxisOrigin(10.9f, halfSpan, cellSize), -22.0f);
    EXPECT_FLOAT_EQ(voxelsprout::render::computeVoxelGiAxisOrigin(-10.1f, halfSpan, cellSize), -43.0f);

    EXPECT_FLOAT_EQ(
        voxelsprout::render::computeVoxelGiStableOriginY(100.0f, 99.0f, true, 2.0f),
        99.0f
    );
    EXPECT_FLOAT_EQ(
        voxelsprout::render::computeVoxelGiStableOriginY(100.0f, 97.0f, true, 2.0f),
        100.0f
    );
    EXPECT_FLOAT_EQ(
        voxelsprout::render::computeVoxelGiStableOriginY(100.0f, 0.0f, false, 2.0f),
        100.0f
    );
}

TEST(FrameMathTest, ComputeSunDirectionUsesYawPitch) {
    const voxelsprout::math::Vector3 dir = voxelsprout::render::computeSunDirection(-90.0f, 30.0f);
    ExpectNear(dir, voxelsprout::math::Vector3{0.0f, 0.5f, -0.8660254f});
}

TEST(FrameMathTest, ComputeVoxelGiFlagsDetectsChanges) {
    const std::array<voxelsprout::math::Vector3, 9> sh = {
        voxelsprout::math::Vector3{0.1f, 0.2f, 0.3f},
        voxelsprout::math::Vector3{0.2f, 0.1f, 0.4f},
        voxelsprout::math::Vector3{0.3f, 0.0f, 0.1f},
        voxelsprout::math::Vector3{0.4f, 0.3f, 0.2f},
        voxelsprout::math::Vector3{0.5f, 0.1f, 0.0f},
        voxelsprout::math::Vector3{0.6f, 0.2f, 0.1f},
        voxelsprout::math::Vector3{0.7f, 0.3f, 0.2f},
        voxelsprout::math::Vector3{0.8f, 0.4f, 0.3f},
        voxelsprout::math::Vector3{0.9f, 0.5f, 0.4f},
    };
    const std::array<std::array<float, 3>, 9> previousSh = {
        std::array<float, 3>{0.1f, 0.2f, 0.3f},
        std::array<float, 3>{0.2f, 0.1f, 0.4f},
        std::array<float, 3>{0.3f, 0.0f, 0.1f},
        std::array<float, 3>{0.4f, 0.3f, 0.2f},
        std::array<float, 3>{0.5f, 0.1f, 0.0f},
        std::array<float, 3>{0.6f, 0.2f, 0.1f},
        std::array<float, 3>{0.7f, 0.3f, 0.2f},
        std::array<float, 3>{0.8f, 0.4f, 0.3f},
        std::array<float, 3>{0.9f, 0.5f, 0.4f},
    };

    const voxelsprout::render::VoxelGiComputeFlags first = voxelsprout::render::computeVoxelGiFlags(
        sh,
        previousSh,
        {1.0f, 2.0f, 3.0f},
        {1.0f, 2.0f, 3.0f},
        true,
        false,
        true,
        voxelsprout::math::Vector3{1.0f, 2.0f, 3.0f},
        voxelsprout::math::Vector3{1.0f, 2.0f, 3.0f},
        voxelsprout::math::Vector3{0.1f, 0.2f, 0.3f},
        voxelsprout::math::Vector3{0.1f, 0.2f, 0.3f},
        1.0f,
        1.0f,
        0.5f,
        0.5f,
        0.001f,
        0.001f,
        0.001f
    );
    EXPECT_FALSE(first.gridMoved);
    EXPECT_FALSE(first.lightingChanged);
    EXPECT_FALSE(first.needsOccupancyUpload);
    EXPECT_FALSE(first.needsComputeUpdate);

    const voxelsprout::render::VoxelGiComputeFlags second = voxelsprout::render::computeVoxelGiFlags(
        sh,
        previousSh,
        {1.0f, 2.0f, 3.0f},
        {1.0f, 5.0f, 3.0f},
        true,
        true,
        true,
        voxelsprout::math::Vector3{2.0f, 2.0f, 3.0f},
        voxelsprout::math::Vector3{1.0f, 2.0f, 3.0f},
        voxelsprout::math::Vector3{0.1f, 0.2f, 0.3f},
        voxelsprout::math::Vector3{0.1f, 0.2f, 0.3f},
        1.0f,
        1.0f,
        0.5f,
        0.5f,
        0.001f,
        0.001f,
        0.001f
    );
    EXPECT_TRUE(second.gridMoved);
    EXPECT_TRUE(second.needsOccupancyUpload);
    EXPECT_TRUE(second.needsComputeUpdate);
    EXPECT_TRUE(second.lightingChanged);
}
