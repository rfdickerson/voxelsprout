#include <gtest/gtest.h>

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

    const voxelsprout::render::CameraFrameDerived frame = voxelsprout::render::ComputeCameraFrame(camera);
    ExpectNear(frame.forward, voxelsprout::math::Vector3{0.0f, 0.0f, 1.0f});
    EXPECT_EQ(frame.chunkX, 1);
    EXPECT_EQ(frame.chunkY, -1);
    EXPECT_EQ(frame.chunkZ, 0);
}

TEST(FrameMathTest, ComputeVoxelGiAxisOriginAndVerticalStability) {
    constexpr float halfSpan = 32.0f;
    constexpr float cellSize = 1.0f;

    EXPECT_FLOAT_EQ(voxelsprout::render::ComputeVoxelGiAxisOrigin(10.9f, halfSpan, cellSize), -22.0f);
    EXPECT_FLOAT_EQ(voxelsprout::render::ComputeVoxelGiAxisOrigin(-10.1f, halfSpan, cellSize), -43.0f);

    EXPECT_FLOAT_EQ(
        voxelsprout::render::ComputeVoxelGiStableOriginY(100.0f, 99.0f, true, 2.0f),
        99.0f
    );
    EXPECT_FLOAT_EQ(
        voxelsprout::render::ComputeVoxelGiStableOriginY(100.0f, 97.0f, true, 2.0f),
        100.0f
    );
    EXPECT_FLOAT_EQ(
        voxelsprout::render::ComputeVoxelGiStableOriginY(100.0f, 0.0f, false, 2.0f),
        100.0f
    );
}

TEST(FrameMathTest, ComputeSunDirectionUsesYawPitch) {
    const voxelsprout::math::Vector3 dir = voxelsprout::render::ComputeSunDirection(-90.0f, 30.0f);
    ExpectNear(dir, voxelsprout::math::Vector3{0.0f, 0.5f, -0.8660254f});
}
