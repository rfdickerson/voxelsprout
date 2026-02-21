#pragma once

namespace voxelsprout::core {

struct NoiseSamples {
    float perlin = 0.0f;
    float worley = 0.0f;
};

[[nodiscard]] NoiseSamples sampleLibNoise(double x, double y, double z);

} // namespace voxelsprout::core
