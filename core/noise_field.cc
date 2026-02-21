#include "core/noise_field.h"

#include <noise/module/perlin.h>
#include <noise/module/voronoi.h>

#include <algorithm>

namespace voxelsprout::core {
namespace {

float clamp01(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

} // namespace

NoiseSamples sampleLibNoise(double x, double y, double z) {
    noise::module::Perlin perlin;
    perlin.SetSeed(1337);
    perlin.SetFrequency(0.08);
    perlin.SetLacunarity(2.0);
    perlin.SetPersistence(0.5);
    perlin.SetOctaveCount(5);

    noise::module::Voronoi voronoi;
    voronoi.SetSeed(4242);
    voronoi.SetFrequency(0.08);
    voronoi.SetDisplacement(0.0);
    voronoi.EnableDistance(true);

    const float perlinValue = static_cast<float>(perlin.GetValue(x, y, z));
    const float worleyDistance = static_cast<float>(voronoi.GetValue(x, y, z));

    NoiseSamples result{};
    result.perlin = clamp01(0.5f + (0.5f * perlinValue));
    result.worley = clamp01(worleyDistance);
    return result;
}

} // namespace voxelsprout::core
