#pragma once

#include "world/chunk_grid.h"

#include <vector>

// Deterministic grass/flower billboard scatter over a chunk's grass voxels.
// Pure CPU, no Vulkan: the renderer copies GrassInstance verbatim into its
// per-instance vertex buffer (layouts are static_asserted at the copy point).
namespace odai::world {

struct GrassInstance {
    float worldPosYaw[4];
    float colorTint[4];
};

struct GrassScatterParams {
    // Center of the resident chunk window; grass only scatters near it.
    int residentCenterChunkX = 0;
    int residentCenterChunkZ = 0;
    // Chebyshev chunk radius for activating grass, and the wider radius that
    // keeps already-grown grass alive (hysteresis against window jitter).
    int activeRadius = 1;
    int retainedRadius = 2;
    // Whether this chunk had grass instances before this rebuild.
    bool previouslyActive = false;
};

// Returns the instances for one chunk; empty when the chunk is outside the
// active radius. Deterministic: same chunk contents + params -> same output.
std::vector<GrassInstance> buildGrassInstances(const Chunk& chunk, const GrassScatterParams& params);

} // namespace odai::world
