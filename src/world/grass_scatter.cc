#include "world/grass_scatter.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace odai::world {

std::vector<GrassInstance> buildGrassInstances(const Chunk& chunk, const GrassScatterParams& params) {
    std::vector<GrassInstance> grassInstances;
    const int grassDistanceX = std::abs(chunk.chunkX() - params.residentCenterChunkX);
    const int grassDistanceZ = std::abs(chunk.chunkZ() - params.residentCenterChunkZ);
    const int grassActiveRadius = params.previouslyActive ? params.retainedRadius : params.activeRadius;
    if (grassDistanceX > grassActiveRadius || grassDistanceZ > grassActiveRadius) {
        return grassInstances;
    }
    grassInstances.reserve(448);

    const float chunkWorldX = static_cast<float>(chunk.chunkX() * Chunk::kSizeX);
    const float chunkWorldY = static_cast<float>(chunk.chunkY() * Chunk::kSizeY);
    const float chunkWorldZ = static_cast<float>(chunk.chunkZ() * Chunk::kSizeZ);

    for (int y = 0; y < Chunk::kSizeY - 1; ++y) {
        for (int z = 0; z < Chunk::kSizeZ; ++z) {
            for (int x = 0; x < Chunk::kSizeX; ++x) {
                if (chunk.voxelAt(x, y, z).type != VoxelType::Grass) {
                    continue;
                }
                if (chunk.voxelAt(x, y + 1, z).type != VoxelType::Empty) {
                    continue;
                }

                const std::uint32_t hash =
                    static_cast<std::uint32_t>(x * 73856093) ^
                    static_cast<std::uint32_t>(y * 19349663) ^
                    static_cast<std::uint32_t>(z * 83492791) ^
                    static_cast<std::uint32_t>((chunk.chunkX() + 101) * 2654435761u) ^
                    static_cast<std::uint32_t>((chunk.chunkZ() + 193) * 2246822519u);
                // Keep grass sparse and deterministic so placement feels natural and stable.
                if ((hash % 100u) >= 22u) {
                    continue;
                }
                const int clumpCount = 2 + static_cast<int>((hash >> 24u) & 0x1u);
                for (int clumpIndex = 0; clumpIndex < clumpCount; ++clumpIndex) {
                    const std::uint32_t clumpHash = hash ^ (0x9E3779B9u * static_cast<std::uint32_t>(clumpIndex + 1));
                    const float rand0 = static_cast<float>(clumpHash & 0xFFu) / 255.0f;
                    const float rand1 = static_cast<float>((clumpHash >> 8u) & 0xFFu) / 255.0f;
                    const float rand2 = static_cast<float>((clumpHash >> 16u) & 0xFFu) / 255.0f;
                    const float rand3 = static_cast<float>((clumpHash >> 24u) & 0xFFu) / 255.0f;
                    const std::uint32_t tintHash = clumpHash ^ 0x85EBCA6Bu;
                    const float tintRand0 = static_cast<float>(tintHash & 0xFFu) / 255.0f;
                    const float tintRand1 = static_cast<float>((tintHash >> 8u) & 0xFFu) / 255.0f;
                    const float tintRand2 = static_cast<float>((tintHash >> 16u) & 0xFFu) / 255.0f;
                    const float radial = 0.06f + (0.18f * rand2);
                    const float angle = rand1 * (2.0f * 3.14159265f);
                    const float jitterX = std::cos(angle) * radial;
                    const float jitterZ = std::sin(angle) * radial;
                    const float yawRadians = rand0 * (2.0f * 3.14159265f);
                    const float yJitter = rand3 * 0.08f;

                    GrassInstance instance{};
                    instance.worldPosYaw[0] = chunkWorldX + static_cast<float>(x) + 0.5f + jitterX;
                    // Lift slightly above the supporting voxel top to avoid depth tie flicker.
                    instance.worldPosYaw[1] = chunkWorldY + static_cast<float>(y) + 1.02f + yJitter;
                    instance.worldPosYaw[2] = chunkWorldZ + static_cast<float>(z) + 0.5f + jitterZ;
                    instance.worldPosYaw[3] = yawRadians;
                    // Mostly green bushes, with some flowers.
                    const bool placeFlower = ((clumpHash >> 5u) % 100u) < 18u;
                    if (placeFlower) {
                        // Bias strongly toward poppies (tiles 1-2), with rarer lighter wildflowers (3-4).
                        const bool choosePoppy = ((clumpHash >> 13u) % 100u) < 74u;
                        const std::uint32_t flowerTile = choosePoppy
                            ? (1u + ((clumpHash >> 9u) & 0x1u))
                            : (3u + ((clumpHash >> 10u) & 0x1u));
                        if (choosePoppy) {
                            const float poppyBoost = 0.96f + (tintRand1 * 0.10f);
                            instance.colorTint[0] = (0.92f + (tintRand0 * 0.14f)) * poppyBoost;
                            instance.colorTint[1] = (0.92f + (tintRand2 * 0.14f)) * poppyBoost;
                            instance.colorTint[2] = (0.92f + (tintRand1 * 0.14f)) * poppyBoost;
                        } else {
                            const float flowerBoost = 0.94f + (tintRand1 * 0.12f);
                            instance.colorTint[0] = (0.94f + (tintRand0 * 0.14f)) * flowerBoost;
                            instance.colorTint[1] = (0.94f + (tintRand2 * 0.14f)) * flowerBoost;
                            instance.colorTint[2] = (0.94f + (tintRand1 * 0.14f)) * flowerBoost;
                        }
                        instance.colorTint[3] = static_cast<float>(flowerTile);
                    } else {
                        // Golden grass variation.
                        const float warmBias = 0.50f + (0.50f * tintRand0);
                        const float dryBias = tintRand2;
                        const float brightness = 0.82f + (tintRand1 * 0.32f);
                        const float redBase = std::lerp(0.90f, 1.28f, warmBias);
                        const float greenBase = std::lerp(0.98f, 1.36f, (warmBias * 0.70f) + (dryBias * 0.30f));
                        const float blueBase = std::lerp(0.56f, 0.20f, warmBias);
                        instance.colorTint[0] = redBase * brightness;
                        instance.colorTint[1] = greenBase * brightness;
                        instance.colorTint[2] = blueBase * brightness;
                        instance.colorTint[3] = 4.0f;
                    }
                    grassInstances.push_back(instance);
                }
            }
        }
    }
    return grassInstances;
}

} // namespace odai::world
