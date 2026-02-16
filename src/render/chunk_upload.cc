#include "render/renderer_backend.h"

#include <GLFW/glfw3.h>
#include "core/grid3.h"
#include "core/log.h"
#include "math/math.h"
#include "sim/network_procedural.h"
#include "world/chunk_mesher.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace render {

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "render/renderer_shared.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

void RendererBackend::clearMagicaVoxelMeshes() {
    for (MagicaMeshDraw& draw : m_magicaMeshDraws) {
        if (draw.vertexBufferHandle != kInvalidBufferHandle) {
            scheduleBufferRelease(draw.vertexBufferHandle, m_lastGraphicsTimelineValue);
            draw.vertexBufferHandle = kInvalidBufferHandle;
        }
        if (draw.indexBufferHandle != kInvalidBufferHandle) {
            scheduleBufferRelease(draw.indexBufferHandle, m_lastGraphicsTimelineValue);
            draw.indexBufferHandle = kInvalidBufferHandle;
        }
        draw.indexCount = 0;
    }
    m_magicaMeshDraws.clear();
}


void RendererBackend::setVoxelBaseColorPalette(const std::array<std::uint32_t, 16>& paletteRgba) {
    m_voxelBaseColorPaletteRgba = paletteRgba;
}


bool RendererBackend::uploadMagicaVoxelMesh(
    const world::ChunkMeshData& mesh,
    float worldOffsetX,
    float worldOffsetY,
    float worldOffsetZ
) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }

    if (mesh.vertices.empty() || mesh.indices.empty()) {
        return false;
    }

    BufferCreateDesc vertexCreateDesc{};
    vertexCreateDesc.size = static_cast<VkDeviceSize>(mesh.vertices.size() * sizeof(world::PackedVoxelVertex));
    vertexCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vertexCreateDesc.initialData = mesh.vertices.data();
    const BufferHandle newVertexHandle = m_bufferAllocator.createBuffer(vertexCreateDesc);
    if (newVertexHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "magica voxel vertex buffer allocation failed";
        return false;
    }

    BufferCreateDesc indexCreateDesc{};
    indexCreateDesc.size = static_cast<VkDeviceSize>(mesh.indices.size() * sizeof(std::uint32_t));
    indexCreateDesc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    indexCreateDesc.initialData = mesh.indices.data();
    const BufferHandle newIndexHandle = m_bufferAllocator.createBuffer(indexCreateDesc);
    if (newIndexHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "magica voxel index buffer allocation failed";
        m_bufferAllocator.destroyBuffer(newVertexHandle);
        return false;
    }

    const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(newVertexHandle);
    if (vertexBuffer != VK_NULL_HANDLE) {
        setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(vertexBuffer), "mesh.magicaVoxel.vertex");
    }
    const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(newIndexHandle);
    if (indexBuffer != VK_NULL_HANDLE) {
        setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(indexBuffer), "mesh.magicaVoxel.index");
    }

    MagicaMeshDraw draw{};
    draw.vertexBufferHandle = newVertexHandle;
    draw.indexBufferHandle = newIndexHandle;
    draw.indexCount = static_cast<uint32_t>(mesh.indices.size());
    draw.offsetX = worldOffsetX;
    draw.offsetY = worldOffsetY;
    draw.offsetZ = worldOffsetZ;
    m_magicaMeshDraws.push_back(draw);
    return true;
}


bool RendererBackend::updateChunkMesh(const world::ChunkGrid& chunkGrid) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    (void)chunkGrid;
    m_chunkMeshRebuildRequested = true;
    m_pendingChunkRemeshIndices.clear();
    m_voxelGiWorldDirty = true;
    return true;
}


bool RendererBackend::updateChunkMesh(const world::ChunkGrid& chunkGrid, std::size_t chunkIndex) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    if (chunkIndex >= chunkGrid.chunks().size()) {
        return false;
    }
    if (m_chunkMeshRebuildRequested) {
        return true;
    }
    if (std::find(m_pendingChunkRemeshIndices.begin(), m_pendingChunkRemeshIndices.end(), chunkIndex) ==
        m_pendingChunkRemeshIndices.end()) {
        m_pendingChunkRemeshIndices.push_back(chunkIndex);
    }
    m_voxelGiWorldDirty = true;
    return true;
}


bool RendererBackend::updateChunkMesh(const world::ChunkGrid& chunkGrid, std::span<const std::size_t> chunkIndices) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    if (chunkIndices.empty()) {
        return true;
    }
    if (m_chunkMeshRebuildRequested) {
        return true;
    }
    for (const std::size_t chunkIndex : chunkIndices) {
        if (chunkIndex >= chunkGrid.chunks().size()) {
            return false;
        }
        if (std::find(m_pendingChunkRemeshIndices.begin(), m_pendingChunkRemeshIndices.end(), chunkIndex) ==
            m_pendingChunkRemeshIndices.end()) {
            m_pendingChunkRemeshIndices.push_back(chunkIndex);
        }
    }
    m_voxelGiWorldDirty = true;
    return true;
}


bool RendererBackend::useSpatialPartitioningQueries() const {
    return m_debugEnableSpatialQueries;
}

world::ClipmapConfig RendererBackend::clipmapQueryConfig() const {
    return m_debugClipmapConfig;
}


void RendererBackend::setSpatialQueryStats(
    bool used,
    const world::SpatialQueryStats& stats,
    std::uint32_t visibleChunkCount
) {
    m_debugSpatialQueriesUsed = used;
    m_debugSpatialQueryStats = stats;
    m_debugSpatialVisibleChunkCount = visibleChunkCount;
}


bool RendererBackend::createChunkBuffers(const world::ChunkGrid& chunkGrid, std::span<const std::size_t> remeshChunkIndices) {
    if (chunkGrid.chunks().empty()) {
        return false;
    }

    const std::vector<world::Chunk>& chunks = chunkGrid.chunks();
    const std::vector<ChunkDrawRange> previousChunkDrawRanges = m_chunkDrawRanges;
    const std::uint32_t previousDebugChunkMeshVertexCount = m_debugChunkMeshVertexCount;
    const std::uint32_t previousDebugChunkMeshIndexCount = m_debugChunkMeshIndexCount;
    auto rollbackChunkDrawState = [&]() {
        m_chunkDrawRanges = previousChunkDrawRanges;
        m_debugChunkMeshVertexCount = previousDebugChunkMeshVertexCount;
        m_debugChunkMeshIndexCount = previousDebugChunkMeshIndexCount;
    };
    const std::size_t expectedDrawRangeCount = chunks.size() * world::kChunkMeshLodCount;
    if (m_chunkDrawRanges.size() != expectedDrawRangeCount) {
        m_chunkDrawRanges.assign(expectedDrawRangeCount, ChunkDrawRange{});
    }
    if (m_chunkLodMeshCache.size() != chunks.size()) {
        m_chunkLodMeshCache.assign(chunks.size(), world::ChunkLodMeshes{});
        m_chunkLodMeshCacheValid = false;
    }
    if (m_chunkGrassInstanceCache.size() != chunks.size()) {
        m_chunkGrassInstanceCache.assign(chunks.size(), std::vector<GrassBillboardInstance>{});
    }

    auto rebuildGrassInstancesForChunk = [&](std::size_t chunkArrayIndex) {
        if (chunkArrayIndex >= chunks.size()) {
            return;
        }
        const world::Chunk& chunk = chunks[chunkArrayIndex];
        std::vector<GrassBillboardInstance>& grassInstances = m_chunkGrassInstanceCache[chunkArrayIndex];
        grassInstances.clear();
        grassInstances.reserve(448);

        const float chunkWorldX = static_cast<float>(chunk.chunkX() * world::Chunk::kSizeX);
        const float chunkWorldY = static_cast<float>(chunk.chunkY() * world::Chunk::kSizeY);
        const float chunkWorldZ = static_cast<float>(chunk.chunkZ() * world::Chunk::kSizeZ);

        for (int y = 0; y < world::Chunk::kSizeY - 1; ++y) {
            for (int z = 0; z < world::Chunk::kSizeZ; ++z) {
                for (int x = 0; x < world::Chunk::kSizeX; ++x) {
                    if (chunk.voxelAt(x, y, z).type != world::VoxelType::Grass) {
                        continue;
                    }
                    if (chunk.voxelAt(x, y + 1, z).type != world::VoxelType::Empty) {
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

                        GrassBillboardInstance instance{};
                        instance.worldPosYaw[0] = chunkWorldX + static_cast<float>(x) + 0.5f + jitterX;
                        // Lift slightly above the supporting voxel top to avoid depth tie flicker.
                        instance.worldPosYaw[1] = chunkWorldY + static_cast<float>(y) + 1.02f + yJitter;
                        instance.worldPosYaw[2] = chunkWorldZ + static_cast<float>(z) + 0.5f + jitterZ;
                        instance.worldPosYaw[3] = yawRadians;
                        // Mostly green bushes, with some flowers.
                        const bool placeFlower = ((clumpHash >> 5u) % 100u) < 18u;
                        if (placeFlower) {
                            // Bias strongly toward poppies (tiles 5-6), with rarer lighter wildflowers (7-8).
                            const bool choosePoppy = ((clumpHash >> 13u) % 100u) < 74u;
                            const std::uint32_t flowerTile = choosePoppy
                                ? (5u + ((clumpHash >> 9u) & 0x1u))
                                : (7u + ((clumpHash >> 10u) & 0x1u));
                            if (choosePoppy) {
                                const float poppyBoost = 0.92f + (tintRand1 * 0.30f);
                                instance.colorTint[0] = (1.05f + (tintRand0 * 0.55f)) * poppyBoost;
                                instance.colorTint[1] = (0.58f + (tintRand2 * 0.38f)) * poppyBoost;
                                instance.colorTint[2] = (0.40f + (tintRand1 * 0.24f)) * poppyBoost;
                            } else {
                                const float flowerBoost = 0.88f + (tintRand1 * 0.30f);
                                instance.colorTint[0] = (0.96f + (tintRand0 * 0.42f)) * flowerBoost;
                                instance.colorTint[1] = (0.96f + (tintRand2 * 0.42f)) * flowerBoost;
                                instance.colorTint[2] = (0.96f + (tintRand1 * 0.42f)) * flowerBoost;
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
    };

    std::size_t remeshedChunkCount = 0;
    std::size_t remeshedActiveVertexCount = 0;
    std::size_t remeshedActiveIndexCount = 0;
    std::size_t remeshedNaiveVertexCount = 0;
    std::size_t remeshedNaiveIndexCount = 0;
    const auto countMeshGeometry = [](const world::ChunkLodMeshes& lodMeshes, std::size_t& outVertices, std::size_t& outIndices) {
        for (const world::ChunkMeshData& lodMesh : lodMeshes.lodMeshes) {
            outVertices += lodMesh.vertices.size();
            outIndices += lodMesh.indices.size();
        }
    };
    const bool fullRemesh = !m_chunkLodMeshCacheValid || remeshChunkIndices.empty();
    const auto remeshStart = std::chrono::steady_clock::now();
    if (fullRemesh) {
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            m_chunkLodMeshCache[chunkArrayIndex] =
                world::buildChunkLodMeshes(chunks[chunkArrayIndex], m_chunkMeshingOptions);
            rebuildGrassInstancesForChunk(chunkArrayIndex);
            countMeshGeometry(
                m_chunkLodMeshCache[chunkArrayIndex],
                remeshedActiveVertexCount,
                remeshedActiveIndexCount
            );
            if (m_chunkMeshingOptions.mode == world::MeshingMode::Naive) {
                remeshedNaiveVertexCount = remeshedActiveVertexCount;
                remeshedNaiveIndexCount = remeshedActiveIndexCount;
            } else {
                const world::ChunkLodMeshes naiveLodMeshes =
                    world::buildChunkLodMeshes(chunks[chunkArrayIndex], world::MeshingOptions{world::MeshingMode::Naive});
                countMeshGeometry(naiveLodMeshes, remeshedNaiveVertexCount, remeshedNaiveIndexCount);
            }
        }
        remeshedChunkCount = chunks.size();
        m_chunkLodMeshCacheValid = true;
    } else {
        std::vector<std::uint8_t> remeshMask(chunks.size(), 0u);
        std::vector<std::size_t> uniqueRemeshChunkIndices;
        uniqueRemeshChunkIndices.reserve(remeshChunkIndices.size());
        for (const std::size_t chunkArrayIndex : remeshChunkIndices) {
            if (chunkArrayIndex >= chunks.size()) {
                rollbackChunkDrawState();
                return false;
            }
            if (remeshMask[chunkArrayIndex] != 0u) {
                continue;
            }
            remeshMask[chunkArrayIndex] = 1u;
            uniqueRemeshChunkIndices.push_back(chunkArrayIndex);
        }

        for (const std::size_t chunkArrayIndex : uniqueRemeshChunkIndices) {
            m_chunkLodMeshCache[chunkArrayIndex] =
                world::buildChunkLodMeshes(chunks[chunkArrayIndex], m_chunkMeshingOptions);
            rebuildGrassInstancesForChunk(chunkArrayIndex);
            countMeshGeometry(
                m_chunkLodMeshCache[chunkArrayIndex],
                remeshedActiveVertexCount,
                remeshedActiveIndexCount
            );
            if (m_chunkMeshingOptions.mode == world::MeshingMode::Naive) {
                remeshedNaiveVertexCount = remeshedActiveVertexCount;
                remeshedNaiveIndexCount = remeshedActiveIndexCount;
            } else {
                const world::ChunkLodMeshes naiveLodMeshes =
                    world::buildChunkLodMeshes(chunks[chunkArrayIndex], world::MeshingOptions{world::MeshingMode::Naive});
                countMeshGeometry(naiveLodMeshes, remeshedNaiveVertexCount, remeshedNaiveIndexCount);
            }
        }
        remeshedChunkCount = uniqueRemeshChunkIndices.size();
    }
    const auto remeshEnd = std::chrono::steady_clock::now();
    const std::chrono::duration<float, std::milli> remeshMs = remeshEnd - remeshStart;
    m_debugChunkLastRemeshedChunkCount = static_cast<std::uint32_t>(remeshedChunkCount);
    m_debugChunkLastRemeshActiveVertexCount = static_cast<std::uint32_t>(remeshedActiveVertexCount);
    m_debugChunkLastRemeshActiveIndexCount = static_cast<std::uint32_t>(remeshedActiveIndexCount);
    m_debugChunkLastRemeshNaiveVertexCount = static_cast<std::uint32_t>(remeshedNaiveVertexCount);
    m_debugChunkLastRemeshNaiveIndexCount = static_cast<std::uint32_t>(remeshedNaiveIndexCount);
    m_debugChunkLastRemeshMs = remeshMs.count();
    if (remeshedNaiveIndexCount > 0) {
        const float ratio = static_cast<float>(remeshedActiveIndexCount) / static_cast<float>(remeshedNaiveIndexCount);
        m_debugChunkLastRemeshReductionPercent = std::clamp(100.0f * (1.0f - ratio), 0.0f, 100.0f);
    } else {
        m_debugChunkLastRemeshReductionPercent = 0.0f;
    }
    if (fullRemesh) {
        m_debugChunkLastFullRemeshMs = remeshMs.count();
    }

    std::vector<GrassBillboardInstance> combinedGrassInstances;
    {
        std::size_t totalGrassInstanceCount = 0;
        for (const std::vector<GrassBillboardInstance>& chunkGrass : m_chunkGrassInstanceCache) {
            totalGrassInstanceCount += chunkGrass.size();
        }
        combinedGrassInstances.reserve(totalGrassInstanceCount);
        for (const std::vector<GrassBillboardInstance>& chunkGrass : m_chunkGrassInstanceCache) {
            combinedGrassInstances.insert(combinedGrassInstances.end(), chunkGrass.begin(), chunkGrass.end());
        }
    }
    // Temporary toggle: disable plant rendering by forcing zero grass billboard instances.
    constexpr bool kDisablePlantRendering = true;
    if (kDisablePlantRendering) {
        combinedGrassInstances.clear();
    }

    if (combinedGrassInstances.empty()) {
        if (m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
            const uint64_t grassReleaseValue = m_lastGraphicsTimelineValue;
            scheduleBufferRelease(m_grassBillboardInstanceBufferHandle, grassReleaseValue);
            m_grassBillboardInstanceBufferHandle = kInvalidBufferHandle;
        }
        m_grassBillboardInstanceCount = 0;
    } else {
        if (fullRemesh) {
            float minR = std::numeric_limits<float>::max();
            float minG = std::numeric_limits<float>::max();
            float minB = std::numeric_limits<float>::max();
            float maxR = std::numeric_limits<float>::lowest();
            float maxG = std::numeric_limits<float>::lowest();
            float maxB = std::numeric_limits<float>::lowest();
            for (const GrassBillboardInstance& instance : combinedGrassInstances) {
                minR = std::min(minR, instance.colorTint[0]);
                minG = std::min(minG, instance.colorTint[1]);
                minB = std::min(minB, instance.colorTint[2]);
                maxR = std::max(maxR, instance.colorTint[0]);
                maxG = std::max(maxG, instance.colorTint[1]);
                maxB = std::max(maxB, instance.colorTint[2]);
            }
            VOX_LOGI("render") << "grass tint range rgb min=("
                              << minR << ", " << minG << ", " << minB
                              << "), max=("
                              << maxR << ", " << maxG << ", " << maxB
                              << "), instances=" << combinedGrassInstances.size() << "\n";
        }

        BufferCreateDesc grassInstanceCreateDesc{};
        grassInstanceCreateDesc.size = static_cast<VkDeviceSize>(combinedGrassInstances.size() * sizeof(GrassBillboardInstance));
        grassInstanceCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        grassInstanceCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        grassInstanceCreateDesc.initialData = combinedGrassInstances.data();

        const BufferHandle newGrassInstanceBufferHandle = m_bufferAllocator.createBuffer(grassInstanceCreateDesc);
        if (newGrassInstanceBufferHandle != kInvalidBufferHandle) {
            const VkBuffer grassInstanceBuffer = m_bufferAllocator.getBuffer(newGrassInstanceBufferHandle);
            if (grassInstanceBuffer != VK_NULL_HANDLE) {
                setObjectName(
                    VK_OBJECT_TYPE_BUFFER,
                    vkHandleToUint64(grassInstanceBuffer),
                    "mesh.grassBillboard.instances"
                );
            }
            if (m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
                const uint64_t grassReleaseValue = m_lastGraphicsTimelineValue;
                scheduleBufferRelease(m_grassBillboardInstanceBufferHandle, grassReleaseValue);
            }
            m_grassBillboardInstanceBufferHandle = newGrassInstanceBufferHandle;
            m_grassBillboardInstanceCount = static_cast<uint32_t>(combinedGrassInstances.size());
        } else {
            VOX_LOGE("render") << "grass billboard instance buffer allocation failed";
        }
    }

    std::vector<world::PackedVoxelVertex> combinedVertices;
    std::vector<std::uint32_t> combinedIndices;
    std::size_t uploadedVertexCount = 0;
    std::size_t uploadedIndexCount = 0;

    for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
        const world::Chunk& chunk = chunks[chunkArrayIndex];
        const world::ChunkLodMeshes& chunkLodMeshes = m_chunkLodMeshCache[chunkArrayIndex];

        for (std::size_t lodIndex = 0; lodIndex < world::kChunkMeshLodCount; ++lodIndex) {
            const world::ChunkMeshData& chunkMesh = chunkLodMeshes.lodMeshes[lodIndex];
            const std::size_t drawRangeArrayIndex = (chunkArrayIndex * world::kChunkMeshLodCount) + lodIndex;
            ChunkDrawRange& drawRange = m_chunkDrawRanges[drawRangeArrayIndex];

            drawRange.offsetX = static_cast<float>(chunk.chunkX() * world::Chunk::kSizeX);
            drawRange.offsetY = static_cast<float>(chunk.chunkY() * world::Chunk::kSizeY);
            drawRange.offsetZ = static_cast<float>(chunk.chunkZ() * world::Chunk::kSizeZ);
            drawRange.firstIndex = 0;
            drawRange.vertexOffset = 0;
            drawRange.indexCount = 0;

            if (chunkMesh.vertices.empty() || chunkMesh.indices.empty()) {
                continue;
            }

            const std::size_t baseVertexSize = combinedVertices.size();
            if (baseVertexSize > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
                VOX_LOGE("render") << "chunk mesh vertex offset exceeds int32 range";
                rollbackChunkDrawState();
                return false;
            }
            const uint32_t baseVertex = static_cast<uint32_t>(baseVertexSize);
            const uint32_t firstIndex = static_cast<uint32_t>(combinedIndices.size());

            combinedVertices.insert(combinedVertices.end(), chunkMesh.vertices.begin(), chunkMesh.vertices.end());
            combinedIndices.reserve(combinedIndices.size() + chunkMesh.indices.size());
            for (const std::uint32_t index : chunkMesh.indices) {
                combinedIndices.push_back(index + baseVertex);
            }

            drawRange.firstIndex = firstIndex;
            // Indices are already rebased into global vertex space.
            drawRange.vertexOffset = 0;
            drawRange.indexCount = static_cast<uint32_t>(chunkMesh.indices.size());
            uploadedVertexCount += chunkMesh.vertices.size();
            uploadedIndexCount += chunkMesh.indices.size();
        }
    }
    m_debugChunkMeshVertexCount = static_cast<std::uint32_t>(uploadedVertexCount);
    m_debugChunkMeshIndexCount = static_cast<std::uint32_t>(uploadedIndexCount);

    std::array<uint32_t, 2> meshQueueFamilies = {
        m_graphicsQueueFamilyIndex,
        m_transferQueueFamilyIndex
    };
    if (meshQueueFamilies[0] == meshQueueFamilies[1]) {
        meshQueueFamilies[1] = UINT32_MAX;
    }

    BufferHandle newChunkVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle newChunkIndexBufferHandle = kInvalidBufferHandle;
    std::optional<FrameArenaSlice> chunkVertexUploadSliceOpt = std::nullopt;
    std::optional<FrameArenaSlice> chunkIndexUploadSliceOpt = std::nullopt;
    auto cleanupPendingAllocations = [&]() {
        if (newChunkVertexBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(newChunkVertexBufferHandle);
            newChunkVertexBufferHandle = kInvalidBufferHandle;
        }
        if (newChunkIndexBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(newChunkIndexBufferHandle);
            newChunkIndexBufferHandle = kInvalidBufferHandle;
        }
    };

    collectCompletedBufferReleases();

    if (m_transferCommandBufferInFlightValue > 0) {
        if (!isTimelineValueReached(m_transferCommandBufferInFlightValue)) {
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }
    }
    m_transferCommandBufferInFlightValue = 0;
    collectCompletedBufferReleases();
    const uint64_t previousChunkReadyTimelineValue = m_currentChunkReadyTimelineValue;
    const bool hasChunkCopies = !combinedVertices.empty() && !combinedIndices.empty();

    if (hasChunkCopies) {
        const VkDeviceSize vertexBufferSize =
            static_cast<VkDeviceSize>(combinedVertices.size() * sizeof(world::PackedVoxelVertex));
        const VkDeviceSize indexBufferSize =
            static_cast<VkDeviceSize>(combinedIndices.size() * sizeof(std::uint32_t));

        BufferCreateDesc vertexCreateDesc{};
        vertexCreateDesc.size = vertexBufferSize;
        vertexCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (meshQueueFamilies[1] != UINT32_MAX) {
            vertexCreateDesc.queueFamilyIndices = meshQueueFamilies.data();
            vertexCreateDesc.queueFamilyIndexCount = 2;
        }
        newChunkVertexBufferHandle = m_bufferAllocator.createBuffer(vertexCreateDesc);
        if (newChunkVertexBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "chunk global vertex buffer allocation failed";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }
        {
            const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(newChunkVertexBufferHandle);
            if (vertexBuffer != VK_NULL_HANDLE) {
                setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(vertexBuffer), "chunk.global.vertex");
            }
        }

        BufferCreateDesc indexCreateDesc{};
        indexCreateDesc.size = indexBufferSize;
        indexCreateDesc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (meshQueueFamilies[1] != UINT32_MAX) {
            indexCreateDesc.queueFamilyIndices = meshQueueFamilies.data();
            indexCreateDesc.queueFamilyIndexCount = 2;
        }
        newChunkIndexBufferHandle = m_bufferAllocator.createBuffer(indexCreateDesc);
        if (newChunkIndexBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "chunk global index buffer allocation failed";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }
        {
            const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(newChunkIndexBufferHandle);
            if (indexBuffer != VK_NULL_HANDLE) {
                setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(indexBuffer), "chunk.global.index");
            }
        }

        chunkVertexUploadSliceOpt = m_frameArena.allocateUpload(
            vertexBufferSize,
            static_cast<VkDeviceSize>(alignof(world::PackedVoxelVertex)),
            FrameArenaUploadKind::Unknown
        );
        if (!chunkVertexUploadSliceOpt.has_value() || chunkVertexUploadSliceOpt->mapped == nullptr) {
            VOX_LOGE("render") << "chunk global vertex upload slice allocation failed";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }
        std::memcpy(
            chunkVertexUploadSliceOpt->mapped,
            combinedVertices.data(),
            static_cast<size_t>(vertexBufferSize)
        );

        chunkIndexUploadSliceOpt = m_frameArena.allocateUpload(
            indexBufferSize,
            static_cast<VkDeviceSize>(alignof(std::uint32_t)),
            FrameArenaUploadKind::Unknown
        );
        if (!chunkIndexUploadSliceOpt.has_value() || chunkIndexUploadSliceOpt->mapped == nullptr) {
            VOX_LOGE("render") << "chunk global index upload slice allocation failed";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }
        std::memcpy(
            chunkIndexUploadSliceOpt->mapped,
            combinedIndices.data(),
            static_cast<size_t>(indexBufferSize)
        );
    }

    uint64_t transferSignalValue = 0;
    if (hasChunkCopies) {
        const VkResult resetResult = vkResetCommandPool(m_device, m_transferCommandPool, 0);
        if (resetResult != VK_SUCCESS) {
            logVkFailure("vkResetCommandPool(transfer)", resetResult);
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(m_transferCommandBuffer, &beginInfo) != VK_SUCCESS) {
            VOX_LOGE("render") << "vkBeginCommandBuffer (transfer) failed\n";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }

        {
            const VkDeviceSize vertexBufferSize = m_bufferAllocator.getSize(newChunkVertexBufferHandle);
            const VkDeviceSize indexBufferSize = m_bufferAllocator.getSize(newChunkIndexBufferHandle);

            VkBufferCopy vertexCopy{};
            vertexCopy.srcOffset = chunkVertexUploadSliceOpt->offset;
            vertexCopy.size = vertexBufferSize;
            vkCmdCopyBuffer(
                m_transferCommandBuffer,
                m_bufferAllocator.getBuffer(chunkVertexUploadSliceOpt->buffer),
                m_bufferAllocator.getBuffer(newChunkVertexBufferHandle),
                1,
                &vertexCopy
            );

            VkBufferCopy indexCopy{};
            indexCopy.srcOffset = chunkIndexUploadSliceOpt->offset;
            indexCopy.size = indexBufferSize;
            vkCmdCopyBuffer(
                m_transferCommandBuffer,
                m_bufferAllocator.getBuffer(chunkIndexUploadSliceOpt->buffer),
                m_bufferAllocator.getBuffer(newChunkIndexBufferHandle),
                1,
                &indexCopy
            );
        }

        if (vkEndCommandBuffer(m_transferCommandBuffer) != VK_SUCCESS) {
            VOX_LOGE("render") << "vkEndCommandBuffer (transfer) failed\n";
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }

        transferSignalValue = m_nextTimelineValue++;
        std::array<VkSemaphore, 1> transferWaitSemaphores{};
        std::array<VkPipelineStageFlags, 1> transferWaitStages{};
        std::array<uint64_t, 1> transferWaitValues{};
        uint32_t transferWaitCount = 0;
        if (m_lastGraphicsTimelineValue > 0) {
            transferWaitSemaphores[0] = m_renderTimelineSemaphore;
            transferWaitStages[0] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            transferWaitValues[0] = m_lastGraphicsTimelineValue;
            transferWaitCount = 1;
        }

        VkSemaphore timelineSemaphore = m_renderTimelineSemaphore;
        VkTimelineSemaphoreSubmitInfo timelineSubmitInfo{};
        timelineSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineSubmitInfo.waitSemaphoreValueCount = transferWaitCount;
        timelineSubmitInfo.pWaitSemaphoreValues =
            transferWaitCount > 0 ? transferWaitValues.data() : nullptr;
        timelineSubmitInfo.signalSemaphoreValueCount = 1;
        timelineSubmitInfo.pSignalSemaphoreValues = &transferSignalValue;

        VkSubmitInfo transferSubmitInfo{};
        transferSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        transferSubmitInfo.pNext = &timelineSubmitInfo;
        transferSubmitInfo.waitSemaphoreCount = transferWaitCount;
        transferSubmitInfo.pWaitSemaphores =
            transferWaitCount > 0 ? transferWaitSemaphores.data() : nullptr;
        transferSubmitInfo.pWaitDstStageMask =
            transferWaitCount > 0 ? transferWaitStages.data() : nullptr;
        transferSubmitInfo.commandBufferCount = 1;
        transferSubmitInfo.pCommandBuffers = &m_transferCommandBuffer;
        transferSubmitInfo.signalSemaphoreCount = 1;
        transferSubmitInfo.pSignalSemaphores = &timelineSemaphore;

        const VkResult submitResult = vkQueueSubmit(m_transferQueue, 1, &transferSubmitInfo, VK_NULL_HANDLE);
        if (submitResult != VK_SUCCESS) {
            logVkFailure("vkQueueSubmit(transfer)", submitResult);
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }

        m_currentChunkReadyTimelineValue = transferSignalValue;
        m_pendingTransferTimelineValue = transferSignalValue;
        m_transferCommandBufferInFlightValue = transferSignalValue;
    }

    const uint64_t oldChunkReleaseValue = std::max(m_lastGraphicsTimelineValue, previousChunkReadyTimelineValue);
    scheduleBufferRelease(m_chunkVertexBufferHandle, oldChunkReleaseValue);
    scheduleBufferRelease(m_chunkIndexBufferHandle, oldChunkReleaseValue);
    m_chunkVertexBufferHandle = newChunkVertexBufferHandle;
    m_chunkIndexBufferHandle = newChunkIndexBufferHandle;
    newChunkVertexBufferHandle = kInvalidBufferHandle;
    newChunkIndexBufferHandle = kInvalidBufferHandle;

    VOX_LOGD("render") << "chunk upload queued (ranges=" << m_chunkDrawRanges.size()
                       << ", remeshedChunks=" << remeshedChunkCount
                       << ", meshingMode="
                       << (m_chunkMeshingOptions.mode == world::MeshingMode::Greedy ? "greedy" : "naive")
                       << ", vertices=" << uploadedVertexCount
                       << ", indices=" << uploadedIndexCount
                       << (hasChunkCopies
                               ? (", timelineValue=" + std::to_string(transferSignalValue))
                               : ", immediate=true")
                       << ")";
    return true;
}
} // namespace render
