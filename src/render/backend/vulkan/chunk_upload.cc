#include "render/backend/vulkan/renderer_backend.h"

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

namespace voxelsprout::render {

#include "render/renderer_shared.h"

namespace {

RtVertex decodePackedVoxelVertexPosition(std::uint32_t packedBits, float offsetX, float offsetY, float offsetZ) {
    const std::uint32_t x =
        (packedBits >> voxelsprout::world::PackedVoxelVertex::kShiftX) & voxelsprout::world::PackedVoxelVertex::kMask5;
    const std::uint32_t y =
        (packedBits >> voxelsprout::world::PackedVoxelVertex::kShiftY) & voxelsprout::world::PackedVoxelVertex::kMask5;
    const std::uint32_t z =
        (packedBits >> voxelsprout::world::PackedVoxelVertex::kShiftZ) & voxelsprout::world::PackedVoxelVertex::kMask5;
    const std::uint32_t face =
        (packedBits >> voxelsprout::world::PackedVoxelVertex::kShiftFace) & voxelsprout::world::PackedVoxelVertex::kMask3;
    const std::uint32_t corner =
        (packedBits >> voxelsprout::world::PackedVoxelVertex::kShiftCorner) & voxelsprout::world::PackedVoxelVertex::kMask2;

    RtVertex vertex{};
    vertex.position[0] = static_cast<float>(x) + offsetX;
    vertex.position[1] = static_cast<float>(y) + offsetY;
    vertex.position[2] = static_cast<float>(z) + offsetZ;
    if (face == 0u) {
        vertex.position[0] += 1.0f;
        vertex.position[1] += (corner == 1u || corner == 2u) ? 1.0f : 0.0f;
        vertex.position[2] += (corner == 2u || corner == 3u) ? 1.0f : 0.0f;
        return vertex;
    }
    if (face == 1u) {
        vertex.position[1] += (corner == 1u || corner == 2u) ? 1.0f : 0.0f;
        vertex.position[2] += (corner == 0u || corner == 1u) ? 1.0f : 0.0f;
        return vertex;
    }
    if (face == 2u) {
        vertex.position[0] += (corner == 2u || corner == 3u) ? 1.0f : 0.0f;
        vertex.position[1] += 1.0f;
        vertex.position[2] += (corner == 1u || corner == 2u) ? 1.0f : 0.0f;
        return vertex;
    }
    if (face == 3u) {
        vertex.position[0] += (corner == 2u || corner == 3u) ? 1.0f : 0.0f;
        vertex.position[2] += (corner == 0u || corner == 3u) ? 1.0f : 0.0f;
        return vertex;
    }
    if (face == 4u) {
        vertex.position[0] += (corner == 0u || corner == 1u) ? 1.0f : 0.0f;
        vertex.position[1] += (corner == 1u || corner == 2u) ? 1.0f : 0.0f;
        vertex.position[2] += 1.0f;
        return vertex;
    }

    vertex.position[0] += (corner == 2u || corner == 3u) ? 1.0f : 0.0f;
    vertex.position[1] += (corner == 1u || corner == 2u) ? 1.0f : 0.0f;
    return vertex;
}

void destroyRtGeometryBuffers(BufferAllocator& allocator, RtGeometryBuffers& geometry) {
    if (geometry.indexBufferHandle != kInvalidBufferHandle) {
        allocator.destroyBuffer(geometry.indexBufferHandle);
        geometry.indexBufferHandle = kInvalidBufferHandle;
    }
    if (geometry.vertexBufferHandle != kInvalidBufferHandle) {
        allocator.destroyBuffer(geometry.vertexBufferHandle);
        geometry.vertexBufferHandle = kInvalidBufferHandle;
    }
    geometry.vertexCount = 0;
    geometry.indexCount = 0;
}

bool createRtGeometryBuffers(
    BufferAllocator& allocator,
    const std::vector<RtVertex>& vertices,
    const std::vector<std::uint32_t>& indices,
    RtGeometryBuffers& outGeometry
) {
    destroyRtGeometryBuffers(allocator, outGeometry);
    if (vertices.empty() || indices.empty()) {
        return true;
    }

    BufferCreateDesc vertexCreateDesc{};
    vertexCreateDesc.size = static_cast<VkDeviceSize>(vertices.size() * sizeof(RtVertex));
    vertexCreateDesc.usage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vertexCreateDesc.initialData = vertices.data();
    outGeometry.vertexBufferHandle = allocator.createBuffer(vertexCreateDesc);
    if (outGeometry.vertexBufferHandle == kInvalidBufferHandle) {
        return false;
    }

    BufferCreateDesc indexCreateDesc{};
    indexCreateDesc.size = static_cast<VkDeviceSize>(indices.size() * sizeof(std::uint32_t));
    indexCreateDesc.usage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    indexCreateDesc.initialData = indices.data();
    outGeometry.indexBufferHandle = allocator.createBuffer(indexCreateDesc);
    if (outGeometry.indexBufferHandle == kInvalidBufferHandle) {
        destroyRtGeometryBuffers(allocator, outGeometry);
        return false;
    }

    outGeometry.vertexCount = static_cast<std::uint32_t>(vertices.size());
    outGeometry.indexCount = static_cast<std::uint32_t>(indices.size());
    return true;
}

} // namespace

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
    for (RtGeometryBuffers& geometry : m_rtMagicaGeometries) {
        destroyRtGeometryBuffers(m_bufferAllocator, geometry);
    }
    m_rtMagicaGeometries.clear();
    markRayTracingSceneDirty();
}


void RendererBackend::setVoxelBaseColorPalette(const std::array<std::uint32_t, 16>& paletteRgba) {
    m_voxelBaseColorPaletteRgba = paletteRgba;
}


bool RendererBackend::uploadMagicaVoxelMesh(
    const voxelsprout::world::ChunkMeshData& mesh,
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
    vertexCreateDesc.size = static_cast<VkDeviceSize>(mesh.vertices.size() * sizeof(voxelsprout::world::PackedVoxelVertex));
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
    if (m_rayTracingCapabilityProbe.rayTracingCoreReady) {
        std::vector<RtVertex> rtVertices;
        rtVertices.reserve(mesh.vertices.size());
        for (const voxelsprout::world::PackedVoxelVertex& vertex : mesh.vertices) {
            rtVertices.push_back(decodePackedVoxelVertexPosition(vertex.bits, worldOffsetX, worldOffsetY, worldOffsetZ));
        }
        RtGeometryBuffers rtGeometry{};
        if (!createRtGeometryBuffers(m_bufferAllocator, rtVertices, mesh.indices, rtGeometry)) {
            VOX_LOGE("render") << "magica voxel RT geometry buffer allocation failed";
        } else {
            m_rtMagicaGeometries.push_back(rtGeometry);
            markRayTracingSceneDirty();
            if (rayTracingRuntimeReady() && !rebuildRayTracingScene()) {
                VOX_LOGE("render") << "magica voxel RT scene rebuild failed";
            }
        }
    }
    return true;
}


bool RendererBackend::updateChunkMesh(const voxelsprout::world::ChunkGrid& chunkGrid) {
    if (m_device == VK_NULL_HANDLE) {
        return false;
    }
    (void)chunkGrid;
    m_chunkMeshRebuildRequested = true;
    m_pendingChunkRemeshIndices.clear();
    m_voxelGiWorldDirty = true;
    ++m_voxelGiWorldVersion;
    m_voxelGiOccupancyFullRebuildInProgress = true;
    m_voxelGiOccupancyFullRebuildNeedsClear = true;
    m_voxelGiOccupancyFullRebuildCursor = 0;
    m_voxelGiDirtyChunkIndices.clear();
    return true;
}


bool RendererBackend::updateChunkMesh(const voxelsprout::world::ChunkGrid& chunkGrid, std::size_t chunkIndex) {
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
    if (!m_voxelGiOccupancyFullRebuildInProgress &&
        std::find(m_voxelGiDirtyChunkIndices.begin(), m_voxelGiDirtyChunkIndices.end(), chunkIndex) ==
            m_voxelGiDirtyChunkIndices.end()) {
        m_voxelGiDirtyChunkIndices.push_back(chunkIndex);
    }
    m_voxelGiWorldDirty = true;
    ++m_voxelGiWorldVersion;
    return true;
}


bool RendererBackend::updateChunkMesh(const voxelsprout::world::ChunkGrid& chunkGrid, std::span<const std::size_t> chunkIndices) {
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
        if (!m_voxelGiOccupancyFullRebuildInProgress &&
            std::find(m_voxelGiDirtyChunkIndices.begin(), m_voxelGiDirtyChunkIndices.end(), chunkIndex) ==
                m_voxelGiDirtyChunkIndices.end()) {
            m_voxelGiDirtyChunkIndices.push_back(chunkIndex);
        }
    }
    m_voxelGiWorldDirty = true;
    ++m_voxelGiWorldVersion;
    return true;
}


bool RendererBackend::useSpatialPartitioningQueries() const {
    return m_debugEnableSpatialQueries;
}

voxelsprout::world::ClipmapConfig RendererBackend::clipmapQueryConfig() const {
    return m_debugClipmapConfig;
}


void RendererBackend::setSpatialQueryStats(
    bool used,
    const voxelsprout::world::SpatialQueryStats& stats,
    std::uint32_t visibleChunkCount
) {
    m_debugSpatialQueriesUsed = used;
    m_debugSpatialQueryStats = stats;
    m_debugSpatialVisibleChunkCount = visibleChunkCount;
}


bool RendererBackend::createChunkBuffers(const voxelsprout::world::ChunkGrid& chunkGrid, std::span<const std::size_t> remeshChunkIndices) {
    if (chunkGrid.chunks().empty()) {
        return false;
    }

    const std::vector<voxelsprout::world::Chunk>& chunks = chunkGrid.chunks();
    const std::vector<ChunkDrawRange> previousChunkDrawRanges = m_chunkDrawRanges;
    const std::uint32_t previousDebugChunkMeshVertexCount = m_debugChunkMeshVertexCount;
    const std::uint32_t previousDebugChunkMeshIndexCount = m_debugChunkMeshIndexCount;
    auto rollbackChunkDrawState = [&]() {
        m_chunkDrawRanges = previousChunkDrawRanges;
        m_debugChunkMeshVertexCount = previousDebugChunkMeshVertexCount;
        m_debugChunkMeshIndexCount = previousDebugChunkMeshIndexCount;
    };
    const std::size_t expectedDrawRangeCount = chunks.size() * voxelsprout::world::kChunkMeshLodCount;
    if (m_chunkDrawRanges.size() != expectedDrawRangeCount) {
        m_chunkDrawRanges.assign(expectedDrawRangeCount, ChunkDrawRange{});
    }
    if (m_chunkLodMeshCache.size() != chunks.size()) {
        m_chunkLodMeshCache.assign(chunks.size(), voxelsprout::world::ChunkLodMeshes{});
        m_chunkLodMeshCacheValid = false;
    }
    if (m_chunkGrassInstanceCache.size() != chunks.size()) {
        m_chunkGrassInstanceCache.assign(chunks.size(), std::vector<GrassBillboardInstance>{});
    }

    auto rebuildGrassInstancesForChunk = [&](std::size_t chunkArrayIndex) {
        if (chunkArrayIndex >= chunks.size()) {
            return;
        }
        const voxelsprout::world::Chunk& chunk = chunks[chunkArrayIndex];
        std::vector<GrassBillboardInstance>& grassInstances = m_chunkGrassInstanceCache[chunkArrayIndex];
        grassInstances.clear();
        grassInstances.reserve(448);

        const float chunkWorldX = static_cast<float>(chunk.chunkX() * voxelsprout::world::Chunk::kSizeX);
        const float chunkWorldY = static_cast<float>(chunk.chunkY() * voxelsprout::world::Chunk::kSizeY);
        const float chunkWorldZ = static_cast<float>(chunk.chunkZ() * voxelsprout::world::Chunk::kSizeZ);

        for (int y = 0; y < voxelsprout::world::Chunk::kSizeY - 1; ++y) {
            for (int z = 0; z < voxelsprout::world::Chunk::kSizeZ; ++z) {
                for (int x = 0; x < voxelsprout::world::Chunk::kSizeX; ++x) {
                    if (chunk.voxelAt(x, y, z).type != voxelsprout::world::VoxelType::Grass) {
                        continue;
                    }
                    if (chunk.voxelAt(x, y + 1, z).type != voxelsprout::world::VoxelType::Empty) {
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
    };

    std::size_t remeshedChunkCount = 0;
    std::size_t remeshedActiveVertexCount = 0;
    std::size_t remeshedActiveIndexCount = 0;
    std::size_t remeshedNaiveVertexCount = 0;
    std::size_t remeshedNaiveIndexCount = 0;
    const auto countMeshGeometry = [](const voxelsprout::world::ChunkLodMeshes& lodMeshes, std::size_t& outVertices, std::size_t& outIndices) {
        for (const voxelsprout::world::ChunkMeshData& lodMesh : lodMeshes.lodMeshes) {
            outVertices += lodMesh.vertices.size();
            outIndices += lodMesh.indices.size();
        }
    };
    const bool fullRemesh = !m_chunkLodMeshCacheValid || remeshChunkIndices.empty();
    const auto remeshStart = std::chrono::steady_clock::now();
    if (fullRemesh) {
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            m_chunkLodMeshCache[chunkArrayIndex] =
                voxelsprout::world::buildChunkLodMeshes(chunks[chunkArrayIndex], m_chunkMeshingOptions);
            rebuildGrassInstancesForChunk(chunkArrayIndex);
            countMeshGeometry(
                m_chunkLodMeshCache[chunkArrayIndex],
                remeshedActiveVertexCount,
                remeshedActiveIndexCount
            );
            if (m_chunkMeshingOptions.mode == voxelsprout::world::MeshingMode::Naive) {
                remeshedNaiveVertexCount = remeshedActiveVertexCount;
                remeshedNaiveIndexCount = remeshedActiveIndexCount;
            } else {
                const voxelsprout::world::ChunkLodMeshes naiveLodMeshes =
                    voxelsprout::world::buildChunkLodMeshes(chunks[chunkArrayIndex], voxelsprout::world::MeshingOptions{voxelsprout::world::MeshingMode::Naive});
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
                voxelsprout::world::buildChunkLodMeshes(chunks[chunkArrayIndex], m_chunkMeshingOptions);
            rebuildGrassInstancesForChunk(chunkArrayIndex);
            countMeshGeometry(
                m_chunkLodMeshCache[chunkArrayIndex],
                remeshedActiveVertexCount,
                remeshedActiveIndexCount
            );
            if (m_chunkMeshingOptions.mode == voxelsprout::world::MeshingMode::Naive) {
                remeshedNaiveVertexCount = remeshedActiveVertexCount;
                remeshedNaiveIndexCount = remeshedActiveIndexCount;
            } else {
                const voxelsprout::world::ChunkLodMeshes naiveLodMeshes =
                    voxelsprout::world::buildChunkLodMeshes(chunks[chunkArrayIndex], voxelsprout::world::MeshingOptions{voxelsprout::world::MeshingMode::Naive});
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

    struct RtChunkSceneMetadata {
        int chunkX = 0;
        int chunkY = 0;
        int chunkZ = 0;
        std::uint32_t vertexCount = 0;
        std::uint32_t indexCount = 0;
        bool geometryResident = false;
    };

    std::vector<voxelsprout::world::PackedVoxelVertex> combinedVertices;
    std::vector<std::uint32_t> combinedIndices;
    std::vector<RtChunkSceneRecord> previousRtChunkSceneRecords = std::move(m_rtChunkSceneRecords);
    const bool previousChunkBlasesInUse = std::any_of(
        previousRtChunkSceneRecords.begin(),
        previousRtChunkSceneRecords.end(),
        [](const RtChunkSceneRecord& record) { return record.blas.handle != VK_NULL_HANDLE; }
    );
    if (previousChunkBlasesInUse && m_graphicsQueue != VK_NULL_HANDLE) {
        const VkResult waitResult = vkQueueWaitIdle(m_graphicsQueue);
        if (waitResult != VK_SUCCESS) {
            logVkFailure("vkQueueWaitIdle(chunkRtRetire)", waitResult);
            m_rtChunkSceneRecords = std::move(previousRtChunkSceneRecords);
            rollbackChunkDrawState();
            return false;
        }
    }
    auto destroyRtAs = [&](RtAccelerationStructure& accelerationStructure) {
        if (accelerationStructure.handle != VK_NULL_HANDLE && m_destroyAccelerationStructureKhr != nullptr) {
            m_destroyAccelerationStructureKhr(m_device, accelerationStructure.handle, nullptr);
            accelerationStructure.handle = VK_NULL_HANDLE;
        }
        if (accelerationStructure.storageBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(accelerationStructure.storageBufferHandle);
            accelerationStructure.storageBufferHandle = kInvalidBufferHandle;
        }
        accelerationStructure.deviceAddress = 0;
        accelerationStructure.primitiveCount = 0;
    };
    std::vector<RtChunkSceneMetadata> previousRtChunkSceneMetadata;
    previousRtChunkSceneMetadata.reserve(previousRtChunkSceneRecords.size());
    for (RtChunkSceneRecord& previousRecord : previousRtChunkSceneRecords) {
        previousRtChunkSceneMetadata.push_back(RtChunkSceneMetadata{
            previousRecord.chunkX,
            previousRecord.chunkY,
            previousRecord.chunkZ,
            previousRecord.vertexCount,
            previousRecord.indexCount,
            previousRecord.geometryResident
        });
        destroyRtAs(previousRecord.blas);
        destroyRtGeometryBuffers(m_bufferAllocator, previousRecord.geometry);
    }
    m_rtChunkSceneRecords.resize(chunks.size());
    m_rtDirtyChunkCount = 0;
    std::size_t uploadedVertexCount = 0;
    std::size_t uploadedIndexCount = 0;

    for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
        const voxelsprout::world::Chunk& chunk = chunks[chunkArrayIndex];
        const voxelsprout::world::ChunkLodMeshes& chunkLodMeshes = m_chunkLodMeshCache[chunkArrayIndex];
        RtChunkSceneRecord& rtChunkRecord = m_rtChunkSceneRecords[chunkArrayIndex];
        rtChunkRecord.chunkX = chunk.chunkX();
        rtChunkRecord.chunkY = chunk.chunkY();
        rtChunkRecord.chunkZ = chunk.chunkZ();

        for (std::size_t lodIndex = 0; lodIndex < voxelsprout::world::kChunkMeshLodCount; ++lodIndex) {
            const voxelsprout::world::ChunkMeshData& chunkMesh = chunkLodMeshes.lodMeshes[lodIndex];
            const std::size_t drawRangeArrayIndex = (chunkArrayIndex * voxelsprout::world::kChunkMeshLodCount) + lodIndex;
            ChunkDrawRange& drawRange = m_chunkDrawRanges[drawRangeArrayIndex];

            drawRange.offsetX = static_cast<float>(chunk.chunkX() * voxelsprout::world::Chunk::kSizeX);
            drawRange.offsetY = static_cast<float>(chunk.chunkY() * voxelsprout::world::Chunk::kSizeY);
            drawRange.offsetZ = static_cast<float>(chunk.chunkZ() * voxelsprout::world::Chunk::kSizeZ);
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
            if (lodIndex == 0u && m_rayTracingCapabilityProbe.rayTracingCoreReady) {
                // RT shadows should trace against the highest-detail chunk mesh.
                rtChunkRecord.vertexCount = static_cast<std::uint32_t>(chunkMesh.vertices.size());
                rtChunkRecord.indexCount = static_cast<std::uint32_t>(chunkMesh.indices.size());
                rtChunkRecord.geometryResident = !chunkMesh.vertices.empty() && !chunkMesh.indices.empty();
                std::vector<RtVertex> rtChunkVertices;
                rtChunkVertices.reserve(chunkMesh.vertices.size());
                for (const voxelsprout::world::PackedVoxelVertex& vertex : chunkMesh.vertices) {
                    rtChunkVertices.push_back(
                        decodePackedVoxelVertexPosition(vertex.bits, drawRange.offsetX, drawRange.offsetY, drawRange.offsetZ)
                    );
                }
                if (!createRtGeometryBuffers(m_bufferAllocator, rtChunkVertices, chunkMesh.indices, rtChunkRecord.geometry)) {
                    VOX_LOGE("render") << "chunk RT geometry buffer allocation failed for chunk ("
                                       << rtChunkRecord.chunkX << ","
                                       << rtChunkRecord.chunkY << ","
                                       << rtChunkRecord.chunkZ << ")";
                    rtChunkRecord.geometryResident = false;
                    rtChunkRecord.vertexCount = 0;
                    rtChunkRecord.indexCount = 0;
                }
            }

            drawRange.firstIndex = firstIndex;
            // Indices are already rebased into global vertex space.
            drawRange.vertexOffset = 0;
            drawRange.indexCount = static_cast<uint32_t>(chunkMesh.indices.size());
            uploadedVertexCount += chunkMesh.vertices.size();
            uploadedIndexCount += chunkMesh.indices.size();
        }

        const auto previousRecordIt = std::find_if(
            previousRtChunkSceneMetadata.begin(),
            previousRtChunkSceneMetadata.end(),
            [&](const RtChunkSceneMetadata& record) {
                return record.chunkX == rtChunkRecord.chunkX &&
                       record.chunkY == rtChunkRecord.chunkY &&
                       record.chunkZ == rtChunkRecord.chunkZ;
            });
        rtChunkRecord.dirty =
            previousRecordIt == previousRtChunkSceneMetadata.end() ||
            previousRecordIt->vertexCount != rtChunkRecord.vertexCount ||
            previousRecordIt->indexCount != rtChunkRecord.indexCount ||
            previousRecordIt->geometryResident != rtChunkRecord.geometryResident ||
            fullRemesh;
        if (rtChunkRecord.dirty) {
            ++m_rtDirtyChunkCount;
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
            static_cast<VkDeviceSize>(combinedVertices.size() * sizeof(voxelsprout::world::PackedVoxelVertex));
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
            static_cast<VkDeviceSize>(alignof(voxelsprout::world::PackedVoxelVertex)),
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
    const bool startupChunkUpload =
        m_lastGraphicsTimelineValue == 0 &&
        previousChunkReadyTimelineValue == 0 &&
        std::all_of(
            m_frameTimelineValues.begin(),
            m_frameTimelineValues.end(),
            [](uint64_t value) { return value == 0; }
        );
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
        std::array<VkPipelineStageFlags2, 1> transferWaitStages{};
        std::array<uint64_t, 1> transferWaitValues{};
        uint32_t transferWaitCount = 0;
        if (m_lastGraphicsTimelineValue > 0) {
            transferWaitSemaphores[0] = m_renderTimelineSemaphore;
            transferWaitStages[0] = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            transferWaitValues[0] = m_lastGraphicsTimelineValue;
            transferWaitCount = 1;
        }

        std::array<VkSemaphoreSubmitInfo, 1> transferWaitSemaphoreInfos{};
        if (transferWaitCount > 0) {
            transferWaitSemaphoreInfos[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            transferWaitSemaphoreInfos[0].semaphore = transferWaitSemaphores[0];
            transferWaitSemaphoreInfos[0].value = transferWaitValues[0];
            transferWaitSemaphoreInfos[0].stageMask = transferWaitStages[0];
            transferWaitSemaphoreInfos[0].deviceIndex = 0;
        }
        VkSemaphoreSubmitInfo transferSignalSemaphoreInfo{};
        transferSignalSemaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
        transferSignalSemaphoreInfo.semaphore = m_renderTimelineSemaphore;
        transferSignalSemaphoreInfo.value = transferSignalValue;
        transferSignalSemaphoreInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        transferSignalSemaphoreInfo.deviceIndex = 0;
        VkCommandBufferSubmitInfo transferCommandBufferInfo{};
        transferCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        transferCommandBufferInfo.commandBuffer = m_transferCommandBuffer;
        VkSubmitInfo2 transferSubmitInfo{};
        transferSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        transferSubmitInfo.waitSemaphoreInfoCount = transferWaitCount;
        transferSubmitInfo.pWaitSemaphoreInfos =
            transferWaitCount > 0 ? transferWaitSemaphoreInfos.data() : nullptr;
        transferSubmitInfo.commandBufferInfoCount = 1;
        transferSubmitInfo.pCommandBufferInfos = &transferCommandBufferInfo;
        transferSubmitInfo.signalSemaphoreInfoCount = 1;
        transferSubmitInfo.pSignalSemaphoreInfos = &transferSignalSemaphoreInfo;

        const VkResult submitResult = vkQueueSubmit2(m_transferQueue, 1, &transferSubmitInfo, VK_NULL_HANDLE);
        if (submitResult != VK_SUCCESS) {
            logVkFailure("vkQueueSubmit2(transfer)", submitResult);
            cleanupPendingAllocations();
            rollbackChunkDrawState();
            return false;
        }

        if (startupChunkUpload) {
            const VkResult transferWaitResult = vkQueueWaitIdle(m_transferQueue);
            if (transferWaitResult != VK_SUCCESS) {
                logVkFailure("vkQueueWaitIdle(startupChunkUpload)", transferWaitResult);
                cleanupPendingAllocations();
                rollbackChunkDrawState();
                return false;
            }
            m_currentChunkReadyTimelineValue = 0;
            m_pendingTransferTimelineValue = 0;
            m_transferCommandBufferInFlightValue = 0;
        } else {
            m_currentChunkReadyTimelineValue = transferSignalValue;
            m_pendingTransferTimelineValue = transferSignalValue;
            m_transferCommandBufferInFlightValue = transferSignalValue;
        }
    }

    const uint64_t oldChunkReleaseValue = std::max(m_lastGraphicsTimelineValue, previousChunkReadyTimelineValue);
    scheduleBufferRelease(m_chunkVertexBufferHandle, oldChunkReleaseValue);
    scheduleBufferRelease(m_chunkIndexBufferHandle, oldChunkReleaseValue);
    m_chunkVertexBufferHandle = newChunkVertexBufferHandle;
    m_chunkIndexBufferHandle = newChunkIndexBufferHandle;
    newChunkVertexBufferHandle = kInvalidBufferHandle;
    newChunkIndexBufferHandle = kInvalidBufferHandle;
    if (m_rayTracingCapabilityProbe.rayTracingCoreReady) {
        markRayTracingSceneDirty();
        if (rayTracingRuntimeReady() && !rebuildRayTracingScene()) {
            VOX_LOGE("render") << "chunk RT scene rebuild failed";
        }
    }

    VOX_LOGD("render") << "chunk upload queued (ranges=" << m_chunkDrawRanges.size()
                       << ", remeshedChunks=" << remeshedChunkCount
                       << ", meshingMode="
                       << (m_chunkMeshingOptions.mode == voxelsprout::world::MeshingMode::Greedy ? "greedy" : "naive")
                       << ", vertices=" << uploadedVertexCount
                       << ", indices=" << uploadedIndexCount
                       << ", rtResidentChunks=" << m_rtChunkSceneRecords.size()
                       << ", rtDirtyChunks=" << m_rtDirtyChunkCount
                       << (hasChunkCopies
                               ? (", timelineValue=" + std::to_string(transferSignalValue))
                               : ", immediate=true")
                       << ")";
    return true;
}

void RendererBackend::markRayTracingSceneDirty() {
    if (!m_rayTracingCapabilityProbe.rayTracingCoreReady) {
        return;
    }
    m_rtSceneDirty = true;
    refreshShadowStats();
}

void RendererBackend::destroyRayTracingScene() {
    const bool hasExistingScene =
        m_rtTlas.handle != VK_NULL_HANDLE ||
        !m_rtChunkSceneRecords.empty() ||
        !m_rtMagicaBlases.empty();
    if (hasExistingScene && m_device != VK_NULL_HANDLE && m_graphicsQueue != VK_NULL_HANDLE) {
        const VkResult waitResult = vkQueueWaitIdle(m_graphicsQueue);
        if (waitResult != VK_SUCCESS) {
            VOX_LOGW("render") << "destroyRayTracingScene: vkQueueWaitIdle failed before AS destruction";
        }
    }
    auto destroyAs = [&](RtAccelerationStructure& accelerationStructure) {
        if (accelerationStructure.handle != VK_NULL_HANDLE && m_destroyAccelerationStructureKhr != nullptr) {
            m_destroyAccelerationStructureKhr(m_device, accelerationStructure.handle, nullptr);
            accelerationStructure.handle = VK_NULL_HANDLE;
        }
        if (accelerationStructure.storageBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(accelerationStructure.storageBufferHandle);
            accelerationStructure.storageBufferHandle = kInvalidBufferHandle;
        }
        accelerationStructure.deviceAddress = 0;
        accelerationStructure.primitiveCount = 0;
    };

    destroyAs(m_rtTlas);
    if (m_rtTlasInstanceBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_rtTlasInstanceBufferHandle);
        m_rtTlasInstanceBufferHandle = kInvalidBufferHandle;
    }
    for (RtChunkSceneRecord& chunkRecord : m_rtChunkSceneRecords) {
        destroyAs(chunkRecord.blas);
        destroyRtGeometryBuffers(m_bufferAllocator, chunkRecord.geometry);
    }
    for (RtAccelerationStructure& blas : m_rtMagicaBlases) {
        destroyAs(blas);
    }
    m_rtMagicaBlases.clear();
    for (RtGeometryBuffers& geometry : m_rtMagicaGeometries) {
        destroyRtGeometryBuffers(m_bufferAllocator, geometry);
    }
    m_rtMagicaGeometries.clear();
    m_rtSceneDirty = false;
    m_rtSceneBuildCount = 0;
    m_rtBlasBuildCount = 0;
    m_rtTlasBuildCount = 0;
    m_rtDirtyChunkCount = 0;
    m_rtChunkSceneRecords.clear();
    refreshShadowStats();
}

bool RendererBackend::rebuildRayTracingScene() {
    if (!rayTracingRuntimeReady()) {
        refreshShadowStats();
        return false;
    }
    const bool hasExistingScene =
        m_rtTlas.handle != VK_NULL_HANDLE ||
        !m_rtChunkSceneRecords.empty() ||
        !m_rtMagicaBlases.empty();
    if (hasExistingScene) {
        const VkResult waitResult = vkQueueWaitIdle(m_graphicsQueue);
        if (waitResult != VK_SUCCESS) {
            VOX_LOGE("render") << "rebuildRayTracingScene: vkQueueWaitIdle failed before AS rebuild";
            refreshShadowStats();
            return false;
        }
    }
    const VkDeviceAddress scratchAlignment = std::max<VkDeviceAddress>(
        1,
        static_cast<VkDeviceAddress>(m_rayTracingCapabilityProbe.scratchAlignment)
    );
    auto alignDeviceAddress = [&](VkDeviceAddress address) -> VkDeviceAddress {
        const VkDeviceAddress mask = scratchAlignment - 1;
        return (address + mask) & ~mask;
    };

    auto destroyAs = [&](RtAccelerationStructure& accelerationStructure) {
        if (accelerationStructure.handle != VK_NULL_HANDLE && m_destroyAccelerationStructureKhr != nullptr) {
            m_destroyAccelerationStructureKhr(m_device, accelerationStructure.handle, nullptr);
            accelerationStructure.handle = VK_NULL_HANDLE;
        }
        if (accelerationStructure.storageBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(accelerationStructure.storageBufferHandle);
            accelerationStructure.storageBufferHandle = kInvalidBufferHandle;
        }
        accelerationStructure.deviceAddress = 0;
        accelerationStructure.primitiveCount = 0;
    };
    auto createAsStorage = [&](VkAccelerationStructureTypeKHR type,
                               VkDeviceSize size,
                               RtAccelerationStructure& outAccelerationStructure) -> bool {
        destroyAs(outAccelerationStructure);
        BufferCreateDesc storageCreateDesc{};
        storageCreateDesc.size = size;
        storageCreateDesc.usage =
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        storageCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        outAccelerationStructure.storageBufferHandle = m_bufferAllocator.createBuffer(storageCreateDesc);
        if (outAccelerationStructure.storageBufferHandle == kInvalidBufferHandle) {
            return false;
        }
        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.buffer = m_bufferAllocator.getBuffer(outAccelerationStructure.storageBufferHandle);
        createInfo.size = size;
        createInfo.type = type;
        if (m_createAccelerationStructureKhr(m_device, &createInfo, nullptr, &outAccelerationStructure.handle) != VK_SUCCESS) {
            m_bufferAllocator.destroyBuffer(outAccelerationStructure.storageBufferHandle);
            outAccelerationStructure.storageBufferHandle = kInvalidBufferHandle;
            return false;
        }
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure = outAccelerationStructure.handle;
        outAccelerationStructure.deviceAddress =
            m_getAccelerationStructureDeviceAddressKhr(m_device, &addressInfo);
        return outAccelerationStructure.deviceAddress != 0;
    };

    std::vector<std::pair<RtGeometryBuffers*, RtAccelerationStructure*>> buildGeometries;
    for (RtChunkSceneRecord& chunkRecord : m_rtChunkSceneRecords) {
        if (!chunkRecord.geometryResident ||
            chunkRecord.geometry.vertexCount == 0 ||
            chunkRecord.geometry.indexCount == 0) {
            destroyAs(chunkRecord.blas);
            continue;
        }
        buildGeometries.push_back({&chunkRecord.geometry, &chunkRecord.blas});
    }
    if (m_rtMagicaBlases.size() > m_rtMagicaGeometries.size()) {
        for (std::size_t i = m_rtMagicaGeometries.size(); i < m_rtMagicaBlases.size(); ++i) {
            destroyAs(m_rtMagicaBlases[i]);
        }
    }
    m_rtMagicaBlases.resize(m_rtMagicaGeometries.size());
    for (std::size_t i = 0; i < m_rtMagicaGeometries.size(); ++i) {
        if (m_rtMagicaGeometries[i].vertexCount == 0 || m_rtMagicaGeometries[i].indexCount == 0) {
            continue;
        }
        buildGeometries.push_back({&m_rtMagicaGeometries[i], &m_rtMagicaBlases[i]});
    }

    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    commandPoolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS) {
        return false;
    }
    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS) {
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        return false;
    }

    struct ScratchAllocation {
        BufferHandle handle = kInvalidBufferHandle;
        VkDeviceAddress alignedAddress = 0;
    };
    std::vector<ScratchAllocation> scratchBuffers;
    scratchBuffers.reserve(buildGeometries.size() + 1u);
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(buildGeometries.size());
    bool buildOk = true;
    bool commandBufferBegun = false;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    buildOk = vkBeginCommandBuffer(commandBuffer, &beginInfo) == VK_SUCCESS;
    commandBufferBegun = buildOk;

    for (const auto& [geometry, outBlas] : buildGeometries) {
        if (!buildOk) {
            break;
        }
        VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = m_bufferAllocator.getDeviceAddress(geometry->vertexBufferHandle);
        triangles.vertexStride = sizeof(RtVertex);
        triangles.maxVertex = geometry->vertexCount > 0 ? (geometry->vertexCount - 1u) : 0u;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = m_bufferAllocator.getDeviceAddress(geometry->indexBufferHandle);

        VkAccelerationStructureGeometryKHR asGeometry{};
        asGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        asGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        asGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        asGeometry.geometry.triangles = triangles;

        const std::uint32_t primitiveCount = geometry->indexCount / 3u;
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &asGeometry;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        m_getAccelerationStructureBuildSizesKhr(
            m_device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfo,
            &primitiveCount,
            &sizeInfo
        );
        buildOk = createAsStorage(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizeInfo.accelerationStructureSize, *outBlas);
        if (!buildOk) {
            break;
        }
        BufferCreateDesc scratchCreateDesc{};
        scratchCreateDesc.size = sizeInfo.buildScratchSize + scratchAlignment - 1;
        scratchCreateDesc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        scratchCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const BufferHandle scratchHandle = m_bufferAllocator.createBuffer(scratchCreateDesc);
        if (scratchHandle == kInvalidBufferHandle) {
            buildOk = false;
            break;
        }
        const VkDeviceAddress scratchAddress = alignDeviceAddress(m_bufferAllocator.getDeviceAddress(scratchHandle));
        if (scratchAddress == 0) {
            m_bufferAllocator.destroyBuffer(scratchHandle);
            buildOk = false;
            break;
        }
        scratchBuffers.push_back({scratchHandle, scratchAddress});

        buildInfo.dstAccelerationStructure = outBlas->handle;
        buildInfo.scratchData.deviceAddress = scratchAddress;
        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = primitiveCount;
        const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = {&rangeInfo};
        m_cmdBuildAccelerationStructuresKhr(commandBuffer, 1, &buildInfo, rangeInfos);
        outBlas->primitiveCount = primitiveCount;

        VkAccelerationStructureInstanceKHR instance{};
        instance.transform.matrix[0][0] = 1.0f;
        instance.transform.matrix[1][1] = 1.0f;
        instance.transform.matrix[2][2] = 1.0f;
        instance.instanceCustomIndex = static_cast<std::uint32_t>(tlasInstances.size());
        instance.mask = 0xFFu;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = outBlas->deviceAddress;
        tlasInstances.push_back(instance);
    }

    if (buildOk && !buildGeometries.empty()) {
        VkMemoryBarrier2 blasBuildBarrier{};
        blasBuildBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        blasBuildBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        blasBuildBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        blasBuildBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        blasBuildBarrier.dstAccessMask =
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.memoryBarrierCount = 1;
        dependencyInfo.pMemoryBarriers = &blasBuildBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }

    if (buildOk && !tlasInstances.empty()) {
        if (m_rtTlasInstanceBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(m_rtTlasInstanceBufferHandle);
            m_rtTlasInstanceBufferHandle = kInvalidBufferHandle;
        }
        BufferCreateDesc instanceCreateDesc{};
        instanceCreateDesc.size = static_cast<VkDeviceSize>(tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR));
        instanceCreateDesc.usage =
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        instanceCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        instanceCreateDesc.initialData = tlasInstances.data();
        m_rtTlasInstanceBufferHandle = m_bufferAllocator.createBuffer(instanceCreateDesc);
        buildOk = m_rtTlasInstanceBufferHandle != kInvalidBufferHandle;
        if (buildOk) {
            VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
            instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            instancesData.data.deviceAddress = m_bufferAllocator.getDeviceAddress(m_rtTlasInstanceBufferHandle);

            VkAccelerationStructureGeometryKHR tlasGeometry{};
            tlasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            tlasGeometry.geometry.instances = instancesData;

            const std::uint32_t primitiveCount = static_cast<std::uint32_t>(tlasInstances.size());
            VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
            buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            buildInfo.geometryCount = 1;
            buildInfo.pGeometries = &tlasGeometry;

            VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
            sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            m_getAccelerationStructureBuildSizesKhr(
                m_device,
                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                &buildInfo,
                &primitiveCount,
                &sizeInfo
            );
            buildOk = createAsStorage(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizeInfo.accelerationStructureSize, m_rtTlas);
            if (buildOk) {
                BufferCreateDesc scratchCreateDesc{};
                scratchCreateDesc.size = sizeInfo.buildScratchSize + scratchAlignment - 1;
                scratchCreateDesc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                scratchCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                const BufferHandle scratchHandle = m_bufferAllocator.createBuffer(scratchCreateDesc);
                VkDeviceAddress scratchAddress = 0;
                buildOk = scratchHandle != kInvalidBufferHandle;
                if (buildOk) {
                    scratchAddress = alignDeviceAddress(m_bufferAllocator.getDeviceAddress(scratchHandle));
                    buildOk = scratchAddress != 0;
                    if (!buildOk) {
                        m_bufferAllocator.destroyBuffer(scratchHandle);
                    } else {
                        scratchBuffers.push_back({scratchHandle, scratchAddress});
                    }
                }
                if (buildOk) {
                    buildInfo.dstAccelerationStructure = m_rtTlas.handle;
                    buildInfo.scratchData.deviceAddress = scratchAddress;
                    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
                    rangeInfo.primitiveCount = primitiveCount;
                    const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = {&rangeInfo};
                    m_cmdBuildAccelerationStructuresKhr(commandBuffer, 1, &buildInfo, rangeInfos);
                    m_rtTlas.primitiveCount = primitiveCount;
                }
            }
        }
    } else {
        if (m_rtTlasInstanceBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(m_rtTlasInstanceBufferHandle);
            m_rtTlasInstanceBufferHandle = kInvalidBufferHandle;
        }
        destroyAs(m_rtTlas);
    }

    if (buildOk && m_rtTlas.handle != VK_NULL_HANDLE) {
        VkMemoryBarrier2 tlasBuildBarrier{};
        tlasBuildBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        tlasBuildBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        tlasBuildBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        tlasBuildBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        tlasBuildBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.memoryBarrierCount = 1;
        dependencyInfo.pMemoryBarriers = &tlasBuildBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }

    if (buildOk) {
        buildOk = vkEndCommandBuffer(commandBuffer) == VK_SUCCESS;
    } else if (commandBufferBegun) {
        vkEndCommandBuffer(commandBuffer);
    }
    if (buildOk) {
        VkCommandBufferSubmitInfo commandBufferInfo{};
        commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        commandBufferInfo.commandBuffer = commandBuffer;
        VkSubmitInfo2 submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandBufferInfo;
        buildOk = vkQueueSubmit2(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS;
    }
    if (buildOk) {
        buildOk = vkQueueWaitIdle(m_graphicsQueue) == VK_SUCCESS;
    }

    for (const ScratchAllocation& scratchAllocation : scratchBuffers) {
        if (scratchAllocation.handle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(scratchAllocation.handle);
        }
    }
    vkDestroyCommandPool(m_device, commandPool, nullptr);

    if (!buildOk) {
        destroyAs(m_rtTlas);
        for (RtChunkSceneRecord& chunkRecord : m_rtChunkSceneRecords) {
            destroyAs(chunkRecord.blas);
        }
        for (RtAccelerationStructure& blas : m_rtMagicaBlases) {
            destroyAs(blas);
        }
        refreshShadowStats();
        return false;
    }

    m_rtSceneDirty = false;
    ++m_rtSceneBuildCount;
    m_rtBlasBuildCount = static_cast<std::uint32_t>(buildGeometries.size());
    m_rtTlasBuildCount = tlasInstances.empty() ? 0u : 1u;
    refreshShadowStats();
    VOX_LOGI("render") << "ray tracing scene rebuilt: blas=" << m_rtBlasBuildCount
                       << ", tlas=" << m_rtTlasBuildCount
                       << ", instances=" << tlasInstances.size()
                       << ", residentChunks=" << m_rtChunkSceneRecords.size()
                       << ", dirtyChunks=" << m_rtDirtyChunkCount
                       << ", sceneBuilds=" << m_rtSceneBuildCount << "\n";
    return true;
}
} // namespace voxelsprout::render
