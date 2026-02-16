#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "sim/network_procedural.h"
#include "world/chunk_mesher.h"

namespace voxelsprout::render {

#include "render/renderer_shared.h"

RendererBackend::FrameChunkDrawData RendererBackend::prepareFrameChunkDrawData(
    const std::vector<voxelsprout::world::Chunk>& chunks,
    std::span<const std::size_t> visibleChunkIndices,
    const std::array<voxelsprout::math::Matrix4, kShadowCascadeCount>& lightViewProjMatrices,
    int cameraChunkX,
    int cameraChunkY,
    int cameraChunkZ
) {
    FrameChunkDrawData out{};
    out.shadowCascadeIndirectBuffers.fill(VK_NULL_HANDLE);
    out.shadowCascadeIndirectDrawCounts.fill(0u);
    out.canDrawShadowChunksIndirectByCascade.fill(false);

    const VkBuffer chunkVertexBuffer = m_bufferAllocator.getBuffer(m_chunkVertexBufferHandle);
    const VkBuffer chunkIndexBuffer = m_bufferAllocator.getBuffer(m_chunkIndexBufferHandle);
    const bool chunkDrawBuffersReady = chunkVertexBuffer != VK_NULL_HANDLE && chunkIndexBuffer != VK_NULL_HANDLE;

    std::vector<ChunkInstanceData> chunkInstanceData;
    chunkInstanceData.reserve(m_chunkDrawRanges.size() + 1);
    chunkInstanceData.push_back(ChunkInstanceData{});
    std::vector<VkDrawIndexedIndirectCommand> chunkIndirectCommands;
    chunkIndirectCommands.reserve(m_chunkDrawRanges.size());

    std::vector<ChunkInstanceData> shadowChunkInstanceData;
    shadowChunkInstanceData.reserve(m_chunkDrawRanges.size() + 1);
    shadowChunkInstanceData.push_back(ChunkInstanceData{});
    std::vector<VkDrawIndexedIndirectCommand> shadowChunkIndirectCommands;
    shadowChunkIndirectCommands.reserve(m_chunkDrawRanges.size());
    std::array<std::vector<VkDrawIndexedIndirectCommand>, kShadowCascadeCount> shadowCascadeIndirectCommands{};
    for (auto& cascadeCommands : shadowCascadeIndirectCommands) {
        cascadeCommands.reserve((m_chunkDrawRanges.size() / kShadowCascadeCount) + 1u);
    }

    const auto appendChunkLods = [&](
                                   std::size_t chunkArrayIndex,
                                   std::vector<ChunkInstanceData>& outInstanceData,
                                   std::vector<VkDrawIndexedIndirectCommand>& outIndirectCommands,
                                   bool countVisibleLodStats
                               ) {
        if (chunkArrayIndex >= chunks.size()) {
            return;
        }
        const voxelsprout::world::Chunk& drawChunk = chunks[chunkArrayIndex];
        const bool allowDetailLods =
            drawChunk.chunkX() == cameraChunkX &&
            drawChunk.chunkY() == cameraChunkY &&
            drawChunk.chunkZ() == cameraChunkZ;
        for (std::size_t lodIndex = 0; lodIndex < voxelsprout::world::kChunkMeshLodCount; ++lodIndex) {
            if (lodIndex > 0 && !allowDetailLods) {
                continue;
            }
            const std::size_t drawRangeIndex = (chunkArrayIndex * voxelsprout::world::kChunkMeshLodCount) + lodIndex;
            if (drawRangeIndex >= m_chunkDrawRanges.size()) {
                continue;
            }
            const ChunkDrawRange& drawRange = m_chunkDrawRanges[drawRangeIndex];
            if (drawRange.indexCount == 0 || !chunkDrawBuffersReady) {
                continue;
            }

            const uint32_t instanceIndex = static_cast<uint32_t>(outInstanceData.size());
            ChunkInstanceData instance{};
            instance.chunkOffset[0] = drawRange.offsetX;
            instance.chunkOffset[1] = drawRange.offsetY;
            instance.chunkOffset[2] = drawRange.offsetZ;
            instance.chunkOffset[3] = 0.0f;
            outInstanceData.push_back(instance);

            VkDrawIndexedIndirectCommand indirectCommand{};
            indirectCommand.indexCount = drawRange.indexCount;
            indirectCommand.instanceCount = 1;
            indirectCommand.firstIndex = drawRange.firstIndex;
            indirectCommand.vertexOffset = drawRange.vertexOffset;
            indirectCommand.firstInstance = instanceIndex;
            outIndirectCommands.push_back(indirectCommand);

            if (countVisibleLodStats) {
                if (lodIndex == 0) {
                    ++m_debugDrawnLod0Ranges;
                } else if (lodIndex == 1) {
                    ++m_debugDrawnLod1Ranges;
                } else {
                    ++m_debugDrawnLod2Ranges;
                }
            }
        }
    };

    if (!visibleChunkIndices.empty()) {
        for (const std::size_t chunkArrayIndex : visibleChunkIndices) {
            appendChunkLods(chunkArrayIndex, chunkInstanceData, chunkIndirectCommands, true);
        }
    } else {
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            appendChunkLods(chunkArrayIndex, chunkInstanceData, chunkIndirectCommands, true);
        }
    }

    const auto appendShadowChunkLods = [&](std::size_t chunkArrayIndex, uint32_t cascadeMask) {
        if (chunkArrayIndex >= chunks.size()) {
            return;
        }
        const voxelsprout::world::Chunk& drawChunk = chunks[chunkArrayIndex];
        const bool allowDetailLods =
            drawChunk.chunkX() == cameraChunkX &&
            drawChunk.chunkY() == cameraChunkY &&
            drawChunk.chunkZ() == cameraChunkZ;
        for (std::size_t lodIndex = 0; lodIndex < voxelsprout::world::kChunkMeshLodCount; ++lodIndex) {
            if (lodIndex > 0 && !allowDetailLods) {
                continue;
            }
            const std::size_t drawRangeIndex = (chunkArrayIndex * voxelsprout::world::kChunkMeshLodCount) + lodIndex;
            if (drawRangeIndex >= m_chunkDrawRanges.size()) {
                continue;
            }
            const ChunkDrawRange& drawRange = m_chunkDrawRanges[drawRangeIndex];
            if (drawRange.indexCount == 0 || !chunkDrawBuffersReady) {
                continue;
            }

            const uint32_t instanceIndex = static_cast<uint32_t>(shadowChunkInstanceData.size());
            ChunkInstanceData instance{};
            instance.chunkOffset[0] = drawRange.offsetX;
            instance.chunkOffset[1] = drawRange.offsetY;
            instance.chunkOffset[2] = drawRange.offsetZ;
            instance.chunkOffset[3] = 0.0f;
            shadowChunkInstanceData.push_back(instance);

            VkDrawIndexedIndirectCommand indirectCommand{};
            indirectCommand.indexCount = drawRange.indexCount;
            indirectCommand.instanceCount = 1;
            indirectCommand.firstIndex = drawRange.firstIndex;
            indirectCommand.vertexOffset = drawRange.vertexOffset;
            indirectCommand.firstInstance = instanceIndex;
            shadowChunkIndirectCommands.push_back(indirectCommand);
            for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
                if ((cascadeMask & (1u << cascadeIndex)) != 0u) {
                    shadowCascadeIndirectCommands[cascadeIndex].push_back(indirectCommand);
                }
            }
        }
    };

    const std::vector<std::uint8_t> shadowCandidateMask = buildShadowCandidateMask(chunks, visibleChunkIndices);
    constexpr float kShadowCasterClipMargin = 0.08f;
    if (!m_shadowDebugSettings.enableOccluderCulling) {
        const uint32_t allCascadeMask = (1u << kShadowCascadeCount) - 1u;
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            appendShadowChunkLods(chunkArrayIndex, allCascadeMask);
        }
    } else {
        for (std::size_t chunkArrayIndex = 0; chunkArrayIndex < chunks.size(); ++chunkArrayIndex) {
            if (!shadowCandidateMask.empty() && shadowCandidateMask[chunkArrayIndex] == 0u) {
                continue;
            }
            const voxelsprout::world::Chunk& chunk = chunks[chunkArrayIndex];
            uint32_t cascadeMask = 0u;
            for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
                if (chunkIntersectsShadowCascadeClip(chunk, lightViewProjMatrices[cascadeIndex], kShadowCasterClipMargin)) {
                    cascadeMask |= (1u << cascadeIndex);
                }
            }
            if (cascadeMask != 0u) {
                appendShadowChunkLods(chunkArrayIndex, cascadeMask);
            }
        }
    }

    const VkDeviceSize chunkInstanceBytes =
        static_cast<VkDeviceSize>(chunkInstanceData.size() * sizeof(ChunkInstanceData));
    if (chunkInstanceBytes > 0) {
        out.chunkInstanceSliceOpt = m_frameArena.allocateUpload(
            chunkInstanceBytes,
            static_cast<VkDeviceSize>(alignof(ChunkInstanceData)),
            FrameArenaUploadKind::InstanceData
        );
        if (out.chunkInstanceSliceOpt.has_value() && out.chunkInstanceSliceOpt->mapped != nullptr) {
            std::memcpy(out.chunkInstanceSliceOpt->mapped, chunkInstanceData.data(), static_cast<size_t>(chunkInstanceBytes));
        } else {
            out.chunkInstanceSliceOpt.reset();
        }
    }

    const VkDeviceSize chunkIndirectBytes =
        static_cast<VkDeviceSize>(chunkIndirectCommands.size() * sizeof(VkDrawIndexedIndirectCommand));
    if (chunkIndirectBytes > 0) {
        out.chunkIndirectSliceOpt = m_frameArena.allocateUpload(
            chunkIndirectBytes,
            static_cast<VkDeviceSize>(alignof(VkDrawIndexedIndirectCommand)),
            FrameArenaUploadKind::Unknown
        );
        if (out.chunkIndirectSliceOpt.has_value() && out.chunkIndirectSliceOpt->mapped != nullptr) {
            std::memcpy(
                out.chunkIndirectSliceOpt->mapped,
                chunkIndirectCommands.data(),
                static_cast<size_t>(chunkIndirectBytes)
            );
        } else {
            out.chunkIndirectSliceOpt.reset();
        }
    }

    const VkDeviceSize shadowChunkInstanceBytes =
        static_cast<VkDeviceSize>(shadowChunkInstanceData.size() * sizeof(ChunkInstanceData));
    if (shadowChunkInstanceBytes > 0) {
        out.shadowChunkInstanceSliceOpt = m_frameArena.allocateUpload(
            shadowChunkInstanceBytes,
            static_cast<VkDeviceSize>(alignof(ChunkInstanceData)),
            FrameArenaUploadKind::InstanceData
        );
        if (out.shadowChunkInstanceSliceOpt.has_value() && out.shadowChunkInstanceSliceOpt->mapped != nullptr) {
            std::memcpy(
                out.shadowChunkInstanceSliceOpt->mapped,
                shadowChunkInstanceData.data(),
                static_cast<size_t>(shadowChunkInstanceBytes)
            );
        } else {
            out.shadowChunkInstanceSliceOpt.reset();
        }
    }

    out.chunkInstanceBuffer =
        out.chunkInstanceSliceOpt.has_value() ? m_bufferAllocator.getBuffer(out.chunkInstanceSliceOpt->buffer) : VK_NULL_HANDLE;
    out.chunkIndirectBuffer =
        out.chunkIndirectSliceOpt.has_value() ? m_bufferAllocator.getBuffer(out.chunkIndirectSliceOpt->buffer) : VK_NULL_HANDLE;
    out.shadowChunkInstanceBuffer =
        out.shadowChunkInstanceSliceOpt.has_value()
            ? m_bufferAllocator.getBuffer(out.shadowChunkInstanceSliceOpt->buffer)
            : VK_NULL_HANDLE;

    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        const VkDeviceSize shadowCascadeIndirectBytes = static_cast<VkDeviceSize>(
            shadowCascadeIndirectCommands[cascadeIndex].size() * sizeof(VkDrawIndexedIndirectCommand)
        );
        if (shadowCascadeIndirectBytes == 0) {
            continue;
        }
        out.shadowCascadeIndirectSliceOpts[cascadeIndex] = m_frameArena.allocateUpload(
            shadowCascadeIndirectBytes,
            static_cast<VkDeviceSize>(alignof(VkDrawIndexedIndirectCommand)),
            FrameArenaUploadKind::Unknown
        );
        if (!out.shadowCascadeIndirectSliceOpts[cascadeIndex].has_value() ||
            out.shadowCascadeIndirectSliceOpts[cascadeIndex]->mapped == nullptr) {
            out.shadowCascadeIndirectSliceOpts[cascadeIndex].reset();
            continue;
        }
        std::memcpy(
            out.shadowCascadeIndirectSliceOpts[cascadeIndex]->mapped,
            shadowCascadeIndirectCommands[cascadeIndex].data(),
            static_cast<size_t>(shadowCascadeIndirectBytes)
        );
        out.shadowCascadeIndirectBuffers[cascadeIndex] =
            m_bufferAllocator.getBuffer(out.shadowCascadeIndirectSliceOpts[cascadeIndex]->buffer);
        out.shadowCascadeIndirectDrawCounts[cascadeIndex] =
            static_cast<uint32_t>(shadowCascadeIndirectCommands[cascadeIndex].size());
    }

    out.chunkIndirectDrawCount = static_cast<uint32_t>(chunkIndirectCommands.size());
    m_debugChunkIndirectCommandCount = out.chunkIndirectDrawCount;
    out.canDrawChunksIndirect =
        out.chunkIndirectDrawCount > 0 &&
        out.chunkInstanceSliceOpt.has_value() &&
        out.chunkIndirectSliceOpt.has_value() &&
        out.chunkInstanceBuffer != VK_NULL_HANDLE &&
        out.chunkIndirectBuffer != VK_NULL_HANDLE &&
        chunkDrawBuffersReady;
    for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
        out.canDrawShadowChunksIndirectByCascade[cascadeIndex] =
            out.shadowCascadeIndirectDrawCounts[cascadeIndex] > 0 &&
            out.shadowChunkInstanceSliceOpt.has_value() &&
            out.shadowCascadeIndirectSliceOpts[cascadeIndex].has_value() &&
            out.shadowChunkInstanceBuffer != VK_NULL_HANDLE &&
            out.shadowCascadeIndirectBuffers[cascadeIndex] != VK_NULL_HANDLE &&
            chunkDrawBuffersReady;
    }

    return out;
}

} // namespace voxelsprout::render
