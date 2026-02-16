#include "render/backend/vulkan/renderer_backend.h"

#include <cstdint>

namespace voxelsprout::render {

void RendererBackend::drawIndirectChunkRanges(
    VkCommandBuffer commandBuffer,
    std::uint32_t& passDrawCounter,
    const FrameChunkDrawData& frameChunkDrawData
) {
    if (!frameChunkDrawData.canDrawChunksIndirect) {
        return;
    }

    const uint32_t drawCount = frameChunkDrawData.chunkIndirectDrawCount;
    const std::optional<FrameArenaSlice>& indirectSlice = frameChunkDrawData.chunkIndirectSliceOpt;
    if (!indirectSlice.has_value()) {
        return;
    }

    m_debugDrawCallsTotal += drawCount;
    passDrawCounter += drawCount;

    if (m_supportsMultiDrawIndirect) {
        vkCmdDrawIndexedIndirect(
            commandBuffer,
            frameChunkDrawData.chunkIndirectBuffer,
            indirectSlice->offset,
            drawCount,
            sizeof(VkDrawIndexedIndirectCommand)
        );
        return;
    }

    const VkDeviceSize stride = static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand));
    VkDeviceSize drawOffset = indirectSlice->offset;
    for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex) {
        vkCmdDrawIndexedIndirect(commandBuffer, frameChunkDrawData.chunkIndirectBuffer, drawOffset, 1, static_cast<uint32_t>(stride));
        drawOffset += stride;
    }
}

void RendererBackend::drawIndirectShadowChunkRanges(
    VkCommandBuffer commandBuffer,
    std::uint32_t& passDrawCounter,
    std::uint32_t cascadeIndex,
    const FrameChunkDrawData& frameChunkDrawData
) {
    if (cascadeIndex >= kShadowCascadeCount || !frameChunkDrawData.canDrawShadowChunksIndirectByCascade[cascadeIndex]) {
        return;
    }

    const uint32_t drawCount = frameChunkDrawData.shadowCascadeIndirectDrawCounts[cascadeIndex];
    const VkBuffer indirectBuffer = frameChunkDrawData.shadowCascadeIndirectBuffers[cascadeIndex];
    const std::optional<FrameArenaSlice>& indirectSlice = frameChunkDrawData.shadowCascadeIndirectSliceOpts[cascadeIndex];
    if (!indirectSlice.has_value()) {
        return;
    }

    m_debugDrawCallsTotal += drawCount;
    passDrawCounter += drawCount;

    if (m_supportsMultiDrawIndirect) {
        vkCmdDrawIndexedIndirect(
            commandBuffer,
            indirectBuffer,
            indirectSlice->offset,
            drawCount,
            sizeof(VkDrawIndexedIndirectCommand)
        );
        return;
    }

    const VkDeviceSize stride = static_cast<VkDeviceSize>(sizeof(VkDrawIndexedIndirectCommand));
    VkDeviceSize drawOffset = indirectSlice->offset;
    for (uint32_t drawIndex = 0; drawIndex < drawCount; ++drawIndex) {
        vkCmdDrawIndexedIndirect(commandBuffer, indirectBuffer, drawOffset, 1, static_cast<uint32_t>(stride));
        drawOffset += stride;
    }
}

}  // namespace voxelsprout::render
