#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include "sim/network_procedural.h"
#include "render/backend/vulkan/frame_graph_runtime.h"

namespace odai::render {

#include "render/renderer_shared.h"

void RendererBackend::recordNormalDepthPrepass(const FrameExecutionContext& context, const PrepassInputs& inputs) {
    VkCommandBuffer commandBuffer = context.commandBuffer;
    VkQueryPool gpuTimestampQueryPool = context.gpuTimestampQueryPool;
    CoreFrameGraphOrderValidator& coreFramePassOrderValidator = *context.frameOrderValidator;
    const CoreFrameGraphPlan& coreFrameGraphPlan = *context.frameGraphPlan;
    const uint32_t aoFrameIndex = context.aoFrameIndex;
    const uint32_t imageIndex = context.imageIndex;
    const VkExtent2D aoExtent = context.aoExtent;
    const VkViewport& aoViewport = context.aoViewport;
    const VkRect2D& aoScissor = context.aoScissor;
    // Voxel chunk inputs: consumed by the chunk draw below, mirroring the main pass.
    // The magica/pipe prepass inputs remain on PrepassInputs but stay unconsumed —
    // those draws belong to the prior factory sim and have no caller left.
    const FrameChunkDrawData& frameChunkDrawData = *inputs.frameChunkDrawData;
    const std::optional<FrameArenaSlice>& chunkInstanceSliceOpt = *inputs.chunkInstanceSliceOpt;
    const VkBuffer chunkInstanceBuffer = inputs.chunkInstanceBuffer;
    const VkBuffer chunkVertexBuffer = inputs.chunkVertexBuffer;
    const VkBuffer chunkIndexBuffer = inputs.chunkIndexBuffer;
    const VkBuffer importedVertexBuffer = inputs.importedVertexBuffer;
    const VkBuffer importedIndexBuffer = inputs.importedIndexBuffer;
    const std::span<const ImportedMeshDraw> importedMeshDraws = inputs.importedMeshDraws;
    const std::uint32_t importedTerrainDrawCount = inputs.importedTerrainDrawCount;
    const VkBuffer importedActorVertexBuffer = inputs.importedActorVertexBuffer;
    const VkDeviceSize importedActorVertexOffset = inputs.importedActorVertexOffset;
    const VkBuffer importedActorIndexBuffer = inputs.importedActorIndexBuffer;
    const VkDeviceSize importedActorIndexOffset = inputs.importedActorIndexOffset;
    const std::span<const ImportedMeshDraw> importedActorMeshDraws = inputs.importedActorMeshDraws;
    const std::span<const ImportedMeshDraw> skinnedActorMeshDraws = inputs.skinnedActorMeshDraws;

    auto countDrawCalls = [&](std::uint32_t& passCounter, std::uint32_t drawCount) {
        passCounter += drawCount;
        m_debugDrawCallsTotal += drawCount;
    };
    const auto writeGpuTimestampTop = [&](uint32_t queryIndex) {
        if (gpuTimestampQueryPool == VK_NULL_HANDLE) {
            return;
        }
        vkCmdWriteTimestamp2(
            commandBuffer,
            VK_PIPELINE_STAGE_2_NONE,
            gpuTimestampQueryPool,
            queryIndex
        );
    };
    const auto writeGpuTimestampBottom = [&](uint32_t queryIndex) {
        if (gpuTimestampQueryPool == VK_NULL_HANDLE) {
            return;
        }
        vkCmdWriteTimestamp2(
            commandBuffer,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            gpuTimestampQueryPool,
            queryIndex
        );
    };

    VkClearValue normalDepthClearValue{};
    normalDepthClearValue.color.float32[0] = 0.5f;
    normalDepthClearValue.color.float32[1] = 0.5f;
    normalDepthClearValue.color.float32[2] = 0.5f;
    normalDepthClearValue.color.float32[3] = 0.0f;

    VkClearValue aoDepthClearValue{};
    aoDepthClearValue.depthStencil.depth = 0.0f;
    aoDepthClearValue.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo normalDepthColorAttachment{};
    normalDepthColorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    normalDepthColorAttachment.imageView = m_normalDepthImageViews[aoFrameIndex];
    normalDepthColorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    normalDepthColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    normalDepthColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    normalDepthColorAttachment.clearValue = normalDepthClearValue;

    VkRenderingAttachmentInfo aoDepthAttachment{};
    aoDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    aoDepthAttachment.imageView = m_aoDepthImageViews[imageIndex];
    aoDepthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    aoDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    aoDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    aoDepthAttachment.clearValue = aoDepthClearValue;

    VkRenderingInfo normalDepthRenderingInfo{};
    normalDepthRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    normalDepthRenderingInfo.renderArea.offset = {0, 0};
    normalDepthRenderingInfo.renderArea.extent = aoExtent;
    normalDepthRenderingInfo.layerCount = 1;
    normalDepthRenderingInfo.colorAttachmentCount = 1;
    normalDepthRenderingInfo.pColorAttachments = &normalDepthColorAttachment;
    normalDepthRenderingInfo.pDepthAttachment = &aoDepthAttachment;

    writeGpuTimestampTop(kGpuTimestampQueryPrepassStart);
    coreFramePassOrderValidator.markPassEntered(coreFrameGraphPlan.prepass, "prepass");
    beginDebugLabel(commandBuffer, "Pass: Normal+Depth Prepass", 0.20f, 0.30f, 0.40f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &normalDepthRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &aoViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &aoScissor);

    // Begin/clear/end even when nothing needs the result: the attachments must
    // still end the frame in the layout the descriptor set was written for, and
    // clearing is a fraction of the cost of re-rendering the scene.
    if (!inputs.normalDepthNeeded) {
        vkCmdEndRendering(commandBuffer);
        writeGpuTimestampBottom(kGpuTimestampQueryPrepassEnd);
        return;
    }

    // Voxel chunks (VoxelCraft). This is what gives ambient occlusion something to
    // occlude against in a voxel world — without it the AO input buffer holds only
    // imported statics and skinned actors, and every AO mode reads flat where the
    // terrain is. Guards mirror the main pass exactly: games with no voxel chunks
    // produce no indirect commands, so canDrawChunksIndirect is false and the whole
    // block is skipped.
    if (m_voxelNormalDepthPipeline != VK_NULL_HANDLE &&
        frameChunkDrawData.canDrawChunksIndirect &&
        chunkVertexBuffer != VK_NULL_HANDLE &&
        chunkIndexBuffer != VK_NULL_HANDLE &&
        chunkInstanceBuffer != VK_NULL_HANDLE &&
        chunkInstanceSliceOpt.has_value()) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_voxelNormalDepthPipeline);
        bindGraphicsDescriptorBuffers(commandBuffer);
        const VkBuffer voxelVertexBuffers[2] = {chunkVertexBuffer, chunkInstanceBuffer};
        const VkDeviceSize voxelVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, voxelVertexBuffers, voxelVertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, chunkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // Per-chunk offsets ride the instance buffer; the push constant block stays
        // zeroed so the shader's chunkOffset/cascadeData path matches the main pass.
        ChunkPushConstants chunkPushConstants{};
        vkCmdPushConstants(
            commandBuffer,
            m_pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(ChunkPushConstants),
            &chunkPushConstants
        );
        drawIndirectChunkRanges(commandBuffer, m_debugDrawCallsPrepass, frameChunkDrawData);
    }

    // Opaque normal+depth for SSAO. Gated on the imported-static normal-depth pipeline
    // (the strategy map's settlements/units/grid); the prior game's magica normal-depth
    // draws remain removed.
    if (m_importedStaticNormalDepthPipeline != VK_NULL_HANDLE) {
        if (m_importedStaticNormalDepthPipeline != VK_NULL_HANDLE &&
            importedVertexBuffer != VK_NULL_HANDLE &&
            importedIndexBuffer != VK_NULL_HANDLE &&
            !importedMeshDraws.empty()) {
            const std::size_t terrainDrawCount = std::min<std::size_t>(importedTerrainDrawCount, importedMeshDraws.size());
            const std::size_t staticDrawStart = terrainDrawCount;
            const VkBuffer importedVertexBuffers[1] = {importedVertexBuffer};
            const VkDeviceSize importedVertexOffsets[1] = {0};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_importedStaticNormalDepthPipeline);
            bindGraphicsDescriptorBuffers(commandBuffer);
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, importedVertexBuffers, importedVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, importedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            for (std::size_t drawIndex = 0; drawIndex < importedMeshDraws.size(); ++drawIndex) {
                if ((drawIndex < terrainDrawCount && !m_debugShowImportedTerrain) ||
                    (drawIndex >= staticDrawStart && !m_debugShowImportedStatics)) {
                    continue;
                }
                const ImportedMeshDraw& importedDraw = importedMeshDraws[drawIndex];
                if (importedDraw.blended) {
                    continue;  // no depth/normal contribution from blended surfaces
                }
                countDrawCalls(m_debugDrawCallsPrepass, 1);
                vkCmdDrawIndexed(
                    commandBuffer, importedDraw.indexCount, 1, importedDraw.firstIndex,
                    importedDraw.vertexOffset, 0);
            }
        }
        if (m_importedStaticNormalDepthPipeline != VK_NULL_HANDLE &&
            importedActorVertexBuffer != VK_NULL_HANDLE &&
            importedActorIndexBuffer != VK_NULL_HANDLE &&
            !importedActorMeshDraws.empty() &&
            m_debugShowImportedStatics) {
            const VkBuffer importedVertexBuffers[1] = {importedActorVertexBuffer};
            const VkDeviceSize importedVertexOffsets[1] = {importedActorVertexOffset};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_importedStaticNormalDepthPipeline);
            bindGraphicsDescriptorBuffers(commandBuffer);
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, importedVertexBuffers, importedVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, importedActorIndexBuffer, importedActorIndexOffset, VK_INDEX_TYPE_UINT32);
            for (const ImportedMeshDraw& importedDraw : importedActorMeshDraws) {
                countDrawCalls(m_debugDrawCallsPrepass, 1);
                vkCmdDrawIndexed(
                    commandBuffer, importedDraw.indexCount, 1, importedDraw.firstIndex,
                    importedDraw.vertexOffset, 0);
            }
        }
        // GPU-skinned actors (Dragon Age touchstone) -- same pipeline as the
        // CPU-skinned block above. Up to kMaxSkinnedInstances independent
        // instance slots means each draw may come from a different
        // vertex/index buffer, so bind is resolved per-draw rather than once
        // for the whole span (see skinning_resources.cc).
        if (m_importedStaticNormalDepthPipeline != VK_NULL_HANDLE &&
            !skinnedActorMeshDraws.empty() &&
            m_debugShowImportedStatics) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_importedStaticNormalDepthPipeline);
            bindGraphicsDescriptorBuffers(commandBuffer);
            VkBuffer boundSkinnedVertexBuffer = VK_NULL_HANDLE;
            VkBuffer boundSkinnedIndexBuffer = VK_NULL_HANDLE;
            for (const ImportedMeshDraw& skinnedDraw : skinnedActorMeshDraws) {
                const VkBuffer drawVertexBuffer = m_bufferAllocator.getBuffer(skinnedDraw.vertexBufferHandle);
                const VkBuffer drawIndexBuffer = m_bufferAllocator.getBuffer(skinnedDraw.indexBufferHandle);
                if (drawVertexBuffer == VK_NULL_HANDLE || drawIndexBuffer == VK_NULL_HANDLE) {
                    continue;
                }
                if (drawVertexBuffer != boundSkinnedVertexBuffer) {
                    const VkBuffer skinnedVertexBuffers[1] = {drawVertexBuffer};
                    const VkDeviceSize skinnedVertexOffsets[1] = {0};
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, skinnedVertexBuffers, skinnedVertexOffsets);
                    boundSkinnedVertexBuffer = drawVertexBuffer;
                }
                if (drawIndexBuffer != boundSkinnedIndexBuffer) {
                    vkCmdBindIndexBuffer(commandBuffer, drawIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
                    boundSkinnedIndexBuffer = drawIndexBuffer;
                }
                countDrawCalls(m_debugDrawCallsPrepass, 1);
                vkCmdDrawIndexed(commandBuffer, skinnedDraw.indexCount, 1, skinnedDraw.firstIndex, 0, 0);
            }
        }
        // Transparent water samples this prepass to estimate the opaque scene below it.
    }

    // (removed) pipe / belt / transport normal-depth prepass draws — legacy factory sim.

    // (removed) grass billboard normal-depth prepass draw. This outlived the main and
    // shadow draws by one commit: with the scatter gone it was inert (zero instances),
    // but had instances returned it would have written normal/depth for geometry the
    // main pass no longer shades, feeding SSAO an occluder with no surface.
    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryPrepassEnd);

    transitionImageLayout(
        commandBuffer,
        m_normalDepthImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
}

}  // namespace odai::render
