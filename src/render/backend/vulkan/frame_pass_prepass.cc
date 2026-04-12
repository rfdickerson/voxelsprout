#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include "sim/network_procedural.h"
#include "render/backend/vulkan/frame_graph_runtime.h"

namespace voxelsprout::render {

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
    const BoundDescriptorSets& boundDescriptorSets = *context.boundDescriptorSets;
    const uint32_t mvpDynamicOffset = context.mvpDynamicOffset;
    const FrameChunkDrawData& frameChunkDrawData = *inputs.frameChunkDrawData;
    const std::optional<FrameArenaSlice>& chunkInstanceSliceOpt = *inputs.chunkInstanceSliceOpt;
    const VkBuffer chunkInstanceBuffer = inputs.chunkInstanceBuffer;
    const VkBuffer chunkVertexBuffer = inputs.chunkVertexBuffer;
    const VkBuffer chunkIndexBuffer = inputs.chunkIndexBuffer;
    const bool canDrawMagica = inputs.canDrawMagica;
    const std::span<const ReadyMagicaDraw> readyMagicaDraws = inputs.readyMagicaDraws;
    const uint32_t pipeInstanceCount = inputs.pipeInstanceCount;
    const std::optional<FrameArenaSlice>& pipeInstanceSliceOpt = *inputs.pipeInstanceSliceOpt;
    const uint32_t transportInstanceCount = inputs.transportInstanceCount;
    const std::optional<FrameArenaSlice>& transportInstanceSliceOpt = *inputs.transportInstanceSliceOpt;
    const uint32_t beltCargoInstanceCount = inputs.beltCargoInstanceCount;
    const std::optional<FrameArenaSlice>& beltCargoInstanceSliceOpt = *inputs.beltCargoInstanceSliceOpt;

    const uint32_t boundDescriptorSetCount = boundDescriptorSets.count;
    auto countDrawCalls = [&](std::uint32_t& passCounter, std::uint32_t drawCount) {
        passCounter += drawCount;
        m_debugDrawCallsTotal += drawCount;
    };
    const auto writeGpuTimestampTop = [&](uint32_t queryIndex) {
        if (gpuTimestampQueryPool == VK_NULL_HANDLE) {
            return;
        }
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            gpuTimestampQueryPool,
            queryIndex
        );
    };
    const auto writeGpuTimestampBottom = [&](uint32_t queryIndex) {
        if (gpuTimestampQueryPool == VK_NULL_HANDLE) {
            return;
        }
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
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

    if (m_voxelNormalDepthPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_voxelNormalDepthPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_pipelineLayout,
            0,
            boundDescriptorSetCount,
            boundDescriptorSets.sets.data(),
            1,
            &mvpDynamicOffset
        );
        if (frameChunkDrawData.canDrawChunksIndirect) {
            const VkBuffer voxelVertexBuffers[2] = {chunkVertexBuffer, chunkInstanceBuffer};
            const VkDeviceSize voxelVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, voxelVertexBuffers, voxelVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, chunkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

            ChunkPushConstants chunkPushConstants{};
            chunkPushConstants.chunkOffset[0] = 0.0f;
            chunkPushConstants.chunkOffset[1] = 0.0f;
            chunkPushConstants.chunkOffset[2] = 0.0f;
            chunkPushConstants.chunkOffset[3] = 0.0f;
            chunkPushConstants.cascadeData[0] = 0.0f;
            chunkPushConstants.cascadeData[1] = 0.0f;
            chunkPushConstants.cascadeData[2] = 0.0f;
            chunkPushConstants.cascadeData[3] = 0.0f;
            vkCmdPushConstants(
                commandBuffer,
                m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                sizeof(ChunkPushConstants),
                &chunkPushConstants
            );
            drawIndirectChunkRanges(commandBuffer, m_debugDrawCallsPrepass, frameChunkDrawData);
        }
        if (canDrawMagica) {
            for (const ReadyMagicaDraw& magicaDraw : readyMagicaDraws) {
                const VkBuffer magicaVertexBuffers[2] = {magicaDraw.vertexBuffer, chunkInstanceBuffer};
                const VkDeviceSize magicaVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
                vkCmdBindVertexBuffers(commandBuffer, 0, 2, magicaVertexBuffers, magicaVertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, magicaDraw.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

                ChunkPushConstants magicaPushConstants{};
                magicaPushConstants.chunkOffset[0] = magicaDraw.offsetX;
                magicaPushConstants.chunkOffset[1] = magicaDraw.offsetY;
                magicaPushConstants.chunkOffset[2] = magicaDraw.offsetZ;
                magicaPushConstants.chunkOffset[3] = 0.0f;
                magicaPushConstants.cascadeData[0] = 0.0f;
                magicaPushConstants.cascadeData[1] = 0.0f;
                magicaPushConstants.cascadeData[2] = 0.0f;
                magicaPushConstants.cascadeData[3] = 0.0f;
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &magicaPushConstants
                );
                countDrawCalls(m_debugDrawCallsPrepass, 1);
                vkCmdDrawIndexed(commandBuffer, magicaDraw.indexCount, 1, 0, 0, 0);
            }
        }
    }

    if (m_pipeNormalDepthPipeline != VK_NULL_HANDLE) {
        auto drawNormalDepthInstances = [&](BufferHandle vertexHandle,
                                            BufferHandle indexHandle,
                                            uint32_t indexCount,
                                            uint32_t instanceCount,
                                            const std::optional<FrameArenaSlice>& instanceSlice) {
            if (instanceCount == 0 || !instanceSlice.has_value() || indexCount == 0) {
                return;
            }
            const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(vertexHandle);
            const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(indexHandle);
            const VkBuffer instanceBuffer = m_bufferAllocator.getBuffer(instanceSlice->buffer);
            if (vertexBuffer == VK_NULL_HANDLE || indexBuffer == VK_NULL_HANDLE || instanceBuffer == VK_NULL_HANDLE) {
                return;
            }
            const VkBuffer vertexBuffers[2] = {vertexBuffer, instanceBuffer};
            const VkDeviceSize vertexOffsets[2] = {0, instanceSlice->offset};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeNormalDepthPipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pipelineLayout,
                0,
                boundDescriptorSetCount,
                boundDescriptorSets.sets.data(),
                1,
                &mvpDynamicOffset
            );
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsPrepass, 1);
            vkCmdDrawIndexed(commandBuffer, indexCount, instanceCount, 0, 0, 0);
        };
        drawNormalDepthInstances(
            m_pipeVertexBufferHandle,
            m_pipeIndexBufferHandle,
            m_pipeIndexCount,
            pipeInstanceCount,
            pipeInstanceSliceOpt
        );
        drawNormalDepthInstances(
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            m_transportIndexCount,
            transportInstanceCount,
            transportInstanceSliceOpt
        );
        drawNormalDepthInstances(
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            m_transportIndexCount,
            beltCargoInstanceCount,
            beltCargoInstanceSliceOpt
        );
    }
    if (m_grassBillboardNormalDepthPipeline != VK_NULL_HANDLE &&
        m_grassBillboardIndexCount > 0 &&
        m_grassBillboardInstanceCount > 0 &&
        m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
        const VkBuffer grassVertexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardVertexBufferHandle);
        const VkBuffer grassIndexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardIndexBufferHandle);
        const VkBuffer grassInstanceBuffer = m_bufferAllocator.getBuffer(m_grassBillboardInstanceBufferHandle);
        if (grassVertexBuffer != VK_NULL_HANDLE &&
            grassIndexBuffer != VK_NULL_HANDLE &&
            grassInstanceBuffer != VK_NULL_HANDLE) {
            const VkBuffer vertexBuffers[2] = {grassVertexBuffer, grassInstanceBuffer};
            const VkDeviceSize vertexOffsets[2] = {0, 0};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_grassBillboardNormalDepthPipeline);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pipelineLayout,
                0,
                boundDescriptorSetCount,
                boundDescriptorSets.sets.data(),
                1,
                &mvpDynamicOffset
            );
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, vertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, grassIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsPrepass, 1);
            vkCmdDrawIndexed(commandBuffer, m_grassBillboardIndexCount, m_grassBillboardInstanceCount, 0, 0, 0);
        }
    }
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

}  // namespace voxelsprout::render
