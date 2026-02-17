#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <string>

#include "sim/network_procedural.h"
#include "render/backend/vulkan/frame_graph_runtime.h"

namespace voxelsprout::render {

#include "render/renderer_shared.h"

void RendererBackend::recordShadowAtlasPass(const FrameExecutionContext& context, const ShadowPassInputs& inputs) {
    VkCommandBuffer commandBuffer = context.commandBuffer;
    VkQueryPool gpuTimestampQueryPool = context.gpuTimestampQueryPool;
    const BoundDescriptorSets& boundDescriptorSets = *context.boundDescriptorSets;
    const uint32_t mvpDynamicOffset = context.mvpDynamicOffset;
    CoreFrameGraphOrderValidator& coreFramePassOrderValidator = *context.frameOrderValidator;
    const CoreFrameGraphPlan& coreFrameGraphPlan = *context.frameGraphPlan;
    const FrameChunkDrawData& frameChunkDrawData = *inputs.frameChunkDrawData;
    const std::optional<FrameArenaSlice>& chunkInstanceSliceOpt = *inputs.chunkInstanceSliceOpt;
    const std::optional<FrameArenaSlice>& shadowChunkInstanceSliceOpt = *inputs.shadowChunkInstanceSliceOpt;
    const VkBuffer chunkInstanceBuffer = inputs.chunkInstanceBuffer;
    const VkBuffer shadowChunkInstanceBuffer = inputs.shadowChunkInstanceBuffer;
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

    writeGpuTimestampTop(kGpuTimestampQueryShadowStart);
    coreFramePassOrderValidator.markPassEntered(coreFrameGraphPlan.shadow, "shadow");
    beginDebugLabel(commandBuffer, "Pass: Shadow Atlas", 0.28f, 0.22f, 0.22f, 1.0f);
    const bool shadowInitialized = m_shadowDepthInitialized;
    transitionImageLayout(
        commandBuffer,
        m_shadowDepthImage,
        shadowInitialized ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        shadowInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        shadowInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        0,
        1
    );

    VkClearValue shadowDepthClearValue{};
    shadowDepthClearValue.depthStencil.depth = 0.0f;
    shadowDepthClearValue.depthStencil.stencil = 0;

    if (m_shadowPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);
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

        for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
            if (m_cmdInsertDebugUtilsLabel != nullptr) {
                const std::string cascadeLabel = "Shadow Cascade " + std::to_string(cascadeIndex);
                insertDebugLabel(commandBuffer, cascadeLabel.c_str(), 0.48f, 0.32f, 0.32f, 1.0f);
            }
            const ShadowAtlasRect atlasRect = kShadowAtlasRects[cascadeIndex];
            VkViewport shadowViewport{};
            shadowViewport.x = static_cast<float>(atlasRect.x);
            shadowViewport.y = static_cast<float>(atlasRect.y);
            shadowViewport.width = static_cast<float>(atlasRect.size);
            shadowViewport.height = static_cast<float>(atlasRect.size);
            shadowViewport.minDepth = 0.0f;
            shadowViewport.maxDepth = 1.0f;

            VkRect2D shadowScissor{};
            shadowScissor.offset = {
                static_cast<int32_t>(atlasRect.x),
                static_cast<int32_t>(atlasRect.y)
            };
            shadowScissor.extent = {atlasRect.size, atlasRect.size};

            VkRenderingAttachmentInfo shadowDepthAttachment{};
            shadowDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            shadowDepthAttachment.imageView = m_shadowDepthImageView;
            shadowDepthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            shadowDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            shadowDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            shadowDepthAttachment.clearValue = shadowDepthClearValue;

            VkRenderingInfo shadowRenderingInfo{};
            shadowRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
            shadowRenderingInfo.renderArea.offset = shadowScissor.offset;
            shadowRenderingInfo.renderArea.extent = shadowScissor.extent;
            shadowRenderingInfo.layerCount = 1;
            shadowRenderingInfo.colorAttachmentCount = 0;
            shadowRenderingInfo.pDepthAttachment = &shadowDepthAttachment;

            vkCmdBeginRendering(commandBuffer, &shadowRenderingInfo);
            vkCmdSetViewport(commandBuffer, 0, 1, &shadowViewport);
            vkCmdSetScissor(commandBuffer, 0, 1, &shadowScissor);
            const float cascadeF = static_cast<float>(cascadeIndex);
            const float constantBias =
                m_shadowDebugSettings.casterConstantBiasBase +
                (m_shadowDebugSettings.casterConstantBiasCascadeScale * cascadeF);
            const float slopeBias =
                m_shadowDebugSettings.casterSlopeBiasBase +
                (m_shadowDebugSettings.casterSlopeBiasCascadeScale * cascadeF);
            // Reverse-Z uses GREATER depth tests, so flip bias sign.
            vkCmdSetDepthBias(commandBuffer, -constantBias, 0.0f, -slopeBias);

            if (cascadeIndex < kShadowCascadeCount) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);
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
                const VkBuffer voxelVertexBuffers[2] = {chunkVertexBuffer, shadowChunkInstanceBuffer};
                const VkDeviceSize voxelVertexOffsets[2] = {0, shadowChunkInstanceSliceOpt->offset};
                vkCmdBindVertexBuffers(commandBuffer, 0, 2, voxelVertexBuffers, voxelVertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, chunkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

                ChunkPushConstants chunkPushConstants{};
                chunkPushConstants.chunkOffset[0] = 0.0f;
                chunkPushConstants.chunkOffset[1] = 0.0f;
                chunkPushConstants.chunkOffset[2] = 0.0f;
                chunkPushConstants.chunkOffset[3] = 0.0f;
                chunkPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
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
                drawIndirectShadowChunkRanges(commandBuffer, m_debugDrawCallsShadow, cascadeIndex, frameChunkDrawData);
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
                    magicaPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
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
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(commandBuffer, magicaDraw.indexCount, 1, 0, 0, 0);
                }
            }

            if (m_pipeShadowPipeline != VK_NULL_HANDLE) {
                auto drawShadowInstances = [&](BufferHandle vertexHandle,
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
                    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeShadowPipeline);
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

                    ChunkPushConstants pipeShadowPushConstants{};
                    pipeShadowPushConstants.chunkOffset[0] = 0.0f;
                    pipeShadowPushConstants.chunkOffset[1] = 0.0f;
                    pipeShadowPushConstants.chunkOffset[2] = 0.0f;
                    pipeShadowPushConstants.chunkOffset[3] = 0.0f;
                    pipeShadowPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                    pipeShadowPushConstants.cascadeData[1] = 0.0f;
                    pipeShadowPushConstants.cascadeData[2] = 0.0f;
                    pipeShadowPushConstants.cascadeData[3] = 0.0f;
                    vkCmdPushConstants(
                        commandBuffer,
                        m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT,
                        0,
                        sizeof(ChunkPushConstants),
                        &pipeShadowPushConstants
                    );
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(commandBuffer, indexCount, instanceCount, 0, 0, 0);
                };
                drawShadowInstances(
                    m_pipeVertexBufferHandle,
                    m_pipeIndexBufferHandle,
                    m_pipeIndexCount,
                    pipeInstanceCount,
                    pipeInstanceSliceOpt
                );
                drawShadowInstances(
                    m_transportVertexBufferHandle,
                    m_transportIndexBufferHandle,
                    m_transportIndexCount,
                    transportInstanceCount,
                    transportInstanceSliceOpt
                );
                drawShadowInstances(
                    m_transportVertexBufferHandle,
                    m_transportIndexBufferHandle,
                    m_transportIndexCount,
                    beltCargoInstanceCount,
                    beltCargoInstanceSliceOpt
                );
            }

            const uint32_t grassShadowCascadeCount = static_cast<uint32_t>(std::clamp(
                m_shadowDebugSettings.grassShadowCascadeCount,
                0,
                static_cast<int>(kShadowCascadeCount)
            ));
            if (cascadeIndex < grassShadowCascadeCount &&
                m_grassBillboardShadowPipeline != VK_NULL_HANDLE &&
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
                    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_grassBillboardShadowPipeline);
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

                    ChunkPushConstants grassShadowPushConstants{};
                    grassShadowPushConstants.chunkOffset[0] = 0.0f;
                    grassShadowPushConstants.chunkOffset[1] = 0.0f;
                    grassShadowPushConstants.chunkOffset[2] = 0.0f;
                    grassShadowPushConstants.chunkOffset[3] = 0.0f;
                    grassShadowPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                    grassShadowPushConstants.cascadeData[1] = 0.0f;
                    grassShadowPushConstants.cascadeData[2] = 0.0f;
                    grassShadowPushConstants.cascadeData[3] = 0.0f;
                    vkCmdPushConstants(
                        commandBuffer,
                        m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT,
                        0,
                        sizeof(ChunkPushConstants),
                        &grassShadowPushConstants
                    );
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(commandBuffer, m_grassBillboardIndexCount, m_grassBillboardInstanceCount, 0, 0, 0);
                }
            }
            if (m_sdfShadowPipeline != VK_NULL_HANDLE) {
                ChunkPushConstants sdfShadowPushConstants{};
                sdfShadowPushConstants.chunkOffset[0] = 0.0f;
                sdfShadowPushConstants.chunkOffset[1] = 0.0f;
                sdfShadowPushConstants.chunkOffset[2] = 0.0f;
                sdfShadowPushConstants.chunkOffset[3] = 0.0f;
                sdfShadowPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                sdfShadowPushConstants.cascadeData[1] = 0.0f;
                sdfShadowPushConstants.cascadeData[2] = 0.0f;
                sdfShadowPushConstants.cascadeData[3] = 0.0f;
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_sdfShadowPipeline);
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
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &sdfShadowPushConstants
                );
                countDrawCalls(m_debugDrawCallsShadow, 1);
                vkCmdDraw(commandBuffer, 3, 1, 0, 0);
            }

            vkCmdEndRendering(commandBuffer);
        }
    }

    transitionImageLayout(
        commandBuffer,
        m_shadowDepthImage,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT,
        0,
        1
    );
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryShadowEnd);
}

}  // namespace voxelsprout::render
