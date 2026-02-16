#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstring>

#include "sim/network_procedural.h"
#include "render/backend/vulkan/frame_graph_runtime.h"

namespace voxelsprout::render {

#include "render/renderer_shared.h"

void RendererBackend::recordMainScenePass(const FrameExecutionContext& context, const MainPassInputs& inputs) {
    VkCommandBuffer commandBuffer = context.commandBuffer;
    VkQueryPool gpuTimestampQueryPool = context.gpuTimestampQueryPool;
    CoreFrameGraphOrderValidator& coreFramePassOrderValidator = *context.frameOrderValidator;
    const CoreFrameGraphPlan& coreFrameGraphPlan = *context.frameGraphPlan;
    const uint32_t aoFrameIndex = context.aoFrameIndex;
    const uint32_t imageIndex = context.imageIndex;
    const VkViewport& viewport = context.viewport;
    const VkRect2D& scissor = context.scissor;
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
    const VoxelPreview& preview = *inputs.preview;

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

    if (!m_msaaColorImageInitialized[imageIndex]) {
        transitionImageLayout(
            commandBuffer,
            m_msaaColorImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }
    const bool hdrResolveInitialized = m_hdrResolveImageInitialized[aoFrameIndex];
    transitionImageLayout(
        commandBuffer,
        m_hdrResolveImages[aoFrameIndex],
        hdrResolveInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        hdrResolveInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
        hdrResolveInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    transitionImageLayout(
        commandBuffer,
        m_depthImages[imageIndex],
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT
    );

    VkClearValue clearValue{};
    clearValue.color.float32[0] = 0.06f;
    clearValue.color.float32[1] = 0.08f;
    clearValue.color.float32[2] = 0.12f;
    clearValue.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = m_msaaColorImageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.clearValue = clearValue;
    colorAttachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
    colorAttachment.resolveImageView = m_hdrResolveImageViews[aoFrameIndex];
    colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkClearValue depthClearValue{};
    depthClearValue.depthStencil.depth = 0.0f;
    depthClearValue.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = m_depthImageViews[imageIndex];
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.clearValue = depthClearValue;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = m_swapchainExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;

    writeGpuTimestampTop(kGpuTimestampQueryMainStart);
    coreFramePassOrderValidator.markPassEntered(coreFrameGraphPlan.main, "main");
    beginDebugLabel(commandBuffer, "Pass: Main Scene", 0.20f, 0.20f, 0.45f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
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
        drawIndirectChunkRanges(commandBuffer, m_debugDrawCallsMain, frameChunkDrawData);
    }
    if (canDrawMagica) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_magicaPipeline);
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
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, magicaDraw.indexCount, 1, 0, 0, 0);
        }
    }

    if (m_pipePipeline != VK_NULL_HANDLE) {
        auto drawLitInstances = [&](BufferHandle vertexHandle,
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

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipePipeline);
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
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, indexCount, instanceCount, 0, 0, 0);
        };
        drawLitInstances(
            m_pipeVertexBufferHandle,
            m_pipeIndexBufferHandle,
            m_pipeIndexCount,
            pipeInstanceCount,
            pipeInstanceSliceOpt
        );
        drawLitInstances(
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            m_transportIndexCount,
            transportInstanceCount,
            transportInstanceSliceOpt
        );
        drawLitInstances(
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            m_transportIndexCount,
            beltCargoInstanceCount,
            beltCargoInstanceSliceOpt
        );
    }

    if (m_grassBillboardPipeline != VK_NULL_HANDLE &&
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
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_grassBillboardPipeline);
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
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, m_grassBillboardIndexCount, m_grassBillboardInstanceCount, 0, 0, 0);
        }
    }

    const VkPipeline activePreviewPipeline =
        (preview.mode == VoxelPreview::Mode::Remove) ? m_previewRemovePipeline : m_previewAddPipeline;
    const bool drawCubePreview = !preview.pipeStyle && preview.visible && activePreviewPipeline != VK_NULL_HANDLE;
    const bool drawFacePreview =
        !preview.pipeStyle && preview.faceVisible && preview.brushSize == 1 && m_previewRemovePipeline != VK_NULL_HANDLE;

    if (preview.pipeStyle && preview.visible && m_pipePipeline != VK_NULL_HANDLE) {
        PipeInstance previewInstance{};
        previewInstance.originLength[0] = static_cast<float>(preview.x);
        previewInstance.originLength[1] = static_cast<float>(preview.y);
        previewInstance.originLength[2] = static_cast<float>(preview.z);
        previewInstance.originLength[3] = 1.0f;
        voxelsprout::math::Vector3 previewAxis =
            voxelsprout::math::normalize(voxelsprout::math::Vector3{preview.pipeAxisX, preview.pipeAxisY, preview.pipeAxisZ});
        if (voxelsprout::math::lengthSquared(previewAxis) <= 0.0001f) {
            previewAxis = voxelsprout::math::Vector3{0.0f, 1.0f, 0.0f};
        }
        previewInstance.axisRadius[0] = previewAxis.x;
        previewInstance.axisRadius[1] = previewAxis.y;
        previewInstance.axisRadius[2] = previewAxis.z;
        previewInstance.axisRadius[3] = std::clamp(preview.pipeRadius, 0.02f, 0.5f);
        if (preview.mode == VoxelPreview::Mode::Remove) {
            previewInstance.tint[0] = 1.0f;
            previewInstance.tint[1] = 0.32f;
            previewInstance.tint[2] = 0.26f;
        } else {
            previewInstance.tint[0] = 0.30f;
            previewInstance.tint[1] = 0.95f;
            previewInstance.tint[2] = 1.0f;
        }
        previewInstance.tint[3] = std::clamp(preview.pipeStyleId, 0.0f, 2.0f);
        previewInstance.extensions[0] = 0.0f;
        previewInstance.extensions[1] = 0.0f;
        previewInstance.extensions[2] = 1.0f;
        previewInstance.extensions[3] = 1.0f;
        if (preview.pipeStyleId > 0.5f && preview.pipeStyleId < 1.5f) {
            previewInstance.extensions[2] = 2.0f;
            previewInstance.extensions[3] = 0.25f;
        }
        if (preview.pipeStyleId > 1.5f) {
            previewInstance.extensions[2] = 2.0f;
            previewInstance.extensions[3] = 0.25f;
        }

        const std::optional<FrameArenaSlice> previewInstanceSlice =
            m_frameArena.allocateUpload(
                sizeof(PipeInstance),
                static_cast<VkDeviceSize>(alignof(PipeInstance)),
                FrameArenaUploadKind::PreviewData
            );
        if (previewInstanceSlice.has_value() && previewInstanceSlice->mapped != nullptr) {
            std::memcpy(previewInstanceSlice->mapped, &previewInstance, sizeof(PipeInstance));
            const bool previewUsesPipeMesh = preview.pipeStyleId < 0.5f;
            const BufferHandle previewVertexHandle =
                previewUsesPipeMesh ? m_pipeVertexBufferHandle : m_transportVertexBufferHandle;
            const BufferHandle previewIndexHandle =
                previewUsesPipeMesh ? m_pipeIndexBufferHandle : m_transportIndexBufferHandle;
            const uint32_t previewIndexCount = previewUsesPipeMesh ? m_pipeIndexCount : m_transportIndexCount;
            if (previewIndexCount == 0) {
                // No mesh data allocated for this preview style.
            }
            const VkBuffer pipeVertexBuffer = m_bufferAllocator.getBuffer(previewVertexHandle);
            const VkBuffer pipeIndexBuffer = m_bufferAllocator.getBuffer(previewIndexHandle);
            const VkBuffer pipeInstanceBuffer = m_bufferAllocator.getBuffer(previewInstanceSlice->buffer);
            if (pipeVertexBuffer != VK_NULL_HANDLE &&
                pipeIndexBuffer != VK_NULL_HANDLE &&
                pipeInstanceBuffer != VK_NULL_HANDLE &&
                previewIndexCount > 0) {
                const VkBuffer vertexBuffers[2] = {pipeVertexBuffer, pipeInstanceBuffer};
                const VkDeviceSize vertexOffsets[2] = {0, previewInstanceSlice->offset};
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipePipeline);
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
                vkCmdBindIndexBuffer(commandBuffer, pipeIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexed(commandBuffer, previewIndexCount, 1, 0, 0, 0);
            }
        }
    }

    if (drawCubePreview || drawFacePreview) {
        constexpr uint32_t kPreviewCubeIndexCount = 36u;
        constexpr uint32_t kPreviewFaceIndexCount = 6u;
        constexpr uint32_t kAddCubeFirstIndex = 0u;
        constexpr uint32_t kRemoveCubeFirstIndex = 36u;
        constexpr uint32_t kFaceFirstIndexBase = kRemoveCubeFirstIndex;
        constexpr float kChunkCoordinateScale = 1.0f;

        const VkBuffer previewVertexBuffer = m_bufferAllocator.getBuffer(m_previewVertexBufferHandle);
        const VkBuffer previewIndexBuffer = m_bufferAllocator.getBuffer(m_previewIndexBufferHandle);
        if (previewVertexBuffer != VK_NULL_HANDLE &&
            previewIndexBuffer != VK_NULL_HANDLE &&
            chunkInstanceSliceOpt.has_value() &&
            chunkInstanceBuffer != VK_NULL_HANDLE) {
            const VkBuffer previewVertexBuffers[2] = {previewVertexBuffer, chunkInstanceBuffer};
            const VkDeviceSize previewVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, previewVertexBuffers, previewVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, previewIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

            auto drawPreviewRange = [&](VkPipeline pipeline, uint32_t indexCount, uint32_t firstIndex, int x, int y, int z) {
                if (pipeline == VK_NULL_HANDLE || indexCount == 0) {
                    return;
                }
                ChunkPushConstants previewChunkPushConstants{};
                previewChunkPushConstants.chunkOffset[0] = static_cast<float>(x) * kChunkCoordinateScale;
                previewChunkPushConstants.chunkOffset[1] = static_cast<float>(y) * kChunkCoordinateScale;
                previewChunkPushConstants.chunkOffset[2] = static_cast<float>(z) * kChunkCoordinateScale;
                previewChunkPushConstants.chunkOffset[3] = 0.0f;
                previewChunkPushConstants.cascadeData[0] = 0.0f;
                previewChunkPushConstants.cascadeData[1] = 0.0f;
                previewChunkPushConstants.cascadeData[2] = 0.0f;
                previewChunkPushConstants.cascadeData[3] = 0.0f;

                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
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
                    &previewChunkPushConstants
                );
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexed(commandBuffer, indexCount, 1, firstIndex, 0, 0);
            };

            if (drawCubePreview) {
                const uint32_t cubeFirstIndex =
                    (preview.mode == VoxelPreview::Mode::Add) ? kAddCubeFirstIndex : kRemoveCubeFirstIndex;
                const int brushSize = std::max(preview.brushSize, 1);
                for (int localY = 0; localY < brushSize; ++localY) {
                    for (int localZ = 0; localZ < brushSize; ++localZ) {
                        for (int localX = 0; localX < brushSize; ++localX) {
                            drawPreviewRange(
                                activePreviewPipeline,
                                kPreviewCubeIndexCount,
                                cubeFirstIndex,
                                preview.x + localX,
                                preview.y + localY,
                                preview.z + localZ
                            );
                        }
                    }
                }
            }

            if (drawFacePreview) {
                const uint32_t faceFirstIndex =
                    kFaceFirstIndexBase + (std::min(preview.faceId, 5u) * kPreviewFaceIndexCount);
                drawPreviewRange(
                    m_previewRemovePipeline,
                    kPreviewFaceIndexCount,
                    faceFirstIndex,
                    preview.faceX,
                    preview.faceY,
                    preview.faceZ
                );
            }
        }
    }

    // Draw skybox last with depth-test so sun/sky only appears where no geometry wrote depth.
    if (m_skyboxPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyboxPipeline);
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
        countDrawCalls(m_debugDrawCallsMain, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }

    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryMainEnd);
}

}  // namespace voxelsprout::render
