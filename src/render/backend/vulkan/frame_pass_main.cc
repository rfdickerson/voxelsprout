#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstring>

#include "sim/network_procedural.h"
#include "render/backend/vulkan/frame_graph_runtime.h"

namespace odai::render {

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
    const VkBuffer importedVertexBuffer = inputs.importedVertexBuffer;
    const VkBuffer importedIndexBuffer = inputs.importedIndexBuffer;
    const std::span<const ImportedMeshDraw> importedMeshDraws = inputs.importedMeshDraws;
    const std::uint32_t importedTerrainDrawCount = inputs.importedTerrainDrawCount;
    const VkBuffer importedActorVertexBuffer = inputs.importedActorVertexBuffer;
    const VkDeviceSize importedActorVertexOffset = inputs.importedActorVertexOffset;
    const VkBuffer importedActorIndexBuffer = inputs.importedActorIndexBuffer;
    const VkDeviceSize importedActorIndexOffset = inputs.importedActorIndexOffset;
    const std::span<const ImportedMeshDraw> importedActorMeshDraws = inputs.importedActorMeshDraws;
    const uint32_t pipeInstanceCount = inputs.pipeInstanceCount;
    const std::optional<FrameArenaSlice>& pipeInstanceSliceOpt = *inputs.pipeInstanceSliceOpt;
    const uint32_t transportInstanceCount = inputs.transportInstanceCount;
    const std::optional<FrameArenaSlice>& transportInstanceSliceOpt = *inputs.transportInstanceSliceOpt;
    const uint32_t beltCargoInstanceCount = inputs.beltCargoInstanceCount;
    const std::optional<FrameArenaSlice>& beltCargoInstanceSliceOpt = *inputs.beltCargoInstanceSliceOpt;
    const VoxelPreview& preview = *inputs.preview;
    const bool useRtMainShadows =
        m_shadowStats.activeMode == ShadowMode::RayTraced &&
        m_shadowStats.mainPassRayTracingReady;
    const bool useRtVoxelShadows = useRtMainShadows && m_pipelineRt != VK_NULL_HANDLE;
    const bool useRtMagicaShadows = useRtMainShadows && m_magicaPipelineRt != VK_NULL_HANDLE;
    m_shadowStats.mainPassRayTracingActive = useRtMainShadows;

    if (useRtMainShadows) {
        VkMemoryBarrier2 rayTracingReadBarrier{};
        rayTracingReadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        rayTracingReadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        rayTracingReadBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        rayTracingReadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        rayTracingReadBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.memoryBarrierCount = 1;
        dependencyInfo.pMemoryBarriers = &rayTracingReadBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }

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
    const bool canDrawImportedWater =
        m_importedWaterPipeline != VK_NULL_HANDLE &&
        m_importedWaterVertexBufferHandle != kInvalidBufferHandle &&
        m_importedWaterIndexBufferHandle != kInvalidBufferHandle &&
        m_importedWaterIndexCount > 0;

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
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
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
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
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

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, useRtVoxelShadows ? m_pipelineRt : m_pipeline);
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
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(ChunkPushConstants),
            &chunkPushConstants
        );
        drawIndirectChunkRanges(commandBuffer, m_debugDrawCallsMain, frameChunkDrawData);
    }
    if (canDrawMagica) {
        vkCmdBindPipeline(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            useRtMagicaShadows ? m_magicaPipelineRt : m_magicaPipeline
        );
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
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(ChunkPushConstants),
                &magicaPushConstants
            );
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, magicaDraw.indexCount, 1, 0, 0, 0);
        }
    }
    if (m_importedStaticPipeline != VK_NULL_HANDLE &&
        importedVertexBuffer != VK_NULL_HANDLE &&
        importedIndexBuffer != VK_NULL_HANDLE &&
        !importedMeshDraws.empty()) {
        const std::size_t terrainDrawCount = std::min<std::size_t>(importedTerrainDrawCount, importedMeshDraws.size());
        const std::size_t staticDrawStart = terrainDrawCount;
        const bool drawTerrain = m_debugShowImportedTerrain;
        const bool drawStatics = m_debugShowImportedStatics;
        const VkBuffer importedVertexBuffers[1] = {importedVertexBuffer};
        const VkDeviceSize importedVertexOffsets[1] = {0};
        vkCmdBindPipeline(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            (useRtMainShadows && m_importedStaticPipelineRt != VK_NULL_HANDLE)
                ? m_importedStaticPipelineRt
                : m_importedStaticPipeline
        );
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
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, importedVertexBuffers, importedVertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, importedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
        ChunkPushConstants importedPushConstants{};
        importedPushConstants.cascadeData[2] = m_debugShowImportedTextures ? 0.0f : 1.0f;
        importedPushConstants.cascadeData[3] = m_debugImportedFlatShading ? 1.0f : 0.0f;
        vkCmdPushConstants(
            commandBuffer,
            m_pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(ChunkPushConstants),
            &importedPushConstants
        );
        for (std::size_t drawIndex = 0; drawIndex < importedMeshDraws.size(); ++drawIndex) {
            if ((drawIndex < terrainDrawCount && !drawTerrain) ||
                (drawIndex >= staticDrawStart && !drawStatics)) {
                continue;
            }
            const ImportedMeshDraw& importedDraw = importedMeshDraws[drawIndex];
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, importedDraw.indexCount, 1, importedDraw.firstIndex, 0, 0);
        }
    }
    if (m_importedStaticPipeline != VK_NULL_HANDLE &&
        importedActorVertexBuffer != VK_NULL_HANDLE &&
        importedActorIndexBuffer != VK_NULL_HANDLE &&
        !importedActorMeshDraws.empty() &&
        m_debugShowImportedStatics) {
        const VkBuffer importedVertexBuffers[1] = {importedActorVertexBuffer};
        const VkDeviceSize importedVertexOffsets[1] = {importedActorVertexOffset};
        vkCmdBindPipeline(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            (useRtMainShadows && m_importedStaticPipelineRt != VK_NULL_HANDLE)
                ? m_importedStaticPipelineRt
                : m_importedStaticPipeline
        );
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
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, importedVertexBuffers, importedVertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, importedActorIndexBuffer, importedActorIndexOffset, VK_INDEX_TYPE_UINT32);
        ChunkPushConstants importedPushConstants{};
        importedPushConstants.cascadeData[2] = m_debugShowImportedTextures ? 0.0f : 1.0f;
        importedPushConstants.cascadeData[3] = m_debugImportedFlatShading ? 1.0f : 0.0f;
        vkCmdPushConstants(
            commandBuffer,
            m_pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(ChunkPushConstants),
            &importedPushConstants
        );
        for (const ImportedMeshDraw& importedDraw : importedActorMeshDraws) {
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, importedDraw.indexCount, 1, importedDraw.firstIndex, 0, 0);
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

    const bool canCaptureWaterRefraction =
        canDrawImportedWater &&
        aoFrameIndex < m_waterRefractionImages.size() &&
        aoFrameIndex < m_waterRefractionImageInitialized.size() &&
        m_waterRefractionImages[aoFrameIndex] != VK_NULL_HANDLE &&
        m_hdrResolveImages[aoFrameIndex] != VK_NULL_HANDLE;
    if (canCaptureWaterRefraction) {
        vkCmdEndRendering(commandBuffer);

        transitionImageLayout(
            commandBuffer,
            m_hdrResolveImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0u,
            1u,
            0u,
            1u
        );
        transitionImageLayout(
            commandBuffer,
            m_waterRefractionImages[aoFrameIndex],
            m_waterRefractionImageInitialized[aoFrameIndex]
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            m_waterRefractionImageInitialized[aoFrameIndex]
                ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                : VK_PIPELINE_STAGE_2_NONE,
            m_waterRefractionImageInitialized[aoFrameIndex]
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );

        VkImageCopy opaqueCopyRegion{};
        opaqueCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        opaqueCopyRegion.srcSubresource.mipLevel = 0;
        opaqueCopyRegion.srcSubresource.baseArrayLayer = 0;
        opaqueCopyRegion.srcSubresource.layerCount = 1;
        opaqueCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        opaqueCopyRegion.dstSubresource.mipLevel = 0;
        opaqueCopyRegion.dstSubresource.baseArrayLayer = 0;
        opaqueCopyRegion.dstSubresource.layerCount = 1;
        opaqueCopyRegion.extent = {m_swapchainExtent.width, m_swapchainExtent.height, 1u};
        vkCmdCopyImage(
            commandBuffer,
            m_hdrResolveImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            m_waterRefractionImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &opaqueCopyRegion
        );

        transitionImageLayout(
            commandBuffer,
            m_waterRefractionImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        transitionImageLayout(
            commandBuffer,
            m_hdrResolveImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0u,
            1u,
            0u,
            1u
        );
        transitionImageLayout(
            commandBuffer,
            m_msaaColorImages[imageIndex],
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        transitionImageLayout(
            commandBuffer,
            m_depthImages[imageIndex],
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT
        );
        m_waterRefractionImageInitialized[aoFrameIndex] = true;

        VkRenderingAttachmentInfo waterColorAttachment = colorAttachment;
        waterColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        waterColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        VkRenderingAttachmentInfo waterDepthAttachment = depthAttachment;
        waterDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        waterDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        VkRenderingInfo waterRenderingInfo = renderingInfo;
        waterRenderingInfo.pColorAttachments = &waterColorAttachment;
        waterRenderingInfo.pDepthAttachment = &waterDepthAttachment;

        vkCmdBeginRendering(commandBuffer, &waterRenderingInfo);
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    }

    if (canDrawImportedWater) {
        const VkBuffer waterVertexBuffer = m_bufferAllocator.getBuffer(m_importedWaterVertexBufferHandle);
        const VkBuffer waterIndexBuffer = m_bufferAllocator.getBuffer(m_importedWaterIndexBufferHandle);
        if (waterVertexBuffer != VK_NULL_HANDLE && waterIndexBuffer != VK_NULL_HANDLE) {
            const bool useRtWaterReflections =
                m_rayTracingRuntimeEnabled &&
                m_rtTlas.handle != VK_NULL_HANDLE &&
                m_importedWaterPipelineRt != VK_NULL_HANDLE;
            const VkBuffer waterVertexBuffers[1] = {waterVertexBuffer};
            const VkDeviceSize waterVertexOffsets[1] = {0};
            vkCmdBindPipeline(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                useRtWaterReflections ? m_importedWaterPipelineRt : m_importedWaterPipeline
            );
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
            ChunkPushConstants waterPushConstants{};
            waterPushConstants.cascadeData[2] = m_debugImportedWaterSolid ? 1.0f : 0.0f;
            vkCmdPushConstants(
                commandBuffer,
                m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(ChunkPushConstants),
                &waterPushConstants
            );
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, waterVertexBuffers, waterVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, waterIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, m_importedWaterIndexCount, 1, 0, 0, 0);
        }
    }
    const VkPipeline activePreviewPipeline =
        (preview.mode == VoxelPreview::Mode::Remove) ? m_previewRemovePipeline : m_previewAddPipeline;
    const bool drawCubePreview = !preview.pipeStyle && preview.visible && activePreviewPipeline != VK_NULL_HANDLE;
    const bool drawFacePreview =
        !preview.pipeStyle && preview.faceVisible && preview.brushSize == 1 && m_previewFaceOutlinePipeline != VK_NULL_HANDLE;

    if (preview.pipeStyle && preview.visible && m_pipePipeline != VK_NULL_HANDLE) {
        PipeInstance previewInstance{};
        previewInstance.originLength[0] = static_cast<float>(preview.x);
        previewInstance.originLength[1] = static_cast<float>(preview.y);
        previewInstance.originLength[2] = static_cast<float>(preview.z);
        previewInstance.originLength[3] = 1.0f;
        odai::math::Vector3 previewAxis =
            odai::math::normalize(odai::math::Vector3{preview.pipeAxisX, preview.pipeAxisY, preview.pipeAxisZ});
        if (odai::math::lengthSquared(previewAxis) <= 0.0001f) {
            previewAxis = odai::math::Vector3{0.0f, 1.0f, 0.0f};
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
        constexpr uint32_t kPreviewFaceOutlineIndexCount = 8u;
        constexpr uint32_t kAddCubeFirstIndex = 0u;
        constexpr uint32_t kRemoveCubeFirstIndex = 36u;
        constexpr uint32_t kAddFaceOutlineFirstIndexBase = 72u;
        constexpr uint32_t kRemoveFaceOutlineFirstIndexBase = 120u;
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
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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
                    ((preview.mode == VoxelPreview::Mode::Add)
                         ? kAddFaceOutlineFirstIndexBase
                         : kRemoveFaceOutlineFirstIndexBase) +
                    (std::min(preview.faceId, 5u) * kPreviewFaceOutlineIndexCount);
                drawPreviewRange(
                    m_previewFaceOutlinePipeline,
                    kPreviewFaceOutlineIndexCount,
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
    if (m_skyCloudPipeline != VK_NULL_HANDLE &&
        m_skyCloudVertexBufferHandle != kInvalidBufferHandle &&
        m_skyCloudIndexBufferHandle != kInvalidBufferHandle &&
        m_skyCloudIndexCount > 0 &&
        m_morrowindSkyTextureImageView != VK_NULL_HANDLE) {
        const VkBuffer skyCloudVertexBuffer = m_bufferAllocator.getBuffer(m_skyCloudVertexBufferHandle);
        const VkBuffer skyCloudIndexBuffer = m_bufferAllocator.getBuffer(m_skyCloudIndexBufferHandle);
        if (skyCloudVertexBuffer != VK_NULL_HANDLE && skyCloudIndexBuffer != VK_NULL_HANDLE) {
            const VkBuffer skyCloudVertexBuffers[1] = {skyCloudVertexBuffer};
            const VkDeviceSize skyCloudVertexOffsets[1] = {0};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyCloudPipeline);
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
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, skyCloudVertexBuffers, skyCloudVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, skyCloudIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, m_skyCloudIndexCount, 1, 0, 0, 0);
        }
    }

    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQueryMainEnd);
}

}  // namespace odai::render
