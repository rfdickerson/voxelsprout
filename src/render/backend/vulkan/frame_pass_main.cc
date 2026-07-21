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
    const uint32_t mvpDynamicOffset = context.mvpDynamicOffset;
    // Legacy voxel/magica/pipe per-frame draw inputs are still present on MainPassInputs
    // but no longer consumed here — those passes were removed (prior voxel/factory game).
    const VkBuffer importedVertexBuffer = inputs.importedVertexBuffer;
    const VkBuffer importedIndexBuffer = inputs.importedIndexBuffer;
    const std::span<const ImportedMeshDraw> importedMeshDraws = inputs.importedMeshDraws;
    const std::uint32_t importedTerrainDrawCount = inputs.importedTerrainDrawCount;
    const VkBuffer importedActorVertexBuffer = inputs.importedActorVertexBuffer;
    const VkDeviceSize importedActorVertexOffset = inputs.importedActorVertexOffset;
    const VkBuffer importedActorIndexBuffer = inputs.importedActorIndexBuffer;
    const VkDeviceSize importedActorIndexOffset = inputs.importedActorIndexOffset;
    const std::span<const ImportedMeshDraw> importedActorMeshDraws = inputs.importedActorMeshDraws;
    const bool renderingImportedScene = !importedMeshDraws.empty() || !importedActorMeshDraws.empty();
    const bool useRtMainShadows =
        m_shadowStats.activeMode == ShadowMode::RayTraced &&
        m_shadowStats.mainPassRayTracingReady;
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

    if (!renderingImportedScene && m_terrainTessPipeline != VK_NULL_HANDLE) {
        constexpr std::uint32_t kTerrainPatchGridResolution = 16u;
        constexpr std::uint32_t kTerrainPatchControlPointCount = 4u;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_terrainTessPipeline);
        bindGraphicsDescriptorBuffers(commandBuffer);
        countDrawCalls(m_debugDrawCallsMain, 1);
        vkCmdDraw(
            commandBuffer,
            kTerrainPatchControlPointCount * kTerrainPatchGridResolution * kTerrainPatchGridResolution,
            1,
            0,
            0
        );
    }

    // Hex strategy-map land: one instanced, tessellated, height-displaced draw of the
    // shared base hex fan (one instance per land tile). The TESC collapses distant
    // tiles to the base fan, so a single all-instances draw is cheap.
    if (m_hexTerrainEnabled && m_hexTerrainPipeline != VK_NULL_HANDLE && m_hexInstanceCount > 0) {
        const VkBuffer hexBaseVertexBuffer = m_bufferAllocator.getBuffer(m_hexBaseVertexBufferHandle);
        const VkBuffer hexBaseIndexBuffer = m_bufferAllocator.getBuffer(m_hexBaseIndexBufferHandle);
        const VkBuffer hexInstanceBuffer = m_bufferAllocator.getBuffer(m_hexInstanceBufferHandle);
        if (hexBaseVertexBuffer != VK_NULL_HANDLE && hexBaseIndexBuffer != VK_NULL_HANDLE &&
            hexInstanceBuffer != VK_NULL_HANDLE) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_hexTerrainPipeline);
            bindGraphicsDescriptorBuffers(commandBuffer);
            const VkBuffer hexVertexBuffers[2] = {hexBaseVertexBuffer, hexInstanceBuffer};
            const VkDeviceSize hexVertexOffsets[2] = {0, 0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 2, hexVertexBuffers, hexVertexOffsets);
            vkCmdBindIndexBuffer(commandBuffer, hexBaseIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, m_hexIndexCount, m_hexInstanceCount, 0, 0, 0);
        }
    }

    // (removed) voxel chunk + magica model main-pass draws — legacy from the prior
    // voxel/factory game; the strategy map renders hex terrain + imported scene instead.
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
        bindGraphicsDescriptorBuffers(commandBuffer);
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, importedVertexBuffers, importedVertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, importedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
        ChunkPushConstants importedPushConstants{};
        importedPushConstants.cascadeData[1] = m_importedSceneInteriorMode ? 1.0f : 0.0f;
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
        bindGraphicsDescriptorBuffers(commandBuffer);
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, importedVertexBuffers, importedVertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, importedActorIndexBuffer, importedActorIndexOffset, VK_INDEX_TYPE_UINT32);
        ChunkPushConstants importedPushConstants{};
        importedPushConstants.cascadeData[1] = m_importedSceneInteriorMode ? 1.0f : 0.0f;
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

    // (removed) pipe / belt / transport instanced main-pass draws — legacy factory-sim
    // rendering from the prior game; the strategy map has no pipes or conveyors.

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
            bindGraphicsDescriptorBuffers(commandBuffer);
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
            bindGraphicsDescriptorBuffers(commandBuffer);
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
            // Vulkan Roadmap 2026 VRS: shade the low-frequency water surface coarsely
            // (2x2) to reclaim GPU. The water pipeline carries the dynamic shading-rate
            // state, so the rate must be set before drawing it when VRS is available.
            if (m_supportsVrs && m_cmdSetFragmentShadingRate != nullptr) {
                const uint32_t rate = m_debugWaterVrsCoarse ? 2u : 1u;
                const VkExtent2D fragmentSize{rate, rate};
                const VkFragmentShadingRateCombinerOpKHR combinerOps[2] = {
                    VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
                    VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
                };
                m_cmdSetFragmentShadingRate(commandBuffer, &fragmentSize, combinerOps);
            }
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(commandBuffer, m_importedWaterIndexCount, 1, 0, 0, 0);
        }
    }
    // (removed) voxel/pipe placement-preview draws — legacy editor overlays from the
    // prior game (cube/face brush + pipe ghost); the strategy map has no voxel editing.

    // Draw skybox last with depth-test so sun/sky only appears where no geometry wrote depth.
    if (m_skyboxPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyboxPipeline);
        bindGraphicsDescriptorBuffers(commandBuffer);
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
            bindGraphicsDescriptorBuffers(commandBuffer);
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
