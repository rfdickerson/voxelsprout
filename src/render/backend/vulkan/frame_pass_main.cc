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
    // Voxel chunk inputs: consumed by the chunk draw below (VoxelCraft). The magica/pipe
    // inputs are still present on MainPassInputs but remain unconsumed here.
    const FrameChunkDrawData& frameChunkDrawData = *inputs.frameChunkDrawData;
    const std::optional<FrameArenaSlice>& chunkInstanceSliceOpt = *inputs.chunkInstanceSliceOpt;
    const VkBuffer chunkInstanceBuffer = inputs.chunkInstanceBuffer;
    const VkBuffer chunkVertexBuffer = inputs.chunkVertexBuffer;
    const VkBuffer chunkIndexBuffer = inputs.chunkIndexBuffer;
    const VkBuffer importedVertexBuffer = inputs.importedVertexBuffer;
    const VkBuffer importedIndexBuffer = inputs.importedIndexBuffer;
    const std::span<const ImportedMeshDraw> importedMeshDraws = inputs.importedMeshDraws;
    const std::uint32_t importedTerrainDrawCount = inputs.importedTerrainDrawCount;
    const std::span<const std::uint32_t> importedBlendedDrawOrder = inputs.importedBlendedDrawOrder;
    const VkBuffer importedActorVertexBuffer = inputs.importedActorVertexBuffer;
    const VkDeviceSize importedActorVertexOffset = inputs.importedActorVertexOffset;
    const VkBuffer importedActorIndexBuffer = inputs.importedActorIndexBuffer;
    const VkDeviceSize importedActorIndexOffset = inputs.importedActorIndexOffset;
    const std::span<const ImportedMeshDraw> importedActorMeshDraws = inputs.importedActorMeshDraws;
    const std::span<const std::uint32_t> importedActorBlendedDrawOrder =
        inputs.importedActorBlendedDrawOrder;
    const std::span<const ImportedMeshDraw> skinnedActorMeshDraws = inputs.skinnedActorMeshDraws;
    const bool renderingImportedScene = !importedMeshDraws.empty() || !importedActorMeshDraws.empty();
    const bool useRtMainShadows =
        m_shadowStats.activeMode == ShadowMode::RayTraced &&
        m_shadowStats.mainPassRayTracingReady;
    const bool useRtVoxelShadows = useRtMainShadows && m_pipelineRt != VK_NULL_HANDLE;
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

    // Render a real planar reflection before the main scene. The transform in
    // imported_static.vert reflects world points across the selected water
    // plane, which is exactly equivalent to mirroring the camera, while the
    // fragment shader clips geometry below the plane. The existing pipelines
    // are single-sample in the showcase configuration, so the half-resolution
    // target needs no resolve image of its own.
    const bool canRenderPlanarWaterReflection =
        canDrawImportedWater &&
        m_waterReflectionPlaneValid &&
        m_colorSampleCount == VK_SAMPLE_COUNT_1_BIT &&
        aoFrameIndex < m_waterReflectionImages.size() &&
        aoFrameIndex < m_waterReflectionImageViews.size() &&
        aoFrameIndex < m_waterReflectionImageInitialized.size() &&
        aoFrameIndex < m_waterReflectionDepthImages.size() &&
        aoFrameIndex < m_waterReflectionDepthImageViews.size() &&
        aoFrameIndex < m_waterReflectionDepthImageInitialized.size() &&
        m_waterReflectionImages[aoFrameIndex] != VK_NULL_HANDLE &&
        m_waterReflectionDepthImages[aoFrameIndex] != VK_NULL_HANDLE &&
        m_importedStaticPipelineTwoSided != VK_NULL_HANDLE &&
        m_importedStaticDepthPrewritePipelineTwoSided != VK_NULL_HANDLE &&
        importedVertexBuffer != VK_NULL_HANDLE &&
        importedIndexBuffer != VK_NULL_HANDLE &&
        !importedMeshDraws.empty();
    const bool useWaterReflectionTemporalResolve =
        m_waterReflectionTemporalEnabled &&
        m_waterReflectionResolvePipeline != VK_NULL_HANDLE &&
        m_waterReflectionResolveBufferSet.valid();
    if (canRenderPlanarWaterReflection) {
        beginDebugLabel(commandBuffer, "Pass: Planar Water Reflection", 0.08f, 0.34f, 0.42f, 1.0f);
        transitionImageLayout(
            commandBuffer,
            m_waterReflectionImages[aoFrameIndex],
            m_waterReflectionImageInitialized[aoFrameIndex]
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            m_waterReflectionImageInitialized[aoFrameIndex]
                ? (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)
                : VK_PIPELINE_STAGE_2_NONE,
            m_waterReflectionImageInitialized[aoFrameIndex]
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        transitionImageLayout(
            commandBuffer,
            m_waterReflectionDepthImages[aoFrameIndex],
            m_waterReflectionDepthImageInitialized[aoFrameIndex]
                ? (m_waterReflectionDepthSampled[aoFrameIndex]
                    ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL)
                : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            m_waterReflectionDepthImageInitialized[aoFrameIndex]
                ? (m_waterReflectionDepthSampled[aoFrameIndex]
                    ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    : (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                       VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT))
                : VK_PIPELINE_STAGE_2_NONE,
            m_waterReflectionDepthImageInitialized[aoFrameIndex]
                ? (m_waterReflectionDepthSampled[aoFrameIndex]
                    ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                    : VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
                : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);
        m_waterReflectionDepthSampled[aoFrameIndex] = false;

        VkClearValue reflectionClear{};
        const bool reflectionHasSky = shouldRenderImportedSky(m_importedInteriorLighting);
        reflectionClear.color.float32[0] = reflectionHasSky ? 0.06f : m_importedInteriorLighting.fogColor[0];
        reflectionClear.color.float32[1] = reflectionHasSky ? 0.08f : m_importedInteriorLighting.fogColor[1];
        reflectionClear.color.float32[2] = reflectionHasSky ? 0.12f : m_importedInteriorLighting.fogColor[2];
        reflectionClear.color.float32[3] = 1.0f;
        VkRenderingAttachmentInfo reflectionColorAttachment{};
        reflectionColorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        reflectionColorAttachment.imageView = m_waterReflectionImageViews[aoFrameIndex];
        reflectionColorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        reflectionColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        reflectionColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        reflectionColorAttachment.clearValue = reflectionClear;
        VkClearValue reflectionDepthClear{};
        reflectionDepthClear.depthStencil.depth = 0.0f;
        VkRenderingAttachmentInfo reflectionDepthAttachment{};
        reflectionDepthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        reflectionDepthAttachment.imageView = m_waterReflectionDepthImageViews[aoFrameIndex];
        reflectionDepthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        reflectionDepthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        reflectionDepthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        reflectionDepthAttachment.clearValue = reflectionDepthClear;
        VkRenderingInfo reflectionRenderingInfo{};
        reflectionRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        reflectionRenderingInfo.renderArea.extent = m_waterReflectionExtent;
        reflectionRenderingInfo.layerCount = 1;
        reflectionRenderingInfo.colorAttachmentCount = 1;
        reflectionRenderingInfo.pColorAttachments = &reflectionColorAttachment;
        reflectionRenderingInfo.pDepthAttachment = &reflectionDepthAttachment;
        vkCmdBeginRendering(commandBuffer, &reflectionRenderingInfo);

        VkViewport reflectionViewport{};
        reflectionViewport.width = static_cast<float>(m_waterReflectionExtent.width);
        reflectionViewport.height = static_cast<float>(m_waterReflectionExtent.height);
        reflectionViewport.minDepth = 0.0f;
        reflectionViewport.maxDepth = 1.0f;
        VkRect2D reflectionScissor{};
        reflectionScissor.extent = m_waterReflectionExtent;
        vkCmdSetViewport(commandBuffer, 0, 1, &reflectionViewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &reflectionScissor);

        vkCmdBindPipeline(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_importedStaticPipelineTwoSided);
        bindGraphicsDescriptorBuffers(commandBuffer);
        const VkBuffer reflectionVertexBuffers[1] = {importedVertexBuffer};
        const VkDeviceSize reflectionVertexOffsets[1] = {0};
        vkCmdBindVertexBuffers(
            commandBuffer, 0, 1, reflectionVertexBuffers, reflectionVertexOffsets);
        vkCmdBindIndexBuffer(
            commandBuffer, importedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

        ChunkPushConstants reflectionPush{};
        reflectionPush.cascadeData[0] = 1.0f;
        reflectionPush.cascadeData[1] = m_importedSceneInteriorMode ? 1.0f : 0.0f;
        reflectionPush.cascadeData[2] = m_debugShowImportedTextures ? 0.0f : 1.0f;
        reflectionPush.cascadeData[3] = m_debugImportedFlatShading ? 1.0f : 0.0f;
        reflectionPush.materialParams[2] = m_debugHighlightUntextured ? 1.0f : 0.0f;
        reflectionPush.materialParams[3] = m_waterReflectionPlaneHeight;
        const std::size_t reflectionTerrainCount = std::min<std::size_t>(
            importedTerrainDrawCount, importedMeshDraws.size());
        const auto includeReflectionDraw = [&](std::size_t drawIndex) {
            return drawIndex < reflectionTerrainCount
                ? m_debugShowImportedTerrain
                : m_debugShowImportedStatics;
        };
        VkBuffer reflectionIndirectBuffer = VK_NULL_HANDLE;
        VkDeviceSize reflectionIndirectBase = 0;
        const bool reflectionUsesIndirect = m_supportsMultiDrawIndirect &&
            buildImportedIndirectBatches(
                importedMeshDraws, includeReflectionDraw,
                reflectionIndirectBuffer, reflectionIndirectBase);
        const auto drawReflectionGeometry = [&](VkPipeline pipeline) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            const auto pushReflectionDraw = [&](const ImportedMeshDraw& draw) {
                reflectionPush.materialParams[0] =
                    static_cast<float>(draw.alphaThreshold) / 255.0f;
                reflectionPush.rigidAnimationParams[0] =
                    sampleImportedRigidAnimationTransform(
                        draw.rigidAnimationIndex,
                        reflectionPush.rigidAnimationTransform)
                        ? 1.0f
                        : 0.0f;
                vkCmdPushConstants(
                    commandBuffer, m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(reflectionPush), &reflectionPush);
            };
            if (reflectionUsesIndirect) {
                reflectionPush.rigidAnimationParams[0] = 0.0f;
                for (const ImportedIndirectBatch& batch : m_importedIndirectBatches) {
                    reflectionPush.materialParams[0] =
                        static_cast<float>(batch.alphaThreshold) / 255.0f;
                    vkCmdPushConstants(
                        commandBuffer, m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        0, sizeof(reflectionPush), &reflectionPush);
                    countDrawCalls(m_debugDrawCallsMain, 1);
                    vkCmdDrawIndexedIndirect(
                        commandBuffer, reflectionIndirectBuffer,
                        reflectionIndirectBase + batch.bufferOffset,
                        batch.drawCount, sizeof(VkDrawIndexedIndirectCommand));
                }
            }
            for (std::size_t drawIndex = 0; drawIndex < importedMeshDraws.size(); ++drawIndex) {
                const ImportedMeshDraw& draw = importedMeshDraws[drawIndex];
                if (draw.blended || !includeReflectionDraw(drawIndex) ||
                    (reflectionUsesIndirect && draw.rigidAnimationIndex == 0xffffffffu)) {
                    continue;
                }
                pushReflectionDraw(draw);
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexed(
                    commandBuffer, draw.indexCount, 1, draw.firstIndex,
                    draw.vertexOffset, 0);
            }
        };
        // The normal imported pipeline is configured for the merged depth
        // prepass: depth test on, writes off. The reflection owns a separate
        // depth image, so replay the exact same batches through the depth-only
        // pipeline first or the sky's EQUAL-to-clear test would paint over all
        // reflected geometry.
        drawReflectionGeometry(m_importedStaticDepthPrewritePipelineTwoSided);
        drawReflectionGeometry(m_importedStaticPipelineTwoSided);

        if (reflectionHasSky && m_skyboxPipeline != VK_NULL_HANDLE) {
            vkCmdBindPipeline(
                commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyboxPipeline);
            bindGraphicsDescriptorBuffers(commandBuffer);
            vkCmdPushConstants(
                commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(reflectionPush), &reflectionPush);
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        }
        vkCmdEndRendering(commandBuffer);
        transitionImageLayout(
            commandBuffer,
            m_waterReflectionImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            useWaterReflectionTemporalResolve
                ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                : VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        if (useWaterReflectionTemporalResolve) {
            transitionImageLayout(
                commandBuffer,
                m_waterReflectionDepthImages[aoFrameIndex],
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_DEPTH_BIT);
            m_waterReflectionDepthSampled[aoFrameIndex] = true;
            writeGpuTimestampTop(kGpuTimestampQueryWaterReflectionResolveStart);
            if (!recordWaterReflectionResolve(commandBuffer, aoFrameIndex)) {
                m_waterReflectionHistoryValid = false;
            }
            writeGpuTimestampBottom(kGpuTimestampQueryWaterReflectionResolveEnd);
        }
        m_waterReflectionImageInitialized[aoFrameIndex] = true;
        m_waterReflectionDepthImageInitialized[aoFrameIndex] = true;
        endDebugLabel(commandBuffer);
    }

    // Null when the sample count is 1: createMsaaColorTargets skips the image
    // entirely and the main pass targets hdrResolve directly. See there.
    const bool msaaEnabled = m_colorSampleCount != VK_SAMPLE_COUNT_1_BIT &&
                             imageIndex < m_msaaColorImages.size() &&
                             m_msaaColorImages[imageIndex] != VK_NULL_HANDLE;
    if (msaaEnabled && !m_msaaColorImageInitialized[imageIndex]) {
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
    // Merged, the prepass already wrote this depth and already put a dependency
    // on it -- transitioning from UNDEFINED here would tell the driver the
    // contents are expendable and it is free to discard exactly what the main
    // pass is about to load.
    const bool mergedDepthPrepass = useMergedDepthPrepass();
    if (!mergedDepthPrepass) {
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
    }

    VkClearValue clearValue{};
    const bool renderImportedSky = shouldRenderImportedSky(m_importedInteriorLighting);
    clearValue.color.float32[0] = renderImportedSky ? 0.06f : m_importedInteriorLighting.fogColor[0];
    clearValue.color.float32[1] = renderImportedSky ? 0.08f : m_importedInteriorLighting.fogColor[1];
    clearValue.color.float32[2] = renderImportedSky ? 0.12f : m_importedInteriorLighting.fogColor[2];
    clearValue.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView =
        msaaEnabled ? m_msaaColorImageViews[imageIndex] : m_hdrResolveImageViews[aoFrameIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearValue;
    if (msaaEnabled) {
        colorAttachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
        colorAttachment.resolveImageView = m_hdrResolveImageViews[aoFrameIndex];
        colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    } else {
        colorAttachment.resolveMode = VK_RESOLVE_MODE_NONE;
        colorAttachment.resolveImageView = VK_NULL_HANDLE;
    }

    VkClearValue depthClearValue{};
    depthClearValue.depthStencil.depth = 0.0f;
    depthClearValue.depthStencil.stencil = 0;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = m_depthImageViews[imageIndex];
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    // LOAD under the merged prepass: that pass cleared and filled this buffer.
    // Clearing here would throw the prepass away and leave main with nothing to
    // early-Z against.
    depthAttachment.loadOp =
        mergedDepthPrepass ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue = depthClearValue;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = m_renderExtent;
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

    // Voxel chunk draws (VoxelCraft). Games with no voxel chunks produce no indirect
    // commands, so canDrawChunksIndirect is false and this whole block is skipped.
    // (Magica model main-pass draws remain removed -- no current game uploads them.)
    if (frameChunkDrawData.canDrawChunksIndirect &&
        m_pipeline != VK_NULL_HANDLE &&
        chunkVertexBuffer != VK_NULL_HANDLE &&
        chunkIndexBuffer != VK_NULL_HANDLE &&
        chunkInstanceBuffer != VK_NULL_HANDLE &&
        chunkInstanceSliceOpt.has_value()) {
        vkCmdBindPipeline(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            useRtVoxelShadows ? m_pipelineRt : m_pipeline
        );
        bindGraphicsDescriptorBuffers(commandBuffer);
        const VkBuffer voxelVertexBuffers[2] = {chunkVertexBuffer, chunkInstanceBuffer};
        const VkDeviceSize voxelVertexOffsets[2] = {0, chunkInstanceSliceOpt->offset};
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, voxelVertexBuffers, voxelVertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, chunkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // Per-chunk offsets ride the instance buffer; the push constant block stays zeroed
        // so the shader's chunkOffset/cascadeData path matches the shadow pass.
        ChunkPushConstants chunkPushConstants{};
        // Neutral alpha-test threshold. Draws that carry an authored one
        // overwrite this; leaving it zeroed would mean nothing cuts out.
        chunkPushConstants.materialParams[0] = 0.5f;
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
        // Neutral alpha-test threshold. Draws that carry an authored one
        // overwrite this; leaving it zeroed would mean nothing cuts out.
        importedPushConstants.materialParams[0] = 0.5f;
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
        // The alpha-test threshold is authored per surface, so it rides the push
        // constants and is re-pushed only when consecutive draws disagree.
        // Sentinel starts outside 0-255 so the first draw always pushes.
        // ODAI_DEBUG_NO_ALPHATEST=1 pushes a threshold below every possible
        // sampled alpha, so the shader's `discard` can never fire. Diagnostic
        // twin of ODAI_DEBUG_NO_CULL: rendering with and without it and
        // diffing separates "this surface is missing because alpha test threw
        // it away" from "because it was culled" from "because it was never
        // submitted", which look identical on screen.
        static const bool s_disableAlphaTest = std::getenv("ODAI_DEBUG_NO_ALPHATEST") != nullptr;
        int lastPushedThreshold = -1;
        const auto pushAlphaThreshold = [&](std::uint8_t threshold) {
            if (static_cast<int>(threshold) == lastPushedThreshold) {
                return;
            }
            lastPushedThreshold = static_cast<int>(threshold);
            importedPushConstants.materialParams[0] =
                s_disableAlphaTest ? -1.0f : (static_cast<float>(threshold) / 255.0f);
            static const bool s_highlightAlphaTest =
                std::getenv("ODAI_DEBUG_ALPHATEST_HIGHLIGHT") != nullptr;
            importedPushConstants.materialParams[1] = s_highlightAlphaTest ? 1.0f : 0.0f;
            importedPushConstants.materialParams[2] = m_debugHighlightUntextured ? 1.0f : 0.0f;
            vkCmdPushConstants(
                commandBuffer,
                m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(ChunkPushConstants),
                &importedPushConstants);
        };
        std::uint32_t lastPushedAnimation = 0xffffffffu;
        const auto pushRigidAnimation = [&](std::uint32_t animationIndex) {
            if (animationIndex == lastPushedAnimation) {
                return;
            }
            lastPushedAnimation = animationIndex;
            importedPushConstants.rigidAnimationParams[0] =
                sampleImportedRigidAnimationTransform(
                    animationIndex, importedPushConstants.rigidAnimationTransform)
                    ? 1.0f
                    : 0.0f;
            vkCmdPushConstants(
                commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(ChunkPushConstants), &importedPushConstants);
        };
        // Opaque imported geometry, as indirect batches grouped by alpha-test
        // threshold. This is the bulk of the frame's draw calls -- roughly 3000
        // of them in the Mojave -- and they all share one vertex and one index
        // buffer, so the whole set collapses into one indirect call per
        // distinct threshold.
        // TERRAIN GOES THROUGH ITS OWN TESSELLATED PIPELINE when one exists.
        // It is drawn first, as its own batch build, because
        // buildImportedIndirectBatches reuses member scratch -- the terrain
        // batches must be recorded before the statics build overwrites them.
        // Patch-list assembly reads the same index buffer (three indices, one
        // triangle patch), so the indirect commands need no translation.
        const std::size_t tessTerrainDrawCount = std::min<std::size_t>(
            m_visibleImportedNearTerrainDrawCount, terrainDrawCount);
        const bool tessellateTerrain = mergedDepthPrepass &&
            m_importedTerrainTessPipeline != VK_NULL_HANDLE && drawTerrain &&
            tessTerrainDrawCount > 0u;
        if (tessellateTerrain) {
            const auto includeTerrainDraw = [&](std::size_t drawIndex) {
                return drawIndex < tessTerrainDrawCount;
            };
            VkBuffer terrainIndirectBuffer = VK_NULL_HANDLE;
            VkDeviceSize terrainIndirectBase = 0;
            if (m_supportsMultiDrawIndirect &&
                buildImportedIndirectBatches(
                    importedMeshDraws, includeTerrainDraw, terrainIndirectBuffer,
                    terrainIndirectBase)) {
                vkCmdBindPipeline(
                    commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_importedTerrainTessPipeline);
                for (const ImportedIndirectBatch& batch : m_importedIndirectBatches) {
                    pushAlphaThreshold(batch.alphaThreshold);
                    pushRigidAnimation(0xffffffffu);
                    countDrawCalls(m_debugDrawCallsMain, 1);
                    vkCmdDrawIndexedIndirect(
                        commandBuffer, terrainIndirectBuffer,
                        terrainIndirectBase + batch.bufferOffset, batch.drawCount,
                        sizeof(VkDrawIndexedIndirectCommand));
                }
            }
        }
        const auto includeOpaqueDraw = [&](std::size_t drawIndex) {
            if (drawIndex < terrainDrawCount) {
                // Far terrain (past the near prefix) always draws flat; the
                // near prefix draws flat only when tessellation is off.
                if (tessellateTerrain && drawIndex < tessTerrainDrawCount) {
                    return false;
                }
                return drawTerrain;
            }
            return drawStatics;
        };
        VkBuffer indirectBuffer = VK_NULL_HANDLE;
        VkDeviceSize indirectBase = 0;
        const bool useIndirect = m_supportsMultiDrawIndirect &&
            buildImportedIndirectBatches(
                importedMeshDraws, includeOpaqueDraw, indirectBuffer, indirectBase);
        if (useIndirect) {
            // DEPTH PREWRITE, in the same render pass instance as the shading
            // draws below.
            //
            // The main pass clears its own depth (loadOp CLEAR) and the SSAO
            // prepass writes into a different image at AO resolution with
            // storeOp DONT_CARE, so nothing reaches this depth buffer before the
            // shading draws do. Every occluded surface was therefore being run
            // through a forward shader carrying cascaded PCF shadows, a 64-light
            // loop, a four-layer terrain blend and PBR, only to fail the depth
            // test afterwards. Measured: disabling the main pass's own depth
            // writes -- removing what little rejection it had -- took main from
            // 25.8 to 53.4 ms, i.e. roughly half the shaded fragments were
            // already invisible.
            //
            // Laying depth first with a shader that does nothing but the alpha
            // test lets the hardware kill those fragments before any of it runs.
            // No barrier is needed: depth written by earlier draws in a render
            // pass instance is visible to later draws in the same instance by
            // rasterization order.
            //
            // The same indirect buffer is replayed, so the prewrite and the
            // shading pass are drawing exactly the same primitives by
            // construction -- there is no second culling path to drift.
            // ODAI_MAIN_PREWRITE=0 skips it, for A/B measurement and as a kill
            // switch if a driver ever disagrees about depth invariance between
            // the two pipelines.
            static const bool s_depthPrewriteEnabled = []() {
                const char* env = std::getenv("ODAI_MAIN_PREWRITE");
                return env == nullptr || (env[0] != '0');
            }();
            // Under the merged prepass this whole block is the thing being
            // deleted: the depth it would lay is already in the buffer.
            if (!mergedDepthPrepass && s_depthPrewriteEnabled &&
                m_importedStaticDepthPrewritePipeline != VK_NULL_HANDLE) {
                // Bracketed separately even though it is inside main's own
                // window: this is a whole extra rasterization of the visible
                // set, and "does laying depth first pay for itself" is not
                // answerable while its cost is folded into the number it is
                // supposed to be reducing.
                writeGpuTimestampTop(kGpuTimestampQueryPrewriteStart);
                VkPipeline boundPrewritePipeline = VK_NULL_HANDLE;
                for (const ImportedIndirectBatch& batch : m_importedIndirectBatches) {
                    VkPipeline wantedPipeline =
                        (batch.twoSided &&
                         m_importedStaticDepthPrewritePipelineTwoSided != VK_NULL_HANDLE)
                            ? m_importedStaticDepthPrewritePipelineTwoSided
                            : m_importedStaticDepthPrewritePipeline;
                    if (wantedPipeline != boundPrewritePipeline) {
                        vkCmdBindPipeline(
                            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                        boundPrewritePipeline = wantedPipeline;
                    }
                    // The alpha test must match the shading pass exactly, or a
                    // cutout texel gets depth written for a surface that is meant
                    // to be see-through.
                    pushAlphaThreshold(batch.alphaThreshold);
                    countDrawCalls(m_debugDrawCallsMain, 1);
                    vkCmdDrawIndexedIndirect(
                        commandBuffer,
                        indirectBuffer,
                        indirectBase + batch.bufferOffset,
                        batch.drawCount,
                        sizeof(VkDrawIndexedIndirectCommand));
                }
                for (std::size_t drawIndex = 0; drawIndex < importedMeshDraws.size(); ++drawIndex) {
                    const ImportedMeshDraw& draw = importedMeshDraws[drawIndex];
                    if (draw.blended || draw.rigidAnimationIndex == 0xffffffffu ||
                        !includeOpaqueDraw(drawIndex)) {
                        continue;
                    }
                    const VkPipeline wantedPipeline =
                        (draw.twoSided && m_importedStaticDepthPrewritePipelineTwoSided != VK_NULL_HANDLE)
                            ? m_importedStaticDepthPrewritePipelineTwoSided
                            : m_importedStaticDepthPrewritePipeline;
                    if (wantedPipeline != boundPrewritePipeline) {
                        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                        boundPrewritePipeline = wantedPipeline;
                    }
                    pushAlphaThreshold(draw.alphaThreshold);
                    pushRigidAnimation(draw.rigidAnimationIndex);
                    countDrawCalls(m_debugDrawCallsMain, 1);
                    vkCmdDrawIndexed(
                        commandBuffer, draw.indexCount, 1, draw.firstIndex,
                        draw.vertexOffset, 0);
                }
                writeGpuTimestampBottom(kGpuTimestampQueryPrewriteEnd);
            }

            // Two-sidedness is a pipeline switch, so the batch order (grouped
            // by it) is also the bind order -- at most one extra bind.
            VkPipeline boundOpaquePipeline = VK_NULL_HANDLE;
            for (const ImportedIndirectBatch& batch : m_importedIndirectBatches) {
                VkPipeline wantedPipeline =
                    (batch.twoSided && m_importedStaticPipelineTwoSided != VK_NULL_HANDLE)
                        ? m_importedStaticPipelineTwoSided
                        : ((useRtMainShadows && m_importedStaticPipelineRt != VK_NULL_HANDLE)
                               ? m_importedStaticPipelineRt
                               : m_importedStaticPipeline);
                if (wantedPipeline != boundOpaquePipeline) {
                    vkCmdBindPipeline(
                        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                    boundOpaquePipeline = wantedPipeline;
                }
                pushAlphaThreshold(batch.alphaThreshold);
                pushRigidAnimation(0xffffffffu);
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexedIndirect(
                    commandBuffer, indirectBuffer, indirectBase + batch.bufferOffset,
                    batch.drawCount, sizeof(VkDrawIndexedIndirectCommand));
            }
            for (std::size_t drawIndex = 0; drawIndex < importedMeshDraws.size(); ++drawIndex) {
                const ImportedMeshDraw& draw = importedMeshDraws[drawIndex];
                if (draw.blended || draw.rigidAnimationIndex == 0xffffffffu ||
                    !includeOpaqueDraw(drawIndex)) {
                    continue;
                }
                const VkPipeline wantedPipeline =
                    (draw.twoSided && m_importedStaticPipelineTwoSided != VK_NULL_HANDLE)
                        ? m_importedStaticPipelineTwoSided
                        : ((useRtMainShadows && m_importedStaticPipelineRt != VK_NULL_HANDLE)
                               ? m_importedStaticPipelineRt
                               : m_importedStaticPipeline);
                if (wantedPipeline != boundOpaquePipeline) {
                    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                    boundOpaquePipeline = wantedPipeline;
                }
                pushAlphaThreshold(draw.alphaThreshold);
                pushRigidAnimation(draw.rigidAnimationIndex);
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexed(
                    commandBuffer, draw.indexCount, 1, draw.firstIndex,
                    draw.vertexOffset, 0);
            }
            if (boundOpaquePipeline != VK_NULL_HANDLE) {
                // Leave the opaque pipeline bound for the blended replay below,
                // which assumes it and only rebinds when it wants a different one.
                vkCmdBindPipeline(
                    commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    (useRtMainShadows && m_importedStaticPipelineRt != VK_NULL_HANDLE)
                        ? m_importedStaticPipelineRt
                        : m_importedStaticPipeline);
            }
        } else {
            // Direct fallback: no multiDrawIndirect, or the frame arena could
            // not serve the command buffer this frame.
            for (std::size_t drawIndex = 0; drawIndex < importedMeshDraws.size(); ++drawIndex) {
                if (!includeOpaqueDraw(drawIndex)) {
                    continue;
                }
                const ImportedMeshDraw& importedDraw = importedMeshDraws[drawIndex];
                if (importedDraw.blended) {
                    continue;  // replayed below, in back-to-front order
                }
                pushAlphaThreshold(importedDraw.alphaThreshold);
                pushRigidAnimation(importedDraw.rigidAnimationIndex);
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexed(
                    commandBuffer, importedDraw.indexCount, 1, importedDraw.firstIndex,
                    importedDraw.vertexOffset, 0);
            }
        }

        // Blended tail, farthest first. Same vertex/index buffers and push
        // constants as the opaque draws, so only the pipeline changes; if that
        // pipeline failed to create, the draws still go out through the opaque
        // one rather than vanishing.
        if (!importedBlendedDrawOrder.empty()) {
            // Two-sidedness is a per-draw property (NiStencilProperty DRAW_BOTH
            // on the source shape), so the pipeline is resolved per draw and
            // rebound only when it actually changes. The sorted order is what
            // decides the sequence -- correctness of the compositing outranks
            // the handful of extra binds a mixed run costs.
            VkPipeline boundBlendedPipeline = VK_NULL_HANDLE;
            for (const std::uint32_t drawIndex : importedBlendedDrawOrder) {
                if (drawIndex >= importedMeshDraws.size()) {
                    continue;
                }
                if ((drawIndex < terrainDrawCount && !drawTerrain) ||
                    (drawIndex >= staticDrawStart && !drawStatics)) {
                    continue;
                }
                const ImportedMeshDraw& importedDraw = importedMeshDraws[drawIndex];
                pushAlphaThreshold(importedDraw.alphaThreshold);
                pushRigidAnimation(importedDraw.rigidAnimationIndex);
                VkPipeline wantedPipeline =
                    (importedDraw.twoSided && m_importedStaticPipelineBlendedTwoSided != VK_NULL_HANDLE)
                        ? m_importedStaticPipelineBlendedTwoSided
                        : m_importedStaticPipelineBlended;
                if (wantedPipeline != VK_NULL_HANDLE && wantedPipeline != boundBlendedPipeline) {
                    vkCmdBindPipeline(
                        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                    boundBlendedPipeline = wantedPipeline;
                }
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexed(
                    commandBuffer, importedDraw.indexCount, 1, importedDraw.firstIndex,
                    importedDraw.vertexOffset, 0);
            }
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
        importedPushConstants.materialParams[2] = m_debugHighlightUntextured ? 1.0f : 0.0f;
        // Per-draw, exactly as the static block above does it. This used to push
        // a single hardcoded 0.5 for every actor in the scene with a comment
        // claiming the draws below overwrote it -- nothing did, so an actor's
        // authored NiAlphaProperty threshold never reached the shader.
        auto pushActorState = [&](std::uint8_t alphaThreshold) {
            importedPushConstants.materialParams[0] =
                static_cast<float>(alphaThreshold) / 255.0f;
            vkCmdPushConstants(
                commandBuffer,
                m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(ChunkPushConstants),
                &importedPushConstants
            );
        };
        const VkPipeline actorOpaqueDefaultPipeline =
            (useRtMainShadows && m_importedStaticPipelineRt != VK_NULL_HANDLE)
                ? m_importedStaticPipelineRt
                : m_importedStaticPipeline;
        VkPipeline boundActorPipeline = actorOpaqueDefaultPipeline;
        for (std::size_t drawIndex = 0; drawIndex < importedActorMeshDraws.size(); ++drawIndex) {
            const ImportedMeshDraw& importedDraw = importedActorMeshDraws[drawIndex];
            // Blended actor parts are replayed sorted, after the opaque ones.
            if (importedDraw.blended) {
                continue;
            }
            const VkPipeline wantedPipeline =
                (importedDraw.twoSided && m_importedStaticPipelineTwoSided != VK_NULL_HANDLE)
                    ? m_importedStaticPipelineTwoSided
                    : actorOpaqueDefaultPipeline;
            if (wantedPipeline != boundActorPipeline) {
                vkCmdBindPipeline(
                    commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                boundActorPipeline = wantedPipeline;
            }
            pushActorState(importedDraw.alphaThreshold);
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(
                commandBuffer, importedDraw.indexCount, 1, importedDraw.firstIndex,
                importedDraw.vertexOffset, 0);
        }
        // Blended tail, farthest first -- the same treatment the static scene
        // gets. Without it an actor's alpha-blended parts (hair cards, eye
        // lashes, the glare quads on Victor's face screen) went through the
        // opaque pipeline and rendered as solid slabs of whatever colour sits
        // under their transparent texels, which for a Fallout texture is black.
        VkPipeline boundActorBlendedPipeline = VK_NULL_HANDLE;
        for (const std::uint32_t drawIndex : importedActorBlendedDrawOrder) {
            if (drawIndex >= importedActorMeshDraws.size()) {
                continue;
            }
            const ImportedMeshDraw& importedDraw = importedActorMeshDraws[drawIndex];
            const VkPipeline wantedPipeline =
                (importedDraw.twoSided && m_importedStaticPipelineBlendedTwoSided != VK_NULL_HANDLE)
                    ? m_importedStaticPipelineBlendedTwoSided
                    : m_importedStaticPipelineBlended;
            if (wantedPipeline != VK_NULL_HANDLE && wantedPipeline != boundActorBlendedPipeline) {
                vkCmdBindPipeline(
                    commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                boundActorBlendedPipeline = wantedPipeline;
            }
            pushActorState(importedDraw.alphaThreshold);
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDrawIndexed(
                commandBuffer, importedDraw.indexCount, 1, importedDraw.firstIndex,
                importedDraw.vertexOffset, 0);
        }
    }
    // GPU-skinned actors (Dragon Age touchstone) -- same pipeline as the
    // CPU-skinned block above (skinning_resources.cc's output buffers are laid
    // out exactly like ImportedMeshVertex). Up to kMaxSkinnedInstances
    // independent instance slots means each draw may come from a different
    // vertex/index buffer, so bind is resolved per-draw.
    if (m_importedStaticPipeline != VK_NULL_HANDLE &&
        !skinnedActorMeshDraws.empty() &&
        m_debugShowImportedStatics) {
        vkCmdBindPipeline(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            (useRtMainShadows && m_importedStaticPipelineRt != VK_NULL_HANDLE)
                ? m_importedStaticPipelineRt
                : m_importedStaticPipeline
        );
        bindGraphicsDescriptorBuffers(commandBuffer);
        ChunkPushConstants skinnedPushConstants{};
        skinnedPushConstants.cascadeData[1] = m_importedSceneInteriorMode ? 1.0f : 0.0f;
        skinnedPushConstants.cascadeData[2] = m_debugShowImportedTextures ? 0.0f : 1.0f;
        skinnedPushConstants.cascadeData[3] = m_debugImportedFlatShading ? 1.0f : 0.0f;
        // Per draw, for the same reason as the actor block above. Skinned draws
        // are not sorted into a blended tail: the skinning path produces one
        // instance slot per actor and its parts are already filtered of the
        // alpha-blended glare quads (see FalloutCharacter part selection), so
        // there is nothing here to sort. Two-sidedness still applies -- a
        // dust-mask or a coat flap is authored DRAW_BOTH like any other thin
        // surface.
        skinnedPushConstants.materialParams[2] = m_debugHighlightUntextured ? 1.0f : 0.0f;
        auto pushSkinnedState = [&](std::uint8_t alphaThreshold) {
            skinnedPushConstants.materialParams[0] =
                static_cast<float>(alphaThreshold) / 255.0f;
            vkCmdPushConstants(
                commandBuffer,
                m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(ChunkPushConstants),
                &skinnedPushConstants
            );
        };
        const VkPipeline skinnedDefaultPipeline =
            (useRtMainShadows && m_importedStaticPipelineRt != VK_NULL_HANDLE)
                ? m_importedStaticPipelineRt
                : m_importedStaticPipeline;
        VkPipeline boundSkinnedPipeline = skinnedDefaultPipeline;
        VkBuffer boundSkinnedVertexBuffer = VK_NULL_HANDLE;
        VkBuffer boundSkinnedIndexBuffer = VK_NULL_HANDLE;
        // Two passes over the same list: opaque parts, then blended ones after
        // them so they composite over what they cover. Not distance-sorted, and
        // deliberately so -- these are the parts of ONE actor, already in the
        // NIF's own part order, and there is no camera-dependent answer to
        // "which of a character's own hair cards is in front" that a per-draw
        // AABB centre would get right anyway.
        const auto drawSkinned = [&](bool wantBlended) {
            for (const ImportedMeshDraw& skinnedDraw : skinnedActorMeshDraws) {
                if (skinnedDraw.blended != wantBlended) {
                    continue;
                }
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
                VkPipeline wantedPipeline = VK_NULL_HANDLE;
                if (wantBlended) {
                    wantedPipeline =
                        (skinnedDraw.twoSided && m_importedStaticPipelineBlendedTwoSided != VK_NULL_HANDLE)
                            ? m_importedStaticPipelineBlendedTwoSided
                            : m_importedStaticPipelineBlended;
                    if (wantedPipeline == VK_NULL_HANDLE) {
                        wantedPipeline = skinnedDefaultPipeline;
                    }
                } else {
                    wantedPipeline =
                        (skinnedDraw.twoSided && m_importedStaticPipelineTwoSided != VK_NULL_HANDLE)
                            ? m_importedStaticPipelineTwoSided
                            : skinnedDefaultPipeline;
                }
                if (wantedPipeline != boundSkinnedPipeline) {
                    vkCmdBindPipeline(
                        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                    boundSkinnedPipeline = wantedPipeline;
                }
                pushSkinnedState(skinnedDraw.alphaThreshold);
                countDrawCalls(m_debugDrawCallsMain, 1);
                vkCmdDrawIndexed(commandBuffer, skinnedDraw.indexCount, 1, skinnedDraw.firstIndex, 0, 0);
            }
        };
        drawSkinned(false);
        drawSkinned(true);
    }

    // (removed) pipe / belt / transport instanced main-pass draws — legacy factory-sim
    // rendering from the prior game; the strategy map has no pipes or conveyors.

    // (removed) grass billboard main-pass draw. The billboards contributed nothing legible
    // from the camera while their shadow casters scattered dark streaks across the ground;
    // chunk_upload.cc no longer scatters instances, so this had nothing left to draw.

    // Imported looping effects. Fire is analytic and stateless on the GPU:
    // SV_InstanceID identifies a lobe, the shared frame clock advances its age,
    // and six generated vertices form its camera-facing quad. That keeps a
    // whole hearth to one draw with no per-frame particle-buffer upload or
    // transfer/graphics barrier. Additive blending is order independent, so
    // emitters from several live Bethesda cells do not need a sort.
    if (m_importedFireParticlePipeline != VK_NULL_HANDLE &&
        !m_importedParticleEmitters.empty()) {
        vkCmdBindPipeline(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_importedFireParticlePipeline);
        bindGraphicsDescriptorBuffers(commandBuffer);
        if (m_supportsVrs && m_cmdSetFragmentShadingRate != nullptr) {
            const VkExtent2D fineRate{1u, 1u};
            const VkFragmentShadingRateCombinerOpKHR combinerOps[2] = {
                VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
                VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
            };
            m_cmdSetFragmentShadingRate(commandBuffer, &fineRate, combinerOps);
        }
        for (const odai::importer::ImportedSceneParticleEmitter& emitter :
             m_importedParticleEmitters) {
            if (emitter.effect != odai::importer::ImportedParticleEffect::Fire ||
                emitter.particleCount == 0u || emitter.intensity <= 0.0f) {
                continue;
            }
            ChunkPushConstants push{};
            std::memcpy(push.chunkOffset, emitter.position, sizeof(emitter.position));
            push.chunkOffset[3] = std::max(emitter.spawnRadius, 0.0f);
            std::memcpy(push.cascadeData, emitter.color, sizeof(emitter.color));
            push.cascadeData[3] = std::max(emitter.intensity, 0.0f);
            push.materialParams[0] = std::max(emitter.particleLifetime, 0.05f);
            push.materialParams[1] = std::max(emitter.upwardSpeed, 0.0f);
            push.materialParams[2] = std::max(emitter.particleSize, 0.5f);
            // Keep the seed small enough that adding SV_InstanceID remains
            // exact in float. A 24-bit form-derived seed put many values near
            // float's integer precision limit and collapsed adjacent lobes
            // onto one billboard instead of spreading them across the fire.
            push.materialParams[3] = static_cast<float>(emitter.seed & 0x00000fffu);
            vkCmdPushConstants(
                commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(push), &push);
            countDrawCalls(m_debugDrawCallsMain, 1);
            vkCmdDraw(
                commandBuffer, 6u,
                std::clamp(emitter.particleCount, 1u, 256u), 0u, 0u);
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
        opaqueCopyRegion.extent = {m_renderExtent.width, m_renderExtent.height, 1u};
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
            msaaEnabled ? m_msaaColorImages[imageIndex] : m_hdrResolveImages[aoFrameIndex],
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
            // Neutral alpha-test threshold. Draws that carry an authored one
            // overwrite this; leaving it zeroed would mean nothing cuts out.
            waterPushConstants.materialParams[0] = 0.5f;
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
    if (renderImportedSky && m_skyboxPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_skyboxPipeline);
        bindGraphicsDescriptorBuffers(commandBuffer);
        // Do not inherit the planar pass's reflection flag in a frame where no
        // later imported draw happened to overwrite the push constants.
        ChunkPushConstants skyPushConstants{};
        vkCmdPushConstants(
            commandBuffer, m_pipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(skyPushConstants), &skyPushConstants);
        countDrawCalls(m_debugDrawCallsMain, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    if (renderImportedSky &&
        m_skyCloudPipeline != VK_NULL_HANDLE &&
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
