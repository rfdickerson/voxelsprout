#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "render/backend/vulkan/frame_graph_runtime.h"

namespace odai::render {

#include "render/renderer_shared.h"

namespace {

constexpr float kImportedShadowConstantBiasScale = 1.9f;
constexpr float kImportedShadowSlopeBiasScale = 2.2f;

}  // namespace

void RendererBackend::recordShadowAtlasPass(const FrameExecutionContext& context, const ShadowPassInputs& inputs) {
    VkCommandBuffer commandBuffer = context.commandBuffer;
    VkQueryPool gpuTimestampQueryPool = context.gpuTimestampQueryPool;
    CoreFrameGraphOrderValidator& coreFramePassOrderValidator = *context.frameOrderValidator;
    const CoreFrameGraphPlan& coreFrameGraphPlan = *context.frameGraphPlan;
    // Voxel chunk shadow inputs: consumed by the per-cascade caster draw below
    // (VoxelCraft). The magica/pipe inputs remain unconsumed here.
    const FrameChunkDrawData& frameChunkDrawData = *inputs.frameChunkDrawData;
    const std::optional<FrameArenaSlice>& shadowChunkInstanceSliceOpt = *inputs.shadowChunkInstanceSliceOpt;
    const VkBuffer shadowChunkInstanceBuffer = inputs.shadowChunkInstanceBuffer;
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
    const bool importedPageCullingEnabled = inputs.importedPageCullingEnabled;

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

    writeGpuTimestampTop(kGpuTimestampQueryShadowStart);
    coreFramePassOrderValidator.markPassEntered(coreFrameGraphPlan.shadow, "shadow");
    if (inputs.renderInteriorPointShadows &&
        inputs.interiorPointShadowLightCount > 0 &&
        m_importedStaticShadowPipeline != VK_NULL_HANDLE &&
        importedVertexBuffer != VK_NULL_HANDLE &&
        importedIndexBuffer != VK_NULL_HANDLE &&
        !importedMeshDraws.empty()) {
        beginDebugLabel(commandBuffer, "Pass: Interior Point Shadow Atlas", 0.34f, 0.20f, 0.16f, 1.0f);
        transitionImageLayout(
            commandBuffer,
            m_shadowDepthImage,
            m_interiorPointShadowAtlasValid
                ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            m_interiorPointShadowAtlasValid
                ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                : VK_PIPELINE_STAGE_2_NONE,
            m_interiorPointShadowAtlasValid
                ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            0,
            1);

        const bool useCompactShadowStream =
            m_importedStaticShadowCompactPipeline != VK_NULL_HANDLE &&
            inputs.importedShadowVertexBuffer != VK_NULL_HANDLE;
        const VkPipeline oneSidedPipeline =
            useCompactShadowStream ? m_importedStaticShadowCompactPipeline
                                   : m_importedStaticShadowPipeline;
        const VkPipeline twoSidedPipeline =
            useCompactShadowStream ? m_importedStaticShadowCompactPipelineTwoSided
                                   : m_importedStaticShadowPipelineTwoSided;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, oneSidedPipeline);
        bindGraphicsDescriptorBuffers(commandBuffer);
        const VkBuffer pointShadowVertexBuffers[1] = {
            useCompactShadowStream ? inputs.importedShadowVertexBuffer : importedVertexBuffer};
        const VkDeviceSize pointShadowVertexOffsets[1] = {0};
        vkCmdBindVertexBuffers(
            commandBuffer, 0, 1, pointShadowVertexBuffers, pointShadowVertexOffsets);
        vkCmdBindIndexBuffer(commandBuffer, importedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

        const auto includeInteriorCaster = [&](std::size_t) {
            return m_debugShowImportedStatics;
        };
        VkBuffer indirectBuffer = VK_NULL_HANDLE;
        VkDeviceSize indirectBase = 0;
        const bool useIndirect = m_supportsMultiDrawIndirect &&
            buildImportedIndirectBatches(
                importedMeshDraws, includeInteriorCaster, indirectBuffer, indirectBase);
        const std::uint32_t shadowLightCount = std::min<std::uint32_t>(
            inputs.interiorPointShadowLightCount, kInteriorPointShadowLightCount);
        VkClearValue depthClear{};
        depthClear.depthStencil.depth = 0.0f;
        vkCmdSetDepthBias(
            commandBuffer,
            -(m_shadowDebugSettings.casterConstantBiasBase * kImportedShadowConstantBiasScale),
            0.0f,
            -(m_shadowDebugSettings.casterSlopeBiasBase * kImportedShadowSlopeBiasScale));

        for (std::uint32_t slot = 0; slot < shadowLightCount; ++slot) {
            const std::uint32_t faceSize = kInteriorPointShadowFaceSize;
            const std::uint32_t cubeX =
                (slot % kInteriorPointShadowCubesPerRow) * (3u * faceSize);
            const std::uint32_t cubeY =
                (slot / kInteriorPointShadowCubesPerRow) * (2u * faceSize);
            for (std::uint32_t face = 0; face < kInteriorPointShadowFaceCount; ++face) {
                const std::uint32_t faceX =
                    cubeX + ((face % 3u) * faceSize);
                const std::uint32_t faceY =
                    cubeY + ((face / 3u) * faceSize);
                VkRect2D faceRect{};
                faceRect.offset = {
                    static_cast<std::int32_t>(faceX), static_cast<std::int32_t>(faceY)};
                faceRect.extent = {
                    faceSize, faceSize};
                VkViewport faceViewport{};
                faceViewport.x = static_cast<float>(faceX);
                faceViewport.y = static_cast<float>(faceY);
                faceViewport.width = static_cast<float>(faceSize);
                faceViewport.height = static_cast<float>(faceSize);
                faceViewport.minDepth = 0.0f;
                faceViewport.maxDepth = 1.0f;

                VkRenderingAttachmentInfo depthAttachment{};
                depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                depthAttachment.imageView = m_shadowDepthImageView;
                depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
                depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                depthAttachment.clearValue = depthClear;
                VkRenderingInfo renderingInfo{};
                renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
                renderingInfo.renderArea = faceRect;
                renderingInfo.layerCount = 1;
                renderingInfo.pDepthAttachment = &depthAttachment;
                vkCmdBeginRendering(commandBuffer, &renderingInfo);
                vkCmdSetViewport(commandBuffer, 0, 1, &faceViewport);
                vkCmdSetScissor(commandBuffer, 0, 1, &faceRect);

                ChunkPushConstants pointPush{};
                pointPush.cascadeData[0] = static_cast<float>(
                    (slot * kInteriorPointShadowFaceCount) + face);
                pointPush.cascadeData[1] = 1.0f;
                pointPush.materialParams[0] = 0.5f;
                int lastThreshold = -1;
                const auto pushPointThreshold = [&](std::uint8_t threshold) {
                    if (static_cast<int>(threshold) == lastThreshold) {
                        return;
                    }
                    lastThreshold = static_cast<int>(threshold);
                    pointPush.materialParams[0] = static_cast<float>(threshold) / 255.0f;
                    vkCmdPushConstants(
                        commandBuffer,
                        m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        0,
                        sizeof(ChunkPushConstants),
                        &pointPush);
                };
                std::uint32_t lastPointAnimation = 0xffffffffu;
                const auto pushPointAnimation = [&](std::uint32_t animationIndex) {
                    if (animationIndex == lastPointAnimation) {
                        return;
                    }
                    lastPointAnimation = animationIndex;
                    pointPush.rigidAnimationParams[0] =
                        sampleImportedRigidAnimationTransform(
                            animationIndex, pointPush.rigidAnimationTransform)
                            ? 1.0f
                            : 0.0f;
                    vkCmdPushConstants(
                        commandBuffer, m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        0, sizeof(ChunkPushConstants), &pointPush);
                };
                if (useIndirect) {
                    VkPipeline boundPipeline = oneSidedPipeline;
                    vkCmdBindPipeline(
                        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, boundPipeline);
                    for (const ImportedIndirectBatch& batch : m_importedIndirectBatches) {
                        const VkPipeline wantedPipeline =
                            (batch.twoSided && twoSidedPipeline != VK_NULL_HANDLE)
                                ? twoSidedPipeline
                                : oneSidedPipeline;
                        if (wantedPipeline != boundPipeline) {
                            vkCmdBindPipeline(
                                commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                            boundPipeline = wantedPipeline;
                        }
                        pushPointThreshold(batch.alphaThreshold);
                        pushPointAnimation(0xffffffffu);
                        countDrawCalls(m_debugDrawCallsShadow, 1);
                        vkCmdDrawIndexedIndirect(
                            commandBuffer,
                            indirectBuffer,
                            indirectBase + batch.bufferOffset,
                            batch.drawCount,
                            sizeof(VkDrawIndexedIndirectCommand));
                    }
                }
                for (const ImportedMeshDraw& draw : importedMeshDraws) {
                    if (draw.blended || !m_debugShowImportedStatics ||
                        (useIndirect && draw.rigidAnimationIndex == 0xffffffffu)) {
                        continue;
                    }
                    const VkPipeline wantedPipeline =
                        (draw.twoSided && twoSidedPipeline != VK_NULL_HANDLE)
                            ? twoSidedPipeline
                            : oneSidedPipeline;
                    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                    pushPointThreshold(draw.alphaThreshold);
                    pushPointAnimation(draw.rigidAnimationIndex);
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(
                        commandBuffer, draw.indexCount, 1, draw.firstIndex,
                        draw.vertexOffset, 0);
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
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            0,
            1);
        vkCmdSetDepthBias(commandBuffer, 0.0f, 0.0f, 0.0f);
        m_interiorPointShadowAtlasValid = true;
        m_shadowRenderedValid = {};
        VOX_LOGI("render") << "interior point shadow atlas rebuilt: lights="
                           << shadowLightCount << ", faces="
                           << (shadowLightCount * kInteriorPointShadowFaceCount)
                           << ", faceSize=" << kInteriorPointShadowFaceSize;
        endDebugLabel(commandBuffer);
        writeGpuTimestampBottom(kGpuTimestampQueryShadowEnd);
        return;
    }
    if (inputs.skipDirectionalShadows) {
        writeGpuTimestampBottom(kGpuTimestampQueryShadowEnd);
        return;
    }
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

    // Shadow caster pass. Gated on the imported-static shadow pipeline (strategy-map
    // settlements/units); the prior game's voxel chunk + magica shadow draws were removed.
    if (m_importedStaticShadowPipeline != VK_NULL_HANDLE) {
        for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
            if ((inputs.skipCascadeMask & (1u << cascadeIndex)) != 0u) {
                // This cascade's tile is being reused -- see skipCascadeMask's
                // declaration. Skipping the whole begin/end block is what keeps
                // the tile intact: the clear is scoped to the block's renderArea.
                continue;
            }
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

            // Voxel chunk shadow casters for this cascade. Skipped entirely when a game
            // has no voxel chunks (no per-cascade indirect commands are produced).
            // (Magica model shadow casters remain removed.)
            if (m_shadowPipeline != VK_NULL_HANDLE &&
                frameChunkDrawData.canDrawShadowChunksIndirectByCascade[cascadeIndex] &&
                shadowChunkInstanceSliceOpt.has_value() &&
                shadowChunkInstanceBuffer != VK_NULL_HANDLE &&
                chunkVertexBuffer != VK_NULL_HANDLE &&
                chunkIndexBuffer != VK_NULL_HANDLE) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);
                bindGraphicsDescriptorBuffers(commandBuffer);
                const VkBuffer voxelVertexBuffers[2] = {chunkVertexBuffer, shadowChunkInstanceBuffer};
                const VkDeviceSize voxelVertexOffsets[2] = {0, shadowChunkInstanceSliceOpt->offset};
                vkCmdBindVertexBuffers(commandBuffer, 0, 2, voxelVertexBuffers, voxelVertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, chunkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

                // cascadeData[0] selects the light matrix for this cascade in the shadow
                // vertex shader; chunk offsets ride the instance buffer.
                ChunkPushConstants chunkPushConstants{};
                // Neutral alpha-test threshold. Draws that carry an authored one
                // overwrite this; leaving it zeroed would mean nothing cuts out.
                chunkPushConstants.materialParams[0] = 0.5f;
                chunkPushConstants.cascadeData[0] = cascadeF;
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &chunkPushConstants
                );
                drawIndirectShadowChunkRanges(
                    commandBuffer, m_debugDrawCallsShadow, cascadeIndex, frameChunkDrawData);
            }

            const std::span<const ImportedMeshDraw> cascadeImportedMeshDraws =
                importedPageCullingEnabled ? inputs.importedMeshDrawsByCascade[cascadeIndex] : importedMeshDraws;
            const std::uint32_t cascadeImportedTerrainDrawCount =
                importedPageCullingEnabled
                    ? inputs.importedTerrainDrawCountsByCascade[cascadeIndex]
                    : importedTerrainDrawCount;
            if (m_importedStaticShadowPipeline != VK_NULL_HANDLE &&
                importedVertexBuffer != VK_NULL_HANDLE &&
                importedIndexBuffer != VK_NULL_HANDLE &&
                !cascadeImportedMeshDraws.empty()) {
                const std::size_t terrainDrawCount =
                    std::min<std::size_t>(cascadeImportedTerrainDrawCount, cascadeImportedMeshDraws.size());
                vkCmdSetDepthBias(
                    commandBuffer,
                    -(constantBias * kImportedShadowConstantBiasScale),
                    0.0f,
                    -(slopeBias * kImportedShadowSlopeBiasScale));
                // Prefer the 28-byte stream. Same shaders and same draws; the
                // cascades read position plus the alpha-test fields and nothing
                // else, so the full 72-byte stride was wasting most of every
                // cache line, four times per frame. Falls back to the full
                // vertex if the compact pipeline or buffer is unavailable.
                const bool useCompactShadowStream =
                    m_importedStaticShadowCompactPipeline != VK_NULL_HANDLE &&
                    inputs.importedShadowVertexBuffer != VK_NULL_HANDLE;
                vkCmdBindPipeline(
                    commandBuffer,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    useCompactShadowStream ? m_importedStaticShadowCompactPipeline
                                           : m_importedStaticShadowPipeline);
                bindGraphicsDescriptorBuffers(commandBuffer);
                const VkBuffer importedVertexBuffers[1] = {
                    useCompactShadowStream ? inputs.importedShadowVertexBuffer : importedVertexBuffer};
                const VkDeviceSize importedVertexOffsets[1] = {0};
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, importedVertexBuffers, importedVertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, importedIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
                ChunkPushConstants importedPushConstants{};
                // Neutral alpha-test threshold. Draws that carry an authored one
                // overwrite this; leaving it zeroed would mean nothing cuts out.
                importedPushConstants.materialParams[0] = 0.5f;
                importedPushConstants.chunkOffset[0] = 0.0f;
                importedPushConstants.chunkOffset[1] = 0.0f;
                importedPushConstants.chunkOffset[2] = 0.0f;
                importedPushConstants.chunkOffset[3] = 0.0f;
                importedPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                importedPushConstants.cascadeData[1] = 0.0f;
                importedPushConstants.cascadeData[2] = 0.0f;
                importedPushConstants.cascadeData[3] = 0.0f;
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &importedPushConstants
                );
                // Same authored threshold as the main pass. A cascade that cut
                // out at a different alpha would cast a shadow whose silhouette
                // did not match the thing casting it.
                int lastPushedThreshold = -1;
                const auto pushAlphaThreshold = [&](std::uint8_t threshold) {
                    if (static_cast<int>(threshold) == lastPushedThreshold) {
                        return;
                    }
                    lastPushedThreshold = static_cast<int>(threshold);
                    importedPushConstants.materialParams[0] =
                        static_cast<float>(threshold) / 255.0f;
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
                // Indirect batching, rebuilt per cascade because each cascade
                // culls its own caster list. This is the biggest single block
                // of draw calls in the frame -- four cascades over the whole
                // caster set -- so it is also where batching pays the most.
                const auto includeDraw = [&](std::size_t drawIndex) {
                    if (drawIndex < terrainDrawCount) {
                        return m_debugShowImportedTerrain;
                    }
                    return m_debugShowImportedStatics;
                };
                VkBuffer indirectBuffer = VK_NULL_HANDLE;
                VkDeviceSize indirectBase = 0;
                const bool useIndirect = m_supportsMultiDrawIndirect &&
                    buildImportedIndirectBatches(
                        cascadeImportedMeshDraws, includeDraw, indirectBuffer, indirectBase);
                if (useIndirect) {
                    // Two-sided casters need cull NONE, matching the main pass.
                    // The variant has to follow the same compact/full stream
                    // choice the bind above made, since the two pipelines
                    // differ in vertex input.
                    const VkPipeline oneSidedShadowPipeline =
                        useCompactShadowStream ? m_importedStaticShadowCompactPipeline
                                               : m_importedStaticShadowPipeline;
                    const VkPipeline twoSidedShadowPipeline =
                        useCompactShadowStream ? m_importedStaticShadowCompactPipelineTwoSided
                                               : m_importedStaticShadowPipelineTwoSided;
                    VkPipeline boundShadowPipeline = oneSidedShadowPipeline;
                    for (const ImportedIndirectBatch& batch : m_importedIndirectBatches) {
                        const VkPipeline wantedPipeline =
                            (batch.twoSided && twoSidedShadowPipeline != VK_NULL_HANDLE)
                                ? twoSidedShadowPipeline
                                : oneSidedShadowPipeline;
                        if (wantedPipeline != boundShadowPipeline) {
                            vkCmdBindPipeline(
                                commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                            boundShadowPipeline = wantedPipeline;
                        }
                        pushAlphaThreshold(batch.alphaThreshold);
                        pushRigidAnimation(0xffffffffu);
                        countDrawCalls(m_debugDrawCallsShadow, 1);
                        vkCmdDrawIndexedIndirect(
                            commandBuffer, indirectBuffer, indirectBase + batch.bufferOffset,
                            batch.drawCount, sizeof(VkDrawIndexedIndirectCommand));
                    }
                    for (std::size_t drawIndex = 0;
                         drawIndex < cascadeImportedMeshDraws.size(); ++drawIndex) {
                        const ImportedMeshDraw& draw = cascadeImportedMeshDraws[drawIndex];
                        if (draw.blended || draw.rigidAnimationIndex == 0xffffffffu ||
                            !includeDraw(drawIndex)) {
                            continue;
                        }
                        const VkPipeline wantedPipeline =
                            (draw.twoSided && twoSidedShadowPipeline != VK_NULL_HANDLE)
                                ? twoSidedShadowPipeline
                                : oneSidedShadowPipeline;
                        if (wantedPipeline != boundShadowPipeline) {
                            vkCmdBindPipeline(
                                commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, wantedPipeline);
                            boundShadowPipeline = wantedPipeline;
                        }
                        pushAlphaThreshold(draw.alphaThreshold);
                        pushRigidAnimation(draw.rigidAnimationIndex);
                        countDrawCalls(m_debugDrawCallsShadow, 1);
                        vkCmdDrawIndexed(
                            commandBuffer, draw.indexCount, 1, draw.firstIndex,
                            draw.vertexOffset, 0);
                    }
                    if (boundShadowPipeline != oneSidedShadowPipeline) {
                        vkCmdBindPipeline(
                            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, oneSidedShadowPipeline);
                    }
                } else {
                    for (std::size_t drawIndex = 0; drawIndex < cascadeImportedMeshDraws.size(); ++drawIndex) {
                        if (!includeDraw(drawIndex)) {
                            continue;
                        }
                        const ImportedMeshDraw& importedDraw = cascadeImportedMeshDraws[drawIndex];
                        if (importedDraw.blended) {
                            continue;  // a blended surface casts no opaque shadow
                        }
                        pushAlphaThreshold(importedDraw.alphaThreshold);
                        pushRigidAnimation(importedDraw.rigidAnimationIndex);
                        countDrawCalls(m_debugDrawCallsShadow, 1);
                        vkCmdDrawIndexed(
                            commandBuffer, importedDraw.indexCount, 1, importedDraw.firstIndex,
                            importedDraw.vertexOffset, 0);
                    }
                }
                vkCmdSetDepthBias(commandBuffer, -constantBias, 0.0f, -slopeBias);
            }
            if (m_importedStaticShadowPipeline != VK_NULL_HANDLE &&
                importedActorVertexBuffer != VK_NULL_HANDLE &&
                importedActorIndexBuffer != VK_NULL_HANDLE &&
                !importedActorMeshDraws.empty() &&
                m_debugShowImportedStatics) {
                vkCmdSetDepthBias(
                    commandBuffer,
                    -(constantBias * kImportedShadowConstantBiasScale),
                    0.0f,
                    -(slopeBias * kImportedShadowSlopeBiasScale));
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_importedStaticShadowPipeline);
                bindGraphicsDescriptorBuffers(commandBuffer);
                const VkBuffer importedVertexBuffers[1] = {importedActorVertexBuffer};
                const VkDeviceSize importedVertexOffsets[1] = {importedActorVertexOffset};
                vkCmdBindVertexBuffers(commandBuffer, 0, 1, importedVertexBuffers, importedVertexOffsets);
                vkCmdBindIndexBuffer(commandBuffer, importedActorIndexBuffer, importedActorIndexOffset, VK_INDEX_TYPE_UINT32);
                ChunkPushConstants importedPushConstants{};
                importedPushConstants.materialParams[0] = 0.5f;
                importedPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                // Per draw, so an actor casts the silhouette its own authored
                // threshold cuts rather than the one 0.5 happens to cut. The
                // cascade index has to ride along in the same push, which is
                // why this is a local helper and not the one the static block
                // above uses.
                int lastPushedActorThreshold = -1;
                const auto pushActorAlphaThreshold = [&](std::uint8_t threshold) {
                    if (static_cast<int>(threshold) == lastPushedActorThreshold) {
                        return;
                    }
                    lastPushedActorThreshold = static_cast<int>(threshold);
                    importedPushConstants.materialParams[0] =
                        static_cast<float>(threshold) / 255.0f;
                    vkCmdPushConstants(
                        commandBuffer,
                        m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        0,
                        sizeof(ChunkPushConstants),
                        &importedPushConstants
                    );
                };
                for (const ImportedMeshDraw& importedDraw : importedActorMeshDraws) {
                    if (importedDraw.blended) {
                        continue;  // a blended surface casts no opaque shadow
                    }
                    pushActorAlphaThreshold(importedDraw.alphaThreshold);
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(
                        commandBuffer, importedDraw.indexCount, 1, importedDraw.firstIndex,
                        importedDraw.vertexOffset, 0);
                }
                vkCmdSetDepthBias(commandBuffer, -constantBias, 0.0f, -slopeBias);
            }
            // GPU-skinned actors (Dragon Age touchstone) -- same pipeline as the
            // CPU-skinned block above (skinning_resources.cc's output buffers are
            // laid out exactly like ImportedMeshVertex). Up to kMaxSkinnedInstances
            // independent instance slots means each draw may come from a
            // different vertex/index buffer, so bind is resolved per-draw.
            if (m_importedStaticShadowPipeline != VK_NULL_HANDLE &&
                !skinnedActorMeshDraws.empty()) {
                vkCmdSetDepthBias(
                    commandBuffer,
                    -(constantBias * kImportedShadowConstantBiasScale),
                    0.0f,
                    -(slopeBias * kImportedShadowSlopeBiasScale));
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_importedStaticShadowPipeline);
                bindGraphicsDescriptorBuffers(commandBuffer);
                ChunkPushConstants skinnedPushConstants{};
                skinnedPushConstants.materialParams[0] = 0.5f;
                skinnedPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                // Per draw, as in the actor block above.
                int lastPushedSkinnedThreshold = -1;
                const auto pushSkinnedAlphaThreshold = [&](std::uint8_t threshold) {
                    if (static_cast<int>(threshold) == lastPushedSkinnedThreshold) {
                        return;
                    }
                    lastPushedSkinnedThreshold = static_cast<int>(threshold);
                    skinnedPushConstants.materialParams[0] =
                        static_cast<float>(threshold) / 255.0f;
                    vkCmdPushConstants(
                        commandBuffer,
                        m_pipelineLayout,
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        0,
                        sizeof(ChunkPushConstants),
                        &skinnedPushConstants
                    );
                };
                VkBuffer boundSkinnedVertexBuffer = VK_NULL_HANDLE;
                VkBuffer boundSkinnedIndexBuffer = VK_NULL_HANDLE;
                for (const ImportedMeshDraw& skinnedDraw : skinnedActorMeshDraws) {
                    if (skinnedDraw.blended) {
                        continue;  // a blended surface casts no opaque shadow
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
                    pushSkinnedAlphaThreshold(skinnedDraw.alphaThreshold);
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(commandBuffer, skinnedDraw.indexCount, 1, skinnedDraw.firstIndex, 0, 0);
                }
                vkCmdSetDepthBias(commandBuffer, -constantBias, 0.0f, -slopeBias);
            }

            // (removed) pipe / belt / transport shadow-caster draws — legacy factory sim.

            // (removed) grass billboard shadow-caster draw. Each quad's alpha test read a
            // minified mip of a mostly-transparent sprite, passed the cutoff, and cast a
            // solid rectangle -- dark streaks over the ground with no visible occluder.
            // Grass is no longer scattered at all (see chunk_upload.cc).
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

    // Dumps the PREVIOUS frame's atlas -- this frame's is still only recorded,
    // not submitted -- which is what you want anyway: a fully streamed one.
    if (const char* dumpPath = std::getenv("ODAI_SHADOW_DUMP")) {
        if (!m_shadowAtlasDumped && m_shadowAtlasDumpCountdown-- <= 0) {
            m_shadowAtlasDumped = true;
            dumpShadowAtlas(dumpPath);
        }
    }
}


// ODAI_SHADOW_DUMP=<path> writes the shadow atlas out as a PGM, once.
//
// An EMPTY atlas and a correctly-populated-but-wrongly-sampled one look
// identical on screen -- everything lit -- because with reverse-Z the clear is
// 0.0 and "receiver depth >= 0" is always true. Reading the image back is the
// only thing that separates those two, and they need opposite fixes.
//
// Reverse-Z, so in the output BLACK IS EMPTY (far/cleared) and brighter is a
// caster nearer the light. A tile that is uniformly black was never drawn into.
void RendererBackend::dumpShadowAtlas(const char* outputPath) {
    if (m_shadowDepthImage == VK_NULL_HANDLE || m_device == VK_NULL_HANDLE) {
        VOX_LOGW("render") << "shadow atlas dump: no atlas image";
        return;
    }
    vkDeviceWaitIdle(m_device);

    const VkDeviceSize pixelCount =
        static_cast<VkDeviceSize>(kShadowAtlasSize) * static_cast<VkDeviceSize>(kShadowAtlasSize);
    const VkDeviceSize bufferSize = pixelCount * sizeof(float);

    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bufferSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &staging) != VK_SUCCESS) {
            VOX_LOGW("render") << "shadow atlas dump: staging buffer creation failed";
            return;
        }
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(m_device, staging, &requirements);
        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = requirements.size;
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);
        constexpr VkMemoryPropertyFlags kWanted =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t memoryTypeIndex = memoryProperties.memoryTypeCount;
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if (((requirements.memoryTypeBits & (1u << i)) != 0u) &&
                ((memoryProperties.memoryTypes[i].propertyFlags & kWanted) == kWanted)) {
                memoryTypeIndex = i;
                break;
            }
        }
        if (memoryTypeIndex == memoryProperties.memoryTypeCount) {
            VOX_LOGW("render") << "shadow atlas dump: no host-visible memory type";
            vkDestroyBuffer(m_device, staging, nullptr);
            return;
        }
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        if (vkAllocateMemory(m_device, &allocateInfo, nullptr, &stagingMemory) != VK_SUCCESS ||
            vkBindBufferMemory(m_device, staging, stagingMemory, 0) != VK_SUCCESS) {
            VOX_LOGW("render") << "shadow atlas dump: staging memory allocation failed";
            vkDestroyBuffer(m_device, staging, nullptr);
            return;
        }
    }

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &pool) == VK_SUCCESS) {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = pool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(m_device, &allocInfo, &cmd) == VK_SUCCESS) {
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            const auto barrier = [&](VkImageLayout oldLayout, VkImageLayout newLayout) {
                VkImageMemoryBarrier2 imageBarrier{};
                imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                imageBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
                imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                imageBarrier.dstAccessMask =
                    VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
                imageBarrier.oldLayout = oldLayout;
                imageBarrier.newLayout = newLayout;
                imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imageBarrier.image = m_shadowDepthImage;
                imageBarrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                VkDependencyInfo dependency{};
                dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dependency.imageMemoryBarrierCount = 1;
                dependency.pImageMemoryBarriers = &imageBarrier;
                vkCmdPipelineBarrier2(cmd, &dependency);
            };

            barrier(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
            region.imageExtent = {kShadowAtlasSize, kShadowAtlasSize, 1u};
            vkCmdCopyImageToBuffer(
                cmd, m_shadowDepthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &region);
            barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            vkEndCommandBuffer(cmd);
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;
            vkQueueSubmit(m_graphicsQueue, 1, &submit, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_graphicsQueue);

            void* mapped = nullptr;
            if (vkMapMemory(m_device, stagingMemory, 0, bufferSize, 0, &mapped) == VK_SUCCESS &&
                mapped != nullptr) {
                const float* depths = static_cast<const float*>(mapped);
                // 16-BIT, not 8. One 8-bit step is 1/255 of a cascade's depth
                // range, which at cascade 3 (~20000 world units deep) is ~79
                // world units -- so a 60-unit occluder separation lands BELOW
                // the quantization floor and reads as "no shadow" no matter
                // what the renderer did. That artifact cost a round of chasing
                // a cascade-3 bug that was not there. 16-bit puts the step at
                // 0.31 units, comfortably under anything worth measuring.
                std::vector<unsigned char> pixels(static_cast<std::size_t>(pixelCount) * 2u);
                double sum = 0.0;
                std::size_t nonZero = 0;
                for (std::size_t i = 0; i < static_cast<std::size_t>(pixelCount); ++i) {
                    const float d = depths[i];
                    sum += d;
                    nonZero += (d > 0.0f) ? 1u : 0u;
                    const auto q = static_cast<std::uint16_t>(
                        std::clamp(d, 0.0f, 1.0f) * 65535.0f + 0.5f);
                    // PGM is big-endian.
                    pixels[i * 2u] = static_cast<unsigned char>(q >> 8);
                    pixels[(i * 2u) + 1u] = static_cast<unsigned char>(q & 0xffu);
                }
                if (std::FILE* file = std::fopen(outputPath, "wb")) {
                    std::fprintf(file, "P5\n%u %u\n65535\n", kShadowAtlasSize, kShadowAtlasSize);
                    std::fwrite(pixels.data(), 1, pixels.size(), file);
                    std::fclose(file);
                }
                // The counts are the actual finding; the image is for looking at
                // WHERE the content is, once you know there is any.
                VOX_LOGI("render") << "shadow atlas dump -> " << outputPath << ": "
                                   << nonZero << "/" << pixels.size() << " texels written ("
                                   << (100.0 * static_cast<double>(nonZero) /
                                       static_cast<double>(pixels.size()))
                                   << "%), mean depth " << (sum / static_cast<double>(pixels.size()));
                // Per cascade tile, because a whole-atlas percentage hides
                // "cascade 0 is empty and cascade 3 is fine".
                for (uint32_t cascadeIndex = 0; cascadeIndex < kShadowCascadeCount; ++cascadeIndex) {
                    const ShadowAtlasRect rect = kShadowAtlasRects[cascadeIndex];
                    std::size_t tileNonZero = 0;
                    float tileMin = 1.0f;
                    float tileMax = 0.0f;
                    double tileSum = 0.0;
                    for (uint32_t y = 0; y < rect.size; ++y) {
                        for (uint32_t x = 0; x < rect.size; ++x) {
                            const std::size_t index =
                                (static_cast<std::size_t>(rect.y + y) * kShadowAtlasSize) +
                                (rect.x + x);
                            const float d = depths[index];
                            if (d > 0.0f) {
                                ++tileNonZero;
                                tileMin = std::min(tileMin, d);
                                tileMax = std::max(tileMax, d);
                                tileSum += d;
                            }
                        }
                    }
                    const std::size_t tileTexels =
                        static_cast<std::size_t>(rect.size) * static_cast<std::size_t>(rect.size);
                    VOX_LOGI("render") << "  cascade " << cascadeIndex << " tile "
                                       << tileNonZero << "/" << tileTexels << " written"
                                       << " depth min/mean/max "
                                       << (tileNonZero != 0u ? tileMin : 0.0f) << "/"
                                       << (tileNonZero != 0u
                                               ? tileSum / static_cast<double>(tileNonZero)
                                               : 0.0)
                                       << "/" << tileMax;
                    // The SAMPLING matrix for this cascade, so a CPU cross-check
                    // can replicate the shader's ref-vs-stored comparison against
                    // this very dump. Row-major, 16 floats.
                    const odai::math::Matrix4& lp = m_shadowRenderedMatrices[cascadeIndex];
                    std::string matrixText;
                    for (int mi = 0; mi < 16; ++mi) {
                        // %.9e, NOT to_string: these coefficients are ~1e-5
                        // against world coordinates ~1e5, so six decimals of
                        // text injects more error than the bug being chased.
                        char field[32];
                        std::snprintf(field, sizeof(field), "%.9e", lp.m[mi]);
                        matrixText += (mi != 0 ? " " : "");
                        matrixText += field;
                    }
                    VOX_LOGI("render") << "  cascade " << cascadeIndex << " sampleMatrix "
                                       << matrixText;
                }
                vkUnmapMemory(m_device, stagingMemory);
            }
        }
        vkDestroyCommandPool(m_device, pool, nullptr);
    }
    vkDestroyBuffer(m_device, staging, nullptr);
    vkFreeMemory(m_device, stagingMemory, nullptr);
}

}  // namespace odai::render
