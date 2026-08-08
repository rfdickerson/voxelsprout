#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>
#include <algorithm>
#include <string>

#include "sim/network_procedural.h"
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
    const uint32_t mvpDynamicOffset = context.mvpDynamicOffset;
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
                const std::size_t staticDrawStart = terrainDrawCount;
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
                for (std::size_t drawIndex = 0; drawIndex < cascadeImportedMeshDraws.size(); ++drawIndex) {
                    if ((drawIndex < terrainDrawCount && !m_debugShowImportedTerrain) ||
                        (drawIndex >= staticDrawStart && !m_debugShowImportedStatics)) {
                        continue;
                    }
                    const ImportedMeshDraw& importedDraw = cascadeImportedMeshDraws[drawIndex];
                    countDrawCalls(m_debugDrawCallsShadow, 1);
                    vkCmdDrawIndexed(
                        commandBuffer, importedDraw.indexCount, 1, importedDraw.firstIndex,
                        importedDraw.vertexOffset, 0);
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
                importedPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &importedPushConstants
                );
                for (const ImportedMeshDraw& importedDraw : importedActorMeshDraws) {
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
                !skinnedActorMeshDraws.empty() &&
                m_debugShowImportedStatics) {
                vkCmdSetDepthBias(
                    commandBuffer,
                    -(constantBias * kImportedShadowConstantBiasScale),
                    0.0f,
                    -(slopeBias * kImportedShadowSlopeBiasScale));
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_importedStaticShadowPipeline);
                bindGraphicsDescriptorBuffers(commandBuffer);
                ChunkPushConstants skinnedPushConstants{};
                skinnedPushConstants.cascadeData[0] = static_cast<float>(cascadeIndex);
                vkCmdPushConstants(
                    commandBuffer,
                    m_pipelineLayout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0,
                    sizeof(ChunkPushConstants),
                    &skinnedPushConstants
                );
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
}

}  // namespace odai::render
