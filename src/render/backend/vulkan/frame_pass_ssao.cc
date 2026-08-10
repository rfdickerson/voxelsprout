#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>

#include "sim/network_procedural.h"

namespace odai::render {

#include "render/renderer_shared.h"

void RendererBackend::recordSsaoPasses(const FrameExecutionContext& context) {
    VkCommandBuffer commandBuffer = context.commandBuffer;
    VkQueryPool gpuTimestampQueryPool = context.gpuTimestampQueryPool;
    const uint32_t aoFrameIndex = context.aoFrameIndex;
    const VkExtent2D aoExtent = context.aoExtent;
    const uint32_t mvpDynamicOffset = context.mvpDynamicOffset;

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

    const uint32_t dispatchX = (aoExtent.width + (kSsaoComputeWorkgroupSize - 1u)) / kSsaoComputeWorkgroupSize;
    const uint32_t dispatchY = (aoExtent.height + (kSsaoComputeWorkgroupSize - 1u)) / kSsaoComputeWorkgroupSize;

    // Which estimator runs this frame. Off dispatches neither the AO nor the blur
    // pass; the world shaders read camera.shadowVoxelGridSize.w (set from
    // m_debugEnableSsao in frame_run.cc) and fall back to an ambient factor of 1,
    // so nothing samples the stale texture left behind.
    const VkPipeline aoPipeline = [&]() -> VkPipeline {
        switch (m_shadowDebugSettings.aoMode) {
            case AoMode::Ssao: return m_ssaoPipeline;
            case AoMode::Hbao: return m_ssaoHbaoPipeline;
            case AoMode::Gtao: return m_ssaoGtaoPipeline;
            case AoMode::Off:  break;
        }
        return VK_NULL_HANDLE;
    }();

    writeGpuTimestampTop(kGpuTimestampQuerySsaoStart);
    beginDebugLabel(commandBuffer, "Pass: SSAO", 0.20f, 0.36f, 0.26f, 1.0f);
    if (aoPipeline != VK_NULL_HANDLE && m_ssaoBufferSet.valid()) {
        // Self-transition: the normal-depth prepass leaves this sampled for fragment
        // shaders (main lighting); sync it for this compute read too.
        transitionImageLayout(
            commandBuffer,
            m_normalDepthImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );

        const bool ssaoRawInitialized = m_ssaoRawImageInitialized[aoFrameIndex];
        transitionImageLayout(
            commandBuffer,
            m_ssaoRawImages[aoFrameIndex],
            ssaoRawInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            ssaoRawInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
            ssaoRawInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );

        SsaoComputePushConstants ssaoPushConstants{};
        ssaoPushConstants.width = std::max(1u, aoExtent.width);
        ssaoPushConstants.height = std::max(1u, aoExtent.height);
        ssaoPushConstants.fineRadiusScale = m_shadowDebugSettings.ssaoFineRadiusScale;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, aoPipeline);
        bindDescriptorBuffer(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoPipelineLayout,
            0, m_ssaoBufferSet, m_currentFrame);
        vkCmdPushConstants(
            commandBuffer,
            m_ssaoPipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(SsaoComputePushConstants),
            &ssaoPushConstants
        );
        vkCmdDispatch(commandBuffer, dispatchX, dispatchY, 1u);

        transitionImageLayout(
            commandBuffer,
            m_ssaoRawImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        m_ssaoRawImageInitialized[aoFrameIndex] = true;
    }
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQuerySsaoEnd);

    writeGpuTimestampTop(kGpuTimestampQuerySsaoBlurStart);
    beginDebugLabel(commandBuffer, "Pass: SSAO Blur", 0.22f, 0.40f, 0.30f, 1.0f);
    if (aoPipeline != VK_NULL_HANDLE && m_ssaoBlurPipeline != VK_NULL_HANDLE &&
        m_ssaoBlurBufferSet.valid()) {
        const bool ssaoBlurInitialized = m_ssaoBlurImageInitialized[aoFrameIndex];
        transitionImageLayout(
            commandBuffer,
            m_ssaoBlurImages[aoFrameIndex],
            ssaoBlurInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            ssaoBlurInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
            ssaoBlurInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );

        struct SsaoComputePushConstants {
            uint32_t width;
            uint32_t height;
        };
        SsaoComputePushConstants ssaoBlurPushConstants{};
        ssaoBlurPushConstants.width = std::max(1u, aoExtent.width);
        ssaoBlurPushConstants.height = std::max(1u, aoExtent.height);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoBlurPipeline);
        bindDescriptorBuffer(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoBlurPipelineLayout,
            0, m_ssaoBlurBufferSet, m_currentFrame);
        vkCmdPushConstants(
            commandBuffer,
            m_ssaoBlurPipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(SsaoComputePushConstants),
            &ssaoBlurPushConstants
        );
        vkCmdDispatch(commandBuffer, dispatchX, dispatchY, 1u);

        // Final consumers of the blurred AO texture are the main-pass ambient terms in
        // imported_static.frag.slang and voxel_packed.frag.slang (both binding 7) plus
        // the tonemap debug-visualize modes -- not another compute pass.
        transitionImageLayout(
            commandBuffer,
            m_ssaoBlurImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        m_ssaoBlurImageInitialized[aoFrameIndex] = true;
    }
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQuerySsaoBlurEnd);
}

}  // namespace odai::render
