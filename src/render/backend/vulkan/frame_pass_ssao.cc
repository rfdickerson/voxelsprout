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

    // The blur/upsample writes m_aoExtent; the estimator writes the smaller
    // m_ssaoRawExtent. These were one number when both passes ran at the same
    // resolution, and conflating them again would either march at full AO
    // resolution (no saving) or leave the top-left quarter of the blur target
    // written and the rest stale.
    const VkExtent2D rawExtent = {
        std::max(1u, m_ssaoRawExtent.width),
        std::max(1u, m_ssaoRawExtent.height)
    };
    const uint32_t rawDispatchX = (rawExtent.width + (kSsaoComputeWorkgroupSize - 1u)) / kSsaoComputeWorkgroupSize;
    const uint32_t rawDispatchY = (rawExtent.height + (kSsaoComputeWorkgroupSize - 1u)) / kSsaoComputeWorkgroupSize;
    const uint32_t dispatchX = (aoExtent.width + (kSsaoComputeWorkgroupSize - 1u)) / kSsaoComputeWorkgroupSize;
    const uint32_t dispatchY = (aoExtent.height + (kSsaoComputeWorkgroupSize - 1u)) / kSsaoComputeWorkgroupSize;

    // Which estimator runs this frame. Off dispatches neither the AO nor the blur
    // pass; the world shaders read camera.shadowVoxelGridSize.w (set from
    // m_debugEnableSsao in frame_run.cc) and fall back to an ambient factor of 1,
    // so nothing samples the stale texture left behind.
    // XeGTAO runs its own two-dispatch sequence below rather than the shared
    // single-dispatch path, so it deliberately reports no aoPipeline here.
    const bool useXeGtao = m_shadowDebugSettings.aoMode == AoMode::Xegtao &&
        m_xegtaoPrefilterPipeline != VK_NULL_HANDLE &&
        m_xegtaoMainPipeline != VK_NULL_HANDLE &&
        m_xegtaoDenoisePipeline != VK_NULL_HANDLE &&
        m_xegtaoPrefilterBufferSet.valid() && m_xegtaoMainBufferSet.valid() &&
        m_xegtaoDenoiseBufferSet.valid() &&
        aoFrameIndex < m_xegtaoBentNormalImageViews.size() &&
        aoFrameIndex < m_xegtaoAoTermImageViews.size();
    const VkPipeline aoPipeline = [&]() -> VkPipeline {
        switch (m_shadowDebugSettings.aoMode) {
            case AoMode::Ssao: return m_ssaoPipeline;
            case AoMode::Hbao: return m_ssaoHbaoPipeline;
            case AoMode::Gtao: return m_ssaoGtaoPipeline;
            // Falls back to the plain GTAO pipeline when XeGTAO could not be
            // built, so selecting it on a build without its shaders still
            // renders AO rather than nothing.
            case AoMode::Xegtao: return useXeGtao ? VK_NULL_HANDLE : m_ssaoGtaoPipeline;
            case AoMode::Off:  break;
        }
        return VK_NULL_HANDLE;
    }();

    writeGpuTimestampTop(kGpuTimestampQuerySsaoStart);
    beginDebugLabel(commandBuffer, "Pass: SSAO", 0.20f, 0.36f, 0.26f, 1.0f);
    if (useXeGtao) {
        // Shared by both dispatches: the pyramid's MIP filter has to fall off on
        // exactly the curve the march will use, or the depths it feeds are
        // biased relative to the horizons computed from them.
        const float effectRadius = std::clamp(m_shadowDebugSettings.ssaoRadius, 0.25f, 512.0f);
        constexpr float kFalloffRange = 0.615f;

        transitionImageLayout(
            commandBuffer, m_normalDepthImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);

        const bool pyramidInitialized = m_xegtaoDepthInitialized[aoFrameIndex];
        for (uint32_t level = 0; level < kXeGtaoDepthMipCount; ++level) {
            transitionImageLayout(
                commandBuffer, m_xegtaoDepthImages[level][aoFrameIndex],
                pyramidInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                   : VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                pyramidInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                   : VK_PIPELINE_STAGE_2_NONE,
                pyramidInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT);
        }

        XeGtaoPrefilterPushConstants prefilterPush{};
        prefilterPush.width = rawExtent.width;
        prefilterPush.height = rawExtent.height;
        prefilterPush.effectRadius = effectRadius;
        prefilterPush.falloffRange = kFalloffRange;

        vkCmdBindPipeline(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_xegtaoPrefilterPipeline);
        bindDescriptorBuffer(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_xegtaoPrefilterPipelineLayout,
            0, m_xegtaoPrefilterBufferSet, m_currentFrame);
        vkCmdPushConstants(
            commandBuffer, m_xegtaoPrefilterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
            sizeof(prefilterPush), &prefilterPush);
        // One group per 16x16 tile of MIP0 -- the group covers 16x16 even though
        // it is 8x8 threads, because each thread owns a 2x2 quad.
        vkCmdDispatch(
            commandBuffer, (rawExtent.width + 15u) / 16u, (rawExtent.height + 15u) / 16u, 1u);

        for (uint32_t level = 0; level < kXeGtaoDepthMipCount; ++level) {
            transitionImageLayout(
                commandBuffer, m_xegtaoDepthImages[level][aoFrameIndex],
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT);
        }
        m_xegtaoDepthInitialized[aoFrameIndex] = true;

        const bool aoTermInitialized = m_xegtaoAoTermInitialized[aoFrameIndex];
        transitionImageLayout(
            commandBuffer, m_xegtaoAoTermImages[aoFrameIndex],
            aoTermInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                              : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            aoTermInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                              : VK_PIPELINE_STAGE_2_NONE,
            aoTermInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const bool bentInitialized = m_xegtaoBentNormalInitialized[aoFrameIndex];
        transitionImageLayout(
            commandBuffer, m_xegtaoBentNormalImages[aoFrameIndex],
            bentInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            bentInitialized ? VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
            bentInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);

        XeGtaoMainPushConstants mainPush{};
        mainPush.width = rawExtent.width;
        mainPush.height = rawExtent.height;
        mainPush.effectRadius = effectRadius;
        mainPush.falloffRange = kFalloffRange;
        mainPush.sampleDistributionPower = 2.0f;
        mainPush.thinOccluderCompensation = 0.0f;
        mainPush.finalValuePower = 2.2f;
        // Advances every frame so the blue noise is decorrelated in time and TAA
        // integrates the sampling error away instead of locking in a dither.
        mainPush.temporalIndex = m_xegtaoTemporalIndex++;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_xegtaoMainPipeline);
        bindDescriptorBuffer(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_xegtaoMainPipelineLayout,
            0, m_xegtaoMainBufferSet, m_currentFrame);
        vkCmdPushConstants(
            commandBuffer, m_xegtaoMainPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
            sizeof(mainPush), &mainPush);
        vkCmdDispatch(commandBuffer, rawDispatchX, rawDispatchY, 1u);

        transitionImageLayout(
            commandBuffer, m_xegtaoAoTermImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        // The bent normals go two places: the denoiser reads their alpha for the
        // packed edges, and the main lighting pass may sample the direction. Both
        // stages have to be in the destination mask or the denoise read races the
        // write.
        transitionImageLayout(
            commandBuffer, m_xegtaoBentNormalImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        m_xegtaoAoTermInitialized[aoFrameIndex] = true;
        m_xegtaoBentNormalInitialized[aoFrameIndex] = true;

        // DENOISE. Edge-aware blur of the AO term into m_ssaoRawImages, which is
        // where every existing consumer already looks -- so the joint bilateral
        // upsample below and the main pass need no changes.
        const bool rawInitialized = m_ssaoRawImageInitialized[aoFrameIndex];
        transitionImageLayout(
            commandBuffer, m_ssaoRawImages[aoFrameIndex],
            rawInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL,
            rawInitialized ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_2_NONE,
            rawInitialized ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);

        // ODAI_XEGTAO_BLUR tunes how far the denoise spreads into edge-connected
        // neighbours. Exposed because the right value depends on how noisy the
        // raw term is, which depends on the adaptive sample counts and on
        // ODAI_AO_DOWNSCALE -- there is no single correct constant across them.
        static const float s_denoiseBlur = []() {
            const char* env = std::getenv("ODAI_XEGTAO_BLUR");
            const float value = (env != nullptr) ? static_cast<float>(std::atof(env)) : -1.0f;
            // 4.0 measured: flicker (frame N vs N+1, static camera) 0.335% at
            // 1.0, 0.252% at 2.0, 0.176% at 4.0, while AO strength against
            // AO-off held at 40.6% throughout. Stronger blur does not wash the
            // effect out because the weights are edge-gated -- it only spreads
            // within a connected surface, which is exactly where AO is
            // low-frequency and the noise is not.
            return (value >= 0.0f) ? value : 4.0f;
        }();
        XeGtaoDenoisePushConstants denoisePush{};
        denoisePush.width = rawExtent.width;
        denoisePush.height = rawExtent.height;
        denoisePush.blurAmount = s_denoiseBlur;

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_xegtaoDenoisePipeline);
        bindDescriptorBuffer(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_xegtaoDenoisePipelineLayout,
            0, m_xegtaoDenoiseBufferSet, m_currentFrame);
        vkCmdPushConstants(
            commandBuffer, m_xegtaoDenoisePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
            sizeof(denoisePush), &denoisePush);
        vkCmdDispatch(commandBuffer, rawDispatchX, rawDispatchY, 1u);

        transitionImageLayout(
            commandBuffer, m_ssaoRawImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        m_ssaoRawImageInitialized[aoFrameIndex] = true;
    }
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
        ssaoPushConstants.width = rawExtent.width;
        ssaoPushConstants.height = rawExtent.height;
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
        vkCmdDispatch(commandBuffer, rawDispatchX, rawDispatchY, 1u);

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
    if ((aoPipeline != VK_NULL_HANDLE || useXeGtao) && m_ssaoBlurPipeline != VK_NULL_HANDLE &&
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
