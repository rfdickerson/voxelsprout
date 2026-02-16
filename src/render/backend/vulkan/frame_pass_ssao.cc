#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include "sim/network_procedural.h"

namespace voxelsprout::render {

#include "render/renderer_shared.h"

void RendererBackend::recordSsaoPasses(const FrameExecutionContext& context) {
    VkCommandBuffer commandBuffer = context.commandBuffer;
    VkQueryPool gpuTimestampQueryPool = context.gpuTimestampQueryPool;
    const uint32_t aoFrameIndex = context.aoFrameIndex;
    const VkExtent2D aoExtent = context.aoExtent;
    const VkViewport& aoViewport = context.aoViewport;
    const VkRect2D& aoScissor = context.aoScissor;
    const BoundDescriptorSets& boundDescriptorSets = *context.boundDescriptorSets;
    const uint32_t mvpDynamicOffset = context.mvpDynamicOffset;

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

    VkClearValue ssaoClearValue{};
    ssaoClearValue.color.float32[0] = 1.0f;
    ssaoClearValue.color.float32[1] = 1.0f;
    ssaoClearValue.color.float32[2] = 1.0f;
    ssaoClearValue.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo ssaoRawAttachment{};
    ssaoRawAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    ssaoRawAttachment.imageView = m_ssaoRawImageViews[aoFrameIndex];
    ssaoRawAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ssaoRawAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ssaoRawAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ssaoRawAttachment.clearValue = ssaoClearValue;

    VkRenderingInfo ssaoRenderingInfo{};
    ssaoRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ssaoRenderingInfo.renderArea.offset = {0, 0};
    ssaoRenderingInfo.renderArea.extent = aoExtent;
    ssaoRenderingInfo.layerCount = 1;
    ssaoRenderingInfo.colorAttachmentCount = 1;
    ssaoRenderingInfo.pColorAttachments = &ssaoRawAttachment;

    writeGpuTimestampTop(kGpuTimestampQuerySsaoStart);
    beginDebugLabel(commandBuffer, "Pass: SSAO", 0.20f, 0.36f, 0.26f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &ssaoRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &aoViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &aoScissor);
    if (m_ssaoPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ssaoPipeline);
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
        countDrawCalls(m_debugDrawCallsPrepass, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQuerySsaoEnd);

    transitionImageLayout(
        commandBuffer,
        m_ssaoRawImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );

    VkRenderingAttachmentInfo ssaoBlurAttachment{};
    ssaoBlurAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    ssaoBlurAttachment.imageView = m_ssaoBlurImageViews[aoFrameIndex];
    ssaoBlurAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ssaoBlurAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ssaoBlurAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    ssaoBlurAttachment.clearValue = ssaoClearValue;

    VkRenderingInfo ssaoBlurRenderingInfo{};
    ssaoBlurRenderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ssaoBlurRenderingInfo.renderArea.offset = {0, 0};
    ssaoBlurRenderingInfo.renderArea.extent = aoExtent;
    ssaoBlurRenderingInfo.layerCount = 1;
    ssaoBlurRenderingInfo.colorAttachmentCount = 1;
    ssaoBlurRenderingInfo.pColorAttachments = &ssaoBlurAttachment;

    writeGpuTimestampTop(kGpuTimestampQuerySsaoBlurStart);
    beginDebugLabel(commandBuffer, "Pass: SSAO Blur", 0.22f, 0.40f, 0.30f, 1.0f);
    vkCmdBeginRendering(commandBuffer, &ssaoBlurRenderingInfo);
    vkCmdSetViewport(commandBuffer, 0, 1, &aoViewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &aoScissor);
    if (m_ssaoBlurPipeline != VK_NULL_HANDLE) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ssaoBlurPipeline);
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
        countDrawCalls(m_debugDrawCallsPrepass, 1);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    }
    vkCmdEndRendering(commandBuffer);
    endDebugLabel(commandBuffer);
    writeGpuTimestampBottom(kGpuTimestampQuerySsaoBlurEnd);

    transitionImageLayout(
        commandBuffer,
        m_ssaoBlurImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
}

}  // namespace voxelsprout::render
