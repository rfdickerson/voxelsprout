#include "render/Renderer.hpp"

namespace render {

namespace {
// Keep these in sync with Renderer.cpp GI constants.
constexpr uint32_t kVoxelGiGridResolution = 64u;
constexpr uint32_t kVoxelGiWorkgroupSize = 4u;
constexpr uint32_t kVoxelGiPropagationIterations = 8u;

void transitionImageLayout(
    VkCommandBuffer commandBuffer,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask,
    VkImageAspectFlags aspectMask,
    uint32_t baseArrayLayer = 0,
    uint32_t layerCount = 1,
    uint32_t baseMipLevel = 0,
    uint32_t levelCount = 1
) {
    VkImageMemoryBarrier2 imageBarrier{};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    imageBarrier.srcStageMask = srcStageMask;
    imageBarrier.srcAccessMask = srcAccessMask;
    imageBarrier.dstStageMask = dstStageMask;
    imageBarrier.dstAccessMask = dstAccessMask;
    imageBarrier.oldLayout = oldLayout;
    imageBarrier.newLayout = newLayout;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = image;
    imageBarrier.subresourceRange.aspectMask = aspectMask;
    imageBarrier.subresourceRange.baseMipLevel = baseMipLevel;
    imageBarrier.subresourceRange.levelCount = levelCount;
    imageBarrier.subresourceRange.baseArrayLayer = baseArrayLayer;
    imageBarrier.subresourceRange.layerCount = layerCount;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &imageBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}
} // namespace

void Renderer::recordVoxelGiDispatchSequence(
    VkCommandBuffer commandBuffer,
    uint32_t mvpDynamicOffset,
    VkQueryPool gpuTimestampQueryPool
) {
    const uint32_t voxelGiDispatchX = (kVoxelGiGridResolution + (kVoxelGiWorkgroupSize - 1u)) / kVoxelGiWorkgroupSize;
    const uint32_t voxelGiDispatchY = (kVoxelGiGridResolution + (kVoxelGiWorkgroupSize - 1u)) / kVoxelGiWorkgroupSize;
    const uint32_t voxelGiDispatchZ = (kVoxelGiGridResolution + (kVoxelGiWorkgroupSize - 1u)) / kVoxelGiWorkgroupSize;
    const uint32_t voxelGiSkyDispatchX = (kVoxelGiGridResolution + 7u) / 8u;
    const uint32_t voxelGiSkyDispatchY = (kVoxelGiGridResolution + 7u) / 8u;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_voxelGiSkyExposurePipeline);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        m_voxelGiPipelineLayout,
        0,
        1,
        &m_voxelGiDescriptorSets[m_currentFrame],
        1,
        &mvpDynamicOffset
    );
    vkCmdDispatch(commandBuffer, voxelGiSkyDispatchX, voxelGiSkyDispatchY, 1);
    transitionImageLayout(
        commandBuffer,
        m_voxelGiSkyExposureImage,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    m_voxelGiSkyExposureInitialized = true;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_voxelGiSurfacePipeline);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        m_voxelGiPipelineLayout,
        0,
        1,
        &m_voxelGiDescriptorSets[m_currentFrame],
        1,
        &mvpDynamicOffset
    );
    vkCmdDispatch(commandBuffer, voxelGiDispatchX, voxelGiDispatchY, voxelGiDispatchZ);
    for (std::size_t faceIndex = 0; faceIndex < m_voxelGiSurfaceFaceImages.size(); ++faceIndex) {
        transitionImageLayout(
            commandBuffer,
            m_voxelGiSurfaceFaceImages[faceIndex],
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }

    if (gpuTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            gpuTimestampQueryPool,
            kGpuTimestampQueryGiInjectStart
        );
    }
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_voxelGiInjectPipeline);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        m_voxelGiPipelineLayout,
        0,
        1,
        &m_voxelGiDescriptorSets[m_currentFrame],
        1,
        &mvpDynamicOffset
    );
    vkCmdDispatch(commandBuffer, voxelGiDispatchX, voxelGiDispatchY, voxelGiDispatchZ);
    if (gpuTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            gpuTimestampQueryPool,
            kGpuTimestampQueryGiInjectEnd
        );
    }

    transitionImageLayout(
        commandBuffer,
        m_voxelGiImages[0],
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    if (gpuTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            gpuTimestampQueryPool,
            kGpuTimestampQueryGiPropagateStart
        );
    }
    for (uint32_t propagateIteration = 0; propagateIteration < kVoxelGiPropagationIterations; ++propagateIteration) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_voxelGiPropagatePipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            m_voxelGiPipelineLayout,
            0,
            1,
            &m_voxelGiDescriptorSets[m_currentFrame],
            1,
            &mvpDynamicOffset
        );
        vkCmdDispatch(commandBuffer, voxelGiDispatchX, voxelGiDispatchY, voxelGiDispatchZ);

        if ((propagateIteration + 1u) < kVoxelGiPropagationIterations) {
            transitionImageLayout(
                commandBuffer,
                m_voxelGiImages[1],
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
            transitionImageLayout(
                commandBuffer,
                m_voxelGiImages[0],
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );

            VkImageCopy copyRegion{};
            copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.srcSubresource.mipLevel = 0;
            copyRegion.srcSubresource.baseArrayLayer = 0;
            copyRegion.srcSubresource.layerCount = 1;
            copyRegion.srcOffset = {0, 0, 0};
            copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.dstSubresource.mipLevel = 0;
            copyRegion.dstSubresource.baseArrayLayer = 0;
            copyRegion.dstSubresource.layerCount = 1;
            copyRegion.dstOffset = {0, 0, 0};
            copyRegion.extent = {
                kVoxelGiGridResolution,
                kVoxelGiGridResolution,
                kVoxelGiGridResolution
            };
            vkCmdCopyImage(
                commandBuffer,
                m_voxelGiImages[1],
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_voxelGiImages[0],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &copyRegion
            );

            transitionImageLayout(
                commandBuffer,
                m_voxelGiImages[1],
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
            transitionImageLayout(
                commandBuffer,
                m_voxelGiImages[0],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
        }
    }
    if (gpuTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp(
            commandBuffer,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            gpuTimestampQueryPool,
            kGpuTimestampQueryGiPropagateEnd
        );
    }

    transitionImageLayout(
        commandBuffer,
        m_voxelGiImages[1],
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    m_voxelGiInitialized = true;
    m_voxelGiWorldDirty = false;
}

} // namespace render
