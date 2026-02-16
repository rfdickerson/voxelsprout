#include "render/backend/vulkan/renderer_backend.h"

namespace voxelsprout::render {

namespace {
// Keep these in sync with renderer.cc GI constants.
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
struct VoxelGiDispatchDims {
    uint32_t volumeX = 1u;
    uint32_t volumeY = 1u;
    uint32_t volumeZ = 1u;
    uint32_t skyX = 1u;
    uint32_t skyY = 1u;
};

struct VoxelGiPassContext {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    uint32_t mvpDynamicOffset = 0u;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkPipeline skyExposurePipeline = VK_NULL_HANDLE;
    VkPipeline surfacePipeline = VK_NULL_HANDLE;
    VkPipeline injectPipeline = VK_NULL_HANDLE;
    VkPipeline propagatePipeline = VK_NULL_HANDLE;
    std::array<VkImage, 6> surfaceFaceImages{};
    std::array<VkImage, 2> voxelGiImages{};
    VkImage skyExposureImage = VK_NULL_HANDLE;
    VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
};

struct VoxelGiPassState {
    bool* skyExposureInitialized = nullptr;
    bool* giInitialized = nullptr;
    bool* giWorldDirty = nullptr;
};

struct VoxelGiTimestampQueryIndices {
    uint32_t injectStart = 0u;
    uint32_t injectEnd = 0u;
    uint32_t propagateStart = 0u;
    uint32_t propagateEnd = 0u;
};

VoxelGiDispatchDims computeVoxelGiDispatchDims() {
    VoxelGiDispatchDims dims{};
    dims.volumeX = (kVoxelGiGridResolution + (kVoxelGiWorkgroupSize - 1u)) / kVoxelGiWorkgroupSize;
    dims.volumeY = (kVoxelGiGridResolution + (kVoxelGiWorkgroupSize - 1u)) / kVoxelGiWorkgroupSize;
    dims.volumeZ = (kVoxelGiGridResolution + (kVoxelGiWorkgroupSize - 1u)) / kVoxelGiWorkgroupSize;
    dims.skyX = (kVoxelGiGridResolution + 7u) / 8u;
    dims.skyY = (kVoxelGiGridResolution + 7u) / 8u;
    return dims;
}

void bindVoxelGiComputePass(const VoxelGiPassContext& context, VkPipeline pipeline) {
    const uint32_t dynamicOffset = context.mvpDynamicOffset;
    vkCmdBindPipeline(context.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(
        context.commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        context.pipelineLayout,
        0,
        1,
        &context.descriptorSet,
        1,
        &dynamicOffset
    );
}

void writeTimestampIfEnabled(
    const VoxelGiPassContext& context,
    VkPipelineStageFlagBits stage,
    uint32_t queryIndex
) {
    if (context.timestampQueryPool == VK_NULL_HANDLE) {
        return;
    }
    vkCmdWriteTimestamp(context.commandBuffer, stage, context.timestampQueryPool, queryIndex);
}

void recordVoxelGiSkyExposurePass(
    const VoxelGiPassContext& context,
    const VoxelGiDispatchDims& dims,
    VoxelGiPassState& state
) {
    bindVoxelGiComputePass(context, context.skyExposurePipeline);
    vkCmdDispatch(context.commandBuffer, dims.skyX, dims.skyY, 1u);
    transitionImageLayout(
        context.commandBuffer,
        context.skyExposureImage,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    *state.skyExposureInitialized = true;
}

void recordVoxelGiSurfacePass(
    const VoxelGiPassContext& context,
    const VoxelGiDispatchDims& dims
) {
    bindVoxelGiComputePass(context, context.surfacePipeline);
    vkCmdDispatch(context.commandBuffer, dims.volumeX, dims.volumeY, dims.volumeZ);
    for (const VkImage surfaceFaceImage : context.surfaceFaceImages) {
        transitionImageLayout(
            context.commandBuffer,
            surfaceFaceImage,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }
}

void recordVoxelGiInjectPass(
    const VoxelGiPassContext& context,
    const VoxelGiDispatchDims& dims,
    const VoxelGiTimestampQueryIndices& timestampQueries
) {
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueries.injectStart);

    bindVoxelGiComputePass(context, context.injectPipeline);
    vkCmdDispatch(context.commandBuffer, dims.volumeX, dims.volumeY, dims.volumeZ);

    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueries.injectEnd);
}

void recordVoxelGiPropagationPass(
    const VoxelGiPassContext& context,
    const VoxelGiDispatchDims& dims,
    const VoxelGiTimestampQueryIndices& timestampQueries
) {
    transitionImageLayout(
        context.commandBuffer,
        context.voxelGiImages[0],
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueries.propagateStart);

    for (uint32_t propagateIteration = 0; propagateIteration < kVoxelGiPropagationIterations; ++propagateIteration) {
        bindVoxelGiComputePass(context, context.propagatePipeline);
        vkCmdDispatch(context.commandBuffer, dims.volumeX, dims.volumeY, dims.volumeZ);

        if ((propagateIteration + 1u) >= kVoxelGiPropagationIterations) {
            continue;
        }

        transitionImageLayout(
            context.commandBuffer,
            context.voxelGiImages[1],
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        transitionImageLayout(
            context.commandBuffer,
            context.voxelGiImages[0],
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
            context.commandBuffer,
            context.voxelGiImages[1],
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            context.voxelGiImages[0],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &copyRegion
        );

        transitionImageLayout(
            context.commandBuffer,
            context.voxelGiImages[1],
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
        transitionImageLayout(
            context.commandBuffer,
            context.voxelGiImages[0],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }

    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueries.propagateEnd);
}

void finalizeVoxelGiPass(
    const VoxelGiPassContext& context,
    VoxelGiPassState& state
) {
    transitionImageLayout(
        context.commandBuffer,
        context.voxelGiImages[1],
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
    *state.giInitialized = true;
    *state.giWorldDirty = false;
}
} // namespace

void RendererBackend::recordVoxelGiDispatchSequence(
    VkCommandBuffer commandBuffer,
    uint32_t mvpDynamicOffset,
    VkQueryPool gpuTimestampQueryPool
) {
    const VoxelGiDispatchDims dispatchDims = computeVoxelGiDispatchDims();
    const VoxelGiPassContext passContext{
        commandBuffer,
        mvpDynamicOffset,
        m_voxelGiPipelineLayout,
        m_voxelGiDescriptorSets[m_currentFrame],
        m_voxelGiSkyExposurePipeline,
        m_voxelGiSurfacePipeline,
        m_voxelGiInjectPipeline,
        m_voxelGiPropagatePipeline,
        m_voxelGiSurfaceFaceImages,
        m_voxelGiImages,
        m_voxelGiSkyExposureImage,
        gpuTimestampQueryPool
    };
    VoxelGiPassState passState{
        &m_voxelGiSkyExposureInitialized,
        &m_voxelGiInitialized,
        &m_voxelGiWorldDirty
    };
    const VoxelGiTimestampQueryIndices timestampQueries{
        kGpuTimestampQueryGiInjectStart,
        kGpuTimestampQueryGiInjectEnd,
        kGpuTimestampQueryGiPropagateStart,
        kGpuTimestampQueryGiPropagateEnd
    };

    recordVoxelGiSkyExposurePass(passContext, dispatchDims, passState);
    recordVoxelGiSurfacePass(passContext, dispatchDims);
    recordVoxelGiInjectPass(passContext, dispatchDims, timestampQueries);
    recordVoxelGiPropagationPass(passContext, dispatchDims, timestampQueries);
    finalizeVoxelGiPass(passContext, passState);
}

} // namespace voxelsprout::render
