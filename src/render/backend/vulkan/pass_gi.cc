#include "render/backend/vulkan/renderer_backend.h"

namespace voxelsprout::render {

namespace {
// Keep these in sync with renderer.cc GI constants.
constexpr uint32_t kVoxelGiGridResolution = 64u;
constexpr uint32_t kVoxelGiWorkgroupSize = 4u;
constexpr uint32_t kVoxelGiPropagationIterations = 8u;
constexpr uint32_t kVoxelGiChunkResolution = 32u;
constexpr uint32_t kVoxelGiOccupancyWorkgroupXY = 8u;

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

void transitionComputeStorageAccess(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier2 memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    memoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memoryBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    memoryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memoryBarrier.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.memoryBarrierCount = 1;
    dependencyInfo.pMemoryBarriers = &memoryBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}
struct VoxelGiDispatchDims {
    uint32_t volumeX = 1u;
    uint32_t volumeY = 1u;
    uint32_t volumeZ = 1u;
    uint32_t occupancyX = 1u;
    uint32_t occupancyY = 1u;
    uint32_t skyX = 1u;
    uint32_t skyY = 1u;
};

struct VoxelGiPassContext {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    uint32_t mvpDynamicOffset = 0u;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkPipeline occupancyPipeline = VK_NULL_HANDLE;
    VkPipeline skyExposurePipeline = VK_NULL_HANDLE;
    VkPipeline surfacePipeline = VK_NULL_HANDLE;
    VkPipeline restirCandidatePipeline = VK_NULL_HANDLE;
    VkPipeline restirTemporalPipeline = VK_NULL_HANDLE;
    VkPipeline restirSpatialPipeline = VK_NULL_HANDLE;
    VkPipeline restirResolvePipeline = VK_NULL_HANDLE;
    VkPipeline injectPipeline = VK_NULL_HANDLE;
    VkPipeline propagatePipeline = VK_NULL_HANDLE;
    VkImage occupancyImage = VK_NULL_HANDLE;
    std::array<VkImage, 6> surfaceFaceImages{};
    std::array<VkImage, 2> voxelGiImages{};
    VkImage skyExposureImage = VK_NULL_HANDLE;
    VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
    bool useRtSurfaceTracing = false;
    bool useRestirSurface = false;
    bool restirHistoryValid = false;
};

struct VoxelGiPassState {
    bool* skyExposureInitialized = nullptr;
    bool* giInitialized = nullptr;
    bool* giWorldDirty = nullptr;
};

struct VoxelGiTimestampQueryIndices {
    uint32_t occupancyStart = 0u;
    uint32_t occupancyEnd = 0u;
    uint32_t surfaceStart = 0u;
    uint32_t surfaceEnd = 0u;
    uint32_t surfaceCandidateStart = 0u;
    uint32_t surfaceCandidateEnd = 0u;
    uint32_t surfaceTemporalStart = 0u;
    uint32_t surfaceTemporalEnd = 0u;
    uint32_t surfaceSpatialStart = 0u;
    uint32_t surfaceSpatialEnd = 0u;
    uint32_t surfaceResolveStart = 0u;
    uint32_t surfaceResolveEnd = 0u;
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
    dims.occupancyX = (kVoxelGiChunkResolution + (kVoxelGiOccupancyWorkgroupXY - 1u)) / kVoxelGiOccupancyWorkgroupXY;
    dims.occupancyY = (kVoxelGiChunkResolution + (kVoxelGiOccupancyWorkgroupXY - 1u)) / kVoxelGiOccupancyWorkgroupXY;
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

void recordVoxelGiOccupancyPass(
    const VoxelGiPassContext& context,
    const VoxelGiDispatchDims& dims,
    const VoxelGiTimestampQueryIndices& timestampQueries,
    uint32_t occupancyDispatchZ
) {
    if (occupancyDispatchZ == 0u) {
        writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueries.occupancyStart);
        writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueries.occupancyEnd);
        return;
    }
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueries.occupancyStart);
    bindVoxelGiComputePass(context, context.occupancyPipeline);
    vkCmdDispatch(context.commandBuffer, dims.occupancyX, dims.occupancyY, occupancyDispatchZ);
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueries.occupancyEnd);
    transitionImageLayout(
        context.commandBuffer,
        context.occupancyImage,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT
    );
}

void recordVoxelGiSurfacePass(
    const VoxelGiPassContext& context,
    const VoxelGiDispatchDims& dims,
    const VoxelGiTimestampQueryIndices& timestampQueries
) {
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueries.surfaceStart);
    if (context.useRtSurfaceTracing) {
        VkMemoryBarrier2 rayTracingReadBarrier{};
        rayTracingReadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        rayTracingReadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        rayTracingReadBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        rayTracingReadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rayTracingReadBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.memoryBarrierCount = 1;
        dependencyInfo.pMemoryBarriers = &rayTracingReadBarrier;
        vkCmdPipelineBarrier2(context.commandBuffer, &dependencyInfo);
    }
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
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueries.surfaceEnd);
}

void writeZeroDurationTimestampPair(
    const VoxelGiPassContext& context,
    uint32_t startQuery,
    uint32_t endQuery
) {
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, startQuery);
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, endQuery);
}

void recordVoxelGiRestirSurfacePass(
    const VoxelGiPassContext& context,
    const VoxelGiDispatchDims& dims,
    const VoxelGiTimestampQueryIndices& timestampQueries
) {
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueries.surfaceStart);
    if (context.useRtSurfaceTracing) {
        VkMemoryBarrier2 rayTracingReadBarrier{};
        rayTracingReadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        rayTracingReadBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        rayTracingReadBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        rayTracingReadBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rayTracingReadBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.memoryBarrierCount = 1;
        dependencyInfo.pMemoryBarriers = &rayTracingReadBarrier;
        vkCmdPipelineBarrier2(context.commandBuffer, &dependencyInfo);
    }

    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueries.surfaceCandidateStart);
    bindVoxelGiComputePass(context, context.restirCandidatePipeline);
    vkCmdDispatch(context.commandBuffer, dims.volumeX, dims.volumeY, dims.volumeZ);
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueries.surfaceCandidateEnd);
    transitionComputeStorageAccess(context.commandBuffer);

    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueries.surfaceTemporalStart);
    bindVoxelGiComputePass(context, context.restirTemporalPipeline);
    vkCmdDispatch(context.commandBuffer, dims.volumeX, dims.volumeY, dims.volumeZ);
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueries.surfaceTemporalEnd);
    transitionComputeStorageAccess(context.commandBuffer);

    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueries.surfaceSpatialStart);
    bindVoxelGiComputePass(context, context.restirSpatialPipeline);
    vkCmdDispatch(context.commandBuffer, dims.volumeX, dims.volumeY, dims.volumeZ);
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueries.surfaceSpatialEnd);
    transitionComputeStorageAccess(context.commandBuffer);

    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampQueries.surfaceResolveStart);
    bindVoxelGiComputePass(context, context.restirResolvePipeline);
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
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueries.surfaceResolveEnd);
    writeTimestampIfEnabled(context, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampQueries.surfaceEnd);
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
    VkQueryPool gpuTimestampQueryPool,
    uint32_t occupancyDispatchZ
) {
    const bool useRtSurfaceTracing =
        m_voxelGiDebugSettings.surfaceMode != VoxelGiSurfaceMode::Legacy &&
        m_rayTracingRuntimeEnabled &&
        m_voxelGiSurfacePipelineRt != VK_NULL_HANDLE &&
        m_rtTlas.handle != VK_NULL_HANDLE;
    const bool useRestirSurface =
        m_voxelGiDebugSettings.surfaceMode == VoxelGiSurfaceMode::RestirSurface &&
        useRtSurfaceTracing &&
        m_voxelGiRestirReady &&
        m_voxelGiRestirCandidatePipeline != VK_NULL_HANDLE &&
        m_voxelGiRestirTemporalPipeline != VK_NULL_HANDLE &&
        m_voxelGiRestirSpatialPipeline != VK_NULL_HANDLE &&
        m_voxelGiRestirResolvePipeline != VK_NULL_HANDLE &&
        m_voxelGiRestirReservoirCurrentBufferHandle != kInvalidBufferHandle &&
        m_voxelGiRestirReservoirPreviousBufferHandle != kInvalidBufferHandle &&
        m_voxelGiRestirReservoirScratchBufferHandle != kInvalidBufferHandle;
    m_voxelGiRtSurfaceActiveThisFrame = useRtSurfaceTracing;
    m_voxelGiRestirActiveThisFrame = useRestirSurface;
    const VoxelGiDispatchDims dispatchDims = computeVoxelGiDispatchDims();
    const VoxelGiPassContext passContext{
        commandBuffer,
        mvpDynamicOffset,
        m_voxelGiPipelineLayout,
        m_voxelGiDescriptorSets[m_currentFrame],
        m_voxelGiOccupancyPipeline,
        m_voxelGiSkyExposurePipeline,
        useRtSurfaceTracing ? m_voxelGiSurfacePipelineRt : m_voxelGiSurfacePipeline,
        m_voxelGiRestirCandidatePipeline,
        m_voxelGiRestirTemporalPipeline,
        m_voxelGiRestirSpatialPipeline,
        m_voxelGiRestirResolvePipeline,
        m_voxelGiInjectPipeline,
        m_voxelGiPropagatePipeline,
        m_voxelGiOccupancyImage,
        m_voxelGiSurfaceFaceImages,
        m_voxelGiImages,
        m_voxelGiSkyExposureImage,
        gpuTimestampQueryPool,
        useRtSurfaceTracing,
        useRestirSurface,
        m_voxelGiRestirHistoryValid
    };
    VoxelGiPassState passState{
        &m_voxelGiSkyExposureInitialized,
        &m_voxelGiInitialized,
        &m_voxelGiWorldDirty
    };
    const VoxelGiTimestampQueryIndices timestampQueries{
        kGpuTimestampQueryGiOccupancyStart,
        kGpuTimestampQueryGiOccupancyEnd,
        kGpuTimestampQueryGiSurfaceStart,
        kGpuTimestampQueryGiSurfaceEnd,
        kGpuTimestampQueryGiSurfaceCandidateStart,
        kGpuTimestampQueryGiSurfaceCandidateEnd,
        kGpuTimestampQueryGiSurfaceTemporalStart,
        kGpuTimestampQueryGiSurfaceTemporalEnd,
        kGpuTimestampQueryGiSurfaceSpatialStart,
        kGpuTimestampQueryGiSurfaceSpatialEnd,
        kGpuTimestampQueryGiSurfaceResolveStart,
        kGpuTimestampQueryGiSurfaceResolveEnd,
        kGpuTimestampQueryGiInjectStart,
        kGpuTimestampQueryGiInjectEnd,
        kGpuTimestampQueryGiPropagateStart,
        kGpuTimestampQueryGiPropagateEnd
    };

    recordVoxelGiOccupancyPass(passContext, dispatchDims, timestampQueries, occupancyDispatchZ);
    recordVoxelGiSkyExposurePass(passContext, dispatchDims, passState);
    if (useRestirSurface) {
        recordVoxelGiRestirSurfacePass(passContext, dispatchDims, timestampQueries);
    } else {
        recordVoxelGiSurfacePass(passContext, dispatchDims, timestampQueries);
        writeZeroDurationTimestampPair(passContext, timestampQueries.surfaceCandidateStart, timestampQueries.surfaceCandidateEnd);
        writeZeroDurationTimestampPair(passContext, timestampQueries.surfaceTemporalStart, timestampQueries.surfaceTemporalEnd);
        writeZeroDurationTimestampPair(passContext, timestampQueries.surfaceSpatialStart, timestampQueries.surfaceSpatialEnd);
        writeZeroDurationTimestampPair(passContext, timestampQueries.surfaceResolveStart, timestampQueries.surfaceResolveEnd);
    }
    recordVoxelGiInjectPass(passContext, dispatchDims, timestampQueries);
    recordVoxelGiPropagationPass(passContext, dispatchDims, timestampQueries);
    finalizeVoxelGiPass(passContext, passState);
    if (useRestirSurface) {
        std::swap(m_voxelGiRestirReservoirCurrentBufferHandle, m_voxelGiRestirReservoirPreviousBufferHandle);
        m_voxelGiRestirHistoryValid = true;
        m_voxelGiRestirHistoryResetReason = "history_valid";
    }
}

} // namespace voxelsprout::render
