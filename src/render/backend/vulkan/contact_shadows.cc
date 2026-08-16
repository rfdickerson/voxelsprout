#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cstdint>

#include "core/log.h"
#include "sim/network_procedural.h"

namespace odai::render {

#include "render/renderer_shared.h"

namespace {

constexpr const char* kDepthShaderPath =
    "../src/render/shaders/contact_shadow_depth.comp.slang.spv";
constexpr const char* kTraceShaderPath =
    "../src/render/shaders/contact_shadow_trace.comp.slang.spv";
constexpr const char* kResolveShaderPath =
    "../src/render/shaders/contact_shadow_resolve.comp.slang.spv";
constexpr std::uint32_t kWorkgroupSize = 8u;
constexpr std::uint32_t kMaxDepthMipCount = 6u;

struct ContactShadowPushConstants {
    std::uint32_t extent[4]{};
    std::uint32_t dispatch[4]{};
    std::uint32_t cluster[4]{};
    float params[4]{};
};
static_assert(sizeof(ContactShadowPushConstants) == 64u);

std::uint32_t divideRoundUp(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

}  // namespace

bool RendererBackend::createContactShadowResources() {
    if (!readBinaryFile(kDepthShaderPath).has_value() ||
        !readBinaryFile(kTraceShaderPath).has_value() ||
        !readBinaryFile(kResolveShaderPath).has_value()) {
        VOX_LOGW("render")
            << "contact-shadow shaders unavailable; authored interiors fall back to shadow maps";
        m_contactShadowAvailable = false;
        return true;
    }

    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    bindings[0] = {0u, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1u, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    for (std::uint32_t binding = 2u; binding < bindings.size(); ++binding) {
        bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u,
                             VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    if (!createDescriptorSetLayout(
            bindings,
            m_contactShadowDescriptorSetLayout,
            "vkCreateDescriptorSetLayout(contactShadows)",
            "renderer.descriptorSetLayout.contactShadows",
            nullptr,
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)) {
        destroyContactShadowResources();
        return false;
    }
    if (!createDescriptorBufferSet(
            m_contactShadowDescriptorSetLayout,
            kMaxFramesInFlight,
            VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
            "renderer.descriptorBuffer.contactShadows",
            m_contactShadowBufferSet)) {
        destroyContactShadowResources();
        return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(ContactShadowPushConstants);
    const std::array<VkPushConstantRange, 1> pushRanges = {pushRange};
    if (!createComputePipelineLayout(
            m_contactShadowDescriptorSetLayout,
            pushRanges,
            m_contactShadowPipelineLayout,
            "vkCreatePipelineLayout(contactShadows)",
            "renderer.pipelineLayout.contactShadows")) {
        destroyContactShadowResources();
        return false;
    }

    const auto createPassPipeline = [&](const char* path,
                                        const char* moduleName,
                                        const char* pipelineName,
                                        VkPipeline& pipeline) -> bool {
        VkShaderModule module = VK_NULL_HANDLE;
        if (!createShaderModuleFromFile(m_device, path, moduleName, module)) {
            return false;
        }
        const bool created = createComputePipeline(
            m_contactShadowPipelineLayout,
            module,
            pipeline,
            "vkCreateComputePipelines(contactShadows)",
            pipelineName,
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
        vkDestroyShaderModule(m_device, module, nullptr);
        return created;
    };
    if (!createPassPipeline(kDepthShaderPath, "contact_shadow_depth.comp",
                            "pipeline.contactShadows.depth", m_contactShadowDepthPipeline) ||
        !createPassPipeline(kTraceShaderPath, "contact_shadow_trace.comp",
                            "pipeline.contactShadows.trace", m_contactShadowTracePipeline) ||
        !createPassPipeline(kResolveShaderPath, "contact_shadow_resolve.comp",
                            "pipeline.contactShadows.resolve", m_contactShadowResolvePipeline)) {
        destroyContactShadowResources();
        return false;
    }

    m_contactShadowAvailable = true;
    return true;
}

bool RendererBackend::createContactShadowBuffers(VkExtent2D renderExtent) {
    const VkExtent2D halfExtent{
        std::max(1u, divideRoundUp(renderExtent.width, 2u)),
        std::max(1u, divideRoundUp(renderExtent.height, 2u))};

    std::uint64_t depthEntryCount = 0u;
    std::uint32_t mipWidth = halfExtent.width;
    std::uint32_t mipHeight = halfExtent.height;
    std::uint32_t mipCount = 0u;
    while (mipCount < kMaxDepthMipCount) {
        depthEntryCount += static_cast<std::uint64_t>(mipWidth) * mipHeight;
        ++mipCount;
        if (mipWidth == 1u && mipHeight == 1u) {
            break;
        }
        mipWidth = std::max(1u, divideRoundUp(mipWidth, 2u));
        mipHeight = std::max(1u, divideRoundUp(mipHeight, 2u));
    }

    const VkDeviceSize depthSize =
        static_cast<VkDeviceSize>(depthEntryCount) * sizeof(float) * 2u;
    const VkDeviceSize halfSize =
        static_cast<VkDeviceSize>(halfExtent.width) * halfExtent.height *
        sizeof(std::uint32_t) * 4u;
    const VkDeviceSize fullSize =
        static_cast<VkDeviceSize>(renderExtent.width) * renderExtent.height *
        sizeof(std::uint32_t) * 2u;
    if (m_contactShadowDepthBufferHandle != kInvalidBufferHandle &&
        m_contactShadowDepthBufferSize >= depthSize &&
        m_contactShadowHalfBufferSize >= halfSize &&
        m_contactShadowFullMaskBufferSize >= fullSize) {
        m_contactShadowHalfExtent = halfExtent;
        m_contactShadowDepthMipCount = mipCount;
        return true;
    }

    const auto destroyBuffer = [&](BufferHandle& handle) {
        if (handle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(handle);
            handle = kInvalidBufferHandle;
        }
    };
    destroyBuffer(m_contactShadowDepthBufferHandle);
    destroyBuffer(m_contactShadowHalfBufferHandle);
    destroyBuffer(m_contactShadowFullMaskBufferHandle);
    m_contactShadowDepthBufferSize = 0;
    m_contactShadowHalfBufferSize = 0;
    m_contactShadowFullMaskBufferSize = 0;

    const auto createStorageBuffer = [&](VkDeviceSize size,
                                         const char* name,
                                         BufferHandle& handle) -> bool {
        BufferCreateDesc desc{};
        desc.size = size;
        desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        desc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        handle = m_bufferAllocator.createBuffer(desc);
        if (handle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "failed to create " << name;
            return false;
        }
        setObjectName(VK_OBJECT_TYPE_BUFFER,
                      vkHandleToUint64(m_bufferAllocator.getBuffer(handle)), name);
        return true;
    };
    if (!createStorageBuffer(depthSize, "contactShadows.depthHierarchy",
                             m_contactShadowDepthBufferHandle) ||
        !createStorageBuffer(halfSize, "contactShadows.halfRecords",
                             m_contactShadowHalfBufferHandle) ||
        !createStorageBuffer(fullSize, "contactShadows.fullMask",
                             m_contactShadowFullMaskBufferHandle)) {
        destroyBuffer(m_contactShadowDepthBufferHandle);
        destroyBuffer(m_contactShadowHalfBufferHandle);
        destroyBuffer(m_contactShadowFullMaskBufferHandle);
        return false;
    }
    m_contactShadowDepthBufferSize = depthSize;
    m_contactShadowHalfBufferSize = halfSize;
    m_contactShadowFullMaskBufferSize = fullSize;
    m_contactShadowHalfExtent = halfExtent;
    m_contactShadowDepthMipCount = mipCount;
    VOX_LOGI("render") << "contact shadows: " << halfExtent.width << "x"
                        << halfExtent.height << ", depthMips=" << mipCount
                        << ", buffers="
                        << ((depthSize + halfSize + fullSize) / (1024u * 1024u)) << " MB";
    return true;
}

void RendererBackend::recordScreenSpaceDepthHierarchyPass(
    const FrameExecutionContext& context) {
    if ((!m_contactShadowActive && !m_screenSpaceGiActive) ||
        !m_contactShadowAvailable) {
        return;
    }
    const VkCommandBuffer commandBuffer = context.commandBuffer;
    const VkQueryPool queryPool = context.gpuTimestampQueryPool;
    const auto timestampTop = [&](std::uint32_t query) {
        if (queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 queryPool, query);
        }
    };
    const auto timestampBottom = [&](std::uint32_t query) {
        if (queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 queryPool, query);
        }
    };
    beginDebugLabel(commandBuffer, "Pass: Screen-space depth hierarchy", 0.24f, 0.18f, 0.08f, 1.0f);
    timestampTop(kGpuTimestampQueryScreenDepthStart);

    ContactShadowPushConstants push{};
    push.extent[0] = m_renderExtent.width;
    push.extent[1] = m_renderExtent.height;
    push.extent[2] = m_contactShadowHalfExtent.width;
    push.extent[3] = m_contactShadowHalfExtent.height;
    push.cluster[0] = m_lightClusterGridX;
    push.cluster[1] = m_lightClusterGridY;
    push.cluster[2] = kLightClusterSliceCount;
    push.cluster[3] = kLightClusterTileSize;
    push.params[0] = m_lightClusterSliceScale;
    push.params[1] = m_lightClusterSliceBias;
    push.params[2] = 640.0f;

    bindDescriptorBuffer(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_contactShadowPipelineLayout, 0,
        m_contactShadowBufferSet, m_currentFrame);
    vkCmdBindPipeline(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_contactShadowDepthPipeline);

    std::uint32_t sourceOffset = 0u;
    std::uint32_t destinationOffset = 0u;
    std::uint32_t sourceWidth = m_contactShadowHalfExtent.width;
    std::uint32_t sourceHeight = m_contactShadowHalfExtent.height;
    for (std::uint32_t mip = 0u; mip < m_contactShadowDepthMipCount; ++mip) {
        const std::uint32_t width = mip == 0u
            ? m_contactShadowHalfExtent.width
            : std::max(1u, divideRoundUp(sourceWidth, 2u));
        const std::uint32_t height = mip == 0u
            ? m_contactShadowHalfExtent.height
            : std::max(1u, divideRoundUp(sourceHeight, 2u));
        push.dispatch[0] = mip;
        push.dispatch[1] = sourceOffset;
        push.dispatch[2] = destinationOffset;
        push.dispatch[3] = m_taaJitterPhase & 3u;
        vkCmdPushConstants(commandBuffer, m_contactShadowPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(commandBuffer, divideRoundUp(width, kWorkgroupSize),
                      divideRoundUp(height, kWorkgroupSize), 1u);

        VkBufferMemoryBarrier2 mipBarrier{};
        mipBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        mipBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mipBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mipBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mipBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mipBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mipBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mipBarrier.buffer = m_bufferAllocator.getBuffer(m_contactShadowDepthBufferHandle);
        mipBarrier.offset = 0;
        mipBarrier.size = VK_WHOLE_SIZE;
        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.bufferMemoryBarrierCount = 1u;
        dependency.pBufferMemoryBarriers = &mipBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);

        sourceOffset = destinationOffset;
        destinationOffset += width * height;
        sourceWidth = width;
        sourceHeight = height;
    }

    timestampBottom(kGpuTimestampQueryScreenDepthEnd);
    endDebugLabel(commandBuffer);
}

void RendererBackend::recordContactShadowPass(const FrameExecutionContext& context) {
    if (!m_contactShadowActive) {
        return;
    }
    const VkCommandBuffer commandBuffer = context.commandBuffer;
    const VkQueryPool queryPool = context.gpuTimestampQueryPool;
    const auto timestampTop = [&](std::uint32_t query) {
        if (queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_NONE, queryPool, query);
        }
    };
    const auto timestampBottom = [&](std::uint32_t query) {
        if (queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(
                commandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queryPool, query);
        }
    };
    beginDebugLabel(commandBuffer, "Pass: Contact shadow trace", 0.34f, 0.16f, 0.08f, 1.0f);
    timestampTop(kGpuTimestampQueryContactShadowTraceStart);

    ContactShadowPushConstants push{};
    push.extent[0] = m_renderExtent.width;
    push.extent[1] = m_renderExtent.height;
    push.extent[2] = m_contactShadowHalfExtent.width;
    push.extent[3] = m_contactShadowHalfExtent.height;
    push.cluster[0] = m_lightClusterGridX;
    push.cluster[1] = m_lightClusterGridY;
    push.cluster[2] = kLightClusterSliceCount;
    push.cluster[3] = kLightClusterTileSize;
    push.params[0] = m_lightClusterSliceScale;
    push.params[1] = m_lightClusterSliceBias;
    push.params[2] = 160.0f;

    bindDescriptorBuffer(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_contactShadowPipelineLayout, 0,
        m_contactShadowBufferSet, m_currentFrame);

    push.dispatch[0] = m_contactShadowDepthMipCount;
    push.dispatch[1] = 0u;
    push.dispatch[2] = 0u;
    push.dispatch[3] = m_taaJitterPhase & 3u;
    vkCmdBindPipeline(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_contactShadowTracePipeline);
    vkCmdPushConstants(commandBuffer, m_contactShadowPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(commandBuffer,
                  divideRoundUp(m_contactShadowHalfExtent.width, kWorkgroupSize),
                  divideRoundUp(m_contactShadowHalfExtent.height, kWorkgroupSize), 1u);
    timestampBottom(kGpuTimestampQueryContactShadowTraceEnd);
    endDebugLabel(commandBuffer);

    VkBufferMemoryBarrier2 halfBarrier{};
    halfBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    halfBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    halfBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    halfBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    halfBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    halfBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    halfBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    halfBarrier.buffer = m_bufferAllocator.getBuffer(m_contactShadowHalfBufferHandle);
    halfBarrier.offset = 0;
    halfBarrier.size = VK_WHOLE_SIZE;
    VkDependencyInfo halfDependency{};
    halfDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    halfDependency.bufferMemoryBarrierCount = 1u;
    halfDependency.pBufferMemoryBarriers = &halfBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &halfDependency);

    beginDebugLabel(commandBuffer, "Pass: Contact shadow resolve", 0.48f, 0.24f, 0.10f, 1.0f);
    timestampTop(kGpuTimestampQueryContactShadowResolveStart);
    vkCmdBindPipeline(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_contactShadowResolvePipeline);
    vkCmdPushConstants(commandBuffer, m_contactShadowPipelineLayout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(commandBuffer, divideRoundUp(m_renderExtent.width, kWorkgroupSize),
                  divideRoundUp(m_renderExtent.height, kWorkgroupSize), 1u);
    timestampBottom(kGpuTimestampQueryContactShadowResolveEnd);
    endDebugLabel(commandBuffer);

    VkBufferMemoryBarrier2 fullBarrier{};
    fullBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    fullBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    fullBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    fullBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    fullBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    fullBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    fullBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    fullBarrier.buffer = m_bufferAllocator.getBuffer(m_contactShadowFullMaskBufferHandle);
    fullBarrier.offset = 0;
    fullBarrier.size = VK_WHOLE_SIZE;
    VkDependencyInfo fullDependency{};
    fullDependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    fullDependency.bufferMemoryBarrierCount = 1u;
    fullDependency.pBufferMemoryBarriers = &fullBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &fullDependency);
}

void RendererBackend::destroyContactShadowResources() {
    m_contactShadowActive = false;
    m_contactShadowAvailable = false;
    const auto destroyPipeline = [&](VkPipeline& pipeline) {
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        }
    };
    destroyPipeline(m_contactShadowDepthPipeline);
    destroyPipeline(m_contactShadowTracePipeline);
    destroyPipeline(m_contactShadowResolvePipeline);
    if (m_contactShadowPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_contactShadowPipelineLayout, nullptr);
        m_contactShadowPipelineLayout = VK_NULL_HANDLE;
    }
    destroyDescriptorBufferSet(m_contactShadowBufferSet);
    if (m_contactShadowDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_contactShadowDescriptorSetLayout, nullptr);
        m_contactShadowDescriptorSetLayout = VK_NULL_HANDLE;
    }
    const auto destroyBuffer = [&](BufferHandle& handle) {
        if (handle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(handle);
            handle = kInvalidBufferHandle;
        }
    };
    destroyBuffer(m_contactShadowDepthBufferHandle);
    destroyBuffer(m_contactShadowHalfBufferHandle);
    destroyBuffer(m_contactShadowFullMaskBufferHandle);
    m_contactShadowDepthBufferSize = 0;
    m_contactShadowHalfBufferSize = 0;
    m_contactShadowFullMaskBufferSize = 0;
    m_contactShadowHalfExtent = {};
    m_contactShadowDepthMipCount = 0u;
}

}  // namespace odai::render
