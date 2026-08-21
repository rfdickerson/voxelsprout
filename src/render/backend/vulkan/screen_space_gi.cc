#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cstdint>
#include <cstring>

#include "core/log.h"
#include "render/backend/vulkan/frame_math.h"

namespace odai::render {

#include "render/renderer_shared.h"

namespace {

constexpr const char* kScreenSpaceGiShaderPath =
    "../src/render/shaders/screen_space_gi.comp.slang.spv";
constexpr std::uint32_t kWorkgroupSize = 8u;

struct ScreenSpaceGiPushConstants {
    float prevViewProj[16]{};
    std::uint32_t extent[4]{};
    std::uint32_t dispatch[4]{};
    float params[4]{};
};
static_assert(sizeof(ScreenSpaceGiPushConstants) == 112u);

std::uint32_t divideRoundUp(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

}  // namespace

bool RendererBackend::createScreenSpaceGiResources() {
    if (!readBinaryFile(kScreenSpaceGiShaderPath).has_value() ||
        !m_contactShadowAvailable) {
        VOX_LOGW("render")
            << "screen-space GI unavailable; authored ambient remains active";
        m_screenSpaceGiAvailable = false;
        return true;
    }

    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    bindings[0] = {0u, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1u,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1u, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2u, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u,
                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    for (std::uint32_t binding = 3u; binding < bindings.size(); ++binding) {
        bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u,
                             VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    if (!createDescriptorSetLayout(
            bindings,
            m_screenSpaceGiDescriptorSetLayout,
            "vkCreateDescriptorSetLayout(screenSpaceGi)",
            "renderer.descriptorSetLayout.screenSpaceGi",
            nullptr,
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)) {
        destroyScreenSpaceGiResources();
        return false;
    }
    if (!createDescriptorBufferSet(
            m_screenSpaceGiDescriptorSetLayout,
            kMaxFramesInFlight,
            VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
            "renderer.descriptorBuffer.screenSpaceGi",
            m_screenSpaceGiBufferSet)) {
        destroyScreenSpaceGiResources();
        return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.size = sizeof(ScreenSpaceGiPushConstants);
    const std::array<VkPushConstantRange, 1> pushRanges = {pushRange};
    if (!createComputePipelineLayout(
            m_screenSpaceGiDescriptorSetLayout,
            pushRanges,
            m_screenSpaceGiPipelineLayout,
            "vkCreatePipelineLayout(screenSpaceGi)",
            "renderer.pipelineLayout.screenSpaceGi")) {
        destroyScreenSpaceGiResources();
        return false;
    }

    VkShaderModule module = VK_NULL_HANDLE;
    if (!createShaderModuleFromFile(
            m_device, kScreenSpaceGiShaderPath, "screen_space_gi.comp", module)) {
        destroyScreenSpaceGiResources();
        return false;
    }
    const bool created = createComputePipeline(
        m_screenSpaceGiPipelineLayout,
        module,
        m_screenSpaceGiPipeline,
        "vkCreateComputePipelines(screenSpaceGi)",
        "pipeline.screenSpaceGi",
        VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT);
    vkDestroyShaderModule(m_device, module, nullptr);
    if (!created) {
        destroyScreenSpaceGiResources();
        return false;
    }

    m_screenSpaceGiAvailable = true;
    return true;
}

bool RendererBackend::createScreenSpaceGiBuffers(VkExtent2D renderExtent) {
    const VkExtent2D quarterExtent{
        screenSpaceGiQuarterExtent(renderExtent.width),
        screenSpaceGiQuarterExtent(renderExtent.height)};
    const VkDeviceSize recordSize =
        static_cast<VkDeviceSize>(quarterExtent.width) * quarterExtent.height *
        sizeof(std::uint32_t) * 4u;
    const bool extentChanged =
        quarterExtent.width != m_screenSpaceGiExtent.width ||
        quarterExtent.height != m_screenSpaceGiExtent.height;
    if (m_screenSpaceGiRecordBufferHandles[0] != kInvalidBufferHandle &&
        m_screenSpaceGiRecordBufferHandles[1] != kInvalidBufferHandle &&
        m_screenSpaceGiRecordBufferSize >= recordSize) {
        m_screenSpaceGiExtent = quarterExtent;
        if (extentChanged) {
            m_screenSpaceGiHistoryValid = false;
        }
        return true;
    }

    for (BufferHandle& handle : m_screenSpaceGiRecordBufferHandles) {
        if (handle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(handle);
            handle = kInvalidBufferHandle;
        }
    }
    BufferCreateDesc desc{};
    desc.size = recordSize;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    desc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    for (std::uint32_t index = 0u; index < 2u; ++index) {
        m_screenSpaceGiRecordBufferHandles[index] = m_bufferAllocator.createBuffer(desc);
        if (m_screenSpaceGiRecordBufferHandles[index] == kInvalidBufferHandle) {
            for (BufferHandle& handle : m_screenSpaceGiRecordBufferHandles) {
                if (handle != kInvalidBufferHandle) {
                    m_bufferAllocator.destroyBuffer(handle);
                    handle = kInvalidBufferHandle;
                }
            }
            m_screenSpaceGiRecordBufferSize = 0u;
            return false;
        }
        const std::string name = "screenSpaceGi.records[" + std::to_string(index) + "]";
        setObjectName(
            VK_OBJECT_TYPE_BUFFER,
            vkHandleToUint64(m_bufferAllocator.getBuffer(
                m_screenSpaceGiRecordBufferHandles[index])),
            name.c_str());
    }
    m_screenSpaceGiRecordBufferSize = recordSize;
    m_screenSpaceGiExtent = quarterExtent;
    m_screenSpaceGiHistoryIndex = 0u;
    m_screenSpaceGiHistoryValid = false;
    VOX_LOGI("render") << "screen-space GI: " << quarterExtent.width << "x"
                        << quarterExtent.height << ", records="
                        << ((recordSize * 2u) / (1024u * 1024u)) << " MB";
    return true;
}

void RendererBackend::recordScreenSpaceGiPass(const FrameExecutionContext& context) {
    if (!m_screenSpaceGiActive) {
        return;
    }
    const VkCommandBuffer commandBuffer = context.commandBuffer;
    beginDebugLabel(commandBuffer, "Pass: Screen-space GI", 0.16f, 0.38f, 0.18f, 1.0f);
    if (context.gpuTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(
            commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            context.gpuTimestampQueryPool, kGpuTimestampQueryScreenSpaceGiStart);
    }

    ScreenSpaceGiPushConstants push{};
    std::memcpy(push.prevViewProj, m_taaPrevViewProjColumnMajor.m,
                sizeof(push.prevViewProj));
    push.extent[0] = m_renderExtent.width;
    push.extent[1] = m_renderExtent.height;
    push.extent[2] = m_screenSpaceGiExtent.width;
    push.extent[3] = m_screenSpaceGiExtent.height;
    push.dispatch[0] = m_contactShadowDepthMipCount;
    push.dispatch[1] = m_taaJitterPhase & 7u;
    push.dispatch[2] =
        (m_screenSpaceGiHistoryValid && m_taaHistoryValid &&
         m_taaPrevViewProjValid) ? 1u : 0u;
    push.params[0] = 640.0f;
    push.params[1] = 4.0f;
    push.params[2] = m_taaPrevJitterNdc[0];
    push.params[3] = m_taaPrevJitterNdc[1];

    bindDescriptorBuffer(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_screenSpaceGiPipelineLayout,
        0, m_screenSpaceGiBufferSet, m_currentFrame);
    vkCmdBindPipeline(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_screenSpaceGiPipeline);
    vkCmdPushConstants(
        commandBuffer, m_screenSpaceGiPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(push), &push);
    vkCmdDispatch(
        commandBuffer,
        divideRoundUp(m_screenSpaceGiExtent.width, kWorkgroupSize),
        divideRoundUp(m_screenSpaceGiExtent.height, kWorkgroupSize), 1u);

    if (context.gpuTimestampQueryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(
            commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            context.gpuTimestampQueryPool, kGpuTimestampQueryScreenSpaceGiEnd);
    }
    endDebugLabel(commandBuffer);

    const std::uint32_t currentIndex = m_screenSpaceGiHistoryIndex ^ 1u;
    VkBufferMemoryBarrier2 recordBarrier{};
    recordBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    recordBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    recordBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    recordBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    recordBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    recordBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    recordBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    recordBarrier.buffer = m_bufferAllocator.getBuffer(
        m_screenSpaceGiRecordBufferHandles[currentIndex]);
    recordBarrier.size = VK_WHOLE_SIZE;
    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.bufferMemoryBarrierCount = 1u;
    dependency.pBufferMemoryBarriers = &recordBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    m_screenSpaceGiHistoryIndex = currentIndex;
    m_screenSpaceGiHistoryValid = true;
}

void RendererBackend::destroyScreenSpaceGiResources() {
    m_screenSpaceGiActive = false;
    m_screenSpaceGiAvailable = false;
    m_screenSpaceGiHistoryValid = false;
    if (m_screenSpaceGiPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_screenSpaceGiPipeline, nullptr);
        m_screenSpaceGiPipeline = VK_NULL_HANDLE;
    }
    if (m_screenSpaceGiPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_screenSpaceGiPipelineLayout, nullptr);
        m_screenSpaceGiPipelineLayout = VK_NULL_HANDLE;
    }
    destroyDescriptorBufferSet(m_screenSpaceGiBufferSet);
    if (m_screenSpaceGiDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(
            m_device, m_screenSpaceGiDescriptorSetLayout, nullptr);
        m_screenSpaceGiDescriptorSetLayout = VK_NULL_HANDLE;
    }
    for (BufferHandle& handle : m_screenSpaceGiRecordBufferHandles) {
        if (handle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(handle);
            handle = kInvalidBufferHandle;
        }
    }
    m_screenSpaceGiRecordBufferSize = 0u;
    m_screenSpaceGiExtent = {};
    m_screenSpaceGiHistoryIndex = 0u;
}

}  // namespace odai::render
