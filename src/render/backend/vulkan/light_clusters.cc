#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>


namespace odai::render {

// Included inside the namespace, like every other frame_pass_*.cc here: this
// header is shared implementation, not an interface.
#include "render/renderer_shared.h"

namespace {

constexpr const char* kLightClusterShaderPath =
    "../src/render/shaders/light_cluster_cull.comp.slang.spv";

// One thread per cluster, so the workgroup size is just the dispatch
// granularity. Matches [numthreads(64,1,1)] in the shader.
constexpr uint32_t kLightClusterWorkgroupSize = 64u;

// THE MASK IS TWO WORDS BECAUSE THE LIGHT CAP IS 64. If the cap ever rises this
// stops being a mask and has to become the usual compacted index list with an
// atomic allocator -- which is a different pass, not a bigger constant.
static_assert(kImportedLocalLightCapacity == 64,
              "the cluster light mask is a fixed 64 bits; a larger light cap needs an index list "
              "(see src/render/shaders/light_clusters.slang)");

}  // namespace

uint32_t RendererBackend::lightClusterGridX(VkExtent2D extent) {
    return std::max(1u, (extent.width + (kLightClusterTileSize - 1u)) / kLightClusterTileSize);
}

uint32_t RendererBackend::lightClusterGridY(VkExtent2D extent) {
    return std::max(1u, (extent.height + (kLightClusterTileSize - 1u)) / kLightClusterTileSize);
}

uint32_t RendererBackend::lightClusterCount(VkExtent2D extent) {
    return lightClusterGridX(extent) * lightClusterGridY(extent) * kLightClusterSliceCount;
}

bool RendererBackend::createLightClusterResources() {
    if (!readBinaryFile(kLightClusterShaderPath).has_value()) {
        VOX_LOGI("render")
            << "clustered light culling shader not found; falling back to per-fragment light "
               "iteration (expected: "
            << kLightClusterShaderPath << ")";
        m_lightClusterAvailable = false;
        return true;  // not fatal: the fragment shader's fallback path still lights the scene
    }

    if (m_lightClusterDescriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding cameraBinding{};
        cameraBinding.binding = 0;
        cameraBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        cameraBinding.descriptorCount = 1;
        cameraBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding maskBinding{};
        maskBinding.binding = 1;
        maskBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        maskBinding.descriptorCount = 1;
        maskBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        const std::array<VkDescriptorSetLayoutBinding, 2> bindings = {cameraBinding, maskBinding};
        if (!createDescriptorSetLayout(
                bindings,
                m_lightClusterDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(lightClusters)",
                "renderer.descriptorSetLayout.lightClusters",
                nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)) {
            destroyLightClusterResources();
            return false;
        }
    }

    if (!m_lightClusterBufferSet.valid()) {
        if (!createDescriptorBufferSet(
                m_lightClusterDescriptorSetLayout,
                kMaxFramesInFlight,
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
                "renderer.descriptorBuffer.lightClusters",
                m_lightClusterBufferSet)) {
            destroyLightClusterResources();
            return false;
        }
    }

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (!createShaderModuleFromFile(
            m_device, kLightClusterShaderPath, "light_cluster_cull.comp", shaderModule)) {
        destroyLightClusterResources();
        return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(LightClusterPushConstants);
    const std::array<VkPushConstantRange, 1> pushConstantRanges = {pushConstantRange};

    if (!createComputePipelineLayout(
            m_lightClusterDescriptorSetLayout,
            pushConstantRanges,
            m_lightClusterPipelineLayout,
            "vkCreatePipelineLayout(lightClusters)",
            "renderer.pipelineLayout.lightClusters")) {
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
        destroyLightClusterResources();
        return false;
    }

    if (!createComputePipeline(
            m_lightClusterPipelineLayout,
            shaderModule,
            m_lightClusterPipeline,
            "vkCreateComputePipelines(lightClusters)",
            "pipeline.lightClusters.cull",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)) {
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
        destroyLightClusterResources();
        return false;
    }
    vkDestroyShaderModule(m_device, shaderModule, nullptr);

    m_lightClusterAvailable = true;
    return true;
}

// Sized from the render extent, so it is rebuilt with the swapchain rather than
// held at a worst case. A 4K render extent is 60x34x24 = 48960 clusters, i.e.
// 392 KB of mask -- small enough that per-frame-in-flight regions are not worth
// the bookkeeping: the pass writes the whole buffer before anything reads it,
// and the barrier below is what orders that.
bool RendererBackend::createLightClusterBuffer(VkExtent2D renderExtent) {
    const uint32_t clusters = lightClusterCount(renderExtent);
    const VkDeviceSize requiredSize =
        static_cast<VkDeviceSize>(clusters) * sizeof(uint32_t) * 2u;
    if (m_lightClusterBufferHandle != kInvalidBufferHandle &&
        m_lightClusterBufferSize >= requiredSize) {
        m_lightClusterGridX = lightClusterGridX(renderExtent);
        m_lightClusterGridY = lightClusterGridY(renderExtent);
        return true;
    }
    if (m_lightClusterBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_lightClusterBufferHandle);
        m_lightClusterBufferHandle = kInvalidBufferHandle;
        m_lightClusterBufferSize = 0;
    }

    BufferCreateDesc desc{};
    desc.size = requiredSize;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    desc.memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    m_lightClusterBufferHandle = m_bufferAllocator.createBuffer(desc);
    if (m_lightClusterBufferHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "failed to create light cluster mask buffer";
        return false;
    }
    m_lightClusterBufferSize = requiredSize;
    m_lightClusterGridX = lightClusterGridX(renderExtent);
    m_lightClusterGridY = lightClusterGridY(renderExtent);
    setObjectName(
        VK_OBJECT_TYPE_BUFFER,
        vkHandleToUint64(m_bufferAllocator.getBuffer(m_lightClusterBufferHandle)),
        "lightClusters.maskBuffer");
    VOX_LOGI("render") << "light clusters: " << m_lightClusterGridX << "x" << m_lightClusterGridY
                       << "x" << kLightClusterSliceCount << " (" << clusters << " clusters, "
                       << (requiredSize / 1024u) << " KB)";
    return true;
}

// The exponential depth mapping, computed on the CPU because both the compute
// pass and the fragment shader must use exactly the same one: a fragment that
// picks a different slice than the cluster it was culled into reads a mask for
// somewhere else, and lights pop across a depth boundary.
void RendererBackend::computeLightClusterSliceParams(
    float nearPlane, float farPlane, float& outScale, float& outBias) const {
    const float safeNear = std::max(nearPlane, 1e-3f);
    const float safeFar = std::max(farPlane, safeNear * 2.0f);
    const float logRatio = std::log2(safeFar / safeNear);
    outScale = static_cast<float>(kLightClusterSliceCount) / std::max(logRatio, 1e-4f);
    outBias = -outScale * std::log2(safeNear);
}

void RendererBackend::recordLightClusterPass(const FrameExecutionContext& context) {
    if (!m_lightClusterCullActive) {
        return;
    }
    VkCommandBuffer commandBuffer = context.commandBuffer;
    beginDebugLabel(commandBuffer, "Pass: Light clusters", 0.86f, 0.72f, 0.24f, 1.0f);

    LightClusterPushConstants push{};
    push.gridX = m_lightClusterGridX;
    push.gridY = m_lightClusterGridY;
    push.gridZ = kLightClusterSliceCount;
    push.tileSize = kLightClusterTileSize;
    push.sliceScale = m_lightClusterSliceScale;
    push.sliceBias = m_lightClusterSliceBias;
    push.extentX = m_renderExtent.width;
    push.extentY = m_renderExtent.height;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_lightClusterPipeline);
    bindDescriptorBuffer(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_lightClusterPipelineLayout, 0,
        m_lightClusterBufferSet, m_currentFrame);
    vkCmdPushConstants(
        commandBuffer, m_lightClusterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
        &push);
    const uint32_t clusters = push.gridX * push.gridY * push.gridZ;
    vkCmdDispatch(
        commandBuffer,
        (clusters + (kLightClusterWorkgroupSize - 1u)) / kLightClusterWorkgroupSize, 1u, 1u);

    // Hand-written, like every barrier here: the main pass reads this buffer
    // from the fragment stage, and the prepass between them does not touch it,
    // so there is nothing else to order against.
    VkBufferMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = m_bufferAllocator.getBuffer(m_lightClusterBufferHandle);
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    endDebugLabel(commandBuffer);
}

void RendererBackend::destroyLightClusterResources() {
    if (m_lightClusterPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_lightClusterPipeline, nullptr);
        m_lightClusterPipeline = VK_NULL_HANDLE;
    }
    if (m_lightClusterPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_lightClusterPipelineLayout, nullptr);
        m_lightClusterPipelineLayout = VK_NULL_HANDLE;
    }
    destroyDescriptorBufferSet(m_lightClusterBufferSet);
    if (m_lightClusterDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_lightClusterDescriptorSetLayout, nullptr);
        m_lightClusterDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (m_lightClusterBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_lightClusterBufferHandle);
        m_lightClusterBufferHandle = kInvalidBufferHandle;
        m_lightClusterBufferSize = 0;
    }
    m_lightClusterAvailable = false;
    m_lightClusterCullActive = false;
}

}  // namespace odai::render
