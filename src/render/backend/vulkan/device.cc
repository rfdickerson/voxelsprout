#include "render/backend/vulkan/renderer_backend.h"

#include <GLFW/glfw3.h>
#include "core/grid3.h"
#include "core/log.h"
#include "math/math.h"
#include "sim/network_procedural.h"
#include "world/chunk_mesher.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace odai::render {

#include "render/renderer_shared.h"

namespace {

constexpr const char* kPipelineCacheFilePath = "odai_pipeline_cache.bin";

} // namespace

bool RendererBackend::createPipelineCache() {
    if (m_pipelineCache != VK_NULL_HANDLE) {
        return true;
    }

    // Prior-run cache data is only valid for the same device and driver build;
    // the header check avoids feeding the driver stale bytes after an upgrade
    // or a GPU swap (implementations must reject mismatches anyway, but a
    // corrupt file could still fail cache creation outright).
    std::vector<char> initialData;
    std::ifstream cacheFile(kPipelineCacheFilePath, std::ios::binary | std::ios::ate);
    if (cacheFile) {
        const std::streamsize size = cacheFile.tellg();
        if (size >= static_cast<std::streamsize>(sizeof(VkPipelineCacheHeaderVersionOne))) {
            cacheFile.seekg(0, std::ios::beg);
            initialData.resize(static_cast<std::size_t>(size));
            if (cacheFile.read(initialData.data(), size)) {
                VkPipelineCacheHeaderVersionOne header{};
                std::memcpy(&header, initialData.data(), sizeof(header));
                VkPhysicalDeviceProperties deviceProperties{};
                vkGetPhysicalDeviceProperties(m_physicalDevice, &deviceProperties);
                const bool headerMatches =
                    header.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
                    header.vendorID == deviceProperties.vendorID &&
                    header.deviceID == deviceProperties.deviceID &&
                    std::memcmp(header.pipelineCacheUUID, deviceProperties.pipelineCacheUUID, VK_UUID_SIZE) == 0;
                if (!headerMatches) {
                    VOX_LOGI("render") << "pipeline cache file stale (device/driver changed); starting empty\n";
                    initialData.clear();
                }
            } else {
                initialData.clear();
            }
        }
    }

    VkPipelineCacheCreateInfo cacheCreateInfo{};
    cacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheCreateInfo.initialDataSize = initialData.size();
    cacheCreateInfo.pInitialData = initialData.empty() ? nullptr : initialData.data();

    VkResult cacheResult = vkCreatePipelineCache(m_device, &cacheCreateInfo, nullptr, &m_pipelineCache);
    if (cacheResult != VK_SUCCESS && !initialData.empty()) {
        // A corrupt cache file must not block startup — retry empty.
        cacheCreateInfo.initialDataSize = 0;
        cacheCreateInfo.pInitialData = nullptr;
        cacheResult = vkCreatePipelineCache(m_device, &cacheCreateInfo, nullptr, &m_pipelineCache);
    }
    if (cacheResult != VK_SUCCESS) {
        logVkFailure("vkCreatePipelineCache", cacheResult);
        m_pipelineCache = VK_NULL_HANDLE;
        return false;
    }

    setObjectName(VK_OBJECT_TYPE_PIPELINE_CACHE, vkHandleToUint64(m_pipelineCache), "renderer.pipelineCache");
    VOX_LOGI("render") << "pipeline cache ready (loaded " << initialData.size() << " bytes from "
                       << kPipelineCacheFilePath << ")\n";
    return true;
}

void RendererBackend::savePipelineCache() {
    if (m_pipelineCache == VK_NULL_HANDLE) {
        return;
    }

    std::size_t dataSize = 0;
    if (vkGetPipelineCacheData(m_device, m_pipelineCache, &dataSize, nullptr) == VK_SUCCESS && dataSize > 0) {
        std::vector<char> data(dataSize);
        if (vkGetPipelineCacheData(m_device, m_pipelineCache, &dataSize, data.data()) == VK_SUCCESS) {
            std::ofstream cacheFile(kPipelineCacheFilePath, std::ios::binary | std::ios::trunc);
            if (cacheFile && cacheFile.write(data.data(), static_cast<std::streamsize>(dataSize))) {
                VOX_LOGI("render") << "pipeline cache saved (" << dataSize << " bytes to "
                                   << kPipelineCacheFilePath << ")\n";
            } else {
                VOX_LOGW("render") << "failed writing pipeline cache file: " << kPipelineCacheFilePath << "\n";
            }
        }
    }
}

void RendererBackend::destroyPipelineCache() {
    if (m_pipelineCache == VK_NULL_HANDLE) {
        return;
    }
    savePipelineCache();
    vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
    m_pipelineCache = VK_NULL_HANDLE;
}

bool RendererBackend::createDescriptorSetLayout(
    std::span<const VkDescriptorSetLayoutBinding> bindings,
    VkDescriptorSetLayout& outDescriptorSetLayout,
    const char* failureContext,
    const char* debugName,
    const void* pNext,
    VkDescriptorSetLayoutCreateFlags flags
) {
    VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCreateInfo.pNext = pNext;
    layoutCreateInfo.flags = flags;
    layoutCreateInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutCreateInfo.pBindings = bindings.empty() ? nullptr : bindings.data();

    const VkResult layoutResult =
        vkCreateDescriptorSetLayout(m_device, &layoutCreateInfo, nullptr, &outDescriptorSetLayout);
    if (layoutResult != VK_SUCCESS) {
        logVkFailure(failureContext, layoutResult);
        return false;
    }

    if (debugName != nullptr) {
        setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, vkHandleToUint64(outDescriptorSetLayout), debugName);
    }
    return true;
}


bool RendererBackend::createDescriptorPool(
    std::span<const VkDescriptorPoolSize> poolSizes,
    uint32_t maxSets,
    VkDescriptorPool& outDescriptorPool,
    const char* failureContext,
    const char* debugName,
    VkDescriptorPoolCreateFlags flags
) {
    VkDescriptorPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.flags = flags;
    poolCreateInfo.maxSets = maxSets;
    poolCreateInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolCreateInfo.pPoolSizes = poolSizes.empty() ? nullptr : poolSizes.data();

    const VkResult poolResult = vkCreateDescriptorPool(m_device, &poolCreateInfo, nullptr, &outDescriptorPool);
    if (poolResult != VK_SUCCESS) {
        logVkFailure(failureContext, poolResult);
        return false;
    }

    if (debugName != nullptr) {
        setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL, vkHandleToUint64(outDescriptorPool), debugName);
    }
    return true;
}


bool RendererBackend::createComputePipelineLayout(
    VkDescriptorSetLayout descriptorSetLayout,
    std::span<const VkPushConstantRange> pushConstantRanges,
    VkPipelineLayout& outPipelineLayout,
    const char* failureContext,
    const char* debugName
) {
    outPipelineLayout = VK_NULL_HANDLE;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
    pipelineLayoutCreateInfo.pPushConstantRanges = pushConstantRanges.empty() ? nullptr : pushConstantRanges.data();

    const VkResult pipelineLayoutResult =
        vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &outPipelineLayout);
    if (pipelineLayoutResult != VK_SUCCESS) {
        logVkFailure(failureContext, pipelineLayoutResult);
        return false;
    }

    if (debugName != nullptr) {
        setObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT, vkHandleToUint64(outPipelineLayout), debugName);
    }

    return true;
}


bool RendererBackend::createComputePipeline(
    VkPipelineLayout pipelineLayout,
    VkShaderModule shaderModule,
    VkPipeline& outPipeline,
    const char* failureContext,
    const char* debugName,
    VkPipelineCreateFlags pipelineFlags
) {
    outPipeline = VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shaderModule;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.flags = pipelineFlags;
    pipelineCreateInfo.stage = stage;
    pipelineCreateInfo.layout = pipelineLayout;
    const VkResult pipelineResult =
        vkCreateComputePipelines(m_device, m_pipelineCache, 1, &pipelineCreateInfo, nullptr, &outPipeline);
    if (pipelineResult != VK_SUCCESS) {
        logVkFailure(failureContext, pipelineResult);
        return false;
    }

    if (debugName != nullptr) {
        setObjectName(VK_OBJECT_TYPE_PIPELINE, vkHandleToUint64(outPipeline), debugName);
    }

    return true;
}


void RendererBackend::setObjectName(VkObjectType objectType, uint64_t objectHandle, const char* name) const {
    if (m_setDebugUtilsObjectName == nullptr || m_device == VK_NULL_HANDLE || objectHandle == 0 || name == nullptr) {
        return;
    }
    VkDebugUtilsObjectNameInfoEXT nameInfo{};
    nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    nameInfo.objectType = objectType;
    nameInfo.objectHandle = objectHandle;
    nameInfo.pObjectName = name;
    m_setDebugUtilsObjectName(m_device, &nameInfo);
}


void RendererBackend::beginDebugLabel(
    VkCommandBuffer commandBuffer,
    const char* name,
    float r,
    float g,
    float b,
    float a
) const {
    if (m_cmdBeginDebugUtilsLabel == nullptr || commandBuffer == VK_NULL_HANDLE || name == nullptr) {
        return;
    }
    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0] = r;
    label.color[1] = g;
    label.color[2] = b;
    label.color[3] = a;
    m_cmdBeginDebugUtilsLabel(commandBuffer, &label);
}


void RendererBackend::endDebugLabel(VkCommandBuffer commandBuffer) const {
    if (m_cmdEndDebugUtilsLabel == nullptr || commandBuffer == VK_NULL_HANDLE) {
        return;
    }
    m_cmdEndDebugUtilsLabel(commandBuffer);
}


void RendererBackend::insertDebugLabel(
    VkCommandBuffer commandBuffer,
    const char* name,
    float r,
    float g,
    float b,
    float a
) const {
    if (m_cmdInsertDebugUtilsLabel == nullptr || commandBuffer == VK_NULL_HANDLE || name == nullptr) {
        return;
    }
    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name;
    label.color[0] = r;
    label.color[1] = g;
    label.color[2] = b;
    label.color[3] = a;
    m_cmdInsertDebugUtilsLabel(commandBuffer, &label);
}


} // namespace odai::render
