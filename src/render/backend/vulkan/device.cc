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

namespace voxelsprout::render {

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "render/renderer_shared.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

bool RendererBackend::allocatePerFrameDescriptorSets(
    VkDescriptorPool descriptorPool,
    VkDescriptorSetLayout descriptorSetLayout,
    std::span<VkDescriptorSet> outDescriptorSets,
    const char* failureContext,
    const char* debugNamePrefix
) {
    if (outDescriptorSets.empty()) {
        return true;
    }

    std::vector<VkDescriptorSetLayout> setLayouts(outDescriptorSets.size(), descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(setLayouts.size());
    allocateInfo.pSetLayouts = setLayouts.data();

    const VkResult allocateResult = vkAllocateDescriptorSets(m_device, &allocateInfo, outDescriptorSets.data());
    if (allocateResult != VK_SUCCESS) {
        logVkFailure(failureContext, allocateResult);
        return false;
    }

    if (debugNamePrefix != nullptr) {
        for (std::size_t frameIndex = 0; frameIndex < outDescriptorSets.size(); ++frameIndex) {
            const std::string setName = std::string(debugNamePrefix) + std::to_string(frameIndex);
            setObjectName(
                VK_OBJECT_TYPE_DESCRIPTOR_SET,
                vkHandleToUint64(outDescriptorSets[frameIndex]),
                setName.c_str()
            );
        }
    }

    return true;
}


bool RendererBackend::createDescriptorSetLayout(
    std::span<const VkDescriptorSetLayoutBinding> bindings,
    VkDescriptorSetLayout& outDescriptorSetLayout,
    const char* failureContext,
    const char* debugName,
    const void* pNext
) {
    VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCreateInfo.pNext = pNext;
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
    const char* debugName
) {
    outPipeline = VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shaderModule;
    stage.pName = "main";

    VkComputePipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = stage;
    pipelineCreateInfo.layout = pipelineLayout;
    const VkResult pipelineResult =
        vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &outPipeline);
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


} // namespace voxelsprout::render
