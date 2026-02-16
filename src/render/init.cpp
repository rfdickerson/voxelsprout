#include "render/Renderer.hpp"

#include <GLFW/glfw3.h>
#include "core/Grid3.hpp"
#include "core/Log.hpp"
#include "math/Math.hpp"
#include "sim/NetworkProcedural.hpp"
#include "world/ChunkMesher.hpp"

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

namespace render {

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "render/Renderer.Shared.hpp"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

bool isDeviceExtensionAvailable(VkPhysicalDevice physicalDevice, const char* extensionName) {
    if (physicalDevice == VK_NULL_HANDLE || extensionName == nullptr || extensionName[0] == '\0') {
        return false;
    }

    uint32_t extensionCount = 0;
    VkResult result = vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
    if (result != VK_SUCCESS || extensionCount == 0) {
        return false;
    }

    std::vector<VkExtensionProperties> extensionProperties(extensionCount);
    result = vkEnumerateDeviceExtensionProperties(
        physicalDevice,
        nullptr,
        &extensionCount,
        extensionProperties.data()
    );
    if (result != VK_SUCCESS) {
        return false;
    }

    for (const VkExtensionProperties& extensionProperty : extensionProperties) {
        if (std::strcmp(extensionProperty.extensionName, extensionName) == 0) {
            return true;
        }
    }
    return false;
}

void appendDeviceExtensionIfMissing(std::vector<const char*>& extensions, const char* extensionName) {
    if (extensionName == nullptr || extensionName[0] == '\0') {
        return;
    }

    for (const char* existingExtension : extensions) {
        if (existingExtension != nullptr && std::strcmp(existingExtension, extensionName) == 0) {
            return;
        }
    }

    extensions.push_back(extensionName);
}


bool Renderer::init(GLFWwindow* window, const world::ChunkGrid& chunkGrid) {
    using Clock = std::chrono::steady_clock;
    const auto initStart = Clock::now();
    auto elapsedMs = [](const Clock::time_point& start) -> std::int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    };
    auto runStep = [&](const char* stepName, auto&& stepFn) -> bool {
        const auto stepStart = Clock::now();
        const bool ok = stepFn();
        VOX_LOGI("render") << "init step " << stepName << " took " << elapsedMs(stepStart) << " ms\n";
        return ok;
    };

    VOX_LOGI("render") << "init begin\n";
    m_window = window;
    if (m_window == nullptr) {
        VOX_LOGE("render") << "init failed: window is null\n";
        return false;
    }
    bool hasPaletteOverride = false;
    for (const std::uint32_t rgba : m_voxelBaseColorPaletteRgba) {
        if (rgba != 0u) {
            hasPaletteOverride = true;
            break;
        }
    }
    if (!hasPaletteOverride) {
        for (std::size_t i = 0; i < m_voxelBaseColorPaletteRgba.size(); ++i) {
            const std::uint8_t shade = static_cast<std::uint8_t>((255u * static_cast<std::uint32_t>(i)) / 15u);
            m_voxelBaseColorPaletteRgba[i] =
                static_cast<std::uint32_t>(shade) |
                (static_cast<std::uint32_t>(shade) << 8u) |
                (static_cast<std::uint32_t>(shade) << 16u) |
                (0xFFu << 24u);
        }
    }

    if (glfwVulkanSupported() == GLFW_FALSE) {
        VOX_LOGE("render") << "init failed: glfwVulkanSupported returned false\n";
        return false;
    }

    if (!runStep("createInstance", [&] { return createInstance(); })) {
        VOX_LOGE("render") << "init failed at createInstance\n";
        shutdown();
        return false;
    }
    if (!runStep("createSurface", [&] { return createSurface(); })) {
        VOX_LOGE("render") << "init failed at createSurface\n";
        shutdown();
        return false;
    }
    if (!runStep("pickPhysicalDevice", [&] { return pickPhysicalDevice(); })) {
        VOX_LOGE("render") << "init failed at pickPhysicalDevice\n";
        shutdown();
        return false;
    }
    if (!runStep("createLogicalDevice", [&] { return createLogicalDevice(); })) {
        VOX_LOGE("render") << "init failed at createLogicalDevice\n";
        shutdown();
        return false;
    }
    if (!runStep("createTimelineSemaphore", [&] { return createTimelineSemaphore(); })) {
        VOX_LOGE("render") << "init failed at createTimelineSemaphore\n";
        shutdown();
        return false;
    }
    if (!runStep("bufferAllocator.init", [&] {
            return m_bufferAllocator.init(
                m_physicalDevice,
                m_device
                ,
                m_vmaAllocator
            );
        })) {
        VOX_LOGE("render") << "init failed at buffer allocator init\n";
        shutdown();
        return false;
    }
    if (!runStep("createUploadRingBuffer", [&] { return createUploadRingBuffer(); })) {
        VOX_LOGE("render") << "init failed at createUploadRingBuffer\n";
        shutdown();
        return false;
    }
    if (!runStep("createTransferResources", [&] { return createTransferResources(); })) {
        VOX_LOGE("render") << "init failed at createTransferResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createEnvironmentResources", [&] { return createEnvironmentResources(); })) {
        VOX_LOGE("render") << "init failed at createEnvironmentResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createShadowResources", [&] { return createShadowResources(); })) {
        VOX_LOGE("render") << "init failed at createShadowResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createVoxelGiResources", [&] { return createVoxelGiResources(); })) {
        VOX_LOGE("render") << "init failed at createVoxelGiResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createAutoExposureResources", [&] { return createAutoExposureResources(); })) {
        VOX_LOGE("render") << "init failed at createAutoExposureResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createSunShaftResources", [&] { return createSunShaftResources(); })) {
        VOX_LOGE("render") << "init failed at createSunShaftResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createSwapchain", [&] { return createSwapchain(); })) {
        VOX_LOGE("render") << "init failed at createSwapchain\n";
        shutdown();
        return false;
    }
    if (!runStep("createDescriptorResources", [&] { return createDescriptorResources(); })) {
        VOX_LOGE("render") << "init failed at createDescriptorResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createGraphicsPipeline", [&] { return createGraphicsPipeline(); })) {
        VOX_LOGE("render") << "init failed at createGraphicsPipeline\n";
        shutdown();
        return false;
    }
    if (!runStep("createPipePipeline", [&] { return createPipePipeline(); })) {
        VOX_LOGE("render") << "init failed at createPipePipeline\n";
        shutdown();
        return false;
    }
    if (!runStep("createAoPipelines", [&] { return createAoPipelines(); })) {
        VOX_LOGE("render") << "init failed at createAoPipelines\n";
        shutdown();
        return false;
    }
    {
        const auto frameArenaStart = Clock::now();
        m_frameArena.beginFrame(0);
        VOX_LOGI("render") << "init step frameArena.beginFrame(0) took " << elapsedMs(frameArenaStart) << " ms\n";
    }
    if (!runStep("createChunkBuffers", [&] { return createChunkBuffers(chunkGrid, {}); })) {
        VOX_LOGE("render") << "init failed at createChunkBuffers\n";
        shutdown();
        return false;
    }
    if (!runStep("createPipeBuffers", [&] { return createPipeBuffers(); })) {
        VOX_LOGE("render") << "init failed at createPipeBuffers\n";
        shutdown();
        return false;
    }
    if (!runStep("createPreviewBuffers", [&] { return createPreviewBuffers(); })) {
        VOX_LOGE("render") << "init failed at createPreviewBuffers\n";
        shutdown();
        return false;
    }
    if (!runStep("createFrameResources", [&] { return createFrameResources(); })) {
        VOX_LOGE("render") << "init failed at createFrameResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createGpuTimestampResources", [&] { return createGpuTimestampResources(); })) {
        VOX_LOGE("render") << "init failed at createGpuTimestampResources\n";
        shutdown();
        return false;
    }
    if (!runStep("createImGuiResources", [&] { return createImGuiResources(); })) {
        VOX_LOGE("render") << "init failed at createImGuiResources\n";
        shutdown();
        return false;
    }

    VOX_LOGI("render") << "init complete in " << elapsedMs(initStart) << " ms\n";
    return true;
}


bool Renderer::createInstance() {
#ifndef NDEBUG
    const bool enableValidationLayers = isLayerAvailable(kValidationLayers[0]);
#else
    const bool enableValidationLayers = false;
#endif
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions == nullptr || glfwExtensionCount == 0) {
        VOX_LOGI("render") << "no GLFW Vulkan instance extensions available\n";
        return false;
    }

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    m_debugUtilsEnabled = isInstanceExtensionAvailable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (m_debugUtilsEnabled) {
        appendInstanceExtensionIfMissing(extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    } else {
        VOX_LOGI("render") << "instance extension unavailable: " << VK_EXT_DEBUG_UTILS_EXTENSION_NAME << "\n";
    }
    VOX_LOGI("render") << "createInstance (validation="
              << (enableValidationLayers ? "on" : "off")
              << ", debugUtils=" << (m_debugUtilsEnabled ? "on" : "off")
              << ")\n";

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "voxel_factory_toy";
    applicationInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    applicationInfo.pEngineName = "none";
    applicationInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    applicationInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &applicationInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();
    }

    const VkResult result = vkCreateInstance(&createInfo, nullptr, &m_instance);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateInstance", result);
        return false;
    }
    return true;
}

bool Renderer::createSurface() {
    const VkResult result = glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface);
    if (result != VK_SUCCESS) {
        logVkFailure("glfwCreateWindowSurface", result);
        return false;
    }
    return true;
}


bool Renderer::pickPhysicalDevice() {
    m_supportsBindlessDescriptors = false;
    m_bindlessTextureCapacity = 0;
    m_supportsDisplayTiming = false;
    m_hasDisplayTimingExtension = false;

    struct CandidateSelection {
        VkPhysicalDevice device = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties properties{};
        uint32_t graphicsQueueFamilyIndex = 0;
        uint32_t graphicsQueueIndex = 0;
        uint32_t transferQueueFamilyIndex = 0;
        uint32_t transferQueueIndex = 0;
        bool supportsWireframe = false;
        bool supportsSamplerAnisotropy = false;
        bool supportsMultiDrawIndirect = false;
        bool supportsDrawIndirectFirstInstance = false;
        bool supportsDisplayTiming = false;
        bool hasDisplayTimingExtension = false;
        uint32_t bindlessTextureCapacity = 0;
        float maxSamplerAnisotropy = 1.0f;
        VkFormat depthFormat = VK_FORMAT_UNDEFINED;
        VkFormat shadowDepthFormat = VK_FORMAT_UNDEFINED;
        VkFormat hdrColorFormat = VK_FORMAT_UNDEFINED;
        VkFormat normalDepthFormat = VK_FORMAT_UNDEFINED;
        VkFormat ssaoFormat = VK_FORMAT_UNDEFINED;
    };
    auto scoreCandidate = [](const CandidateSelection& candidate) -> int {
        int score = 0;
        if (candidate.supportsDisplayTiming) {
            score += 8;
        }
        if (candidate.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 2;
        }
        if (candidate.supportsMultiDrawIndirect) {
            score += 1;
        }
        return score;
    };
    std::optional<CandidateSelection> bestCandidate;
    bool anyCandidateSupportsDisplayTiming = false;

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        VOX_LOGI("render") << "no Vulkan physical devices found\n";
        return false;
    }
    VOX_LOGI("render") << "physical devices found: " << deviceCount << "\n";

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        VOX_LOGI("render") << "evaluating GPU: " << properties.deviceName
                  << ", apiVersion=" << VK_VERSION_MAJOR(properties.apiVersion) << "."
                  << VK_VERSION_MINOR(properties.apiVersion) << "."
                  << VK_VERSION_PATCH(properties.apiVersion) << "\n";
        if (properties.apiVersion < VK_API_VERSION_1_3) {
            VOX_LOGI("render") << "skip GPU: Vulkan 1.3 required\n";
            continue;
        }
        if ((properties.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0) {
            VOX_LOGI("render") << "skip GPU: 4x MSAA color attachments not supported\n";
            continue;
        }
        if ((properties.limits.framebufferDepthSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0) {
            VOX_LOGI("render") << "skip GPU: 4x MSAA depth attachments not supported\n";
            continue;
        }

        const QueueFamilyChoice queueFamily = findQueueFamily(candidate, m_surface);
        if (!queueFamily.valid()) {
            VOX_LOGI("render") << "skip GPU: missing graphics/present/transfer queue support\n";
            continue;
        }
        if (!hasRequiredDeviceExtensions(candidate)) {
            VOX_LOGI("render") << "skip GPU: missing required device extensions\n";
            continue;
        }

        const SwapchainSupport swapchainSupport = querySwapchainSupport(candidate, m_surface);
        if (swapchainSupport.formats.empty() || swapchainSupport.presentModes.empty()) {
            VOX_LOGI("render") << "skip GPU: swapchain support incomplete\n";
            continue;
        }
        const VkFormat depthFormat = findSupportedDepthFormat(candidate);
        if (depthFormat == VK_FORMAT_UNDEFINED) {
            VOX_LOGI("render") << "skip GPU: no supported depth format\n";
            continue;
        }
        const VkFormat shadowDepthFormat = findSupportedShadowDepthFormat(candidate);
        if (shadowDepthFormat == VK_FORMAT_UNDEFINED) {
            VOX_LOGI("render") << "skip GPU: no supported shadow depth format\n";
            continue;
        }
        const VkFormat hdrColorFormat = findSupportedHdrColorFormat(candidate);
        if (hdrColorFormat == VK_FORMAT_UNDEFINED) {
            VOX_LOGI("render") << "skip GPU: no supported HDR color format\n";
            continue;
        }
        const VkFormat normalDepthFormat = findSupportedNormalDepthFormat(candidate);
        if (normalDepthFormat == VK_FORMAT_UNDEFINED) {
            VOX_LOGI("render") << "skip GPU: no supported normal-depth color format\n";
            continue;
        }
        const VkFormat ssaoFormat = findSupportedSsaoFormat(candidate);
        if (ssaoFormat == VK_FORMAT_UNDEFINED) {
            VOX_LOGI("render") << "skip GPU: no supported SSAO format\n";
            continue;
        }

        VkPhysicalDeviceVulkan11Features vulkan11Features{};
        vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12Features.pNext = &vulkan11Features;
        VkPhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13Features.pNext = &vulkan12Features;
        VkPhysicalDeviceMemoryPriorityFeaturesEXT memoryPriorityFeatures{};
        memoryPriorityFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
        memoryPriorityFeatures.pNext = &vulkan13Features;
        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &memoryPriorityFeatures;
        vkGetPhysicalDeviceFeatures2(candidate, &features2);
        if (vulkan13Features.dynamicRendering != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: dynamicRendering not supported\n";
            continue;
        }
        if (vulkan12Features.timelineSemaphore != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: timelineSemaphore not supported\n";
            continue;
        }
        if (vulkan13Features.synchronization2 != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: synchronization2 not supported\n";
            continue;
        }
        if (vulkan13Features.maintenance4 != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: maintenance4 not supported\n";
            continue;
        }
        if (vulkan12Features.bufferDeviceAddress != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: bufferDeviceAddress not supported\n";
            continue;
        }
        if (memoryPriorityFeatures.memoryPriority != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: memoryPriority not supported\n";
            continue;
        }
        if (features2.features.drawIndirectFirstInstance != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: drawIndirectFirstInstance not supported\n";
            continue;
        }
        if (vulkan11Features.shaderDrawParameters != VK_TRUE) {
            VOX_LOGI("render") << "skip GPU: shaderDrawParameters not supported\n";
            continue;
        }
        const bool supportsBindlessDescriptors =
            vulkan12Features.descriptorIndexing == VK_TRUE &&
            vulkan12Features.runtimeDescriptorArray == VK_TRUE &&
            vulkan12Features.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
            vulkan12Features.descriptorBindingPartiallyBound == VK_TRUE;
        if (!supportsBindlessDescriptors) {
            VOX_LOGI("render") << "skip GPU: bindless descriptor indexing not supported\n";
            continue;
        }

        uint32_t bindlessTextureCapacity = 0;
        const uint32_t perStageSamplerLimit = properties.limits.maxPerStageDescriptorSamplers;
        const uint32_t perStageSampledLimit = properties.limits.maxPerStageDescriptorSampledImages;
        const uint32_t descriptorSetSampledLimit = properties.limits.maxDescriptorSetSampledImages;
        uint32_t safeBudget = std::min({perStageSamplerLimit, perStageSampledLimit, descriptorSetSampledLimit});
        if (safeBudget > kBindlessReservedSampledDescriptors) {
            safeBudget -= kBindlessReservedSampledDescriptors;
        } else {
            safeBudget = 0;
        }
        bindlessTextureCapacity = std::min(kBindlessTargetTextureCapacity, safeBudget);
        if (bindlessTextureCapacity < kBindlessMinTextureCapacity) {
            VOX_LOGI("render") << "skip GPU: bindless descriptor budget too small\n";
            continue;
        }

        const bool displayTimingExtensionAvailable =
            isDeviceExtensionAvailable(candidate, VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME);
        const bool supportsDisplayTiming = displayTimingExtensionAvailable;

        CandidateSelection candidateSelection{};
        candidateSelection.device = candidate;
        candidateSelection.properties = properties;
        candidateSelection.graphicsQueueFamilyIndex = queueFamily.graphicsAndPresent.value();
        candidateSelection.graphicsQueueIndex = queueFamily.graphicsQueueIndex;
        candidateSelection.transferQueueFamilyIndex = queueFamily.transfer.value();
        candidateSelection.transferQueueIndex = queueFamily.transferQueueIndex;
        candidateSelection.supportsWireframe = features2.features.fillModeNonSolid == VK_TRUE;
        candidateSelection.supportsSamplerAnisotropy = features2.features.samplerAnisotropy == VK_TRUE;
        candidateSelection.supportsDrawIndirectFirstInstance = features2.features.drawIndirectFirstInstance == VK_TRUE;
        candidateSelection.supportsMultiDrawIndirect = features2.features.multiDrawIndirect == VK_TRUE;
        candidateSelection.supportsDisplayTiming = supportsDisplayTiming;
        candidateSelection.hasDisplayTimingExtension = displayTimingExtensionAvailable;
        if (supportsDisplayTiming) {
            anyCandidateSupportsDisplayTiming = true;
        }
        candidateSelection.bindlessTextureCapacity = bindlessTextureCapacity;
        candidateSelection.maxSamplerAnisotropy = properties.limits.maxSamplerAnisotropy;
        candidateSelection.depthFormat = depthFormat;
        candidateSelection.shadowDepthFormat = shadowDepthFormat;
        candidateSelection.hdrColorFormat = hdrColorFormat;
        candidateSelection.normalDepthFormat = normalDepthFormat;
        candidateSelection.ssaoFormat = ssaoFormat;

        VOX_LOGI("render") << "candidate presentation timing: gpu=" << properties.deviceName
                           << ", displayTimingSupport=" << (candidateSelection.supportsDisplayTiming ? "yes" : "no")
                           << "(ext=" << (candidateSelection.hasDisplayTimingExtension ? "yes" : "no") << ")\n";

        if (!bestCandidate.has_value() ||
            scoreCandidate(candidateSelection) > scoreCandidate(bestCandidate.value())) {
            bestCandidate = candidateSelection;
        }
    }

    if (bestCandidate.has_value()) {
        const CandidateSelection& selected = bestCandidate.value();
        m_physicalDevice = selected.device;
        m_graphicsQueueFamilyIndex = selected.graphicsQueueFamilyIndex;
        m_graphicsQueueIndex = selected.graphicsQueueIndex;
        m_transferQueueFamilyIndex = selected.transferQueueFamilyIndex;
        m_transferQueueIndex = selected.transferQueueIndex;
        m_supportsWireframePreview = selected.supportsWireframe;
        m_supportsSamplerAnisotropy = selected.supportsSamplerAnisotropy;
        m_supportsMultiDrawIndirect = selected.supportsMultiDrawIndirect;
        m_supportsBindlessDescriptors = true;
        m_supportsDisplayTiming = selected.supportsDisplayTiming;
        m_hasDisplayTimingExtension = selected.hasDisplayTimingExtension;
        m_enableDisplayTiming = m_supportsDisplayTiming;
        m_bindlessTextureCapacity = selected.bindlessTextureCapacity;
        m_maxSamplerAnisotropy = selected.maxSamplerAnisotropy;
        m_depthFormat = selected.depthFormat;
        m_shadowDepthFormat = selected.shadowDepthFormat;
        m_hdrColorFormat = selected.hdrColorFormat;
        m_normalDepthFormat = selected.normalDepthFormat;
        m_ssaoFormat = selected.ssaoFormat;
        m_colorSampleCount = VK_SAMPLE_COUNT_4_BIT;

        VOX_LOGI("render") << "selected GPU: " << selected.properties.deviceName
                           << ", graphicsQueueFamily=" << m_graphicsQueueFamilyIndex
                           << ", graphicsQueueIndex=" << m_graphicsQueueIndex
                           << ", transferQueueFamily=" << m_transferQueueFamilyIndex
                           << ", transferQueueIndex=" << m_transferQueueIndex
                           << ", wireframePreview=" << (m_supportsWireframePreview ? "yes" : "no")
                           << ", samplerAnisotropy=" << (m_supportsSamplerAnisotropy ? "yes" : "no")
                           << ", drawIndirectFirstInstance="
                           << (selected.supportsDrawIndirectFirstInstance ? "yes" : "no")
                           << ", multiDrawIndirect=" << (m_supportsMultiDrawIndirect ? "yes" : "no")
                           << ", bindlessDescriptors=" << (m_supportsBindlessDescriptors ? "yes" : "no")
                           << ", bindlessTextureCapacity=" << m_bindlessTextureCapacity
                           << ", displayTiming=" << (m_supportsDisplayTiming ? "yes" : "no")
                           << "(ext=" << (selected.hasDisplayTimingExtension ? "yes" : "no") << ")"
                           << ", maxSamplerAnisotropy=" << m_maxSamplerAnisotropy
                           << ", msaaSamples=" << static_cast<uint32_t>(m_colorSampleCount)
                           << ", shadowDepthFormat=" << static_cast<int>(m_shadowDepthFormat)
                           << ", hdrColorFormat=" << static_cast<int>(m_hdrColorFormat)
                           << ", normalDepthFormat=" << static_cast<int>(m_normalDepthFormat)
                           << ", ssaoFormat=" << static_cast<int>(m_ssaoFormat)
                           << "\n";
        if (!anyCandidateSupportsDisplayTiming) {
            VOX_LOGI("render")
                << "display timing unavailable: no enumerated physical device exposes "
                << VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME << "\n";
        }
        return true;
    }

    VOX_LOGI("render") << "no suitable GPU found\n";
    return false;
}


bool Renderer::createLogicalDevice() {
    const bool sameFamily = (m_graphicsQueueFamilyIndex == m_transferQueueFamilyIndex);
    std::array<VkDeviceQueueCreateInfo, 2> queueCreateInfos{};
    uint32_t queueCreateInfoCount = 0;
    std::array<float, 2> sharedFamilyPriorities = {1.0f, 1.0f};
    float graphicsQueuePriority = 1.0f;
    float transferQueuePriority = 1.0f;

    if (sameFamily) {
        const uint32_t queueCount = std::max(m_graphicsQueueIndex, m_transferQueueIndex) + 1;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
        queueCreateInfo.queueCount = queueCount;
        queueCreateInfo.pQueuePriorities = sharedFamilyPriorities.data();
        queueCreateInfos[queueCreateInfoCount++] = queueCreateInfo;
    } else {
        VkDeviceQueueCreateInfo graphicsQueueCreateInfo{};
        graphicsQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        graphicsQueueCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
        graphicsQueueCreateInfo.queueCount = m_graphicsQueueIndex + 1;
        graphicsQueueCreateInfo.pQueuePriorities = &graphicsQueuePriority;
        queueCreateInfos[queueCreateInfoCount++] = graphicsQueueCreateInfo;

        VkDeviceQueueCreateInfo transferQueueCreateInfo{};
        transferQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        transferQueueCreateInfo.queueFamilyIndex = m_transferQueueFamilyIndex;
        transferQueueCreateInfo.queueCount = m_transferQueueIndex + 1;
        transferQueueCreateInfo.pQueuePriorities = &transferQueuePriority;
        queueCreateInfos[queueCreateInfoCount++] = transferQueueCreateInfo;
    }

    VkPhysicalDeviceFeatures2 enabledFeatures2{};
    enabledFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enabledFeatures2.features.fillModeNonSolid = m_supportsWireframePreview ? VK_TRUE : VK_FALSE;
    enabledFeatures2.features.samplerAnisotropy = m_supportsSamplerAnisotropy ? VK_TRUE : VK_FALSE;
    enabledFeatures2.features.multiDrawIndirect = m_supportsMultiDrawIndirect ? VK_TRUE : VK_FALSE;
    enabledFeatures2.features.drawIndirectFirstInstance = VK_TRUE;

    VkPhysicalDeviceVulkan11Features vulkan11Features{};
    vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan11Features.shaderDrawParameters = VK_TRUE;

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.pNext = &vulkan11Features;
    vulkan12Features.timelineSemaphore = VK_TRUE;
    vulkan12Features.bufferDeviceAddress = VK_TRUE;
    if (m_supportsBindlessDescriptors) {
        vulkan12Features.descriptorIndexing = VK_TRUE;
        vulkan12Features.runtimeDescriptorArray = VK_TRUE;
        vulkan12Features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        vulkan12Features.descriptorBindingPartiallyBound = VK_TRUE;
    }

    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &vulkan12Features;
    vulkan13Features.dynamicRendering = VK_TRUE;
    vulkan13Features.synchronization2 = VK_TRUE;
    vulkan13Features.maintenance4 = VK_TRUE;

    VkPhysicalDeviceMemoryPriorityFeaturesEXT memoryPriorityFeatures{};
    memoryPriorityFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
    memoryPriorityFeatures.pNext = &vulkan13Features;
    memoryPriorityFeatures.memoryPriority = VK_TRUE;
    enabledFeatures2.pNext = &memoryPriorityFeatures;

    std::vector<const char*> enabledDeviceExtensions(kDeviceExtensions.begin(), kDeviceExtensions.end());
    if (m_supportsDisplayTiming && m_hasDisplayTimingExtension) {
        appendDeviceExtensionIfMissing(enabledDeviceExtensions, VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME);
    }

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &enabledFeatures2;
    createInfo.queueCreateInfoCount = queueCreateInfoCount;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = nullptr;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledDeviceExtensions.data();

    const VkResult result = vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateDevice", result);
        return false;
    }
    VOX_LOGI("render") << "device features enabled: dynamicRendering=1, synchronization2=1, maintenance4=1, "
        << "timelineSemaphore=1, bufferDeviceAddress=1, memoryPriority=1, shaderDrawParameters=1, drawIndirectFirstInstance=1, "
        << "multiDrawIndirect=" << (m_supportsMultiDrawIndirect ? 1 : 0)
        << ", descriptorIndexing=" << (m_supportsBindlessDescriptors ? 1 : 0)
        << ", runtimeDescriptorArray=" << (m_supportsBindlessDescriptors ? 1 : 0)
        << ", sampledImageArrayNonUniformIndexing=" << (m_supportsBindlessDescriptors ? 1 : 0)
        << ", descriptorBindingPartiallyBound=" << (m_supportsBindlessDescriptors ? 1 : 0)
        << ", displayTiming=" << (m_supportsDisplayTiming ? 1 : 0)
        << "\n";
    {
        std::string extensionLog;
        for (std::size_t i = 0; i < enabledDeviceExtensions.size(); ++i) {
            if (i > 0) {
                extensionLog += ", ";
            }
            extensionLog += enabledDeviceExtensions[i];
        }
        VOX_LOGI("render") << "device extensions enabled: " << extensionLog << "\n";
    }
    if (m_supportsBindlessDescriptors) {
        VOX_LOGI("render") << "bindless descriptor support enabled (capacity="
            << m_bindlessTextureCapacity << ")\n";
    } else {
        VOX_LOGI("render") << "bindless descriptor support disabled (missing descriptor-indexing features)\n";
    }

    vkGetDeviceQueue(m_device, m_graphicsQueueFamilyIndex, m_graphicsQueueIndex, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, m_transferQueueFamilyIndex, m_transferQueueIndex, &m_transferQueue);
    m_getRefreshCycleDurationGoogle = reinterpret_cast<PFN_vkGetRefreshCycleDurationGOOGLE>(
        vkGetDeviceProcAddr(m_device, "vkGetRefreshCycleDurationGOOGLE")
    );
    m_getPastPresentationTimingGoogle = reinterpret_cast<PFN_vkGetPastPresentationTimingGOOGLE>(
        vkGetDeviceProcAddr(m_device, "vkGetPastPresentationTimingGOOGLE")
    );
    if (m_supportsDisplayTiming &&
        (m_getRefreshCycleDurationGoogle == nullptr || m_getPastPresentationTimingGoogle == nullptr)) {
        VOX_LOGI("render") << "display_timing extension enabled but function pointers were not loaded; disabling display timing\n";
        m_supportsDisplayTiming = false;
        m_enableDisplayTiming = false;
    }
    VOX_LOGI("render") << "present runtime: displayTimingSupport=" << (m_supportsDisplayTiming ? "yes" : "no")
        << ", displayTimingExtension=" << (m_hasDisplayTimingExtension ? "yes" : "no")
        << ", displayTimingEnabled=" << (m_enableDisplayTiming ? "yes" : "no")
        << "\n";
    loadDebugUtilsFunctions();
    setObjectName(VK_OBJECT_TYPE_DEVICE, vkHandleToUint64(m_device), "renderer.device");
    setObjectName(VK_OBJECT_TYPE_QUEUE, vkHandleToUint64(m_graphicsQueue), "renderer.queue.graphics");
    setObjectName(VK_OBJECT_TYPE_QUEUE, vkHandleToUint64(m_transferQueue), "renderer.queue.transfer");

    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(m_physicalDevice, &deviceProperties);
    m_uniformBufferAlignment = std::max<VkDeviceSize>(
        deviceProperties.limits.minUniformBufferOffsetAlignment,
        static_cast<VkDeviceSize>(16)
    );
    m_gpuTimestampPeriodNs = deviceProperties.limits.timestampPeriod;
    std::uint32_t queueFamilyPropertyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyPropertyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(
        m_physicalDevice,
        &queueFamilyPropertyCount,
        queueFamilyProperties.data()
    );
    const bool graphicsQueueHasTimestamps =
        m_graphicsQueueFamilyIndex < queueFamilyProperties.size() &&
        queueFamilyProperties[m_graphicsQueueFamilyIndex].timestampValidBits > 0;
    m_gpuTimestampsSupported = graphicsQueueHasTimestamps && m_gpuTimestampPeriodNs > 0.0f;
    VOX_LOGI("render") << "GPU timestamps: supported=" << (m_gpuTimestampsSupported ? "yes" : "no")
        << ", periodNs=" << m_gpuTimestampPeriodNs
        << ", graphicsTimestampBits="
        << (graphicsQueueHasTimestamps
                ? queueFamilyProperties[m_graphicsQueueFamilyIndex].timestampValidBits
                : 0u)
        << "\n";

    if (m_vmaAllocator == VK_NULL_HANDLE) {
        VmaAllocatorCreateInfo allocatorCreateInfo{};
        allocatorCreateInfo.flags =
            VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
            VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT |
            VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
        allocatorCreateInfo.physicalDevice = m_physicalDevice;
        allocatorCreateInfo.device = m_device;
        allocatorCreateInfo.instance = m_instance;
        allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        const VkResult allocatorResult = vmaCreateAllocator(&allocatorCreateInfo, &m_vmaAllocator);
        if (allocatorResult != VK_SUCCESS) {
            logVkFailure("vmaCreateAllocator", allocatorResult);
            return false;
        }
        VOX_LOGI("render") << "VMA allocator created: flags="
            << "BUFFER_DEVICE_ADDRESS|EXT_MEMORY_BUDGET|EXT_MEMORY_PRIORITY\n";
    }
    return true;
}


void Renderer::loadDebugUtilsFunctions() {
    m_setDebugUtilsObjectName = nullptr;
    m_cmdBeginDebugUtilsLabel = nullptr;
    m_cmdEndDebugUtilsLabel = nullptr;
    m_cmdInsertDebugUtilsLabel = nullptr;

    if (!m_debugUtilsEnabled || m_device == VK_NULL_HANDLE) {
        return;
    }

    m_setDebugUtilsObjectName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(m_device, "vkSetDebugUtilsObjectNameEXT")
    );
    m_cmdBeginDebugUtilsLabel = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(m_device, "vkCmdBeginDebugUtilsLabelEXT")
    );
    m_cmdEndDebugUtilsLabel = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(m_device, "vkCmdEndDebugUtilsLabelEXT")
    );
    m_cmdInsertDebugUtilsLabel = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(m_device, "vkCmdInsertDebugUtilsLabelEXT")
    );

    const bool namesReady = m_setDebugUtilsObjectName != nullptr;
    const bool labelsReady = m_cmdBeginDebugUtilsLabel != nullptr && m_cmdEndDebugUtilsLabel != nullptr;
    if (!namesReady && !labelsReady) {
        VOX_LOGI("render") << "debug utils extension enabled but debug functions were not loaded\n";
        m_debugUtilsEnabled = false;
        return;
    }

    VOX_LOGI("render") << "debug utils loaded: objectNames=" << (namesReady ? "yes" : "no")
        << ", cmdLabels=" << (labelsReady ? "yes" : "no")
        << ", cmdInsertLabel=" << (m_cmdInsertDebugUtilsLabel != nullptr ? "yes" : "no")
        << "\n";
}


bool Renderer::createTimelineSemaphore() {
    if (m_renderTimelineSemaphore != VK_NULL_HANDLE) {
        return true;
    }

    VkSemaphoreTypeCreateInfo timelineCreateInfo{};
    timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCreateInfo.initialValue = 0;

    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = &timelineCreateInfo;

    const VkResult result = vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_renderTimelineSemaphore);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateSemaphore(timeline)", result);
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_SEMAPHORE,
        vkHandleToUint64(m_renderTimelineSemaphore),
        "renderer.timeline.render"
    );

    m_frameTimelineValues.fill(0);
    m_pendingTransferTimelineValue = 0;
    m_currentChunkReadyTimelineValue = 0;
    m_transferCommandBufferInFlightValue = 0;
    m_lastGraphicsTimelineValue = 0;
    m_nextTimelineValue = 1;
    return true;
}


bool Renderer::createUploadRingBuffer() {
    // FrameArena layer A foundation: one persistently mapped upload arena per frame-in-flight.
    FrameArenaConfig config{};
    config.uploadBytesPerFrame = 1024ull * 1024ull * 64ull;
    config.frameCount = kMaxFramesInFlight;
    config.uploadUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    const bool ok = m_frameArena.init(
        &m_bufferAllocator,
        m_physicalDevice,
        m_device,
        config
        , m_vmaAllocator
    );
    if (!ok) {
        VOX_LOGE("render") << "frame arena init failed\n";
    } else {
        const BufferHandle uploadHandle = m_frameArena.uploadBufferHandle();
        if (uploadHandle != kInvalidBufferHandle) {
            const VkBuffer uploadBuffer = m_bufferAllocator.getBuffer(uploadHandle);
            if (uploadBuffer != VK_NULL_HANDLE) {
                setObjectName(
                    VK_OBJECT_TYPE_BUFFER,
                    vkHandleToUint64(uploadBuffer),
                    "framearena.uploadRing"
                );
            }
        }
    }
    return ok;
}


bool Renderer::createTransferResources() {
    if (m_transferCommandPool != VK_NULL_HANDLE && m_transferCommandBuffer != VK_NULL_HANDLE) {
        return true;
    }

    VkCommandPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolCreateInfo.queueFamilyIndex = m_transferQueueFamilyIndex;

    const VkResult poolResult = vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &m_transferCommandPool);
    if (poolResult != VK_SUCCESS) {
        logVkFailure("vkCreateCommandPool(transfer)", poolResult);
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_COMMAND_POOL,
        vkHandleToUint64(m_transferCommandPool),
        "renderer.transfer.commandPool"
    );

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = m_transferCommandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    const VkResult commandBufferResult = vkAllocateCommandBuffers(m_device, &allocateInfo, &m_transferCommandBuffer);
    if (commandBufferResult != VK_SUCCESS) {
        logVkFailure("vkAllocateCommandBuffers(transfer)", commandBufferResult);
        vkDestroyCommandPool(m_device, m_transferCommandPool, nullptr);
        m_transferCommandPool = VK_NULL_HANDLE;
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_COMMAND_BUFFER,
        vkHandleToUint64(m_transferCommandBuffer),
        "renderer.transfer.commandBuffer"
    );

    return true;
}


bool Renderer::createPipeBuffers() {
    if (m_pipeVertexBufferHandle != kInvalidBufferHandle &&
        m_pipeIndexBufferHandle != kInvalidBufferHandle &&
        m_transportVertexBufferHandle != kInvalidBufferHandle &&
        m_transportIndexBufferHandle != kInvalidBufferHandle &&
        m_grassBillboardVertexBufferHandle != kInvalidBufferHandle &&
        m_grassBillboardIndexBufferHandle != kInvalidBufferHandle) {
        return true;
    }

    const PipeMeshData pipeMesh = buildPipeCylinderMesh();
    const PipeMeshData transportMesh = buildTransportBoxMesh();
    if (pipeMesh.vertices.empty() || pipeMesh.indices.empty()) {
        VOX_LOGE("render") << "pipe cylinder mesh build failed\n";
        return false;
    }
    if (transportMesh.vertices.empty() || transportMesh.indices.empty()) {
        VOX_LOGE("render") << "transport box mesh build failed\n";
        return false;
    }

    auto createMeshBuffers = [&](const PipeMeshData& mesh, BufferHandle& outVertex, BufferHandle& outIndex, const char* label) -> bool {
        if (outVertex != kInvalidBufferHandle || outIndex != kInvalidBufferHandle) {
            return true;
        }
        BufferCreateDesc vertexCreateDesc{};
        vertexCreateDesc.size = static_cast<VkDeviceSize>(mesh.vertices.size() * sizeof(PipeMeshData::Vertex));
        vertexCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        vertexCreateDesc.initialData = mesh.vertices.data();
        outVertex = m_bufferAllocator.createBuffer(vertexCreateDesc);
        if (outVertex == kInvalidBufferHandle) {
            VOX_LOGE("render") << label << " vertex buffer allocation failed\n";
            return false;
        }
        const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(outVertex);
        if (vertexBuffer != VK_NULL_HANDLE) {
            const std::string vertexName = std::string("mesh.") + label + ".vertex";
            setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(vertexBuffer), vertexName.c_str());
        }

        BufferCreateDesc indexCreateDesc{};
        indexCreateDesc.size = static_cast<VkDeviceSize>(mesh.indices.size() * sizeof(std::uint32_t));
        indexCreateDesc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        indexCreateDesc.initialData = mesh.indices.data();
        outIndex = m_bufferAllocator.createBuffer(indexCreateDesc);
        if (outIndex == kInvalidBufferHandle) {
            VOX_LOGE("render") << label << " index buffer allocation failed\n";
            m_bufferAllocator.destroyBuffer(outVertex);
            outVertex = kInvalidBufferHandle;
            return false;
        }
        const VkBuffer indexBuffer = m_bufferAllocator.getBuffer(outIndex);
        if (indexBuffer != VK_NULL_HANDLE) {
            const std::string indexName = std::string("mesh.") + label + ".index";
            setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(indexBuffer), indexName.c_str());
        }
        return true;
    };

    if (!createMeshBuffers(pipeMesh, m_pipeVertexBufferHandle, m_pipeIndexBufferHandle, "pipe")) {
        return false;
    }
    if (!createMeshBuffers(
            transportMesh,
            m_transportVertexBufferHandle,
            m_transportIndexBufferHandle,
            "transport"
        )) {
        VOX_LOGE("render") << "transport mesh buffer setup failed\n";
        return false;
    }

    if (m_grassBillboardVertexBufferHandle == kInvalidBufferHandle ||
        m_grassBillboardIndexBufferHandle == kInvalidBufferHandle) {
        constexpr std::array<GrassBillboardVertex, 8> kGrassBillboardVertices = {{
            // Plane 0 (X axis).
            GrassBillboardVertex{{-0.38f, 0.0f}, {0.0f, 1.0f}, 0.0f},
            GrassBillboardVertex{{ 0.38f, 0.0f}, {1.0f, 1.0f}, 0.0f},
            GrassBillboardVertex{{-0.38f, 0.88f}, {0.0f, 0.0f}, 0.0f},
            GrassBillboardVertex{{ 0.38f, 0.88f}, {1.0f, 0.0f}, 0.0f},
            // Plane 1 (Z axis).
            GrassBillboardVertex{{-0.38f, 0.0f}, {0.0f, 1.0f}, 1.0f},
            GrassBillboardVertex{{ 0.38f, 0.0f}, {1.0f, 1.0f}, 1.0f},
            GrassBillboardVertex{{-0.38f, 0.88f}, {0.0f, 0.0f}, 1.0f},
            GrassBillboardVertex{{ 0.38f, 0.88f}, {1.0f, 0.0f}, 1.0f},
        }};
        constexpr std::array<std::uint32_t, 12> kGrassBillboardIndices = {{
            0u, 1u, 2u, 2u, 1u, 3u,
            4u, 5u, 6u, 6u, 5u, 7u
        }};

        BufferCreateDesc grassVertexCreateDesc{};
        grassVertexCreateDesc.size = static_cast<VkDeviceSize>(kGrassBillboardVertices.size() * sizeof(GrassBillboardVertex));
        grassVertexCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        grassVertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        grassVertexCreateDesc.initialData = kGrassBillboardVertices.data();
        m_grassBillboardVertexBufferHandle = m_bufferAllocator.createBuffer(grassVertexCreateDesc);
        if (m_grassBillboardVertexBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "grass billboard vertex buffer allocation failed\n";
            return false;
        }
        {
            const VkBuffer grassVertexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardVertexBufferHandle);
            if (grassVertexBuffer != VK_NULL_HANDLE) {
                setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(grassVertexBuffer), "mesh.grassBillboard.vertex");
            }
        }

        BufferCreateDesc grassIndexCreateDesc{};
        grassIndexCreateDesc.size = static_cast<VkDeviceSize>(kGrassBillboardIndices.size() * sizeof(std::uint32_t));
        grassIndexCreateDesc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        grassIndexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        grassIndexCreateDesc.initialData = kGrassBillboardIndices.data();
        m_grassBillboardIndexBufferHandle = m_bufferAllocator.createBuffer(grassIndexCreateDesc);
        if (m_grassBillboardIndexBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "grass billboard index buffer allocation failed\n";
            m_bufferAllocator.destroyBuffer(m_grassBillboardVertexBufferHandle);
            m_grassBillboardVertexBufferHandle = kInvalidBufferHandle;
            return false;
        }
        {
            const VkBuffer grassIndexBuffer = m_bufferAllocator.getBuffer(m_grassBillboardIndexBufferHandle);
            if (grassIndexBuffer != VK_NULL_HANDLE) {
                setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(grassIndexBuffer), "mesh.grassBillboard.index");
            }
        }
        m_grassBillboardIndexCount = static_cast<uint32_t>(kGrassBillboardIndices.size());
    }

    m_pipeIndexCount = static_cast<uint32_t>(pipeMesh.indices.size());
    m_transportIndexCount = static_cast<uint32_t>(transportMesh.indices.size());
    return true;
}


bool Renderer::createPreviewBuffers() {
    if (m_previewVertexBufferHandle != kInvalidBufferHandle && m_previewIndexBufferHandle != kInvalidBufferHandle) {
        return true;
    }

    const world::ChunkMeshData addMesh = buildSingleVoxelPreviewMesh(0, 0, 0, 3, 250);
    const world::ChunkMeshData removeMesh = buildSingleVoxelPreviewMesh(0, 0, 0, 3, 251);
    if (addMesh.vertices.empty() || addMesh.indices.empty() || removeMesh.vertices.empty() || removeMesh.indices.empty()) {
        VOX_LOGE("render") << "preview mesh build failed\n";
        return false;
    }

    world::ChunkMeshData mesh{};
    mesh.vertices = addMesh.vertices;
    mesh.indices = addMesh.indices;
    mesh.vertices.insert(mesh.vertices.end(), removeMesh.vertices.begin(), removeMesh.vertices.end());
    mesh.indices.reserve(mesh.indices.size() + removeMesh.indices.size());
    const uint32_t removeBaseVertex = static_cast<uint32_t>(addMesh.vertices.size());
    for (const uint32_t index : removeMesh.indices) {
        mesh.indices.push_back(index + removeBaseVertex);
    }

    BufferCreateDesc vertexCreateDesc{};
    vertexCreateDesc.size = static_cast<VkDeviceSize>(mesh.vertices.size() * sizeof(world::PackedVoxelVertex));
    vertexCreateDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    vertexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vertexCreateDesc.initialData = mesh.vertices.data();
    m_previewVertexBufferHandle = m_bufferAllocator.createBuffer(vertexCreateDesc);
    if (m_previewVertexBufferHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "preview vertex buffer allocation failed\n";
        return false;
    }
    {
        const VkBuffer previewVertexBuffer = m_bufferAllocator.getBuffer(m_previewVertexBufferHandle);
        if (previewVertexBuffer != VK_NULL_HANDLE) {
            setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(previewVertexBuffer), "preview.voxel.vertex");
        }
    }

    BufferCreateDesc indexCreateDesc{};
    indexCreateDesc.size = static_cast<VkDeviceSize>(mesh.indices.size() * sizeof(std::uint32_t));
    indexCreateDesc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    indexCreateDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    indexCreateDesc.initialData = mesh.indices.data();
    m_previewIndexBufferHandle = m_bufferAllocator.createBuffer(indexCreateDesc);
    if (m_previewIndexBufferHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "preview index buffer allocation failed\n";
        m_bufferAllocator.destroyBuffer(m_previewVertexBufferHandle);
        m_previewVertexBufferHandle = kInvalidBufferHandle;
        return false;
    }
    {
        const VkBuffer previewIndexBuffer = m_bufferAllocator.getBuffer(m_previewIndexBufferHandle);
        if (previewIndexBuffer != VK_NULL_HANDLE) {
            setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(previewIndexBuffer), "preview.voxel.index");
        }
    }

    m_previewIndexCount = static_cast<uint32_t>(mesh.indices.size());
    return true;
}


bool Renderer::createEnvironmentResources() {
    if (!createDiffuseTextureResources()) {
        VOX_LOGE("render") << "diffuse texture creation failed\n";
        return false;
    }
    VOX_LOGI("render") << "environment uses procedural sky + SH irradiance + diffuse albedo texture\n";
    return true;
}


bool Renderer::createDiffuseTextureResources() {
    bool hasDiffuseAllocation = (m_diffuseTextureMemory != VK_NULL_HANDLE);
    if (m_vmaAllocator != VK_NULL_HANDLE) {
        hasDiffuseAllocation = (m_diffuseTextureAllocation != VK_NULL_HANDLE);
    }
    if (
        m_diffuseTextureImage != VK_NULL_HANDLE &&
        hasDiffuseAllocation &&
        m_diffuseTextureImageView != VK_NULL_HANDLE &&
        m_diffuseTextureSampler != VK_NULL_HANDLE &&
        m_diffuseTexturePlantSampler != VK_NULL_HANDLE
    ) {
        return true;
    }

    constexpr uint32_t kTileSize = 16;
    constexpr uint32_t kTextureTilesX = 9;
    constexpr uint32_t kTextureTilesY = 1;
    constexpr uint32_t kTextureWidth = kTileSize * kTextureTilesX;
    constexpr uint32_t kTextureHeight = kTileSize * kTextureTilesY;
    constexpr VkFormat kTextureFormat = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t diffuseMipLevels = 1u;
    for (uint32_t tileExtent = kTileSize; tileExtent > 1u; tileExtent >>= 1u) {
        ++diffuseMipLevels;
    }
    constexpr VkDeviceSize kTextureBytes = kTextureWidth * kTextureHeight * 4;

    std::vector<std::uint8_t> pixels(static_cast<size_t>(kTextureBytes), 0);
    auto hash8 = [](uint32_t x, uint32_t y, uint32_t seed) -> std::uint8_t {
        uint32_t h = x * 374761393u;
        h += y * 668265263u;
        h += seed * 2246822519u;
        h = (h ^ (h >> 13u)) * 1274126177u;
        return static_cast<std::uint8_t>((h >> 24u) & 0xFFu);
    };
    auto writePixel = [&](uint32_t px, uint32_t py, std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a = 255u) {
        const size_t i = static_cast<size_t>((py * kTextureWidth + px) * 4u);
        pixels[i + 0] = r;
        pixels[i + 1] = g;
        pixels[i + 2] = b;
        pixels[i + 3] = a;
    };

    for (uint32_t y = 0; y < kTextureHeight; ++y) {
        for (uint32_t x = 0; x < kTextureWidth; ++x) {
            const uint32_t tileIndex = x / kTileSize;
            const uint32_t localX = x % kTileSize;
            const uint32_t localY = y % kTileSize;
            const std::uint8_t noiseA = hash8(localX, localY, tileIndex + 11u);
            const std::uint8_t noiseB = hash8(localX, localY, tileIndex + 37u);

            std::uint8_t r = 128u;
            std::uint8_t g = 128u;
            std::uint8_t b = 128u;
            if (tileIndex == 0u) {
                // Stone.
                const int tone = 108 + static_cast<int>(noiseA % 34u) - 17;
                r = static_cast<std::uint8_t>(std::clamp(tone, 72, 146));
                g = static_cast<std::uint8_t>(std::clamp(tone - 5, 66, 140));
                b = static_cast<std::uint8_t>(std::clamp(tone - 10, 58, 132));
            } else if (tileIndex == 1u) {
                // Dirt.
                const int warm = 94 + static_cast<int>(noiseA % 28u) - 14;
                const int cool = 68 + static_cast<int>(noiseB % 20u) - 10;
                r = static_cast<std::uint8_t>(std::clamp(warm + 20, 70, 138));
                g = static_cast<std::uint8_t>(std::clamp(warm - 2, 48, 112));
                b = static_cast<std::uint8_t>(std::clamp(cool - 8, 26, 84));
            } else if (tileIndex == 2u) {
                // Grass.
                const int green = 118 + static_cast<int>(noiseA % 32u) - 16;
                r = static_cast<std::uint8_t>(std::clamp(52 + static_cast<int>(noiseB % 18u) - 9, 34, 74));
                g = static_cast<std::uint8_t>(std::clamp(green, 82, 154));
                b = static_cast<std::uint8_t>(std::clamp(44 + static_cast<int>(noiseA % 14u) - 7, 26, 64));
            } else if (tileIndex == 3u) {
                // Wood.
                const int stripe = ((localX / 3u) + (localY / 5u)) % 3u;
                const int base = (stripe == 0) ? 112 : (stripe == 1 ? 96 : 84);
                const int grain = static_cast<int>(noiseA % 16u) - 8;
                r = static_cast<std::uint8_t>(std::clamp(base + 34 + grain, 78, 168));
                g = static_cast<std::uint8_t>(std::clamp(base + 12 + grain, 56, 136));
                b = static_cast<std::uint8_t>(std::clamp(base - 6 + (grain / 2), 36, 110));
            } else if (tileIndex == 4u) {
                // Billboard grass-bush sprite (transparent background).
                const int ix = static_cast<int>(localX);
                const int iy = static_cast<int>(localY);
                const int rowFromBottom = static_cast<int>(kTileSize - 1u - localY);

                auto circleWeight = [&](int cx, int cy, int radius) -> float {
                    const int dx = ix - cx;
                    const int dy = iy - cy;
                    const int distSq = (dx * dx) + (dy * dy);
                    const int radiusSq = radius * radius;
                    if (distSq >= radiusSq) {
                        return 0.0f;
                    }
                    return 1.0f - (static_cast<float>(distSq) / static_cast<float>(radiusSq));
                };

                float leafWeight = 0.0f;
                leafWeight = std::max(leafWeight, circleWeight(4, 8, 5));
                leafWeight = std::max(leafWeight, circleWeight(8, 7, 6));
                leafWeight = std::max(leafWeight, circleWeight(11, 8, 5));
                leafWeight = std::max(leafWeight, circleWeight(8, 4, 4));

                const bool stemA = (std::abs(ix - 7) <= 1) && rowFromBottom <= 7;
                const bool stemB = (std::abs(ix - 9) <= 1) && rowFromBottom <= 6;
                const bool baseTuft = (rowFromBottom <= 3) && (std::abs(ix - 8) <= 5);
                const float stemWeight = (stemA || stemB || baseTuft) ? 0.75f : 0.0f;
                const float bushWeight = std::max(leafWeight, stemWeight);
                if (bushWeight <= 0.02f) {
                    writePixel(x, y, 0u, 0u, 0u, 0u);
                    continue;
                }

                const float edgeNoise = static_cast<float>(noiseA % 100u) / 100.0f;
                if (bushWeight < (0.22f + (edgeNoise * 0.24f))) {
                    writePixel(x, y, 0u, 0u, 0u, 0u);
                    continue;
                }

                const int green = 122 + static_cast<int>(noiseA % 66u) - 22;
                const int red = 42 + static_cast<int>(noiseB % 26u) - 9;
                const int blue = 30 + static_cast<int>(noiseA % 16u) - 5;
                r = static_cast<std::uint8_t>(std::clamp(red, 22, 88));
                g = static_cast<std::uint8_t>(std::clamp(green, 82, 200));
                b = static_cast<std::uint8_t>(std::clamp(blue, 16, 84));
                const int alphaBase = static_cast<int>(120.0f + (bushWeight * 140.0f));
                const std::uint8_t alpha = static_cast<std::uint8_t>(std::clamp(alphaBase + static_cast<int>(noiseB % 28u) - 10, 120, 250));
                writePixel(x, y, r, g, b, alpha);
                continue;
            } else {
                // Procedural flower sprites (tiles 5..8):
                // 5-6 = poppies (red/orange-red), 7-8 = light wildflowers.
                const int ix = static_cast<int>(localX);
                const int iy = static_cast<int>(localY);
                const int rowFromBottom = static_cast<int>(kTileSize - 1u - localY);
                const uint32_t flowerVariant = (tileIndex - 5u) & 3u;
                const bool poppyVariant = flowerVariant < 2u;

                constexpr std::array<std::array<int, 3>, 4> kPetalPalette = {{
                    {226, 42, 28},   // poppy red
                    {242, 88, 34},   // poppy orange-red
                    {236, 212, 244}, // lavender
                    {246, 232, 198}  // cream
                }};

                auto circleWeight = [&](int cx, int cy, int radius) -> float {
                    const int dx = ix - cx;
                    const int dy = iy - cy;
                    const int distSq = (dx * dx) + (dy * dy);
                    const int radiusSq = radius * radius;
                    if (distSq >= radiusSq) {
                        return 0.0f;
                    }
                    return 1.0f - (static_cast<float>(distSq) / static_cast<float>(radiusSq));
                };

                const bool stem = (std::abs(ix - (7 + static_cast<int>(flowerVariant & 1u))) <= 0) && rowFromBottom <= 9;
                const bool leafA = (rowFromBottom >= 2 && rowFromBottom <= 5) && (ix >= 5 && ix <= 7);
                const bool leafB = (rowFromBottom >= 3 && rowFromBottom <= 6) && (ix >= 8 && ix <= 10);
                float stemWeight = (stem || leafA || leafB) ? 0.75f : 0.0f;
                stemWeight += circleWeight(6, 11, 2) * 0.5f;
                stemWeight += circleWeight(10, 10, 2) * 0.5f;
                stemWeight = std::clamp(stemWeight, 0.0f, 1.0f);

                const int flowerCenterX = 8 + ((flowerVariant == 1u) ? 1 : (flowerVariant == 2u ? -1 : 0));
                const int flowerCenterY = 6 + ((flowerVariant >= 2u) ? 1 : 0);
                float petalWeight = 0.0f;
                petalWeight = std::max(petalWeight, circleWeight(flowerCenterX, flowerCenterY, 3));
                petalWeight = std::max(petalWeight, circleWeight(flowerCenterX - 2, flowerCenterY, 3));
                petalWeight = std::max(petalWeight, circleWeight(flowerCenterX + 2, flowerCenterY, 3));
                petalWeight = std::max(petalWeight, circleWeight(flowerCenterX, flowerCenterY - 2, 3));
                petalWeight = std::max(petalWeight, circleWeight(flowerCenterX, flowerCenterY + 2, 3));
                const float centerWeight = circleWeight(flowerCenterX, flowerCenterY, 2);

                if (petalWeight <= 0.04f && stemWeight <= 0.03f) {
                    writePixel(x, y, 0u, 0u, 0u, 0u);
                    continue;
                }

                const float edgeNoise = static_cast<float>(noiseA % 100u) / 100.0f;
                if (petalWeight > 0.0f && petalWeight < (0.20f + (edgeNoise * 0.18f)) && stemWeight < 0.45f) {
                    writePixel(x, y, 0u, 0u, 0u, 0u);
                    continue;
                }

                const std::array<int, 3> petalColor = kPetalPalette[flowerVariant];
                if (petalWeight > stemWeight) {
                    const int petalShade = static_cast<int>(noiseB % 22u) - 10;
                    r = static_cast<std::uint8_t>(std::clamp(petalColor[0] + petalShade, 80, 255));
                    g = static_cast<std::uint8_t>(std::clamp(petalColor[1] + petalShade, 80, 255));
                    b = static_cast<std::uint8_t>(std::clamp(petalColor[2] + petalShade, 80, 255));
                    if (centerWeight > 0.42f) {
                        if (poppyVariant) {
                            // Dark poppy center.
                            r = static_cast<std::uint8_t>(std::clamp(34 + static_cast<int>(noiseA % 14u) - 7, 14, 58));
                            g = static_cast<std::uint8_t>(std::clamp(24 + static_cast<int>(noiseB % 14u) - 7, 10, 46));
                            b = static_cast<std::uint8_t>(std::clamp(24 + static_cast<int>(noiseA % 12u) - 6, 10, 44));
                        } else {
                            r = static_cast<std::uint8_t>(std::clamp(246 + static_cast<int>(noiseA % 16u) - 8, 200, 255));
                            g = static_cast<std::uint8_t>(std::clamp(212 + static_cast<int>(noiseB % 22u) - 11, 150, 248));
                            b = static_cast<std::uint8_t>(std::clamp(94 + static_cast<int>(noiseA % 16u) - 8, 52, 140));
                        }
                    }
                    const int alphaBase = static_cast<int>(130.0f + (petalWeight * 120.0f));
                    const std::uint8_t alpha = static_cast<std::uint8_t>(
                        std::clamp(alphaBase + static_cast<int>(noiseA % 24u) - 12, 128, 250)
                    );
                    writePixel(x, y, r, g, b, alpha);
                } else {
                    const int green = 116 + static_cast<int>(noiseA % 36u) - 14;
                    const int red = 62 + static_cast<int>(noiseB % 24u) - 10;
                    const int blue = 40 + static_cast<int>(noiseA % 20u) - 10;
                    r = static_cast<std::uint8_t>(std::clamp(red, 34, 104));
                    g = static_cast<std::uint8_t>(std::clamp(green, 74, 176));
                    b = static_cast<std::uint8_t>(std::clamp(blue, 18, 90));
                    const int alphaBase = static_cast<int>(112.0f + (stemWeight * 122.0f));
                    const std::uint8_t alpha = static_cast<std::uint8_t>(
                        std::clamp(alphaBase + static_cast<int>(noiseB % 20u) - 8, 108, 240)
                    );
                    writePixel(x, y, r, g, b, alpha);
                }
                continue;
            }
            writePixel(x, y, r, g, b);
        }
    }

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkBufferCreateInfo stagingCreateInfo{};
    stagingCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingCreateInfo.size = kTextureBytes;
    stagingCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = vkCreateBuffer(m_device, &stagingCreateInfo, nullptr, &stagingBuffer);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateBuffer(diffuseStaging)", result);
        return false;
    }
    setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(stagingBuffer), "diffuse.staging.buffer");

    VkMemoryRequirements stagingMemReq{};
    vkGetBufferMemoryRequirements(m_device, stagingBuffer, &stagingMemReq);
    uint32_t memoryTypeIndex = findMemoryTypeIndex(
        m_physicalDevice,
        stagingMemReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
        VOX_LOGI("render") << "no staging memory type for diffuse texture\n";
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }

    VkMemoryAllocateInfo stagingAllocInfo{};
    stagingAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    stagingAllocInfo.allocationSize = stagingMemReq.size;
    stagingAllocInfo.memoryTypeIndex = memoryTypeIndex;
    result = vkAllocateMemory(m_device, &stagingAllocInfo, nullptr, &stagingMemory);
    if (result != VK_SUCCESS) {
        logVkFailure("vkAllocateMemory(diffuseStaging)", result);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }
    result = vkBindBufferMemory(m_device, stagingBuffer, stagingMemory, 0);
    if (result != VK_SUCCESS) {
        logVkFailure("vkBindBufferMemory(diffuseStaging)", result);
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }

    void* mapped = nullptr;
    result = vkMapMemory(m_device, stagingMemory, 0, kTextureBytes, 0, &mapped);
    if (result != VK_SUCCESS || mapped == nullptr) {
        logVkFailure("vkMapMemory(diffuseStaging)", result);
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }
    std::memcpy(mapped, pixels.data(), static_cast<size_t>(kTextureBytes));
    vkUnmapMemory(m_device, stagingMemory);

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = kTextureFormat;
    imageCreateInfo.extent = {kTextureWidth, kTextureHeight, 1};
    imageCreateInfo.mipLevels = diffuseMipLevels;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_diffuseTextureMemory = VK_NULL_HANDLE;
    m_diffuseTextureAllocation = VK_NULL_HANDLE;
    if (m_vmaAllocator != VK_NULL_HANDLE) {
        VmaAllocationCreateInfo allocationCreateInfo{};
        allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        result = vmaCreateImage(
            m_vmaAllocator,
            &imageCreateInfo,
            &allocationCreateInfo,
            &m_diffuseTextureImage,
            &m_diffuseTextureAllocation,
            nullptr
        );
        if (result != VK_SUCCESS) {
            logVkFailure("vmaCreateImage(diffuseTexture)", result);
            vkFreeMemory(m_device, stagingMemory, nullptr);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }
    } else
    {
        result = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_diffuseTextureImage);
        if (result != VK_SUCCESS) {
            logVkFailure("vkCreateImage(diffuseTexture)", result);
            vkFreeMemory(m_device, stagingMemory, nullptr);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }

        VkMemoryRequirements imageMemReq{};
        vkGetImageMemoryRequirements(m_device, m_diffuseTextureImage, &imageMemReq);
        memoryTypeIndex = findMemoryTypeIndex(
            m_physicalDevice,
            imageMemReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            VOX_LOGI("render") << "no device-local memory for diffuse texture\n";
            vkDestroyImage(m_device, m_diffuseTextureImage, nullptr);
            m_diffuseTextureImage = VK_NULL_HANDLE;
            vkFreeMemory(m_device, stagingMemory, nullptr);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }

        VkMemoryAllocateInfo imageAllocInfo{};
        imageAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        imageAllocInfo.allocationSize = imageMemReq.size;
        imageAllocInfo.memoryTypeIndex = memoryTypeIndex;
        result = vkAllocateMemory(m_device, &imageAllocInfo, nullptr, &m_diffuseTextureMemory);
        if (result != VK_SUCCESS) {
            logVkFailure("vkAllocateMemory(diffuseTexture)", result);
            vkDestroyImage(m_device, m_diffuseTextureImage, nullptr);
            m_diffuseTextureImage = VK_NULL_HANDLE;
            vkFreeMemory(m_device, stagingMemory, nullptr);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }
        result = vkBindImageMemory(m_device, m_diffuseTextureImage, m_diffuseTextureMemory, 0);
        if (result != VK_SUCCESS) {
            logVkFailure("vkBindImageMemory(diffuseTexture)", result);
            destroyDiffuseTextureResources();
            vkFreeMemory(m_device, stagingMemory, nullptr);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }
    }
    setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_diffuseTextureImage), "diffuse.albedo.image");

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    result = vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateCommandPool(diffuseUpload)", result);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }
    setObjectName(VK_OBJECT_TYPE_COMMAND_POOL, vkHandleToUint64(commandPool), "diffuse.upload.commandPool");

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &commandBuffer);
    if (result != VK_SUCCESS) {
        logVkFailure("vkAllocateCommandBuffers(diffuseUpload)", result);
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }
    setObjectName(VK_OBJECT_TYPE_COMMAND_BUFFER, vkHandleToUint64(commandBuffer), "diffuse.upload.commandBuffer");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        logVkFailure("vkBeginCommandBuffer(diffuseUpload)", result);
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }

    transitionImageLayout(
        commandBuffer,
        m_diffuseTextureImage,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0u,
        1u,
        0u,
        diffuseMipLevels
    );

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {kTextureWidth, kTextureHeight, 1};
    vkCmdCopyBufferToImage(
        commandBuffer,
        stagingBuffer,
        m_diffuseTextureImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &copyRegion
    );

    for (uint32_t mipLevel = 1u; mipLevel < diffuseMipLevels; ++mipLevel) {
        const uint32_t srcMip = mipLevel - 1u;
        transitionImageLayout(
            commandBuffer,
            m_diffuseTextureImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0u,
            1u,
            srcMip,
            1u
        );

        const int32_t srcTileWidth = static_cast<int32_t>(std::max(1u, kTileSize >> srcMip));
        const int32_t srcTileHeight = static_cast<int32_t>(std::max(1u, kTileSize >> srcMip));
        const int32_t dstTileWidth = static_cast<int32_t>(std::max(1u, kTileSize >> mipLevel));
        const int32_t dstTileHeight = static_cast<int32_t>(std::max(1u, kTileSize >> mipLevel));

        for (uint32_t tileY = 0; tileY < kTextureTilesY; ++tileY) {
            for (uint32_t tileX = 0; tileX < kTextureTilesX; ++tileX) {
                VkImageBlit blitRegion{};
                blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blitRegion.srcSubresource.mipLevel = srcMip;
                blitRegion.srcSubresource.baseArrayLayer = 0;
                blitRegion.srcSubresource.layerCount = 1;
                blitRegion.srcOffsets[0] = {
                    static_cast<int32_t>(tileX) * srcTileWidth,
                    static_cast<int32_t>(tileY) * srcTileHeight,
                    0
                };
                blitRegion.srcOffsets[1] = {
                    blitRegion.srcOffsets[0].x + srcTileWidth,
                    blitRegion.srcOffsets[0].y + srcTileHeight,
                    1
                };

                blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blitRegion.dstSubresource.mipLevel = mipLevel;
                blitRegion.dstSubresource.baseArrayLayer = 0;
                blitRegion.dstSubresource.layerCount = 1;
                blitRegion.dstOffsets[0] = {
                    static_cast<int32_t>(tileX) * dstTileWidth,
                    static_cast<int32_t>(tileY) * dstTileHeight,
                    0
                };
                blitRegion.dstOffsets[1] = {
                    blitRegion.dstOffsets[0].x + dstTileWidth,
                    blitRegion.dstOffsets[0].y + dstTileHeight,
                    1
                };

                vkCmdBlitImage(
                    commandBuffer,
                    m_diffuseTextureImage,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    m_diffuseTextureImage,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1,
                    &blitRegion,
                    VK_FILTER_LINEAR
                );
            }
        }
    }

    if (diffuseMipLevels > 1u) {
        transitionImageLayout(
            commandBuffer,
            m_diffuseTextureImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            0u,
            1u,
            0u,
            diffuseMipLevels - 1u
        );
    }

    transitionImageLayout(
        commandBuffer,
        m_diffuseTextureImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        0u,
        1u,
        diffuseMipLevels - 1u,
        1u
    );

    result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        logVkFailure("vkEndCommandBuffer(diffuseUpload)", result);
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    result = vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        logVkFailure("vkQueueSubmit(diffuseUpload)", result);
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }
    result = vkQueueWaitIdle(m_graphicsQueue);
    if (result != VK_SUCCESS) {
        logVkFailure("vkQueueWaitIdle(diffuseUpload)", result);
        vkDestroyCommandPool(m_device, commandPool, nullptr);
        destroyDiffuseTextureResources();
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        return false;
    }

    vkDestroyCommandPool(m_device, commandPool, nullptr);
    vkFreeMemory(m_device, stagingMemory, nullptr);
    vkDestroyBuffer(m_device, stagingBuffer, nullptr);

    VkImageViewCreateInfo viewCreateInfo{};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.image = m_diffuseTextureImage;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = kTextureFormat;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = diffuseMipLevels;
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = 1;
    result = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_diffuseTextureImageView);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateImageView(diffuseTexture)", result);
        destroyDiffuseTextureResources();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_IMAGE_VIEW,
        vkHandleToUint64(m_diffuseTextureImageView),
        "diffuse.albedo.imageView"
    );

    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.anisotropyEnable = m_supportsSamplerAnisotropy ? VK_TRUE : VK_FALSE;
    samplerCreateInfo.maxAnisotropy = m_supportsSamplerAnisotropy
        ? std::min(8.0f, m_maxSamplerAnisotropy)
        : 1.0f;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = static_cast<float>(diffuseMipLevels - 1u);
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
    result = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_diffuseTextureSampler);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateSampler(diffuseTexture)", result);
        destroyDiffuseTextureResources();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_SAMPLER,
        vkHandleToUint64(m_diffuseTextureSampler),
        "diffuse.albedo.sampler"
    );

    VkSamplerCreateInfo plantSamplerCreateInfo = samplerCreateInfo;
    plantSamplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    plantSamplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    plantSamplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    plantSamplerCreateInfo.anisotropyEnable = VK_FALSE;
    plantSamplerCreateInfo.maxAnisotropy = 1.0f;
    result = vkCreateSampler(m_device, &plantSamplerCreateInfo, nullptr, &m_diffuseTexturePlantSampler);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateSampler(diffuseTexturePlant)", result);
        destroyDiffuseTextureResources();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_SAMPLER,
        vkHandleToUint64(m_diffuseTexturePlantSampler),
        "diffuse.albedo.plantSampler"
    );

    VOX_LOGI("render") << "diffuse atlas mipmaps generated: levels=" << diffuseMipLevels
                       << ", tileSize=" << kTileSize << ", atlas=" << kTextureWidth << "x" << kTextureHeight << "\n";

    return true;
}


bool Renderer::createShadowResources() {
    if (
        m_shadowDepthImage != VK_NULL_HANDLE &&
        m_shadowDepthImageView != VK_NULL_HANDLE &&
        m_shadowDepthSampler != VK_NULL_HANDLE
    ) {
        return true;
    }

    if (m_shadowDepthFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "shadow depth format is undefined\n";
        return false;
    }

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = m_shadowDepthFormat;
    imageCreateInfo.extent.width = kShadowAtlasSize;
    imageCreateInfo.extent.height = kShadowAtlasSize;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (m_vmaAllocator != VK_NULL_HANDLE) {
        VmaAllocationCreateInfo allocationCreateInfo{};
        allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const VkResult imageResult = vmaCreateImage(
            m_vmaAllocator,
            &imageCreateInfo,
            &allocationCreateInfo,
            &m_shadowDepthImage,
            &m_shadowDepthAllocation,
            nullptr
        );
        if (imageResult != VK_SUCCESS) {
            logVkFailure("vmaCreateImage(shadowDepth)", imageResult);
            return false;
        }
        setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_shadowDepthImage), "shadow.atlas.image");
        VOX_LOGI("render") << "alloc shadow depth atlas (VMA): "
                  << kShadowAtlasSize << "x" << kShadowAtlasSize
                  << ", format=" << static_cast<int>(m_shadowDepthFormat)
                  << ", cascades=" << kShadowCascadeCount << "\n";
    } else
    {
        const VkResult imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_shadowDepthImage);
        if (imageResult != VK_SUCCESS) {
            logVkFailure("vkCreateImage(shadowDepth)", imageResult);
            return false;
        }
        setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_shadowDepthImage), "shadow.atlas.image");

        VkMemoryRequirements memoryRequirements{};
        vkGetImageMemoryRequirements(m_device, m_shadowDepthImage, &memoryRequirements);
        const uint32_t memoryTypeIndex = findMemoryTypeIndex(
            m_physicalDevice,
            memoryRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            VOX_LOGI("render") << "no memory type for shadow depth image\n";
            destroyShadowResources();
            return false;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        const VkResult allocResult = vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_shadowDepthMemory);
        if (allocResult != VK_SUCCESS) {
            logVkFailure("vkAllocateMemory(shadowDepth)", allocResult);
            destroyShadowResources();
            return false;
        }

        const VkResult bindResult = vkBindImageMemory(m_device, m_shadowDepthImage, m_shadowDepthMemory, 0);
        if (bindResult != VK_SUCCESS) {
            logVkFailure("vkBindImageMemory(shadowDepth)", bindResult);
            destroyShadowResources();
            return false;
        }
        VOX_LOGI("render") << "alloc shadow depth atlas (vk): "
                  << kShadowAtlasSize << "x" << kShadowAtlasSize
                  << ", format=" << static_cast<int>(m_shadowDepthFormat)
                  << ", cascades=" << kShadowCascadeCount << "\n";
    }

    VkImageViewCreateInfo viewCreateInfo{};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.image = m_shadowDepthImage;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = m_shadowDepthFormat;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = 1;
    const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_shadowDepthImageView);
    if (viewResult != VK_SUCCESS) {
        logVkFailure("vkCreateImageView(shadowDepth)", viewResult);
        destroyShadowResources();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_IMAGE_VIEW,
        vkHandleToUint64(m_shadowDepthImageView),
        "shadow.atlas.imageView"
    );

    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.compareEnable = VK_TRUE;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
    const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_shadowDepthSampler);
    if (samplerResult != VK_SUCCESS) {
        logVkFailure("vkCreateSampler(shadowDepth)", samplerResult);
        destroyShadowResources();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_SAMPLER,
        vkHandleToUint64(m_shadowDepthSampler),
        "shadow.atlas.sampler"
    );

    m_shadowDepthInitialized = false;
    VOX_LOGI("render") << "shadow resources ready (atlas " << kShadowAtlasSize << "x" << kShadowAtlasSize
              << ", cascades=" << kShadowCascadeCount << ")\n";
    return true;
}


bool Renderer::createVoxelGiResources() {
    const bool surfaceFacesReady = std::all_of(
        m_voxelGiSurfaceFaceImageViews.begin(),
        m_voxelGiSurfaceFaceImageViews.end(),
        [](VkImageView view) { return view != VK_NULL_HANDLE; }
    );
    if (m_voxelGiSampler != VK_NULL_HANDLE &&
        m_voxelGiImageViews[0] != VK_NULL_HANDLE &&
        m_voxelGiImageViews[1] != VK_NULL_HANDLE &&
        surfaceFacesReady &&
        m_voxelGiSkyExposureImageView != VK_NULL_HANDLE &&
        m_voxelGiOccupancySampler != VK_NULL_HANDLE &&
        m_voxelGiOccupancyImageView != VK_NULL_HANDLE) {
        return true;
    }

    if (m_voxelGiFormat == VK_FORMAT_UNDEFINED) {
        m_voxelGiFormat = findSupportedVoxelGiFormat(m_physicalDevice);
    }
    if (m_voxelGiFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "voxel GI format unsupported (requires sampled+storage 3D image)\n";
        return false;
    }
    if (m_voxelGiOccupancyFormat == VK_FORMAT_UNDEFINED) {
        m_voxelGiOccupancyFormat = findSupportedVoxelGiOccupancyFormat(m_physicalDevice);
    }
    if (m_voxelGiOccupancyFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "voxel GI occupancy format unsupported (requires sampled 3D image)\n";
        return false;
    }

    for (std::size_t volumeIndex = 0; volumeIndex < m_voxelGiImages.size(); ++volumeIndex) {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
        imageCreateInfo.format = m_voxelGiFormat;
        imageCreateInfo.extent.width = kVoxelGiGridResolution;
        imageCreateInfo.extent.height = kVoxelGiGridResolution;
        imageCreateInfo.extent.depth = kVoxelGiGridResolution;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage =
            VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (m_vmaAllocator != VK_NULL_HANDLE) {
            VmaAllocationCreateInfo allocationCreateInfo{};
            allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            const VkResult imageResult = vmaCreateImage(
                m_vmaAllocator,
                &imageCreateInfo,
                &allocationCreateInfo,
                &m_voxelGiImages[volumeIndex],
                &m_voxelGiImageAllocations[volumeIndex],
                nullptr
            );
            if (imageResult != VK_SUCCESS) {
                logVkFailure("vmaCreateImage(voxelGi)", imageResult);
                destroyVoxelGiResources();
                return false;
            }
        } else
        {
            const VkResult imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_voxelGiImages[volumeIndex]);
            if (imageResult != VK_SUCCESS) {
                logVkFailure("vkCreateImage(voxelGi)", imageResult);
                destroyVoxelGiResources();
                return false;
            }

            VkMemoryRequirements memoryRequirements{};
            vkGetImageMemoryRequirements(m_device, m_voxelGiImages[volumeIndex], &memoryRequirements);
            const uint32_t memoryTypeIndex = findMemoryTypeIndex(
                m_physicalDevice,
                memoryRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );
            if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                VOX_LOGE("render") << "no memory type for voxel GI image\n";
                destroyVoxelGiResources();
                return false;
            }

            VkMemoryAllocateInfo allocateInfo{};
            allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocateInfo.allocationSize = memoryRequirements.size;
            allocateInfo.memoryTypeIndex = memoryTypeIndex;
            const VkResult allocResult =
                vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_voxelGiImageMemories[volumeIndex]);
            if (allocResult != VK_SUCCESS) {
                logVkFailure("vkAllocateMemory(voxelGi)", allocResult);
                destroyVoxelGiResources();
                return false;
            }

            const VkResult bindResult =
                vkBindImageMemory(m_device, m_voxelGiImages[volumeIndex], m_voxelGiImageMemories[volumeIndex], 0);
            if (bindResult != VK_SUCCESS) {
                logVkFailure("vkBindImageMemory(voxelGi)", bindResult);
                destroyVoxelGiResources();
                return false;
            }
        }

        const std::string imageName = "voxelGi.radiance.image." + std::to_string(volumeIndex);
        setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_voxelGiImages[volumeIndex]), imageName.c_str());

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_voxelGiImages[volumeIndex];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        viewCreateInfo.format = m_voxelGiFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;
        const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_voxelGiImageViews[volumeIndex]);
        if (viewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(voxelGi)", viewResult);
            destroyVoxelGiResources();
            return false;
        }
        const std::string viewName = "voxelGi.radiance.imageView." + std::to_string(volumeIndex);
        setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_voxelGiImageViews[volumeIndex]), viewName.c_str());
    }

    {
        static constexpr std::array<const char*, 6> kSurfaceFaceNames = {
            "posX",
            "negX",
            "posY",
            "negY",
            "posZ",
            "negZ"
        };
        for (std::size_t faceIndex = 0; faceIndex < kSurfaceFaceNames.size(); ++faceIndex) {
            VkImageCreateInfo surfaceFaceImageCreateInfo{};
            surfaceFaceImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            surfaceFaceImageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
            surfaceFaceImageCreateInfo.format = m_voxelGiFormat;
            surfaceFaceImageCreateInfo.extent.width = kVoxelGiGridResolution;
            surfaceFaceImageCreateInfo.extent.height = kVoxelGiGridResolution;
            surfaceFaceImageCreateInfo.extent.depth = kVoxelGiGridResolution;
            surfaceFaceImageCreateInfo.mipLevels = 1;
            surfaceFaceImageCreateInfo.arrayLayers = 1;
            surfaceFaceImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            surfaceFaceImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            surfaceFaceImageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
            surfaceFaceImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            surfaceFaceImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            if (m_vmaAllocator != VK_NULL_HANDLE) {
                VmaAllocationCreateInfo surfaceFaceAllocCreateInfo{};
                surfaceFaceAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                surfaceFaceAllocCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                const VkResult surfaceFaceImageResult = vmaCreateImage(
                    m_vmaAllocator,
                    &surfaceFaceImageCreateInfo,
                    &surfaceFaceAllocCreateInfo,
                    &m_voxelGiSurfaceFaceImages[faceIndex],
                    &m_voxelGiSurfaceFaceAllocations[faceIndex],
                    nullptr
                );
                if (surfaceFaceImageResult != VK_SUCCESS) {
                    logVkFailure("vmaCreateImage(voxelGiSurfaceFace)", surfaceFaceImageResult);
                    destroyVoxelGiResources();
                    return false;
                }
            } else
            {
                const VkResult surfaceFaceImageResult =
                    vkCreateImage(m_device, &surfaceFaceImageCreateInfo, nullptr, &m_voxelGiSurfaceFaceImages[faceIndex]);
                if (surfaceFaceImageResult != VK_SUCCESS) {
                    logVkFailure("vkCreateImage(voxelGiSurfaceFace)", surfaceFaceImageResult);
                    destroyVoxelGiResources();
                    return false;
                }

                VkMemoryRequirements surfaceFaceMemoryRequirements{};
                vkGetImageMemoryRequirements(m_device, m_voxelGiSurfaceFaceImages[faceIndex], &surfaceFaceMemoryRequirements);
                const uint32_t surfaceFaceMemoryTypeIndex = findMemoryTypeIndex(
                    m_physicalDevice,
                    surfaceFaceMemoryRequirements.memoryTypeBits,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                );
                if (surfaceFaceMemoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                    VOX_LOGE("render") << "no memory type for voxel GI surface face image\n";
                    destroyVoxelGiResources();
                    return false;
                }

                VkMemoryAllocateInfo surfaceFaceAllocateInfo{};
                surfaceFaceAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                surfaceFaceAllocateInfo.allocationSize = surfaceFaceMemoryRequirements.size;
                surfaceFaceAllocateInfo.memoryTypeIndex = surfaceFaceMemoryTypeIndex;
                const VkResult surfaceFaceAllocResult =
                    vkAllocateMemory(m_device, &surfaceFaceAllocateInfo, nullptr, &m_voxelGiSurfaceFaceMemories[faceIndex]);
                if (surfaceFaceAllocResult != VK_SUCCESS) {
                    logVkFailure("vkAllocateMemory(voxelGiSurfaceFace)", surfaceFaceAllocResult);
                    destroyVoxelGiResources();
                    return false;
                }

                const VkResult surfaceFaceBindResult =
                    vkBindImageMemory(m_device, m_voxelGiSurfaceFaceImages[faceIndex], m_voxelGiSurfaceFaceMemories[faceIndex], 0);
                if (surfaceFaceBindResult != VK_SUCCESS) {
                    logVkFailure("vkBindImageMemory(voxelGiSurfaceFace)", surfaceFaceBindResult);
                    destroyVoxelGiResources();
                    return false;
                }
            }

            const std::string faceImageName = "voxelGi.surfaceFace." + std::string(kSurfaceFaceNames[faceIndex]) + ".image";
            setObjectName(
                VK_OBJECT_TYPE_IMAGE,
                vkHandleToUint64(m_voxelGiSurfaceFaceImages[faceIndex]),
                faceImageName.c_str()
            );

            VkImageViewCreateInfo surfaceFaceViewCreateInfo{};
            surfaceFaceViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            surfaceFaceViewCreateInfo.image = m_voxelGiSurfaceFaceImages[faceIndex];
            surfaceFaceViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
            surfaceFaceViewCreateInfo.format = m_voxelGiFormat;
            surfaceFaceViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            surfaceFaceViewCreateInfo.subresourceRange.baseMipLevel = 0;
            surfaceFaceViewCreateInfo.subresourceRange.levelCount = 1;
            surfaceFaceViewCreateInfo.subresourceRange.baseArrayLayer = 0;
            surfaceFaceViewCreateInfo.subresourceRange.layerCount = 1;
            const VkResult surfaceFaceViewResult =
                vkCreateImageView(m_device, &surfaceFaceViewCreateInfo, nullptr, &m_voxelGiSurfaceFaceImageViews[faceIndex]);
            if (surfaceFaceViewResult != VK_SUCCESS) {
                logVkFailure("vkCreateImageView(voxelGiSurfaceFace)", surfaceFaceViewResult);
                destroyVoxelGiResources();
                return false;
            }
            const std::string faceImageViewName = "voxelGi.surfaceFace." + std::string(kSurfaceFaceNames[faceIndex]) + ".imageView";
            setObjectName(
                VK_OBJECT_TYPE_IMAGE_VIEW,
                vkHandleToUint64(m_voxelGiSurfaceFaceImageViews[faceIndex]),
                faceImageViewName.c_str()
            );
        }
    }

    {
        VkImageCreateInfo skyExposureImageCreateInfo{};
        skyExposureImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        skyExposureImageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
        skyExposureImageCreateInfo.format = m_voxelGiFormat;
        skyExposureImageCreateInfo.extent.width = kVoxelGiGridResolution;
        skyExposureImageCreateInfo.extent.height = kVoxelGiGridResolution;
        skyExposureImageCreateInfo.extent.depth = kVoxelGiGridResolution;
        skyExposureImageCreateInfo.mipLevels = 1;
        skyExposureImageCreateInfo.arrayLayers = 1;
        skyExposureImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        skyExposureImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        skyExposureImageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        skyExposureImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        skyExposureImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (m_vmaAllocator != VK_NULL_HANDLE) {
            VmaAllocationCreateInfo skyExposureAllocCreateInfo{};
            skyExposureAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            skyExposureAllocCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            const VkResult skyExposureImageResult = vmaCreateImage(
                m_vmaAllocator,
                &skyExposureImageCreateInfo,
                &skyExposureAllocCreateInfo,
                &m_voxelGiSkyExposureImage,
                &m_voxelGiSkyExposureAllocation,
                nullptr
            );
            if (skyExposureImageResult != VK_SUCCESS) {
                logVkFailure("vmaCreateImage(voxelGiSkyExposure)", skyExposureImageResult);
                destroyVoxelGiResources();
                return false;
            }
        } else
        {
            const VkResult skyExposureImageResult =
                vkCreateImage(m_device, &skyExposureImageCreateInfo, nullptr, &m_voxelGiSkyExposureImage);
            if (skyExposureImageResult != VK_SUCCESS) {
                logVkFailure("vkCreateImage(voxelGiSkyExposure)", skyExposureImageResult);
                destroyVoxelGiResources();
                return false;
            }

            VkMemoryRequirements skyExposureMemoryRequirements{};
            vkGetImageMemoryRequirements(m_device, m_voxelGiSkyExposureImage, &skyExposureMemoryRequirements);
            const uint32_t skyExposureMemoryTypeIndex = findMemoryTypeIndex(
                m_physicalDevice,
                skyExposureMemoryRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );
            if (skyExposureMemoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                VOX_LOGE("render") << "no memory type for voxel GI sky exposure image\n";
                destroyVoxelGiResources();
                return false;
            }

            VkMemoryAllocateInfo skyExposureAllocateInfo{};
            skyExposureAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            skyExposureAllocateInfo.allocationSize = skyExposureMemoryRequirements.size;
            skyExposureAllocateInfo.memoryTypeIndex = skyExposureMemoryTypeIndex;
            const VkResult skyExposureAllocResult =
                vkAllocateMemory(m_device, &skyExposureAllocateInfo, nullptr, &m_voxelGiSkyExposureMemory);
            if (skyExposureAllocResult != VK_SUCCESS) {
                logVkFailure("vkAllocateMemory(voxelGiSkyExposure)", skyExposureAllocResult);
                destroyVoxelGiResources();
                return false;
            }

            const VkResult skyExposureBindResult =
                vkBindImageMemory(m_device, m_voxelGiSkyExposureImage, m_voxelGiSkyExposureMemory, 0);
            if (skyExposureBindResult != VK_SUCCESS) {
                logVkFailure("vkBindImageMemory(voxelGiSkyExposure)", skyExposureBindResult);
                destroyVoxelGiResources();
                return false;
            }
        }

        setObjectName(
            VK_OBJECT_TYPE_IMAGE,
            vkHandleToUint64(m_voxelGiSkyExposureImage),
            "voxelGi.skyExposure.image"
        );

        VkImageViewCreateInfo skyExposureViewCreateInfo{};
        skyExposureViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        skyExposureViewCreateInfo.image = m_voxelGiSkyExposureImage;
        skyExposureViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        skyExposureViewCreateInfo.format = m_voxelGiFormat;
        skyExposureViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        skyExposureViewCreateInfo.subresourceRange.baseMipLevel = 0;
        skyExposureViewCreateInfo.subresourceRange.levelCount = 1;
        skyExposureViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        skyExposureViewCreateInfo.subresourceRange.layerCount = 1;
        const VkResult skyExposureViewResult =
            vkCreateImageView(m_device, &skyExposureViewCreateInfo, nullptr, &m_voxelGiSkyExposureImageView);
        if (skyExposureViewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(voxelGiSkyExposure)", skyExposureViewResult);
            destroyVoxelGiResources();
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_IMAGE_VIEW,
            vkHandleToUint64(m_voxelGiSkyExposureImageView),
            "voxelGi.skyExposure.imageView"
        );
    }

    {
        VkImageCreateInfo occupancyImageCreateInfo{};
        occupancyImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        occupancyImageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
        occupancyImageCreateInfo.format = m_voxelGiOccupancyFormat;
        occupancyImageCreateInfo.extent.width = kVoxelGiGridResolution;
        occupancyImageCreateInfo.extent.height = kVoxelGiGridResolution;
        occupancyImageCreateInfo.extent.depth = kVoxelGiGridResolution;
        occupancyImageCreateInfo.mipLevels = 1;
        occupancyImageCreateInfo.arrayLayers = 1;
        occupancyImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        occupancyImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        occupancyImageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        occupancyImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        occupancyImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if (m_vmaAllocator != VK_NULL_HANDLE) {
            VmaAllocationCreateInfo occupancyAllocCreateInfo{};
            occupancyAllocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            occupancyAllocCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            const VkResult occupancyImageResult = vmaCreateImage(
                m_vmaAllocator,
                &occupancyImageCreateInfo,
                &occupancyAllocCreateInfo,
                &m_voxelGiOccupancyImage,
                &m_voxelGiOccupancyAllocation,
                nullptr
            );
            if (occupancyImageResult != VK_SUCCESS) {
                logVkFailure("vmaCreateImage(voxelGiOccupancy)", occupancyImageResult);
                destroyVoxelGiResources();
                return false;
            }
        } else
        {
            const VkResult occupancyImageResult =
                vkCreateImage(m_device, &occupancyImageCreateInfo, nullptr, &m_voxelGiOccupancyImage);
            if (occupancyImageResult != VK_SUCCESS) {
                logVkFailure("vkCreateImage(voxelGiOccupancy)", occupancyImageResult);
                destroyVoxelGiResources();
                return false;
            }

            VkMemoryRequirements occupancyMemoryRequirements{};
            vkGetImageMemoryRequirements(m_device, m_voxelGiOccupancyImage, &occupancyMemoryRequirements);
            const uint32_t occupancyMemoryTypeIndex = findMemoryTypeIndex(
                m_physicalDevice,
                occupancyMemoryRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );
            if (occupancyMemoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                VOX_LOGE("render") << "no memory type for voxel GI occupancy image\n";
                destroyVoxelGiResources();
                return false;
            }

            VkMemoryAllocateInfo occupancyAllocateInfo{};
            occupancyAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            occupancyAllocateInfo.allocationSize = occupancyMemoryRequirements.size;
            occupancyAllocateInfo.memoryTypeIndex = occupancyMemoryTypeIndex;
            const VkResult occupancyAllocResult =
                vkAllocateMemory(m_device, &occupancyAllocateInfo, nullptr, &m_voxelGiOccupancyMemory);
            if (occupancyAllocResult != VK_SUCCESS) {
                logVkFailure("vkAllocateMemory(voxelGiOccupancy)", occupancyAllocResult);
                destroyVoxelGiResources();
                return false;
            }

            const VkResult occupancyBindResult =
                vkBindImageMemory(m_device, m_voxelGiOccupancyImage, m_voxelGiOccupancyMemory, 0);
            if (occupancyBindResult != VK_SUCCESS) {
                logVkFailure("vkBindImageMemory(voxelGiOccupancy)", occupancyBindResult);
                destroyVoxelGiResources();
                return false;
            }
        }

        setObjectName(
            VK_OBJECT_TYPE_IMAGE,
            vkHandleToUint64(m_voxelGiOccupancyImage),
            "voxelGi.occupancy.image"
        );

        VkImageViewCreateInfo occupancyViewCreateInfo{};
        occupancyViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        occupancyViewCreateInfo.image = m_voxelGiOccupancyImage;
        occupancyViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        occupancyViewCreateInfo.format = m_voxelGiOccupancyFormat;
        occupancyViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        occupancyViewCreateInfo.subresourceRange.baseMipLevel = 0;
        occupancyViewCreateInfo.subresourceRange.levelCount = 1;
        occupancyViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        occupancyViewCreateInfo.subresourceRange.layerCount = 1;
        const VkResult occupancyViewResult =
            vkCreateImageView(m_device, &occupancyViewCreateInfo, nullptr, &m_voxelGiOccupancyImageView);
        if (occupancyViewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(voxelGiOccupancy)", occupancyViewResult);
            destroyVoxelGiResources();
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_IMAGE_VIEW,
            vkHandleToUint64(m_voxelGiOccupancyImageView),
            "voxelGi.occupancy.imageView"
        );
    }

    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
    const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_voxelGiSampler);
    if (samplerResult != VK_SUCCESS) {
        logVkFailure("vkCreateSampler(voxelGi)", samplerResult);
        destroyVoxelGiResources();
        return false;
    }
    setObjectName(VK_OBJECT_TYPE_SAMPLER, vkHandleToUint64(m_voxelGiSampler), "voxelGi.radiance.sampler");

    VkSamplerCreateInfo occupancySamplerCreateInfo = samplerCreateInfo;
    occupancySamplerCreateInfo.magFilter = VK_FILTER_NEAREST;
    occupancySamplerCreateInfo.minFilter = VK_FILTER_NEAREST;
    const VkResult occupancySamplerResult = vkCreateSampler(
        m_device,
        &occupancySamplerCreateInfo,
        nullptr,
        &m_voxelGiOccupancySampler
    );
    if (occupancySamplerResult != VK_SUCCESS) {
        logVkFailure("vkCreateSampler(voxelGiOccupancy)", occupancySamplerResult);
        destroyVoxelGiResources();
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_SAMPLER,
        vkHandleToUint64(m_voxelGiOccupancySampler),
        "voxelGi.occupancy.sampler"
    );

    constexpr const char* kVoxelGiSkyExposureShaderPath = "../src/render/shaders/voxel_gi_sky_exposure.comp.slang.spv";
    constexpr const char* kVoxelGiSurfaceShaderPath = "../src/render/shaders/voxel_gi_surface.comp.slang.spv";
    constexpr const char* kVoxelGiInjectShaderPath = "../src/render/shaders/voxel_gi_inject.comp.slang.spv";
    constexpr const char* kVoxelGiPropagateShaderPath = "../src/render/shaders/voxel_gi_propagate.comp.slang.spv";
    const bool hasSkyExposureShader = readBinaryFile(kVoxelGiSkyExposureShaderPath).has_value();
    const bool hasSurfaceShader = readBinaryFile(kVoxelGiSurfaceShaderPath).has_value();
    const bool hasInjectShader = readBinaryFile(kVoxelGiInjectShaderPath).has_value();
    const bool hasPropagateShader = readBinaryFile(kVoxelGiPropagateShaderPath).has_value();
    if (!hasSkyExposureShader || !hasSurfaceShader || !hasInjectShader || !hasPropagateShader) {
        VOX_LOGI("render")
            << "voxel GI compute shaders not found; keeping static volume fallback (expected: "
            << kVoxelGiSkyExposureShaderPath << ", " << kVoxelGiSurfaceShaderPath << ", " << kVoxelGiInjectShaderPath
            << ", " << kVoxelGiPropagateShaderPath << ")\n";
        m_voxelGiComputeAvailable = false;
        m_voxelGiInitialized = false;
        m_voxelGiSkyExposureInitialized = false;
        m_voxelGiOccupancyInitialized = false;
        return true;
    }

    if (m_voxelGiDescriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding cameraBinding{};
        cameraBinding.binding = 0;
        cameraBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        cameraBinding.descriptorCount = 1;
        cameraBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding shadowBinding{};
        shadowBinding.binding = 1;
        shadowBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadowBinding.descriptorCount = 1;
        shadowBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding injectWriteBinding{};
        injectWriteBinding.binding = 2;
        injectWriteBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        injectWriteBinding.descriptorCount = 1;
        injectWriteBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding propagateReadBinding{};
        propagateReadBinding.binding = 3;
        propagateReadBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        propagateReadBinding.descriptorCount = 1;
        propagateReadBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding propagateWriteBinding{};
        propagateWriteBinding.binding = 4;
        propagateWriteBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        propagateWriteBinding.descriptorCount = 1;
        propagateWriteBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding occupancyBinding{};
        occupancyBinding.binding = 5;
        occupancyBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        occupancyBinding.descriptorCount = 1;
        occupancyBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding surfaceBinding{};
        surfaceBinding.binding = 6;
        surfaceBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        surfaceBinding.descriptorCount = 1;
        surfaceBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding surfaceNegBinding{};
        surfaceNegBinding.binding = 7;
        surfaceNegBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        surfaceNegBinding.descriptorCount = 1;
        surfaceNegBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding surfacePosYBinding{};
        surfacePosYBinding.binding = 8;
        surfacePosYBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        surfacePosYBinding.descriptorCount = 1;
        surfacePosYBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding surfaceNegYBinding{};
        surfaceNegYBinding.binding = 9;
        surfaceNegYBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        surfaceNegYBinding.descriptorCount = 1;
        surfaceNegYBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding surfacePosZBinding{};
        surfacePosZBinding.binding = 10;
        surfacePosZBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        surfacePosZBinding.descriptorCount = 1;
        surfacePosZBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding surfaceNegZBinding{};
        surfaceNegZBinding.binding = 11;
        surfaceNegZBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        surfaceNegZBinding.descriptorCount = 1;
        surfaceNegZBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding skyExposureBinding{};
        skyExposureBinding.binding = 12;
        skyExposureBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        skyExposureBinding.descriptorCount = 1;
        skyExposureBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        const std::array<VkDescriptorSetLayoutBinding, 13> bindings = {
            cameraBinding,
            shadowBinding,
            injectWriteBinding,
            propagateReadBinding,
            propagateWriteBinding,
            occupancyBinding,
            surfaceBinding,
            surfaceNegBinding,
            surfacePosYBinding,
            surfaceNegYBinding,
            surfacePosZBinding,
            surfaceNegZBinding,
            skyExposureBinding
        };

        if (!createDescriptorSetLayout(
                bindings,
                m_voxelGiDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(voxelGi)",
                "renderer.descriptorSetLayout.voxelGi"
            )) {
            destroyVoxelGiResources();
            return false;
        }
    }

    if (m_voxelGiDescriptorPool == VK_NULL_HANDLE) {
        const std::array<VkDescriptorPoolSize, 4> poolSizes = {
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2 * kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 9 * kMaxFramesInFlight}
        };
        if (!createDescriptorPool(
                poolSizes,
                kMaxFramesInFlight,
                m_voxelGiDescriptorPool,
                "vkCreateDescriptorPool(voxelGi)",
                "renderer.descriptorPool.voxelGi"
            )) {
            destroyVoxelGiResources();
            return false;
        }
    }

    if (!allocatePerFrameDescriptorSets(
            m_voxelGiDescriptorPool,
            m_voxelGiDescriptorSetLayout,
            std::span<VkDescriptorSet>(m_voxelGiDescriptorSets),
            "vkAllocateDescriptorSets(voxelGi)",
            "renderer.descriptorSet.voxelGi.frame"
        )) {
        destroyVoxelGiResources();
        return false;
    }

    std::array<VkShaderModule, 4> shaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& skyExposureShaderModule = shaderModules[0];
    VkShaderModule& surfaceShaderModule = shaderModules[1];
    VkShaderModule& injectShaderModule = shaderModules[2];
    VkShaderModule& propagateShaderModule = shaderModules[3];
    if (!createShaderModuleFromFile(
            m_device,
            kVoxelGiSkyExposureShaderPath,
            "voxel_gi_sky_exposure.comp",
            skyExposureShaderModule
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyVoxelGiResources();
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kVoxelGiSurfaceShaderPath,
            "voxel_gi_surface.comp",
            surfaceShaderModule
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyVoxelGiResources();
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kVoxelGiInjectShaderPath,
            "voxel_gi_inject.comp",
            injectShaderModule
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyVoxelGiResources();
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kVoxelGiPropagateShaderPath,
            "voxel_gi_propagate.comp",
            propagateShaderModule
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyVoxelGiResources();
        return false;
    }
    if (!createComputePipelineLayout(
            m_voxelGiDescriptorSetLayout,
            std::span<const VkPushConstantRange>{},
            m_voxelGiPipelineLayout,
            "vkCreatePipelineLayout(voxelGi)",
            "renderer.pipelineLayout.voxelGi"
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyVoxelGiResources();
        return false;
    }

    if (!createComputePipeline(
            m_voxelGiPipelineLayout,
            skyExposureShaderModule,
            m_voxelGiSkyExposurePipeline,
            "vkCreateComputePipelines(voxelGiSkyExposure)",
            "pipeline.voxelGi.skyExposure"
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyVoxelGiResources();
        return false;
    }
    if (!createComputePipeline(
            m_voxelGiPipelineLayout,
            surfaceShaderModule,
            m_voxelGiSurfacePipeline,
            "vkCreateComputePipelines(voxelGiSurface)",
            "pipeline.voxelGi.surface"
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyVoxelGiResources();
        return false;
    }
    if (!createComputePipeline(
            m_voxelGiPipelineLayout,
            injectShaderModule,
            m_voxelGiInjectPipeline,
            "vkCreateComputePipelines(voxelGiInject)",
            "pipeline.voxelGi.inject"
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyVoxelGiResources();
        return false;
    }
    if (!createComputePipeline(
            m_voxelGiPipelineLayout,
            propagateShaderModule,
            m_voxelGiPropagatePipeline,
            "vkCreateComputePipelines(voxelGiPropagate)",
            "pipeline.voxelGi.propagate"
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyVoxelGiResources();
        return false;
    }
    destroyShaderModules(m_device, shaderModules);

    m_voxelGiComputeAvailable = true;
    m_voxelGiInitialized = false;
    m_voxelGiSkyExposureInitialized = false;
    m_voxelGiOccupancyInitialized = false;
    VOX_LOGI("render") << "voxel GI resources ready: "
                       << kVoxelGiGridResolution << "^3, format=" << static_cast<int>(m_voxelGiFormat)
                       << ", occupancyFormat=" << static_cast<int>(m_voxelGiOccupancyFormat)
                       << ", compute=enabled\n";
    return true;
}


bool Renderer::createSwapchain() {
    const SwapchainSupport support = querySwapchainSupport(m_physicalDevice, m_surface);
    if (support.formats.empty() || support.presentModes.empty()) {
        VOX_LOGI("render") << "swapchain support query returned no formats or present modes\n";
        return false;
    }

    const VkSurfaceFormatKHR surfaceFormat = chooseSwapchainFormat(support.formats);
    const VkPresentModeKHR presentMode = choosePresentMode(support.presentModes);
    const VkExtent2D extent = chooseExtent(m_window, support.capabilities);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = m_surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    const VkResult swapchainResult = vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain);
    if (swapchainResult != VK_SUCCESS) {
        logVkFailure("vkCreateSwapchainKHR", swapchainResult);
        return false;
    }
    setObjectName(VK_OBJECT_TYPE_SWAPCHAIN_KHR, vkHandleToUint64(m_swapchain), "swapchain.main");

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, m_swapchainImages.data());
    for (uint32_t i = 0; i < imageCount; ++i) {
        const std::string imageName = "swapchain.image." + std::to_string(i);
        setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_swapchainImages[i]), imageName.c_str());
    }

    m_swapchainFormat = surfaceFormat.format;
    m_swapchainExtent = extent;

    m_swapchainImageViews.resize(imageCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_swapchainImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_swapchainFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_swapchainImageViews[i]) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed to create swapchain image view " << i << "\n";
            return false;
        }
        const std::string viewName = "swapchain.imageView." + std::to_string(i);
        setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_swapchainImageViews[i]), viewName.c_str());
    }

    VOX_LOGI("render") << "swapchain ready: images=" << imageCount
              << ", extent=" << m_swapchainExtent.width << "x" << m_swapchainExtent.height << "\n";
    m_swapchainImageInitialized.assign(imageCount, false);
    m_swapchainImageTimelineValues.assign(imageCount, 0);
    if (!createHdrResolveTargets()) {
        VOX_LOGE("render") << "HDR resolve target creation failed\n";
        return false;
    }
    if (!createMsaaColorTargets()) {
        VOX_LOGE("render") << "MSAA color target creation failed\n";
        return false;
    }
    if (!createDepthTargets()) {
        VOX_LOGE("render") << "depth target creation failed\n";
        return false;
    }
    if (!createAoTargets()) {
        VOX_LOGE("render") << "AO target creation failed\n";
        return false;
    }
    m_renderFinishedSemaphores.resize(imageCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        const VkResult semaphoreResult =
            vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_renderFinishedSemaphores[i]);
        if (semaphoreResult != VK_SUCCESS) {
            logVkFailure("vkCreateSemaphore(renderFinishedPerImage)", semaphoreResult);
            return false;
        }
        const std::string semaphoreName = "swapchain.renderFinished." + std::to_string(i);
        setObjectName(VK_OBJECT_TYPE_SEMAPHORE, vkHandleToUint64(m_renderFinishedSemaphores[i]), semaphoreName.c_str());
    }

    return true;
}


bool Renderer::createImGuiResources() {
    if (m_imguiInitialized) {
        return true;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    if (!ImGui_ImplGlfw_InitForVulkan(m_window, true)) {
        VOX_LOGE("imgui") << "ImGui_ImplGlfw_InitForVulkan failed\n";
        ImGui::DestroyContext();
        return false;
    }

    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_SAMPLER, 256},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256},
        {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 256},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 256},
        {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 256},
    };

    if (!createDescriptorPool(
            poolSizes,
            256,
            m_imguiDescriptorPool,
            "vkCreateDescriptorPool(imgui)",
            "imgui.descriptorPool",
            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
        )) {
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        return false;
    }

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = m_instance;
    initInfo.PhysicalDevice = m_physicalDevice;
    initInfo.Device = m_device;
    initInfo.QueueFamily = m_graphicsQueueFamilyIndex;
    initInfo.Queue = m_graphicsQueue;
    initInfo.DescriptorPool = m_imguiDescriptorPool;
    initInfo.MinImageCount = std::max<uint32_t>(2u, static_cast<uint32_t>(m_swapchainImages.size()));
    initInfo.ImageCount = static_cast<uint32_t>(m_swapchainImages.size());
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &m_swapchainFormat;
    initInfo.PipelineRenderingCreateInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    initInfo.CheckVkResultFn = imguiCheckVkResult;
    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        VOX_LOGE("imgui") << "ImGui_ImplVulkan_Init failed\n";
        vkDestroyDescriptorPool(m_device, m_imguiDescriptorPool, nullptr);
        m_imguiDescriptorPool = VK_NULL_HANDLE;
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        return false;
    }

    if (!ImGui_ImplVulkan_CreateFontsTexture()) {
        VOX_LOGE("imgui") << "ImGui_ImplVulkan_CreateFontsTexture failed\n";
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(m_device, m_imguiDescriptorPool, nullptr);
        m_imguiDescriptorPool = VK_NULL_HANDLE;
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        return false;
    }

    m_imguiInitialized = true;
    return true;
}

void Renderer::destroyImGuiResources() {
    if (!m_imguiInitialized) {
        return;
    }

    VOX_LOGI("imgui") << "destroy begin\n";
    ImGui_ImplVulkan_DestroyFontsTexture();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (m_imguiDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_imguiDescriptorPool, nullptr);
        m_imguiDescriptorPool = VK_NULL_HANDLE;
    }
    m_imguiInitialized = false;
    VOX_LOGI("imgui") << "destroy complete\n";
}


bool Renderer::recreateSwapchain() {
    VOX_LOGI("render") << "recreateSwapchain begin\n";
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(m_window, &width, &height);
    while ((width == 0 || height == 0) && glfwWindowShouldClose(m_window) == GLFW_FALSE) {
        // Keep swapchain recreation responsive when minimized without hard-blocking shutdown.
        glfwWaitEventsTimeout(0.05);
        glfwGetFramebufferSize(m_window, &width, &height);
    }
    if (glfwWindowShouldClose(m_window) == GLFW_TRUE) {
        return false;
    }

    vkDeviceWaitIdle(m_device);

    destroyPipeline();
    destroySwapchain();

    if (!createSwapchain()) {
        VOX_LOGE("render") << "recreateSwapchain failed: createSwapchain\n";
        return false;
    }
    if (!createGraphicsPipeline()) {
        VOX_LOGE("render") << "recreateSwapchain failed: createGraphicsPipeline\n";
        return false;
    }
    if (!createPipePipeline()) {
        VOX_LOGE("render") << "recreateSwapchain failed: createPipePipeline\n";
        return false;
    }
    if (!createAoPipelines()) {
        VOX_LOGE("render") << "recreateSwapchain failed: createAoPipelines\n";
        return false;
    }
    if (m_imguiInitialized) {
        ImGui_ImplVulkan_SetMinImageCount(std::max<uint32_t>(2u, static_cast<uint32_t>(m_swapchainImages.size())));
    }
    VOX_LOGI("render") << "recreateSwapchain complete\n";
    return true;
}


void Renderer::destroySwapchain() {
    destroyHdrResolveTargets();
    destroyMsaaColorTargets();
    destroyDepthTargets();
    destroyAoTargets();
    const uint32_t orphanedFrameArenaImages = m_frameArena.liveImageCount();
    if (orphanedFrameArenaImages > 0) {
        VOX_LOGI("render") << "destroySwapchain: cleaning up "
            << orphanedFrameArenaImages
            << " orphaned FrameArena image(s)\n";
        m_frameArena.destroyAllImages();
    }
    m_aoExtent = VkExtent2D{};

    for (VkSemaphore semaphore : m_renderFinishedSemaphores) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, semaphore, nullptr);
        }
    }
    m_renderFinishedSemaphores.clear();

    for (VkImageView imageView : m_swapchainImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_swapchainImageViews.clear();
    m_swapchainImages.clear();
    m_swapchainImageInitialized.clear();
    m_swapchainImageTimelineValues.clear();

    if (m_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
}


void Renderer::destroyTransferResources() {
    m_transferCommandBuffer = VK_NULL_HANDLE;
    if (m_transferCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_transferCommandPool, nullptr);
        m_transferCommandPool = VK_NULL_HANDLE;
    }
}


void Renderer::destroyPreviewBuffers() {
    if (m_previewIndexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_previewIndexBufferHandle);
        m_previewIndexBufferHandle = kInvalidBufferHandle;
    }
    if (m_previewVertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_previewVertexBufferHandle);
        m_previewVertexBufferHandle = kInvalidBufferHandle;
    }
    m_previewIndexCount = 0;
}


void Renderer::destroyMagicaBuffers() {
    for (MagicaMeshDraw& draw : m_magicaMeshDraws) {
        if (draw.indexBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(draw.indexBufferHandle);
            draw.indexBufferHandle = kInvalidBufferHandle;
        }
        if (draw.vertexBufferHandle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(draw.vertexBufferHandle);
            draw.vertexBufferHandle = kInvalidBufferHandle;
        }
        draw.indexCount = 0;
        draw.offsetX = 0.0f;
        draw.offsetY = 0.0f;
        draw.offsetZ = 0.0f;
    }
    m_magicaMeshDraws.clear();
}


void Renderer::destroyPipeBuffers() {
    if (m_grassBillboardIndexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_grassBillboardIndexBufferHandle);
        m_grassBillboardIndexBufferHandle = kInvalidBufferHandle;
    }
    if (m_grassBillboardVertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_grassBillboardVertexBufferHandle);
        m_grassBillboardVertexBufferHandle = kInvalidBufferHandle;
    }
    m_grassBillboardIndexCount = 0;

    if (m_transportIndexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_transportIndexBufferHandle);
        m_transportIndexBufferHandle = kInvalidBufferHandle;
    }
    if (m_transportVertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_transportVertexBufferHandle);
        m_transportVertexBufferHandle = kInvalidBufferHandle;
    }
    m_transportIndexCount = 0;

    if (m_pipeIndexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_pipeIndexBufferHandle);
        m_pipeIndexBufferHandle = kInvalidBufferHandle;
    }
    if (m_pipeVertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_pipeVertexBufferHandle);
        m_pipeVertexBufferHandle = kInvalidBufferHandle;
    }
    m_pipeIndexCount = 0;
}


void Renderer::destroyEnvironmentResources() {
    destroyDiffuseTextureResources();
}


void Renderer::destroyDiffuseTextureResources() {
    if (m_diffuseTexturePlantSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_diffuseTexturePlantSampler, nullptr);
        m_diffuseTexturePlantSampler = VK_NULL_HANDLE;
    }
    if (m_diffuseTextureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_diffuseTextureSampler, nullptr);
        m_diffuseTextureSampler = VK_NULL_HANDLE;
    }
    if (m_diffuseTextureImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_diffuseTextureImageView, nullptr);
        m_diffuseTextureImageView = VK_NULL_HANDLE;
    }
    if (m_diffuseTextureImage != VK_NULL_HANDLE) {
        if (m_vmaAllocator != VK_NULL_HANDLE && m_diffuseTextureAllocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_diffuseTextureImage, m_diffuseTextureAllocation);
            m_diffuseTextureAllocation = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, m_diffuseTextureImage, nullptr);
        }
        m_diffuseTextureImage = VK_NULL_HANDLE;
    }
    if (m_diffuseTextureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_diffuseTextureMemory, nullptr);
        m_diffuseTextureMemory = VK_NULL_HANDLE;
    }
    m_diffuseTextureAllocation = VK_NULL_HANDLE;
}


void Renderer::destroyShadowResources() {
    if (m_shadowDepthSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_shadowDepthSampler, nullptr);
        m_shadowDepthSampler = VK_NULL_HANDLE;
    }
    if (m_shadowDepthImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_shadowDepthImageView, nullptr);
        m_shadowDepthImageView = VK_NULL_HANDLE;
    }
    if (m_shadowDepthImage != VK_NULL_HANDLE) {
        if (m_vmaAllocator != VK_NULL_HANDLE && m_shadowDepthAllocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_shadowDepthImage, m_shadowDepthAllocation);
            m_shadowDepthAllocation = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, m_shadowDepthImage, nullptr);
        }
        m_shadowDepthImage = VK_NULL_HANDLE;
    }
    if (m_shadowDepthMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_shadowDepthMemory, nullptr);
        m_shadowDepthMemory = VK_NULL_HANDLE;
    }
    m_shadowDepthInitialized = false;
}


void Renderer::destroyVoxelGiResources() {
    m_pipelineManager.destroyVoxelGiPipelines(m_device);
    m_descriptorManager.destroyVoxelGi(m_device);

    if (m_voxelGiOccupancySampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_voxelGiOccupancySampler, nullptr);
        m_voxelGiOccupancySampler = VK_NULL_HANDLE;
    }
    if (m_voxelGiSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_voxelGiSampler, nullptr);
        m_voxelGiSampler = VK_NULL_HANDLE;
    }
    if (m_voxelGiOccupancyImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_voxelGiOccupancyImageView, nullptr);
        m_voxelGiOccupancyImageView = VK_NULL_HANDLE;
    }
    if (m_voxelGiSkyExposureImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_voxelGiSkyExposureImageView, nullptr);
        m_voxelGiSkyExposureImageView = VK_NULL_HANDLE;
    }
    for (std::size_t faceIndex = 0; faceIndex < m_voxelGiSurfaceFaceImageViews.size(); ++faceIndex) {
        if (m_voxelGiSurfaceFaceImageViews[faceIndex] != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, m_voxelGiSurfaceFaceImageViews[faceIndex], nullptr);
            m_voxelGiSurfaceFaceImageViews[faceIndex] = VK_NULL_HANDLE;
        }
    }
    if (m_voxelGiOccupancyImage != VK_NULL_HANDLE) {
        if (m_vmaAllocator != VK_NULL_HANDLE && m_voxelGiOccupancyAllocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_voxelGiOccupancyImage, m_voxelGiOccupancyAllocation);
            m_voxelGiOccupancyAllocation = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, m_voxelGiOccupancyImage, nullptr);
        }
        m_voxelGiOccupancyImage = VK_NULL_HANDLE;
    }
    if (m_voxelGiSkyExposureImage != VK_NULL_HANDLE) {
        if (m_vmaAllocator != VK_NULL_HANDLE && m_voxelGiSkyExposureAllocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_voxelGiSkyExposureImage, m_voxelGiSkyExposureAllocation);
            m_voxelGiSkyExposureAllocation = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, m_voxelGiSkyExposureImage, nullptr);
        }
        m_voxelGiSkyExposureImage = VK_NULL_HANDLE;
    }
    for (std::size_t faceIndex = 0; faceIndex < m_voxelGiSurfaceFaceImages.size(); ++faceIndex) {
        if (m_voxelGiSurfaceFaceImages[faceIndex] != VK_NULL_HANDLE) {
            if (m_vmaAllocator != VK_NULL_HANDLE && m_voxelGiSurfaceFaceAllocations[faceIndex] != VK_NULL_HANDLE) {
                vmaDestroyImage(
                    m_vmaAllocator,
                    m_voxelGiSurfaceFaceImages[faceIndex],
                    m_voxelGiSurfaceFaceAllocations[faceIndex]
                );
                m_voxelGiSurfaceFaceAllocations[faceIndex] = VK_NULL_HANDLE;
            } else {
                vkDestroyImage(m_device, m_voxelGiSurfaceFaceImages[faceIndex], nullptr);
            }
            m_voxelGiSurfaceFaceImages[faceIndex] = VK_NULL_HANDLE;
        }
    }
    if (m_voxelGiOccupancyMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_voxelGiOccupancyMemory, nullptr);
        m_voxelGiOccupancyMemory = VK_NULL_HANDLE;
    }
    if (m_voxelGiSkyExposureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_voxelGiSkyExposureMemory, nullptr);
        m_voxelGiSkyExposureMemory = VK_NULL_HANDLE;
    }
    for (std::size_t faceIndex = 0; faceIndex < m_voxelGiSurfaceFaceMemories.size(); ++faceIndex) {
        if (m_voxelGiSurfaceFaceMemories[faceIndex] != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, m_voxelGiSurfaceFaceMemories[faceIndex], nullptr);
            m_voxelGiSurfaceFaceMemories[faceIndex] = VK_NULL_HANDLE;
        }
    }
    for (std::size_t volumeIndex = 0; volumeIndex < m_voxelGiImageViews.size(); ++volumeIndex) {
        if (m_voxelGiImageViews[volumeIndex] != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, m_voxelGiImageViews[volumeIndex], nullptr);
            m_voxelGiImageViews[volumeIndex] = VK_NULL_HANDLE;
        }
        if (m_voxelGiImages[volumeIndex] != VK_NULL_HANDLE) {
            if (m_vmaAllocator != VK_NULL_HANDLE && m_voxelGiImageAllocations[volumeIndex] != VK_NULL_HANDLE) {
                vmaDestroyImage(m_vmaAllocator, m_voxelGiImages[volumeIndex], m_voxelGiImageAllocations[volumeIndex]);
                m_voxelGiImageAllocations[volumeIndex] = VK_NULL_HANDLE;
            } else {
                vkDestroyImage(m_device, m_voxelGiImages[volumeIndex], nullptr);
            }
            m_voxelGiImages[volumeIndex] = VK_NULL_HANDLE;
        }
        if (m_voxelGiImageMemories[volumeIndex] != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, m_voxelGiImageMemories[volumeIndex], nullptr);
            m_voxelGiImageMemories[volumeIndex] = VK_NULL_HANDLE;
        }
    }
    m_voxelGiImageAllocations = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    m_voxelGiSurfaceFaceAllocations.fill(VK_NULL_HANDLE);
    m_voxelGiSkyExposureAllocation = VK_NULL_HANDLE;
    m_voxelGiOccupancyAllocation = VK_NULL_HANDLE;
    m_voxelGiInitialized = false;
    m_voxelGiSkyExposureInitialized = false;
    m_voxelGiOccupancyInitialized = false;
    m_voxelGiComputeAvailable = false;
    m_voxelGiWorldDirty = true;
    m_voxelGiHasPreviousFrameState = false;
    m_voxelGiPreviousBounceStrength = 0.0f;
    m_voxelGiPreviousDiffusionSoftness = 0.0f;
}


void Renderer::destroyChunkBuffers() {
    for (ChunkDrawRange& drawRange : m_chunkDrawRanges) {
        drawRange.firstIndex = 0;
        drawRange.vertexOffset = 0;
        drawRange.indexCount = 0;
    }

    for (const DeferredBufferRelease& release : m_deferredBufferReleases) {
        if (release.handle != kInvalidBufferHandle) {
            m_bufferAllocator.destroyBuffer(release.handle);
        }
    }
    m_deferredBufferReleases.clear();

    m_chunkDrawRanges.clear();
    m_chunkLodMeshCache.clear();
    m_chunkGrassInstanceCache.clear();
    m_chunkLodMeshCacheValid = false;
    if (m_grassBillboardInstanceBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_grassBillboardInstanceBufferHandle);
        m_grassBillboardInstanceBufferHandle = kInvalidBufferHandle;
    }
    m_grassBillboardInstanceCount = 0;
    m_bufferAllocator.destroyBuffer(m_chunkVertexBufferHandle);
    m_chunkVertexBufferHandle = kInvalidBufferHandle;
    m_bufferAllocator.destroyBuffer(m_chunkIndexBufferHandle);
    m_chunkIndexBufferHandle = kInvalidBufferHandle;
    m_pendingTransferTimelineValue = 0;
    m_currentChunkReadyTimelineValue = 0;
    m_transferCommandBufferInFlightValue = 0;
}


void Renderer::destroyPipeline() {
    m_pipelineManager.destroyMainPipelines(m_device);
}


void Renderer::shutdown() {
    VOX_LOGI("render") << "shutdown begin\n";
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    if (m_device != VK_NULL_HANDLE) {
        destroyImGuiResources();
        destroyFrameResources();
        destroyGpuTimestampResources();
        destroyTransferResources();
        if (m_renderTimelineSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, m_renderTimelineSemaphore, nullptr);
            m_renderTimelineSemaphore = VK_NULL_HANDLE;
        }
        destroyPipeBuffers();
        destroyPreviewBuffers();
        destroyMagicaBuffers();
        destroyEnvironmentResources();
        destroyShadowResources();
        destroyVoxelGiResources();
        destroyAutoExposureResources();
        destroySunShaftResources();
        destroyChunkBuffers();
        destroyPipeline();
        m_descriptorManager.destroyMain(m_device);
        destroySwapchain();
        const uint32_t liveFrameArenaImagesBeforeShutdown = m_frameArena.liveImageCount();
        if (liveFrameArenaImagesBeforeShutdown > 0) {
            VOX_LOGI("render") << "shutdown: forcing cleanup of "
                << liveFrameArenaImagesBeforeShutdown
                << " remaining FrameArena image(s) before allocator shutdown\n";
            m_frameArena.destroyAllImages();
        }
        m_frameArena.shutdown(&m_bufferAllocator);
        m_bufferAllocator.shutdown();

        uint32_t rendererOwnedLiveImages = 0;
        auto logLiveImage = [&](const char* name, VkImage image) {
            if (image == VK_NULL_HANDLE) {
                return;
            }
            ++rendererOwnedLiveImages;
            VOX_LOGI("render") << "shutdown leak check: live image '" << name
                << "' handle=0x" << std::hex
                << static_cast<unsigned long long>(vkHandleToUint64(image))
                << std::dec << "\n";
        };
        logLiveImage("diffuse.albedo.image", m_diffuseTextureImage);
        logLiveImage("shadow.atlas.image", m_shadowDepthImage);
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_voxelGiImages.size()); ++i) {
            logLiveImage(("voxelGi.radiance.image[" + std::to_string(i) + "]").c_str(), m_voxelGiImages[i]);
        }
        for (std::size_t faceIndex = 0; faceIndex < m_voxelGiSurfaceFaceImages.size(); ++faceIndex) {
            logLiveImage(
                ("voxelGi.surfaceFace.image[" + std::to_string(faceIndex) + "]").c_str(),
                m_voxelGiSurfaceFaceImages[faceIndex]
            );
        }
        logLiveImage("voxelGi.skyExposure.image", m_voxelGiSkyExposureImage);
        logLiveImage("voxelGi.occupancy.image", m_voxelGiOccupancyImage);
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_depthImages.size()); ++i) {
            logLiveImage(("depth.msaa.image[" + std::to_string(i) + "]").c_str(), m_depthImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_msaaColorImages.size()); ++i) {
            logLiveImage(("hdr.msaaColor.image[" + std::to_string(i) + "]").c_str(), m_msaaColorImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_hdrResolveImages.size()); ++i) {
            logLiveImage(("hdr.resolve.image[" + std::to_string(i) + "]").c_str(), m_hdrResolveImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_normalDepthImages.size()); ++i) {
            logLiveImage(("ao.normalDepth.image[" + std::to_string(i) + "]").c_str(), m_normalDepthImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_aoDepthImages.size()); ++i) {
            logLiveImage(("ao.depth.image[" + std::to_string(i) + "]").c_str(), m_aoDepthImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_ssaoRawImages.size()); ++i) {
            logLiveImage(("ao.ssaoRaw.image[" + std::to_string(i) + "]").c_str(), m_ssaoRawImages[i]);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(m_ssaoBlurImages.size()); ++i) {
            logLiveImage(("ao.ssaoBlur.image[" + std::to_string(i) + "]").c_str(), m_ssaoBlurImages[i]);
        }
        if (rendererOwnedLiveImages == 0) {
            VOX_LOGI("render") << "shutdown leak check: no renderer-owned live VkImage handles\n";
        }

        if (m_vmaAllocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(m_vmaAllocator);
            m_vmaAllocator = VK_NULL_HANDLE;
        }

        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }

    if (m_surface != VK_NULL_HANDLE && m_instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }

    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }

    m_physicalDevice = VK_NULL_HANDLE;
    m_debugUtilsEnabled = false;
    m_setDebugUtilsObjectName = nullptr;
    m_cmdBeginDebugUtilsLabel = nullptr;
    m_cmdEndDebugUtilsLabel = nullptr;
    m_cmdInsertDebugUtilsLabel = nullptr;
    m_graphicsQueue = VK_NULL_HANDLE;
    m_transferQueue = VK_NULL_HANDLE;
    m_graphicsQueueFamilyIndex = 0;
    m_graphicsQueueIndex = 0;
    m_transferQueueFamilyIndex = 0;
    m_transferQueueIndex = 0;
    m_aoExtent = VkExtent2D{};
    m_depthFormat = VK_FORMAT_UNDEFINED;
    m_shadowDepthFormat = VK_FORMAT_UNDEFINED;
    m_hdrColorFormat = VK_FORMAT_UNDEFINED;
    m_normalDepthFormat = VK_FORMAT_UNDEFINED;
    m_ssaoFormat = VK_FORMAT_UNDEFINED;
    m_voxelGiFormat = VK_FORMAT_UNDEFINED;
    m_voxelGiOccupancyFormat = VK_FORMAT_UNDEFINED;
    m_voxelGiWorldDirty = true;
    m_voxelGiHasPreviousFrameState = false;
    m_voxelGiPreviousGridOrigin = {0.0f, 0.0f, 0.0f};
    m_voxelGiPreviousSunDirection = {0.0f, 0.0f, 0.0f};
    m_voxelGiPreviousSunColor = {0.0f, 0.0f, 0.0f};
    m_voxelGiPreviousShIrradiance = {};
    m_voxelGiPreviousBounceStrength = 0.0f;
    m_voxelGiPreviousDiffusionSoftness = 0.0f;
    m_autoExposureHistogramBufferHandle = kInvalidBufferHandle;
    m_autoExposureStateBufferHandle = kInvalidBufferHandle;
    m_autoExposureComputeAvailable = false;
    m_autoExposureHistoryValid = false;
    m_sunShaftComputeAvailable = false;
    m_sunShaftShaderAvailable = false;
    m_supportsWireframePreview = false;
    m_supportsSamplerAnisotropy = false;
    m_supportsMultiDrawIndirect = false;
    m_supportsDisplayTiming = false;
    m_hasDisplayTimingExtension = false;
    m_enableDisplayTiming = false;
    m_chunkMeshingOptions = world::MeshingOptions{};
    m_chunkMeshRebuildRequested = false;
    m_pendingChunkRemeshIndices.clear();
    m_gpuTimestampsSupported = false;
    m_gpuTimestampPeriodNs = 0.0f;
    m_gpuTimestampQueryPools.fill(VK_NULL_HANDLE);
    m_debugGpuFrameTimeMs = 0.0f;
    m_debugGpuShadowTimeMs = 0.0f;
    m_debugGpuGiInjectTimeMs = 0.0f;
    m_debugGpuGiPropagateTimeMs = 0.0f;
    m_debugGpuAutoExposureTimeMs = 0.0f;
    m_debugGpuSunShaftTimeMs = 0.0f;
    m_debugGpuPrepassTimeMs = 0.0f;
    m_debugGpuSsaoTimeMs = 0.0f;
    m_debugGpuSsaoBlurTimeMs = 0.0f;
    m_debugGpuMainTimeMs = 0.0f;
    m_debugGpuPostTimeMs = 0.0f;
    m_debugDisplayRefreshMs = 0.0f;
    m_debugDisplayPresentMarginMs = 0.0f;
    m_debugDisplayActualEarliestDeltaMs = 0.0f;
    m_debugDisplayTimingSampleCount = 0;
    m_debugChunkMeshVertexCount = 0;
    m_debugChunkMeshIndexCount = 0;
    m_debugChunkLastRemeshedChunkCount = 0;
    m_debugChunkLastRemeshActiveVertexCount = 0;
    m_debugChunkLastRemeshActiveIndexCount = 0;
    m_debugChunkLastRemeshNaiveVertexCount = 0;
    m_debugChunkLastRemeshNaiveIndexCount = 0;
    m_debugChunkLastRemeshReductionPercent = 0.0f;
    m_debugChunkLastRemeshMs = 0.0f;
    m_debugChunkLastFullRemeshMs = 0.0f;
    m_debugEnableSpatialQueries = true;
    m_debugClipmapConfig = world::ClipmapConfig{};
    m_debugSpatialQueriesUsed = false;
    m_debugSpatialQueryStats = {};
    m_debugSpatialVisibleChunkCount = 0;
    m_debugCpuFrameTotalMsHistory.fill(0.0f);
    m_debugCpuFrameWorkMsHistory.fill(0.0f);
    m_debugCpuFrameEwmaMsHistory.fill(0.0f);
    m_debugCpuFrameTimingMsHistoryWrite = 0;
    m_debugCpuFrameTimingMsHistoryCount = 0;
    m_debugCpuFrameWorkMs = 0.0f;
    m_debugCpuFrameEwmaMs = 0.0f;
    m_debugCpuFrameEwmaInitialized = false;
    m_debugGpuFrameTimingMsHistory.fill(0.0f);
    m_debugGpuFrameTimingMsHistoryWrite = 0;
    m_debugGpuFrameTimingMsHistoryCount = 0;
    m_frameTimelineValues.fill(0);
    m_pendingTransferTimelineValue = 0;
    m_currentChunkReadyTimelineValue = 0;
    m_transferCommandBufferInFlightValue = 0;
    m_lastGraphicsTimelineValue = 0;
    m_nextTimelineValue = 1;
    m_nextDisplayTimingPresentId = 1;
    m_lastSubmittedDisplayTimingPresentId = 0;
    m_lastPresentedDisplayTimingPresentId = 0;
    m_getRefreshCycleDurationGoogle = nullptr;
    m_getPastPresentationTimingGoogle = nullptr;
    m_currentFrame = 0;
    m_window = nullptr;
    VOX_LOGI("render") << "shutdown complete\n";
}

} // namespace render
