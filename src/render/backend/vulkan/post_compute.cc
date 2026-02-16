#include "render/backend/vulkan/renderer_backend.h"

#include "core/log.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

namespace voxelsprout::render {

namespace {

constexpr uint32_t kAutoExposureHistogramBins = 64u;

template <typename VkHandleT>
uint64_t vkHandleToUint64(VkHandleT handle) {
    if constexpr (std::is_pointer_v<VkHandleT>) {
        return reinterpret_cast<uint64_t>(handle);
    } else {
        return static_cast<uint64_t>(handle);
    }
}

const char* vkResultName(VkResult result) {
    switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    default: return "VK_RESULT_UNKNOWN";
    }
}

void logVkFailure(const char* context, VkResult result) {
    VOX_LOGE("render") << context << " failed: "
                       << vkResultName(result) << " (" << static_cast<int>(result) << ")";
}

std::optional<std::vector<std::uint8_t>> readBinaryFile(const char* filePath) {
    if (filePath == nullptr) {
        return std::nullopt;
    }

    const std::filesystem::path path(filePath);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return std::nullopt;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return std::nullopt;
    }
    file.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> data(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        return std::nullopt;
    }
    return data;
}

bool createShaderModuleFromFile(
    VkDevice device,
    const char* filePath,
    const char* debugName,
    VkShaderModule& outShaderModule
) {
    outShaderModule = VK_NULL_HANDLE;

    const std::optional<std::vector<std::uint8_t>> shaderFileData = readBinaryFile(filePath);
    if (!shaderFileData.has_value()) {
        VOX_LOGE("render") << "missing shader file for " << debugName << ": "
                           << (filePath != nullptr ? filePath : "<null>") << "\n";
        return false;
    }
    if ((shaderFileData->size() % sizeof(std::uint32_t)) != 0) {
        VOX_LOGE("render") << "invalid SPIR-V byte size for " << debugName << ": " << filePath << "\n";
        return false;
    }

    const std::uint32_t* code = reinterpret_cast<const std::uint32_t*>(shaderFileData->data());
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = shaderFileData->size();
    createInfo.pCode = code;

    const VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &outShaderModule);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateShaderModule(fileOrFallback)", result);
        return false;
    }
    return true;
}

void destroyShaderModules(VkDevice device, std::span<const VkShaderModule> shaderModules) {
    for (const VkShaderModule shaderModule : shaderModules) {
        if (shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shaderModule, nullptr);
        }
    }
}

struct alignas(16) AutoExposureHistogramPushConstants {
    uint32_t width = 1u;
    uint32_t height = 1u;
    uint32_t totalPixels = 1u;
    uint32_t binCount = kAutoExposureHistogramBins;
    float minLogLuminance = -10.0f;
    float maxLogLuminance = 4.0f;
    float sourceMipLevel = 0.0f;
    float _pad1 = 0.0f;
};

struct alignas(16) AutoExposureUpdatePushConstants {
    uint32_t totalPixels = 1u;
    uint32_t binCount = kAutoExposureHistogramBins;
    uint32_t resetHistory = 1u;
    uint32_t _pad0 = 0u;
    float minLogLuminance = -10.0f;
    float maxLogLuminance = 4.0f;
    float lowPercentile = 0.5f;
    float highPercentile = 0.98f;
    float keyValue = 0.18f;
    float minExposure = 0.25f;
    float maxExposure = 2.2f;
    float adaptUpRate = 3.0f;
    float adaptDownRate = 1.4f;
    float deltaTimeSeconds = 0.016f;
    float _pad1 = 0.0f;
    float _pad2 = 0.0f;
};

struct alignas(16) SunShaftPushConstants {
    uint32_t width = 1u;
    uint32_t height = 1u;
    uint32_t sampleCount = 10u;
    uint32_t _pad0 = 0u;
};

} // namespace

bool RendererBackend::createAutoExposureResources() {
    const float initialExposure = std::clamp(m_skyDebugSettings.manualExposure, 0.05f, 8.0f);
    if (m_autoExposureStateBufferHandle == kInvalidBufferHandle) {
        const std::array<float, 4> initialState = {initialExposure, initialExposure, 1.0f, 0.0f};
        BufferCreateDesc exposureStateBufferDesc{};
        exposureStateBufferDesc.size = sizeof(initialState);
        exposureStateBufferDesc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        exposureStateBufferDesc.memoryProperties =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        exposureStateBufferDesc.initialData = initialState.data();
        m_autoExposureStateBufferHandle = m_bufferAllocator.createBuffer(exposureStateBufferDesc);
        if (m_autoExposureStateBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "failed to create auto exposure state buffer";
            destroyAutoExposureResources();
            return false;
        }
        const VkBuffer autoExposureStateBuffer = m_bufferAllocator.getBuffer(m_autoExposureStateBufferHandle);
        setObjectName(
            VK_OBJECT_TYPE_BUFFER,
            vkHandleToUint64(autoExposureStateBuffer),
            "autoExposure.stateBuffer"
        );
    }

    if (m_autoExposureHistogramBufferHandle == kInvalidBufferHandle) {
        BufferCreateDesc histogramBufferDesc{};
        histogramBufferDesc.size = static_cast<VkDeviceSize>(kAutoExposureHistogramBins * sizeof(uint32_t));
        histogramBufferDesc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        histogramBufferDesc.memoryProperties =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        m_autoExposureHistogramBufferHandle = m_bufferAllocator.createBuffer(histogramBufferDesc);
        if (m_autoExposureHistogramBufferHandle == kInvalidBufferHandle) {
            VOX_LOGE("render") << "failed to create auto exposure histogram buffer";
            destroyAutoExposureResources();
            return false;
        }
        const VkBuffer autoExposureHistogramBuffer = m_bufferAllocator.getBuffer(m_autoExposureHistogramBufferHandle);
        setObjectName(
            VK_OBJECT_TYPE_BUFFER,
            vkHandleToUint64(autoExposureHistogramBuffer),
            "autoExposure.histogramBuffer"
        );
    }

    constexpr const char* kHistogramShaderPath = "../src/render/shaders/auto_exposure_histogram.comp.slang.spv";
    constexpr const char* kUpdateShaderPath = "../src/render/shaders/auto_exposure_update.comp.slang.spv";
    const bool hasHistogramShader = readBinaryFile(kHistogramShaderPath).has_value();
    const bool hasUpdateShader = readBinaryFile(kUpdateShaderPath).has_value();
    if (!hasHistogramShader || !hasUpdateShader) {
        VOX_LOGI("render")
            << "auto exposure compute shaders not found; using manual exposure fallback (expected: "
            << kHistogramShaderPath << ", " << kUpdateShaderPath << ")\n";
        m_autoExposureComputeAvailable = false;
        m_autoExposureHistoryValid = false;
        return true;
    }

    if (m_autoExposureDescriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding hdrSceneBinding{};
        hdrSceneBinding.binding = 0;
        hdrSceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        hdrSceneBinding.descriptorCount = 1;
        hdrSceneBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding histogramBinding{};
        histogramBinding.binding = 1;
        histogramBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        histogramBinding.descriptorCount = 1;
        histogramBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding exposureStateBinding{};
        exposureStateBinding.binding = 2;
        exposureStateBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        exposureStateBinding.descriptorCount = 1;
        exposureStateBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        const std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            hdrSceneBinding,
            histogramBinding,
            exposureStateBinding
        };

        if (!createDescriptorSetLayout(
                bindings,
                m_autoExposureDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(autoExposure)",
                "renderer.descriptorSetLayout.autoExposure"
            )) {
            destroyAutoExposureResources();
            return false;
        }
    }

    if (m_autoExposureDescriptorPool == VK_NULL_HANDLE) {
        const std::array<VkDescriptorPoolSize, 2> poolSizes = {
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kMaxFramesInFlight}
        };
        if (!createDescriptorPool(
                poolSizes,
                kMaxFramesInFlight,
                m_autoExposureDescriptorPool,
                "vkCreateDescriptorPool(autoExposure)",
                "renderer.descriptorPool.autoExposure"
            )) {
            destroyAutoExposureResources();
            return false;
        }
    }

    if (!allocatePerFrameDescriptorSets(
            m_autoExposureDescriptorPool,
            m_autoExposureDescriptorSetLayout,
            std::span<VkDescriptorSet>(m_autoExposureDescriptorSets),
            "vkAllocateDescriptorSets(autoExposure)",
            "renderer.descriptorSet.autoExposure.frame"
        )) {
        destroyAutoExposureResources();
        return false;
    }

    std::array<VkShaderModule, 2> shaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& histogramShaderModule = shaderModules[0];
    VkShaderModule& updateShaderModule = shaderModules[1];
    if (!createShaderModuleFromFile(
            m_device,
            kHistogramShaderPath,
            "auto_exposure_histogram.comp",
            histogramShaderModule
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyAutoExposureResources();
        return false;
    }
    if (!createShaderModuleFromFile(
            m_device,
            kUpdateShaderPath,
            "auto_exposure_update.comp",
            updateShaderModule
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyAutoExposureResources();
        return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = static_cast<uint32_t>(std::max(
        sizeof(AutoExposureHistogramPushConstants),
        sizeof(AutoExposureUpdatePushConstants)
    ));

    const std::array<VkPushConstantRange, 1> pushConstantRanges = {pushConstantRange};
    if (!createComputePipelineLayout(
            m_autoExposureDescriptorSetLayout,
            pushConstantRanges,
            m_autoExposurePipelineLayout,
            "vkCreatePipelineLayout(autoExposure)",
            "renderer.pipelineLayout.autoExposure"
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyAutoExposureResources();
        return false;
    }

    if (!createComputePipeline(
            m_autoExposurePipelineLayout,
            histogramShaderModule,
            m_autoExposureHistogramPipeline,
            "vkCreateComputePipelines(autoExposureHistogram)",
            "pipeline.autoExposure.histogram"
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyAutoExposureResources();
        return false;
    }

    if (!createComputePipeline(
            m_autoExposurePipelineLayout,
            updateShaderModule,
            m_autoExposureUpdatePipeline,
            "vkCreateComputePipelines(autoExposureUpdate)",
            "pipeline.autoExposure.update"
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroyAutoExposureResources();
        return false;
    }

    destroyShaderModules(m_device, shaderModules);

    m_autoExposureComputeAvailable = true;
    m_autoExposureHistoryValid = false;
    VOX_LOGI("render")
        << "auto exposure resources ready: bins=" << kAutoExposureHistogramBins
        << ", compute=enabled\n";
    return true;
}

bool RendererBackend::createSunShaftResources() {
    constexpr const char* kSunShaftShaderPath = "../src/render/shaders/sun_shafts.comp.slang.spv";
    const bool hasSunShaftShader = readBinaryFile(kSunShaftShaderPath).has_value();
    if (!hasSunShaftShader) {
        VOX_LOGI("render")
            << "sun shafts compute shader not found; disabling dedicated pass (expected: "
            << kSunShaftShaderPath << ")\n";
        m_sunShaftShaderAvailable = false;
        m_sunShaftComputeAvailable = false;
        return true;
    }

    m_sunShaftShaderAvailable = true;

    if (m_sunShaftDescriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding cameraBinding{};
        cameraBinding.binding = 0;
        cameraBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        cameraBinding.descriptorCount = 1;
        cameraBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding normalDepthBinding{};
        normalDepthBinding.binding = 1;
        normalDepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        normalDepthBinding.descriptorCount = 1;
        normalDepthBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding shadowBinding{};
        shadowBinding.binding = 2;
        shadowBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadowBinding.descriptorCount = 1;
        shadowBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding outputBinding{};
        outputBinding.binding = 3;
        outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        outputBinding.descriptorCount = 1;
        outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        const std::array<VkDescriptorSetLayoutBinding, 4> bindings = {
            cameraBinding,
            normalDepthBinding,
            shadowBinding,
            outputBinding
        };

        if (!createDescriptorSetLayout(
                bindings,
                m_sunShaftDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(sunShaft)",
                "renderer.descriptorSetLayout.sunShaft"
            )) {
            destroySunShaftResources();
            return false;
        }
    }

    if (m_sunShaftDescriptorPool == VK_NULL_HANDLE) {
        const std::array<VkDescriptorPoolSize, 3> poolSizes = {
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxFramesInFlight}
        };
        if (!createDescriptorPool(
                poolSizes,
                kMaxFramesInFlight,
                m_sunShaftDescriptorPool,
                "vkCreateDescriptorPool(sunShaft)",
                "renderer.descriptorPool.sunShaft"
            )) {
            destroySunShaftResources();
            return false;
        }
    }

    if (!allocatePerFrameDescriptorSets(
            m_sunShaftDescriptorPool,
            m_sunShaftDescriptorSetLayout,
            std::span<VkDescriptorSet>(m_sunShaftDescriptorSets),
            "vkAllocateDescriptorSets(sunShaft)",
            "renderer.descriptorSet.sunShaft.frame"
        )) {
        destroySunShaftResources();
        return false;
    }

    VkShaderModule sunShaftShaderModule = VK_NULL_HANDLE;
    if (!createShaderModuleFromFile(
            m_device,
            kSunShaftShaderPath,
            "sun_shafts.comp",
            sunShaftShaderModule
        )) {
        destroySunShaftResources();
        return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(SunShaftPushConstants);

    const std::array<VkPushConstantRange, 1> pushConstantRanges = {pushConstantRange};
    if (!createComputePipelineLayout(
            m_sunShaftDescriptorSetLayout,
            pushConstantRanges,
            m_sunShaftPipelineLayout,
            "vkCreatePipelineLayout(sunShaft)",
            "renderer.pipelineLayout.sunShaft"
        )) {
        vkDestroyShaderModule(m_device, sunShaftShaderModule, nullptr);
        destroySunShaftResources();
        return false;
    }

    if (!createComputePipeline(
            m_sunShaftPipelineLayout,
            sunShaftShaderModule,
            m_sunShaftPipeline,
            "vkCreateComputePipelines(sunShaft)",
            "pipeline.sunShaft.compute"
        )) {
        vkDestroyShaderModule(m_device, sunShaftShaderModule, nullptr);
        destroySunShaftResources();
        return false;
    }
    vkDestroyShaderModule(m_device, sunShaftShaderModule, nullptr);

    m_sunShaftComputeAvailable = true;
    VOX_LOGI("render") << "sun shafts compute resources ready\n";
    return true;
}

void RendererBackend::destroyAutoExposureResources() {
    if (m_autoExposureHistogramPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_autoExposureHistogramPipeline, nullptr);
        m_autoExposureHistogramPipeline = VK_NULL_HANDLE;
    }
    if (m_autoExposureUpdatePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_autoExposureUpdatePipeline, nullptr);
        m_autoExposureUpdatePipeline = VK_NULL_HANDLE;
    }
    if (m_autoExposurePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_autoExposurePipelineLayout, nullptr);
        m_autoExposurePipelineLayout = VK_NULL_HANDLE;
    }
    if (m_autoExposureDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_autoExposureDescriptorPool, nullptr);
        m_autoExposureDescriptorPool = VK_NULL_HANDLE;
    }
    if (m_autoExposureDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_autoExposureDescriptorSetLayout, nullptr);
        m_autoExposureDescriptorSetLayout = VK_NULL_HANDLE;
    }
    m_autoExposureDescriptorSets.fill(VK_NULL_HANDLE);

    if (m_autoExposureHistogramBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_autoExposureHistogramBufferHandle);
        m_autoExposureHistogramBufferHandle = kInvalidBufferHandle;
    }
    if (m_autoExposureStateBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_autoExposureStateBufferHandle);
        m_autoExposureStateBufferHandle = kInvalidBufferHandle;
    }
    m_autoExposureComputeAvailable = false;
    m_autoExposureHistoryValid = false;
}

void RendererBackend::destroySunShaftResources() {
    if (m_sunShaftPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_sunShaftPipeline, nullptr);
        m_sunShaftPipeline = VK_NULL_HANDLE;
    }
    if (m_sunShaftPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_sunShaftPipelineLayout, nullptr);
        m_sunShaftPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_sunShaftDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_sunShaftDescriptorPool, nullptr);
        m_sunShaftDescriptorPool = VK_NULL_HANDLE;
    }
    if (m_sunShaftDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_sunShaftDescriptorSetLayout, nullptr);
        m_sunShaftDescriptorSetLayout = VK_NULL_HANDLE;
    }
    m_sunShaftDescriptorSets.fill(VK_NULL_HANDLE);
    m_sunShaftComputeAvailable = false;
    m_sunShaftShaderAvailable = false;
}

} // namespace voxelsprout::render
