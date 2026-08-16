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

namespace odai::render {

namespace {

// Local copy of the shared single-image barrier helper. renderer_shared.h owns
// the canonical one, but this file cannot include it -- their anonymous-
// namespace helpers collide -- and the shared definition does not link across
// translation units.
void taaTransitionImage(
    VkCommandBuffer commandBuffer,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask) {
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
    imageBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &imageBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
}

}  // namespace

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

// No alignas here: it would pad sizeof to 16 and declare a 16-byte push
// constant range, while ssao.comp.slang declares exactly two uints and the
// dispatch in frame_pass_ssao.cc pushes 8 bytes. The upper half would then be
// permanently undefined -- harmless only because nothing reads it, and
// BestPractices-PushConstants flags it on every dispatch. The structs above
// reach a 16-byte multiple through explicit _pad fields instead, which is why
// they can carry alignas without the size drifting from what is pushed.
} // namespace

// Material library table (set 0, binding 13). Created unconditionally at init,
// before any scene exists: the binding is declared in the descriptor layout for
// every pipeline, so it must resolve to a real buffer from the first frame
// rather than only once something has materials.
//
// One buffer holding kMaxFramesInFlight copies of the table. The descriptor is
// pointed at the current frame's region each frame, so an edit applied between
// frames never mutates memory a frame in flight is still reading. At 8 KB per
// region that costs 16 KB total and a 8 KB memcpy on frames where the table
// actually changed -- which is only when someone moves a slider.
bool RendererBackend::createImportedMaterialResources() {
    if (m_importedMaterialBufferHandle != kInvalidBufferHandle) {
        return true;
    }
    // Slot 0 is the reserved sentinel and is never read; the rest default to a
    // fully rough dielectric with a white tint, i.e. the legacy response.
    m_importedMaterialTable.fill(importer::GpuImportedMaterial{});
    m_importedMaterialTableDirtyFrames = kMaxFramesInFlight;

    BufferCreateDesc desc{};
    desc.size = static_cast<VkDeviceSize>(sizeof(importer::GpuImportedMaterial)) *
                importer::kImportedSceneMaterialTableCapacity * kMaxFramesInFlight;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    desc.memoryProperties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    m_importedMaterialBufferHandle = m_bufferAllocator.createBuffer(desc);
    if (m_importedMaterialBufferHandle == kInvalidBufferHandle) {
        VOX_LOGE("render") << "failed to create imported material table buffer";
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_BUFFER,
        vkHandleToUint64(m_bufferAllocator.getBuffer(m_importedMaterialBufferHandle)),
        "importedMaterial.tableBuffer"
    );
    return true;
}

namespace {

// One CPU-side library entry -> the packed GPU record. Emissive strength is
// premultiplied here so the shader adds a radiance directly and never has to
// know the authoring split between colour and intensity.
importer::GpuImportedMaterial toGpuMaterial(const importer::ImportedSceneMaterial& material) {
    importer::GpuImportedMaterial out{};
    out.baseColorMetallic[0] = material.baseColorTint[0];
    out.baseColorMetallic[1] = material.baseColorTint[1];
    out.baseColorMetallic[2] = material.baseColorTint[2];
    out.baseColorMetallic[3] = std::clamp(material.metallic, 0.0f, 1.0f);
    out.emissiveRoughness[0] = material.emissive[0] * material.emissiveStrength;
    out.emissiveRoughness[1] = material.emissive[1] * material.emissiveStrength;
    out.emissiveRoughness[2] = material.emissive[2] * material.emissiveStrength;
    out.emissiveRoughness[3] = std::clamp(material.roughness, 0.0f, 1.0f);
    return out;
}

}  // namespace

void RendererBackend::setImportedMaterial(std::uint32_t index,
                                          const importer::ImportedSceneMaterial& material) {
    // Slot 0 is the reserved sentinel; writing it would be silently ignored by
    // the shader anyway, so reject it here where it is visible.
    if (index == 0u || index >= importer::kImportedSceneMaterialTableCapacity) {
        return;
    }
    m_importedMaterialTable[index] = toGpuMaterial(material);
    // Every frame-in-flight region needs the new value before the edit is fully
    // applied; see the countdown in updateFrameDescriptorSets().
    m_importedMaterialTableDirtyFrames = kMaxFramesInFlight;
}

void RendererBackend::setImportedMaterialTable(
    const std::vector<importer::ImportedSceneMaterial>& materials) {
    m_importedMaterialTable.fill(importer::GpuImportedMaterial{});
    const std::size_t count =
        std::min<std::size_t>(materials.size(), importer::kImportedSceneMaterialTableCapacity);
    if (materials.size() > importer::kImportedSceneMaterialTableCapacity) {
        VOX_LOGW("render") << "material table truncated to "
                           << importer::kImportedSceneMaterialTableCapacity << " of "
                           << materials.size() << " entries";
    }
    // Starts at 1: slot 0 stays the identity sentinel no matter what the caller
    // put in materials[0], so the flag index and the table index never diverge.
    for (std::size_t i = 1; i < count; ++i) {
        m_importedMaterialTable[i] = toGpuMaterial(materials[i]);
    }
    m_importedMaterialTableDirtyFrames = kMaxFramesInFlight;
}

bool RendererBackend::createAutoExposureResources() {
    const float initialExposure = std::clamp(m_skyDebugSettings.manualExposure, 0.05f, 8.0f);
    if (m_autoExposureStateBufferHandle == kInvalidBufferHandle) {
        const std::array<float, 4> initialState = {initialExposure, initialExposure, 1.0f, 0.0f};
        BufferCreateDesc exposureStateBufferDesc{};
        exposureStateBufferDesc.size = sizeof(initialState);
        exposureStateBufferDesc.usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
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
        histogramBufferDesc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
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
                "renderer.descriptorSetLayout.autoExposure",
                nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
            )) {
            destroyAutoExposureResources();
            return false;
        }
    }

    // Descriptor-buffer backing: one region per frame-in-flight. The set has a
    // combined image sampler (hdr scene) so it needs both resource + sampler usage.
    if (!m_autoExposureBufferSet.valid()) {
        if (!createDescriptorBufferSet(
                m_autoExposureDescriptorSetLayout,
                kMaxFramesInFlight,
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                    VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
                "renderer.descriptorBuffer.autoExposure",
                m_autoExposureBufferSet
            )) {
            destroyAutoExposureResources();
            return false;
        }
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
            "pipeline.autoExposure.histogram",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
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
            "pipeline.autoExposure.update",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
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
        cameraBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
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
                "renderer.descriptorSetLayout.sunShaft",
                nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
            )) {
            destroySunShaftResources();
            return false;
        }
    }

    // Descriptor-buffer backing: camera UBO + 2 combined image samplers + storage image.
    if (!m_sunShaftBufferSet.valid()) {
        if (!createDescriptorBufferSet(
                m_sunShaftDescriptorSetLayout,
                kMaxFramesInFlight,
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                    VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
                "renderer.descriptorBuffer.sunShaft",
                m_sunShaftBufferSet
            )) {
            destroySunShaftResources();
            return false;
        }
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
            "pipeline.sunShaft.compute",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
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
    destroyDescriptorBufferSet(m_autoExposureBufferSet);
    if (m_autoExposureDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_autoExposureDescriptorSetLayout, nullptr);
        m_autoExposureDescriptorSetLayout = VK_NULL_HANDLE;
    }

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
    destroyDescriptorBufferSet(m_sunShaftBufferSet);
    if (m_sunShaftDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_sunShaftDescriptorSetLayout, nullptr);
        m_sunShaftDescriptorSetLayout = VK_NULL_HANDLE;
    }
    m_sunShaftComputeAvailable = false;
    m_sunShaftShaderAvailable = false;
}

bool RendererBackend::createSsaoComputeResources() {
    constexpr const char* kSsaoShaderPath = "../src/render/shaders/ssao.comp.slang.spv";
    constexpr const char* kSsaoHbaoShaderPath = "../src/render/shaders/ssao_hbao.comp.slang.spv";
    constexpr const char* kSsaoGtaoShaderPath = "../src/render/shaders/ssao_gtao.comp.slang.spv";
    constexpr const char* kSsaoBlurShaderPath = "../src/render/shaders/ssao_blur.comp.slang.spv";

    if (m_ssaoDescriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding cameraBinding{};
        cameraBinding.binding = 0;
        cameraBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        cameraBinding.descriptorCount = 1;
        cameraBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding normalDepthBinding{};
        normalDepthBinding.binding = 1;
        normalDepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        normalDepthBinding.descriptorCount = 1;
        normalDepthBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding ssaoRawOutputBinding{};
        ssaoRawOutputBinding.binding = 2;
        ssaoRawOutputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ssaoRawOutputBinding.descriptorCount = 1;
        ssaoRawOutputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        const std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            cameraBinding,
            normalDepthBinding,
            ssaoRawOutputBinding
        };

        if (!createDescriptorSetLayout(
                bindings,
                m_ssaoDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(ssao)",
                "renderer.descriptorSetLayout.ssao",
                nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
            )) {
            destroySsaoComputeResources();
            return false;
        }
    }

    // Descriptor-buffer backing: camera UBO + normal-depth sampler + storage image.
    if (!m_ssaoBufferSet.valid()) {
        if (!createDescriptorBufferSet(
                m_ssaoDescriptorSetLayout,
                kMaxFramesInFlight,
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                    VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
                "renderer.descriptorBuffer.ssao",
                m_ssaoBufferSet
            )) {
            destroySsaoComputeResources();
            return false;
        }
    }

    if (m_ssaoBlurDescriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding normalDepthBinding{};
        normalDepthBinding.binding = 0;
        normalDepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        normalDepthBinding.descriptorCount = 1;
        normalDepthBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding ssaoRawBinding{};
        ssaoRawBinding.binding = 1;
        ssaoRawBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ssaoRawBinding.descriptorCount = 1;
        ssaoRawBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding ssaoBlurOutputBinding{};
        ssaoBlurOutputBinding.binding = 2;
        ssaoBlurOutputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ssaoBlurOutputBinding.descriptorCount = 1;
        ssaoBlurOutputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        const std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            normalDepthBinding,
            ssaoRawBinding,
            ssaoBlurOutputBinding
        };

        if (!createDescriptorSetLayout(
                bindings,
                m_ssaoBlurDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(ssaoBlur)",
                "renderer.descriptorSetLayout.ssaoBlur",
                nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
            )) {
            destroySsaoComputeResources();
            return false;
        }
    }

    // Descriptor-buffer backing: 2 combined image samplers + storage image (no camera).
    if (!m_ssaoBlurBufferSet.valid()) {
        if (!createDescriptorBufferSet(
                m_ssaoBlurDescriptorSetLayout,
                kMaxFramesInFlight,
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                    VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
                "renderer.descriptorBuffer.ssaoBlur",
                m_ssaoBlurBufferSet
            )) {
            destroySsaoComputeResources();
            return false;
        }
    }

    std::array<VkShaderModule, 4> shaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& ssaoShaderModule = shaderModules[0];
    VkShaderModule& ssaoBlurShaderModule = shaderModules[1];
    VkShaderModule& ssaoHbaoShaderModule = shaderModules[2];
    VkShaderModule& ssaoGtaoShaderModule = shaderModules[3];
    if (!createShaderModuleFromFile(m_device, kSsaoShaderPath, "ssao.comp", ssaoShaderModule)) {
        destroyShaderModules(m_device, shaderModules);
        destroySsaoComputeResources();
        return false;
    }
    if (!createShaderModuleFromFile(m_device, kSsaoHbaoShaderPath, "ssao_hbao.comp", ssaoHbaoShaderModule)) {
        destroyShaderModules(m_device, shaderModules);
        destroySsaoComputeResources();
        return false;
    }
    if (!createShaderModuleFromFile(m_device, kSsaoGtaoShaderPath, "ssao_gtao.comp", ssaoGtaoShaderModule)) {
        destroyShaderModules(m_device, shaderModules);
        destroySsaoComputeResources();
        return false;
    }
    if (!createShaderModuleFromFile(m_device, kSsaoBlurShaderPath, "ssao_blur.comp", ssaoBlurShaderModule)) {
        destroyShaderModules(m_device, shaderModules);
        destroySsaoComputeResources();
        return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(SsaoComputePushConstants);
    const std::array<VkPushConstantRange, 1> pushConstantRanges = {pushConstantRange};

    if (!createComputePipelineLayout(
            m_ssaoDescriptorSetLayout,
            pushConstantRanges,
            m_ssaoPipelineLayout,
            "vkCreatePipelineLayout(ssao)",
            "renderer.pipelineLayout.ssao"
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroySsaoComputeResources();
        return false;
    }
    if (!createComputePipeline(
            m_ssaoPipelineLayout,
            ssaoShaderModule,
            m_ssaoPipeline,
            "vkCreateComputePipelines(ssao)",
            "pipeline.ssao",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroySsaoComputeResources();
        return false;
    }

    // HBAO and GTAO share the SSAO layout and descriptor set; only the shader differs,
    // so they are three pipelines over one binding model rather than three passes.
    if (!createComputePipeline(
            m_ssaoPipelineLayout,
            ssaoHbaoShaderModule,
            m_ssaoHbaoPipeline,
            "vkCreateComputePipelines(ssaoHbao)",
            "pipeline.ssao.hbao",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroySsaoComputeResources();
        return false;
    }
    if (!createComputePipeline(
            m_ssaoPipelineLayout,
            ssaoGtaoShaderModule,
            m_ssaoGtaoPipeline,
            "vkCreateComputePipelines(ssaoGtao)",
            "pipeline.ssao.gtao",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroySsaoComputeResources();
        return false;
    }

    if (!createComputePipelineLayout(
            m_ssaoBlurDescriptorSetLayout,
            pushConstantRanges,
            m_ssaoBlurPipelineLayout,
            "vkCreatePipelineLayout(ssaoBlur)",
            "renderer.pipelineLayout.ssaoBlur"
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroySsaoComputeResources();
        return false;
    }
    if (!createComputePipeline(
            m_ssaoBlurPipelineLayout,
            ssaoBlurShaderModule,
            m_ssaoBlurPipeline,
            "vkCreateComputePipelines(ssaoBlur)",
            "pipeline.ssaoBlur",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
        )) {
        destroyShaderModules(m_device, shaderModules);
        destroySsaoComputeResources();
        return false;
    }

    destroyShaderModules(m_device, shaderModules);

    VOX_LOGI("render") << "ssao compute resources ready\n";
    return true;
}

void RendererBackend::destroySsaoComputeResources() {
    if (m_ssaoPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_ssaoPipeline, nullptr);
        m_ssaoPipeline = VK_NULL_HANDLE;
    }
    if (m_ssaoBlurPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_ssaoBlurPipeline, nullptr);
        m_ssaoBlurPipeline = VK_NULL_HANDLE;
    }
    if (m_ssaoPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_ssaoPipelineLayout, nullptr);
        m_ssaoPipelineLayout = VK_NULL_HANDLE;
    }
    destroyDescriptorBufferSet(m_ssaoBufferSet);
    if (m_ssaoDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_ssaoDescriptorSetLayout, nullptr);
        m_ssaoDescriptorSetLayout = VK_NULL_HANDLE;
    }

    if (m_ssaoBlurPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_ssaoBlurPipelineLayout, nullptr);
        m_ssaoBlurPipelineLayout = VK_NULL_HANDLE;
    }
    destroyDescriptorBufferSet(m_ssaoBlurBufferSet);
    if (m_ssaoBlurDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_ssaoBlurDescriptorSetLayout, nullptr);
        m_ssaoBlurDescriptorSetLayout = VK_NULL_HANDLE;
    }
}


// ---------------------------------------------------------------------------
// Temporal AA. See taa.comp.slang for what this fixes and why reprojection is
// camera-only. The pass samples hdrResolve mip0, blends clamped history, and
// copies the result back over mip0 so bloom and tonemap stay untouched.

bool RendererBackend::createTaaComputeResources() {
    constexpr const char* kTaaShaderPath = "../src/render/shaders/taa.comp.slang.spv";

    if (m_taaDescriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding cameraBinding{};
        cameraBinding.binding = 0;
        cameraBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        cameraBinding.descriptorCount = 1;
        cameraBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding currentColorBinding{};
        currentColorBinding.binding = 1;
        currentColorBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        currentColorBinding.descriptorCount = 1;
        currentColorBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding historyBinding{};
        historyBinding.binding = 2;
        historyBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        historyBinding.descriptorCount = 1;
        historyBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding normalDepthBinding{};
        normalDepthBinding.binding = 3;
        normalDepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        normalDepthBinding.descriptorCount = 1;
        normalDepthBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding outputBinding{};
        outputBinding.binding = 4;
        outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        outputBinding.descriptorCount = 1;
        outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding taaUniformBinding{};
        taaUniformBinding.binding = 5;
        taaUniformBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        taaUniformBinding.descriptorCount = 1;
        taaUniformBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // Binding 6: per-object motion vectors. Unused by taa.comp.slang and
        // read by temporal_upscale.comp.slang; a binding a shader does not
        // reference is legal, and one layout for both is what lets the pass pick
        // a pipeline rather than duplicate its descriptor plumbing.
        VkDescriptorSetLayoutBinding velocityBinding{};
        velocityBinding.binding = 6;
        velocityBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        velocityBinding.descriptorCount = 1;
        velocityBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        const std::array<VkDescriptorSetLayoutBinding, 7> bindings = {
            cameraBinding, currentColorBinding, historyBinding,
            normalDepthBinding, outputBinding, taaUniformBinding, velocityBinding
        };
        if (!createDescriptorSetLayout(
                bindings,
                m_taaDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(taa)",
                "renderer.descriptorSetLayout.taa",
                nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
            )) {
            destroyTaaComputeResources();
            return false;
        }
    }

    if (!m_taaBufferSet.valid()) {
        if (!createDescriptorBufferSet(
                m_taaDescriptorSetLayout,
                kMaxFramesInFlight,
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                    VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
                "renderer.descriptorBuffer.taa",
                m_taaBufferSet
            )) {
            destroyTaaComputeResources();
            return false;
        }
    }

    if (m_taaPipeline != VK_NULL_HANDLE) {
        return true;
    }

    VkShaderModule taaShaderModule = VK_NULL_HANDLE;
    if (!createShaderModuleFromFile(m_device, kTaaShaderPath, "taa.comp", taaShaderModule)) {
        destroyTaaComputeResources();
        return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(TaaPushConstants);
    const std::array<VkPushConstantRange, 1> pushConstantRanges = {pushConstantRange};

    if (!createComputePipelineLayout(
            m_taaDescriptorSetLayout,
            pushConstantRanges,
            m_taaPipelineLayout,
            "vkCreatePipelineLayout(taa)",
            "renderer.pipelineLayout.taa"
        )) {
        vkDestroyShaderModule(m_device, taaShaderModule, nullptr);
        destroyTaaComputeResources();
        return false;
    }
    if (!createComputePipeline(
            m_taaPipelineLayout,
            taaShaderModule,
            m_taaPipeline,
            "vkCreateComputePipelines(taa)",
            "pipeline.taa",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
        )) {
        vkDestroyShaderModule(m_device, taaShaderModule, nullptr);
        destroyTaaComputeResources();
        return false;
    }

    // The upscaler shares the TAA pipeline layout, so it is built here and
    // selected per frame. Optional: if its shader is missing the Temporal
    // backend still resolves, and recordTaaPass falls back to same-resolution
    // TAA -- a softer image than a real upscale, but not a broken one.
    {
        constexpr const char* kTemporalUpscaleShaderPath =
            "../src/render/shaders/temporal_upscale.comp.slang.spv";
        if (std::filesystem::exists(kTemporalUpscaleShaderPath)) {
            VkShaderModule upscaleModule = VK_NULL_HANDLE;
            if (createShaderModuleFromFile(
                    m_device, kTemporalUpscaleShaderPath, "temporal_upscale.comp", upscaleModule)) {
                if (!createComputePipeline(
                        m_taaPipelineLayout, upscaleModule, m_temporalUpscalePipeline,
                        "vkCreateComputePipelines(temporalUpscale)", "pipeline.temporalUpscale",
                        VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)) {
                    VOX_LOGW("render") << "temporal upscale pipeline creation failed; "
                                          "Temporal backend will run same-resolution TAA";
                }
                vkDestroyShaderModule(m_device, upscaleModule, nullptr);
            }
        } else {
            VOX_LOGW("render") << "temporal_upscale shader missing; Temporal backend will run "
                                  "same-resolution TAA";
        }
    }
    vkDestroyShaderModule(m_device, taaShaderModule, nullptr);
    return true;
}

void RendererBackend::destroyTaaComputeResources() {
    // BEFORE the pipelines below, and more importantly before vkDestroyDevice.
    // The backend owns VkPipelines built against this device, and a unique_ptr
    // member would otherwise be destroyed at ~RendererBackend -- after the
    // device is gone. That is a use-after-free the loader catches as
    // "vkDestroyPipeline: Invalid device" and the process aborts on exit, long
    // after the frame that looked fine.
    m_upscaler.reset();
    if (m_taaPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_taaPipeline, nullptr);
        m_taaPipeline = VK_NULL_HANDLE;
    }
    if (m_temporalUpscalePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_temporalUpscalePipeline, nullptr);
        m_temporalUpscalePipeline = VK_NULL_HANDLE;
    }
    if (m_taaPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_taaPipelineLayout, nullptr);
        m_taaPipelineLayout = VK_NULL_HANDLE;
    }
    destroyDescriptorBufferSet(m_taaBufferSet);
    if (m_taaDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_taaDescriptorSetLayout, nullptr);
        m_taaDescriptorSetLayout = VK_NULL_HANDLE;
    }
}

upscale::HostServices RendererBackend::makeUpscalerHostServices() {
    upscale::HostServices host{};
    host.device = m_device;
    // Descriptor provisioning stays here on purpose: this engine writes
    // VK_EXT_descriptor_buffer sets, which is not something a vendor SDK knows
    // about and not something a backend should have to.
    host.bindDescriptors = [this](VkCommandBuffer cmd, VkPipelineLayout layout,
                                  std::uint32_t frameIndex) {
        bindDescriptorBuffer(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, m_taaBufferSet, frameIndex);
    };
    host.transitionImage = [](VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout,
                              VkImageLayout newLayout, VkPipelineStageFlags2 srcStage,
                              VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage,
                              VkAccessFlags2 dstAccess) {
        taaTransitionImage(
            cmd, image, oldLayout, newLayout, srcStage, srcAccess, dstStage, dstAccess);
    };
    host.beginDebugLabel = [this](VkCommandBuffer cmd, const char* label, float r, float g,
                                  float b, float a) {
        beginDebugLabel(cmd, label, r, g, b, a);
    };
    host.endDebugLabel = [this](VkCommandBuffer cmd) { endDebugLabel(cmd); };
    host.loadShader = [this](const char* path, const char* debugName, VkShaderModule& out) {
        return createShaderModuleFromFile(m_device, path, debugName, out);
    };
    return host;
}

bool RendererBackend::createUpscalerBackend() {
    m_upscaler = upscale::createUpscaler(m_upscalerStatus.active, makeUpscalerHostServices());
    if (!m_upscaler) {
        // Off resolves to no technique at all, which is correct rather than an
        // error. Anything else reaching here means resolveUpscaler() picked a
        // backend createUpscaler() cannot build, which is a bug in one of them.
        return m_upscalerStatus.active == UpscalerBackend::Off;
    }
    upscale::SetupInfo setup{};
    setup.renderExtent = {m_renderExtent.width, m_renderExtent.height};
    setup.displayExtent = {m_swapchainExtent.width, m_swapchainExtent.height};
    setup.invertedDepth = true;
    setup.hdrInput = true;
    setup.pipelineLayout = m_taaPipelineLayout;
    if (!m_upscaler->setup(setup)) {
        VOX_LOGW("render") << "upscaler backend "
                           << upscalerBackendName(m_upscaler->id())
                           << " failed setup; the frame will run without it";
        m_upscaler.reset();
        return false;
    }
    const upscale::Capabilities caps = m_upscaler->capabilities();
    if (!caps.available) {
        VOX_LOGW("render") << "upscaler backend " << upscalerBackendName(m_upscaler->id())
                           << " unavailable: " << caps.unavailableReason;
        m_upscaler.reset();
        return false;
    }
    return true;
}

RendererBackend::TaaPassOutcome RendererBackend::recordTaaPass(
    VkCommandBuffer commandBuffer,
    std::uint32_t aoFrameIndex,
    VkQueryPool gpuTimestampQueryPool) {
    if (!m_taaEnabled || !m_upscaler || !m_taaBufferSet.valid()) {
        return {};
    }
    if (aoFrameIndex >= m_hdrResolveImages.size() ||
        m_hdrResolveImages[aoFrameIndex] == VK_NULL_HANDLE) {
        return {};
    }
    const std::uint32_t currentImage = m_taaHistoryIndex ^ 1u;
    const std::uint32_t historyImage = m_taaHistoryIndex;
    if (m_taaImages[currentImage] == VK_NULL_HANDLE ||
        m_taaImages[historyImage] == VK_NULL_HANDLE) {
        return {};
    }

    // Covers the TAA resolve AND the temporal upscale -- they are one pass with
    // two output paths, and separating them would make the upscaled and
    // non-upscaled frames report different pass sets for the same work.
    const auto writeTaaTimestamp = [&](uint32_t queryIndex, VkPipelineStageFlags2 stage) {
        if (gpuTimestampQueryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(commandBuffer, stage, gpuTimestampQueryPool, queryIndex);
        }
    };
    writeTaaTimestamp(kGpuTimestampQueryTaaStart, VK_PIPELINE_STAGE_2_NONE);

    // EVERYTHING BETWEEN HERE AND THE COPY-BACK BELOW IS THE BACKEND'S.
    //
    // The transitions, the pipeline choice and the dispatch grid are properties
    // of the technique, not of this frame -- XeSS wants different resource
    // states and a different dispatch than this does, and DLSS different again.
    // What stays here is what the HOST owns either way: which images exist,
    // whose descriptors point at them, and what happens to the result.
    upscale::DispatchInfo dispatch{};
    dispatch.commandBuffer = commandBuffer;
    dispatch.frameIndex = m_currentFrame;
    dispatch.colorInput = m_hdrResolveImages[aoFrameIndex];
    dispatch.colorInputLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    dispatch.history = m_taaImages[historyImage];
    dispatch.historyInitialized = m_taaImageInitialized[historyImage];
    dispatch.output = m_taaImages[currentImage];
    dispatch.outputInitialized = m_taaImageInitialized[currentImage];
    dispatch.jitter = {m_taaJitterNdc[0], m_taaJitterNdc[1]};
    dispatch.resetHistory = !m_taaHistoryValid;

    const upscale::DispatchResult dispatched = m_upscaler->dispatch(dispatch);
    if (!dispatched.ran) {
        writeTaaTimestamp(kGpuTimestampQueryTaaEnd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        return {};
    }
    m_taaImageInitialized[historyImage] = true;
    const bool upscaling = dispatched.resultInOutput;

    if (upscaling) {
        // No copy-back: the result is larger than hdrResolve and there is
        // nothing to copy it into. The tonemap pass samples the upscaled image
        // directly instead (see updateFrameDescriptorSets), which is also where
        // the resolution change now happens rather than in a bilinear fetch.
        //
        // The cost is that the bloom chain, which is built from hdrResolve's mip
        // pyramid, keeps reading the pre-upscale image. Bloom is low frequency
        // and this is the usual arrangement, but it does mean bloom no longer
        // sees the temporally resolved result the way it did at native
        // resolution.
        taaTransitionImage(
            commandBuffer, m_taaImages[currentImage],
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        // hdrResolve is left where the main pass put it; nothing transitioned it
        // for a copy that is not happening.
        taaTransitionImage(
            commandBuffer, m_hdrResolveImages[aoFrameIndex],
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        m_taaImageInitialized[currentImage] = true;
        endDebugLabel(commandBuffer);
        m_taaHistoryIndex = currentImage;
        // Both of these, not just the index. m_taaHistoryValid is what the
        // uniform's history-valid flag is built from, and leaving it false here
        // multiplied the history weight to zero on every frame -- the pass ran,
        // wrote a correct image, and accumulated nothing, which is
        // indistinguishable from a working upscaler with the weight turned down.
        m_taaHistoryValid = true;
        writeTaaTimestamp(kGpuTimestampQueryTaaEnd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
        return TaaPassOutcome{
            true,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    }

    // Copy the result back over hdrResolve mip0 so the bloom/tonemap chain
    // reads TAA output without knowing TAA exists.
    taaTransitionImage(
        commandBuffer, m_taaImages[currentImage],
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    taaTransitionImage(
        commandBuffer, m_hdrResolveImages[aoFrameIndex],
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u};
    copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 0u, 1u};
    copyRegion.extent = {m_renderExtent.width, m_renderExtent.height, 1u};
    vkCmdCopyImage(
        commandBuffer,
        m_taaImages[currentImage], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_hdrResolveImages[aoFrameIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1u, &copyRegion);

    // The output becomes next frame's history input.
    taaTransitionImage(
        commandBuffer, m_taaImages[currentImage],
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    m_taaImageInitialized[currentImage] = true;

    endDebugLabel(commandBuffer);

    m_taaHistoryIndex = currentImage;
    m_taaHistoryValid = true;
    writeTaaTimestamp(kGpuTimestampQueryTaaEnd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    return TaaPassOutcome{
        true,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT};
}


// XeGTAO resource creation. Two pipelines: the depth prefilter that builds the
// pyramid, and the GTAO integral that marches it.
//
// Failure here is NOT fatal. The three original estimators keep their own
// pipelines and are unaffected; XeGTAO simply reports unavailable and the AO
// mode selection falls back. That matters because this needs two extra shaders
// on disk and a descriptor layout with a five-element sampler array, and a build
// without slangc has neither.
bool RendererBackend::createXeGtaoResources() {
    constexpr const char* kPrefilterShaderPath =
        "../src/render/shaders/xegtao_prefilter.comp.slang.spv";
    constexpr const char* kMainShaderPath =
        "../src/render/shaders/xegtao_main.comp.slang.spv";
    constexpr const char* kDenoiseShaderPath =
        "../src/render/shaders/xegtao_denoise.comp.slang.spv";

    if (!std::filesystem::exists(kPrefilterShaderPath) || !std::filesystem::exists(kMainShaderPath) ||
        !std::filesystem::exists(kDenoiseShaderPath)) {
        VOX_LOGW("render") << "XeGTAO shaders missing; XeGTAO mode unavailable";
        return false;
    }

    const auto computeBinding = [](uint32_t binding, VkDescriptorType type, uint32_t count) {
        VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = binding;
        layoutBinding.descriptorType = type;
        layoutBinding.descriptorCount = count;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        return layoutBinding;
    };

    // Prefilter: camera, the normal-depth source, and one storage image per
    // pyramid level.
    if (m_xegtaoPrefilterDescriptorSetLayout == VK_NULL_HANDLE) {
        const std::array<VkDescriptorSetLayoutBinding, 7> bindings = {
            computeBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
            computeBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1),
            computeBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
            computeBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
            computeBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
            computeBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
            computeBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
        };
        if (!createDescriptorSetLayout(
                bindings, m_xegtaoPrefilterDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(xegtaoPrefilter)",
                "renderer.descriptorSetLayout.xegtaoPrefilter", nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)) {
            destroyXeGtaoResources();
            return false;
        }
    }
    // Main: camera, normal-depth, the pyramid as a 5-element sampler array (the
    // march indexes it by computed level, so it has to be one array binding
    // rather than five), the AO output and the bent normal output.
    if (m_xegtaoMainDescriptorSetLayout == VK_NULL_HANDLE) {
        const std::array<VkDescriptorSetLayoutBinding, 5> bindings = {
            computeBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1),
            computeBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1),
            computeBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kXeGtaoDepthMipCount),
            computeBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
            computeBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
        };
        if (!createDescriptorSetLayout(
                bindings, m_xegtaoMainDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(xegtaoMain)",
                "renderer.descriptorSetLayout.xegtaoMain", nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)) {
            destroyXeGtaoResources();
            return false;
        }
    }

    // Denoise: the pre-denoise AO term, the bent-normal target (for its packed
    // edges in .w) and the final AO image.
    if (m_xegtaoDenoiseDescriptorSetLayout == VK_NULL_HANDLE) {
        const std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            computeBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1),
            computeBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1),
            computeBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1),
        };
        if (!createDescriptorSetLayout(
                bindings, m_xegtaoDenoiseDescriptorSetLayout,
                "vkCreateDescriptorSetLayout(xegtaoDenoise)",
                "renderer.descriptorSetLayout.xegtaoDenoise", nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)) {
            destroyXeGtaoResources();
            return false;
        }
    }

    const VkBufferUsageFlags descriptorBufferUsage =
        VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
        VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
    if (!m_xegtaoPrefilterBufferSet.valid() &&
        !createDescriptorBufferSet(
            m_xegtaoPrefilterDescriptorSetLayout, kMaxFramesInFlight, descriptorBufferUsage,
            "renderer.descriptorBuffer.xegtaoPrefilter", m_xegtaoPrefilterBufferSet)) {
        destroyXeGtaoResources();
        return false;
    }
    if (!m_xegtaoMainBufferSet.valid() &&
        !createDescriptorBufferSet(
            m_xegtaoMainDescriptorSetLayout, kMaxFramesInFlight, descriptorBufferUsage,
            "renderer.descriptorBuffer.xegtaoMain", m_xegtaoMainBufferSet)) {
        destroyXeGtaoResources();
        return false;
    }

    if (!m_xegtaoDenoiseBufferSet.valid() &&
        !createDescriptorBufferSet(
            m_xegtaoDenoiseDescriptorSetLayout, kMaxFramesInFlight, descriptorBufferUsage,
            "renderer.descriptorBuffer.xegtaoDenoise", m_xegtaoDenoiseBufferSet)) {
        destroyXeGtaoResources();
        return false;
    }

    std::array<VkShaderModule, 3> shaderModules = {
        VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    if (!createShaderModuleFromFile(
            m_device, kPrefilterShaderPath, "xegtao_prefilter.comp", shaderModules[0]) ||
        !createShaderModuleFromFile(
            m_device, kMainShaderPath, "xegtao_main.comp", shaderModules[1]) ||
        !createShaderModuleFromFile(
            m_device, kDenoiseShaderPath, "xegtao_denoise.comp", shaderModules[2])) {
        destroyShaderModules(m_device, shaderModules);
        destroyXeGtaoResources();
        return false;
    }

    VkPushConstantRange prefilterRange{};
    prefilterRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    prefilterRange.size = sizeof(XeGtaoPrefilterPushConstants);
    const std::array<VkPushConstantRange, 1> prefilterRanges = {prefilterRange};

    VkPushConstantRange mainRange{};
    mainRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    mainRange.size = sizeof(XeGtaoMainPushConstants);
    const std::array<VkPushConstantRange, 1> mainRanges = {mainRange};

    VkPushConstantRange denoiseRange{};
    denoiseRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    denoiseRange.size = sizeof(XeGtaoDenoisePushConstants);
    const std::array<VkPushConstantRange, 1> denoiseRanges = {denoiseRange};

    if (!createComputePipelineLayout(
            m_xegtaoPrefilterDescriptorSetLayout, prefilterRanges, m_xegtaoPrefilterPipelineLayout,
            "vkCreatePipelineLayout(xegtaoPrefilter)", "renderer.pipelineLayout.xegtaoPrefilter") ||
        !createComputePipeline(
            m_xegtaoPrefilterPipelineLayout, shaderModules[0], m_xegtaoPrefilterPipeline,
            "vkCreateComputePipelines(xegtaoPrefilter)", "pipeline.xegtaoPrefilter",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT) ||
        !createComputePipelineLayout(
            m_xegtaoMainDescriptorSetLayout, mainRanges, m_xegtaoMainPipelineLayout,
            "vkCreatePipelineLayout(xegtaoMain)", "renderer.pipelineLayout.xegtaoMain") ||
        !createComputePipeline(
            m_xegtaoMainPipelineLayout, shaderModules[1], m_xegtaoMainPipeline,
            "vkCreateComputePipelines(xegtaoMain)", "pipeline.xegtaoMain",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT) ||
        !createComputePipelineLayout(
            m_xegtaoDenoiseDescriptorSetLayout, denoiseRanges, m_xegtaoDenoisePipelineLayout,
            "vkCreatePipelineLayout(xegtaoDenoise)", "renderer.pipelineLayout.xegtaoDenoise") ||
        !createComputePipeline(
            m_xegtaoDenoisePipelineLayout, shaderModules[2], m_xegtaoDenoisePipeline,
            "vkCreateComputePipelines(xegtaoDenoise)", "pipeline.xegtaoDenoise",
            VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)) {
        destroyShaderModules(m_device, shaderModules);
        destroyXeGtaoResources();
        return false;
    }

    destroyShaderModules(m_device, shaderModules);
    VOX_LOGI("render") << "XeGTAO resources ready (depth pyramid levels="
                       << kXeGtaoDepthMipCount << ")";
    return true;
}

void RendererBackend::destroyXeGtaoResources() {
    if (m_xegtaoPrefilterPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_xegtaoPrefilterPipeline, nullptr);
        m_xegtaoPrefilterPipeline = VK_NULL_HANDLE;
    }
    if (m_xegtaoMainPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_xegtaoMainPipeline, nullptr);
        m_xegtaoMainPipeline = VK_NULL_HANDLE;
    }
    if (m_xegtaoDenoisePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_xegtaoDenoisePipeline, nullptr);
        m_xegtaoDenoisePipeline = VK_NULL_HANDLE;
    }
    if (m_xegtaoPrefilterPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_xegtaoPrefilterPipelineLayout, nullptr);
        m_xegtaoPrefilterPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_xegtaoMainPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_xegtaoMainPipelineLayout, nullptr);
        m_xegtaoMainPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_xegtaoDenoisePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_xegtaoDenoisePipelineLayout, nullptr);
        m_xegtaoDenoisePipelineLayout = VK_NULL_HANDLE;
    }
    destroyDescriptorBufferSet(m_xegtaoDenoiseBufferSet);
    if (m_xegtaoDenoiseDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_xegtaoDenoiseDescriptorSetLayout, nullptr);
        m_xegtaoDenoiseDescriptorSetLayout = VK_NULL_HANDLE;
    }
    destroyDescriptorBufferSet(m_xegtaoPrefilterBufferSet);
    destroyDescriptorBufferSet(m_xegtaoMainBufferSet);
    if (m_xegtaoPrefilterDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_xegtaoPrefilterDescriptorSetLayout, nullptr);
        m_xegtaoPrefilterDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (m_xegtaoMainDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_xegtaoMainDescriptorSetLayout, nullptr);
        m_xegtaoMainDescriptorSetLayout = VK_NULL_HANDLE;
    }
}

// Per-frame descriptor writes. The AO targets are per-frame-in-flight, so the
// views change with aoFrameIndex and these cannot be written once at creation.
void RendererBackend::writeXeGtaoDescriptors(
    uint32_t frameIndex, uint32_t aoFrameIndex, VkDeviceAddress cameraAddress,
    VkDeviceSize cameraRange, const VkDescriptorImageInfo& normalDepthInfo) {
    if (!m_xegtaoPrefilterBufferSet.valid() || !m_xegtaoMainBufferSet.valid() ||
        aoFrameIndex >= m_xegtaoBentNormalImageViews.size() ||
        m_xegtaoBentNormalImageViews[aoFrameIndex] == VK_NULL_HANDLE) {
        return;
    }

    // Prefilter set.
    writeDescriptorBufferUniform(
        m_xegtaoPrefilterBufferSet, frameIndex,
        descriptorBufferBindingOffset(m_xegtaoPrefilterDescriptorSetLayout, 0),
        cameraAddress, cameraRange);
    writeDescriptorBufferCombinedImageSampler(
        m_xegtaoPrefilterBufferSet, frameIndex,
        descriptorBufferBindingOffset(m_xegtaoPrefilterDescriptorSetLayout, 1), 0,
        normalDepthInfo.imageView, normalDepthInfo.sampler, normalDepthInfo.imageLayout);
    for (uint32_t level = 0; level < kXeGtaoDepthMipCount; ++level) {
        writeDescriptorBufferStorageImage(
            m_xegtaoPrefilterBufferSet, frameIndex,
            descriptorBufferBindingOffset(m_xegtaoPrefilterDescriptorSetLayout, 2 + level),
            m_xegtaoDepthImageViews[level][aoFrameIndex], VK_IMAGE_LAYOUT_GENERAL);
    }

    // Main set.
    writeDescriptorBufferUniform(
        m_xegtaoMainBufferSet, frameIndex,
        descriptorBufferBindingOffset(m_xegtaoMainDescriptorSetLayout, 0),
        cameraAddress, cameraRange);
    writeDescriptorBufferCombinedImageSampler(
        m_xegtaoMainBufferSet, frameIndex,
        descriptorBufferBindingOffset(m_xegtaoMainDescriptorSetLayout, 1), 0,
        normalDepthInfo.imageView, normalDepthInfo.sampler, normalDepthInfo.imageLayout);
    for (uint32_t level = 0; level < kXeGtaoDepthMipCount; ++level) {
        writeDescriptorBufferCombinedImageSamplerArray(
            m_xegtaoMainBufferSet, frameIndex,
            descriptorBufferBindingOffset(m_xegtaoMainDescriptorSetLayout, 2), level,
            kXeGtaoDepthMipCount, m_xegtaoDepthImageViews[level][aoFrameIndex],
            m_xegtaoPointSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    writeDescriptorBufferStorageImage(
        m_xegtaoMainBufferSet, frameIndex,
        descriptorBufferBindingOffset(m_xegtaoMainDescriptorSetLayout, 3),
        m_xegtaoAoTermImageViews[aoFrameIndex], VK_IMAGE_LAYOUT_GENERAL);
    writeDescriptorBufferStorageImage(
        m_xegtaoMainBufferSet, frameIndex,
        descriptorBufferBindingOffset(m_xegtaoMainDescriptorSetLayout, 4),
        m_xegtaoBentNormalImageViews[aoFrameIndex], VK_IMAGE_LAYOUT_GENERAL);

    // Denoise set: pre-denoise AO in, edges from the bent normal's alpha, final
    // AO out into the image every existing consumer already reads.
    writeDescriptorBufferCombinedImageSampler(
        m_xegtaoDenoiseBufferSet, frameIndex,
        descriptorBufferBindingOffset(m_xegtaoDenoiseDescriptorSetLayout, 0), 0,
        m_xegtaoAoTermImageViews[aoFrameIndex], m_xegtaoPointSampler,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writeDescriptorBufferCombinedImageSampler(
        m_xegtaoDenoiseBufferSet, frameIndex,
        descriptorBufferBindingOffset(m_xegtaoDenoiseDescriptorSetLayout, 1), 0,
        m_xegtaoBentNormalImageViews[aoFrameIndex], m_xegtaoPointSampler,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    writeDescriptorBufferStorageImage(
        m_xegtaoDenoiseBufferSet, frameIndex,
        descriptorBufferBindingOffset(m_xegtaoDenoiseDescriptorSetLayout, 2),
        m_ssaoRawImageViews[aoFrameIndex], VK_IMAGE_LAYOUT_GENERAL);
}

} // namespace odai::render
