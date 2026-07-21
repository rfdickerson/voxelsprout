#include "render/backend/vulkan/renderer_backend.h"

#include "core/log.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace odai::render {

namespace {

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

constexpr uint32_t kBindlessTextureIndexDiffuse = 0u;
constexpr uint32_t kBindlessTextureIndexHdrResolved = 1u;
constexpr uint32_t kBindlessTextureIndexShadowAtlas = 2u;
constexpr uint32_t kBindlessTextureIndexNormalDepth = 3u;
constexpr uint32_t kBindlessTextureIndexSsaoBlur = 4u;
constexpr uint32_t kBindlessTextureIndexSsaoRaw = 5u;
constexpr uint32_t kBindlessTextureIndexPlantDiffuse = 6u;
constexpr uint32_t kBindlessTextureIndexSkyDaylight = 7u;
constexpr uint32_t kBindlessTextureIndexWaterNormal = 8u;
constexpr uint32_t kBindlessTextureIndexTerrainDetail = 9u;
constexpr uint32_t kBindlessTextureIndexFogMap = 10u;
constexpr uint32_t kBindlessTextureStaticCount = 11u;
constexpr uint32_t kAutoExposureHistogramBins = 64u;

} // namespace

bool RendererBackend::createDescriptorResources() {
    if (m_descriptorSetLayout == VK_NULL_HANDLE) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(13);

        VkDescriptorSetLayoutBinding mvpBinding{};
        mvpBinding.binding = 0;
        mvpBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        mvpBinding.descriptorCount = 1;
        mvpBinding.stageFlags =
            VK_SHADER_STAGE_VERTEX_BIT |
            VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
            VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
            VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(mvpBinding);

        VkDescriptorSetLayoutBinding diffuseTextureBinding{};
        diffuseTextureBinding.binding = 1;
        diffuseTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        diffuseTextureBinding.descriptorCount = 1;
        diffuseTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(diffuseTextureBinding);

        VkDescriptorSetLayoutBinding exposureStateBinding{};
        exposureStateBinding.binding = 2;
        exposureStateBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        exposureStateBinding.descriptorCount = 1;
        exposureStateBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(exposureStateBinding);

        VkDescriptorSetLayoutBinding hdrSceneBinding{};
        hdrSceneBinding.binding = 3;
        hdrSceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        hdrSceneBinding.descriptorCount = 1;
        hdrSceneBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(hdrSceneBinding);

        VkDescriptorSetLayoutBinding shadowMapBinding{};
        shadowMapBinding.binding = 4;
        shadowMapBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadowMapBinding.descriptorCount = 1;
        shadowMapBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(shadowMapBinding);

        VkDescriptorSetLayoutBinding waterRefractionBinding{};
        waterRefractionBinding.binding = 5;
        waterRefractionBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        waterRefractionBinding.descriptorCount = 1;
        waterRefractionBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(waterRefractionBinding);

        VkDescriptorSetLayoutBinding normalDepthBinding{};
        normalDepthBinding.binding = 6;
        normalDepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        normalDepthBinding.descriptorCount = 1;
        normalDepthBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(normalDepthBinding);

        VkDescriptorSetLayoutBinding ssaoBlurBinding{};
        ssaoBlurBinding.binding = 7;
        ssaoBlurBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ssaoBlurBinding.descriptorCount = 1;
        ssaoBlurBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(ssaoBlurBinding);

        VkDescriptorSetLayoutBinding ssaoRawBinding{};
        ssaoRawBinding.binding = 8;
        ssaoRawBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ssaoRawBinding.descriptorCount = 1;
        ssaoRawBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(ssaoRawBinding);

        VkDescriptorSetLayoutBinding voxelGiBinding{};
        voxelGiBinding.binding = 9;
        voxelGiBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        voxelGiBinding.descriptorCount = 1;
        voxelGiBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(voxelGiBinding);

        VkDescriptorSetLayoutBinding sunShaftBinding{};
        sunShaftBinding.binding = 10;
        sunShaftBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sunShaftBinding.descriptorCount = 1;
        sunShaftBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(sunShaftBinding);

        VkDescriptorSetLayoutBinding voxelGiOccupancyDebugBinding{};
        voxelGiOccupancyDebugBinding.binding = 11;
        voxelGiOccupancyDebugBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        voxelGiOccupancyDebugBinding.descriptorCount = 1;
        voxelGiOccupancyDebugBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings.push_back(voxelGiOccupancyDebugBinding);

        if (m_rayTracingRuntimeEnabled) {
            VkDescriptorSetLayoutBinding shadowSceneBinding{};
            shadowSceneBinding.binding = 12;
            shadowSceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            shadowSceneBinding.descriptorCount = 1;
            shadowSceneBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            bindings.push_back(shadowSceneBinding);
        }

        if (!createDescriptorSetLayout(
                bindings,
                m_descriptorSetLayout,
                "vkCreateDescriptorSetLayout",
                "renderer.descriptorSetLayout.main",
                nullptr,
                VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
            )) {
            return false;
        }
    }

    // Descriptor-buffer backing for the main per-frame set (set 0): camera UBO +
    // storage buffer + combined image samplers (+ optional accel structure).
    if (!m_mainBufferSet.valid()) {
        if (!createDescriptorBufferSet(
                m_descriptorSetLayout,
                kMaxFramesInFlight,
                VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                    VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
                "renderer.descriptorBuffer.main",
                m_mainBufferSet
            )) {
            return false;
        }
    }

    if (m_supportsBindlessDescriptors && m_bindlessTextureCapacity > 0) {
        if (m_bindlessDescriptorSetLayout == VK_NULL_HANDLE) {
            VkDescriptorSetLayoutBinding bindlessTexturesBinding{};
            bindlessTexturesBinding.binding = 0;
            bindlessTexturesBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindlessTexturesBinding.descriptorCount = m_bindlessTextureCapacity;
            bindlessTexturesBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            const VkDescriptorBindingFlags bindlessBindingFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
            VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsCreateInfo{};
            bindingFlagsCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
            bindingFlagsCreateInfo.bindingCount = 1;
            bindingFlagsCreateInfo.pBindingFlags = &bindlessBindingFlags;

            const std::array<VkDescriptorSetLayoutBinding, 1> bindlessBindings = {bindlessTexturesBinding};
            if (!createDescriptorSetLayout(
                    bindlessBindings,
                    m_bindlessDescriptorSetLayout,
                    "vkCreateDescriptorSetLayout(bindless)",
                    "renderer.descriptorSetLayout.bindless",
                    &bindingFlagsCreateInfo,
                    VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
                )) {
                return false;
            }
        }

        // Descriptor-buffer backing for the bindless texture array (set 1). One
        // region (shared across frames); partially-bound slots simply stay unwritten.
        if (!m_bindlessBufferSet.valid()) {
            if (!createDescriptorBufferSet(
                    m_bindlessDescriptorSetLayout,
                    1u,
                    VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                        VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT,
                    "renderer.descriptorBuffer.bindless",
                    m_bindlessBufferSet
                )) {
                return false;
            }
        }
    }

    return true;
}

void RendererBackend::updateFrameDescriptorSets(
    uint32_t aoFrameIndex,
    const VkDescriptorBufferInfo& cameraBufferInfo,
    VkBuffer autoExposureHistogramBuffer,
    VkBuffer autoExposureStateBuffer,
    const VkDescriptorBufferInfo* voxelGiChunkMetaBufferInfo,
    const VkDescriptorBufferInfo* voxelGiChunkVoxelBufferInfo
) {
    // Camera UBO device address (frame-arena slice) for descriptor-buffer writes.
    VkBufferDeviceAddressInfo cameraAddressInfo{};
    cameraAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    cameraAddressInfo.buffer = cameraBufferInfo.buffer;
    const VkDeviceAddress cameraDeviceAddress =
        (cameraBufferInfo.buffer != VK_NULL_HANDLE)
            ? vkGetBufferDeviceAddress(m_device, &cameraAddressInfo) + cameraBufferInfo.offset
            : 0;

    VkDescriptorImageInfo hdrSceneImageInfo{};
    hdrSceneImageInfo.sampler = m_hdrResolveSampler;
    hdrSceneImageInfo.imageView = m_hdrResolveSampleImageViews[aoFrameIndex];
    hdrSceneImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo diffuseTextureImageInfo{};
    diffuseTextureImageInfo.sampler = m_diffuseTextureSampler;
    diffuseTextureImageInfo.imageView = m_diffuseTextureImageView;
    diffuseTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo plantDiffuseTextureImageInfo{};
    plantDiffuseTextureImageInfo.sampler = m_diffuseTexturePlantSampler;
    plantDiffuseTextureImageInfo.imageView =
        (m_plantDiffuseTextureImageView != VK_NULL_HANDLE) ? m_plantDiffuseTextureImageView : m_diffuseTextureImageView;
    plantDiffuseTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo morrowindSkyTextureImageInfo{};
    morrowindSkyTextureImageInfo.sampler =
        (m_morrowindSkyTextureSampler != VK_NULL_HANDLE) ? m_morrowindSkyTextureSampler : m_diffuseTextureSampler;
    morrowindSkyTextureImageInfo.imageView =
        (m_morrowindSkyTextureImageView != VK_NULL_HANDLE) ? m_morrowindSkyTextureImageView : m_diffuseTextureImageView;
    morrowindSkyTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo waterNormalTextureImageInfo{};
    waterNormalTextureImageInfo.sampler =
        (m_waterNormalTextureSampler != VK_NULL_HANDLE) ? m_waterNormalTextureSampler : m_diffuseTextureSampler;
    waterNormalTextureImageInfo.imageView =
        (m_waterNormalTextureImageView != VK_NULL_HANDLE) ? m_waterNormalTextureImageView : m_diffuseTextureImageView;
    waterNormalTextureImageInfo.imageLayout = m_hostCopyFinalLayout;

    VkDescriptorImageInfo terrainDetailTextureImageInfo{};
    terrainDetailTextureImageInfo.sampler =
        (m_terrainDetailTextureSampler != VK_NULL_HANDLE) ? m_terrainDetailTextureSampler : m_diffuseTextureSampler;
    terrainDetailTextureImageInfo.imageView =
        (m_terrainDetailTextureImageView != VK_NULL_HANDLE) ? m_terrainDetailTextureImageView : m_diffuseTextureImageView;
    terrainDetailTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo fogMapTextureImageInfo{};
    fogMapTextureImageInfo.sampler =
        (m_fogMapSampler != VK_NULL_HANDLE) ? m_fogMapSampler : m_diffuseTextureSampler;
    fogMapTextureImageInfo.imageView =
        (m_fogMapTextureResource.imageView != VK_NULL_HANDLE) ? m_fogMapTextureResource.imageView : m_diffuseTextureImageView;
    fogMapTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo shadowMapImageInfo{};
    shadowMapImageInfo.sampler = m_shadowDepthSampler;
    shadowMapImageInfo.imageView = m_shadowDepthImageView;
    shadowMapImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo waterRefractionImageInfo{};
    waterRefractionImageInfo.sampler = m_hdrResolveSampler;
    waterRefractionImageInfo.imageView =
        (aoFrameIndex < m_waterRefractionImageViews.size()) ? m_waterRefractionImageViews[aoFrameIndex] : VK_NULL_HANDLE;
    waterRefractionImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo normalDepthImageInfo{};
    normalDepthImageInfo.sampler = m_normalDepthSampler;
    normalDepthImageInfo.imageView = m_normalDepthImageViews[aoFrameIndex];
    normalDepthImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo ssaoBlurImageInfo{};
    ssaoBlurImageInfo.sampler = m_ssaoSampler;
    ssaoBlurImageInfo.imageView = m_ssaoBlurImageViews[aoFrameIndex];
    ssaoBlurImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo ssaoRawImageInfo{};
    ssaoRawImageInfo.sampler = m_ssaoSampler;
    ssaoRawImageInfo.imageView = m_ssaoRawImageViews[aoFrameIndex];
    ssaoRawImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo voxelGiVolumeImageInfo{};
    voxelGiVolumeImageInfo.sampler = m_voxelGiSampler;
    voxelGiVolumeImageInfo.imageView = m_voxelGiImageViews[1];
    voxelGiVolumeImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo voxelGiOccupancyDebugImageInfo{};
    voxelGiOccupancyDebugImageInfo.sampler = m_voxelGiOccupancySampler;
    voxelGiOccupancyDebugImageInfo.imageView = m_voxelGiOccupancyImageView;
    voxelGiOccupancyDebugImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo sunShaftImageInfo{};
    sunShaftImageInfo.sampler = m_sunShaftSampler;
    sunShaftImageInfo.imageView =
        (aoFrameIndex < m_sunShaftImageViews.size()) ? m_sunShaftImageViews[aoFrameIndex] : VK_NULL_HANDLE;
    sunShaftImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo autoExposureStateBufferInfo{};
    autoExposureStateBufferInfo.buffer = autoExposureStateBuffer;
    autoExposureStateBufferInfo.offset = 0;
    autoExposureStateBufferInfo.range = sizeof(float) * 4u;
    const bool hasRayTracingSceneDescriptor = m_rayTracingRuntimeEnabled && m_rtTlas.handle != VK_NULL_HANDLE;
    VkWriteDescriptorSetAccelerationStructureKHR rayTracingSceneWriteInfo{};
    rayTracingSceneWriteInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    rayTracingSceneWriteInfo.accelerationStructureCount = hasRayTracingSceneDescriptor ? 1u : 0u;
    rayTracingSceneWriteInfo.pAccelerationStructures = hasRayTracingSceneDescriptor ? &m_rtTlas.handle : nullptr;

    if (m_mainBufferSet.valid()) {
        const uint32_t region = m_currentFrame;
        const VkDescriptorSetLayout layout = m_descriptorSetLayout;
        auto mainOffset = [&](uint32_t binding) { return descriptorBufferBindingOffset(layout, binding); };
        auto sampler = [&](uint32_t binding, const VkDescriptorImageInfo& info) {
            writeDescriptorBufferCombinedImageSampler(
                m_mainBufferSet, region, mainOffset(binding), 0, info.imageView, info.sampler, info.imageLayout);
        };
        // Camera UBO (binding 0) and exposure-state storage buffer (binding 2).
        writeDescriptorBufferUniform(m_mainBufferSet, region, mainOffset(0), cameraDeviceAddress, cameraBufferInfo.range);
        VkBufferDeviceAddressInfo exposureAddrInfo{};
        exposureAddrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        exposureAddrInfo.buffer = autoExposureStateBufferInfo.buffer;
        const VkDeviceAddress exposureAddress = (autoExposureStateBufferInfo.buffer != VK_NULL_HANDLE)
            ? vkGetBufferDeviceAddress(m_device, &exposureAddrInfo) : 0;
        writeDescriptorBufferStorage(m_mainBufferSet, region, mainOffset(2), exposureAddress, autoExposureStateBufferInfo.range);
        // Combined image samplers (bindings 1, 3-11).
        sampler(1, diffuseTextureImageInfo);
        sampler(3, hdrSceneImageInfo);
        sampler(4, shadowMapImageInfo);
        sampler(5, waterRefractionImageInfo);
        sampler(6, normalDepthImageInfo);
        sampler(7, ssaoBlurImageInfo);
        sampler(8, ssaoRawImageInfo);
        sampler(9, voxelGiVolumeImageInfo);
        sampler(10, sunShaftImageInfo);
        sampler(11, voxelGiOccupancyDebugImageInfo);
        // Ray-traced scene acceleration structure (binding 12) when RT is live.
        if (hasRayTracingSceneDescriptor) {
            writeDescriptorBufferAccelerationStructure(m_mainBufferSet, region, mainOffset(12), m_rtTlas.deviceAddress);
        }
    }

    if (m_voxelGiComputeAvailable && m_voxelGiBufferSet.valid()) {
        const uint32_t region = m_currentFrame;
        const VkDescriptorSetLayout layout = m_voxelGiDescriptorSetLayout;
        auto bindOffset = [&](uint32_t binding) {
            return descriptorBufferBindingOffset(layout, binding);
        };
        auto bufferAddress = [&](const VkDescriptorBufferInfo& info) -> VkDeviceAddress {
            if (info.buffer == VK_NULL_HANDLE) {
                return 0;
            }
            VkBufferDeviceAddressInfo addrInfo{};
            addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addrInfo.buffer = info.buffer;
            return vkGetBufferDeviceAddress(m_device, &addrInfo) + info.offset;
        };

        // Camera UBO + shadow sampler.
        writeDescriptorBufferUniform(m_voxelGiBufferSet, region, bindOffset(0), cameraDeviceAddress, cameraBufferInfo.range);
        writeDescriptorBufferCombinedImageSampler(m_voxelGiBufferSet, region, bindOffset(1), 0,
            shadowMapImageInfo.imageView, shadowMapImageInfo.sampler, shadowMapImageInfo.imageLayout);
        // Radiance A (storage) / read (sampled) / B (storage).
        writeDescriptorBufferStorageImage(m_voxelGiBufferSet, region, bindOffset(2), m_voxelGiImageViews[0], VK_IMAGE_LAYOUT_GENERAL);
        writeDescriptorBufferSampledImage(m_voxelGiBufferSet, region, bindOffset(3), m_voxelGiImageViews[0], VK_IMAGE_LAYOUT_GENERAL);
        writeDescriptorBufferStorageImage(m_voxelGiBufferSet, region, bindOffset(4), m_voxelGiImageViews[1], VK_IMAGE_LAYOUT_GENERAL);
        // Occupancy sampled (read-only) then the 6 surface faces (storage).
        writeDescriptorBufferSampledImage(m_voxelGiBufferSet, region, bindOffset(5), m_voxelGiOccupancyImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        for (uint32_t faceIndex = 0; faceIndex < 6u; ++faceIndex) {
            writeDescriptorBufferStorageImage(m_voxelGiBufferSet, region, bindOffset(6u + faceIndex),
                m_voxelGiSurfaceFaceImageViews[faceIndex], VK_IMAGE_LAYOUT_GENERAL);
        }
        // Sky exposure + occupancy storage view.
        writeDescriptorBufferStorageImage(m_voxelGiBufferSet, region, bindOffset(12), m_voxelGiSkyExposureImageView, VK_IMAGE_LAYOUT_GENERAL);
        writeDescriptorBufferStorageImage(m_voxelGiBufferSet, region, bindOffset(13), m_voxelGiOccupancyImageView, VK_IMAGE_LAYOUT_GENERAL);
        // Chunk meta / voxel storage buffers (fall back to the exposure buffer when absent).
        VkDescriptorBufferInfo voxelGiFallbackStorageInfo{};
        voxelGiFallbackStorageInfo.buffer = autoExposureStateBufferInfo.buffer;
        voxelGiFallbackStorageInfo.offset = 0;
        voxelGiFallbackStorageInfo.range = autoExposureStateBufferInfo.range;
        const VkDescriptorBufferInfo& voxelGiChunkMetaInfo =
            (voxelGiChunkMetaBufferInfo != nullptr) ? *voxelGiChunkMetaBufferInfo : voxelGiFallbackStorageInfo;
        const VkDescriptorBufferInfo& voxelGiChunkVoxelInfo =
            (voxelGiChunkVoxelBufferInfo != nullptr) ? *voxelGiChunkVoxelBufferInfo : voxelGiFallbackStorageInfo;
        writeDescriptorBufferStorage(m_voxelGiBufferSet, region, bindOffset(14),
            bufferAddress(voxelGiChunkMetaInfo), voxelGiChunkMetaInfo.range);
        writeDescriptorBufferStorage(m_voxelGiBufferSet, region, bindOffset(15),
            bufferAddress(voxelGiChunkVoxelInfo), voxelGiChunkVoxelInfo.range);

        // Ray-traced surface tracing + ReSTIR reservoirs (only when RT is live).
        const bool hasVoxelGiRayTracingSceneDescriptor =
            m_rayTracingRuntimeEnabled && m_rtTlas.handle != VK_NULL_HANDLE;
        if (hasVoxelGiRayTracingSceneDescriptor) {
            writeDescriptorBufferAccelerationStructure(m_voxelGiBufferSet, region, bindOffset(16), m_rtTlas.deviceAddress);
        }
        const VkBuffer restirCurrent = m_bufferAllocator.getBuffer(m_voxelGiRestirReservoirCurrentBufferHandle);
        const VkBuffer restirPrevious = m_bufferAllocator.getBuffer(m_voxelGiRestirReservoirPreviousBufferHandle);
        const VkBuffer restirScratch = m_bufferAllocator.getBuffer(m_voxelGiRestirReservoirScratchBufferHandle);
        if (m_rayTracingRuntimeEnabled &&
            restirCurrent != VK_NULL_HANDLE && restirPrevious != VK_NULL_HANDLE && restirScratch != VK_NULL_HANDLE) {
            writeDescriptorBufferStorage(m_voxelGiBufferSet, region, bindOffset(17),
                m_bufferAllocator.getDeviceAddress(m_voxelGiRestirReservoirCurrentBufferHandle),
                m_bufferAllocator.getSize(m_voxelGiRestirReservoirCurrentBufferHandle));
            writeDescriptorBufferStorage(m_voxelGiBufferSet, region, bindOffset(18),
                m_bufferAllocator.getDeviceAddress(m_voxelGiRestirReservoirPreviousBufferHandle),
                m_bufferAllocator.getSize(m_voxelGiRestirReservoirPreviousBufferHandle));
            writeDescriptorBufferStorage(m_voxelGiBufferSet, region, bindOffset(19),
                m_bufferAllocator.getDeviceAddress(m_voxelGiRestirReservoirScratchBufferHandle),
                m_bufferAllocator.getSize(m_voxelGiRestirReservoirScratchBufferHandle));
        }
    }

    if (m_autoExposureComputeAvailable &&
        m_autoExposureBufferSet.valid() &&
        autoExposureHistogramBuffer != VK_NULL_HANDLE &&
        autoExposureStateBuffer != VK_NULL_HANDLE) {
        // Descriptor-buffer writes are cheap (a memcpy into mapped memory), so we
        // write the region unconditionally each frame rather than diff against a key.
        const uint32_t region = m_currentFrame;
        const VkDeviceSize hdrOffset = descriptorBufferBindingOffset(m_autoExposureDescriptorSetLayout, 0);
        const VkDeviceSize histogramOffset = descriptorBufferBindingOffset(m_autoExposureDescriptorSetLayout, 1);
        const VkDeviceSize stateOffset = descriptorBufferBindingOffset(m_autoExposureDescriptorSetLayout, 2);
        writeDescriptorBufferCombinedImageSampler(
            m_autoExposureBufferSet, region, hdrOffset, 0,
            hdrSceneImageInfo.imageView, hdrSceneImageInfo.sampler, hdrSceneImageInfo.imageLayout);
        writeDescriptorBufferStorage(
            m_autoExposureBufferSet, region, histogramOffset,
            m_bufferAllocator.getDeviceAddress(m_autoExposureHistogramBufferHandle),
            static_cast<VkDeviceSize>(kAutoExposureHistogramBins * sizeof(uint32_t)));
        writeDescriptorBufferStorage(
            m_autoExposureBufferSet, region, stateOffset,
            m_bufferAllocator.getDeviceAddress(m_autoExposureStateBufferHandle),
            sizeof(float) * 4u);
    }

    if (m_sunShaftComputeAvailable &&
        m_sunShaftBufferSet.valid() &&
        aoFrameIndex < m_sunShaftImageViews.size() &&
        m_sunShaftImageViews[aoFrameIndex] != VK_NULL_HANDLE) {
        const uint32_t region = m_currentFrame;
        const VkDeviceSize cameraOffset = descriptorBufferBindingOffset(m_sunShaftDescriptorSetLayout, 0);
        const VkDeviceSize normalDepthOffset = descriptorBufferBindingOffset(m_sunShaftDescriptorSetLayout, 1);
        const VkDeviceSize shadowOffset = descriptorBufferBindingOffset(m_sunShaftDescriptorSetLayout, 2);
        const VkDeviceSize outputOffset = descriptorBufferBindingOffset(m_sunShaftDescriptorSetLayout, 3);
        writeDescriptorBufferUniform(
            m_sunShaftBufferSet, region, cameraOffset, cameraDeviceAddress, cameraBufferInfo.range);
        writeDescriptorBufferCombinedImageSampler(
            m_sunShaftBufferSet, region, normalDepthOffset, 0,
            normalDepthImageInfo.imageView, normalDepthImageInfo.sampler, normalDepthImageInfo.imageLayout);
        writeDescriptorBufferCombinedImageSampler(
            m_sunShaftBufferSet, region, shadowOffset, 0,
            shadowMapImageInfo.imageView, shadowMapImageInfo.sampler, shadowMapImageInfo.imageLayout);
        writeDescriptorBufferStorageImage(
            m_sunShaftBufferSet, region, outputOffset,
            m_sunShaftImageViews[aoFrameIndex], VK_IMAGE_LAYOUT_GENERAL);
    }

    if (m_ssaoBufferSet.valid() &&
        aoFrameIndex < m_ssaoRawImageViews.size() &&
        m_ssaoRawImageViews[aoFrameIndex] != VK_NULL_HANDLE) {
        const uint32_t region = m_currentFrame;
        const VkDeviceSize cameraOffset = descriptorBufferBindingOffset(m_ssaoDescriptorSetLayout, 0);
        const VkDeviceSize normalDepthOffset = descriptorBufferBindingOffset(m_ssaoDescriptorSetLayout, 1);
        const VkDeviceSize outputOffset = descriptorBufferBindingOffset(m_ssaoDescriptorSetLayout, 2);
        writeDescriptorBufferUniform(
            m_ssaoBufferSet, region, cameraOffset, cameraDeviceAddress, cameraBufferInfo.range);
        writeDescriptorBufferCombinedImageSampler(
            m_ssaoBufferSet, region, normalDepthOffset, 0,
            normalDepthImageInfo.imageView, normalDepthImageInfo.sampler, normalDepthImageInfo.imageLayout);
        writeDescriptorBufferStorageImage(
            m_ssaoBufferSet, region, outputOffset,
            m_ssaoRawImageViews[aoFrameIndex], VK_IMAGE_LAYOUT_GENERAL);
    }

    if (m_ssaoBlurBufferSet.valid() &&
        aoFrameIndex < m_ssaoBlurImageViews.size() &&
        m_ssaoBlurImageViews[aoFrameIndex] != VK_NULL_HANDLE) {
        const uint32_t region = m_currentFrame;
        const VkDeviceSize normalDepthOffset = descriptorBufferBindingOffset(m_ssaoBlurDescriptorSetLayout, 0);
        const VkDeviceSize ssaoRawOffset = descriptorBufferBindingOffset(m_ssaoBlurDescriptorSetLayout, 1);
        const VkDeviceSize outputOffset = descriptorBufferBindingOffset(m_ssaoBlurDescriptorSetLayout, 2);
        writeDescriptorBufferCombinedImageSampler(
            m_ssaoBlurBufferSet, region, normalDepthOffset, 0,
            normalDepthImageInfo.imageView, normalDepthImageInfo.sampler, normalDepthImageInfo.imageLayout);
        writeDescriptorBufferCombinedImageSampler(
            m_ssaoBlurBufferSet, region, ssaoRawOffset, 0,
            ssaoRawImageInfo.imageView, ssaoRawImageInfo.sampler, ssaoRawImageInfo.imageLayout);
        writeDescriptorBufferStorageImage(
            m_ssaoBlurBufferSet, region, outputOffset,
            m_ssaoBlurImageViews[aoFrameIndex], VK_IMAGE_LAYOUT_GENERAL);
    }

    const std::size_t bindlessTextureCount =
        kBindlessTextureStaticCount + std::min<std::size_t>(
            m_importedTextureResources.size(),
            (m_bindlessTextureCapacity > kBindlessTextureStaticCount)
                ? static_cast<std::size_t>(m_bindlessTextureCapacity - kBindlessTextureStaticCount)
                : 0u);
    if (m_bindlessBufferSet.valid() && m_bindlessTextureCapacity >= bindlessTextureCount) {
        std::vector<VkDescriptorImageInfo> bindlessImageInfos(bindlessTextureCount);
        bindlessImageInfos[kBindlessTextureIndexDiffuse] = diffuseTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexHdrResolved] = hdrSceneImageInfo;
        bindlessImageInfos[kBindlessTextureIndexShadowAtlas] = shadowMapImageInfo;
        bindlessImageInfos[kBindlessTextureIndexNormalDepth] = normalDepthImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSsaoBlur] = ssaoBlurImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSsaoRaw] = ssaoRawImageInfo;
        bindlessImageInfos[kBindlessTextureIndexPlantDiffuse] = plantDiffuseTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSkyDaylight] = morrowindSkyTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexWaterNormal] = waterNormalTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexTerrainDetail] = terrainDetailTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexFogMap] = fogMapTextureImageInfo;
        for (std::size_t textureIndex = 0; textureIndex < m_importedTextureResources.size(); ++textureIndex) {
            const std::size_t bindlessIndex = kBindlessTextureStaticCount + textureIndex;
            if (bindlessIndex >= bindlessImageInfos.size()) {
                break;
            }
            const ImportedTextureResource& texture = m_importedTextureResources[textureIndex];
            if (texture.imageView == VK_NULL_HANDLE || m_importedTextureSampler == VK_NULL_HANDLE) {
                continue;
            }
            bindlessImageInfos[bindlessIndex].sampler = m_importedTextureSampler;
            bindlessImageInfos[bindlessIndex].imageView = texture.imageView;
            bindlessImageInfos[bindlessIndex].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        // Bindless array binding 0: one combined-image-sampler descriptor per array
        // element (single region shared across frames). Empty slots stay unwritten
        // (partially bound); the shader never indexes them.
        const VkDeviceSize bindlessBindingOffset = descriptorBufferBindingOffset(m_bindlessDescriptorSetLayout, 0);
        for (std::size_t index = 0; index < bindlessImageInfos.size(); ++index) {
            const VkDescriptorImageInfo& info = bindlessImageInfos[index];
            if (info.imageView == VK_NULL_HANDLE || info.sampler == VK_NULL_HANDLE) {
                continue;
            }
            writeDescriptorBufferCombinedImageSamplerArray(
                m_bindlessBufferSet, 0u, bindlessBindingOffset,
                static_cast<uint32_t>(index), m_bindlessTextureCapacity,
                info.imageView, info.sampler, info.imageLayout);
        }
    }
}

} // namespace odai::render
