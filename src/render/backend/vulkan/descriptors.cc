#include "render/backend/vulkan/renderer_backend.h"

#include "core/log.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace voxelsprout::render {

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
constexpr uint32_t kBindlessTextureStaticCount = 7u;
constexpr uint32_t kAutoExposureHistogramBins = 64u;

} // namespace

bool RendererBackend::createDescriptorResources() {
    if (m_descriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding mvpBinding{};
        mvpBinding.binding = 0;
        mvpBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        mvpBinding.descriptorCount = 1;
        mvpBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding diffuseTextureBinding{};
        diffuseTextureBinding.binding = 1;
        diffuseTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        diffuseTextureBinding.descriptorCount = 1;
        diffuseTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding exposureStateBinding{};
        exposureStateBinding.binding = 2;
        exposureStateBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        exposureStateBinding.descriptorCount = 1;
        exposureStateBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding hdrSceneBinding{};
        hdrSceneBinding.binding = 3;
        hdrSceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        hdrSceneBinding.descriptorCount = 1;
        hdrSceneBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding shadowMapBinding{};
        shadowMapBinding.binding = 4;
        shadowMapBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadowMapBinding.descriptorCount = 1;
        shadowMapBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding normalDepthBinding{};
        normalDepthBinding.binding = 6;
        normalDepthBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        normalDepthBinding.descriptorCount = 1;
        normalDepthBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding ssaoBlurBinding{};
        ssaoBlurBinding.binding = 7;
        ssaoBlurBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ssaoBlurBinding.descriptorCount = 1;
        ssaoBlurBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding ssaoRawBinding{};
        ssaoRawBinding.binding = 8;
        ssaoRawBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ssaoRawBinding.descriptorCount = 1;
        ssaoRawBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding voxelGiBinding{};
        voxelGiBinding.binding = 9;
        voxelGiBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        voxelGiBinding.descriptorCount = 1;
        voxelGiBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding sunShaftBinding{};
        sunShaftBinding.binding = 10;
        sunShaftBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sunShaftBinding.descriptorCount = 1;
        sunShaftBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding voxelGiOccupancyDebugBinding{};
        voxelGiOccupancyDebugBinding.binding = 11;
        voxelGiOccupancyDebugBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        voxelGiOccupancyDebugBinding.descriptorCount = 1;
        voxelGiOccupancyDebugBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        const std::array<VkDescriptorSetLayoutBinding, 11> bindings = {
            mvpBinding,
            diffuseTextureBinding,
            exposureStateBinding,
            hdrSceneBinding,
            shadowMapBinding,
            normalDepthBinding,
            ssaoBlurBinding,
            ssaoRawBinding,
            voxelGiBinding,
            sunShaftBinding,
            voxelGiOccupancyDebugBinding
        };

        if (!createDescriptorSetLayout(
                bindings,
                m_descriptorSetLayout,
                "vkCreateDescriptorSetLayout",
                "renderer.descriptorSetLayout.main"
            )) {
            return false;
        }
    }

    if (m_descriptorPool == VK_NULL_HANDLE) {
        const std::array<VkDescriptorPoolSize, 3> poolSizes = {
            VkDescriptorPoolSize{
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                kMaxFramesInFlight
            },
            VkDescriptorPoolSize{
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                9 * kMaxFramesInFlight
            },
            VkDescriptorPoolSize{
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                kMaxFramesInFlight
            }
        };

        if (!createDescriptorPool(
                poolSizes,
                kMaxFramesInFlight,
                m_descriptorPool,
                "vkCreateDescriptorPool",
                "renderer.descriptorPool.main"
            )) {
            return false;
        }
    }

    if (!allocatePerFrameDescriptorSets(
            m_descriptorPool,
            m_descriptorSetLayout,
            std::span<VkDescriptorSet>(m_descriptorSets),
            "vkAllocateDescriptorSets",
            "renderer.descriptorSet.frame"
        )) {
        return false;
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
                    &bindingFlagsCreateInfo
                )) {
                return false;
            }
        }

        if (m_bindlessDescriptorPool == VK_NULL_HANDLE) {
            const std::array<VkDescriptorPoolSize, 1> bindlessPoolSizes = {VkDescriptorPoolSize{
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                m_bindlessTextureCapacity
            }};
            if (!createDescriptorPool(
                    bindlessPoolSizes,
                    1,
                    m_bindlessDescriptorPool,
                    "vkCreateDescriptorPool(bindless)",
                    "renderer.descriptorPool.bindless"
                )) {
                return false;
            }
        }

        if (m_bindlessDescriptorSet == VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo bindlessAllocateInfo{};
            bindlessAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            bindlessAllocateInfo.descriptorPool = m_bindlessDescriptorPool;
            bindlessAllocateInfo.descriptorSetCount = 1;
            bindlessAllocateInfo.pSetLayouts = &m_bindlessDescriptorSetLayout;
            const VkResult bindlessAllocateResult = vkAllocateDescriptorSets(
                m_device,
                &bindlessAllocateInfo,
                &m_bindlessDescriptorSet
            );
            if (bindlessAllocateResult != VK_SUCCESS) {
                logVkFailure("vkAllocateDescriptorSets(bindless)", bindlessAllocateResult);
                return false;
            }
            setObjectName(
                VK_OBJECT_TYPE_DESCRIPTOR_SET,
                vkHandleToUint64(m_bindlessDescriptorSet),
                "renderer.descriptorSet.bindless"
            );
        }
    }

    return true;
}

RendererBackend::BoundDescriptorSets RendererBackend::updateFrameDescriptorSets(
    uint32_t aoFrameIndex,
    const VkDescriptorBufferInfo& cameraBufferInfo,
    VkBuffer autoExposureHistogramBuffer,
    VkBuffer autoExposureStateBuffer
) {
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

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
    plantDiffuseTextureImageInfo.imageView = m_diffuseTextureImageView;
    plantDiffuseTextureImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo shadowMapImageInfo{};
    shadowMapImageInfo.sampler = m_shadowDepthSampler;
    shadowMapImageInfo.imageView = m_shadowDepthImageView;
    shadowMapImageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

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

    std::array<VkWriteDescriptorSet, 11> writes{};
    writes[0] = write;
    writes[0].dstSet = m_descriptorSets[m_currentFrame];
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    writes[0].pBufferInfo = &cameraBufferInfo;

    writes[1] = write;
    writes[1].dstSet = m_descriptorSets[m_currentFrame];
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &diffuseTextureImageInfo;

    writes[2] = write;
    writes[2].dstSet = m_descriptorSets[m_currentFrame];
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &autoExposureStateBufferInfo;

    writes[3] = write;
    writes[3].dstSet = m_descriptorSets[m_currentFrame];
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[3].pImageInfo = &hdrSceneImageInfo;

    writes[4] = write;
    writes[4].dstSet = m_descriptorSets[m_currentFrame];
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[4].pImageInfo = &shadowMapImageInfo;

    writes[5] = write;
    writes[5].dstSet = m_descriptorSets[m_currentFrame];
    writes[5].dstBinding = 6;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[5].pImageInfo = &normalDepthImageInfo;

    writes[6] = write;
    writes[6].dstSet = m_descriptorSets[m_currentFrame];
    writes[6].dstBinding = 7;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[6].pImageInfo = &ssaoBlurImageInfo;

    writes[7] = write;
    writes[7].dstSet = m_descriptorSets[m_currentFrame];
    writes[7].dstBinding = 8;
    writes[7].descriptorCount = 1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[7].pImageInfo = &ssaoRawImageInfo;

    writes[8] = write;
    writes[8].dstSet = m_descriptorSets[m_currentFrame];
    writes[8].dstBinding = 9;
    writes[8].descriptorCount = 1;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[8].pImageInfo = &voxelGiVolumeImageInfo;

    writes[9] = write;
    writes[9].dstSet = m_descriptorSets[m_currentFrame];
    writes[9].dstBinding = 10;
    writes[9].descriptorCount = 1;
    writes[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[9].pImageInfo = &sunShaftImageInfo;

    writes[10] = write;
    writes[10].dstSet = m_descriptorSets[m_currentFrame];
    writes[10].dstBinding = 11;
    writes[10].descriptorCount = 1;
    writes[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[10].pImageInfo = &voxelGiOccupancyDebugImageInfo;

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    if (m_voxelGiComputeAvailable && m_voxelGiDescriptorSets[m_currentFrame] != VK_NULL_HANDLE) {
        VkDescriptorImageInfo voxelGiStorageAInfo{};
        voxelGiStorageAInfo.sampler = VK_NULL_HANDLE;
        voxelGiStorageAInfo.imageView = m_voxelGiImageViews[0];
        voxelGiStorageAInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo voxelGiStorageReadInfo{};
        voxelGiStorageReadInfo.sampler = VK_NULL_HANDLE;
        voxelGiStorageReadInfo.imageView = m_voxelGiImageViews[0];
        voxelGiStorageReadInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo voxelGiStorageBInfo{};
        voxelGiStorageBInfo.sampler = VK_NULL_HANDLE;
        voxelGiStorageBInfo.imageView = m_voxelGiImageViews[1];
        voxelGiStorageBInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorImageInfo voxelGiOccupancyInfo{};
        voxelGiOccupancyInfo.sampler = VK_NULL_HANDLE;
        voxelGiOccupancyInfo.imageView = m_voxelGiOccupancyImageView;
        voxelGiOccupancyInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        std::array<VkDescriptorImageInfo, 6> voxelGiSurfaceFaceInfos{};
        for (std::size_t faceIndex = 0; faceIndex < voxelGiSurfaceFaceInfos.size(); ++faceIndex) {
            voxelGiSurfaceFaceInfos[faceIndex].sampler = VK_NULL_HANDLE;
            voxelGiSurfaceFaceInfos[faceIndex].imageView = m_voxelGiSurfaceFaceImageViews[faceIndex];
            voxelGiSurfaceFaceInfos[faceIndex].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }

        VkDescriptorImageInfo voxelGiSkyExposureInfo{};
        voxelGiSkyExposureInfo.sampler = VK_NULL_HANDLE;
        voxelGiSkyExposureInfo.imageView = m_voxelGiSkyExposureImageView;
        voxelGiSkyExposureInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 13> voxelGiWrites{};
        voxelGiWrites[0] = write;
        voxelGiWrites[0].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[0].dstBinding = 0;
        voxelGiWrites[0].descriptorCount = 1;
        voxelGiWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        voxelGiWrites[0].pBufferInfo = &cameraBufferInfo;

        voxelGiWrites[1] = write;
        voxelGiWrites[1].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[1].dstBinding = 1;
        voxelGiWrites[1].descriptorCount = 1;
        voxelGiWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        voxelGiWrites[1].pImageInfo = &shadowMapImageInfo;

        voxelGiWrites[2] = write;
        voxelGiWrites[2].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[2].dstBinding = 2;
        voxelGiWrites[2].descriptorCount = 1;
        voxelGiWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[2].pImageInfo = &voxelGiStorageAInfo;

        voxelGiWrites[3] = write;
        voxelGiWrites[3].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[3].dstBinding = 3;
        voxelGiWrites[3].descriptorCount = 1;
        voxelGiWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        voxelGiWrites[3].pImageInfo = &voxelGiStorageReadInfo;

        voxelGiWrites[4] = write;
        voxelGiWrites[4].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[4].dstBinding = 4;
        voxelGiWrites[4].descriptorCount = 1;
        voxelGiWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[4].pImageInfo = &voxelGiStorageBInfo;

        voxelGiWrites[5] = write;
        voxelGiWrites[5].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[5].dstBinding = 5;
        voxelGiWrites[5].descriptorCount = 1;
        voxelGiWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        voxelGiWrites[5].pImageInfo = &voxelGiOccupancyInfo;

        voxelGiWrites[6] = write;
        voxelGiWrites[6].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[6].dstBinding = 6;
        voxelGiWrites[6].descriptorCount = 1;
        voxelGiWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[6].pImageInfo = &voxelGiSurfaceFaceInfos[0];

        voxelGiWrites[7] = write;
        voxelGiWrites[7].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[7].dstBinding = 7;
        voxelGiWrites[7].descriptorCount = 1;
        voxelGiWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[7].pImageInfo = &voxelGiSurfaceFaceInfos[1];

        voxelGiWrites[8] = write;
        voxelGiWrites[8].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[8].dstBinding = 8;
        voxelGiWrites[8].descriptorCount = 1;
        voxelGiWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[8].pImageInfo = &voxelGiSurfaceFaceInfos[2];

        voxelGiWrites[9] = write;
        voxelGiWrites[9].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[9].dstBinding = 9;
        voxelGiWrites[9].descriptorCount = 1;
        voxelGiWrites[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[9].pImageInfo = &voxelGiSurfaceFaceInfos[3];

        voxelGiWrites[10] = write;
        voxelGiWrites[10].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[10].dstBinding = 10;
        voxelGiWrites[10].descriptorCount = 1;
        voxelGiWrites[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[10].pImageInfo = &voxelGiSurfaceFaceInfos[4];

        voxelGiWrites[11] = write;
        voxelGiWrites[11].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[11].dstBinding = 11;
        voxelGiWrites[11].descriptorCount = 1;
        voxelGiWrites[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[11].pImageInfo = &voxelGiSurfaceFaceInfos[5];

        voxelGiWrites[12] = write;
        voxelGiWrites[12].dstSet = m_voxelGiDescriptorSets[m_currentFrame];
        voxelGiWrites[12].dstBinding = 12;
        voxelGiWrites[12].descriptorCount = 1;
        voxelGiWrites[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        voxelGiWrites[12].pImageInfo = &voxelGiSkyExposureInfo;

        vkUpdateDescriptorSets(
            m_device,
            static_cast<uint32_t>(voxelGiWrites.size()),
            voxelGiWrites.data(),
            0,
            nullptr
        );
    }

    if (m_autoExposureComputeAvailable &&
        m_autoExposureDescriptorSets[m_currentFrame] != VK_NULL_HANDLE &&
        autoExposureHistogramBuffer != VK_NULL_HANDLE &&
        autoExposureStateBuffer != VK_NULL_HANDLE) {
        VkDescriptorBufferInfo autoExposureHistogramBufferInfo{};
        autoExposureHistogramBufferInfo.buffer = autoExposureHistogramBuffer;
        autoExposureHistogramBufferInfo.offset = 0;
        autoExposureHistogramBufferInfo.range = static_cast<VkDeviceSize>(kAutoExposureHistogramBins * sizeof(uint32_t));

        VkDescriptorBufferInfo autoExposureStateComputeBufferInfo{};
        autoExposureStateComputeBufferInfo.buffer = autoExposureStateBuffer;
        autoExposureStateComputeBufferInfo.offset = 0;
        autoExposureStateComputeBufferInfo.range = sizeof(float) * 4u;

        std::array<VkWriteDescriptorSet, 3> autoExposureWrites{};
        autoExposureWrites[0] = write;
        autoExposureWrites[0].dstSet = m_autoExposureDescriptorSets[m_currentFrame];
        autoExposureWrites[0].dstBinding = 0;
        autoExposureWrites[0].descriptorCount = 1;
        autoExposureWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        autoExposureWrites[0].pImageInfo = &hdrSceneImageInfo;

        autoExposureWrites[1] = write;
        autoExposureWrites[1].dstSet = m_autoExposureDescriptorSets[m_currentFrame];
        autoExposureWrites[1].dstBinding = 1;
        autoExposureWrites[1].descriptorCount = 1;
        autoExposureWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        autoExposureWrites[1].pBufferInfo = &autoExposureHistogramBufferInfo;

        autoExposureWrites[2] = write;
        autoExposureWrites[2].dstSet = m_autoExposureDescriptorSets[m_currentFrame];
        autoExposureWrites[2].dstBinding = 2;
        autoExposureWrites[2].descriptorCount = 1;
        autoExposureWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        autoExposureWrites[2].pBufferInfo = &autoExposureStateComputeBufferInfo;

        vkUpdateDescriptorSets(
            m_device,
            static_cast<uint32_t>(autoExposureWrites.size()),
            autoExposureWrites.data(),
            0,
            nullptr
        );
    }

    if (m_sunShaftComputeAvailable &&
        m_sunShaftDescriptorSets[m_currentFrame] != VK_NULL_HANDLE &&
        aoFrameIndex < m_sunShaftImageViews.size() &&
        m_sunShaftImageViews[aoFrameIndex] != VK_NULL_HANDLE) {
        VkDescriptorImageInfo sunShaftOutputImageInfo{};
        sunShaftOutputImageInfo.sampler = VK_NULL_HANDLE;
        sunShaftOutputImageInfo.imageView = m_sunShaftImageViews[aoFrameIndex];
        sunShaftOutputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 4> sunShaftWrites{};
        sunShaftWrites[0] = write;
        sunShaftWrites[0].dstSet = m_sunShaftDescriptorSets[m_currentFrame];
        sunShaftWrites[0].dstBinding = 0;
        sunShaftWrites[0].descriptorCount = 1;
        sunShaftWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        sunShaftWrites[0].pBufferInfo = &cameraBufferInfo;

        sunShaftWrites[1] = write;
        sunShaftWrites[1].dstSet = m_sunShaftDescriptorSets[m_currentFrame];
        sunShaftWrites[1].dstBinding = 1;
        sunShaftWrites[1].descriptorCount = 1;
        sunShaftWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sunShaftWrites[1].pImageInfo = &normalDepthImageInfo;

        sunShaftWrites[2] = write;
        sunShaftWrites[2].dstSet = m_sunShaftDescriptorSets[m_currentFrame];
        sunShaftWrites[2].dstBinding = 2;
        sunShaftWrites[2].descriptorCount = 1;
        sunShaftWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sunShaftWrites[2].pImageInfo = &shadowMapImageInfo;

        sunShaftWrites[3] = write;
        sunShaftWrites[3].dstSet = m_sunShaftDescriptorSets[m_currentFrame];
        sunShaftWrites[3].dstBinding = 3;
        sunShaftWrites[3].descriptorCount = 1;
        sunShaftWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        sunShaftWrites[3].pImageInfo = &sunShaftOutputImageInfo;

        vkUpdateDescriptorSets(
            m_device,
            static_cast<uint32_t>(sunShaftWrites.size()),
            sunShaftWrites.data(),
            0,
            nullptr
        );
    }

    if (m_bindlessDescriptorSet != VK_NULL_HANDLE && m_bindlessTextureCapacity >= kBindlessTextureStaticCount) {
        std::array<VkDescriptorImageInfo, kBindlessTextureStaticCount> bindlessImageInfos{};
        bindlessImageInfos[kBindlessTextureIndexDiffuse] = diffuseTextureImageInfo;
        bindlessImageInfos[kBindlessTextureIndexHdrResolved] = hdrSceneImageInfo;
        bindlessImageInfos[kBindlessTextureIndexShadowAtlas] = shadowMapImageInfo;
        bindlessImageInfos[kBindlessTextureIndexNormalDepth] = normalDepthImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSsaoBlur] = ssaoBlurImageInfo;
        bindlessImageInfos[kBindlessTextureIndexSsaoRaw] = ssaoRawImageInfo;
        bindlessImageInfos[kBindlessTextureIndexPlantDiffuse] = plantDiffuseTextureImageInfo;

        VkWriteDescriptorSet bindlessWrite{};
        bindlessWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        bindlessWrite.dstSet = m_bindlessDescriptorSet;
        bindlessWrite.dstBinding = 0;
        bindlessWrite.dstArrayElement = 0;
        bindlessWrite.descriptorCount = kBindlessTextureStaticCount;
        bindlessWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindlessWrite.pImageInfo = bindlessImageInfos.data();
        vkUpdateDescriptorSets(m_device, 1, &bindlessWrite, 0, nullptr);
    }

    return m_descriptorManager.buildBoundDescriptorSets(m_currentFrame);
}

} // namespace voxelsprout::render
