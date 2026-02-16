#include "render/Renderer.hpp"

#include "core/Log.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace render {

namespace {

constexpr uint32_t kHdrResolveBloomMipCount = 6u;

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

uint32_t findMemoryTypeIndex(
    VkPhysicalDevice physicalDevice,
    uint32_t typeBits,
    VkMemoryPropertyFlags requiredProperties
) {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const bool typeMatches = (typeBits & (1u << i)) != 0;
        const bool propertiesMatch =
            (memoryProperties.memoryTypes[i].propertyFlags & requiredProperties) == requiredProperties;
        if (typeMatches && propertiesMatch) {
            return i;
        }
    }
    return std::numeric_limits<uint32_t>::max();
}

} // namespace

bool Renderer::createDepthTargets() {
    if (m_depthFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "depth format is undefined\n";
        return false;
    }

    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    m_depthImages.assign(imageCount, VK_NULL_HANDLE);
    m_depthImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_depthImageViews.assign(imageCount, VK_NULL_HANDLE);
    m_depthImageAllocations.assign(imageCount, VK_NULL_HANDLE);

    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = m_depthFormat;
        imageCreateInfo.extent.width = m_swapchainExtent.width;
        imageCreateInfo.extent.height = m_swapchainExtent.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = m_colorSampleCount;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkResult imageResult = VK_ERROR_INITIALIZATION_FAILED;
        if (m_vmaAllocator != VK_NULL_HANDLE) {
            VmaAllocationCreateInfo allocationCreateInfo{};
            allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            imageResult = vmaCreateImage(
                m_vmaAllocator,
                &imageCreateInfo,
                &allocationCreateInfo,
                &m_depthImages[i],
                &m_depthImageAllocations[i],
                nullptr
            );
            if (imageResult != VK_SUCCESS) {
                logVkFailure("vmaCreateImage(depth)", imageResult);
                return false;
            }
        } else
        {
            imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_depthImages[i]);
            if (imageResult != VK_SUCCESS) {
                logVkFailure("vkCreateImage(depth)", imageResult);
                return false;
            }

            VkMemoryRequirements memoryRequirements{};
            vkGetImageMemoryRequirements(m_device, m_depthImages[i], &memoryRequirements);

            const uint32_t memoryTypeIndex = findMemoryTypeIndex(
                m_physicalDevice,
                memoryRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );
            if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                VOX_LOGI("render") << "no memory type for depth image\n";
                return false;
            }

            VkMemoryAllocateInfo allocateInfo{};
            allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocateInfo.allocationSize = memoryRequirements.size;
            allocateInfo.memoryTypeIndex = memoryTypeIndex;

            const VkResult allocResult = vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_depthImageMemories[i]);
            if (allocResult != VK_SUCCESS) {
                logVkFailure("vkAllocateMemory(depth)", allocResult);
                return false;
            }

            const VkResult bindResult = vkBindImageMemory(m_device, m_depthImages[i], m_depthImageMemories[i], 0);
            if (bindResult != VK_SUCCESS) {
                logVkFailure("vkBindImageMemory(depth)", bindResult);
                return false;
            }
        }
        {
            const std::string imageName = "depth.msaa.image." + std::to_string(i);
            setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_depthImages[i]), imageName.c_str());
        }

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_depthImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_depthFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;

        const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_depthImageViews[i]);
        if (viewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(depth)", viewResult);
            return false;
        }
        {
            const std::string viewName = "depth.msaa.imageView." + std::to_string(i);
            setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_depthImageViews[i]), viewName.c_str());
        }
    }

    return true;
}

bool Renderer::createAoTargets() {
    if (m_normalDepthFormat == VK_FORMAT_UNDEFINED || m_ssaoFormat == VK_FORMAT_UNDEFINED || m_hdrColorFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "AO formats are undefined\n";
        return false;
    }
    if (m_depthFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "depth format is undefined for AO targets\n";
        return false;
    }

    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    const uint32_t frameTargetCount = kMaxFramesInFlight;
    m_aoExtent.width = std::max(1u, m_swapchainExtent.width / 2u);
    m_aoExtent.height = std::max(1u, m_swapchainExtent.height / 2u);

    auto createColorTargets = [&](VkFormat format,
                                  std::vector<VkImage>& outImages,
                                  std::vector<VkDeviceMemory>& outMemories,
                                  std::vector<VkImageView>& outViews,
                                  std::vector<TransientImageHandle>& outHandles,
                                  const char* debugLabel,
                                  FrameArenaPass firstPass,
                                  FrameArenaPass lastPass) -> bool {
        outImages.assign(frameTargetCount, VK_NULL_HANDLE);
        outMemories.assign(frameTargetCount, VK_NULL_HANDLE);
        outViews.assign(frameTargetCount, VK_NULL_HANDLE);
        outHandles.assign(frameTargetCount, kInvalidTransientImageHandle);
        for (uint32_t i = 0; i < frameTargetCount; ++i) {
            TransientImageDesc imageDesc{};
            imageDesc.imageType = VK_IMAGE_TYPE_2D;
            imageDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageDesc.format = format;
            imageDesc.extent = {m_aoExtent.width, m_aoExtent.height, 1u};
            imageDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageDesc.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            imageDesc.mipLevels = 1;
            imageDesc.arrayLayers = 1;
            imageDesc.samples = VK_SAMPLE_COUNT_1_BIT;
            imageDesc.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageDesc.firstPass = firstPass;
            imageDesc.lastPass = lastPass;
            imageDesc.debugName = std::string(debugLabel) + "[" + std::to_string(i) + "]";
            const TransientImageHandle handle = m_frameArena.createTransientImage(
                imageDesc,
                FrameArenaImageLifetime::Persistent
            );
            if (handle == kInvalidTransientImageHandle) {
                VOX_LOGE("render") << "failed creating transient image " << debugLabel << "\n";
                return false;
            }
            const TransientImageInfo* imageInfo = m_frameArena.getTransientImage(handle);
            if (imageInfo == nullptr || imageInfo->image == VK_NULL_HANDLE || imageInfo->view == VK_NULL_HANDLE) {
                VOX_LOGE("render") << "invalid transient image " << debugLabel << "\n";
                return false;
            }
            outHandles[i] = handle;
            outImages[i] = imageInfo->image;
            outViews[i] = imageInfo->view;
            outMemories[i] = VK_NULL_HANDLE;
            setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(outImages[i]), imageDesc.debugName.c_str());
            {
                const std::string viewName = std::string(debugLabel) + ".view[" + std::to_string(i) + "]";
                setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(outViews[i]), viewName.c_str());
            }
        }
        return true;
    };

    m_normalDepthImageInitialized.assign(frameTargetCount, false);
    m_aoDepthImageInitialized.assign(imageCount, false);
    m_ssaoRawImageInitialized.assign(frameTargetCount, false);
    m_ssaoBlurImageInitialized.assign(frameTargetCount, false);
    m_sunShaftImageInitialized.assign(frameTargetCount, false);

    if (!createColorTargets(
            m_normalDepthFormat,
            m_normalDepthImages,
            m_normalDepthImageMemories,
            m_normalDepthImageViews,
            m_normalDepthTransientHandles,
            "ao.normalDepth",
            FrameArenaPass::Ssao,
            FrameArenaPass::Ssao
        )) {
        return false;
    }

    m_aoDepthImages.assign(imageCount, VK_NULL_HANDLE);
    m_aoDepthImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_aoDepthImageViews.assign(imageCount, VK_NULL_HANDLE);
    m_aoDepthTransientHandles.assign(imageCount, kInvalidTransientImageHandle);
    for (uint32_t i = 0; i < imageCount; ++i) {
        TransientImageDesc depthDesc{};
        depthDesc.imageType = VK_IMAGE_TYPE_2D;
        depthDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        depthDesc.format = m_depthFormat;
        depthDesc.extent = {m_aoExtent.width, m_aoExtent.height, 1u};
        depthDesc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        depthDesc.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        depthDesc.mipLevels = 1;
        depthDesc.arrayLayers = 1;
        depthDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        depthDesc.tiling = VK_IMAGE_TILING_OPTIMAL;
        depthDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthDesc.firstPass = FrameArenaPass::Ssao;
        depthDesc.lastPass = FrameArenaPass::Ssao;
        depthDesc.debugName = "ao.depth[" + std::to_string(i) + "]";
        const TransientImageHandle depthHandle = m_frameArena.createTransientImage(
            depthDesc,
            FrameArenaImageLifetime::Persistent
        );
        if (depthHandle == kInvalidTransientImageHandle) {
            VOX_LOGE("render") << "failed creating AO depth transient image\n";
            return false;
        }
        const TransientImageInfo* depthInfo = m_frameArena.getTransientImage(depthHandle);
        if (depthInfo == nullptr || depthInfo->image == VK_NULL_HANDLE || depthInfo->view == VK_NULL_HANDLE) {
            VOX_LOGE("render") << "invalid AO depth transient image info\n";
            return false;
        }
        m_aoDepthTransientHandles[i] = depthHandle;
        m_aoDepthImages[i] = depthInfo->image;
        m_aoDepthImageViews[i] = depthInfo->view;
        m_aoDepthImageMemories[i] = VK_NULL_HANDLE;
        setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_aoDepthImages[i]), depthDesc.debugName.c_str());
        {
            const std::string viewName = "ao.depth.view[" + std::to_string(i) + "]";
            setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_aoDepthImageViews[i]), viewName.c_str());
        }
    }

    if (!createColorTargets(
            m_ssaoFormat,
            m_ssaoRawImages,
            m_ssaoRawImageMemories,
            m_ssaoRawImageViews,
            m_ssaoRawTransientHandles,
            "ao.ssaoRaw",
            FrameArenaPass::Ssao,
            FrameArenaPass::Ssao
        )) {
        return false;
    }
    if (!createColorTargets(
            m_ssaoFormat,
            m_ssaoBlurImages,
            m_ssaoBlurImageMemories,
            m_ssaoBlurImageViews,
            m_ssaoBlurTransientHandles,
            "ao.ssaoBlur",
            FrameArenaPass::Ssao,
            FrameArenaPass::Main
        )) {
        return false;
    }

    m_sunShaftImages.assign(frameTargetCount, VK_NULL_HANDLE);
    m_sunShaftImageMemories.assign(frameTargetCount, VK_NULL_HANDLE);
    m_sunShaftImageViews.assign(frameTargetCount, VK_NULL_HANDLE);
    m_sunShaftTransientHandles.assign(frameTargetCount, kInvalidTransientImageHandle);
    for (uint32_t i = 0; i < frameTargetCount; ++i) {
        TransientImageDesc sunShaftDesc{};
        sunShaftDesc.imageType = VK_IMAGE_TYPE_2D;
        sunShaftDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        sunShaftDesc.format = m_hdrColorFormat;
        sunShaftDesc.extent = {m_aoExtent.width, m_aoExtent.height, 1u};
        sunShaftDesc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        sunShaftDesc.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        sunShaftDesc.mipLevels = 1;
        sunShaftDesc.arrayLayers = 1;
        sunShaftDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        sunShaftDesc.tiling = VK_IMAGE_TILING_OPTIMAL;
        sunShaftDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        sunShaftDesc.firstPass = FrameArenaPass::Post;
        sunShaftDesc.lastPass = FrameArenaPass::Post;
        sunShaftDesc.debugName = "post.sunShaft[" + std::to_string(i) + "]";
        const TransientImageHandle sunShaftHandle = m_frameArena.createTransientImage(
            sunShaftDesc,
            FrameArenaImageLifetime::Persistent
        );
        if (sunShaftHandle == kInvalidTransientImageHandle) {
            VOX_LOGE("render") << "failed creating sun shaft transient image\n";
            return false;
        }
        const TransientImageInfo* sunShaftInfo = m_frameArena.getTransientImage(sunShaftHandle);
        if (sunShaftInfo == nullptr || sunShaftInfo->image == VK_NULL_HANDLE || sunShaftInfo->view == VK_NULL_HANDLE) {
            VOX_LOGE("render") << "invalid sun shaft transient image info\n";
            return false;
        }
        m_sunShaftTransientHandles[i] = sunShaftHandle;
        m_sunShaftImages[i] = sunShaftInfo->image;
        m_sunShaftImageViews[i] = sunShaftInfo->view;
        m_sunShaftImageMemories[i] = VK_NULL_HANDLE;
        setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_sunShaftImages[i]), sunShaftDesc.debugName.c_str());
        {
            const std::string viewName = "post.sunShaft.view[" + std::to_string(i) + "]";
            setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_sunShaftImageViews[i]), viewName.c_str());
        }
    }

    if (m_normalDepthSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samplerCreateInfo{};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
        samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 0.0f;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.compareEnable = VK_FALSE;
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
        const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_normalDepthSampler);
        if (samplerResult != VK_SUCCESS) {
            logVkFailure("vkCreateSampler(normalDepth)", samplerResult);
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_SAMPLER,
            vkHandleToUint64(m_normalDepthSampler),
            "normalDepth.sampler"
        );
    }

    if (m_ssaoSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samplerCreateInfo{};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 0.0f;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.compareEnable = VK_FALSE;
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
        const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_ssaoSampler);
        if (samplerResult != VK_SUCCESS) {
            logVkFailure("vkCreateSampler(ssao)", samplerResult);
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_SAMPLER,
            vkHandleToUint64(m_ssaoSampler),
            "ssao.sampler"
        );
    }

    if (m_sunShaftSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samplerCreateInfo{};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 0.0f;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.compareEnable = VK_FALSE;
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
        const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_sunShaftSampler);
        if (samplerResult != VK_SUCCESS) {
            logVkFailure("vkCreateSampler(sunShaft)", samplerResult);
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_SAMPLER,
            vkHandleToUint64(m_sunShaftSampler),
            "sunShaft.sampler"
        );
    }

    return true;
}

bool Renderer::createHdrResolveTargets() {
    if (m_hdrColorFormat == VK_FORMAT_UNDEFINED) {
        VOX_LOGE("render") << "HDR color format is undefined\n";
        return false;
    }

    VkFormatProperties hdrFormatProperties{};
    vkGetPhysicalDeviceFormatProperties(m_physicalDevice, m_hdrColorFormat, &hdrFormatProperties);
    const VkFormatFeatureFlags bloomMipFeatures =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT |
        VK_FORMAT_FEATURE_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    const bool supportsBloomMipBlit =
        (hdrFormatProperties.optimalTilingFeatures & bloomMipFeatures) == bloomMipFeatures;

    const uint32_t maxDimension = std::max(m_swapchainExtent.width, m_swapchainExtent.height);
    uint32_t preferredMipLevels = 1u;
    for (uint32_t mipDimension = maxDimension;
         mipDimension > 1u && preferredMipLevels < kHdrResolveBloomMipCount;
         mipDimension >>= 1u) {
        ++preferredMipLevels;
    }
    m_hdrResolveMipLevels = supportsBloomMipBlit ? std::max(1u, preferredMipLevels) : 1u;
    if (!supportsBloomMipBlit) {
        VOX_LOGW("render") << "HDR format lacks linear blit mip support; bloom mip chain disabled";
    }

    const uint32_t frameTargetCount = kMaxFramesInFlight;
    m_hdrResolveImages.assign(frameTargetCount, VK_NULL_HANDLE);
    m_hdrResolveImageMemories.assign(frameTargetCount, VK_NULL_HANDLE);
    m_hdrResolveImageViews.assign(frameTargetCount, VK_NULL_HANDLE);
    m_hdrResolveSampleImageViews.assign(frameTargetCount, VK_NULL_HANDLE);
    m_hdrResolveTransientHandles.assign(frameTargetCount, kInvalidTransientImageHandle);
    m_hdrResolveImageInitialized.assign(frameTargetCount, false);

    for (uint32_t i = 0; i < frameTargetCount; ++i) {
        TransientImageDesc hdrResolveDesc{};
        hdrResolveDesc.imageType = VK_IMAGE_TYPE_2D;
        hdrResolveDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        hdrResolveDesc.format = m_hdrColorFormat;
        hdrResolveDesc.extent = {m_swapchainExtent.width, m_swapchainExtent.height, 1u};
        hdrResolveDesc.usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        hdrResolveDesc.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        hdrResolveDesc.mipLevels = m_hdrResolveMipLevels;
        hdrResolveDesc.arrayLayers = 1;
        hdrResolveDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        hdrResolveDesc.tiling = VK_IMAGE_TILING_OPTIMAL;
        hdrResolveDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        hdrResolveDesc.firstPass = FrameArenaPass::Main;
        hdrResolveDesc.lastPass = FrameArenaPass::Post;
        hdrResolveDesc.debugName = "hdr.resolve[" + std::to_string(i) + "]";
        const TransientImageHandle hdrResolveHandle = m_frameArena.createTransientImage(
            hdrResolveDesc,
            FrameArenaImageLifetime::Persistent
        );
        if (hdrResolveHandle == kInvalidTransientImageHandle) {
            VOX_LOGE("render") << "failed creating HDR resolve transient image\n";
            return false;
        }
        const TransientImageInfo* hdrResolveInfo = m_frameArena.getTransientImage(hdrResolveHandle);
        if (hdrResolveInfo == nullptr || hdrResolveInfo->image == VK_NULL_HANDLE || hdrResolveInfo->view == VK_NULL_HANDLE) {
            VOX_LOGE("render") << "invalid HDR resolve transient image info\n";
            return false;
        }
        m_hdrResolveTransientHandles[i] = hdrResolveHandle;
        m_hdrResolveImages[i] = hdrResolveInfo->image;
        m_hdrResolveSampleImageViews[i] = hdrResolveInfo->view;
        m_hdrResolveImageMemories[i] = VK_NULL_HANDLE;
        setObjectName(
            VK_OBJECT_TYPE_IMAGE,
            vkHandleToUint64(m_hdrResolveImages[i]),
            hdrResolveDesc.debugName.c_str()
        );

        VkImageViewCreateInfo baseMipViewCreateInfo{};
        baseMipViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        baseMipViewCreateInfo.image = m_hdrResolveImages[i];
        baseMipViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        baseMipViewCreateInfo.format = m_hdrColorFormat;
        baseMipViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        baseMipViewCreateInfo.subresourceRange.baseMipLevel = 0;
        baseMipViewCreateInfo.subresourceRange.levelCount = 1;
        baseMipViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        baseMipViewCreateInfo.subresourceRange.layerCount = 1;
        const VkResult baseMipViewResult = vkCreateImageView(
            m_device,
            &baseMipViewCreateInfo,
            nullptr,
            &m_hdrResolveImageViews[i]
        );
        if (baseMipViewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(hdrResolveBaseMip)", baseMipViewResult);
            return false;
        }
        {
            const std::string resolveViewName = "hdr.resolve.baseMip.view[" + std::to_string(i) + "]";
            const std::string sampleViewName = "hdr.resolve.sample.view[" + std::to_string(i) + "]";
            setObjectName(
                VK_OBJECT_TYPE_IMAGE_VIEW,
                vkHandleToUint64(m_hdrResolveImageViews[i]),
                resolveViewName.c_str()
            );
            setObjectName(
                VK_OBJECT_TYPE_IMAGE_VIEW,
                vkHandleToUint64(m_hdrResolveSampleImageViews[i]),
                sampleViewName.c_str()
            );
        }
    }

    if (m_hdrResolveSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_hdrResolveSampler, nullptr);
        m_hdrResolveSampler = VK_NULL_HANDLE;
    }
    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;
    samplerCreateInfo.compareEnable = VK_FALSE;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = static_cast<float>(std::max(1u, m_hdrResolveMipLevels) - 1u);
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;

    const VkResult samplerResult = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_hdrResolveSampler);
    if (samplerResult != VK_SUCCESS) {
        logVkFailure("vkCreateSampler(hdrResolve)", samplerResult);
        return false;
    }
    setObjectName(
        VK_OBJECT_TYPE_SAMPLER,
        vkHandleToUint64(m_hdrResolveSampler),
        "hdrResolve.sampler"
    );

    return true;
}

bool Renderer::createMsaaColorTargets() {
    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    m_msaaColorImages.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageViews.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageInitialized.assign(imageCount, false);
    m_msaaColorImageAllocations.assign(imageCount, VK_NULL_HANDLE);

    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = m_hdrColorFormat;
        imageCreateInfo.extent.width = m_swapchainExtent.width;
        imageCreateInfo.extent.height = m_swapchainExtent.height;
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = m_colorSampleCount;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkResult imageResult = VK_ERROR_INITIALIZATION_FAILED;
        if (m_vmaAllocator != VK_NULL_HANDLE) {
            VmaAllocationCreateInfo allocationCreateInfo{};
            allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            imageResult = vmaCreateImage(
                m_vmaAllocator,
                &imageCreateInfo,
                &allocationCreateInfo,
                &m_msaaColorImages[i],
                &m_msaaColorImageAllocations[i],
                nullptr
            );
            if (imageResult != VK_SUCCESS) {
                logVkFailure("vmaCreateImage(msaaColor)", imageResult);
                return false;
            }
        } else
        {
            imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_msaaColorImages[i]);
            if (imageResult != VK_SUCCESS) {
                logVkFailure("vkCreateImage(msaaColor)", imageResult);
                return false;
            }

            VkMemoryRequirements memoryRequirements{};
            vkGetImageMemoryRequirements(m_device, m_msaaColorImages[i], &memoryRequirements);

            const uint32_t memoryTypeIndex = findMemoryTypeIndex(
                m_physicalDevice,
                memoryRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );
            if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                VOX_LOGI("render") << "no memory type for MSAA color image\n";
                return false;
            }

            VkMemoryAllocateInfo allocateInfo{};
            allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocateInfo.allocationSize = memoryRequirements.size;
            allocateInfo.memoryTypeIndex = memoryTypeIndex;

            const VkResult allocResult = vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_msaaColorImageMemories[i]);
            if (allocResult != VK_SUCCESS) {
                logVkFailure("vkAllocateMemory(msaaColor)", allocResult);
                return false;
            }

            const VkResult bindResult = vkBindImageMemory(m_device, m_msaaColorImages[i], m_msaaColorImageMemories[i], 0);
            if (bindResult != VK_SUCCESS) {
                logVkFailure("vkBindImageMemory(msaaColor)", bindResult);
                return false;
            }
        }
        {
            const std::string imageName = "hdr.msaaColor.image." + std::to_string(i);
            setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_msaaColorImages[i]), imageName.c_str());
        }

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_msaaColorImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_hdrColorFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;

        const VkResult viewResult = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_msaaColorImageViews[i]);
        if (viewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(msaaColor)", viewResult);
            return false;
        }
        {
            const std::string viewName = "hdr.msaaColor.imageView." + std::to_string(i);
            setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_msaaColorImageViews[i]), viewName.c_str());
        }
    }

    return true;
}

void Renderer::destroyHdrResolveTargets() {
    if (m_hdrResolveSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_hdrResolveSampler, nullptr);
        m_hdrResolveSampler = VK_NULL_HANDLE;
    }

    for (VkImageView imageView : m_hdrResolveImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }

    for (TransientImageHandle handle : m_hdrResolveTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_hdrResolveImageViews.clear();
    m_hdrResolveSampleImageViews.clear();
    m_hdrResolveImages.clear();
    m_hdrResolveImageMemories.clear();
    m_hdrResolveTransientHandles.clear();
    m_hdrResolveImageInitialized.clear();
    m_hdrResolveMipLevels = 1;
}

void Renderer::destroyMsaaColorTargets() {
    for (VkImageView imageView : m_msaaColorImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_msaaColorImageViews.clear();

    for (size_t i = 0; i < m_msaaColorImages.size(); ++i) {
        const VkImage image = m_msaaColorImages[i];
        if (image == VK_NULL_HANDLE) {
            continue;
        }
        if (m_vmaAllocator != VK_NULL_HANDLE &&
            i < m_msaaColorImageAllocations.size() &&
            m_msaaColorImageAllocations[i] != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, image, m_msaaColorImageAllocations[i]);
            m_msaaColorImageAllocations[i] = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, image, nullptr);
        }
    }
    m_msaaColorImages.clear();

    for (VkDeviceMemory memory : m_msaaColorImageMemories) {
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, memory, nullptr);
        }
    }
    m_msaaColorImageMemories.clear();
    m_msaaColorImageAllocations.clear();
    m_msaaColorImageInitialized.clear();
}

void Renderer::destroyDepthTargets() {
    for (VkImageView imageView : m_depthImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_depthImageViews.clear();

    for (size_t i = 0; i < m_depthImages.size(); ++i) {
        const VkImage image = m_depthImages[i];
        if (image == VK_NULL_HANDLE) {
            continue;
        }
        if (m_vmaAllocator != VK_NULL_HANDLE &&
            i < m_depthImageAllocations.size() &&
            m_depthImageAllocations[i] != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, image, m_depthImageAllocations[i]);
            m_depthImageAllocations[i] = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, image, nullptr);
        }
    }
    m_depthImages.clear();

    for (VkDeviceMemory memory : m_depthImageMemories) {
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, memory, nullptr);
        }
    }
    m_depthImageMemories.clear();
    m_depthImageAllocations.clear();
}

void Renderer::destroyAoTargets() {
    if (m_sunShaftSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_sunShaftSampler, nullptr);
        m_sunShaftSampler = VK_NULL_HANDLE;
    }
    if (m_ssaoSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_ssaoSampler, nullptr);
        m_ssaoSampler = VK_NULL_HANDLE;
    }
    if (m_normalDepthSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_normalDepthSampler, nullptr);
        m_normalDepthSampler = VK_NULL_HANDLE;
    }

    for (TransientImageHandle handle : m_ssaoBlurTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_ssaoBlurImageViews.clear();
    m_ssaoBlurImages.clear();
    m_ssaoBlurImageMemories.clear();
    m_ssaoBlurTransientHandles.clear();
    m_ssaoBlurImageInitialized.clear();

    for (TransientImageHandle handle : m_sunShaftTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_sunShaftImageViews.clear();
    m_sunShaftImages.clear();
    m_sunShaftImageMemories.clear();
    m_sunShaftTransientHandles.clear();
    m_sunShaftImageInitialized.clear();

    for (TransientImageHandle handle : m_ssaoRawTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_ssaoRawImageViews.clear();
    m_ssaoRawImages.clear();
    m_ssaoRawImageMemories.clear();
    m_ssaoRawTransientHandles.clear();
    m_ssaoRawImageInitialized.clear();

    for (TransientImageHandle handle : m_aoDepthTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_aoDepthImageViews.clear();
    m_aoDepthImages.clear();
    m_aoDepthImageMemories.clear();
    m_aoDepthTransientHandles.clear();
    m_aoDepthImageInitialized.clear();

    for (TransientImageHandle handle : m_normalDepthTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_normalDepthImageViews.clear();
    m_normalDepthImages.clear();
    m_normalDepthImageMemories.clear();
    m_normalDepthTransientHandles.clear();
    m_normalDepthImageInitialized.clear();
}

} // namespace render
