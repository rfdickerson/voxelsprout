#include "render/backend/vulkan/renderer_backend.h"

#include "core/log.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace odai::render {

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

bool RendererBackend::createDepthTargets() {
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
        imageCreateInfo.extent.width = m_renderExtent.width;
        imageCreateInfo.extent.height = m_renderExtent.height;
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

bool RendererBackend::createAoTargets() {
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
    m_aoExtent.width = std::max(1u, m_renderExtent.width / 2u);
    m_aoExtent.height = std::max(1u, m_renderExtent.height / 2u);
    // The AO ESTIMATOR runs at its own, lower resolution than the AO chain's
    // other targets. The horizon march is by far the most expensive thing in
    // the frame after the main pass -- measured 6.4 ms of a 37.9 ms frame at a
    // 2560x1440 swapchain, already at m_aoExtent (half the render extent) --
    // and its cost is per output texel, so this is the one knob that moves it
    // proportionally. The blur pass reads the result at whatever resolution it
    // lands at and joint-bilateral upsamples back to m_aoExtent, which is why
    // only this one target shrinks and every consumer is unaffected.
    m_ssaoRawExtent.width = std::max(1u, m_aoExtent.width / m_aoDownscale);
    m_ssaoRawExtent.height = std::max(1u, m_aoExtent.height / m_aoDownscale);

    m_normalDepthExtent = useMergedDepthPrepass() ? m_renderExtent : m_aoExtent;

    // The cluster grid is a function of the render extent, so it is rebuilt
    // here rather than at init: a resize that left it stale would have every
    // fragment reading a cluster computed for the old tiling.
    if (!createLightClusterBuffer(m_renderExtent)) {
        return false;
    }
    if (!createContactShadowBuffers(m_renderExtent)) {
        VOX_LOGW("render") << "contact-shadow buffers unavailable; hybrid mode will use maps only";
        m_contactShadowAvailable = false;
    }
    if (m_screenSpaceGiAvailable && !createScreenSpaceGiBuffers(m_renderExtent)) {
        VOX_LOGW("render") << "screen-space GI buffers unavailable; using authored ambient only";
        m_screenSpaceGiAvailable = false;
    }

    auto createColorTargets = [&](VkFormat format,
                                  std::vector<VkImage>& outImages,
                                  std::vector<VkDeviceMemory>& outMemories,
                                  std::vector<VkImageView>& outViews,
                                  std::vector<TransientImageHandle>& outHandles,
                                  const char* debugLabel,
                                  FrameArenaPass firstPass,
                                  FrameArenaPass lastPass,
                                  VkExtent2D extent) -> bool {
        outImages.assign(frameTargetCount, VK_NULL_HANDLE);
        outMemories.assign(frameTargetCount, VK_NULL_HANDLE);
        outViews.assign(frameTargetCount, VK_NULL_HANDLE);
        outHandles.assign(frameTargetCount, kInvalidTransientImageHandle);
        for (uint32_t i = 0; i < frameTargetCount; ++i) {
            TransientImageDesc imageDesc{};
            imageDesc.imageType = VK_IMAGE_TYPE_2D;
            imageDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageDesc.format = format;
            imageDesc.extent = {extent.width, extent.height, 1u};
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
    m_xegtaoDepthInitialized.assign(frameTargetCount, false);
    m_xegtaoAoTermInitialized.assign(frameTargetCount, false);
    m_xegtaoBentNormalInitialized.assign(frameTargetCount, false);
    m_ssaoBlurImageInitialized.assign(frameTargetCount, false);
    m_sunShaftImageInitialized.assign(frameTargetCount, false);
    m_sunShaftImageHasContent.assign(frameTargetCount, false);

    if (!createColorTargets(
            m_normalDepthFormat,
            m_normalDepthImages,
            m_normalDepthImageMemories,
            m_normalDepthImageViews,
            m_normalDepthTransientHandles,
            "ao.normalDepth",
            FrameArenaPass::Ssao,
            FrameArenaPass::Ssao,
            m_normalDepthExtent
        )) {
        return false;
    }

    // Motion vectors for geometry that moved independently of the camera.
    //
    // RGBA16F, at RENDER extent rather than the AO extent: a consumer looks up
    // history with this, and a half-resolution motion vector reprojects a
    // silhouette to the wrong pixel. .xy is the NDC vector, .z is a validity
    // flag (see skinned_velocity.frag), .w unused. Cleared to zero every frame,
    // so "nothing drew here" reads as z = 0 and falls back to depth reprojection.
    if (!createColorTargets(
            m_velocityFormat,
            m_velocityImages,
            m_velocityImageMemories,
            m_velocityImageViews,
            m_velocityTransientHandles,
            "velocity",
            FrameArenaPass::Main,
            FrameArenaPass::Post,
            m_renderExtent
        )) {
        return false;
    }
    m_velocityImageInitialized.assign(frameTargetCount, false);

    m_aoDepthImages.assign(imageCount, VK_NULL_HANDLE);
    m_aoDepthImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_aoDepthImageViews.assign(imageCount, VK_NULL_HANDLE);
    m_aoDepthTransientHandles.assign(imageCount, kInvalidTransientImageHandle);
    // Not created under the merged prepass: it depth-tests against the real
    // depth buffer instead, which is the whole point. Left as null handles so
    // any accidental use faults loudly rather than reading a stale image.
    const uint32_t aoDepthCount = useMergedDepthPrepass() ? 0u : imageCount;
    for (uint32_t i = 0; i < aoDepthCount; ++i) {
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

    // SSAO raw + blur targets are written by compute (storage image) and sampled by
    // later passes, so they skip the color-attachment usage createColorTargets assumes.
    auto createSsaoComputeTarget = [&](std::vector<VkImage>& outImages,
                                        std::vector<VkDeviceMemory>& outMemories,
                                        std::vector<VkImageView>& outViews,
                                        std::vector<TransientImageHandle>& outHandles,
                                        const char* debugLabel,
                                        FrameArenaPass firstPass,
                                        FrameArenaPass lastPass,
                                        VkExtent2D targetExtent,
                                        VkFormat targetFormat) -> bool {
        outImages.assign(frameTargetCount, VK_NULL_HANDLE);
        outMemories.assign(frameTargetCount, VK_NULL_HANDLE);
        outViews.assign(frameTargetCount, VK_NULL_HANDLE);
        outHandles.assign(frameTargetCount, kInvalidTransientImageHandle);
        for (uint32_t i = 0; i < frameTargetCount; ++i) {
            TransientImageDesc imageDesc{};
            imageDesc.imageType = VK_IMAGE_TYPE_2D;
            imageDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
            imageDesc.format = targetFormat;
            imageDesc.extent = {targetExtent.width, targetExtent.height, 1u};
            imageDesc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
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

    if (!createSsaoComputeTarget(
            m_ssaoRawImages,
            m_ssaoRawImageMemories,
            m_ssaoRawImageViews,
            m_ssaoRawTransientHandles,
            "ao.ssaoRaw",
            FrameArenaPass::Ssao,
            FrameArenaPass::Ssao,
            m_ssaoRawExtent,
            m_ssaoFormat
        )) {
        return false;
    }
    if (!createSsaoComputeTarget(
            m_ssaoBlurImages,
            m_ssaoBlurImageMemories,
            m_ssaoBlurImageViews,
            m_ssaoBlurTransientHandles,
            "ao.ssaoBlur",
            FrameArenaPass::Ssao,
            FrameArenaPass::Main,
            m_aoExtent,
            m_ssaoFormat
        )) {
        return false;
    }

    // XeGTAO depth pyramid: five levels, each half the previous, starting at the
    // resolution the GTAO pass runs at.
    //
    // R32_SFLOAT rather than the R16F the AO targets use. These hold VIEWSPACE
    // DEPTH in world units, and this game runs a 120000-unit far plane -- past
    // R16F's 65504 ceiling, where every distant texel would become +inf and every
    // horizon behind it would be garbage.
    for (uint32_t level = 0; level < kXeGtaoDepthMipCount; ++level) {
        VkExtent2D levelExtent{
            std::max(1u, m_ssaoRawExtent.width >> level),
            std::max(1u, m_ssaoRawExtent.height >> level)
        };
        m_xegtaoDepthExtents[level] = levelExtent;
        if (!createSsaoComputeTarget(
                m_xegtaoDepthImages[level],
                m_xegtaoDepthImageMemories[level],
                m_xegtaoDepthImageViews[level],
                m_xegtaoDepthTransientHandles[level],
                ("ao.xegtaoDepthMip" + std::to_string(level)).c_str(),
                FrameArenaPass::Ssao,
                FrameArenaPass::Ssao,
                levelExtent,
                VK_FORMAT_R32_SFLOAT
            )) {
            return false;
        }
    }

    if (!createSsaoComputeTarget(
            m_xegtaoAoTermImages,
            m_xegtaoAoTermImageMemories,
            m_xegtaoAoTermImageViews,
            m_xegtaoAoTermTransientHandles,
            "ao.xegtaoAoTerm",
            FrameArenaPass::Ssao,
            FrameArenaPass::Ssao,
            m_ssaoRawExtent,
            m_ssaoFormat
        )) {
        return false;
    }

    // Bent normals, view space, encoded to [0,1]. RGBA8 is enough: this is a
    // direction consumed by a low-frequency ambient term, and the denoiser it
    // feeds is already averaging across a neighbourhood.
    if (!createSsaoComputeTarget(
            m_xegtaoBentNormalImages,
            m_xegtaoBentNormalImageMemories,
            m_xegtaoBentNormalImageViews,
            m_xegtaoBentNormalTransientHandles,
            "ao.xegtaoBentNormal",
            FrameArenaPass::Ssao,
            FrameArenaPass::Main,
            m_ssaoRawExtent,
            VK_FORMAT_R8G8B8A8_UNORM
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

    // Point sampler for the XeGTAO depth pyramid. NEAREST is load-bearing, not a
    // default: linear filtering blends neighbouring depths on the same level,
    // which fabricates a surface between two real ones and feeds the horizon
    // search a occluder that is not there.
    if (m_xegtaoPointSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo pointSamplerInfo{};
        pointSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        pointSamplerInfo.magFilter = VK_FILTER_NEAREST;
        pointSamplerInfo.minFilter = VK_FILTER_NEAREST;
        pointSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        pointSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        pointSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        pointSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        pointSamplerInfo.maxLod = 0.0f;
        pointSamplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        const VkResult pointSamplerResult =
            vkCreateSampler(m_device, &pointSamplerInfo, nullptr, &m_xegtaoPointSampler);
        if (pointSamplerResult != VK_SUCCESS) {
            logVkFailure("vkCreateSampler(xegtaoPoint)", pointSamplerResult);
            return false;
        }
        setObjectName(
            VK_OBJECT_TYPE_SAMPLER,
            vkHandleToUint64(m_xegtaoPointSampler),
            "renderer.sampler.xegtaoPoint");
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

bool RendererBackend::createHdrResolveTargets() {
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

    const uint32_t maxDimension = std::max(m_renderExtent.width, m_renderExtent.height);
    uint32_t preferredMipLevels = 1u;
    for (uint32_t mipDimension = maxDimension;
         mipDimension > 1u && preferredMipLevels < kHdrResolveBloomMipCount;
         mipDimension >>= 1u) {
        ++preferredMipLevels;
    }
    // ODAI_BLOOM_MIPS caps the chain.
    //
    // Every mip past the first costs a full-image layout transition plus a blit,
    // and the cost is per-mip rather than per-pixel: the 2x1 tail of the pyramid
    // is as many barriers as the top and moves almost no data. On an iGPU that
    // sync traffic is a real share of the post pass. The mips that actually
    // carry visible glow are the first few; the rest widen a halo that is
    // already wider than the screen.
    static const uint32_t s_bloomMipCap = []() {
        const char* env = std::getenv("ODAI_BLOOM_MIPS");
        const int value = (env != nullptr) ? std::atoi(env) : 0;
        return (value > 0) ? static_cast<uint32_t>(value) : 0u;
    }();
    if (s_bloomMipCap > 0u) {
        preferredMipLevels = std::min(preferredMipLevels, s_bloomMipCap);
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
    m_waterRefractionImages.assign(frameTargetCount, VK_NULL_HANDLE);
    m_waterRefractionImageViews.assign(frameTargetCount, VK_NULL_HANDLE);
    m_waterRefractionTransientHandles.assign(frameTargetCount, kInvalidTransientImageHandle);
    m_waterRefractionImageInitialized.assign(frameTargetCount, false);
    m_waterReflectionExtent = {
        std::max(1u, (m_renderExtent.width + 1u) / 2u),
        std::max(1u, (m_renderExtent.height + 1u) / 2u),
    };
    m_waterReflectionImages.assign(frameTargetCount, VK_NULL_HANDLE);
    m_waterReflectionImageViews.assign(frameTargetCount, VK_NULL_HANDLE);
    m_waterReflectionTransientHandles.assign(frameTargetCount, kInvalidTransientImageHandle);
    m_waterReflectionImageInitialized.assign(frameTargetCount, false);
    m_waterReflectionDepthImages.assign(frameTargetCount, VK_NULL_HANDLE);
    m_waterReflectionDepthImageViews.assign(frameTargetCount, VK_NULL_HANDLE);
    m_waterReflectionDepthTransientHandles.assign(frameTargetCount, kInvalidTransientImageHandle);
    m_waterReflectionDepthImageInitialized.assign(frameTargetCount, false);
    m_waterReflectionDepthSampled.assign(frameTargetCount, false);

    for (uint32_t i = 0; i < frameTargetCount; ++i) {
        TransientImageDesc hdrResolveDesc{};
        hdrResolveDesc.imageType = VK_IMAGE_TYPE_2D;
        hdrResolveDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        hdrResolveDesc.format = m_hdrColorFormat;
        hdrResolveDesc.extent = {m_renderExtent.width, m_renderExtent.height, 1u};
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

        TransientImageDesc waterRefractionDesc{};
        waterRefractionDesc.imageType = VK_IMAGE_TYPE_2D;
        waterRefractionDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        waterRefractionDesc.format = m_hdrColorFormat;
        waterRefractionDesc.extent = {m_renderExtent.width, m_renderExtent.height, 1u};
        waterRefractionDesc.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        waterRefractionDesc.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        waterRefractionDesc.mipLevels = 1;
        waterRefractionDesc.arrayLayers = 1;
        waterRefractionDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        waterRefractionDesc.tiling = VK_IMAGE_TILING_OPTIMAL;
        waterRefractionDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        waterRefractionDesc.firstPass = FrameArenaPass::Main;
        waterRefractionDesc.lastPass = FrameArenaPass::Main;
        waterRefractionDesc.debugName = "water.refraction.opaque[" + std::to_string(i) + "]";
        const TransientImageHandle waterRefractionHandle = m_frameArena.createTransientImage(
            waterRefractionDesc,
            FrameArenaImageLifetime::Persistent
        );
        if (waterRefractionHandle == kInvalidTransientImageHandle) {
            VOX_LOGE("render") << "failed creating water refraction transient image\n";
            return false;
        }
        const TransientImageInfo* waterRefractionInfo = m_frameArena.getTransientImage(waterRefractionHandle);
        if (waterRefractionInfo == nullptr ||
            waterRefractionInfo->image == VK_NULL_HANDLE ||
            waterRefractionInfo->view == VK_NULL_HANDLE) {
            VOX_LOGE("render") << "invalid water refraction transient image info\n";
            return false;
        }
        m_waterRefractionTransientHandles[i] = waterRefractionHandle;
        m_waterRefractionImages[i] = waterRefractionInfo->image;
        m_waterRefractionImageViews[i] = waterRefractionInfo->view;
        setObjectName(
            VK_OBJECT_TYPE_IMAGE,
            vkHandleToUint64(m_waterRefractionImages[i]),
            waterRefractionDesc.debugName.c_str()
        );

        // Planar reflections are deliberately half resolution: their projected
        // sample is normal-distorted by the water surface, so full-resolution
        // raster cost does not survive to the final pixel. These targets are
        // single-sampled and the pass is enabled only when the main pipelines
        // are single-sampled too (the viewer/showcase default).
        TransientImageDesc waterReflectionDesc{};
        waterReflectionDesc.imageType = VK_IMAGE_TYPE_2D;
        waterReflectionDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        waterReflectionDesc.format = m_hdrColorFormat;
        waterReflectionDesc.extent = {
            m_waterReflectionExtent.width, m_waterReflectionExtent.height, 1u};
        waterReflectionDesc.usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        waterReflectionDesc.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        waterReflectionDesc.mipLevels = 1;
        waterReflectionDesc.arrayLayers = 1;
        waterReflectionDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        waterReflectionDesc.tiling = VK_IMAGE_TILING_OPTIMAL;
        waterReflectionDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        waterReflectionDesc.firstPass = FrameArenaPass::Main;
        waterReflectionDesc.lastPass = FrameArenaPass::Main;
        waterReflectionDesc.debugName =
            "water.reflection.planar[" + std::to_string(i) + "]";
        const TransientImageHandle waterReflectionHandle =
            m_frameArena.createTransientImage(
                waterReflectionDesc, FrameArenaImageLifetime::Persistent);
        if (waterReflectionHandle == kInvalidTransientImageHandle) {
            VOX_LOGE("render") << "failed creating planar water reflection image\n";
            return false;
        }
        const TransientImageInfo* waterReflectionInfo =
            m_frameArena.getTransientImage(waterReflectionHandle);
        if (waterReflectionInfo == nullptr ||
            waterReflectionInfo->image == VK_NULL_HANDLE ||
            waterReflectionInfo->view == VK_NULL_HANDLE) {
            VOX_LOGE("render") << "failed creating planar water reflection image\n";
            return false;
        }
        m_waterReflectionTransientHandles[i] = waterReflectionHandle;
        m_waterReflectionImages[i] = waterReflectionInfo->image;
        m_waterReflectionImageViews[i] = waterReflectionInfo->view;
        setObjectName(
            VK_OBJECT_TYPE_IMAGE,
            vkHandleToUint64(m_waterReflectionImages[i]),
            waterReflectionDesc.debugName.c_str());

        TransientImageDesc waterReflectionDepthDesc{};
        waterReflectionDepthDesc.imageType = VK_IMAGE_TYPE_2D;
        waterReflectionDepthDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        waterReflectionDepthDesc.format = m_depthFormat;
        waterReflectionDepthDesc.extent = waterReflectionDesc.extent;
        waterReflectionDepthDesc.usage =
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        waterReflectionDepthDesc.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        waterReflectionDepthDesc.mipLevels = 1;
        waterReflectionDepthDesc.arrayLayers = 1;
        waterReflectionDepthDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        waterReflectionDepthDesc.tiling = VK_IMAGE_TILING_OPTIMAL;
        waterReflectionDepthDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        waterReflectionDepthDesc.firstPass = FrameArenaPass::Main;
        waterReflectionDepthDesc.lastPass = FrameArenaPass::Main;
        waterReflectionDepthDesc.debugName =
            "water.reflection.depth[" + std::to_string(i) + "]";
        const TransientImageHandle waterReflectionDepthHandle =
            m_frameArena.createTransientImage(
                waterReflectionDepthDesc, FrameArenaImageLifetime::Persistent);
        if (waterReflectionDepthHandle == kInvalidTransientImageHandle) {
            VOX_LOGE("render") << "failed creating planar water reflection depth image\n";
            return false;
        }
        const TransientImageInfo* waterReflectionDepthInfo =
            m_frameArena.getTransientImage(waterReflectionDepthHandle);
        if (waterReflectionDepthInfo == nullptr ||
            waterReflectionDepthInfo->image == VK_NULL_HANDLE ||
            waterReflectionDepthInfo->view == VK_NULL_HANDLE) {
            VOX_LOGE("render") << "failed creating planar water reflection depth image\n";
            return false;
        }
        m_waterReflectionDepthTransientHandles[i] = waterReflectionDepthHandle;
        m_waterReflectionDepthImages[i] = waterReflectionDepthInfo->image;
        m_waterReflectionDepthImageViews[i] = waterReflectionDepthInfo->view;
        setObjectName(
            VK_OBJECT_TYPE_IMAGE,
            vkHandleToUint64(m_waterReflectionDepthImages[i]),
            waterReflectionDepthDesc.debugName.c_str());
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

bool RendererBackend::createWaterReflectionHistoryTargets() {
    destroyWaterReflectionHistoryTargets();
    if (m_renderExtent.width == 0u || m_renderExtent.height == 0u) {
        return false;
    }

    for (std::uint32_t i = 0; i < 2u; ++i) {
        TransientImageDesc colorDesc{};
        colorDesc.imageType = VK_IMAGE_TYPE_2D;
        colorDesc.viewType = VK_IMAGE_VIEW_TYPE_2D;
        colorDesc.format = m_hdrColorFormat;
        colorDesc.extent = {m_renderExtent.width, m_renderExtent.height, 1u};
        colorDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        colorDesc.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorDesc.mipLevels = 1;
        colorDesc.arrayLayers = 1;
        colorDesc.samples = VK_SAMPLE_COUNT_1_BIT;
        colorDesc.tiling = VK_IMAGE_TILING_OPTIMAL;
        colorDesc.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorDesc.firstPass = FrameArenaPass::Main;
        colorDesc.lastPass = FrameArenaPass::Main;
        colorDesc.debugName =
            "water.reflection.history.color[" + std::to_string(i) + "]";
        const TransientImageHandle colorHandle = m_frameArena.createTransientImage(
            colorDesc, FrameArenaImageLifetime::Persistent);
        const TransientImageInfo* colorInfo = m_frameArena.getTransientImage(colorHandle);
        if (colorHandle == kInvalidTransientImageHandle || colorInfo == nullptr ||
            colorInfo->image == VK_NULL_HANDLE || colorInfo->view == VK_NULL_HANDLE) {
            VOX_LOGE("render") << "failed creating water reflection color history\n";
            destroyWaterReflectionHistoryTargets();
            return false;
        }
        m_waterReflectionHistoryTransientHandles[i] = colorHandle;
        m_waterReflectionHistoryImages[i] = colorInfo->image;
        m_waterReflectionHistoryImageViews[i] = colorInfo->view;
        setObjectName(
            VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(colorInfo->image),
            colorDesc.debugName.c_str());

        TransientImageDesc depthDesc = colorDesc;
        depthDesc.format = VK_FORMAT_R32_SFLOAT;
        depthDesc.debugName =
            "water.reflection.history.depth[" + std::to_string(i) + "]";
        const TransientImageHandle depthHandle = m_frameArena.createTransientImage(
            depthDesc, FrameArenaImageLifetime::Persistent);
        const TransientImageInfo* depthInfo = m_frameArena.getTransientImage(depthHandle);
        if (depthHandle == kInvalidTransientImageHandle || depthInfo == nullptr ||
            depthInfo->image == VK_NULL_HANDLE || depthInfo->view == VK_NULL_HANDLE) {
            VOX_LOGE("render") << "failed creating water reflection depth history\n";
            destroyWaterReflectionHistoryTargets();
            return false;
        }
        m_waterReflectionHistoryDepthTransientHandles[i] = depthHandle;
        m_waterReflectionHistoryDepthImages[i] = depthInfo->image;
        m_waterReflectionHistoryDepthImageViews[i] = depthInfo->view;
        setObjectName(
            VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(depthInfo->image),
            depthDesc.debugName.c_str());
    }
    m_waterReflectionHistoryImageInitialized = {false, false};
    m_waterReflectionHistoryDepthInitialized = {false, false};
    m_waterReflectionHistoryIndex = 0u;
    m_waterReflectionHistoryValid = false;
    m_waterReflectionPreviousAvailable = false;
    m_waterReflectionPreviousPlaneValid = false;
    return true;
}

void RendererBackend::destroyWaterReflectionHistoryTargets() {
    for (std::uint32_t i = 0; i < 2u; ++i) {
        if (m_waterReflectionHistoryTransientHandles[i] != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(m_waterReflectionHistoryTransientHandles[i]);
        }
        if (m_waterReflectionHistoryDepthTransientHandles[i] != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(m_waterReflectionHistoryDepthTransientHandles[i]);
        }
        m_waterReflectionHistoryTransientHandles[i] = kInvalidTransientImageHandle;
        m_waterReflectionHistoryDepthTransientHandles[i] = kInvalidTransientImageHandle;
        m_waterReflectionHistoryImages[i] = VK_NULL_HANDLE;
        m_waterReflectionHistoryImageViews[i] = VK_NULL_HANDLE;
        m_waterReflectionHistoryDepthImages[i] = VK_NULL_HANDLE;
        m_waterReflectionHistoryDepthImageViews[i] = VK_NULL_HANDLE;
        m_waterReflectionHistoryImageInitialized[i] = false;
        m_waterReflectionHistoryDepthInitialized[i] = false;
    }
    m_waterReflectionHistoryIndex = 0u;
    m_waterReflectionHistoryValid = false;
    m_waterReflectionPreviousAvailable = false;
    m_waterReflectionPreviousPlaneValid = false;
}

bool RendererBackend::useMergedDepthPrepass() const {
    static const bool s_disabled = []() {
        const char* env = std::getenv("ODAI_MERGED_PREPASS");
        return env != nullptr && env[0] == '0';
    }();
    return !s_disabled && m_colorSampleCount == VK_SAMPLE_COUNT_1_BIT;
}

bool RendererBackend::createMsaaColorTargets() {
    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    m_msaaColorImages.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageViews.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageInitialized.assign(imageCount, false);
    m_msaaColorImageAllocations.assign(imageCount, VK_NULL_HANDLE);

    // AT ONE SAMPLE THERE IS NOTHING TO RESOLVE, so there is no reason for this
    // image to exist. It used to be created 1-sample anyway and the main pass
    // resolved into hdrResolve regardless -- an average-resolve from a 1-sample
    // image to an identical 1-sample image, which is both a full-resolution copy
    // of the whole HDR target every frame and a spec violation
    // (VUID-VkRenderingAttachmentInfo-imageView-06861 requires RESOLVE_MODE_NONE
    // for a 1-sample view). The FNV viewer defaults to ODAI_MSAA=1, so every
    // frame it has ever rendered paid for it.
    //
    // Leaving the handles null is the signal: the main pass renders straight
    // into hdrResolve instead. At 3412x1920 R16G16B16A16 that is ~52 MB per
    // swapchain image of VRAM back as well.
    if (m_colorSampleCount == VK_SAMPLE_COUNT_1_BIT) {
        VOX_LOGI("render") << "MSAA disabled (1 sample): main pass renders directly into "
                              "hdrResolve, no resolve attachment";
        return true;
    }

    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = m_hdrColorFormat;
        imageCreateInfo.extent.width = m_renderExtent.width;
        imageCreateInfo.extent.height = m_renderExtent.height;
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


bool RendererBackend::createTaaTargets() {
    // Two swapchain-sized HDR images ping-ponged as TAA history. NOT arena
    // transients: the arena reclaims per frame, and history must survive into
    // the next one. Dedicated allocations -- two images, once per resize.
    for (std::uint32_t i = 0; i < 2u; ++i) {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = m_hdrColorFormat;
        // Output extent when the temporal upscaler is running -- history has to
        // live on the grid the result is reconstructed onto, not the grid the
        // scene was rasterized on. Render extent otherwise, which is what plain
        // TAA wants.
        const VkExtent2D historyExtent = temporalOutputExtent();
        imageCreateInfo.extent = {historyExtent.width, historyExtent.height, 1u};
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        const VkResult imageResult =
            vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_taaImages[i]);
        if (imageResult != VK_SUCCESS) {
            logVkFailure("vkCreateImage(taa)", imageResult);
            return false;
        }
        VkMemoryRequirements memoryRequirements{};
        vkGetImageMemoryRequirements(m_device, m_taaImages[i], &memoryRequirements);
        const uint32_t memoryTypeIndex = findMemoryTypeIndex(
            m_physicalDevice, memoryRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            VOX_LOGI("render") << "no memory type for TAA image\n";
            return false;
        }
        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        const VkResult allocResult =
            vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_taaImageMemories[i]);
        if (allocResult != VK_SUCCESS) {
            logVkFailure("vkAllocateMemory(taa)", allocResult);
            return false;
        }
        const VkResult bindResult =
            vkBindImageMemory(m_device, m_taaImages[i], m_taaImageMemories[i], 0);
        if (bindResult != VK_SUCCESS) {
            logVkFailure("vkBindImageMemory(taa)", bindResult);
            return false;
        }
        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_taaImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_hdrColorFormat;
        viewCreateInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0u, 1u, 0u, 1u};
        const VkResult viewResult =
            vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_taaImageViews[i]);
        if (viewResult != VK_SUCCESS) {
            logVkFailure("vkCreateImageView(taa)", viewResult);
            return false;
        }
        const std::string imageName = "taa.history.image." + std::to_string(i);
        setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_taaImages[i]), imageName.c_str());
    }
    // Fresh images: whatever history existed died with the old swapchain.
    m_taaImageInitialized = {false, false};
    m_taaHistoryValid = false;
    m_screenSpaceGiHistoryValid = false;
    m_taaHistoryIndex = 0;
    return true;
}

void RendererBackend::destroyTaaTargets() {
    for (std::uint32_t i = 0; i < 2u; ++i) {
        if (m_taaImageViews[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, m_taaImageViews[i], nullptr);
            m_taaImageViews[i] = VK_NULL_HANDLE;
        }
        if (m_taaImages[i] != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, m_taaImages[i], nullptr);
            m_taaImages[i] = VK_NULL_HANDLE;
        }
        if (m_taaImageMemories[i] != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, m_taaImageMemories[i], nullptr);
            m_taaImageMemories[i] = VK_NULL_HANDLE;
        }
    }
    m_taaImageInitialized = {false, false};
    m_taaHistoryValid = false;
    m_screenSpaceGiHistoryValid = false;
}

void RendererBackend::destroyHdrResolveTargets() {
    if (m_hdrResolveSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_hdrResolveSampler, nullptr);
        m_hdrResolveSampler = VK_NULL_HANDLE;
    }

    for (TransientImageHandle handle : m_waterReflectionDepthTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_waterReflectionDepthImageViews.clear();
    m_waterReflectionDepthImages.clear();
    m_waterReflectionDepthTransientHandles.clear();
    m_waterReflectionDepthImageInitialized.clear();
    m_waterReflectionDepthSampled.clear();

    for (TransientImageHandle handle : m_waterReflectionTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_waterReflectionImageViews.clear();
    m_waterReflectionImages.clear();
    m_waterReflectionTransientHandles.clear();
    m_waterReflectionImageInitialized.clear();
    m_waterReflectionExtent = {};

    for (TransientImageHandle handle : m_waterRefractionTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    m_waterRefractionImageViews.clear();
    m_waterRefractionImages.clear();
    m_waterRefractionTransientHandles.clear();
    m_waterRefractionImageInitialized.clear();

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

void RendererBackend::destroyMsaaColorTargets() {
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

void RendererBackend::destroyDepthTargets() {
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

void RendererBackend::destroyAoTargets() {
    if (m_sunShaftSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_sunShaftSampler, nullptr);
        m_sunShaftSampler = VK_NULL_HANDLE;
    }
    if (m_ssaoSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_ssaoSampler, nullptr);
        m_ssaoSampler = VK_NULL_HANDLE;
    }
    if (m_xegtaoPointSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_xegtaoPointSampler, nullptr);
        m_xegtaoPointSampler = VK_NULL_HANDLE;
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
    m_sunShaftImageHasContent.clear();

    for (TransientImageHandle handle : m_ssaoRawTransientHandles) {
        if (handle != kInvalidTransientImageHandle) {
            m_frameArena.destroyTransientImage(handle);
        }
    }
    for (uint32_t level = 0; level < kXeGtaoDepthMipCount; ++level) {
        m_xegtaoDepthImageViews[level].clear();
        m_xegtaoDepthImages[level].clear();
        m_xegtaoDepthImageMemories[level].clear();
        m_xegtaoDepthTransientHandles[level].clear();
    }
    m_xegtaoDepthInitialized.clear();
    m_xegtaoAoTermImageViews.clear();
    m_xegtaoAoTermImages.clear();
    m_xegtaoAoTermImageMemories.clear();
    m_xegtaoAoTermTransientHandles.clear();
    m_xegtaoAoTermInitialized.clear();
    m_xegtaoBentNormalImageViews.clear();
    m_xegtaoBentNormalImages.clear();
    m_xegtaoBentNormalImageMemories.clear();
    m_xegtaoBentNormalTransientHandles.clear();
    m_xegtaoBentNormalInitialized.clear();
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
    m_velocityImageViews.clear();
    m_velocityImages.clear();
    m_velocityImageMemories.clear();
    m_velocityTransientHandles.clear();
    m_velocityImageInitialized.clear();
    m_normalDepthImages.clear();
    m_normalDepthImageMemories.clear();
    m_normalDepthTransientHandles.clear();
    m_normalDepthImageInitialized.clear();
}

} // namespace odai::render
