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
#include <bit>
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

constexpr std::uint32_t makeFourCc(char a, char b, char c, char d) {
    return
        static_cast<std::uint32_t>(static_cast<std::uint8_t>(a)) |
        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(b)) << 8u) |
        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(c)) << 16u) |
        (static_cast<std::uint32_t>(static_cast<std::uint8_t>(d)) << 24u);
}

struct DdsPixelFormat {
    std::uint32_t size = 0;
    std::uint32_t flags = 0;
    std::uint32_t fourCc = 0;
    std::uint32_t rgbBitCount = 0;
    std::uint32_t rMask = 0;
    std::uint32_t gMask = 0;
    std::uint32_t bMask = 0;
    std::uint32_t aMask = 0;
};

struct DdsHeader {
    std::uint32_t size = 0;
    std::uint32_t flags = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    std::uint32_t pitchOrLinearSize = 0;
    std::uint32_t depth = 0;
    std::uint32_t mipMapCount = 0;
    std::uint32_t reserved1[11]{};
    DdsPixelFormat pixelFormat{};
    std::uint32_t caps = 0;
    std::uint32_t caps2 = 0;
    std::uint32_t caps3 = 0;
    std::uint32_t caps4 = 0;
    std::uint32_t reserved2 = 0;
};

struct DdsMipInfo {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
};

struct DdsCompressedImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t mipLevels = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::vector<std::uint8_t> pixelData;
    std::vector<DdsMipInfo> mipInfos;
};

struct DdsRgbaImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> pixelData;
};

bool supportsBc1DiffuseAtlas(VkPhysicalDevice physicalDevice) {
    VkFormatProperties formatProperties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, &formatProperties);
    const VkFormatFeatureFlags features = formatProperties.optimalTilingFeatures;
    return
        (features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0 &&
        (features & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) != 0;
}

bool supportsBc3PlantAtlas(VkPhysicalDevice physicalDevice) {
    VkFormatProperties formatProperties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_BC3_UNORM_BLOCK, &formatProperties);
    const VkFormatFeatureFlags features = formatProperties.optimalTilingFeatures;
    return
        (features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0 &&
        (features & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) != 0;
}

bool supportsBc5WaterNormalTexture(VkPhysicalDevice physicalDevice) {
    VkFormatProperties formatProperties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_BC5_UNORM_BLOCK, &formatProperties);
    const VkFormatFeatureFlags features = formatProperties.optimalTilingFeatures;
    return
        (features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0 &&
        (features & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) != 0;
}

bool loadCompressedDdsFile(
    const std::filesystem::path& path,
    std::uint32_t expectedFourCc,
    VkFormat format,
    std::uint32_t bytesPerBlock,
    DdsCompressedImage& outImage
) {
    outImage = DdsCompressedImage{};

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return false;
    }

    const std::streamsize fileSize = stream.tellg();
    if (fileSize <= 0) {
        return false;
    }
    stream.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(fileSize));
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), fileSize)) {
        return false;
    }
    if (bytes.size() < (4u + sizeof(DdsHeader))) {
        return false;
    }
    if (std::memcmp(bytes.data(), "DDS ", 4) != 0) {
        return false;
    }

    DdsHeader header{};
    std::memcpy(&header, bytes.data() + 4u, sizeof(DdsHeader));
    if (header.size != 124u || header.pixelFormat.size != 32u) {
        return false;
    }
    if (header.pixelFormat.fourCc != expectedFourCc) {
        return false;
    }
    if (header.width == 0u || header.height == 0u) {
        return false;
    }

    const std::uint32_t mipLevels = std::max(header.mipMapCount, 1u);
    const std::size_t dataOffset = 4u + sizeof(DdsHeader);
    if (bytes.size() < dataOffset) {
        return false;
    }

    std::vector<DdsMipInfo> mipInfos;
    mipInfos.reserve(mipLevels);
    VkDeviceSize runningOffset = 0;
    for (std::uint32_t mipIndex = 0; mipIndex < mipLevels; ++mipIndex) {
        const std::uint32_t mipWidth = std::max(1u, header.width >> mipIndex);
        const std::uint32_t mipHeight = std::max(1u, header.height >> mipIndex);
        const std::uint32_t blockWidth = std::max(1u, (mipWidth + 3u) / 4u);
        const std::uint32_t blockHeight = std::max(1u, (mipHeight + 3u) / 4u);
        const VkDeviceSize mipSize =
            static_cast<VkDeviceSize>(blockWidth) * static_cast<VkDeviceSize>(blockHeight) * bytesPerBlock;
        mipInfos.push_back(DdsMipInfo{mipWidth, mipHeight, runningOffset, mipSize});
        runningOffset += mipSize;
    }

    const std::size_t payloadSize = bytes.size() - dataOffset;
    if (payloadSize < static_cast<std::size_t>(runningOffset)) {
        return false;
    }

    outImage.width = header.width;
    outImage.height = header.height;
    outImage.mipLevels = mipLevels;
    outImage.format = format;
    outImage.pixelData.assign(bytes.begin() + static_cast<std::ptrdiff_t>(dataOffset), bytes.end());
    outImage.mipInfos = std::move(mipInfos);
    return true;
}

std::uint8_t decodeDdsChannel(std::uint32_t packedPixel, std::uint32_t mask, std::uint8_t defaultValue) {
    if (mask == 0u) {
        return defaultValue;
    }

    const std::uint32_t shift = std::countr_zero(mask);
    const std::uint32_t shiftedMask = mask >> shift;
    const std::uint32_t componentBits = std::popcount(shiftedMask);
    if (componentBits == 0u) {
        return defaultValue;
    }

    const std::uint32_t componentValue = (packedPixel & mask) >> shift;
    const std::uint32_t componentMax = (1u << componentBits) - 1u;
    if (componentMax == 0u) {
        return defaultValue;
    }

    return static_cast<std::uint8_t>((componentValue * 255u + (componentMax / 2u)) / componentMax);
}

bool loadUncompressedRgbaDdsFile(const std::filesystem::path& path, DdsRgbaImage& outImage) {
    outImage = DdsRgbaImage{};

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return false;
    }

    const std::streamsize fileSize = stream.tellg();
    if (fileSize <= 0) {
        return false;
    }
    stream.seekg(0, std::ios::beg);

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(fileSize));
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), fileSize)) {
        return false;
    }
    if (bytes.size() < (4u + sizeof(DdsHeader))) {
        return false;
    }
    if (std::memcmp(bytes.data(), "DDS ", 4) != 0) {
        return false;
    }

    DdsHeader header{};
    std::memcpy(&header, bytes.data() + 4u, sizeof(DdsHeader));
    if (header.size != 124u || header.pixelFormat.size != 32u) {
        return false;
    }
    if (header.width == 0u || header.height == 0u) {
        return false;
    }
    if (header.pixelFormat.fourCc != 0u || header.pixelFormat.rgbBitCount != 32u) {
        return false;
    }

    const std::size_t dataOffset = 4u + sizeof(DdsHeader);
    const std::size_t pixelCount = static_cast<std::size_t>(header.width) * static_cast<std::size_t>(header.height);
    const std::size_t firstMipSize = pixelCount * 4u;
    if ((bytes.size() - dataOffset) < firstMipSize) {
        return false;
    }

    outImage.width = header.width;
    outImage.height = header.height;
    outImage.pixelData.resize(firstMipSize);

    for (std::size_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex) {
        std::uint32_t packedPixel = 0u;
        std::memcpy(
            &packedPixel,
            bytes.data() + dataOffset + (pixelIndex * sizeof(std::uint32_t)),
            sizeof(std::uint32_t));
        outImage.pixelData[pixelIndex * 4u + 0u] =
            decodeDdsChannel(packedPixel, header.pixelFormat.rMask, 0u);
        outImage.pixelData[pixelIndex * 4u + 1u] =
            decodeDdsChannel(packedPixel, header.pixelFormat.gMask, 0u);
        outImage.pixelData[pixelIndex * 4u + 2u] =
            decodeDdsChannel(packedPixel, header.pixelFormat.bMask, 0u);
        outImage.pixelData[pixelIndex * 4u + 3u] =
            decodeDdsChannel(packedPixel, header.pixelFormat.aMask, 255u);
    }

    return true;
}

std::filesystem::path resolveRendererAssetPath(const std::filesystem::path& relativePath) {
    std::vector<std::filesystem::path> baseCandidates;
    baseCandidates.reserve(6);

#if defined(ODAI_PROJECT_SOURCE_DIR)
    baseCandidates.emplace_back(std::filesystem::path{ODAI_PROJECT_SOURCE_DIR});
#endif

    std::error_code cwdError;
    const std::filesystem::path cwd = std::filesystem::current_path(cwdError);
    if (!cwdError) {
        baseCandidates.push_back(cwd);
        baseCandidates.push_back(cwd / "..");
        baseCandidates.push_back(cwd / ".." / "..");
        baseCandidates.push_back(cwd / ".." / ".." / "..");
    }

    for (const std::filesystem::path& base : baseCandidates) {
        const std::filesystem::path candidate = base / relativePath;
        std::error_code existsError;
        if (!std::filesystem::exists(candidate, existsError) || existsError) {
            continue;
        }

        std::error_code canonicalError;
        const std::filesystem::path canonicalPath = std::filesystem::weakly_canonical(candidate, canonicalError);
        if (!canonicalError) {
            return canonicalPath;
        }
        return candidate;
    }

    return relativePath;
}

constexpr uint32_t kGeneratedWaterNormalTextureSize = 128u;

float wrapUnit(float value) {
    return value - std::floor(value);
}

float sampleGeneratedWaterNormalHeight(float u, float v) {
    constexpr float kTwoPi = 6.28318530718f;
    const float warpX =
        std::sin(((u * 1.0f) + (v * 0.8f)) * kTwoPi) * 0.035f +
        std::cos(((u * 2.2f) - (v * 1.7f)) * kTwoPi) * 0.018f;
    const float warpY =
        std::cos(((u * 0.9f) - (v * 1.1f)) * kTwoPi) * 0.032f +
        std::sin(((u * 1.8f) + (v * 2.4f)) * kTwoPi) * 0.016f;
    const float uu = wrapUnit(u + warpX);
    const float vv = wrapUnit(v + warpY);
    return
        std::sin(((uu * 1.0f) + (vv * 0.75f)) * kTwoPi) * 0.44f +
        std::cos(((uu * 1.65f) - (vv * 1.25f)) * kTwoPi) * 0.28f +
        std::sin(((uu * 3.20f) + (vv * 2.65f)) * kTwoPi) * 0.14f +
        std::cos(((uu * 4.80f) - (vv * 4.20f)) * kTwoPi) * 0.07f;
}

std::vector<std::uint8_t> generateWaterNormalTexturePixels() {
    constexpr uint32_t kSize = kGeneratedWaterNormalTextureSize;
    constexpr float kNormalStrength = 2.2f;
    const float du = 1.0f / static_cast<float>(kSize);
    const float dv = 1.0f / static_cast<float>(kSize);
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(kSize) * static_cast<std::size_t>(kSize) * 4u);
    for (uint32_t y = 0; y < kSize; ++y) {
        for (uint32_t x = 0; x < kSize; ++x) {
            const float u = (static_cast<float>(x) + 0.5f) * du;
            const float v = (static_cast<float>(y) + 0.5f) * dv;
            const float heightL = sampleGeneratedWaterNormalHeight(wrapUnit(u - du), v);
            const float heightR = sampleGeneratedWaterNormalHeight(wrapUnit(u + du), v);
            const float heightD = sampleGeneratedWaterNormalHeight(u, wrapUnit(v - dv));
            const float heightU = sampleGeneratedWaterNormalHeight(u, wrapUnit(v + dv));
            const float dHdU = (heightR - heightL) * kNormalStrength;
            const float dHdV = (heightU - heightD) * kNormalStrength;
            const float tangentX = -dHdU;
            const float tangentY = -dHdV;
            const float tangentZ = 1.0f;
            const float invLen =
                1.0f / std::sqrt((tangentX * tangentX) + (tangentY * tangentY) + (tangentZ * tangentZ));
            const float packedX = (tangentX * invLen * 0.5f) + 0.5f;
            const float packedY = (tangentY * invLen * 0.5f) + 0.5f;
            const float packedZ = (tangentZ * invLen * 0.5f) + 0.5f;
            const std::size_t pixelIndex =
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(kSize) + static_cast<std::size_t>(x)) * 4u;
            pixels[pixelIndex + 0u] = static_cast<std::uint8_t>(std::clamp(packedX, 0.0f, 1.0f) * 255.0f);
            pixels[pixelIndex + 1u] = static_cast<std::uint8_t>(std::clamp(packedY, 0.0f, 1.0f) * 255.0f);
            pixels[pixelIndex + 2u] = static_cast<std::uint8_t>(std::clamp(packedZ, 0.0f, 1.0f) * 255.0f);
            pixels[pixelIndex + 3u] = 255u;
        }
    }
    return pixels;
}

} // namespace

bool RendererBackend::createEnvironmentResources() {
    if (!createDiffuseTextureResources()) {
        VOX_LOGE("render") << "diffuse texture creation failed\n";
        return false;
    }
    if (!createWaterNormalTextureResources()) {
        VOX_LOGW("render") << "generated water normal texture creation failed; keeping procedural water normals only";
    }
    VOX_LOGI("render") << "environment resources ready";
    return true;
}


bool RendererBackend::createWaterNormalTextureResources() {
    bool hasWaterAllocation = (m_waterNormalTextureMemory != VK_NULL_HANDLE);
    if (m_vmaAllocator != VK_NULL_HANDLE) {
        hasWaterAllocation = (m_waterNormalTextureAllocation != VK_NULL_HANDLE);
    }
    if (m_waterNormalTextureImage != VK_NULL_HANDLE &&
        hasWaterAllocation &&
        m_waterNormalTextureImageView != VK_NULL_HANDLE &&
        m_waterNormalTextureSampler != VK_NULL_HANDLE) {
        return true;
    }

    auto destroyWaterTextureResourcesPartial = [&]() {
        if (m_waterNormalTextureSampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_device, m_waterNormalTextureSampler, nullptr);
            m_waterNormalTextureSampler = VK_NULL_HANDLE;
        }
        if (m_waterNormalTextureImageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, m_waterNormalTextureImageView, nullptr);
            m_waterNormalTextureImageView = VK_NULL_HANDLE;
        }
        if (m_waterNormalTextureImage != VK_NULL_HANDLE) {
            if (m_vmaAllocator != VK_NULL_HANDLE && m_waterNormalTextureAllocation != VK_NULL_HANDLE) {
                vmaDestroyImage(m_vmaAllocator, m_waterNormalTextureImage, m_waterNormalTextureAllocation);
                m_waterNormalTextureAllocation = VK_NULL_HANDLE;
            } else {
                vkDestroyImage(m_device, m_waterNormalTextureImage, nullptr);
            }
            m_waterNormalTextureImage = VK_NULL_HANDLE;
        }
        if (m_waterNormalTextureMemory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, m_waterNormalTextureMemory, nullptr);
            m_waterNormalTextureMemory = VK_NULL_HANDLE;
        }
        m_waterNormalTextureAllocation = VK_NULL_HANDLE;
    };

    const std::filesystem::path waterNormalPath = resolveRendererAssetPath("assets/water.dds");
    if (supportsBc5WaterNormalTexture(m_physicalDevice)) {
        DdsCompressedImage ddsImage{};
        if (loadCompressedDdsFile(
                waterNormalPath,
                makeFourCc('B', 'C', '5', 'U'),
                VK_FORMAT_BC5_UNORM_BLOCK,
                16u,
                ddsImage
            )) {
            VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
            VkBufferCreateInfo stagingCreateInfo{};
            stagingCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            stagingCreateInfo.size = static_cast<VkDeviceSize>(ddsImage.pixelData.size());
            stagingCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            stagingCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkResult result = vkCreateBuffer(m_device, &stagingCreateInfo, nullptr, &stagingBuffer);
            if (result != VK_SUCCESS) {
                logVkFailure("vkCreateBuffer(waterNormalDdsStaging)", result);
                return false;
            }
            setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(stagingBuffer), "water.normal.dds.staging.buffer");

            VkMemoryRequirements stagingMemReq{};
            vkGetBufferMemoryRequirements(m_device, stagingBuffer, &stagingMemReq);
            uint32_t memoryTypeIndex = findMemoryTypeIndex(
                m_physicalDevice,
                stagingMemReq.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            );
            if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                vkDestroyBuffer(m_device, stagingBuffer, nullptr);
                VOX_LOGW("render") << "no staging memory type for DDS water normal texture";
            } else {
                VkMemoryAllocateInfo stagingAllocInfo{};
                stagingAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                stagingAllocInfo.allocationSize = stagingMemReq.size;
                stagingAllocInfo.memoryTypeIndex = memoryTypeIndex;
                result = vkAllocateMemory(m_device, &stagingAllocInfo, nullptr, &stagingMemory);
                if (result == VK_SUCCESS && vkBindBufferMemory(m_device, stagingBuffer, stagingMemory, 0) == VK_SUCCESS) {
                    void* mapped = nullptr;
                    result = vkMapMemory(
                        m_device,
                        stagingMemory,
                        0,
                        static_cast<VkDeviceSize>(ddsImage.pixelData.size()),
                        0,
                        &mapped
                    );
                    if (result == VK_SUCCESS && mapped != nullptr) {
                        std::memcpy(mapped, ddsImage.pixelData.data(), ddsImage.pixelData.size());
                        vkUnmapMemory(m_device, stagingMemory);

                        VkImageCreateInfo imageCreateInfo{};
                        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
                        imageCreateInfo.format = ddsImage.format;
                        imageCreateInfo.extent = {ddsImage.width, ddsImage.height, 1u};
                        imageCreateInfo.mipLevels = ddsImage.mipLevels;
                        imageCreateInfo.arrayLayers = 1;
                        imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
                        imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
                        imageCreateInfo.usage = (m_copyMemoryToImage != nullptr)
                            ? (VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT | VK_IMAGE_USAGE_SAMPLED_BIT)
                            : (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
                        imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                        imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                        if (m_vmaAllocator != VK_NULL_HANDLE) {
                            VmaAllocationCreateInfo allocationCreateInfo{};
                            allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                            allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                            result = vmaCreateImage(
                                m_vmaAllocator,
                                &imageCreateInfo,
                                &allocationCreateInfo,
                                &m_waterNormalTextureImage,
                                &m_waterNormalTextureAllocation,
                                nullptr
                            );
                        } else {
                            result = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_waterNormalTextureImage);
                            if (result == VK_SUCCESS) {
                                VkMemoryRequirements imageMemReq{};
                                vkGetImageMemoryRequirements(m_device, m_waterNormalTextureImage, &imageMemReq);
                                memoryTypeIndex = findMemoryTypeIndex(
                                    m_physicalDevice,
                                    imageMemReq.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                );
                                if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                                    result = VK_ERROR_FEATURE_NOT_PRESENT;
                                } else {
                                    VkMemoryAllocateInfo imageAllocInfo{};
                                    imageAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                                    imageAllocInfo.allocationSize = imageMemReq.size;
                                    imageAllocInfo.memoryTypeIndex = memoryTypeIndex;
                                    result = vkAllocateMemory(m_device, &imageAllocInfo, nullptr, &m_waterNormalTextureMemory);
                                    if (result == VK_SUCCESS) {
                                        result = vkBindImageMemory(m_device, m_waterNormalTextureImage, m_waterNormalTextureMemory, 0);
                                    }
                                }
                            }
                        }

                        if (result == VK_SUCCESS) {
                            setObjectName(
                                VK_OBJECT_TYPE_IMAGE,
                                vkHandleToUint64(m_waterNormalTextureImage),
                                "water.normal.image"
                            );

                            const bool useDdsHostCopy =
                                m_copyMemoryToImage != nullptr && m_transitionImageLayout != nullptr;
                            if (useDdsHostCopy) {
                                VkImageSubresourceRange subresourceRange{};
                                subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                                subresourceRange.baseMipLevel = 0;
                                subresourceRange.levelCount = ddsImage.mipLevels;
                                subresourceRange.baseArrayLayer = 0;
                                subresourceRange.layerCount = 1;

                                VkHostImageLayoutTransitionInfoEXT transition{};
                                transition.sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT;
                                transition.image = m_waterNormalTextureImage;
                                transition.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                                transition.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                                transition.subresourceRange = subresourceRange;
                                result = m_transitionImageLayout(m_device, 1, &transition);

                                if (result == VK_SUCCESS) {
                                    std::vector<VkMemoryToImageCopyEXT> copyRegions;
                                    copyRegions.reserve(ddsImage.mipInfos.size());
                                    for (const DdsMipInfo& mipInfo : ddsImage.mipInfos) {
                                        VkMemoryToImageCopyEXT region{};
                                        region.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT;
                                        region.pHostPointer = ddsImage.pixelData.data() + mipInfo.offset;
                                        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                                        region.imageSubresource.mipLevel =
                                            static_cast<uint32_t>(copyRegions.size());
                                        region.imageSubresource.baseArrayLayer = 0;
                                        region.imageSubresource.layerCount = 1;
                                        region.imageExtent = {mipInfo.width, mipInfo.height, 1u};
                                        copyRegions.push_back(region);
                                    }

                                    VkCopyMemoryToImageInfoEXT copyInfo{};
                                    copyInfo.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT;
                                    copyInfo.dstImage = m_waterNormalTextureImage;
                                    copyInfo.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
                                    copyInfo.regionCount = static_cast<uint32_t>(copyRegions.size());
                                    copyInfo.pRegions = copyRegions.data();
                                    result = m_copyMemoryToImage(m_device, &copyInfo);
                                }
                                if (result == VK_SUCCESS) {
                                    transition.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                                    transition.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                    result = m_transitionImageLayout(m_device, 1, &transition);
                                }
                            }

                            VkCommandPool commandPool = VK_NULL_HANDLE;
                            if (!useDdsHostCopy) {
                            VkCommandPoolCreateInfo commandPoolCreateInfo{};
                            commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                            commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
                            commandPoolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
                            result = vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &commandPool);
                            if (result == VK_SUCCESS) {
                                VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
                                VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
                                commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                                commandBufferAllocateInfo.commandPool = commandPool;
                                commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                                commandBufferAllocateInfo.commandBufferCount = 1;
                                result = vkAllocateCommandBuffers(m_device, &commandBufferAllocateInfo, &commandBuffer);
                                if (result == VK_SUCCESS) {
                                    VkCommandBufferBeginInfo beginInfo{};
                                    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                                    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                                    result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
                                    if (result == VK_SUCCESS) {
                                        VkImageMemoryBarrier barrier{};
                                        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                                        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                                        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                                        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                                        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                                        barrier.image = m_waterNormalTextureImage;
                                        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                                        barrier.subresourceRange.baseMipLevel = 0;
                                        barrier.subresourceRange.levelCount = ddsImage.mipLevels;
                                        barrier.subresourceRange.baseArrayLayer = 0;
                                        barrier.subresourceRange.layerCount = 1;
                                        barrier.srcAccessMask = 0;
                                        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                                        vkCmdPipelineBarrier(
                                            commandBuffer,
                                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                                            0,
                                            0,
                                            nullptr,
                                            0,
                                            nullptr,
                                            1,
                                            &barrier
                                        );

                                        std::vector<VkBufferImageCopy> copyRegions;
                                        copyRegions.reserve(ddsImage.mipInfos.size());
                                        for (const DdsMipInfo& mipInfo : ddsImage.mipInfos) {
                                            VkBufferImageCopy copyRegion{};
                                            copyRegion.bufferOffset = mipInfo.offset;
                                            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                                            copyRegion.imageSubresource.mipLevel =
                                                static_cast<uint32_t>(copyRegions.size());
                                            copyRegion.imageSubresource.baseArrayLayer = 0;
                                            copyRegion.imageSubresource.layerCount = 1;
                                            copyRegion.imageExtent = {mipInfo.width, mipInfo.height, 1u};
                                            copyRegions.push_back(copyRegion);
                                        }
                                        vkCmdCopyBufferToImage(
                                            commandBuffer,
                                            stagingBuffer,
                                            m_waterNormalTextureImage,
                                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                            static_cast<uint32_t>(copyRegions.size()),
                                            copyRegions.data()
                                        );

                                        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                                        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                                        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                                        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                                        vkCmdPipelineBarrier(
                                            commandBuffer,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                            0,
                                            0,
                                            nullptr,
                                            0,
                                            nullptr,
                                            1,
                                            &barrier
                                        );

                                        result = vkEndCommandBuffer(commandBuffer);
                                        if (result == VK_SUCCESS) {
                                            VkSubmitInfo submitInfo{};
                                            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                                            submitInfo.commandBufferCount = 1;
                                            submitInfo.pCommandBuffers = &commandBuffer;
                                            result = vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
                                            if (result == VK_SUCCESS) {
                                                result = vkQueueWaitIdle(m_graphicsQueue);
                                            }
                                        }
                                    }
                                }
                            }
                            } // end if (!useDdsHostCopy)

                            if (result == VK_SUCCESS) {
                                VkImageViewCreateInfo viewCreateInfo{};
                                viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                                viewCreateInfo.image = m_waterNormalTextureImage;
                                viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                                viewCreateInfo.format = ddsImage.format;
                                viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                                viewCreateInfo.subresourceRange.baseMipLevel = 0;
                                viewCreateInfo.subresourceRange.levelCount = ddsImage.mipLevels;
                                viewCreateInfo.subresourceRange.baseArrayLayer = 0;
                                viewCreateInfo.subresourceRange.layerCount = 1;
                                result = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_waterNormalTextureImageView);
                            }
                            if (result == VK_SUCCESS) {
                                setObjectName(
                                    VK_OBJECT_TYPE_IMAGE_VIEW,
                                    vkHandleToUint64(m_waterNormalTextureImageView),
                                    "water.normal.view"
                                );
                                VkSamplerCreateInfo samplerCreateInfo{};
                                samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                                samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
                                samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
                                samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                                samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                                samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                                samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
                                samplerCreateInfo.anisotropyEnable = m_supportsSamplerAnisotropy ? VK_TRUE : VK_FALSE;
                                samplerCreateInfo.maxAnisotropy = m_supportsSamplerAnisotropy
                                    ? std::min(8.0f, m_maxSamplerAnisotropy)
                                    : 1.0f;
                                samplerCreateInfo.minLod = 0.0f;
                                samplerCreateInfo.maxLod = static_cast<float>(ddsImage.mipLevels - 1u);
                                result = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_waterNormalTextureSampler);
                                if (result == VK_SUCCESS) {
                                    setObjectName(
                                        VK_OBJECT_TYPE_SAMPLER,
                                        vkHandleToUint64(m_waterNormalTextureSampler),
                                        "water.normal.sampler"
                                    );
                                    VOX_LOGI("render") << "water normal texture loaded from "
                                                       << waterNormalPath.string()
                                                       << ": " << ddsImage.width << "x" << ddsImage.height
                                                       << ", mips=" << ddsImage.mipLevels
                                                       << ", format=BC5";
                                }
                            }
                            if (result != VK_SUCCESS) {
                                destroyWaterTextureResourcesPartial();
                            }
                            if (commandPool != VK_NULL_HANDLE) {
                                vkDestroyCommandPool(m_device, commandPool, nullptr);
                            }
                        } else {
                            destroyWaterTextureResourcesPartial();
                        }
                    }
                }
                if (stagingMemory != VK_NULL_HANDLE) {
                    vkFreeMemory(m_device, stagingMemory, nullptr);
                }
                vkDestroyBuffer(m_device, stagingBuffer, nullptr);
                if (m_waterNormalTextureSampler != VK_NULL_HANDLE &&
                    m_waterNormalTextureImageView != VK_NULL_HANDLE &&
                    m_waterNormalTextureImage != VK_NULL_HANDLE) {
                    return true;
                }
                VOX_LOGW("render") << "failed to upload DDS water normal texture " << waterNormalPath.string()
                                   << "; falling back to generated normal";
            }
        } else if (!waterNormalPath.empty()) {
            VOX_LOGW("render") << "failed to parse DDS water normal texture " << waterNormalPath.string()
                               << "; falling back to generated normal";
        }
    } else {
        VOX_LOGW("render") << "BC5 water normal textures unsupported on this GPU; falling back to generated normal";
    }

    const std::vector<std::uint8_t> pixelData = generateWaterNormalTexturePixels();

    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageCreateInfo.extent = {kGeneratedWaterNormalTextureSize, kGeneratedWaterNormalTextureSize, 1u};
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    const bool useHostCopy = m_copyMemoryToImage != nullptr && m_transitionImageLayout != nullptr;
    imageCreateInfo.usage = useHostCopy
        ? (VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT | VK_IMAGE_USAGE_SAMPLED_BIT)
        : (VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkResult result = VK_SUCCESS;
    if (m_vmaAllocator != VK_NULL_HANDLE) {
        VmaAllocationCreateInfo allocationCreateInfo{};
        allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        result = vmaCreateImage(
            m_vmaAllocator,
            &imageCreateInfo,
            &allocationCreateInfo,
            &m_waterNormalTextureImage,
            &m_waterNormalTextureAllocation,
            nullptr);
        if (result != VK_SUCCESS) {
            logVkFailure("vmaCreateImage(waterNormal)", result);
            return false;
        }
    } else {
        result = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_waterNormalTextureImage);
        if (result != VK_SUCCESS) {
            logVkFailure("vkCreateImage(waterNormal)", result);
            return false;
        }
        VkMemoryRequirements imageMemReq{};
        vkGetImageMemoryRequirements(m_device, m_waterNormalTextureImage, &imageMemReq);
        uint32_t memoryTypeIndex = findMemoryTypeIndex(
            m_physicalDevice,
            imageMemReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
            vkDestroyImage(m_device, m_waterNormalTextureImage, nullptr);
            m_waterNormalTextureImage = VK_NULL_HANDLE;
            VOX_LOGW("render") << "no device-local memory for water normal texture";
            return false;
        }
        VkMemoryAllocateInfo imageAllocInfo{};
        imageAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        imageAllocInfo.allocationSize = imageMemReq.size;
        imageAllocInfo.memoryTypeIndex = memoryTypeIndex;
        result = vkAllocateMemory(m_device, &imageAllocInfo, nullptr, &m_waterNormalTextureMemory);
        if (result != VK_SUCCESS) {
            logVkFailure("vkAllocateMemory(waterNormal)", result);
            vkDestroyImage(m_device, m_waterNormalTextureImage, nullptr);
            m_waterNormalTextureImage = VK_NULL_HANDLE;
            return false;
        }
        result = vkBindImageMemory(m_device, m_waterNormalTextureImage, m_waterNormalTextureMemory, 0);
        if (result != VK_SUCCESS) {
            logVkFailure("vkBindImageMemory(waterNormal)", result);
            vkDestroyImage(m_device, m_waterNormalTextureImage, nullptr);
            m_waterNormalTextureImage = VK_NULL_HANDLE;
            vkFreeMemory(m_device, m_waterNormalTextureMemory, nullptr);
            m_waterNormalTextureMemory = VK_NULL_HANDLE;
            return false;
        }
    }
    setObjectName(VK_OBJECT_TYPE_IMAGE, vkHandleToUint64(m_waterNormalTextureImage), "water.normal.image");

    if (useHostCopy) {
        VkImageSubresourceRange subresourceRange{};
        subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subresourceRange.baseMipLevel = 0;
        subresourceRange.levelCount = 1;
        subresourceRange.baseArrayLayer = 0;
        subresourceRange.layerCount = 1;

        VkHostImageLayoutTransitionInfoEXT transition{};
        transition.sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT;
        transition.image = m_waterNormalTextureImage;
        transition.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        transition.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        transition.subresourceRange = subresourceRange;
        result = m_transitionImageLayout(m_device, 1, &transition);
        if (result != VK_SUCCESS) {
            logVkFailure("vkTransitionImageLayoutEXT(waterNormal UNDEFINED→GENERAL)", result);
            return false;
        }

        VkMemoryToImageCopyEXT copyRegion{};
        copyRegion.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT;
        copyRegion.pHostPointer = pixelData.data();
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = {kGeneratedWaterNormalTextureSize, kGeneratedWaterNormalTextureSize, 1u};

        VkCopyMemoryToImageInfoEXT copyInfo{};
        copyInfo.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT;
        copyInfo.dstImage = m_waterNormalTextureImage;
        copyInfo.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
        copyInfo.regionCount = 1;
        copyInfo.pRegions = &copyRegion;
        result = m_copyMemoryToImage(m_device, &copyInfo);
        if (result != VK_SUCCESS) {
            logVkFailure("vkCopyMemoryToImageEXT(waterNormal)", result);
            return false;
        }

        transition.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        transition.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        result = m_transitionImageLayout(m_device, 1, &transition);
        if (result != VK_SUCCESS) {
            logVkFailure("vkTransitionImageLayoutEXT(waterNormal GENERAL→SHADER_READ_ONLY)", result);
            return false;
        }
    } else {
        // Staging buffer fallback for drivers without host image copy.
        const VkDeviceSize textureBytes = static_cast<VkDeviceSize>(pixelData.size());
        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        VkBufferCreateInfo stagingCreateInfo{};
        stagingCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingCreateInfo.size = textureBytes;
        stagingCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        result = vkCreateBuffer(m_device, &stagingCreateInfo, nullptr, &stagingBuffer);
        if (result != VK_SUCCESS) {
            logVkFailure("vkCreateBuffer(waterNormalStaging)", result);
            return false;
        }

        VkMemoryRequirements stagingMemReq{};
        vkGetBufferMemoryRequirements(m_device, stagingBuffer, &stagingMemReq);
        const uint32_t stagingMemType = findMemoryTypeIndex(
            m_physicalDevice,
            stagingMemReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (stagingMemType == std::numeric_limits<uint32_t>::max()) {
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            VOX_LOGW("render") << "no staging memory type for water normal texture";
            return false;
        }

        VkMemoryAllocateInfo stagingAllocInfo{};
        stagingAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        stagingAllocInfo.allocationSize = stagingMemReq.size;
        stagingAllocInfo.memoryTypeIndex = stagingMemType;
        result = vkAllocateMemory(m_device, &stagingAllocInfo, nullptr, &stagingMemory);
        if (result == VK_SUCCESS) {
            result = vkBindBufferMemory(m_device, stagingBuffer, stagingMemory, 0);
        }
        if (result != VK_SUCCESS) {
            logVkFailure("vkAllocateMemory/BindBuffer(waterNormalStaging)", result);
            if (stagingMemory != VK_NULL_HANDLE) {
                vkFreeMemory(m_device, stagingMemory, nullptr);
            }
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }

        void* mapped = nullptr;
        result = vkMapMemory(m_device, stagingMemory, 0, textureBytes, 0, &mapped);
        if (result != VK_SUCCESS || mapped == nullptr) {
            logVkFailure("vkMapMemory(waterNormalStaging)", result);
            vkFreeMemory(m_device, stagingMemory, nullptr);
            vkDestroyBuffer(m_device, stagingBuffer, nullptr);
            return false;
        }
        std::memcpy(mapped, pixelData.data(), pixelData.size());
        vkUnmapMemory(m_device, stagingMemory);

        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo commandPoolCreateInfo{};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        commandPoolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
        result = vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &commandPool);
        if (result == VK_SUCCESS) {
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo cbAllocInfo{};
            cbAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cbAllocInfo.commandPool = commandPool;
            cbAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbAllocInfo.commandBufferCount = 1;
            result = vkAllocateCommandBuffers(m_device, &cbAllocInfo, &commandBuffer);
            if (result == VK_SUCCESS) {
                VkCommandBufferBeginInfo beginInfo{};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
                if (result == VK_SUCCESS) {
                    VkImageMemoryBarrier barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    barrier.image = m_waterNormalTextureImage;
                    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    barrier.srcAccessMask = 0;
                    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    vkCmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &barrier);

                    VkBufferImageCopy copyRegion{};
                    copyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    copyRegion.imageExtent = {kGeneratedWaterNormalTextureSize, kGeneratedWaterNormalTextureSize, 1u};
                    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, m_waterNormalTextureImage,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

                    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkCmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &barrier);

                    result = vkEndCommandBuffer(commandBuffer);
                    if (result == VK_SUCCESS) {
                        VkSubmitInfo submitInfo{};
                        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                        submitInfo.commandBufferCount = 1;
                        submitInfo.pCommandBuffers = &commandBuffer;
                        result = vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
                        if (result == VK_SUCCESS) {
                            result = vkQueueWaitIdle(m_graphicsQueue);
                        }
                    }
                }
            }
            vkDestroyCommandPool(m_device, commandPool, nullptr);
        }
        vkFreeMemory(m_device, stagingMemory, nullptr);
        vkDestroyBuffer(m_device, stagingBuffer, nullptr);
        if (result != VK_SUCCESS) {
            logVkFailure("waterNormal staging upload", result);
            return false;
        }
    }

    VkImageViewCreateInfo viewCreateInfo{};
    viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewCreateInfo.image = m_waterNormalTextureImage;
    viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewCreateInfo.subresourceRange.baseMipLevel = 0;
    viewCreateInfo.subresourceRange.levelCount = 1;
    viewCreateInfo.subresourceRange.baseArrayLayer = 0;
    viewCreateInfo.subresourceRange.layerCount = 1;
    result = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_waterNormalTextureImageView);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateImageView(waterNormal)", result);
        return false;
    }
    setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, vkHandleToUint64(m_waterNormalTextureImageView), "water.normal.view");

    VkSamplerCreateInfo samplerCreateInfo{};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.maxAnisotropy = 1.0f;
    result = vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_waterNormalTextureSampler);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateSampler(waterNormal)", result);
        return false;
    }
    setObjectName(VK_OBJECT_TYPE_SAMPLER, vkHandleToUint64(m_waterNormalTextureSampler), "water.normal.sampler");
    VOX_LOGI("render") << "generated subtle water normal texture ready: "
                       << kGeneratedWaterNormalTextureSize << "x" << kGeneratedWaterNormalTextureSize;
    return true;
}


bool RendererBackend::createDiffuseTextureResources() {
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

    constexpr uint32_t kTileSize = 32;
    constexpr uint32_t kTextureTilesX = 16;
    constexpr uint32_t kTextureTilesY = 1;
    constexpr uint32_t kTextureWidth = kTileSize * kTextureTilesX;
    constexpr uint32_t kTextureHeight = kTileSize * kTextureTilesY;
    constexpr VkFormat kCompressedTextureFormat = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    constexpr VkFormat kTextureFormat = VK_FORMAT_R8G8B8A8_UNORM;
    uint32_t diffuseMipLevels = 1u;
    for (uint32_t tileExtent = kTileSize; tileExtent > 1u; tileExtent >>= 1u) {
        ++diffuseMipLevels;
    }
    constexpr VkDeviceSize kTextureBytes = kTextureWidth * kTextureHeight * 4;
    const std::filesystem::path faithfulAtlasPath =
        resolveRendererAssetPath("assets/textures/faithful_block_atlas.dds");

    if (supportsBc1DiffuseAtlas(m_physicalDevice)) {
        DdsCompressedImage ddsImage{};
        if (loadCompressedDdsFile(
                faithfulAtlasPath,
                makeFourCc('D', 'X', 'T', '1'),
                kCompressedTextureFormat,
                8u,
                ddsImage
            ) &&
            ddsImage.width == kTextureWidth &&
            ddsImage.height == kTextureHeight) {
            diffuseMipLevels = ddsImage.mipLevels;

            VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
            VkBufferCreateInfo stagingCreateInfo{};
            stagingCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            stagingCreateInfo.size = static_cast<VkDeviceSize>(ddsImage.pixelData.size());
            stagingCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            stagingCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkResult result = vkCreateBuffer(m_device, &stagingCreateInfo, nullptr, &stagingBuffer);
            if (result != VK_SUCCESS) {
                logVkFailure("vkCreateBuffer(diffuseDdsStaging)", result);
                return false;
            }
            setObjectName(VK_OBJECT_TYPE_BUFFER, vkHandleToUint64(stagingBuffer), "diffuse.dds.staging.buffer");

            VkMemoryRequirements stagingMemReq{};
            vkGetBufferMemoryRequirements(m_device, stagingBuffer, &stagingMemReq);
            uint32_t memoryTypeIndex = findMemoryTypeIndex(
                m_physicalDevice,
                stagingMemReq.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            );
            if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
                VOX_LOGI("render") << "no staging memory type for Faithful BC1 atlas\n";
                vkDestroyBuffer(m_device, stagingBuffer, nullptr);
                return false;
            }

            VkMemoryAllocateInfo stagingAllocInfo{};
            stagingAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            stagingAllocInfo.allocationSize = stagingMemReq.size;
            stagingAllocInfo.memoryTypeIndex = memoryTypeIndex;
            result = vkAllocateMemory(m_device, &stagingAllocInfo, nullptr, &stagingMemory);
            if (result != VK_SUCCESS) {
                logVkFailure("vkAllocateMemory(diffuseDdsStaging)", result);
                vkDestroyBuffer(m_device, stagingBuffer, nullptr);
                return false;
            }
            result = vkBindBufferMemory(m_device, stagingBuffer, stagingMemory, 0);
            if (result != VK_SUCCESS) {
                logVkFailure("vkBindBufferMemory(diffuseDdsStaging)", result);
                vkFreeMemory(m_device, stagingMemory, nullptr);
                vkDestroyBuffer(m_device, stagingBuffer, nullptr);
                return false;
            }

            void* mapped = nullptr;
            result = vkMapMemory(
                m_device,
                stagingMemory,
                0,
                static_cast<VkDeviceSize>(ddsImage.pixelData.size()),
                0,
                &mapped
            );
            if (result != VK_SUCCESS || mapped == nullptr) {
                logVkFailure("vkMapMemory(diffuseDdsStaging)", result);
                vkFreeMemory(m_device, stagingMemory, nullptr);
                vkDestroyBuffer(m_device, stagingBuffer, nullptr);
                return false;
            }
            std::memcpy(mapped, ddsImage.pixelData.data(), ddsImage.pixelData.size());
            vkUnmapMemory(m_device, stagingMemory);

            VkImageCreateInfo imageCreateInfo{};
            imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
            imageCreateInfo.format = kCompressedTextureFormat;
            imageCreateInfo.extent = {kTextureWidth, kTextureHeight, 1};
            imageCreateInfo.mipLevels = diffuseMipLevels;
            imageCreateInfo.arrayLayers = 1;
            imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
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
                    logVkFailure("vmaCreateImage(diffuseDdsTexture)", result);
                    vkFreeMemory(m_device, stagingMemory, nullptr);
                    vkDestroyBuffer(m_device, stagingBuffer, nullptr);
                    return false;
                }
            } else {
                result = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_diffuseTextureImage);
                if (result != VK_SUCCESS) {
                    logVkFailure("vkCreateImage(diffuseDdsTexture)", result);
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
                    VOX_LOGI("render") << "no device-local memory for Faithful BC1 atlas\n";
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
                    logVkFailure("vkAllocateMemory(diffuseDdsTexture)", result);
                    vkDestroyImage(m_device, m_diffuseTextureImage, nullptr);
                    m_diffuseTextureImage = VK_NULL_HANDLE;
                    vkFreeMemory(m_device, stagingMemory, nullptr);
                    vkDestroyBuffer(m_device, stagingBuffer, nullptr);
                    return false;
                }
                result = vkBindImageMemory(m_device, m_diffuseTextureImage, m_diffuseTextureMemory, 0);
                if (result != VK_SUCCESS) {
                    logVkFailure("vkBindImageMemory(diffuseDdsTexture)", result);
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
                logVkFailure("vkCreateCommandPool(diffuseDdsUpload)", result);
                destroyDiffuseTextureResources();
                vkFreeMemory(m_device, stagingMemory, nullptr);
                vkDestroyBuffer(m_device, stagingBuffer, nullptr);
                return false;
            }
            setObjectName(VK_OBJECT_TYPE_COMMAND_POOL, vkHandleToUint64(commandPool), "diffuse.dds.upload.commandPool");

            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo cmdAllocInfo{};
            cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdAllocInfo.commandPool = commandPool;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAllocInfo.commandBufferCount = 1;
            result = vkAllocateCommandBuffers(m_device, &cmdAllocInfo, &commandBuffer);
            if (result != VK_SUCCESS) {
                logVkFailure("vkAllocateCommandBuffers(diffuseDdsUpload)", result);
                vkDestroyCommandPool(m_device, commandPool, nullptr);
                destroyDiffuseTextureResources();
                vkFreeMemory(m_device, stagingMemory, nullptr);
                vkDestroyBuffer(m_device, stagingBuffer, nullptr);
                return false;
            }

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
            if (result != VK_SUCCESS) {
                logVkFailure("vkBeginCommandBuffer(diffuseDdsUpload)", result);
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

            std::vector<VkBufferImageCopy> copyRegions;
            copyRegions.reserve(ddsImage.mipInfos.size());
            for (std::uint32_t mipIndex = 0; mipIndex < ddsImage.mipInfos.size(); ++mipIndex) {
                const DdsMipInfo& mipInfo = ddsImage.mipInfos[mipIndex];
                VkBufferImageCopy copyRegion{};
                copyRegion.bufferOffset = mipInfo.offset;
                copyRegion.bufferRowLength = 0;
                copyRegion.bufferImageHeight = 0;
                copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copyRegion.imageSubresource.mipLevel = mipIndex;
                copyRegion.imageSubresource.baseArrayLayer = 0;
                copyRegion.imageSubresource.layerCount = 1;
                copyRegion.imageOffset = {0, 0, 0};
                copyRegion.imageExtent = {mipInfo.width, mipInfo.height, 1};
                copyRegions.push_back(copyRegion);
            }
            vkCmdCopyBufferToImage(
                commandBuffer,
                stagingBuffer,
                m_diffuseTextureImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                static_cast<std::uint32_t>(copyRegions.size()),
                copyRegions.data()
            );

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
                0u,
                diffuseMipLevels
            );

            result = vkEndCommandBuffer(commandBuffer);
            if (result != VK_SUCCESS) {
                logVkFailure("vkEndCommandBuffer(diffuseDdsUpload)", result);
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
                logVkFailure("vkQueueSubmit(diffuseDdsUpload)", result);
                vkDestroyCommandPool(m_device, commandPool, nullptr);
                destroyDiffuseTextureResources();
                vkFreeMemory(m_device, stagingMemory, nullptr);
                vkDestroyBuffer(m_device, stagingBuffer, nullptr);
                return false;
            }
            result = vkQueueWaitIdle(m_graphicsQueue);
            if (result != VK_SUCCESS) {
                logVkFailure("vkQueueWaitIdle(diffuseDdsUpload)", result);
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
            viewCreateInfo.format = kCompressedTextureFormat;
            viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewCreateInfo.subresourceRange.baseMipLevel = 0;
            viewCreateInfo.subresourceRange.levelCount = diffuseMipLevels;
            viewCreateInfo.subresourceRange.baseArrayLayer = 0;
            viewCreateInfo.subresourceRange.layerCount = 1;
            result = vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_diffuseTextureImageView);
            if (result != VK_SUCCESS) {
                logVkFailure("vkCreateImageView(diffuseDdsTexture)", result);
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
                logVkFailure("vkCreateSampler(diffuseDdsTexture)", result);
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
                logVkFailure("vkCreateSampler(diffuseDdsPlant)", result);
                destroyDiffuseTextureResources();
                return false;
            }
            setObjectName(
                VK_OBJECT_TYPE_SAMPLER,
                vkHandleToUint64(m_diffuseTexturePlantSampler),
                "diffuse.albedo.plantSampler"
            );

            const std::filesystem::path faithfulPlantAtlasPath =
                resolveRendererAssetPath("assets/textures/faithful_plant_atlas.dds");
            if (supportsBc3PlantAtlas(m_physicalDevice)) {
                DdsCompressedImage plantDdsImage{};
                if (loadCompressedDdsFile(
                        faithfulPlantAtlasPath,
                        makeFourCc('D', 'X', 'T', '5'),
                        VK_FORMAT_BC3_UNORM_BLOCK,
                        16u,
                        plantDdsImage
                    ) &&
                    plantDdsImage.width == 256u &&
                    plantDdsImage.height == 32u) {
                    VkBuffer plantStagingBuffer = VK_NULL_HANDLE;
                    VkDeviceMemory plantStagingMemory = VK_NULL_HANDLE;
                    VkBufferCreateInfo plantStagingCreateInfo{};
                    plantStagingCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                    plantStagingCreateInfo.size = static_cast<VkDeviceSize>(plantDdsImage.pixelData.size());
                    plantStagingCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                    plantStagingCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                    result = vkCreateBuffer(m_device, &plantStagingCreateInfo, nullptr, &plantStagingBuffer);
                    if (result == VK_SUCCESS) {
                        VkMemoryRequirements plantStagingMemReq{};
                        vkGetBufferMemoryRequirements(m_device, plantStagingBuffer, &plantStagingMemReq);
                        uint32_t plantMemoryTypeIndex = findMemoryTypeIndex(
                            m_physicalDevice,
                            plantStagingMemReq.memoryTypeBits,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                        );
                        if (plantMemoryTypeIndex != std::numeric_limits<uint32_t>::max()) {
                            VkMemoryAllocateInfo plantStagingAllocInfo{};
                            plantStagingAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                            plantStagingAllocInfo.allocationSize = plantStagingMemReq.size;
                            plantStagingAllocInfo.memoryTypeIndex = plantMemoryTypeIndex;
                            result = vkAllocateMemory(m_device, &plantStagingAllocInfo, nullptr, &plantStagingMemory);
                            if (result == VK_SUCCESS &&
                                vkBindBufferMemory(m_device, plantStagingBuffer, plantStagingMemory, 0) == VK_SUCCESS) {
                                void* plantMapped = nullptr;
                                result = vkMapMemory(
                                    m_device,
                                    plantStagingMemory,
                                    0,
                                    static_cast<VkDeviceSize>(plantDdsImage.pixelData.size()),
                                    0,
                                    &plantMapped
                                );
                                if (result == VK_SUCCESS && plantMapped != nullptr) {
                                    std::memcpy(plantMapped, plantDdsImage.pixelData.data(), plantDdsImage.pixelData.size());
                                    vkUnmapMemory(m_device, plantStagingMemory);

                                    VkImageCreateInfo plantImageCreateInfo{};
                                    plantImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                                    plantImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
                                    plantImageCreateInfo.format = VK_FORMAT_BC3_UNORM_BLOCK;
                                    plantImageCreateInfo.extent = {plantDdsImage.width, plantDdsImage.height, 1};
                                    plantImageCreateInfo.mipLevels = plantDdsImage.mipLevels;
                                    plantImageCreateInfo.arrayLayers = 1;
                                    plantImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
                                    plantImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
                                    plantImageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                                    plantImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                                    plantImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                                    if (m_vmaAllocator != VK_NULL_HANDLE) {
                                        VmaAllocationCreateInfo plantAllocationCreateInfo{};
                                        plantAllocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                                        plantAllocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
                                        result = vmaCreateImage(
                                            m_vmaAllocator,
                                            &plantImageCreateInfo,
                                            &plantAllocationCreateInfo,
                                            &m_plantDiffuseTextureImage,
                                            &m_plantDiffuseTextureAllocation,
                                            nullptr
                                        );
                                    } else {
                                        result = vkCreateImage(m_device, &plantImageCreateInfo, nullptr, &m_plantDiffuseTextureImage);
                                        if (result == VK_SUCCESS) {
                                            VkMemoryRequirements plantImageMemReq{};
                                            vkGetImageMemoryRequirements(m_device, m_plantDiffuseTextureImage, &plantImageMemReq);
                                            plantMemoryTypeIndex = findMemoryTypeIndex(
                                                m_physicalDevice,
                                                plantImageMemReq.memoryTypeBits,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                            );
                                            if (plantMemoryTypeIndex != std::numeric_limits<uint32_t>::max()) {
                                                VkMemoryAllocateInfo plantImageAllocInfo{};
                                                plantImageAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                                                plantImageAllocInfo.allocationSize = plantImageMemReq.size;
                                                plantImageAllocInfo.memoryTypeIndex = plantMemoryTypeIndex;
                                                result = vkAllocateMemory(
                                                    m_device,
                                                    &plantImageAllocInfo,
                                                    nullptr,
                                                    &m_plantDiffuseTextureMemory
                                                );
                                                if (result == VK_SUCCESS) {
                                                    result = vkBindImageMemory(
                                                        m_device,
                                                        m_plantDiffuseTextureImage,
                                                        m_plantDiffuseTextureMemory,
                                                        0
                                                    );
                                                }
                                            } else {
                                                result = VK_ERROR_FORMAT_NOT_SUPPORTED;
                                            }
                                        }
                                    }

                                    if (result == VK_SUCCESS) {
                                        VkCommandPool plantCommandPool = VK_NULL_HANDLE;
                                        VkCommandPoolCreateInfo plantPoolCreateInfo{};
                                        plantPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                                        plantPoolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
                                        plantPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
                                        result = vkCreateCommandPool(m_device, &plantPoolCreateInfo, nullptr, &plantCommandPool);
                                        if (result == VK_SUCCESS) {
                                            VkCommandBuffer plantCommandBuffer = VK_NULL_HANDLE;
                                            VkCommandBufferAllocateInfo plantCmdAllocInfo{};
                                            plantCmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                                            plantCmdAllocInfo.commandPool = plantCommandPool;
                                            plantCmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                                            plantCmdAllocInfo.commandBufferCount = 1;
                                            result = vkAllocateCommandBuffers(m_device, &plantCmdAllocInfo, &plantCommandBuffer);
                                            if (result == VK_SUCCESS) {
                                                VkCommandBufferBeginInfo plantBeginInfo{};
                                                plantBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                                                plantBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                                                result = vkBeginCommandBuffer(plantCommandBuffer, &plantBeginInfo);
                                                if (result == VK_SUCCESS) {
                                                    transitionImageLayout(
                                                        plantCommandBuffer,
                                                        m_plantDiffuseTextureImage,
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
                                                        plantDdsImage.mipLevels
                                                    );
                                                    std::vector<VkBufferImageCopy> plantCopyRegions;
                                                    plantCopyRegions.reserve(plantDdsImage.mipInfos.size());
                                                    for (std::uint32_t mipIndex = 0; mipIndex < plantDdsImage.mipInfos.size(); ++mipIndex) {
                                                        const DdsMipInfo& mipInfo = plantDdsImage.mipInfos[mipIndex];
                                                        VkBufferImageCopy copyRegion{};
                                                        copyRegion.bufferOffset = mipInfo.offset;
                                                        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                                                        copyRegion.imageSubresource.mipLevel = mipIndex;
                                                        copyRegion.imageSubresource.baseArrayLayer = 0;
                                                        copyRegion.imageSubresource.layerCount = 1;
                                                        copyRegion.imageExtent = {mipInfo.width, mipInfo.height, 1};
                                                        plantCopyRegions.push_back(copyRegion);
                                                    }
                                                    vkCmdCopyBufferToImage(
                                                        plantCommandBuffer,
                                                        plantStagingBuffer,
                                                        m_plantDiffuseTextureImage,
                                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                        static_cast<std::uint32_t>(plantCopyRegions.size()),
                                                        plantCopyRegions.data()
                                                    );
                                                    transitionImageLayout(
                                                        plantCommandBuffer,
                                                        m_plantDiffuseTextureImage,
                                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                        VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                                        VK_IMAGE_ASPECT_COLOR_BIT,
                                                        0u,
                                                        1u,
                                                        0u,
                                                        plantDdsImage.mipLevels
                                                    );
                                                    result = vkEndCommandBuffer(plantCommandBuffer);
                                                    if (result == VK_SUCCESS) {
                                                        VkSubmitInfo plantSubmitInfo{};
                                                        plantSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                                                        plantSubmitInfo.commandBufferCount = 1;
                                                        plantSubmitInfo.pCommandBuffers = &plantCommandBuffer;
                                                        result = vkQueueSubmit(m_graphicsQueue, 1, &plantSubmitInfo, VK_NULL_HANDLE);
                                                        if (result == VK_SUCCESS) {
                                                            result = vkQueueWaitIdle(m_graphicsQueue);
                                                        }
                                                    }
                                                }
                                            }
                                            vkDestroyCommandPool(m_device, plantCommandPool, nullptr);
                                        }
                                    }

                                    if (result == VK_SUCCESS) {
                                        VkImageViewCreateInfo plantViewCreateInfo{};
                                        plantViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                                        plantViewCreateInfo.image = m_plantDiffuseTextureImage;
                                        plantViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                                        plantViewCreateInfo.format = VK_FORMAT_BC3_UNORM_BLOCK;
                                        plantViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                                        plantViewCreateInfo.subresourceRange.baseMipLevel = 0;
                                        plantViewCreateInfo.subresourceRange.levelCount = plantDdsImage.mipLevels;
                                        plantViewCreateInfo.subresourceRange.baseArrayLayer = 0;
                                        plantViewCreateInfo.subresourceRange.layerCount = 1;
                                        result = vkCreateImageView(
                                            m_device,
                                            &plantViewCreateInfo,
                                            nullptr,
                                            &m_plantDiffuseTextureImageView
                                        );
                                    }
                                }
                            }
                        }
                    }

                    if (plantStagingMemory != VK_NULL_HANDLE) {
                        vkFreeMemory(m_device, plantStagingMemory, nullptr);
                    }
                    if (plantStagingBuffer != VK_NULL_HANDLE) {
                        vkDestroyBuffer(m_device, plantStagingBuffer, nullptr);
                    }
                }
            }
            if (m_plantDiffuseTextureImageView == VK_NULL_HANDLE) {
                VOX_LOGI("render") << "Faithful plant atlas unavailable or invalid at "
                                   << faithfulPlantAtlasPath.string()
                                   << "; using block atlas for plant sampling\n";
            } else {
                VOX_LOGI("render") << "Faithful plant atlas loaded with alpha-capable compression from "
                                   << faithfulPlantAtlasPath.string() << "\n";
            }

            VOX_LOGI("render") << "Faithful BC1 diffuse atlas loaded from " << faithfulAtlasPath.string()
                               << ": mips=" << diffuseMipLevels
                               << ", tileSize=" << kTileSize << ", atlas=" << kTextureWidth << "x" << kTextureHeight << "\n";
            return true;
        }

        VOX_LOGI("render") << "Faithful BC1 diffuse atlas unavailable or invalid at "
                           << faithfulAtlasPath.string()
                           << "; falling back to procedural atlas\n";
    } else {
        VOX_LOGI("render") << "BC1 atlas sampling unsupported; falling back to procedural atlas\n";
    }

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
                // Grass top.
                const int green = 118 + static_cast<int>(noiseA % 32u) - 16;
                r = static_cast<std::uint8_t>(std::clamp(52 + static_cast<int>(noiseB % 18u) - 9, 34, 74));
                g = static_cast<std::uint8_t>(std::clamp(green, 82, 154));
                b = static_cast<std::uint8_t>(std::clamp(44 + static_cast<int>(noiseA % 14u) - 7, 26, 64));
            } else if (tileIndex == 3u) {
                // Grass side.
                const bool grassyBand = localY < (kTileSize / 3u);
                if (grassyBand) {
                    const int green = 108 + static_cast<int>(noiseA % 28u) - 14;
                    r = static_cast<std::uint8_t>(std::clamp(58 + static_cast<int>(noiseB % 18u) - 9, 38, 84));
                    g = static_cast<std::uint8_t>(std::clamp(green, 78, 148));
                    b = static_cast<std::uint8_t>(std::clamp(42 + static_cast<int>(noiseA % 14u) - 7, 24, 68));
                } else {
                    const int warm = 96 + static_cast<int>(noiseA % 26u) - 13;
                    const int cool = 70 + static_cast<int>(noiseB % 18u) - 9;
                    r = static_cast<std::uint8_t>(std::clamp(warm + 16, 74, 142));
                    g = static_cast<std::uint8_t>(std::clamp(warm - 6, 50, 106));
                    b = static_cast<std::uint8_t>(std::clamp(cool - 10, 24, 82));
                }
            } else if (tileIndex == 4u) {
                // Wood side.
                const int stripe = ((localX / 3u) + (localY / 5u)) % 3u;
                const int base = (stripe == 0) ? 112 : (stripe == 1 ? 96 : 84);
                const int grain = static_cast<int>(noiseA % 16u) - 8;
                r = static_cast<std::uint8_t>(std::clamp(base + 34 + grain, 78, 168));
                g = static_cast<std::uint8_t>(std::clamp(base + 12 + grain, 56, 136));
                b = static_cast<std::uint8_t>(std::clamp(base - 6 + (grain / 2), 36, 110));
            } else if (tileIndex == 5u) {
                // Wood top.
                const int dx = static_cast<int>(localX) - static_cast<int>(kTileSize / 2u);
                const int dy = static_cast<int>(localY) - static_cast<int>(kTileSize / 2u);
                const float ring = std::sqrt(static_cast<float>((dx * dx) + (dy * dy)));
                const int ringBand = static_cast<int>(ring / 3.4f) % 3;
                const int base = (ringBand == 0) ? 126 : (ringBand == 1 ? 108 : 92);
                const int grain = static_cast<int>(noiseA % 18u) - 9;
                r = static_cast<std::uint8_t>(std::clamp(base + 28 + grain, 84, 176));
                g = static_cast<std::uint8_t>(std::clamp(base + 8 + grain, 60, 142));
                b = static_cast<std::uint8_t>(std::clamp(base - 10 + (grain / 2), 36, 116));
            } else if (tileIndex == 6u) {
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
            } else if (tileIndex >= 7u && tileIndex <= 10u) {
                // Procedural flower sprites (tiles 7..10):
                // 7-8 = poppies (red/orange-red), 9-10 = light wildflowers.
                const int ix = static_cast<int>(localX);
                const int iy = static_cast<int>(localY);
                const int rowFromBottom = static_cast<int>(kTileSize - 1u - localY);
                const uint32_t flowerVariant = (tileIndex - 7u) & 3u;
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
            } else if (tileIndex == 11u) {
                // Leaves.
                const int ix = static_cast<int>(localX);
                const int iy = static_cast<int>(localY);
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
                leafWeight = std::max(leafWeight, circleWeight(9, 11, 9));
                leafWeight = std::max(leafWeight, circleWeight(16, 8, 10));
                leafWeight = std::max(leafWeight, circleWeight(23, 14, 8));
                leafWeight = std::max(leafWeight, circleWeight(15, 22, 9));
                const float edgeNoise = static_cast<float>(noiseA % 100u) / 100.0f;
                if (leafWeight < (0.16f + (edgeNoise * 0.20f))) {
                    writePixel(x, y, 0u, 0u, 0u, 0u);
                    continue;
                }
                r = static_cast<std::uint8_t>(std::clamp(52 + static_cast<int>(noiseB % 30u) - 12, 28, 94));
                g = static_cast<std::uint8_t>(std::clamp(124 + static_cast<int>(noiseA % 54u) - 22, 86, 188));
                b = static_cast<std::uint8_t>(std::clamp(44 + static_cast<int>(noiseB % 18u) - 7, 22, 76));
                const int alphaBase = static_cast<int>(116.0f + (leafWeight * 138.0f));
                const std::uint8_t alpha = static_cast<std::uint8_t>(
                    std::clamp(alphaBase + static_cast<int>(noiseB % 24u) - 10, 118, 250)
                );
                writePixel(x, y, r, g, b, alpha);
                continue;
            } else if (tileIndex == 12u) {
                // Red concrete.
                r = static_cast<std::uint8_t>(std::clamp(182 + static_cast<int>(noiseA % 14u) - 7, 160, 204));
                g = static_cast<std::uint8_t>(std::clamp(48 + static_cast<int>(noiseB % 10u) - 5, 34, 64));
                b = static_cast<std::uint8_t>(std::clamp(42 + static_cast<int>(noiseA % 10u) - 5, 28, 58));
            } else {
                writePixel(x, y, 0u, 0u, 0u, 0u);
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


bool RendererBackend::createShadowResources() {
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


bool RendererBackend::createVoxelGiResources() {
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
        VOX_LOGE("render") << "voxel GI occupancy format unsupported (requires sampled+storage 3D image)\n";
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
        occupancyImageCreateInfo.usage =
            VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_STORAGE_BIT;
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
    constexpr const char* kVoxelGiOccupancyShaderPath = "../src/render/shaders/voxel_gi_occupancy.comp.slang.spv";
    constexpr const char* kVoxelGiSurfaceShaderPath = "../src/render/shaders/voxel_gi_surface.comp.slang.spv";
    constexpr const char* kVoxelGiSurfaceRtShaderPath = "../src/render/shaders/voxel_gi_surface_rt.comp.slang.spv";
    constexpr const char* kVoxelGiRestirCandidateShaderPath = "../src/render/shaders/voxel_gi_restir_candidate.comp.slang.spv";
    constexpr const char* kVoxelGiRestirTemporalShaderPath = "../src/render/shaders/voxel_gi_restir_temporal.comp.slang.spv";
    constexpr const char* kVoxelGiRestirSpatialShaderPath = "../src/render/shaders/voxel_gi_restir_spatial.comp.slang.spv";
    constexpr const char* kVoxelGiRestirResolveShaderPath = "../src/render/shaders/voxel_gi_restir_resolve.comp.slang.spv";
    constexpr const char* kVoxelGiInjectShaderPath = "../src/render/shaders/voxel_gi_inject.comp.slang.spv";
    constexpr const char* kVoxelGiPropagateShaderPath = "../src/render/shaders/voxel_gi_propagate.comp.slang.spv";
    const bool hasSkyExposureShader = readBinaryFile(kVoxelGiSkyExposureShaderPath).has_value();
    const bool hasOccupancyShader = readBinaryFile(kVoxelGiOccupancyShaderPath).has_value();
    const bool hasSurfaceShader = readBinaryFile(kVoxelGiSurfaceShaderPath).has_value();
    const bool hasInjectShader = readBinaryFile(kVoxelGiInjectShaderPath).has_value();
    const bool hasPropagateShader = readBinaryFile(kVoxelGiPropagateShaderPath).has_value();
    const bool hasRtSurfaceShaderVariant =
        m_rayTracingRuntimeEnabled &&
        readBinaryFile(kVoxelGiSurfaceRtShaderPath).has_value();
    const bool hasRestirShaderSet =
        m_rayTracingRuntimeEnabled &&
        readBinaryFile(kVoxelGiRestirCandidateShaderPath).has_value() &&
        readBinaryFile(kVoxelGiRestirTemporalShaderPath).has_value() &&
        readBinaryFile(kVoxelGiRestirSpatialShaderPath).has_value() &&
        readBinaryFile(kVoxelGiRestirResolveShaderPath).has_value();
    m_voxelGiRtSurfaceReady = false;
    m_voxelGiRestirReady = false;
    if (!hasSkyExposureShader || !hasOccupancyShader || !hasSurfaceShader || !hasInjectShader || !hasPropagateShader) {
        VOX_LOGI("render")
            << "voxel GI compute shaders not found; keeping static volume fallback (expected: "
            << kVoxelGiSkyExposureShaderPath << ", " << kVoxelGiOccupancyShaderPath << ", "
            << kVoxelGiSurfaceShaderPath << ", " << kVoxelGiInjectShaderPath << ", "
            << kVoxelGiPropagateShaderPath << ")\n";
        VOX_LOGI("render") << "voxel GI runtime: compute=no, rtSurfaceVariant="
                           << (hasRtSurfaceShaderVariant ? "yes" : "no")
                           << ", rtSurfaceReady=no, restirSurfaceReady=no, rtSurfaceTracingRequested="
                           << (m_voxelGiDebugSettings.surfaceMode != VoxelGiSurfaceMode::Legacy ? "yes" : "no") << "\n";
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

        VkDescriptorSetLayoutBinding occupancyWriteBinding{};
        occupancyWriteBinding.binding = 13;
        occupancyWriteBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        occupancyWriteBinding.descriptorCount = 1;
        occupancyWriteBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding chunkMetaBinding{};
        chunkMetaBinding.binding = 14;
        chunkMetaBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        chunkMetaBinding.descriptorCount = 1;
        chunkMetaBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding chunkVoxelBinding{};
        chunkVoxelBinding.binding = 15;
        chunkVoxelBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        chunkVoxelBinding.descriptorCount = 1;
        chunkVoxelBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        std::vector<VkDescriptorSetLayoutBinding> bindings = {
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
            skyExposureBinding,
            occupancyWriteBinding,
            chunkMetaBinding,
            chunkVoxelBinding
        };
        if (m_rayTracingRuntimeEnabled) {
            VkDescriptorSetLayoutBinding rtSurfaceSceneBinding{};
            rtSurfaceSceneBinding.binding = 16;
            rtSurfaceSceneBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            rtSurfaceSceneBinding.descriptorCount = 1;
            rtSurfaceSceneBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(rtSurfaceSceneBinding);

            VkDescriptorSetLayoutBinding restirCurrentBinding{};
            restirCurrentBinding.binding = 17;
            restirCurrentBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            restirCurrentBinding.descriptorCount = 1;
            restirCurrentBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(restirCurrentBinding);

            VkDescriptorSetLayoutBinding restirPreviousBinding{};
            restirPreviousBinding.binding = 18;
            restirPreviousBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            restirPreviousBinding.descriptorCount = 1;
            restirPreviousBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(restirPreviousBinding);

            VkDescriptorSetLayoutBinding restirScratchBinding{};
            restirScratchBinding.binding = 19;
            restirScratchBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            restirScratchBinding.descriptorCount = 1;
            restirScratchBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(restirScratchBinding);
        }

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
        std::vector<VkDescriptorPoolSize> poolSizes = {
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2 * kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10 * kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 * kMaxFramesInFlight}
        };
        if (m_rayTracingRuntimeEnabled) {
            poolSizes.push_back(VkDescriptorPoolSize{
                VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                kMaxFramesInFlight
            });
        }
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
    m_voxelGiDescriptorWriteKeyValid.fill(false);

    std::array<VkShaderModule, 10> shaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& skyExposureShaderModule = shaderModules[0];
    VkShaderModule& occupancyShaderModule = shaderModules[1];
    VkShaderModule& surfaceShaderModule = shaderModules[2];
    VkShaderModule& surfaceRtShaderModule = shaderModules[3];
    VkShaderModule& injectShaderModule = shaderModules[4];
    VkShaderModule& propagateShaderModule = shaderModules[5];
    VkShaderModule& restirCandidateShaderModule = shaderModules[6];
    VkShaderModule& restirTemporalShaderModule = shaderModules[7];
    VkShaderModule& restirSpatialShaderModule = shaderModules[8];
    VkShaderModule& restirResolveShaderModule = shaderModules[9];
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
            kVoxelGiOccupancyShaderPath,
            "voxel_gi_occupancy.comp",
            occupancyShaderModule
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
    bool createdRtSurfacePipeline = false;
    if (hasRtSurfaceShaderVariant) {
        if (!createShaderModuleFromFile(
                m_device,
                kVoxelGiSurfaceRtShaderPath,
                "voxel_gi_surface_rt.comp",
                surfaceRtShaderModule
            )) {
            destroyShaderModules(m_device, shaderModules);
            destroyVoxelGiResources();
            return false;
        }
    }
    if (hasRestirShaderSet) {
        if (!createShaderModuleFromFile(
                m_device,
                kVoxelGiRestirCandidateShaderPath,
                "voxel_gi_restir_candidate.comp",
                restirCandidateShaderModule
            ) ||
            !createShaderModuleFromFile(
                m_device,
                kVoxelGiRestirTemporalShaderPath,
                "voxel_gi_restir_temporal.comp",
                restirTemporalShaderModule
            ) ||
            !createShaderModuleFromFile(
                m_device,
                kVoxelGiRestirSpatialShaderPath,
                "voxel_gi_restir_spatial.comp",
                restirSpatialShaderModule
            ) ||
            !createShaderModuleFromFile(
                m_device,
                kVoxelGiRestirResolveShaderPath,
                "voxel_gi_restir_resolve.comp",
                restirResolveShaderModule
            )) {
            destroyShaderModules(m_device, shaderModules);
            destroyVoxelGiResources();
            return false;
        }
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
            occupancyShaderModule,
            m_voxelGiOccupancyPipeline,
            "vkCreateComputePipelines(voxelGiOccupancy)",
            "pipeline.voxelGi.occupancy"
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
    if (hasRtSurfaceShaderVariant) {
        if (!createComputePipeline(
                m_voxelGiPipelineLayout,
                surfaceRtShaderModule,
                m_voxelGiSurfacePipelineRt,
                "vkCreateComputePipelines(voxelGiSurfaceRt)",
                "pipeline.voxelGi.surface.rt"
            )) {
            destroyShaderModules(m_device, shaderModules);
            destroyVoxelGiResources();
            return false;
        }
        createdRtSurfacePipeline = true;
    }
    bool createdRestirPipelines = false;
    if (hasRestirShaderSet) {
        if (!createComputePipeline(
                m_voxelGiPipelineLayout,
                restirCandidateShaderModule,
                m_voxelGiRestirCandidatePipeline,
                "vkCreateComputePipelines(voxelGiRestirCandidate)",
                "pipeline.voxelGi.restirCandidate"
            ) ||
            !createComputePipeline(
                m_voxelGiPipelineLayout,
                restirTemporalShaderModule,
                m_voxelGiRestirTemporalPipeline,
                "vkCreateComputePipelines(voxelGiRestirTemporal)",
                "pipeline.voxelGi.restirTemporal"
            ) ||
            !createComputePipeline(
                m_voxelGiPipelineLayout,
                restirSpatialShaderModule,
                m_voxelGiRestirSpatialPipeline,
                "vkCreateComputePipelines(voxelGiRestirSpatial)",
                "pipeline.voxelGi.restirSpatial"
            ) ||
            !createComputePipeline(
                m_voxelGiPipelineLayout,
                restirResolveShaderModule,
                m_voxelGiRestirResolvePipeline,
                "vkCreateComputePipelines(voxelGiRestirResolve)",
                "pipeline.voxelGi.restirResolve"
            )) {
            destroyShaderModules(m_device, shaderModules);
            destroyVoxelGiResources();
            return false;
        }
        createdRestirPipelines = true;
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
    m_voxelGiRtSurfaceReady =
        m_rayTracingRuntimeEnabled &&
        createdRtSurfacePipeline &&
        m_voxelGiSurfacePipelineRt != VK_NULL_HANDLE;
    m_voxelGiRestirReady =
        m_rayTracingRuntimeEnabled &&
        createdRestirPipelines &&
        m_voxelGiRestirCandidatePipeline != VK_NULL_HANDLE &&
        m_voxelGiRestirTemporalPipeline != VK_NULL_HANDLE &&
        m_voxelGiRestirSpatialPipeline != VK_NULL_HANDLE &&
        m_voxelGiRestirResolvePipeline != VK_NULL_HANDLE;
    if (m_rayTracingRuntimeEnabled) {
        struct alignas(16) VoxelGiRestirReservoirInit {
            float radiance[4];
            float state[4];
        };
        constexpr std::size_t kReservoirFaceCount = 6u;
        const VkDeviceSize reservoirCount =
            static_cast<VkDeviceSize>(kVoxelGiGridResolution) *
            static_cast<VkDeviceSize>(kVoxelGiGridResolution) *
            static_cast<VkDeviceSize>(kVoxelGiGridResolution) *
            static_cast<VkDeviceSize>(kReservoirFaceCount);
        const VkDeviceSize reservoirBytes =
            reservoirCount * static_cast<VkDeviceSize>(sizeof(VoxelGiRestirReservoirInit));
        const BufferCreateDesc reservoirCreateDesc{
            reservoirBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            0,
            nullptr,
            0,
            nullptr
        };
        m_voxelGiRestirReservoirCurrentBufferHandle = m_bufferAllocator.createBuffer(reservoirCreateDesc);
        m_voxelGiRestirReservoirPreviousBufferHandle = m_bufferAllocator.createBuffer(reservoirCreateDesc);
        m_voxelGiRestirReservoirScratchBufferHandle = m_bufferAllocator.createBuffer(reservoirCreateDesc);
        if (m_voxelGiRestirReservoirCurrentBufferHandle == kInvalidBufferHandle ||
            m_voxelGiRestirReservoirPreviousBufferHandle == kInvalidBufferHandle ||
            m_voxelGiRestirReservoirScratchBufferHandle == kInvalidBufferHandle) {
            destroyVoxelGiResources();
            return false;
        }
    }
    VOX_LOGI("render") << "voxel GI resources ready: "
                       << kVoxelGiGridResolution << "^3, format=" << static_cast<int>(m_voxelGiFormat)
                       << ", occupancyFormat=" << static_cast<int>(m_voxelGiOccupancyFormat)
                       << ", compute=enabled\n";
    VOX_LOGI("render") << "voxel GI runtime: compute=yes, rtSurfaceVariant="
                       << (hasRtSurfaceShaderVariant ? "yes" : "no")
                       << ", rtSurfaceReady=" << (m_voxelGiRtSurfaceReady ? "yes" : "no")
                       << ", restirSurfaceReady=" << (m_voxelGiRestirReady ? "yes" : "no")
                       << ", rtSurfaceTracingRequested="
                       << (m_voxelGiDebugSettings.surfaceMode != VoxelGiSurfaceMode::Legacy ? "yes" : "no") << "\n";
    return true;
}



void RendererBackend::destroyEnvironmentResources() {
    destroyDiffuseTextureResources();
}


void RendererBackend::destroyDiffuseTextureResources() {
    if (m_skyCloudIndexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_skyCloudIndexBufferHandle);
        m_skyCloudIndexBufferHandle = kInvalidBufferHandle;
    }
    if (m_skyCloudVertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_skyCloudVertexBufferHandle);
        m_skyCloudVertexBufferHandle = kInvalidBufferHandle;
    }
    m_skyCloudIndexCount = 0;
    if (m_waterNormalTextureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_waterNormalTextureSampler, nullptr);
        m_waterNormalTextureSampler = VK_NULL_HANDLE;
    }
    if (m_waterNormalTextureImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_waterNormalTextureImageView, nullptr);
        m_waterNormalTextureImageView = VK_NULL_HANDLE;
    }
    if (m_waterNormalTextureImage != VK_NULL_HANDLE) {
        if (m_vmaAllocator != VK_NULL_HANDLE && m_waterNormalTextureAllocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_waterNormalTextureImage, m_waterNormalTextureAllocation);
            m_waterNormalTextureAllocation = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, m_waterNormalTextureImage, nullptr);
        }
        m_waterNormalTextureImage = VK_NULL_HANDLE;
    }
    if (m_waterNormalTextureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_waterNormalTextureMemory, nullptr);
        m_waterNormalTextureMemory = VK_NULL_HANDLE;
    }
    m_waterNormalTextureAllocation = VK_NULL_HANDLE;
    if (m_terrainDetailTextureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_terrainDetailTextureSampler, nullptr);
        m_terrainDetailTextureSampler = VK_NULL_HANDLE;
    }
    if (m_terrainDetailTextureImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_terrainDetailTextureImageView, nullptr);
        m_terrainDetailTextureImageView = VK_NULL_HANDLE;
    }
    if (m_terrainDetailTextureImage != VK_NULL_HANDLE) {
        if (m_vmaAllocator != VK_NULL_HANDLE && m_terrainDetailTextureAllocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_terrainDetailTextureImage, m_terrainDetailTextureAllocation);
            m_terrainDetailTextureAllocation = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, m_terrainDetailTextureImage, nullptr);
        }
        m_terrainDetailTextureImage = VK_NULL_HANDLE;
    }
    if (m_terrainDetailTextureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_terrainDetailTextureMemory, nullptr);
        m_terrainDetailTextureMemory = VK_NULL_HANDLE;
    }
    m_terrainDetailTextureAllocation = VK_NULL_HANDLE;
    if (m_morrowindSkyTextureSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_morrowindSkyTextureSampler, nullptr);
        m_morrowindSkyTextureSampler = VK_NULL_HANDLE;
    }
    if (m_morrowindSkyTextureImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_morrowindSkyTextureImageView, nullptr);
        m_morrowindSkyTextureImageView = VK_NULL_HANDLE;
    }
    if (m_morrowindSkyTextureImage != VK_NULL_HANDLE) {
        if (m_vmaAllocator != VK_NULL_HANDLE && m_morrowindSkyTextureAllocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_morrowindSkyTextureImage, m_morrowindSkyTextureAllocation);
            m_morrowindSkyTextureAllocation = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, m_morrowindSkyTextureImage, nullptr);
        }
        m_morrowindSkyTextureImage = VK_NULL_HANDLE;
    }
    if (m_morrowindSkyTextureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_morrowindSkyTextureMemory, nullptr);
        m_morrowindSkyTextureMemory = VK_NULL_HANDLE;
    }
    m_morrowindSkyTextureAllocation = VK_NULL_HANDLE;
    if (m_diffuseTexturePlantSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_diffuseTexturePlantSampler, nullptr);
        m_diffuseTexturePlantSampler = VK_NULL_HANDLE;
    }
    if (m_plantDiffuseTextureImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, m_plantDiffuseTextureImageView, nullptr);
        m_plantDiffuseTextureImageView = VK_NULL_HANDLE;
    }
    if (m_plantDiffuseTextureImage != VK_NULL_HANDLE) {
        if (m_vmaAllocator != VK_NULL_HANDLE && m_plantDiffuseTextureAllocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_vmaAllocator, m_plantDiffuseTextureImage, m_plantDiffuseTextureAllocation);
            m_plantDiffuseTextureAllocation = VK_NULL_HANDLE;
        } else {
            vkDestroyImage(m_device, m_plantDiffuseTextureImage, nullptr);
        }
        m_plantDiffuseTextureImage = VK_NULL_HANDLE;
    }
    if (m_plantDiffuseTextureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_plantDiffuseTextureMemory, nullptr);
        m_plantDiffuseTextureMemory = VK_NULL_HANDLE;
    }
    m_plantDiffuseTextureAllocation = VK_NULL_HANDLE;
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


void RendererBackend::destroyShadowResources() {
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


void RendererBackend::destroyVoxelGiResources() {
    m_pipelineManager.destroyVoxelGiPipelines(m_device);
    m_descriptorManager.destroyVoxelGi(m_device);
    m_voxelGiDescriptorWriteKeyValid.fill(false);
    m_bufferAllocator.destroyBuffer(m_voxelGiRestirReservoirCurrentBufferHandle);
    m_bufferAllocator.destroyBuffer(m_voxelGiRestirReservoirPreviousBufferHandle);
    m_bufferAllocator.destroyBuffer(m_voxelGiRestirReservoirScratchBufferHandle);
    m_voxelGiRestirReservoirCurrentBufferHandle = kInvalidBufferHandle;
    m_voxelGiRestirReservoirPreviousBufferHandle = kInvalidBufferHandle;
    m_voxelGiRestirReservoirScratchBufferHandle = kInvalidBufferHandle;

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
    m_voxelGiRestirReady = false;
    m_voxelGiRestirActiveThisFrame = false;
    m_voxelGiRestirHistoryValid = false;
    m_voxelGiRestirHistoryResetReason = "destroyed";
    m_voxelGiWorldDirty = true;
    ++m_voxelGiWorldVersion;
    m_voxelGiHasPreviousFrameState = false;
    m_voxelGiPreviousBounceStrength = 0.0f;
    m_voxelGiPreviousDiffusionSoftness = 0.0f;
    m_voxelGiPreviousSurfaceMode = VoxelGiSurfaceMode::RestirSurface;
    m_voxelGiOccupancyBuildOrigin = {0.0f, 0.0f, 0.0f};
    m_voxelGiOccupancyFullRebuildCursor = 0;
    m_voxelGiOccupancyFullRebuildInProgress = false;
    m_voxelGiOccupancyFullRebuildNeedsClear = false;
    m_voxelGiDirtyChunkIndices.clear();
    m_voxelGiPreviousRtSurfaceTracingEnabled = false;
    m_voxelGiPreviousRtSurfaceSampleCount = 0.0f;
    m_voxelGiPreviousRtSurfaceBiasScale = 0.0f;
    m_voxelGiPreviousRtSunAngularRadiusDegrees = 0.0f;
    m_voxelGiPreviousRestirCandidateCount = 0.0f;
    m_voxelGiPreviousRestirTemporalReuseEnabled = false;
    m_voxelGiPreviousRestirSpatialReuseEnabled = false;
    m_voxelGiPreviousRestirSpatialRadius = 0.0f;
}



} // namespace odai::render
