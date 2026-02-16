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

bool RendererBackend::createEnvironmentResources() {
    if (!createDiffuseTextureResources()) {
        VOX_LOGE("render") << "diffuse texture creation failed\n";
        return false;
    }
    VOX_LOGI("render") << "environment uses procedural sky + SH irradiance + diffuse albedo texture\n";
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
    constexpr const char* kVoxelGiOccupancyShaderPath = "../src/render/shaders/voxel_gi_occupancy.comp.slang.spv";
    constexpr const char* kVoxelGiSurfaceShaderPath = "../src/render/shaders/voxel_gi_surface.comp.slang.spv";
    constexpr const char* kVoxelGiInjectShaderPath = "../src/render/shaders/voxel_gi_inject.comp.slang.spv";
    constexpr const char* kVoxelGiPropagateShaderPath = "../src/render/shaders/voxel_gi_propagate.comp.slang.spv";
    const bool hasSkyExposureShader = readBinaryFile(kVoxelGiSkyExposureShaderPath).has_value();
    const bool hasOccupancyShader = readBinaryFile(kVoxelGiOccupancyShaderPath).has_value();
    const bool hasSurfaceShader = readBinaryFile(kVoxelGiSurfaceShaderPath).has_value();
    const bool hasInjectShader = readBinaryFile(kVoxelGiInjectShaderPath).has_value();
    const bool hasPropagateShader = readBinaryFile(kVoxelGiPropagateShaderPath).has_value();
    if (!hasSkyExposureShader || !hasOccupancyShader || !hasSurfaceShader || !hasInjectShader || !hasPropagateShader) {
        VOX_LOGI("render")
            << "voxel GI compute shaders not found; keeping static volume fallback (expected: "
            << kVoxelGiSkyExposureShaderPath << ", " << kVoxelGiOccupancyShaderPath << ", "
            << kVoxelGiSurfaceShaderPath << ", " << kVoxelGiInjectShaderPath << ", "
            << kVoxelGiPropagateShaderPath << ")\n";
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

        const std::array<VkDescriptorSetLayoutBinding, 16> bindings = {
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
        const std::array<VkDescriptorPoolSize, 5> poolSizes = {
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2 * kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10 * kMaxFramesInFlight},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kMaxFramesInFlight}
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
    m_voxelGiDescriptorWriteKeyValid.fill(false);

    std::array<VkShaderModule, 5> shaderModules = {
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE
    };
    VkShaderModule& skyExposureShaderModule = shaderModules[0];
    VkShaderModule& occupancyShaderModule = shaderModules[1];
    VkShaderModule& surfaceShaderModule = shaderModules[2];
    VkShaderModule& injectShaderModule = shaderModules[3];
    VkShaderModule& propagateShaderModule = shaderModules[4];
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



void RendererBackend::destroyEnvironmentResources() {
    destroyDiffuseTextureResources();
}


void RendererBackend::destroyDiffuseTextureResources() {
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
    ++m_voxelGiWorldVersion;
    m_voxelGiHasPreviousFrameState = false;
    m_voxelGiPreviousBounceStrength = 0.0f;
    m_voxelGiPreviousDiffusionSoftness = 0.0f;
    m_voxelGiOccupancyBuildOrigin = {0.0f, 0.0f, 0.0f};
    m_voxelGiOccupancyFullRebuildCursor = 0;
    m_voxelGiOccupancyFullRebuildInProgress = false;
    m_voxelGiOccupancyFullRebuildNeedsClear = false;
    m_voxelGiDirtyChunkIndices.clear();
}



} // namespace voxelsprout::render
