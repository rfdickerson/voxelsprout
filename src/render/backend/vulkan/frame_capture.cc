// Frame capture: copy the last presented swapchain image to a file.
//
// This exists because there was no way to see what the renderer actually drew
// without a human looking at a monitor. On a Wayland desktop an external
// screenshot is not a fallback -- the compositor refuses unsandboxed capture
// requests, and a native Wayland surface is invisible to X11 grabbers -- so
// visual bugs could only be described second-hand. Several rounds of diagnosing
// "everything looks grey" from cooked-data statistics produced two confident,
// wrong answers before this was written.
//
// Output is binary PPM (P6). Deliberately not PNG: the encoder would be a new
// dependency for every target that compiles the Vulkan backend, and this is a
// diagnostic, not a feature. Any image tool reads PPM; `ffmpeg -i shot.ppm
// shot.png` converts one.

#include "render/backend/vulkan/renderer_backend.h"

#include "core/log.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace odai::render {

namespace {

// Swizzle table for the swapchain's channel order. The swapchain is
// B8G8R8A8_UNORM (see the format note in renderer.h), so the bytes come back
// BGRA and have to be reordered for PPM's RGB. Written as a lookup rather than
// hardcoded so an R8G8B8A8 swapchain does not silently produce a blue-tinted
// capture -- which would read as a rendering bug and send someone chasing it.
bool channelOrderForFormat(VkFormat format, int& outRedByte, int& outGreenByte, int& outBlueByte) {
    switch (format) {
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
            outRedByte = 2;
            outGreenByte = 1;
            outBlueByte = 0;
            return true;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
            outRedByte = 0;
            outGreenByte = 1;
            outBlueByte = 2;
            return true;
        default:
            return false;
    }
}

}  // namespace

void RendererBackend::destroyFrameCaptureResources() {
    if (m_device == VK_NULL_HANDLE) {
        return;
    }
    if (m_captureCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_captureCommandPool, nullptr);
        m_captureCommandPool = VK_NULL_HANDLE;
        m_captureCommandBuffer = VK_NULL_HANDLE;
    }
    if (m_captureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_captureMemory, nullptr);
        m_captureMemory = VK_NULL_HANDLE;
    }
    if (m_captureBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device, m_captureBuffer, nullptr);
        m_captureBuffer = VK_NULL_HANDLE;
    }
    m_captureBufferBytes = 0;
}

bool RendererBackend::captureLastFrameToFile(const std::string& outputPath) {
    std::vector<std::uint8_t> rgb;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    if (!captureLastFrameRgb(rgb, width, height)) {
        return false;
    }
    std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
    bool wrote = false;
    if (output) {
        output << "P6\n" << width << " " << height << "\n255\n";
        output.write(reinterpret_cast<const char*>(rgb.data()),
                     static_cast<std::streamsize>(rgb.size()));
        wrote = output.good();
    }
    if (wrote) {
        VOX_LOGI("render") << "frame capture written: " << outputPath << " (" << width << "x"
                           << height << ")";
    } else {
        VOX_LOGE("render") << "frame capture failed to write " << outputPath;
    }
    return wrote;
}

bool RendererBackend::captureLastFrameRgb(std::vector<std::uint8_t>& outRgb,
                                          std::uint32_t& outWidth,
                                          std::uint32_t& outHeight) {
    if (m_device == VK_NULL_HANDLE || m_swapchainImages.empty()) {
        VOX_LOGE("render") << "frame capture: renderer not initialized";
        return false;
    }
    if (m_lastPresentedImageIndex >= m_swapchainImages.size()) {
        VOX_LOGE("render") << "frame capture: no frame has been presented yet";
        return false;
    }
    int redByte = 0;
    int greenByte = 0;
    int blueByte = 0;
    if (!channelOrderForFormat(m_swapchainFormat, redByte, greenByte, blueByte)) {
        VOX_LOGE("render") << "frame capture: unsupported swapchain format "
                           << static_cast<int>(m_swapchainFormat);
        return false;
    }

    const VkImage sourceImage = m_swapchainImages[m_lastPresentedImageIndex];
    const uint32_t width = m_swapchainExtent.width;
    const uint32_t height = m_swapchainExtent.height;
    const VkDeviceSize imageBytes = static_cast<VkDeviceSize>(width) * height * 4u;

    // Build the readback resources once and keep them. A video capture calls
    // this every frame, and creating a buffer, an allocation and a command pool
    // per call -- plus a full-device wait -- dominated the frame time by more
    // than an order of magnitude over the render itself.
    if (m_captureBufferBytes != imageBytes) {
        destroyFrameCaptureResources();

        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.size = imageBytes;
        bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(m_device, &bufferCreateInfo, nullptr, &m_captureBuffer) != VK_SUCCESS) {
            VOX_LOGE("render") << "frame capture: readback buffer creation failed";
            return false;
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(m_device, m_captureBuffer, &memoryRequirements);
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);
        uint32_t memoryTypeIndex = UINT32_MAX;
        const VkMemoryPropertyFlags required =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        // HOST_CACHED first. Plain HOST_VISIBLE|HOST_COHERENT is typically
        // write-combined, which is fine to write and dreadful to READ -- and a
        // readback does nothing but read. Measured on the LNL iGPU at
        // 2560x1440, the swizzle loop below pulling bytes straight out of an
        // uncached mapping cost ~1.1 s per frame, roughly 40x the render it was
        // capturing. Uncached memory does not show up as GPU time or as I/O; it
        // shows up as userspace CPU time, which is what made it look like the
        // encoder's fault.
        for (uint32_t pass = 0; pass < 2u && memoryTypeIndex == UINT32_MAX; ++pass) {
            const VkMemoryPropertyFlags wanted =
                (pass == 0u) ? (required | VK_MEMORY_PROPERTY_HOST_CACHED_BIT) : required;
            for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
                if ((memoryRequirements.memoryTypeBits & (1u << i)) != 0u &&
                    (memoryProperties.memoryTypes[i].propertyFlags & wanted) == wanted) {
                    memoryTypeIndex = i;
                    break;
                }
            }
        }
        if (memoryTypeIndex == UINT32_MAX) {
            VOX_LOGE("render") << "frame capture: no host-visible memory type";
            destroyFrameCaptureResources();
            return false;
        }

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = memoryTypeIndex;
        if (vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_captureMemory) != VK_SUCCESS ||
            vkBindBufferMemory(m_device, m_captureBuffer, m_captureMemory, 0) != VK_SUCCESS) {
            VOX_LOGE("render") << "frame capture: readback memory allocation failed";
            destroyFrameCaptureResources();
            return false;
        }

        VkCommandPoolCreateInfo poolCreateInfo{};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
        bool built =
            vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &m_captureCommandPool) ==
            VK_SUCCESS;
        if (built) {
            VkCommandBufferAllocateInfo commandAllocateInfo{};
            commandAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandAllocateInfo.commandPool = m_captureCommandPool;
            commandAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandAllocateInfo.commandBufferCount = 1;
            built = vkAllocateCommandBuffers(m_device, &commandAllocateInfo,
                                             &m_captureCommandBuffer) == VK_SUCCESS;
        }
        if (!built) {
            VOX_LOGE("render") << "frame capture: command buffer setup failed";
            destroyFrameCaptureResources();
            return false;
        }
        m_captureBufferBytes = imageBytes;
    }

    VkBuffer readbackBuffer = m_captureBuffer;
    VkDeviceMemory readbackMemory = m_captureMemory;
    VkCommandBuffer commandBuffer = m_captureCommandBuffer;

    // The frame being read is the one already presented, so waiting on the
    // graphics queue after the copy below is enough -- the old full-device wait
    // here was belt-and-braces and cost more than everything else combined.
    vkResetCommandBuffer(commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        VOX_LOGE("render") << "frame capture: vkBeginCommandBuffer failed";
        return false;
    }

    // The presented image sits in PRESENT_SRC_KHR and must be put back there:
    // the swapchain image is not ours to leave in another layout, and the next
    // acquire of this index would otherwise transition from a layout the
    // renderer did not expect.
    const auto barrier = [&](VkImageLayout oldLayout,
                             VkImageLayout newLayout,
                             VkPipelineStageFlags2 srcStage,
                             VkAccessFlags2 srcAccess,
                             VkPipelineStageFlags2 dstStage,
                             VkAccessFlags2 dstAccess) {
        VkImageMemoryBarrier2 imageBarrier{};
        imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        imageBarrier.srcStageMask = srcStage;
        imageBarrier.srcAccessMask = srcAccess;
        imageBarrier.dstStageMask = dstStage;
        imageBarrier.dstAccessMask = dstAccess;
        imageBarrier.oldLayout = oldLayout;
        imageBarrier.newLayout = newLayout;
        imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.image = sourceImage;
        imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBarrier.subresourceRange.baseMipLevel = 0;
        imageBarrier.subresourceRange.levelCount = 1;
        imageBarrier.subresourceRange.baseArrayLayer = 0;
        imageBarrier.subresourceRange.layerCount = 1;
        VkDependencyInfo dependencyInfo{};
        dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers = &imageBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    };

    barrier(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(
        commandBuffer, sourceImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer, 1, &copyRegion);

    barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT);

    bool submitted = vkEndCommandBuffer(commandBuffer) == VK_SUCCESS;
    if (submitted) {
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitted = vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS;
    }
    if (submitted) {
        submitted = vkQueueWaitIdle(m_graphicsQueue) == VK_SUCCESS;
    }

    bool read = false;
    if (submitted) {
        void* mapped = nullptr;
        if (vkMapMemory(m_device, readbackMemory, 0, imageBytes, 0, &mapped) == VK_SUCCESS &&
            mapped != nullptr) {
            // ONE bulk read out of the mapping, then swizzle from the copy.
            // Even on a HOST_CACHED heap this is worth it, and where the driver
            // only offers write-combined memory it is the difference between a
            // capture that keeps up and one that costs a second a frame:
            // memcpy issues wide sequential loads that a write-combining
            // mapping handles well, while the three strided byte reads per
            // pixel below do not.
            m_captureStaging.resize(static_cast<std::size_t>(imageBytes));
            std::memcpy(m_captureStaging.data(), mapped, static_cast<std::size_t>(imageBytes));
            vkUnmapMemory(m_device, readbackMemory);

            const std::uint8_t* pixels = m_captureStaging.data();
            const std::size_t pixelCount = static_cast<std::size_t>(width) * height;
            outRgb.resize(pixelCount * 3u);
            for (std::size_t i = 0; i < pixelCount; ++i) {
                outRgb[(i * 3u) + 0] = pixels[(i * 4u) + static_cast<std::size_t>(redByte)];
                outRgb[(i * 3u) + 1] = pixels[(i * 4u) + static_cast<std::size_t>(greenByte)];
                outRgb[(i * 3u) + 2] = pixels[(i * 4u) + static_cast<std::size_t>(blueByte)];
            }
            outWidth = width;
            outHeight = height;
            read = true;
        }
    }
    if (!read) {
        VOX_LOGE("render") << "frame capture: readback failed";
    }
    return read;
}

}  // namespace odai::render
