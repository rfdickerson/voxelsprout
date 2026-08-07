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

bool RendererBackend::captureLastFrameToFile(const std::string& outputPath) {
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

    // Everything below is one-shot and torn down before returning. A capture
    // happens once per run at most, so holding no persistent state is worth
    // more than avoiding the allocation.
    vkDeviceWaitIdle(m_device);

    VkBuffer readbackBuffer = VK_NULL_HANDLE;
    VkDeviceMemory readbackMemory = VK_NULL_HANDLE;
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = imageBytes;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(m_device, &bufferCreateInfo, nullptr, &readbackBuffer) != VK_SUCCESS) {
        VOX_LOGE("render") << "frame capture: readback buffer creation failed";
        return false;
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(m_device, readbackBuffer, &memoryRequirements);
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memoryProperties);
    uint32_t memoryTypeIndex = UINT32_MAX;
    const VkMemoryPropertyFlags wanted =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((memoryRequirements.memoryTypeBits & (1u << i)) != 0u &&
            (memoryProperties.memoryTypes[i].propertyFlags & wanted) == wanted) {
            memoryTypeIndex = i;
            break;
        }
    }
    if (memoryTypeIndex == UINT32_MAX) {
        VOX_LOGE("render") << "frame capture: no host-visible memory type";
        vkDestroyBuffer(m_device, readbackBuffer, nullptr);
        return false;
    }

    VkMemoryAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = memoryTypeIndex;
    if (vkAllocateMemory(m_device, &allocateInfo, nullptr, &readbackMemory) != VK_SUCCESS ||
        vkBindBufferMemory(m_device, readbackBuffer, readbackMemory, 0) != VK_SUCCESS) {
        VOX_LOGE("render") << "frame capture: readback memory allocation failed";
        if (readbackMemory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, readbackMemory, nullptr);
        }
        vkDestroyBuffer(m_device, readbackBuffer, nullptr);
        return false;
    }

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    bool ok = vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &commandPool) == VK_SUCCESS;
    if (ok) {
        VkCommandBufferAllocateInfo commandAllocateInfo{};
        commandAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandAllocateInfo.commandPool = commandPool;
        commandAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandAllocateInfo.commandBufferCount = 1;
        ok = vkAllocateCommandBuffers(m_device, &commandAllocateInfo, &commandBuffer) == VK_SUCCESS;
    }
    if (ok) {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        ok = vkBeginCommandBuffer(commandBuffer, &beginInfo) == VK_SUCCESS;
    }
    if (!ok) {
        VOX_LOGE("render") << "frame capture: command buffer setup failed";
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device, commandPool, nullptr);
        }
        vkFreeMemory(m_device, readbackMemory, nullptr);
        vkDestroyBuffer(m_device, readbackBuffer, nullptr);
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

    bool wrote = false;
    if (submitted) {
        void* mapped = nullptr;
        if (vkMapMemory(m_device, readbackMemory, 0, imageBytes, 0, &mapped) == VK_SUCCESS &&
            mapped != nullptr) {
            const auto* pixels = static_cast<const std::uint8_t*>(mapped);
            std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3u);
            for (std::size_t i = 0; i < static_cast<std::size_t>(width) * height; ++i) {
                rgb[(i * 3u) + 0] = pixels[(i * 4u) + static_cast<std::size_t>(redByte)];
                rgb[(i * 3u) + 1] = pixels[(i * 4u) + static_cast<std::size_t>(greenByte)];
                rgb[(i * 3u) + 2] = pixels[(i * 4u) + static_cast<std::size_t>(blueByte)];
            }
            vkUnmapMemory(m_device, readbackMemory);

            std::ofstream output(outputPath, std::ios::binary | std::ios::trunc);
            if (output) {
                output << "P6\n" << width << " " << height << "\n255\n";
                output.write(reinterpret_cast<const char*>(rgb.data()),
                             static_cast<std::streamsize>(rgb.size()));
                wrote = output.good();
            }
        }
    }

    vkDestroyCommandPool(m_device, commandPool, nullptr);
    vkFreeMemory(m_device, readbackMemory, nullptr);
    vkDestroyBuffer(m_device, readbackBuffer, nullptr);

    if (wrote) {
        VOX_LOGI("render") << "frame capture written: " << outputPath << " (" << width << "x" << height << ")";
    } else {
        VOX_LOGE("render") << "frame capture failed to write " << outputPath;
    }
    return wrote;
}

}  // namespace odai::render
