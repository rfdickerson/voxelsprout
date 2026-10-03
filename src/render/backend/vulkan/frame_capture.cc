// Frame capture: copy the acquired image before presentation, then read it on the CPU.
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
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace odai::render {

#include "render/renderer_shared.h"

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
    if (m_captureBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device, m_captureBuffer, nullptr);
        m_captureBuffer = VK_NULL_HANDLE;
    }
    if (m_captureMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, m_captureMemory, nullptr);
        m_captureMemory = VK_NULL_HANDLE;
    }
    m_captureRequested = false;
    m_captureRecorded = false;
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

bool RendererBackend::prepareFrameCapture() {
    m_captureRecorded = false;
    m_captureRequested = false;
    if (m_device == VK_NULL_HANDLE || !m_captureSupported || m_swapchainImages.empty()) {
        return false;
    }
    const VkDeviceSize imageBytes = static_cast<VkDeviceSize>(m_swapchainExtent.width) *
        m_swapchainExtent.height * 4u;
    // Build the readback resources once and keep them. A video capture calls
    // this every frame, and creating a buffer, an allocation and a command pool
    // per call -- plus a full-device wait -- dominated the frame time by more
    // than an order of magnitude over the render itself.
    if (m_captureBufferBytes != imageBytes) {
        vkQueueWaitIdle(m_graphicsQueue);
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

        m_captureBufferBytes = imageBytes;
    }
    m_captureRequested = true;
    return true;
}

void RendererBackend::recordFrameCapture(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    if (!m_captureRequested || m_captureBuffer == VK_NULL_HANDLE ||
        m_captureBufferBytes != static_cast<VkDeviceSize>(m_swapchainExtent.width) *
            m_swapchainExtent.height * 4u) {
        return;
    }
    m_captureRequested = false;
    transitionImageLayout(commandBuffer, m_swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    VkBufferMemoryBarrier2 reuse{};
    reuse.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    reuse.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    reuse.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    reuse.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    reuse.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    reuse.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    reuse.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    reuse.buffer = m_captureBuffer;
    reuse.size = VK_WHOLE_SIZE;
    VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &reuse;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {m_swapchainExtent.width, m_swapchainExtent.height, 1};
    vkCmdCopyImageToBuffer(commandBuffer, m_swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_captureBuffer, 1, &copy);
    reuse.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    reuse.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    reuse.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
    transitionImageLayout(commandBuffer, m_swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    m_captureRecorded = true;
}

bool RendererBackend::captureLastFrameRgb(std::vector<std::uint8_t>& outRgb,
                                          std::uint32_t& outWidth,
                                          std::uint32_t& outHeight) {
    if (m_device == VK_NULL_HANDLE || !m_captureRecorded) {
        VOX_LOGE("render") << "frame capture: prepareFrameCapture must precede a rendered frame";
        return false;
    }
    int redByte = 0, greenByte = 0, blueByte = 0;
    if (!channelOrderForFormat(m_swapchainFormat, redByte, greenByte, blueByte)) {
        return false;
    }
    const uint32_t width = m_swapchainExtent.width;
    const uint32_t height = m_swapchainExtent.height;
    const VkDeviceSize imageBytes = static_cast<VkDeviceSize>(width) * height * 4u;
    const VkDeviceMemory readbackMemory = m_captureMemory;
    bool read = false;
    if (vkQueueWaitIdle(m_graphicsQueue) == VK_SUCCESS) {
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

bool RendererBackend::captureVoxelGi(VoxelGiCapture& output) {
    output = {};
    if (!m_voxelGiInitialized || m_device == VK_NULL_HANDLE ||
        (m_voxelGiFormat != VK_FORMAT_R16G16B16A16_SFLOAT &&
         m_voxelGiFormat != VK_FORMAT_R32G32B32A32_SFLOAT)) return false;
    if (vkQueueWaitIdle(m_graphicsQueue) != VK_SUCCESS) return false;
    const bool half = m_voxelGiFormat == VK_FORMAT_R16G16B16A16_SFLOAT;
    const std::size_t cells = std::size_t(kVoxelGiGridResolution) * kVoxelGiGridResolution * kVoxelGiGridResolution;
    BufferCreateDesc desc{};
    desc.size = cells * (half ? 8 : 16);
    desc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const auto buffer = m_bufferAllocator.createBuffer(desc);
    if (buffer == kInvalidBufferHandle) return false;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
    bool ok = vkCreateCommandPool(m_device, &poolInfo, nullptr, &pool) == VK_SUCCESS;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    if (ok) ok = vkAllocateCommandBuffers(m_device, &allocate, &command) == VK_SUCCESS;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (ok) ok = vkBeginCommandBuffer(command, &begin) == VK_SUCCESS;
    if (ok) {
        transitionImageLayout(command, m_voxelGiImages[1],
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {kVoxelGiGridResolution, kVoxelGiGridResolution, kVoxelGiGridResolution};
        vkCmdCopyImageToBuffer(command, m_voxelGiImages[1], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            m_bufferAllocator.getBuffer(buffer), 1, &copy);
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(command, &dependency);
        transitionImageLayout(command, m_voxelGiImages[1],
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        ok = vkEndCommandBuffer(command) == VK_SUCCESS;
    }
    if (ok) {
        VkCommandBufferSubmitInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
        cb.commandBuffer = command;
        VkSubmitInfo2 submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
        submit.commandBufferInfoCount = 1;
        submit.pCommandBufferInfos = &cb;
        ok = vkQueueSubmit2(m_graphicsQueue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS;
        if (ok) ok = vkQueueWaitIdle(m_graphicsQueue) == VK_SUCCESS;
    }
    if (ok) {
        const void* data = m_bufferAllocator.mapBuffer(buffer, 0, desc.size);
        ok = data != nullptr;
        if (ok) {
            output.origin = m_voxelGiPreviousGridOrigin;
            output.cellSize = kVoxelGiCellSize;
            output.resolution = kVoxelGiGridResolution;
            output.rgb.resize(cells * 3);
            for (std::size_t cell = 0; cell < cells; ++cell) {
                for (unsigned c = 0; c < 3; ++c) {
                    float value;
                    if (half) {
                        const auto h = static_cast<const std::uint16_t*>(data)[cell * 4 + c];
                        const int exponent = (h >> 10) & 31;
                        const int mantissa = h & 1023;
                        value = exponent == 31 ? (mantissa ? std::numeric_limits<float>::quiet_NaN() :
                            std::numeric_limits<float>::infinity()) :
                            std::ldexp(float(exponent ? mantissa + 1024 : mantissa), exponent ? exponent - 25 : -24);
                        if (h & 0x8000) value = -value;
                    } else value = static_cast<const float*>(data)[cell * 4 + c];
                    output.rgb[cell * 3 + c] = value;
                }
            }
            m_bufferAllocator.unmapBuffer(buffer);
        }
    }
    if (pool) vkDestroyCommandPool(m_device, pool, nullptr);
    m_bufferAllocator.destroyBuffer(buffer);
    return ok;
}

}  // namespace odai::render
