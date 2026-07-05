#include "offscreen_capture.h"

#include "render/backend/vulkan/buffer_helpers.h"
#include "render/backend/vulkan/ui_renderer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <vector>

using odai::render::BufferAllocator;
using odai::render::FrameArena;
using odai::render::FrameArenaConfig;
using odai::render::UiRenderer;

namespace odai::uistress {

namespace {

std::filesystem::path resolveRepoPath(const std::filesystem::path& relativePath) {
    std::vector<std::filesystem::path> baseCandidates;
#if defined(ODAI_PROJECT_SOURCE_DIR)
    baseCandidates.emplace_back(std::filesystem::path{ODAI_PROJECT_SOURCE_DIR});
#endif
    std::error_code cwdError;
    const std::filesystem::path cwd = std::filesystem::current_path(cwdError);
    if (!cwdError) {
        baseCandidates.push_back(cwd);
        baseCandidates.push_back(cwd / "..");
    }
    for (const std::filesystem::path& base : baseCandidates) {
        const std::filesystem::path candidate = base / relativePath;
        std::error_code existsError;
        if (std::filesystem::exists(candidate, existsError) && !existsError) {
            return candidate;
        }
    }
    return relativePath;
}

}  // namespace

struct OffscreenCapture::Impl {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queueFamily = 0;
    VmaAllocator vmaAllocator = VK_NULL_HANDLE;

    VkImage colorImage = VK_NULL_HANDLE;
    VmaAllocation colorImageAllocation = VK_NULL_HANDLE;
    VkImageView colorImageView = VK_NULL_HANDLE;
    VkImageLayout colorImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkBuffer readbackBuffer = VK_NULL_HANDLE;
    VmaAllocation readbackAllocation = VK_NULL_HANDLE;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    BufferAllocator bufferAllocator;
    FrameArena frameArena;
    UiRenderer uiRenderer;

    std::uint32_t width = 0;
    std::uint32_t height = 0;

    ~Impl() {
        if (device == VK_NULL_HANDLE) {
            return;
        }
        vkDeviceWaitIdle(device);
        uiRenderer.shutdown();
        frameArena.shutdown(&bufferAllocator);
        if (readbackBuffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(vmaAllocator, readbackBuffer, readbackAllocation);
        }
        if (colorImageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, colorImageView, nullptr);
        }
        if (colorImage != VK_NULL_HANDLE) {
            vmaDestroyImage(vmaAllocator, colorImage, colorImageAllocation);
        }
        bufferAllocator.shutdown();
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, commandPool, nullptr);
        }
        if (vmaAllocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(vmaAllocator);
        }
        if (device != VK_NULL_HANDLE) {
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }
};

namespace {

bool pickHeadlessDevice(OffscreenCapture::Impl& impl) {
    std::uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(impl.instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::fprintf(stderr, "[offscreen_capture] no Vulkan physical devices found\n");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(impl.instance, &deviceCount, devices.data());

    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceVulkan12Features features12{};
        features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceVulkan13Features features13{};
        features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        features13.pNext = &features12;
        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features13;
        vkGetPhysicalDeviceFeatures2(candidate, &features2);

        if (features13.dynamicRendering != VK_TRUE) continue;
        if (features12.descriptorBindingPartiallyBound != VK_TRUE) continue;
        if (features12.descriptorBindingSampledImageUpdateAfterBind != VK_TRUE) continue;
        if (features12.shaderSampledImageArrayNonUniformIndexing != VK_TRUE) continue;

        std::uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, queueFamilies.data());
        for (std::uint32_t i = 0; i < queueFamilyCount; ++i) {
            if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
                impl.physicalDevice = candidate;
                impl.queueFamily = i;
                break;
            }
        }
        if (impl.physicalDevice != VK_NULL_HANDLE) break;
    }
    if (impl.physicalDevice == VK_NULL_HANDLE) {
        std::fprintf(stderr, "[offscreen_capture] no GPU with dynamicRendering + bindless "
                             "descriptor indexing + a graphics queue found\n");
        return false;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = impl.queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan12Features enable12{};
    enable12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    enable12.descriptorIndexing = VK_TRUE;
    enable12.descriptorBindingPartiallyBound = VK_TRUE;
    enable12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    enable12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    VkPhysicalDeviceVulkan13Features enable13{};
    enable13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    enable13.dynamicRendering = VK_TRUE;
    enable13.pNext = &enable12;
    VkPhysicalDeviceFeatures2 enableFeatures2{};
    enableFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    enableFeatures2.pNext = &enable13;

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext = &enableFeatures2;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    if (vkCreateDevice(impl.physicalDevice, &deviceInfo, nullptr, &impl.device) != VK_SUCCESS) {
        std::fprintf(stderr, "[offscreen_capture] vkCreateDevice failed\n");
        return false;
    }
    vkGetDeviceQueue(impl.device, impl.queueFamily, 0, &impl.queue);
    return true;
}

void transitionImage(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                      VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                      VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

}  // namespace

bool OffscreenCapture::init(const Config& config) {
    m_width = config.width;
    m_height = config.height;
    m_impl = new Impl();
    Impl& impl = *m_impl;
    impl.width = config.width;
    impl.height = config.height;

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "odai_ui_stress_test_capture";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    if (vkCreateInstance(&instanceInfo, nullptr, &impl.instance) != VK_SUCCESS) {
        std::fprintf(stderr, "[offscreen_capture] vkCreateInstance failed\n");
        return false;
    }
    if (!pickHeadlessDevice(impl)) {
        return false;
    }

    VmaAllocatorCreateInfo vmaInfo{};
    vmaInfo.physicalDevice = impl.physicalDevice;
    vmaInfo.device = impl.device;
    vmaInfo.instance = impl.instance;
    vmaInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    if (vmaCreateAllocator(&vmaInfo, &impl.vmaAllocator) != VK_SUCCESS) {
        std::fprintf(stderr, "[offscreen_capture] vmaCreateAllocator failed\n");
        return false;
    }

    // Off-screen color target. R8G8B8A8_UNORM so the raw bytes we read back are
    // already sRGB-encoded (matching how the UI shader's linearToSrgb() output
    // is meant to be interpreted), which is exactly what a PNG expects.
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {config.width, config.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo imageAllocInfo{};
    imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateImage(impl.vmaAllocator, &imageInfo, &imageAllocInfo, &impl.colorImage,
                        &impl.colorImageAllocation, nullptr) != VK_SUCCESS) {
        std::fprintf(stderr, "[offscreen_capture] vmaCreateImage failed\n");
        return false;
    }
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = impl.colorImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(impl.device, &viewInfo, nullptr, &impl.colorImageView) != VK_SUCCESS) {
        std::fprintf(stderr, "[offscreen_capture] vkCreateImageView failed\n");
        return false;
    }

    const VkDeviceSize readbackSize = VkDeviceSize{config.width} * config.height * 4u;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = readbackSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo bufferAllocInfo{};
    bufferAllocInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
    bufferAllocInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo readbackAllocInfo{};
    if (vmaCreateBuffer(impl.vmaAllocator, &bufferInfo, &bufferAllocInfo, &impl.readbackBuffer,
                         &impl.readbackAllocation, &readbackAllocInfo) != VK_SUCCESS) {
        std::fprintf(stderr, "[offscreen_capture] vmaCreateBuffer (readback) failed\n");
        return false;
    }

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = impl.queueFamily;
    if (vkCreateCommandPool(impl.device, &poolInfo, nullptr, &impl.commandPool) != VK_SUCCESS) {
        std::fprintf(stderr, "[offscreen_capture] vkCreateCommandPool failed\n");
        return false;
    }
    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = impl.commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(impl.device, &cmdAllocInfo, &impl.commandBuffer) != VK_SUCCESS) {
        std::fprintf(stderr, "[offscreen_capture] vkAllocateCommandBuffers failed\n");
        return false;
    }

    if (!impl.bufferAllocator.init(impl.physicalDevice, impl.device, impl.vmaAllocator)) {
        std::fprintf(stderr, "[offscreen_capture] BufferAllocator::init failed\n");
        return false;
    }
    FrameArenaConfig arenaConfig{};
    arenaConfig.uploadBytesPerFrame = 2 * 1024 * 1024;
    arenaConfig.frameCount = 1;
    arenaConfig.uploadUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (!impl.frameArena.init(&impl.bufferAllocator, impl.physicalDevice, impl.device, arenaConfig,
                               impl.vmaAllocator)) {
        std::fprintf(stderr, "[offscreen_capture] FrameArena::init failed\n");
        return false;
    }

    UiRenderer::InitInfo uiInfo{};
    uiInfo.device = impl.device;
    uiInfo.physicalDevice = impl.physicalDevice;
    uiInfo.vmaAllocator = impl.vmaAllocator;
    uiInfo.bufferAllocator = &impl.bufferAllocator;
    uiInfo.uploadQueue = impl.queue;
    uiInfo.uploadQueueFamily = impl.queueFamily;
    uiInfo.colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
    uiInfo.maxTextureCount = config.maxTextureCount;
    uiInfo.shaderDir = config.shaderDir.empty()
                           ? resolveRepoPath("src/render/shaders").string()
                           : config.shaderDir;
    if (!impl.uiRenderer.init(uiInfo)) {
        std::fprintf(stderr, "[offscreen_capture] UiRenderer::init failed (shaderDir=%s)\n",
                     uiInfo.shaderDir.c_str());
        return false;
    }
    return true;
}

void OffscreenCapture::shutdown() {
    delete m_impl;
    m_impl = nullptr;
}

bool OffscreenCapture::loadPrimaryFont(odai::ui::Font& font, const std::string& ttfPath, float pixelHeight) {
    const std::filesystem::path resolved = resolveRepoPath(ttfPath);
    if (!font.loadFromFile(resolved.string(), pixelHeight)) {
        std::fprintf(stderr, "[offscreen_capture] Font::loadFromFile failed for %s\n", resolved.string().c_str());
        return false;
    }
    return m_impl->uiRenderer.setFontAtlasR8(font.atlasPixels().data(), font.atlasWidth(), font.atlasHeight());
}

odai::ui::UiTextureId OffscreenCapture::registerFontAtlas(const odai::ui::Font& font) {
    return m_impl->uiRenderer.registerFontAtlasR8(font.atlasPixels().data(), font.atlasWidth(), font.atlasHeight());
}

odai::ui::UiTextureId OffscreenCapture::registerTextureRgba8(const std::uint8_t* pixels, std::uint32_t w,
                                                              std::uint32_t h) {
    return m_impl->uiRenderer.registerTextureRgba8(pixels, w, h);
}

odai::ui::UiTextureId OffscreenCapture::registerTextureRgba8Mipmapped(const std::uint8_t* pixels, std::uint32_t w,
                                                                       std::uint32_t h) {
    return m_impl->uiRenderer.registerTextureRgba8Mipmapped(pixels, w, h);
}

CaptureResult OffscreenCapture::captureToPng(const odai::ui::UiDrawData& drawData, const std::string& outPngPath) {
    CaptureResult result{};
    Impl& impl = *m_impl;

    vkResetCommandBuffer(impl.commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(impl.commandBuffer, &beginInfo);

    impl.frameArena.beginFrame(0);
    impl.uiRenderer.uploadGeometry(impl.commandBuffer, impl.frameArena, drawData);

    transitionImage(impl.commandBuffer, impl.colorImage, impl.colorImageLayout,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    impl.colorImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = impl.colorImageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.06f, 0.07f, 0.09f, 1.0f}};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.extent = {impl.width, impl.height};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    vkCmdBeginRendering(impl.commandBuffer, &renderingInfo);

    VkViewport viewport{0.0f, 0.0f, static_cast<float>(impl.width), static_cast<float>(impl.height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {impl.width, impl.height}};
    vkCmdSetViewport(impl.commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(impl.commandBuffer, 0, 1, &scissor);

    impl.uiRenderer.record(impl.commandBuffer, 0, drawData, {impl.width, impl.height});
    vkCmdEndRendering(impl.commandBuffer);

    transitionImage(impl.commandBuffer, impl.colorImage, impl.colorImageLayout,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    impl.colorImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {impl.width, impl.height, 1};
    vkCmdCopyImageToBuffer(impl.commandBuffer, impl.colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            impl.readbackBuffer, 1, &region);
    vkEndCommandBuffer(impl.commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &impl.commandBuffer;

    const auto start = std::chrono::steady_clock::now();
    vkQueueSubmit(impl.queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(impl.queue);
    const auto end = std::chrono::steady_clock::now();
    result.submitToIdleMs = std::chrono::duration<double, std::milli>(end - start).count();

    const UiRenderer::Stats& stats = impl.uiRenderer.stats();
    result.drawCallCount = stats.drawCallCount;
    result.commandCount = stats.commandCount;
    result.vertexCount = static_cast<std::uint32_t>(drawData.vertices.size());
    result.indexCount = static_cast<std::uint32_t>(drawData.indices.size());

    VmaAllocationInfo readbackInfo{};
    vmaGetAllocationInfo(impl.vmaAllocator, impl.readbackAllocation, &readbackInfo);
    if (readbackInfo.pMappedData == nullptr) {
        std::fprintf(stderr, "[offscreen_capture] readback buffer is not host-mapped\n");
        return result;
    }
    // Tightly packed rows (bufferRowLength=0 in the copy region), so stride == width*4.
    const int writeOk = stbi_write_png(outPngPath.c_str(), static_cast<int>(impl.width),
                                        static_cast<int>(impl.height), 4, readbackInfo.pMappedData,
                                        static_cast<int>(impl.width) * 4);
    if (writeOk == 0) {
        std::fprintf(stderr, "[offscreen_capture] stbi_write_png failed for %s\n", outPngPath.c_str());
        return result;
    }
    result.success = true;
    return result;
}

}  // namespace odai::uistress
