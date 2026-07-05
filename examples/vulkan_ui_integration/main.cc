// Minimal standalone integration sample for odai_ui + odai_ui_vulkan.
//
// Purpose: prove the SDK contract works with NO dependency on this repo's app/,
// game/, or world/ modules — everything below the "--- odai_ui contract ---"
// markers is generic Vulkan bootstrap any engine already has; everything after
// is the actual integration surface a host engine implements.
//
// This is deliberately not production-hardened: single queue family, no
// swapchain-resize handling, no depth buffer, 2 frames in flight. See README.md
// in this directory for the step-by-step contract walkthrough.
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <GLFW/glfw3.h>

#include "render/backend/vulkan/buffer_helpers.h"
#include "render/backend/vulkan/ui_renderer.h"
#include "ui/odai_ui.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace odai::ui;
using namespace odai::render;

namespace {

constexpr std::uint32_t kFramesInFlight = 2;
constexpr std::uint32_t kMaxUiTextures = 64;

// Resolves an asset path relative to the repo root (via a compile-time macro)
// or the current working directory, whichever exists. A real vendoring project
// would just hardcode its own asset root instead of this search.
std::filesystem::path resolveAssetPath(const std::filesystem::path& relativePath) {
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

// --- Generic Vulkan bootstrap (any engine already has something like this) ---

struct VulkanApp {
    GLFWwindow* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queueFamily = 0;
    VmaAllocator vmaAllocator = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, kFramesInFlight> commandBuffers{};
    std::array<VkSemaphore, kFramesInFlight> imageAvailable{};
    std::array<VkSemaphore, kFramesInFlight> renderFinished{};
    std::array<VkFence, kFramesInFlight> inFlight{};

    BufferAllocator bufferAllocator;
    FrameArena frameArena;
    UiRenderer uiRenderer;
};

bool createInstanceAndSurface(VulkanApp& app) {
    std::uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    if (glfwExts == nullptr) {
        std::fprintf(stderr, "GLFW reports no Vulkan support\n");
        return false;
    }
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "odai_ui_integration_sample";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    if (vkCreateInstance(&createInfo, nullptr, &app.instance) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateInstance failed\n");
        return false;
    }
    if (glfwCreateWindowSurface(app.instance, app.window, nullptr, &app.surface) != VK_SUCCESS) {
        std::fprintf(stderr, "glfwCreateWindowSurface failed\n");
        return false;
    }
    return true;
}

// Selects a GPU supporting exactly what odai_ui_vulkan's bindless UI pipeline
// needs: Vulkan 1.3 dynamicRendering, plus the 1.2 descriptor-indexing features
// behind its update-after-bind texture array. Real engines already do a more
// thorough device selection than this for their own 3D rendering needs; a UI-
// only consumer needs nothing beyond what's checked here.
bool pickDeviceAndCreateLogical(VulkanApp& app) {
    std::uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(app.instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::fprintf(stderr, "no Vulkan physical devices found\n");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(app.instance, &deviceCount, devices.data());

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
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, app.surface, &presentSupport);
            if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0 || presentSupport == VK_FALSE) {
                continue;
            }
            app.physicalDevice = candidate;
            app.queueFamily = i;
            break;
        }
        if (app.physicalDevice != VK_NULL_HANDLE) break;
    }
    if (app.physicalDevice == VK_NULL_HANDLE) {
        std::fprintf(stderr, "no GPU found with dynamicRendering + bindless descriptor indexing "
                             "+ a combined graphics/present queue\n");
        return false;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = app.queueFamily;
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

    const std::array<const char*, 1> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext = &enableFeatures2;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;
    deviceInfo.enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
    if (vkCreateDevice(app.physicalDevice, &deviceInfo, nullptr, &app.device) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateDevice failed\n");
        return false;
    }
    vkGetDeviceQueue(app.device, app.queueFamily, 0, &app.queue);

    VmaAllocatorCreateInfo vmaInfo{};
    vmaInfo.physicalDevice = app.physicalDevice;
    vmaInfo.device = app.device;
    vmaInfo.instance = app.instance;
    vmaInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    if (vmaCreateAllocator(&vmaInfo, &app.vmaAllocator) != VK_SUCCESS) {
        std::fprintf(stderr, "vmaCreateAllocator failed\n");
        return false;
    }
    return true;
}

bool createSwapchain(VulkanApp& app) {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(app.physicalDevice, app.surface, &caps);

    std::uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(app.physicalDevice, app.surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(app.physicalDevice, app.surface, &formatCount, formats.data());
    VkSurfaceFormatKHR chosenFormat = formats[0];
    for (const VkSurfaceFormatKHR& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) {
            chosenFormat = f;
            break;
        }
    }
    app.swapchainFormat = chosenFormat.format;

    int width = 0, height = 0;
    glfwGetFramebufferSize(app.window, &width, &height);
    app.swapchainExtent.width = std::clamp(static_cast<std::uint32_t>(width), caps.minImageExtent.width,
                                            caps.maxImageExtent.width);
    app.swapchainExtent.height = std::clamp(static_cast<std::uint32_t>(height), caps.minImageExtent.height,
                                             caps.maxImageExtent.height);

    std::uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchainInfo{};
    swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainInfo.surface = app.surface;
    swapchainInfo.minImageCount = imageCount;
    swapchainInfo.imageFormat = chosenFormat.format;
    swapchainInfo.imageColorSpace = chosenFormat.colorSpace;
    swapchainInfo.imageExtent = app.swapchainExtent;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainInfo.preTransform = caps.currentTransform;
    swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // Universally supported; vsynced.
    swapchainInfo.clipped = VK_TRUE;
    if (vkCreateSwapchainKHR(app.device, &swapchainInfo, nullptr, &app.swapchain) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateSwapchainKHR failed\n");
        return false;
    }

    std::uint32_t actualImageCount = 0;
    vkGetSwapchainImagesKHR(app.device, app.swapchain, &actualImageCount, nullptr);
    app.swapchainImages.resize(actualImageCount);
    vkGetSwapchainImagesKHR(app.device, app.swapchain, &actualImageCount, app.swapchainImages.data());

    app.swapchainImageViews.resize(actualImageCount);
    for (std::size_t i = 0; i < app.swapchainImages.size(); ++i) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = app.swapchainImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = app.swapchainFormat;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(app.device, &viewInfo, nullptr, &app.swapchainImageViews[i]) != VK_SUCCESS) {
            std::fprintf(stderr, "vkCreateImageView failed for swapchain image %zu\n", i);
            return false;
        }
    }
    return true;
}

bool createCommandsAndSync(VulkanApp& app) {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = app.queueFamily;
    if (vkCreateCommandPool(app.device, &poolInfo, nullptr, &app.commandPool) != VK_SUCCESS) {
        return false;
    }

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = app.commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = kFramesInFlight;
    if (vkAllocateCommandBuffers(app.device, &cmdAllocInfo, app.commandBuffers.data()) != VK_SUCCESS) {
        return false;
    }

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (std::uint32_t i = 0; i < kFramesInFlight; ++i) {
        if (vkCreateSemaphore(app.device, &semInfo, nullptr, &app.imageAvailable[i]) != VK_SUCCESS ||
            vkCreateSemaphore(app.device, &semInfo, nullptr, &app.renderFinished[i]) != VK_SUCCESS ||
            vkCreateFence(app.device, &fenceInfo, nullptr, &app.inFlight[i]) != VK_SUCCESS) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    if (glfwInit() != GLFW_TRUE) {
        std::fprintf(stderr, "glfwInit failed\n");
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    VulkanApp app;
    app.window = glfwCreateWindow(1024, 720, "odai_ui integration sample", nullptr, nullptr);
    if (app.window == nullptr || !createInstanceAndSurface(app) || !pickDeviceAndCreateLogical(app) ||
        !createSwapchain(app) || !createCommandsAndSync(app)) {
        std::fprintf(stderr, "Vulkan bootstrap failed\n");
        return 1;
    }
    if (!app.bufferAllocator.init(app.physicalDevice, app.device, app.vmaAllocator)) {
        std::fprintf(stderr, "BufferAllocator::init failed\n");
        return 1;
    }
    FrameArenaConfig arenaConfig{};
    arenaConfig.uploadBytesPerFrame = 1 * 1024 * 1024;  // 1 MiB/frame is plenty for UI vertex/index streaming.
    arenaConfig.frameCount = kFramesInFlight;
    arenaConfig.uploadUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (!app.frameArena.init(&app.bufferAllocator, app.physicalDevice, app.device, arenaConfig, app.vmaAllocator)) {
        std::fprintf(stderr, "FrameArena::init failed\n");
        return 1;
    }

    // --- odai_ui contract: step 1, initialize the Vulkan bridge ---
    UiRenderer::InitInfo uiInfo{};
    uiInfo.device = app.device;
    uiInfo.physicalDevice = app.physicalDevice;
    uiInfo.vmaAllocator = app.vmaAllocator;
    uiInfo.bufferAllocator = &app.bufferAllocator;
    uiInfo.uploadQueue = app.queue;
    uiInfo.uploadQueueFamily = app.queueFamily;
    uiInfo.colorFormat = app.swapchainFormat;
    uiInfo.maxTextureCount = kMaxUiTextures;
    uiInfo.shaderDir = resolveAssetPath("src/render/shaders").string();
    if (!app.uiRenderer.init(uiInfo)) {
        std::fprintf(stderr, "UiRenderer::init failed (are ui.vert/frag.slang.spv built? see README)\n");
        return 1;
    }

    // --- odai_ui contract: step 2, bake a font atlas and upload it ---
    Font font;
    const std::filesystem::path fontPath = resolveAssetPath("assets/fonts/EBGaramond-Regular.ttf");
    if (!font.loadFromFile(fontPath.string(), 18.0f)) {
        std::fprintf(stderr, "Font::loadFromFile failed for %s\n", fontPath.string().c_str());
        return 1;
    }
    if (!app.uiRenderer.setFontAtlasR8(font.atlasPixels().data(), font.atlasWidth(), font.atlasHeight())) {
        std::fprintf(stderr, "UiRenderer::setFontAtlasR8 failed\n");
        return 1;
    }

    // --- odai_ui contract: step 3, build a widget tree (Vulkan-free) ---
    UiContext uiContext;
    uiContext.setViewport(UiVec2{static_cast<float>(app.swapchainExtent.width),
                                 static_cast<float>(app.swapchainExtent.height)});
    auto root = std::make_unique<Panel>();
    root->mousePassthrough = true;
    root->background = UiColor{0.10f, 0.11f, 0.13f, 1.0f};

    auto card = std::make_unique<Panel>();
    card->setRect(UiRect::fromXYWH(40.0f, 40.0f, 320.0f, 160.0f));
    card->styleCard(1.0f);
    Panel* cardPtr = card.get();

    auto label = std::make_unique<Label>(&font, "odai_ui SDK sample");
    label->setRect(UiRect::fromXYWH(20.0f, 16.0f, 280.0f, 28.0f));
    cardPtr->addChild(std::move(label));

    auto button = std::make_unique<Button>(&font, "Click me", nullptr);
    button->setRect(UiRect::fromXYWH(20.0f, 60.0f, 140.0f, 36.0f));
    int clickCount = 0;
    Label* countLabelPtr = nullptr;
    {
        auto countLabel = std::make_unique<Label>(&font, "Clicks: 0");
        countLabel->setRect(UiRect::fromXYWH(20.0f, 108.0f, 280.0f, 24.0f));
        countLabelPtr = countLabel.get();
        cardPtr->addChild(std::move(countLabel));
    }
    button->activated.connect([&clickCount, countLabelPtr]() {
        ++clickCount;
        countLabelPtr->setText("Clicks: " + std::to_string(clickCount));
    });
    cardPtr->addChild(std::move(button));
    root->addChild(std::move(card));
    uiContext.setRoot(std::move(root));

    UiDrawList drawList;
    UiInput input;

    // --- Frame loop ---
    std::uint32_t frameIndex = 0;
    while (!glfwWindowShouldClose(app.window)) {
        glfwPollEvents();

        double cursorX = 0.0, cursorY = 0.0;
        glfwGetCursorPos(app.window, &cursorX, &cursorY);
        input.beginFrame();
        input.mousePx = UiVec2{static_cast<float>(cursorX), static_cast<float>(cursorY)};
        input.setButton(UiMouseButton::Left, glfwGetMouseButton(app.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);

        // --- odai_ui contract: step 4, per-frame update + build ---
        uiContext.update(input);
        uiContext.build(drawList);

        vkWaitForFences(app.device, 1, &app.inFlight[frameIndex], VK_TRUE, UINT64_MAX);
        std::uint32_t imageIndex = 0;
        const VkResult acquireResult = vkAcquireNextImageKHR(
            app.device, app.swapchain, UINT64_MAX, app.imageAvailable[frameIndex], VK_NULL_HANDLE, &imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            continue;  // Resize handling is out of scope for this minimal sample.
        }
        vkResetFences(app.device, 1, &app.inFlight[frameIndex]);

        VkCommandBuffer cmd = app.commandBuffers[frameIndex];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginInfo);

        app.frameArena.beginFrame(frameIndex);
        // uploadGeometry() must run outside the rendering pass — see ui_renderer.h.
        app.uiRenderer.uploadGeometry(cmd, app.frameArena, drawList.data());

        transitionImage(cmd, app.swapchainImages[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = app.swapchainImageViews[imageIndex];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue.color = {{0.02f, 0.02f, 0.03f, 1.0f}};

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.extent = app.swapchainExtent;
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        vkCmdBeginRendering(cmd, &renderingInfo);

        VkViewport viewport{0.0f, 0.0f, static_cast<float>(app.swapchainExtent.width),
                            static_cast<float>(app.swapchainExtent.height), 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, app.swapchainExtent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // --- odai_ui contract: step 5, record inside the open rendering pass ---
        app.uiRenderer.record(cmd, frameIndex, drawList.data(), app.swapchainExtent);

        vkCmdEndRendering(cmd);
        transitionImage(cmd, app.swapchainImages[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
        vkEndCommandBuffer(cmd);

        const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &app.imageAvailable[frameIndex];
        submitInfo.pWaitDstStageMask = &waitStage;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &app.renderFinished[frameIndex];
        vkQueueSubmit(app.queue, 1, &submitInfo, app.inFlight[frameIndex]);

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &app.renderFinished[frameIndex];
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &app.swapchain;
        presentInfo.pImageIndices = &imageIndex;
        vkQueuePresentKHR(app.queue, &presentInfo);

        frameIndex = (frameIndex + 1) % kFramesInFlight;
    }

    vkDeviceWaitIdle(app.device);
    glfwDestroyWindow(app.window);
    glfwTerminate();
    return 0;
}
