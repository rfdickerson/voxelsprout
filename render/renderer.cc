#include "render/renderer.h"

#include "core/log.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace voxelsprout::render {
namespace {

constexpr std::uint32_t kFramesInFlight = 2;
constexpr std::uint32_t kTimestampCount = 4;
constexpr std::uint32_t kTimestampFrameStart = 0;
constexpr std::uint32_t kTimestampCloudEnd = 1;
constexpr std::uint32_t kTimestampToneEnd = 2;
constexpr std::uint32_t kTimestampFrameEnd = 3;

constexpr const char* kCloudShaderPath = "shaders/cloud_path_trace.comp.slang.spv";
constexpr const char* kToneMapShaderPath = "shaders/tone_map.comp.slang.spv";

float absDiff(float a, float b) {
    return std::fabs(a - b);
}

bool almostEqual(float a, float b, float epsilon = 1e-4f) {
    return absDiff(a, b) <= epsilon;
}

bool paramsDiffer(const RenderParameters& a, const RenderParameters& b) {
    const auto& ac = a.camera;
    const auto& bc = b.camera;
    const auto& av = a.scene.volume;
    const auto& bv = b.scene.volume;
    const auto& as = a.scene.sun;
    const auto& bs = b.scene.sun;

    return
        !almostEqual(ac.position.x, bc.position.x) ||
        !almostEqual(ac.position.y, bc.position.y) ||
        !almostEqual(ac.position.z, bc.position.z) ||
        !almostEqual(ac.yawDegrees, bc.yawDegrees) ||
        !almostEqual(ac.pitchDegrees, bc.pitchDegrees) ||
        !almostEqual(ac.fovDegrees, bc.fovDegrees) ||
        !almostEqual(av.densityScale, bv.densityScale) ||
        !almostEqual(av.anisotropyG, bv.anisotropyG) ||
        !almostEqual(av.albedo, bv.albedo) ||
        !almostEqual(av.macroScale, bv.macroScale) ||
        !almostEqual(av.detailScale, bv.detailScale) ||
        !almostEqual(av.densityCutoff, bv.densityCutoff) ||
        !almostEqual(av.chunkiness, bv.chunkiness) ||
        !almostEqual(av.coverage, bv.coverage) ||
        !almostEqual(av.weatherScale, bv.weatherScale) ||
        !almostEqual(av.cloudBase, bv.cloudBase) ||
        !almostEqual(av.cloudTop, bv.cloudTop) ||
        !almostEqual(av.warpStrength, bv.warpStrength) ||
        !almostEqual(av.erosionStrength, bv.erosionStrength) ||
        !almostEqual(av.brightnessBoost, bv.brightnessBoost) ||
        !almostEqual(av.ambientLift, bv.ambientLift) ||
        av.maxBounces != bv.maxBounces ||
        !almostEqual(as.direction.x, bs.direction.x) ||
        !almostEqual(as.direction.y, bs.direction.y) ||
        !almostEqual(as.direction.z, bs.direction.z) ||
        !almostEqual(as.intensity, bs.intensity) ||
        !almostEqual(a.exposure, b.exposure) ||
        a.enableAccumulation != b.enableAccumulation;
}

std::vector<char> loadBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        return {};
    }

    std::vector<char> bytes(static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);
    file.read(bytes.data(), size);
    return bytes;
}

VkImageMemoryBarrier2 imageBarrier(
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags2 srcStage,
    VkPipelineStageFlags2 dstStage,
    VkAccessFlags2 srcAccess,
    VkAccessFlags2 dstAccess) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    return barrier;
}

struct CameraPush {
    float cameraPositionFov[4];
    float cameraForward[4];
    float cameraRight[4];
    float cameraUp[4];
    float sunDirectionIntensity[4];
    float mediumParams[4];
    float cloudShapeParams[4];
    float cloudProfileParams[4];
    float cloudWarpParams[4];
    float cloudLightParams[4];
    float frameParams[4];
};

struct ToneMapPush {
    float exposure = 1.0f;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t sampleCount = 1;
};

} // namespace

struct Renderer::Impl {
    struct ImageResource {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    struct FrameResources {
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderComplete = VK_NULL_HANDLE;
        VkQueryPool timestampQueryPool = VK_NULL_HANDLE;
        std::uint64_t submittedTimelineValue = 0;
    };

    struct CloudPathTracePass {
        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    };

    struct ToneMapPass {
        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    };

    GLFWwindow* window = nullptr;

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    std::uint32_t queueFamilyIndex = std::numeric_limits<std::uint32_t>::max();
    VkQueue queue = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent{};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    std::vector<bool> swapchainImageInitialized;

    ImageResource accumulationImage;
    ImageResource rngStateImage;
    ImageResource toneMapImage;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorPool imguiDescriptorPool = VK_NULL_HANDLE;

    CloudPathTracePass cloudPathTracePass;
    ToneMapPass toneMapPass;

    std::array<FrameResources, kFramesInFlight> frames{};
    std::uint32_t frameSlot = 0;

    VkSemaphore timelineSemaphore = VK_NULL_HANDLE;
    std::uint64_t timelineValue = 0;
    float timestampPeriod = 1.0f;

    bool imguiInitialized = false;
    bool swapchainDirty = false;

    bool hasPreviousParams = false;
    RenderParameters previousParams{};
    std::uint32_t accumulationFrameIndex = 0;
    std::uint64_t presentFrameIndex = 0;
    GpuTimingInfo timings{};

    bool init(GLFWwindow* inWindow);
    bool render(const RenderParameters& params);
    void shutdown();

    bool createInstance();
    bool pickPhysicalDevice();
    bool createDevice();
    bool createAllocator();
    bool createSwapchain();
    void destroySwapchain();
    bool createStorageImages();
    void destroyStorageImages();
    bool createDescriptors();
    bool createPipelines();
    bool createFrameResources();
    bool createTimelineSemaphore();
    bool initImGui();
    bool recreateSwapchain();

    VkShaderModule createShaderModuleFromSpv(const char* relativePath) const;
    void destroyDescriptors();
    void destroyPipelines();
    void destroyFrameResources();

    void fetchTimings(const FrameResources& frame);
};

Renderer::~Renderer() {
    shutdown();
}

bool Renderer::init(GLFWwindow* window) {
    if (m_impl != nullptr) {
        return true;
    }

    m_impl = new Impl();
    if (!m_impl->init(window)) {
        delete m_impl;
        m_impl = nullptr;
        return false;
    }

    return true;
}

void Renderer::beginUiFrame() {
    if (m_impl == nullptr || !m_impl->imguiInitialized) {
        return;
    }
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

bool Renderer::renderFrame(const RenderParameters& parameters) {
    if (m_impl == nullptr) {
        return false;
    }
    return m_impl->render(parameters);
}

void Renderer::shutdown() {
    if (m_impl == nullptr) {
        return;
    }
    m_impl->shutdown();
    delete m_impl;
    m_impl = nullptr;
}

std::uint32_t Renderer::frameIndex() const {
    return (m_impl == nullptr) ? 0u : m_impl->accumulationFrameIndex;
}

const GpuTimingInfo& Renderer::gpuTimings() const {
    static GpuTimingInfo kEmpty{};
    return (m_impl == nullptr) ? kEmpty : m_impl->timings;
}

bool Renderer::Impl::createInstance() {
    std::uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions == nullptr || glfwExtensionCount == 0) {
        VOX_LOGE("render") << "GLFW did not return required Vulkan instance extensions";
        return false;
    }

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Voxelsprout Compute Lab";
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.pEngineName = "None";
    appInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create Vulkan instance";
        return false;
    }

    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create Vulkan surface";
        return false;
    }

    return true;
}

bool Renderer::Impl::pickPhysicalDevice() {
    std::uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        VOX_LOGE("render") << "no Vulkan physical devices available";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (VkPhysicalDevice candidate : devices) {
        std::uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, queueFamilies.data());

        for (std::uint32_t family = 0; family < queueFamilyCount; ++family) {
            VkBool32 supportsPresent = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(candidate, family, surface, &supportsPresent);
            const bool supportsGraphics = (queueFamilies[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            const bool supportsCompute = (queueFamilies[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
            if (supportsPresent && supportsGraphics && supportsCompute) {
                physicalDevice = candidate;
                queueFamilyIndex = family;
                VkPhysicalDeviceProperties properties{};
                vkGetPhysicalDeviceProperties(physicalDevice, &properties);
                timestampPeriod = properties.limits.timestampPeriod;
                VOX_LOGI("render") << "using GPU: " << properties.deviceName;
                return true;
            }
        }
    }

    VOX_LOGE("render") << "no suitable queue family with graphics+compute+present";
    return false;
}

bool Renderer::Impl::createDevice() {
    const float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.synchronization2 = VK_TRUE;
    vulkan13Features.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.timelineSemaphore = VK_TRUE;
    vulkan12Features.pNext = &vulkan13Features;

    const std::array<const char*, 1> extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    createInfo.pNext = &vulkan12Features;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create logical device";
        return false;
    }

    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    return true;
}

bool Renderer::Impl::createAllocator() {
    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.instance = instance;
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;

    if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create VMA allocator";
        return false;
    }
    return true;
}

bool Renderer::Impl::createSwapchain() {
    VkSurfaceCapabilitiesKHR capabilities{};
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to query surface capabilities";
        return false;
    }

    std::uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());

    VkSurfaceFormatKHR chosenFormat = formats.front();
    for (const VkSurfaceFormatKHR format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosenFormat = format;
            break;
        }
    }

    std::uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());

    VkPresentModeKHR chosenPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    (void)presentModes;

    int fbWidth = 0;
    int fbHeight = 0;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    if (fbWidth <= 0 || fbHeight <= 0) {
        fbWidth = 1;
        fbHeight = 1;
    }

    VkExtent2D extent{};
    extent.width = std::clamp(
        static_cast<std::uint32_t>(fbWidth),
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width);
    extent.height = std::clamp(
        static_cast<std::uint32_t>(fbHeight),
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height);

    std::uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        imageCount = std::min(imageCount, capabilities.maxImageCount);
    }
    imageCount = std::max(imageCount, kFramesInFlight);

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = chosenFormat.format;
    createInfo.imageColorSpace = chosenFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = chosenPresentMode;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create swapchain";
        return false;
    }

    std::uint32_t swapchainImageCount = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCount, nullptr);
    swapchainImages.resize(swapchainImageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCount, swapchainImages.data());
    swapchainImageViews.resize(swapchainImageCount, VK_NULL_HANDLE);
    swapchainImageInitialized.assign(swapchainImageCount, false);

    swapchainFormat = chosenFormat.format;
    swapchainExtent = extent;

    for (std::size_t i = 0; i < swapchainImages.size(); ++i) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = swapchainImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapchainFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &swapchainImageViews[i]) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed to create swapchain image view";
            return false;
        }
    }

    return true;
}

void Renderer::Impl::destroySwapchain() {
    for (VkImageView imageView : swapchainImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(device, imageView, nullptr);
        }
    }
    swapchainImageViews.clear();
    swapchainImages.clear();
    swapchainImageInitialized.clear();

    if (swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
}

bool Renderer::Impl::createStorageImages() {
    auto createImage = [&](ImageResource& outImage, VkFormat format, VkImageUsageFlags usage) -> bool {
        outImage.width = swapchainExtent.width;
        outImage.height = swapchainExtent.height;
        outImage.format = format;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = format;
        imageInfo.extent.width = outImage.width;
        imageInfo.extent.height = outImage.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = usage;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        if (vmaCreateImage(allocator, &imageInfo, &allocInfo, &outImage.image, &outImage.allocation, nullptr) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed to create storage image";
            return false;
        }

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = outImage.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(device, &viewInfo, nullptr, &outImage.view) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed to create storage image view";
            return false;
        }

        outImage.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        return true;
    };

    if (!createImage(
            accumulationImage,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT)) {
        return false;
    }
    if (!createImage(
            rngStateImage,
            VK_FORMAT_R32_UINT,
            VK_IMAGE_USAGE_STORAGE_BIT)) {
        return false;
    }
    if (!createImage(
            toneMapImage,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) {
        return false;
    }

    return true;
}

void Renderer::Impl::destroyStorageImages() {
    auto destroyImage = [&](ImageResource& image) {
        if (image.view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, image.view, nullptr);
            image.view = VK_NULL_HANDLE;
        }
        if (image.image != VK_NULL_HANDLE && image.allocation != VK_NULL_HANDLE) {
            vmaDestroyImage(allocator, image.image, image.allocation);
            image.image = VK_NULL_HANDLE;
            image.allocation = VK_NULL_HANDLE;
        }
        image.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    destroyImage(accumulationImage);
    destroyImage(rngStateImage);
    destroyImage(toneMapImage);
}

VkShaderModule Renderer::Impl::createShaderModuleFromSpv(const char* relativePath) const {
    const std::string fullPath = std::string(VOXEL_PROJECT_SOURCE_DIR) + "/" + relativePath;
    const std::vector<char> bytes = loadBinaryFile(fullPath);
    if (bytes.empty() || (bytes.size() % 4) != 0) {
        VOX_LOGE("render") << "failed to read shader or invalid size: " << fullPath;
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = bytes.size();
    createInfo.pCode = reinterpret_cast<const std::uint32_t*>(bytes.data());

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create shader module: " << fullPath;
        return VK_NULL_HANDLE;
    }
    return module;
}

bool Renderer::Impl::createDescriptors() {
    std::array<VkDescriptorPoolSize, 1> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 5;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create descriptor pool";
        return false;
    }

    VkDescriptorPoolSize imguiPoolSize{};
    imguiPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    imguiPoolSize.descriptorCount = 128;

    VkDescriptorPoolCreateInfo imguiPoolInfo{};
    imguiPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    imguiPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    imguiPoolInfo.maxSets = 128;
    imguiPoolInfo.poolSizeCount = 1;
    imguiPoolInfo.pPoolSizes = &imguiPoolSize;

    if (vkCreateDescriptorPool(device, &imguiPoolInfo, nullptr, &imguiDescriptorPool) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create ImGui descriptor pool";
        return false;
    }

    std::array<VkDescriptorSetLayoutBinding, 2> cloudBindings{};
    cloudBindings[0].binding = 0;
    cloudBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    cloudBindings[0].descriptorCount = 1;
    cloudBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    cloudBindings[1].binding = 1;
    cloudBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    cloudBindings[1].descriptorCount = 1;
    cloudBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo cloudLayoutInfo{};
    cloudLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    cloudLayoutInfo.bindingCount = static_cast<std::uint32_t>(cloudBindings.size());
    cloudLayoutInfo.pBindings = cloudBindings.data();

    if (vkCreateDescriptorSetLayout(device, &cloudLayoutInfo, nullptr, &cloudPathTracePass.descriptorSetLayout) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create cloud descriptor set layout";
        return false;
    }

    std::array<VkDescriptorSetLayoutBinding, 2> toneBindings{};
    toneBindings[0].binding = 0;
    toneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    toneBindings[0].descriptorCount = 1;
    toneBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    toneBindings[1].binding = 1;
    toneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    toneBindings[1].descriptorCount = 1;
    toneBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo toneLayoutInfo{};
    toneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    toneLayoutInfo.bindingCount = static_cast<std::uint32_t>(toneBindings.size());
    toneLayoutInfo.pBindings = toneBindings.data();

    if (vkCreateDescriptorSetLayout(device, &toneLayoutInfo, nullptr, &toneMapPass.descriptorSetLayout) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create tone map descriptor set layout";
        return false;
    }

    std::array<VkDescriptorSetLayout, 1> cloudSetLayouts = {cloudPathTracePass.descriptorSetLayout};
    VkDescriptorSetAllocateInfo cloudAllocInfo{};
    cloudAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    cloudAllocInfo.descriptorPool = descriptorPool;
    cloudAllocInfo.descriptorSetCount = static_cast<std::uint32_t>(cloudSetLayouts.size());
    cloudAllocInfo.pSetLayouts = cloudSetLayouts.data();

    if (vkAllocateDescriptorSets(device, &cloudAllocInfo, &cloudPathTracePass.descriptorSet) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to allocate cloud descriptor set";
        return false;
    }

    std::array<VkDescriptorSetLayout, 1> toneSetLayouts = {toneMapPass.descriptorSetLayout};
    VkDescriptorSetAllocateInfo toneAllocInfo{};
    toneAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    toneAllocInfo.descriptorPool = descriptorPool;
    toneAllocInfo.descriptorSetCount = static_cast<std::uint32_t>(toneSetLayouts.size());
    toneAllocInfo.pSetLayouts = toneSetLayouts.data();

    if (vkAllocateDescriptorSets(device, &toneAllocInfo, &toneMapPass.descriptorSet) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to allocate tone map descriptor set";
        return false;
    }

    VkDescriptorImageInfo accumInfo{};
    accumInfo.imageView = accumulationImage.view;
    accumInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo rngInfo{};
    rngInfo.imageView = rngStateImage.view;
    rngInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo toneInfo{};
    toneInfo.imageView = toneMapImage.view;
    toneInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 4> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = cloudPathTracePass.descriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &accumInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = cloudPathTracePass.descriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &rngInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = toneMapPass.descriptorSet;
    writes[2].dstBinding = 0;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].descriptorCount = 1;
    writes[2].pImageInfo = &accumInfo;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = toneMapPass.descriptorSet;
    writes[3].dstBinding = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[3].descriptorCount = 1;
    writes[3].pImageInfo = &toneInfo;

    vkUpdateDescriptorSets(device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return true;
}

void Renderer::Impl::destroyDescriptors() {
    if (cloudPathTracePass.descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, cloudPathTracePass.descriptorSetLayout, nullptr);
        cloudPathTracePass.descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (toneMapPass.descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, toneMapPass.descriptorSetLayout, nullptr);
        toneMapPass.descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }
    if (imguiDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, imguiDescriptorPool, nullptr);
        imguiDescriptorPool = VK_NULL_HANDLE;
    }
}

bool Renderer::Impl::createPipelines() {
    VkShaderModule cloudModule = createShaderModuleFromSpv(kCloudShaderPath);
    VkShaderModule toneModule = createShaderModuleFromSpv(kToneMapShaderPath);
    if (cloudModule == VK_NULL_HANDLE || toneModule == VK_NULL_HANDLE) {
        if (cloudModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, cloudModule, nullptr);
        }
        if (toneModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, toneModule, nullptr);
        }
        return false;
    }

    VkPushConstantRange cloudPushRange{};
    cloudPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    cloudPushRange.offset = 0;
    cloudPushRange.size = sizeof(CameraPush);

    VkPipelineLayoutCreateInfo cloudLayoutInfo{};
    cloudLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    cloudLayoutInfo.setLayoutCount = 1;
    cloudLayoutInfo.pSetLayouts = &cloudPathTracePass.descriptorSetLayout;
    cloudLayoutInfo.pushConstantRangeCount = 1;
    cloudLayoutInfo.pPushConstantRanges = &cloudPushRange;

    if (vkCreatePipelineLayout(device, &cloudLayoutInfo, nullptr, &cloudPathTracePass.pipelineLayout) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create cloud pipeline layout";
        vkDestroyShaderModule(device, cloudModule, nullptr);
        vkDestroyShaderModule(device, toneModule, nullptr);
        return false;
    }

    VkPushConstantRange tonePushRange{};
    tonePushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tonePushRange.offset = 0;
    tonePushRange.size = sizeof(ToneMapPush);

    VkPipelineLayoutCreateInfo toneLayoutInfo{};
    toneLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    toneLayoutInfo.setLayoutCount = 1;
    toneLayoutInfo.pSetLayouts = &toneMapPass.descriptorSetLayout;
    toneLayoutInfo.pushConstantRangeCount = 1;
    toneLayoutInfo.pPushConstantRanges = &tonePushRange;

    if (vkCreatePipelineLayout(device, &toneLayoutInfo, nullptr, &toneMapPass.pipelineLayout) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create tone map pipeline layout";
        vkDestroyShaderModule(device, cloudModule, nullptr);
        vkDestroyShaderModule(device, toneModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo cloudStage{};
    cloudStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cloudStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cloudStage.module = cloudModule;
    cloudStage.pName = "main";

    VkComputePipelineCreateInfo cloudPipelineInfo{};
    cloudPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cloudPipelineInfo.stage = cloudStage;
    cloudPipelineInfo.layout = cloudPathTracePass.pipelineLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cloudPipelineInfo, nullptr, &cloudPathTracePass.pipeline) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create cloud compute pipeline";
        vkDestroyShaderModule(device, cloudModule, nullptr);
        vkDestroyShaderModule(device, toneModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo toneStage{};
    toneStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    toneStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    toneStage.module = toneModule;
    toneStage.pName = "main";

    VkComputePipelineCreateInfo tonePipelineInfo{};
    tonePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    tonePipelineInfo.stage = toneStage;
    tonePipelineInfo.layout = toneMapPass.pipelineLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &tonePipelineInfo, nullptr, &toneMapPass.pipeline) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create tone map compute pipeline";
        vkDestroyShaderModule(device, cloudModule, nullptr);
        vkDestroyShaderModule(device, toneModule, nullptr);
        return false;
    }

    vkDestroyShaderModule(device, cloudModule, nullptr);
    vkDestroyShaderModule(device, toneModule, nullptr);
    return true;
}

void Renderer::Impl::destroyPipelines() {
    if (cloudPathTracePass.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, cloudPathTracePass.pipeline, nullptr);
        cloudPathTracePass.pipeline = VK_NULL_HANDLE;
    }
    if (cloudPathTracePass.pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, cloudPathTracePass.pipelineLayout, nullptr);
        cloudPathTracePass.pipelineLayout = VK_NULL_HANDLE;
    }
    if (toneMapPass.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, toneMapPass.pipeline, nullptr);
        toneMapPass.pipeline = VK_NULL_HANDLE;
    }
    if (toneMapPass.pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, toneMapPass.pipelineLayout, nullptr);
        toneMapPass.pipelineLayout = VK_NULL_HANDLE;
    }
}

bool Renderer::Impl::createFrameResources() {
    for (FrameResources& frame : frames) {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &frame.commandPool) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed to create command pool";
            return false;
        }

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = frame.commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &allocInfo, &frame.commandBuffer) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed to allocate command buffer";
            return false;
        }

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &frame.imageAvailable) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed to create imageAvailable semaphore";
            return false;
        }
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &frame.renderComplete) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed to create renderComplete semaphore";
            return false;
        }

        VkQueryPoolCreateInfo queryInfo{};
        queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount = kTimestampCount;

        if (vkCreateQueryPool(device, &queryInfo, nullptr, &frame.timestampQueryPool) != VK_SUCCESS) {
            VOX_LOGE("render") << "failed to create timestamp query pool";
            return false;
        }
    }

    return true;
}

void Renderer::Impl::destroyFrameResources() {
    for (FrameResources& frame : frames) {
        if (frame.timestampQueryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(device, frame.timestampQueryPool, nullptr);
            frame.timestampQueryPool = VK_NULL_HANDLE;
        }
        if (frame.imageAvailable != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, frame.imageAvailable, nullptr);
            frame.imageAvailable = VK_NULL_HANDLE;
        }
        if (frame.renderComplete != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, frame.renderComplete, nullptr);
            frame.renderComplete = VK_NULL_HANDLE;
        }
        if (frame.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device, frame.commandPool, nullptr);
            frame.commandPool = VK_NULL_HANDLE;
        }
    }
}

bool Renderer::Impl::createTimelineSemaphore() {
    VkSemaphoreTypeCreateInfo typeInfo{};
    typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    typeInfo.initialValue = 0;

    VkSemaphoreCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    createInfo.pNext = &typeInfo;

    if (vkCreateSemaphore(device, &createInfo, nullptr, &timelineSemaphore) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to create timeline semaphore";
        return false;
    }

    timelineValue = 0;
    return true;
}

bool Renderer::Impl::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    if (!ImGui_ImplGlfw_InitForVulkan(window, true)) {
        VOX_LOGE("render") << "ImGui GLFW init failed";
        return false;
    }

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = instance;
    initInfo.PhysicalDevice = physicalDevice;
    initInfo.Device = device;
    initInfo.QueueFamily = queueFamilyIndex;
    initInfo.Queue = queue;
    initInfo.DescriptorPool = imguiDescriptorPool;
    initInfo.MinImageCount = static_cast<std::uint32_t>(swapchainImages.size());
    initInfo.ImageCount = static_cast<std::uint32_t>(swapchainImages.size());
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainFormat;

    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        VOX_LOGE("render") << "ImGui Vulkan init failed";
        return false;
    }

    if (!ImGui_ImplVulkan_CreateFontsTexture()) {
        VOX_LOGE("render") << "ImGui font texture upload failed";
        return false;
    }

    imguiInitialized = true;
    return true;
}

void Renderer::Impl::fetchTimings(const FrameResources& frame) {
    if (frame.submittedTimelineValue == 0) {
        return;
    }

    std::array<std::uint64_t, kTimestampCount> values{};
    const VkResult result = vkGetQueryPoolResults(
        device,
        frame.timestampQueryPool,
        0,
        kTimestampCount,
        sizeof(values),
        values.data(),
        sizeof(std::uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (result != VK_SUCCESS) {
        return;
    }

    const float nsToMs = timestampPeriod * 1e-6f;
    timings.cloudPathTraceMs = static_cast<float>(values[kTimestampCloudEnd] - values[kTimestampFrameStart]) * nsToMs;
    timings.toneMapMs = static_cast<float>(values[kTimestampToneEnd] - values[kTimestampCloudEnd]) * nsToMs;
    timings.totalMs = static_cast<float>(values[kTimestampFrameEnd] - values[kTimestampFrameStart]) * nsToMs;
}

bool Renderer::Impl::recreateSwapchain() {
    vkDeviceWaitIdle(device);

    if (imguiInitialized) {
        ImGui_ImplVulkan_SetMinImageCount(static_cast<std::uint32_t>(swapchainImages.size()));
    }

    destroyStorageImages();
    destroySwapchain();

    if (!createSwapchain()) {
        return false;
    }
    if (!createStorageImages()) {
        return false;
    }

    destroyDescriptors();
    if (!createDescriptors()) {
        return false;
    }

    if (imguiInitialized) {
        ImGui_ImplVulkan_SetMinImageCount(static_cast<std::uint32_t>(swapchainImages.size()));
    }

    accumulationFrameIndex = 0;
    hasPreviousParams = false;
    return true;
}

bool Renderer::Impl::init(GLFWwindow* inWindow) {
    window = inWindow;

    if (!createInstance()) {
        return false;
    }
    if (!pickPhysicalDevice()) {
        return false;
    }
    if (!createDevice()) {
        return false;
    }
    if (!createAllocator()) {
        return false;
    }
    if (!createSwapchain()) {
        return false;
    }
    if (!createStorageImages()) {
        return false;
    }
    if (!createDescriptors()) {
        return false;
    }
    if (!createPipelines()) {
        return false;
    }
    if (!createFrameResources()) {
        return false;
    }
    if (!createTimelineSemaphore()) {
        return false;
    }
    if (!initImGui()) {
        return false;
    }

    return true;
}

bool Renderer::Impl::render(const RenderParameters& params) {
    int fbWidth = 0;
    int fbHeight = 0;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    if (fbWidth <= 0 || fbHeight <= 0) {
        return true;
    }

    if (static_cast<std::uint32_t>(fbWidth) != swapchainExtent.width || static_cast<std::uint32_t>(fbHeight) != swapchainExtent.height) {
        if (!recreateSwapchain()) {
            return false;
        }
    }

    FrameResources& frame = frames[frameSlot];

    if (frame.submittedTimelineValue > 0) {
        VkSemaphoreWaitInfo waitInfo{};
        waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &timelineSemaphore;
        waitInfo.pValues = &frame.submittedTimelineValue;
        vkWaitSemaphores(device, &waitInfo, std::numeric_limits<std::uint64_t>::max());
        fetchTimings(frame);
    }

    std::uint32_t imageIndex = 0;
    VkResult acquireResult = vkAcquireNextImageKHR(
        device,
        swapchain,
        std::numeric_limits<std::uint64_t>::max(),
        frame.imageAvailable,
        VK_NULL_HANDLE,
        &imageIndex);

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        return recreateSwapchain();
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        VOX_LOGE("render") << "failed to acquire swapchain image";
        return false;
    }

    const bool paramsChanged = hasPreviousParams ? paramsDiffer(params, previousParams) : true;
    const bool resetAccumulation = paramsChanged || !params.enableAccumulation || params.forceReset;
    const std::uint32_t cloudInterval = std::max(1u, params.cloudUpdateInterval);

    if (resetAccumulation) {
        accumulationFrameIndex = 0;
    }
    const std::uint32_t maxAccumulationSamples = std::max(1u, params.maxAccumulationSamples);
    const bool accumulationComplete =
        params.enableAccumulation && (accumulationFrameIndex >= maxAccumulationSamples);
    const bool runCloudPassThisFrame =
        !accumulationComplete && (resetAccumulation || ((presentFrameIndex % cloudInterval) == 0u));

    previousParams = params;
    hasPreviousParams = true;

    vkResetCommandPool(device, frame.commandPool, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(frame.commandBuffer, &beginInfo) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to begin command buffer";
        return false;
    }

    vkCmdResetQueryPool(frame.commandBuffer, frame.timestampQueryPool, 0, kTimestampCount);
    vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame.timestampQueryPool, kTimestampFrameStart);

    std::array<VkImageMemoryBarrier2, 4> startupBarriers{};
    std::uint32_t startupBarrierCount = 0;

    const VkPipelineStageFlags2 accumSrcStage =
        (accumulationImage.layout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    const VkAccessFlags2 accumSrcAccess =
        (accumulationImage.layout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_ACCESS_2_NONE : VK_ACCESS_2_MEMORY_WRITE_BIT;
    startupBarriers[startupBarrierCount++] = imageBarrier(
        accumulationImage.image,
        accumulationImage.layout,
        VK_IMAGE_LAYOUT_GENERAL,
        accumSrcStage,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        accumSrcAccess,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    const VkPipelineStageFlags2 toneSrcStage =
        (toneMapImage.layout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    const VkAccessFlags2 toneSrcAccess =
        (toneMapImage.layout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_ACCESS_2_NONE : VK_ACCESS_2_MEMORY_WRITE_BIT;
    startupBarriers[startupBarrierCount++] = imageBarrier(
        toneMapImage.image,
        toneMapImage.layout,
        VK_IMAGE_LAYOUT_GENERAL,
        toneSrcStage,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        toneSrcAccess,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    const VkPipelineStageFlags2 rngSrcStage =
        (rngStateImage.layout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    const VkAccessFlags2 rngSrcAccess =
        (rngStateImage.layout == VK_IMAGE_LAYOUT_UNDEFINED) ? VK_ACCESS_2_NONE : VK_ACCESS_2_MEMORY_WRITE_BIT;
    startupBarriers[startupBarrierCount++] = imageBarrier(
        rngStateImage.image,
        rngStateImage.layout,
        VK_IMAGE_LAYOUT_GENERAL,
        rngSrcStage,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        rngSrcAccess,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    VkDependencyInfo startupDep{};
    startupDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    startupDep.imageMemoryBarrierCount = startupBarrierCount;
    startupDep.pImageMemoryBarriers = startupBarriers.data();
    vkCmdPipelineBarrier2(frame.commandBuffer, &startupDep);

    accumulationImage.layout = VK_IMAGE_LAYOUT_GENERAL;
    toneMapImage.layout = VK_IMAGE_LAYOUT_GENERAL;
    rngStateImage.layout = VK_IMAGE_LAYOUT_GENERAL;

    const std::uint32_t dispatchX = (swapchainExtent.width + 7u) / 8u;
    const std::uint32_t dispatchY = (swapchainExtent.height + 7u) / 8u;
    if (runCloudPassThisFrame) {
        vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cloudPathTracePass.pipeline);
        vkCmdBindDescriptorSets(
            frame.commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            cloudPathTracePass.pipelineLayout,
            0,
            1,
            &cloudPathTracePass.descriptorSet,
            0,
            nullptr);

        const voxelsprout::core::Vec3 forward = params.camera.forward();
        const voxelsprout::core::Vec3 right = params.camera.right();
        const voxelsprout::core::Vec3 up = voxelsprout::core::normalize(voxelsprout::core::cross(right, forward));
        voxelsprout::core::Vec3 sunDir = voxelsprout::core::normalize(params.scene.sun.direction);
        if (voxelsprout::core::length(sunDir) <= 0.0f) {
            sunDir = voxelsprout::core::normalize(voxelsprout::core::Vec3{1.0f, 1.0f, 0.5f});
        }

        CameraPush cloudPush{};
        cloudPush.cameraPositionFov[0] = params.camera.position.x;
        cloudPush.cameraPositionFov[1] = params.camera.position.y;
        cloudPush.cameraPositionFov[2] = params.camera.position.z;
        cloudPush.cameraPositionFov[3] = params.camera.fovDegrees;
        cloudPush.cameraForward[0] = forward.x;
        cloudPush.cameraForward[1] = forward.y;
        cloudPush.cameraForward[2] = forward.z;
        cloudPush.cameraForward[3] = 0.0f;
        cloudPush.cameraRight[0] = right.x;
        cloudPush.cameraRight[1] = right.y;
        cloudPush.cameraRight[2] = right.z;
        cloudPush.cameraRight[3] = 0.0f;
        cloudPush.cameraUp[0] = up.x;
        cloudPush.cameraUp[1] = up.y;
        cloudPush.cameraUp[2] = up.z;
        cloudPush.cameraUp[3] = 0.0f;
        cloudPush.sunDirectionIntensity[0] = sunDir.x;
        cloudPush.sunDirectionIntensity[1] = sunDir.y;
        cloudPush.sunDirectionIntensity[2] = sunDir.z;
        cloudPush.sunDirectionIntensity[3] = params.scene.sun.intensity;
        cloudPush.mediumParams[0] = params.scene.volume.densityScale;
        cloudPush.mediumParams[1] = params.scene.volume.anisotropyG;
        cloudPush.mediumParams[2] = params.scene.volume.albedo;
        cloudPush.mediumParams[3] = 0.0f;
        cloudPush.cloudShapeParams[0] = params.scene.volume.macroScale;
        cloudPush.cloudShapeParams[1] = params.scene.volume.detailScale;
        cloudPush.cloudShapeParams[2] = params.scene.volume.densityCutoff;
        cloudPush.cloudShapeParams[3] = params.scene.volume.chunkiness;
        cloudPush.cloudProfileParams[0] = params.scene.volume.coverage;
        cloudPush.cloudProfileParams[1] = params.scene.volume.weatherScale;
        cloudPush.cloudProfileParams[2] = params.scene.volume.cloudBase;
        cloudPush.cloudProfileParams[3] = params.scene.volume.cloudTop;
        cloudPush.cloudWarpParams[0] = params.scene.volume.warpStrength;
        cloudPush.cloudWarpParams[1] = params.scene.volume.erosionStrength;
        cloudPush.cloudWarpParams[2] = 0.0f;
        cloudPush.cloudWarpParams[3] = 0.0f;
        cloudPush.cloudLightParams[0] = params.scene.volume.brightnessBoost;
        cloudPush.cloudLightParams[1] = params.scene.volume.ambientLift;
        cloudPush.cloudLightParams[2] = static_cast<float>(params.scene.volume.maxBounces);
        cloudPush.cloudLightParams[3] = 0.0f;
        cloudPush.frameParams[0] = static_cast<float>(swapchainExtent.width);
        cloudPush.frameParams[1] = static_cast<float>(swapchainExtent.height);
        cloudPush.frameParams[2] = static_cast<float>(accumulationFrameIndex);
        cloudPush.frameParams[3] = (resetAccumulation || !params.enableAccumulation) ? 1.0f : 0.0f;

        vkCmdPushConstants(
            frame.commandBuffer,
            cloudPathTracePass.pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(CameraPush),
            &cloudPush);

        vkCmdDispatch(frame.commandBuffer, dispatchX, dispatchY, 1);
        vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, frame.timestampQueryPool, kTimestampCloudEnd);

        VkImageMemoryBarrier2 cloudToToneBarrier = imageBarrier(
            accumulationImage.image,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

        VkDependencyInfo cloudToToneDep{};
        cloudToToneDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        cloudToToneDep.imageMemoryBarrierCount = 1;
        cloudToToneDep.pImageMemoryBarriers = &cloudToToneBarrier;
        vkCmdPipelineBarrier2(frame.commandBuffer, &cloudToToneDep);

        vkCmdBindPipeline(frame.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, toneMapPass.pipeline);
        vkCmdBindDescriptorSets(
            frame.commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            toneMapPass.pipelineLayout,
            0,
            1,
            &toneMapPass.descriptorSet,
            0,
            nullptr);

        ToneMapPush tonePush{};
        tonePush.exposure = params.exposure;
        tonePush.width = swapchainExtent.width;
        tonePush.height = swapchainExtent.height;
        tonePush.sampleCount = params.enableAccumulation ? (accumulationFrameIndex + 1u) : 1u;

        vkCmdPushConstants(
            frame.commandBuffer,
            toneMapPass.pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(ToneMapPush),
            &tonePush);
        vkCmdDispatch(frame.commandBuffer, dispatchX, dispatchY, 1);
        vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, frame.timestampQueryPool, kTimestampToneEnd);
    } else {
        vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame.timestampQueryPool, kTimestampCloudEnd);
        vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, frame.timestampQueryPool, kTimestampToneEnd);
    }

    const VkImageLayout oldSwapchainLayout = swapchainImageInitialized[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED;

    std::array<VkImageMemoryBarrier2, 2> copyPrep{};
    copyPrep[0] = imageBarrier(
        toneMapImage.image,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_ACCESS_2_TRANSFER_READ_BIT);
    copyPrep[1] = imageBarrier(
        swapchainImages[imageIndex],
        oldSwapchainLayout,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_NONE,
        VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkDependencyInfo copyPrepDep{};
    copyPrepDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    copyPrepDep.imageMemoryBarrierCount = static_cast<std::uint32_t>(copyPrep.size());
    copyPrepDep.pImageMemoryBarriers = copyPrep.data();
    vkCmdPipelineBarrier2(frame.commandBuffer, &copyPrepDep);

    toneMapImage.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkImageBlit blit{};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.layerCount = 1;
    blit.srcOffsets[1].x = static_cast<std::int32_t>(swapchainExtent.width);
    blit.srcOffsets[1].y = static_cast<std::int32_t>(swapchainExtent.height);
    blit.srcOffsets[1].z = 1;
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.layerCount = 1;
    blit.dstOffsets[1].x = static_cast<std::int32_t>(swapchainExtent.width);
    blit.dstOffsets[1].y = static_cast<std::int32_t>(swapchainExtent.height);
    blit.dstOffsets[1].z = 1;

    vkCmdBlitImage(
        frame.commandBuffer,
        toneMapImage.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &blit,
        VK_FILTER_NEAREST);

    VkImageMemoryBarrier2 toUiBarrier = imageBarrier(
        swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkDependencyInfo toUiDep{};
    toUiDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    toUiDep.imageMemoryBarrierCount = 1;
    toUiDep.pImageMemoryBarriers = &toUiBarrier;
    vkCmdPipelineBarrier2(frame.commandBuffer, &toUiDep);

    ImGui::Render();
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = swapchainImageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = swapchainExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(frame.commandBuffer, &renderingInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), frame.commandBuffer);
    vkCmdEndRendering(frame.commandBuffer);

    VkImageMemoryBarrier2 toPresentBarrier = imageBarrier(
        swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_2_NONE);

    VkDependencyInfo toPresentDep{};
    toPresentDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    toPresentDep.imageMemoryBarrierCount = 1;
    toPresentDep.pImageMemoryBarriers = &toPresentBarrier;
    vkCmdPipelineBarrier2(frame.commandBuffer, &toPresentDep);

    swapchainImageInitialized[imageIndex] = true;

    vkCmdWriteTimestamp2(frame.commandBuffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, frame.timestampQueryPool, kTimestampFrameEnd);

    if (vkEndCommandBuffer(frame.commandBuffer) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to end command buffer";
        return false;
    }

    const std::uint64_t signalTimelineValue = ++timelineValue;

    VkSemaphoreSubmitInfo waitSemaphore{};
    waitSemaphore.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    waitSemaphore.semaphore = frame.imageAvailable;
    waitSemaphore.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    waitSemaphore.deviceIndex = 0;

    VkCommandBufferSubmitInfo commandBufferInfo{};
    commandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    commandBufferInfo.commandBuffer = frame.commandBuffer;

    std::array<VkSemaphoreSubmitInfo, 2> signalSemaphores{};
    signalSemaphores[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemaphores[0].semaphore = frame.renderComplete;
    signalSemaphores[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    signalSemaphores[1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemaphores[1].semaphore = timelineSemaphore;
    signalSemaphores[1].value = signalTimelineValue;
    signalSemaphores[1].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitSemaphore;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    submitInfo.signalSemaphoreInfoCount = static_cast<std::uint32_t>(signalSemaphores.size());
    submitInfo.pSignalSemaphoreInfos = signalSemaphores.data();

    if (vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to submit queue";
        return false;
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &frame.renderComplete;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain;
    presentInfo.pImageIndices = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(queue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        if (!recreateSwapchain()) {
            return false;
        }
    } else if (presentResult != VK_SUCCESS) {
        VOX_LOGE("render") << "failed to present";
        return false;
    }

    frame.submittedTimelineValue = signalTimelineValue;

    if (runCloudPassThisFrame && params.enableAccumulation) {
        accumulationFrameIndex += 1;
    }
    presentFrameIndex += 1;

    frameSlot = (frameSlot + 1u) % kFramesInFlight;
    return true;
}

void Renderer::Impl::shutdown() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
    }

    if (imguiInitialized) {
        ImGui_ImplVulkan_DestroyFontsTexture();
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        imguiInitialized = false;
    }

    if (timelineSemaphore != VK_NULL_HANDLE) {
        vkDestroySemaphore(device, timelineSemaphore, nullptr);
        timelineSemaphore = VK_NULL_HANDLE;
    }

    destroyFrameResources();
    destroyPipelines();
    destroyDescriptors();
    destroyStorageImages();
    destroySwapchain();

    if (allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator);
        allocator = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }

    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

} // namespace voxelsprout::render
