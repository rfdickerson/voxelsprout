#include "render/Renderer.hpp"

#include <GLFW/glfw3.h>
#include "math/Math.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

namespace render {

namespace {

constexpr std::array<const char*, 1> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};
constexpr std::array<const char*, 5> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_MAINTENANCE_4_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,
    VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
};

struct Vertex {
    float x;
    float y;
};

constexpr std::array<Vertex, 6> kGroundQuadVertices = {
    Vertex{-0.8f, -0.8f},
    Vertex{0.8f, -0.8f},
    Vertex{0.8f, 0.2f},
    Vertex{-0.8f, -0.8f},
    Vertex{0.8f, 0.2f},
    Vertex{-0.8f, 0.2f},
};

struct alignas(16) CameraUniform {
    float mvp[16];
};

math::Matrix4 transpose(const math::Matrix4& matrix) {
    math::Matrix4 result{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            result(row, col) = matrix(col, row);
        }
    }
    return result;
}

math::Matrix4 perspectiveVulkan(float fovYRadians, float aspectRatio, float nearPlane, float farPlane) {
    math::Matrix4 result{};
    for (int i = 0; i < 16; ++i) {
        result.m[i] = 0.0f;
    }

    const float f = 1.0f / std::tan(fovYRadians * 0.5f);
    result(0, 0) = f / aspectRatio;
    result(1, 1) = -f;
    result(2, 2) = farPlane / (nearPlane - farPlane);
    result(2, 3) = (farPlane * nearPlane) / (nearPlane - farPlane);
    result(3, 2) = -1.0f;
    return result;
}

math::Matrix4 lookAt(const math::Vector3& eye, const math::Vector3& target, const math::Vector3& up) {
    const math::Vector3 forward = math::normalize(target - eye);
    const math::Vector3 right = math::normalize(math::cross(forward, up));
    const math::Vector3 cameraUp = math::cross(right, forward);

    math::Matrix4 view = math::Matrix4::identity();
    view(0, 0) = right.x;
    view(0, 1) = right.y;
    view(0, 2) = right.z;
    view(0, 3) = -math::dot(right, eye);

    view(1, 0) = cameraUp.x;
    view(1, 1) = cameraUp.y;
    view(1, 2) = cameraUp.z;
    view(1, 3) = -math::dot(cameraUp, eye);

    view(2, 0) = -forward.x;
    view(2, 1) = -forward.y;
    view(2, 2) = -forward.z;
    view(2, 3) = math::dot(forward, eye);
    return view;
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

void transitionImageLayout(
    VkCommandBuffer commandBuffer,
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    VkPipelineStageFlags2 srcStageMask,
    VkAccessFlags2 srcAccessMask,
    VkPipelineStageFlags2 dstStageMask,
    VkAccessFlags2 dstAccessMask
) {
    VkImageMemoryBarrier2 imageBarrier{};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    imageBarrier.srcStageMask = srcStageMask;
    imageBarrier.srcAccessMask = srcAccessMask;
    imageBarrier.dstStageMask = dstStageMask;
    imageBarrier.dstAccessMask = dstAccessMask;
    imageBarrier.oldLayout = oldLayout;
    imageBarrier.newLayout = newLayout;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = image;
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
}

// Embedded shaders keep this bootstrap renderer self-contained.
// Future asset/shader systems can replace this with a shader pipeline.
static const uint32_t kVertShaderSpirv[] = {
0x07230203, 0x00010000, 0x000d000b, 0x00000023, 0x00000000, 0x00020011,
0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0007000f, 0x00000000,
0x00000004, 0x6e69616d, 0x00000000, 0x0000000d, 0x00000019, 0x00030003,
0x00000002, 0x000001c2, 0x000a0004, 0x475f4c47, 0x4c474f4f, 0x70635f45,
0x74735f70, 0x5f656c79, 0x656e696c, 0x7269645f, 0x69746365, 0x00006576,
0x00080004, 0x475f4c47, 0x4c474f4f, 0x6e695f45, 0x64756c63, 0x69645f65,
0x74636572, 0x00657669, 0x00040005, 0x00000004, 0x6e69616d, 0x00000000,
0x00060005, 0x0000000b, 0x505f6c67, 0x65567265, 0x78657472, 0x00000000,
0x00060006, 0x0000000b, 0x00000000, 0x505f6c67, 0x7469736f, 0x006e6f69,
0x00070006, 0x0000000b, 0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953,
0x00000000, 0x00070006, 0x0000000b, 0x00000002, 0x435f6c67, 0x4470696c,
0x61747369, 0x0065636e, 0x00070006, 0x0000000b, 0x00000003, 0x435f6c67,
0x446c6c75, 0x61747369, 0x0065636e, 0x00030005, 0x0000000d, 0x00000000,
0x00060005, 0x00000011, 0x656d6143, 0x6e556172, 0x726f6669, 0x0000006d,
0x00040006, 0x00000011, 0x00000000, 0x0070766d, 0x00040005, 0x00000013,
0x656d6163, 0x00006172, 0x00050005, 0x00000019, 0x6f506e69, 0x69746973,
0x00006e6f, 0x00050048, 0x0000000b, 0x00000000, 0x0000000b, 0x00000000,
0x00050048, 0x0000000b, 0x00000001, 0x0000000b, 0x00000001, 0x00050048,
0x0000000b, 0x00000002, 0x0000000b, 0x00000003, 0x00050048, 0x0000000b,
0x00000003, 0x0000000b, 0x00000004, 0x00030047, 0x0000000b, 0x00000002,
0x00040048, 0x00000011, 0x00000000, 0x00000005, 0x00050048, 0x00000011,
0x00000000, 0x00000023, 0x00000000, 0x00050048, 0x00000011, 0x00000000,
0x00000007, 0x00000010, 0x00030047, 0x00000011, 0x00000002, 0x00040047,
0x00000013, 0x00000022, 0x00000000, 0x00040047, 0x00000013, 0x00000021,
0x00000000, 0x00040047, 0x00000019, 0x0000001e, 0x00000000, 0x00020013,
0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040015,
0x00000008, 0x00000020, 0x00000000, 0x0004002b, 0x00000008, 0x00000009,
0x00000001, 0x0004001c, 0x0000000a, 0x00000006, 0x00000009, 0x0006001e,
0x0000000b, 0x00000007, 0x00000006, 0x0000000a, 0x0000000a, 0x00040020,
0x0000000c, 0x00000003, 0x0000000b, 0x0004003b, 0x0000000c, 0x0000000d,
0x00000003, 0x00040015, 0x0000000e, 0x00000020, 0x00000001, 0x0004002b,
0x0000000e, 0x0000000f, 0x00000000, 0x00040018, 0x00000010, 0x00000007,
0x00000004, 0x0003001e, 0x00000011, 0x00000010, 0x00040020, 0x00000012,
0x00000002, 0x00000011, 0x0004003b, 0x00000012, 0x00000013, 0x00000002,
0x00040020, 0x00000014, 0x00000002, 0x00000010, 0x00040017, 0x00000017,
0x00000006, 0x00000002, 0x00040020, 0x00000018, 0x00000001, 0x00000017,
0x0004003b, 0x00000018, 0x00000019, 0x00000001, 0x0004002b, 0x00000006,
0x0000001b, 0x00000000, 0x0004002b, 0x00000006, 0x0000001c, 0x3f800000,
0x00040020, 0x00000021, 0x00000003, 0x00000007, 0x00050036, 0x00000002,
0x00000004, 0x00000000, 0x00000003, 0x000200f8, 0x00000005, 0x00050041,
0x00000014, 0x00000015, 0x00000013, 0x0000000f, 0x0004003d, 0x00000010,
0x00000016, 0x00000015, 0x0004003d, 0x00000017, 0x0000001a, 0x00000019,
0x00050051, 0x00000006, 0x0000001d, 0x0000001a, 0x00000000, 0x00050051,
0x00000006, 0x0000001e, 0x0000001a, 0x00000001, 0x00070050, 0x00000007,
0x0000001f, 0x0000001d, 0x0000001e, 0x0000001b, 0x0000001c, 0x00050091,
0x00000007, 0x00000020, 0x00000016, 0x0000001f, 0x00050041, 0x00000021,
0x00000022, 0x0000000d, 0x0000000f, 0x0003003e, 0x00000022, 0x00000020,
0x000100fd, 0x00010038, 
};

static const uint32_t kFragShaderSpirv[] = {
0x07230203, 0x00010000, 0x000d000b, 0x0000000f, 0x00000000, 0x00020011,
0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000004,
0x00000004, 0x6e69616d, 0x00000000, 0x00000009, 0x00030010, 0x00000004,
0x00000007, 0x00030003, 0x00000002, 0x000001c2, 0x000a0004, 0x475f4c47,
0x4c474f4f, 0x70635f45, 0x74735f70, 0x5f656c79, 0x656e696c, 0x7269645f,
0x69746365, 0x00006576, 0x00080004, 0x475f4c47, 0x4c474f4f, 0x6e695f45,
0x64756c63, 0x69645f65, 0x74636572, 0x00657669, 0x00040005, 0x00000004,
0x6e69616d, 0x00000000, 0x00050005, 0x00000009, 0x4374756f, 0x726f6c6f,
0x00000000, 0x00040047, 0x00000009, 0x0000001e, 0x00000000, 0x00020013,
0x00000002, 0x00030021, 0x00000003, 0x00000002, 0x00030016, 0x00000006,
0x00000020, 0x00040017, 0x00000007, 0x00000006, 0x00000004, 0x00040020,
0x00000008, 0x00000003, 0x00000007, 0x0004003b, 0x00000008, 0x00000009,
0x00000003, 0x0004002b, 0x00000006, 0x0000000a, 0x3e3851ec, 0x0004002b,
0x00000006, 0x0000000b, 0x3f3ae148, 0x0004002b, 0x00000006, 0x0000000c,
0x3e9eb852, 0x0004002b, 0x00000006, 0x0000000d, 0x3f800000, 0x0007002c,
0x00000007, 0x0000000e, 0x0000000a, 0x0000000b, 0x0000000c, 0x0000000d,
0x00050036, 0x00000002, 0x00000004, 0x00000000, 0x00000003, 0x000200f8,
0x00000005, 0x0003003e, 0x00000009, 0x0000000e, 0x000100fd, 0x00010038,
};

struct QueueFamilyChoice {
    std::optional<uint32_t> graphicsAndPresent;

    [[nodiscard]] bool valid() const {
        return graphicsAndPresent.has_value();
    }
};

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

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
    std::cerr << "[render] " << context << " failed: "
              << vkResultName(result) << " (" << static_cast<int>(result) << ")\n";
}

bool isLayerAvailable(const char* layerName) {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> layers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layers.data());

    for (const VkLayerProperties& layer : layers) {
        if (std::strcmp(layer.layerName, layerName) == 0) {
            return true;
        }
    }
    return false;
}

QueueFamilyChoice findQueueFamily(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    QueueFamilyChoice choice;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &familyCount, families.data());

    for (uint32_t familyIndex = 0; familyIndex < familyCount; ++familyIndex) {
        const bool hasGraphics = (families[familyIndex].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        if (!hasGraphics) {
            continue;
        }

        VkBool32 hasPresent = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, familyIndex, surface, &hasPresent);
        if (hasPresent == VK_TRUE) {
            choice.graphicsAndPresent = familyIndex;
            return choice;
        }
    }

    return choice;
}

bool hasRequiredDeviceExtensions(VkPhysicalDevice physicalDevice) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data());

    for (const char* required : kDeviceExtensions) {
        bool found = false;
        for (const VkExtensionProperties& available : extensions) {
            if (std::strcmp(required, available.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    return true;
}

SwapchainSupport querySwapchainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    SwapchainSupport support;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &support.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    support.formats.resize(formatCount);
    if (formatCount > 0) {
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, support.formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    support.presentModes.resize(presentModeCount);
    if (presentModeCount > 0) {
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, support.presentModes.data());
    }

    return support;
}

VkSurfaceFormatKHR chooseSwapchainFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const VkSurfaceFormatKHR format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return formats.front();
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& presentModes) {
    for (const VkPresentModeKHR presentMode : presentModes) {
        if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return presentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseExtent(GLFWwindow* window, const VkSurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D extent{};
    extent.width = std::clamp(
        static_cast<uint32_t>(std::max(width, 1)),
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width
    );
    extent.height = std::clamp(
        static_cast<uint32_t>(std::max(height, 1)),
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height
    );
    return extent;
}

} // namespace

bool Renderer::init(GLFWwindow* window) {
    std::cerr << "[render] init begin\n";
    m_window = window;
    if (m_window == nullptr) {
        std::cerr << "[render] init failed: window is null\n";
        return false;
    }

    if (glfwVulkanSupported() == GLFW_FALSE) {
        std::cerr << "[render] init failed: glfwVulkanSupported returned false\n";
        return false;
    }

    if (!createInstance()) {
        std::cerr << "[render] init failed at createInstance\n";
        shutdown();
        return false;
    }
    if (!createSurface()) {
        std::cerr << "[render] init failed at createSurface\n";
        shutdown();
        return false;
    }
    if (!pickPhysicalDevice()) {
        std::cerr << "[render] init failed at pickPhysicalDevice\n";
        shutdown();
        return false;
    }
    if (!createLogicalDevice()) {
        std::cerr << "[render] init failed at createLogicalDevice\n";
        shutdown();
        return false;
    }
    if (!createTimelineSemaphore()) {
        std::cerr << "[render] init failed at createTimelineSemaphore\n";
        shutdown();
        return false;
    }
    if (!m_bufferAllocator.init(m_physicalDevice, m_device)) {
        std::cerr << "[render] init failed at buffer allocator init\n";
        shutdown();
        return false;
    }
    if (!createUploadRingBuffer()) {
        std::cerr << "[render] init failed at createUploadRingBuffer\n";
        shutdown();
        return false;
    }
    if (!createSwapchain()) {
        std::cerr << "[render] init failed at createSwapchain\n";
        shutdown();
        return false;
    }
    if (!createDescriptorResources()) {
        std::cerr << "[render] init failed at createDescriptorResources\n";
        shutdown();
        return false;
    }
    if (!createGraphicsPipeline()) {
        std::cerr << "[render] init failed at createGraphicsPipeline\n";
        shutdown();
        return false;
    }
    if (!createVertexBuffer()) {
        std::cerr << "[render] init failed at createVertexBuffer\n";
        shutdown();
        return false;
    }
    if (!createFrameResources()) {
        std::cerr << "[render] init failed at createFrameResources\n";
        shutdown();
        return false;
    }

    std::cerr << "[render] init complete\n";
    return true;
}

bool Renderer::createInstance() {
#ifndef NDEBUG
    const bool enableValidationLayers = isLayerAvailable(kValidationLayers[0]);
#else
    const bool enableValidationLayers = false;
#endif
    std::cerr << "[render] createInstance (validation="
              << (enableValidationLayers ? "on" : "off") << ")\n";

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (glfwExtensions == nullptr || glfwExtensionCount == 0) {
        std::cerr << "[render] no GLFW Vulkan instance extensions available\n";
        return false;
    }

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "voxel_factory_toy";
    applicationInfo.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    applicationInfo.pEngineName = "none";
    applicationInfo.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    applicationInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &applicationInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(kValidationLayers.size());
        createInfo.ppEnabledLayerNames = kValidationLayers.data();
    }

    const VkResult result = vkCreateInstance(&createInfo, nullptr, &m_instance);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateInstance", result);
        return false;
    }
    return true;
}

bool Renderer::createSurface() {
    const VkResult result = glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface);
    if (result != VK_SUCCESS) {
        logVkFailure("glfwCreateWindowSurface", result);
        return false;
    }
    return true;
}

bool Renderer::pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "[render] no Vulkan physical devices found\n";
        return false;
    }
    std::cerr << "[render] physical devices found: " << deviceCount << "\n";

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        std::cerr << "[render] evaluating GPU: " << properties.deviceName
                  << ", apiVersion=" << VK_VERSION_MAJOR(properties.apiVersion) << "."
                  << VK_VERSION_MINOR(properties.apiVersion) << "."
                  << VK_VERSION_PATCH(properties.apiVersion) << "\n";
        if (properties.apiVersion < VK_API_VERSION_1_3) {
            std::cerr << "[render] skip GPU: Vulkan 1.3 required\n";
            continue;
        }
        if ((properties.limits.framebufferColorSampleCounts & VK_SAMPLE_COUNT_4_BIT) == 0) {
            std::cerr << "[render] skip GPU: 4x MSAA color attachments not supported\n";
            continue;
        }

        const QueueFamilyChoice queueFamily = findQueueFamily(candidate, m_surface);
        if (!queueFamily.valid()) {
            std::cerr << "[render] skip GPU: no graphics+present queue family\n";
            continue;
        }
        if (!hasRequiredDeviceExtensions(candidate)) {
            std::cerr << "[render] skip GPU: missing required device extensions\n";
            continue;
        }

        const SwapchainSupport swapchainSupport = querySwapchainSupport(candidate, m_surface);
        if (swapchainSupport.formats.empty() || swapchainSupport.presentModes.empty()) {
            std::cerr << "[render] skip GPU: swapchain support incomplete\n";
            continue;
        }

        VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13Features.pNext = &vulkan12Features;
        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &vulkan13Features;
        vkGetPhysicalDeviceFeatures2(candidate, &features2);
        if (vulkan13Features.dynamicRendering != VK_TRUE) {
            std::cerr << "[render] skip GPU: dynamicRendering not supported\n";
            continue;
        }
        if (vulkan12Features.timelineSemaphore != VK_TRUE) {
            std::cerr << "[render] skip GPU: timelineSemaphore not supported\n";
            continue;
        }
        if (vulkan13Features.synchronization2 != VK_TRUE) {
            std::cerr << "[render] skip GPU: synchronization2 not supported\n";
            continue;
        }
        if (vulkan13Features.maintenance4 != VK_TRUE) {
            std::cerr << "[render] skip GPU: maintenance4 not supported\n";
            continue;
        }

        m_physicalDevice = candidate;
        m_graphicsQueueFamilyIndex = queueFamily.graphicsAndPresent.value();
        std::cerr << "[render] selected GPU: " << properties.deviceName
                  << ", queueFamily=" << m_graphicsQueueFamilyIndex << "\n";
        return true;
    }

    std::cerr << "[render] no suitable GPU found\n";
    return false;
}

bool Renderer::createLogicalDevice() {
    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12Features.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13Features.pNext = &vulkan12Features;
    vulkan13Features.dynamicRendering = VK_TRUE;
    vulkan13Features.synchronization2 = VK_TRUE;
    vulkan13Features.maintenance4 = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &vulkan13Features;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = kDeviceExtensions.data();

    const VkResult result = vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateDevice", result);
        return false;
    }

    vkGetDeviceQueue(m_device, m_graphicsQueueFamilyIndex, 0, &m_graphicsQueue);

    VkPhysicalDeviceProperties deviceProperties{};
    vkGetPhysicalDeviceProperties(m_physicalDevice, &deviceProperties);
    m_uniformBufferAlignment = std::max<VkDeviceSize>(
        deviceProperties.limits.minUniformBufferOffsetAlignment,
        static_cast<VkDeviceSize>(16)
    );
    return true;
}

bool Renderer::createTimelineSemaphore() {
    if (m_renderTimelineSemaphore != VK_NULL_HANDLE) {
        return true;
    }

    VkSemaphoreTypeCreateInfo timelineCreateInfo{};
    timelineCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineCreateInfo.initialValue = 0;

    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreCreateInfo.pNext = &timelineCreateInfo;

    const VkResult result = vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_renderTimelineSemaphore);
    if (result != VK_SUCCESS) {
        logVkFailure("vkCreateSemaphore(timeline)", result);
        return false;
    }

    m_frameTimelineValues.fill(0);
    m_nextTimelineValue = 1;
    return true;
}

bool Renderer::createUploadRingBuffer() {
    // Minimal per-frame ring buffer used for small CPU uploads.
    // Future streaming code can replace this with dedicated staging allocators.
    const bool ok = m_uploadRing.init(
        &m_bufferAllocator,
        1024 * 64,
        kMaxFramesInFlight,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
    );
    if (!ok) {
        std::cerr << "[render] upload ring buffer init failed\n";
    }
    return ok;
}

bool Renderer::createSwapchain() {
    const SwapchainSupport support = querySwapchainSupport(m_physicalDevice, m_surface);
    if (support.formats.empty() || support.presentModes.empty()) {
        std::cerr << "[render] swapchain support query returned no formats or present modes\n";
        return false;
    }

    const VkSurfaceFormatKHR surfaceFormat = chooseSwapchainFormat(support.formats);
    const VkPresentModeKHR presentMode = choosePresentMode(support.presentModes);
    const VkExtent2D extent = chooseExtent(m_window, support.capabilities);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = m_surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    const VkResult swapchainResult = vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain);
    if (swapchainResult != VK_SUCCESS) {
        logVkFailure("vkCreateSwapchainKHR", swapchainResult);
        return false;
    }

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, m_swapchainImages.data());

    m_swapchainFormat = surfaceFormat.format;
    m_swapchainExtent = extent;

    m_swapchainImageViews.resize(imageCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_swapchainImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_swapchainFormat;
        viewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewCreateInfo.subresourceRange.baseMipLevel = 0;
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.subresourceRange.baseArrayLayer = 0;
        viewCreateInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(m_device, &viewCreateInfo, nullptr, &m_swapchainImageViews[i]) != VK_SUCCESS) {
            std::cerr << "[render] failed to create swapchain image view " << i << "\n";
            return false;
        }
    }

    std::cerr << "[render] swapchain ready: images=" << imageCount
              << ", extent=" << m_swapchainExtent.width << "x" << m_swapchainExtent.height << "\n";
    m_swapchainImageInitialized.assign(imageCount, false);
    m_swapchainImageTimelineValues.assign(imageCount, 0);
    if (!createMsaaColorTargets()) {
        std::cerr << "[render] MSAA color target creation failed\n";
        return false;
    }
    m_renderFinishedSemaphores.resize(imageCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < imageCount; ++i) {
        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        const VkResult semaphoreResult =
            vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &m_renderFinishedSemaphores[i]);
        if (semaphoreResult != VK_SUCCESS) {
            logVkFailure("vkCreateSemaphore(renderFinishedPerImage)", semaphoreResult);
            return false;
        }
    }

    return true;
}

bool Renderer::createMsaaColorTargets() {
    const uint32_t imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    m_msaaColorImages.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageMemories.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageViews.assign(imageCount, VK_NULL_HANDLE);
    m_msaaColorImageInitialized.assign(imageCount, false);

    for (uint32_t i = 0; i < imageCount; ++i) {
        VkImageCreateInfo imageCreateInfo{};
        imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
        imageCreateInfo.format = m_swapchainFormat;
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

        const VkResult imageResult = vkCreateImage(m_device, &imageCreateInfo, nullptr, &m_msaaColorImages[i]);
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
            std::cerr << "[render] no memory type for MSAA color image\n";
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

        VkImageViewCreateInfo viewCreateInfo{};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.image = m_msaaColorImages[i];
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = m_swapchainFormat;
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
    }

    return true;
}

bool Renderer::createDescriptorResources() {
    if (m_descriptorSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding mvpBinding{};
        mvpBinding.binding = 0;
        mvpBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        mvpBinding.descriptorCount = 1;
        mvpBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutCreateInfo.bindingCount = 1;
        layoutCreateInfo.pBindings = &mvpBinding;

        const VkResult layoutResult =
            vkCreateDescriptorSetLayout(m_device, &layoutCreateInfo, nullptr, &m_descriptorSetLayout);
        if (layoutResult != VK_SUCCESS) {
            logVkFailure("vkCreateDescriptorSetLayout", layoutResult);
            return false;
        }
    }

    if (m_descriptorPool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = kMaxFramesInFlight;

        VkDescriptorPoolCreateInfo poolCreateInfo{};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCreateInfo.maxSets = kMaxFramesInFlight;
        poolCreateInfo.poolSizeCount = 1;
        poolCreateInfo.pPoolSizes = &poolSize;

        const VkResult poolResult = vkCreateDescriptorPool(m_device, &poolCreateInfo, nullptr, &m_descriptorPool);
        if (poolResult != VK_SUCCESS) {
            logVkFailure("vkCreateDescriptorPool", poolResult);
            return false;
        }
    }

    std::array<VkDescriptorSetLayout, kMaxFramesInFlight> setLayouts{};
    setLayouts.fill(m_descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = m_descriptorPool;
    allocateInfo.descriptorSetCount = static_cast<uint32_t>(setLayouts.size());
    allocateInfo.pSetLayouts = setLayouts.data();

    const VkResult allocateResult = vkAllocateDescriptorSets(m_device, &allocateInfo, m_descriptorSets.data());
    if (allocateResult != VK_SUCCESS) {
        logVkFailure("vkAllocateDescriptorSets", allocateResult);
        return false;
    }

    return true;
}

bool Renderer::createGraphicsPipeline() {
    if (m_pipelineLayout == VK_NULL_HANDLE) {
        VkPipelineLayoutCreateInfo layoutCreateInfo{};
        layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCreateInfo.setLayoutCount = 1;
        layoutCreateInfo.pSetLayouts = &m_descriptorSetLayout;
        const VkResult layoutResult = vkCreatePipelineLayout(m_device, &layoutCreateInfo, nullptr, &m_pipelineLayout);
        if (layoutResult != VK_SUCCESS) {
            logVkFailure("vkCreatePipelineLayout", layoutResult);
            return false;
        }
    }

    VkShaderModule vertShaderModule = VK_NULL_HANDLE;
    VkShaderModule fragShaderModule = VK_NULL_HANDLE;

    VkShaderModuleCreateInfo vertCreateInfo{};
    vertCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vertCreateInfo.codeSize = sizeof(kVertShaderSpirv);
    vertCreateInfo.pCode = kVertShaderSpirv;
    const VkResult vertModuleResult = vkCreateShaderModule(m_device, &vertCreateInfo, nullptr, &vertShaderModule);
    if (vertModuleResult != VK_SUCCESS) {
        logVkFailure("vkCreateShaderModule(vertex)", vertModuleResult);
        return false;
    }

    VkShaderModuleCreateInfo fragCreateInfo{};
    fragCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    fragCreateInfo.codeSize = sizeof(kFragShaderSpirv);
    fragCreateInfo.pCode = kFragShaderSpirv;
    const VkResult fragModuleResult = vkCreateShaderModule(m_device, &fragCreateInfo, nullptr, &fragShaderModule);
    if (fragModuleResult != VK_SUCCESS) {
        logVkFailure("vkCreateShaderModule(fragment)", fragModuleResult);
        vkDestroyShaderModule(m_device, vertShaderModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo vertexShaderStage{};
    vertexShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertexShaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertexShaderStage.module = vertShaderModule;
    vertexShaderStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragmentShaderStage{};
    fragmentShaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragmentShaderStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragmentShaderStage.module = fragShaderModule;
    fragmentShaderStage.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {vertexShaderStage, fragmentShaderStage};

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescription{};
    attributeDescription.location = 0;
    attributeDescription.binding = 0;
    attributeDescription.format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescription.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = 1;
    vertexInputInfo.pVertexAttributeDescriptions = &attributeDescription;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = m_colorSampleCount;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineRenderingCreateInfo renderingCreateInfo{};
    renderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingCreateInfo.colorAttachmentCount = 1;
    renderingCreateInfo.pColorAttachmentFormats = &m_swapchainFormat;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.pNext = &renderingCreateInfo;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisampling;
    pipelineCreateInfo.pColorBlendState = &colorBlending;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.layout = m_pipelineLayout;
    pipelineCreateInfo.renderPass = VK_NULL_HANDLE;
    pipelineCreateInfo.subpass = 0;

    VkPipeline newPipeline = VK_NULL_HANDLE;
    const VkResult pipelineResult = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &pipelineCreateInfo,
        nullptr,
        &newPipeline
    );

    vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, vertShaderModule, nullptr);

    if (pipelineResult != VK_SUCCESS) {
        logVkFailure("vkCreateGraphicsPipelines", pipelineResult);
        return false;
    }

    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
    }
    m_pipeline = newPipeline;
    std::cerr << "[render] graphics pipeline ready\n";
    return true;
}

bool Renderer::createVertexBuffer() {
    m_vertexCount = static_cast<uint32_t>(kGroundQuadVertices.size());
    BufferCreateDesc createDesc{};
    createDesc.size = sizeof(kGroundQuadVertices);
    createDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    createDesc.memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    createDesc.initialData = kGroundQuadVertices.data();
    m_vertexBufferHandle = m_bufferAllocator.createBuffer(createDesc);
    if (m_vertexBufferHandle == kInvalidBufferHandle) {
        std::cerr << "[render] vertex buffer allocation failed\n";
        return false;
    }
    std::cerr << "[render] vertex buffer ready (handle=" << m_vertexBufferHandle
              << ", bytes=" << sizeof(kGroundQuadVertices) << ")\n";
    return true;
}

bool Renderer::createFrameResources() {
    for (FrameResources& frame : m_frames) {
        VkCommandPoolCreateInfo poolCreateInfo{};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolCreateInfo.queueFamilyIndex = m_graphicsQueueFamilyIndex;

        if (vkCreateCommandPool(m_device, &poolCreateInfo, nullptr, &frame.commandPool) != VK_SUCCESS) {
            std::cerr << "[render] failed creating command pool for frame resource\n";
            return false;
        }

        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        if (vkCreateSemaphore(m_device, &semaphoreCreateInfo, nullptr, &frame.imageAvailable) != VK_SUCCESS) {
            std::cerr << "[render] failed creating imageAvailable semaphore\n";
            return false;
        }
    }

    std::cerr << "[render] frame resources ready (" << kMaxFramesInFlight << " frames in flight)\n";
    return true;
}

void Renderer::renderFrame(
    const world::ChunkGrid& chunkGrid,
    const sim::Simulation& simulation,
    const CameraPose& camera
) {
    (void)chunkGrid;
    (void)simulation;

    if (m_device == VK_NULL_HANDLE || m_swapchain == VK_NULL_HANDLE) {
        return;
    }

    m_uploadRing.beginFrame(m_currentFrame);

    FrameResources& frame = m_frames[m_currentFrame];
    if (m_frameTimelineValues[m_currentFrame] > 0) {
        VkSemaphore waitSemaphore = m_renderTimelineSemaphore;
        VkSemaphoreWaitInfo waitInfo{};
        waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &waitSemaphore;
        waitInfo.pValues = &m_frameTimelineValues[m_currentFrame];
        const VkResult waitResult = vkWaitSemaphores(m_device, &waitInfo, std::numeric_limits<uint64_t>::max());
        if (waitResult != VK_SUCCESS) {
            logVkFailure("vkWaitSemaphores(frame)", waitResult);
            return;
        }
    }

    uint32_t imageIndex = 0;
    const VkResult acquireResult = vkAcquireNextImageKHR(
        m_device,
        m_swapchain,
        std::numeric_limits<uint64_t>::max(),
        frame.imageAvailable,
        VK_NULL_HANDLE,
        &imageIndex
    );

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        std::cerr << "[render] swapchain out of date during acquire, recreating\n";
        recreateSwapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        logVkFailure("vkAcquireNextImageKHR", acquireResult);
        return;
    }

    if (m_swapchainImageTimelineValues[imageIndex] > 0) {
        VkSemaphore waitSemaphore = m_renderTimelineSemaphore;
        VkSemaphoreWaitInfo waitInfo{};
        waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &waitSemaphore;
        waitInfo.pValues = &m_swapchainImageTimelineValues[imageIndex];
        const VkResult waitResult = vkWaitSemaphores(m_device, &waitInfo, std::numeric_limits<uint64_t>::max());
        if (waitResult != VK_SUCCESS) {
            logVkFailure("vkWaitSemaphores(image)", waitResult);
            return;
        }
    }
    const VkSemaphore renderFinishedSemaphore = m_renderFinishedSemaphores[imageIndex];

    vkResetCommandPool(m_device, frame.commandPool, 0);

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = frame.commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(m_device, &allocateInfo, &commandBuffer) != VK_SUCCESS) {
        std::cerr << "[render] vkAllocateCommandBuffers failed\n";
        return;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        std::cerr << "[render] vkBeginCommandBuffer failed\n";
        return;
    }

    const float aspectRatio = static_cast<float>(m_swapchainExtent.width) / static_cast<float>(m_swapchainExtent.height);
    const float yawRadians = math::radians(camera.yawDegrees);
    const float pitchRadians = math::radians(camera.pitchDegrees);
    const float cosPitch = std::cos(pitchRadians);
    const math::Vector3 eye{camera.x, camera.y, camera.z};
    const math::Vector3 forward{
        std::cos(yawRadians) * cosPitch,
        std::sin(pitchRadians),
        std::sin(yawRadians) * cosPitch
    };
    const math::Matrix4 view = lookAt(eye, eye + forward, math::Vector3{0.0f, 1.0f, 0.0f});
    const math::Matrix4 projection = perspectiveVulkan(math::radians(camera.fovDegrees), aspectRatio, 0.1f, 500.0f);
    const math::Matrix4 mvp = projection * view;
    const math::Matrix4 mvpColumnMajor = transpose(mvp);

    const std::optional<RingBufferSlice> mvpSliceOpt =
        m_uploadRing.allocate(sizeof(CameraUniform), m_uniformBufferAlignment);
    if (!mvpSliceOpt.has_value() || mvpSliceOpt->mapped == nullptr) {
        std::cerr << "[render] failed to allocate MVP uniform slice\n";
        return;
    }

    CameraUniform mvpUniform{};
    std::memcpy(mvpUniform.mvp, mvpColumnMajor.m, sizeof(mvpUniform.mvp));
    std::memcpy(mvpSliceOpt->mapped, &mvpUniform, sizeof(mvpUniform));

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_bufferAllocator.getBuffer(mvpSliceOpt->buffer);
    bufferInfo.offset = mvpSliceOpt->offset;
    bufferInfo.range = mvpSliceOpt->size;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptorSets[m_currentFrame];
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

    transitionImageLayout(
        commandBuffer,
        m_swapchainImages[imageIndex],
        m_swapchainImageInitialized[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
    );
    if (!m_msaaColorImageInitialized[imageIndex]) {
        transitionImageLayout(
            commandBuffer,
            m_msaaColorImages[imageIndex],
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE,
            VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
        );
    }

    VkClearValue clearValue{};
    clearValue.color.float32[0] = 0.06f;
    clearValue.color.float32[1] = 0.08f;
    clearValue.color.float32[2] = 0.12f;
    clearValue.color.float32[3] = 1.0f;

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = m_msaaColorImageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.clearValue = clearValue;
    colorAttachment.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
    colorAttachment.resolveImageView = m_swapchainImageViews[imageIndex];
    colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea.offset = {0, 0};
    renderingInfo.renderArea.extent = m_swapchainExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_swapchainExtent.width);
    viewport.height = static_cast<float>(m_swapchainExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = m_swapchainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipelineLayout,
        0,
        1,
        &m_descriptorSets[m_currentFrame],
        0,
        nullptr
    );
    const VkBuffer vertexBuffer = m_bufferAllocator.getBuffer(m_vertexBufferHandle);
    if (vertexBuffer == VK_NULL_HANDLE) {
        std::cerr << "[render] missing vertex buffer for draw\n";
        return;
    }
    const VkDeviceSize vertexBufferOffset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexBufferOffset);
    vkCmdDraw(commandBuffer, m_vertexCount, 1, 0, 0);

    vkCmdEndRendering(commandBuffer);

    transitionImageLayout(
        commandBuffer,
        m_swapchainImages[imageIndex],
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE
    );

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        std::cerr << "[render] vkEndCommandBuffer failed\n";
        return;
    }

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const uint64_t signalTimelineValue = m_nextTimelineValue++;
    std::array<VkSemaphore, 2> signalSemaphores = {
        renderFinishedSemaphore,
        m_renderTimelineSemaphore
    };
    std::array<uint64_t, 2> signalSemaphoreValues = {
        0,
        signalTimelineValue
    };
    std::array<uint64_t, 1> waitSemaphoreValues = {0};
    VkTimelineSemaphoreSubmitInfo timelineSubmitInfo{};
    timelineSubmitInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineSubmitInfo.waitSemaphoreValueCount = static_cast<uint32_t>(waitSemaphoreValues.size());
    timelineSubmitInfo.pWaitSemaphoreValues = waitSemaphoreValues.data();
    timelineSubmitInfo.signalSemaphoreValueCount = static_cast<uint32_t>(signalSemaphoreValues.size());
    timelineSubmitInfo.pSignalSemaphoreValues = signalSemaphoreValues.data();

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineSubmitInfo;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &frame.imageAvailable;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
    submitInfo.pSignalSemaphores = signalSemaphores.data();

    if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        std::cerr << "[render] vkQueueSubmit failed\n";
        return;
    }
    m_frameTimelineValues[m_currentFrame] = signalTimelineValue;
    m_swapchainImageTimelineValues[imageIndex] = signalTimelineValue;

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapchain;
    presentInfo.pImageIndices = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(m_graphicsQueue, &presentInfo);
    m_swapchainImageInitialized[imageIndex] = true;
    m_msaaColorImageInitialized[imageIndex] = true;

    if (
        acquireResult == VK_SUBOPTIMAL_KHR ||
        presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
        presentResult == VK_SUBOPTIMAL_KHR
    ) {
        std::cerr << "[render] swapchain needs recreate after present\n";
        recreateSwapchain();
    } else if (presentResult != VK_SUCCESS) {
        logVkFailure("vkQueuePresentKHR", presentResult);
    }

    m_currentFrame = (m_currentFrame + 1) % kMaxFramesInFlight;
}

bool Renderer::recreateSwapchain() {
    std::cerr << "[render] recreateSwapchain begin\n";
    int width = 0;
    int height = 0;
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(m_window, &width, &height);
        if (glfwWindowShouldClose(m_window) == GLFW_TRUE) {
            return false;
        }
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(m_device);

    destroyPipeline();
    destroySwapchain();

    if (!createSwapchain()) {
        std::cerr << "[render] recreateSwapchain failed: createSwapchain\n";
        return false;
    }
    if (!createGraphicsPipeline()) {
        std::cerr << "[render] recreateSwapchain failed: createGraphicsPipeline\n";
        return false;
    }
    std::cerr << "[render] recreateSwapchain complete\n";
    return true;
}

void Renderer::destroyMsaaColorTargets() {
    for (VkImageView imageView : m_msaaColorImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_msaaColorImageViews.clear();

    for (VkImage image : m_msaaColorImages) {
        if (image != VK_NULL_HANDLE) {
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
    m_msaaColorImageInitialized.clear();
}

void Renderer::destroySwapchain() {
    destroyMsaaColorTargets();

    for (VkSemaphore semaphore : m_renderFinishedSemaphores) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, semaphore, nullptr);
        }
    }
    m_renderFinishedSemaphores.clear();

    for (VkImageView imageView : m_swapchainImageViews) {
        if (imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    m_swapchainImageViews.clear();
    m_swapchainImages.clear();
    m_swapchainImageInitialized.clear();
    m_swapchainImageTimelineValues.clear();

    if (m_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
}

void Renderer::destroyFrameResources() {
    for (FrameResources& frame : m_frames) {
        if (frame.imageAvailable != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, frame.imageAvailable, nullptr);
            frame.imageAvailable = VK_NULL_HANDLE;
        }
        if (frame.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(m_device, frame.commandPool, nullptr);
            frame.commandPool = VK_NULL_HANDLE;
        }
    }
}

void Renderer::destroyVertexBuffer() {
    if (m_vertexBufferHandle != kInvalidBufferHandle) {
        m_bufferAllocator.destroyBuffer(m_vertexBufferHandle);
        m_vertexBufferHandle = kInvalidBufferHandle;
    }
    m_vertexCount = 0;
}

void Renderer::destroyPipeline() {
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
}

void Renderer::shutdown() {
    std::cerr << "[render] shutdown begin\n";
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    if (m_device != VK_NULL_HANDLE) {
        destroyFrameResources();
        if (m_renderTimelineSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, m_renderTimelineSemaphore, nullptr);
            m_renderTimelineSemaphore = VK_NULL_HANDLE;
        }
        m_uploadRing.shutdown(&m_bufferAllocator);
        destroyVertexBuffer();
        destroyPipeline();
        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }
        if (m_descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
            m_descriptorSetLayout = VK_NULL_HANDLE;
        }
        m_descriptorSets.fill(VK_NULL_HANDLE);
        destroySwapchain();
        m_bufferAllocator.shutdown();

        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }

    if (m_surface != VK_NULL_HANDLE && m_instance != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }

    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }

    m_physicalDevice = VK_NULL_HANDLE;
    m_graphicsQueue = VK_NULL_HANDLE;
    m_graphicsQueueFamilyIndex = 0;
    m_frameTimelineValues.fill(0);
    m_nextTimelineValue = 1;
    m_currentFrame = 0;
    m_window = nullptr;
    std::cerr << "[render] shutdown complete\n";
}

} // namespace render
