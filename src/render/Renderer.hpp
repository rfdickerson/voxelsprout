#pragma once

#include "render/BufferHelpers.hpp"
#include "sim/Simulation.hpp"
#include "world/ChunkGrid.hpp"

#include <array>
#include <vector>

#include <vulkan/vulkan.h>

struct GLFWwindow;

// Render subsystem
// Responsible for: owning the rendering interface used by the app.
// Should NOT do: gameplay simulation, world editing rules, or graphics API specifics yet.
namespace render {

struct CameraPose {
    float x;
    float y;
    float z;
    float yawDegrees;
    float pitchDegrees;
    float fovDegrees;
};

class Renderer {
public:
    bool init(GLFWwindow* window);
    void renderFrame(const world::ChunkGrid& chunkGrid, const sim::Simulation& simulation, const CameraPose& camera);
    void shutdown();

private:
    static constexpr uint32_t kMaxFramesInFlight = 2;

    struct FrameResources {
        // Per-frame command pool to allocate fresh command buffers every frame.
        // Future frame-graph systems will replace this with transient allocators.
        VkCommandPool commandPool = VK_NULL_HANDLE;
        // Signals when swapchain image acquisition is complete for this frame.
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
    };

    bool createInstance();
    bool createSurface();
    bool pickPhysicalDevice();
    bool createLogicalDevice();
    bool createSwapchain();
    bool createMsaaColorTargets();
    bool createTimelineSemaphore();
    bool createGraphicsPipeline();
    bool createUploadRingBuffer();
    bool createDescriptorResources();
    bool createVertexBuffer();
    bool createFrameResources();
    bool recreateSwapchain();
    void destroySwapchain();
    void destroyMsaaColorTargets();
    void destroyFrameResources();
    void destroyVertexBuffer();
    void destroyPipeline();

    GLFWwindow* m_window = nullptr;

    // Global Vulkan API root object.
    // Future renderer versions can add debug utils and extra instance extensions.
    VkInstance m_instance = VK_NULL_HANDLE;
    // Connection between GLFW window and Vulkan presentation.
    // Future multi-window tooling can own multiple surfaces.
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    // Selected GPU used for rendering and present support.
    // Future device selection may become score-based for features/perf tiers.
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    // Logical device with one graphics queue for draw+present.
    // Future systems can add dedicated transfer/compute queues.
    VkDevice m_device = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueFamilyIndex = 0;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;

    // Presentable image chain for the window.
    // Future render-graph integration can manage this as a backend target.
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D m_swapchainExtent{};
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<bool> m_swapchainImageInitialized;
    std::vector<VkImage> m_msaaColorImages;
    std::vector<VkDeviceMemory> m_msaaColorImageMemories;
    std::vector<VkImageView> m_msaaColorImageViews;
    std::vector<bool> m_msaaColorImageInitialized;
    std::vector<uint64_t> m_swapchainImageTimelineValues;
    // One render-finished semaphore per swapchain image avoids reusing a semaphore
    // while presentation may still be waiting on it.
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    VkSampleCountFlagBits m_colorSampleCount = VK_SAMPLE_COUNT_4_BIT;

    // Minimal one-pass pipeline using dynamic rendering.
    // Future material systems will replace this single hardcoded pipeline.
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kMaxFramesInFlight> m_descriptorSets{};
    VkDeviceSize m_uniformBufferAlignment = 256;

    // Hardcoded geometry buffer (single flat quad).
    // Future meshing systems will stream chunk meshes instead.
    BufferAllocator m_bufferAllocator;
    FrameRingBuffer m_uploadRing;
    BufferHandle m_vertexBufferHandle = kInvalidBufferHandle;
    uint32_t m_vertexCount = 0;

    std::array<FrameResources, kMaxFramesInFlight> m_frames{};
    std::array<uint64_t, kMaxFramesInFlight> m_frameTimelineValues{};
    VkSemaphore m_renderTimelineSemaphore = VK_NULL_HANDLE;
    uint64_t m_nextTimelineValue = 1;
    uint32_t m_currentFrame = 0;
};

} // namespace render
