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

struct VoxelPreview {
    enum class Mode {
        Add,
        Remove
    };

    bool visible = false;
    int x = 0;
    int y = 0;
    int z = 0;
    Mode mode = Mode::Add;
};

class Renderer {
public:
    struct ShadowDebugSettings {
        bool enableRpdb = true;
        bool enableRotatedPoissonPcf = false;
        bool enableHybridNearVoxelRay = false;

        float casterConstantBiasBase = 1.1f;
        float casterConstantBiasCascadeScale = 0.9f;
        float casterSlopeBiasBase = 1.7f;
        float casterSlopeBiasCascadeScale = 0.85f;

        float receiverNormalOffsetNear = 0.03f;
        float receiverNormalOffsetFar = 0.12f;
        float receiverBaseBiasNearTexel = 2.2f;
        float receiverBaseBiasFarTexel = 4.6f;
        float receiverSlopeBiasNearTexel = 3.8f;
        float receiverSlopeBiasFarTexel = 7.2f;

        float cascadeBlendMin = 6.0f;
        float cascadeBlendFactor = 0.30f;

        float hybridRayStep = 0.45f;
        float hybridRayMaxDistance = 28.0f;
        float rpdbScale = 2.5f;
        float pcfRadius = 2.0f;
        int poissonSampleCount = 16;
    };

    bool init(GLFWwindow* window, const world::ChunkGrid& chunkGrid);
    bool updateChunkMesh(const world::ChunkGrid& chunkGrid);
    void renderFrame(
        const world::ChunkGrid& chunkGrid,
        const sim::Simulation& simulation,
        const CameraPose& camera,
        const VoxelPreview& preview
    );
    void setDebugUiVisible(bool visible);
    bool isDebugUiVisible() const;
    void shutdown();

private:
    static constexpr uint32_t kMaxFramesInFlight = 2;
    static constexpr uint32_t kShadowCascadeCount = 4;
    static constexpr uint32_t kShadowMapSize = 2048;

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
    bool createHdrResolveTargets();
    bool createMsaaColorTargets();
    bool createDepthTargets();
    bool createShadowResources();
    bool createTimelineSemaphore();
    bool createGraphicsPipeline();
    bool createUploadRingBuffer();
    bool createTransferResources();
    bool createPreviewBuffers();
    bool createEnvironmentResources();
    bool createDescriptorResources();
    bool createChunkBuffers(const world::ChunkGrid& chunkGrid);
    bool updateShadowVoxelGrid(const world::ChunkGrid& chunkGrid);
    bool createFrameResources();
#if defined(VOXEL_HAS_IMGUI)
    bool createImGuiResources();
    void destroyImGuiResources();
    void buildShadowDebugUi();
#endif
    bool recreateSwapchain();
    void destroySwapchain();
    void destroyHdrResolveTargets();
    void destroyMsaaColorTargets();
    void destroyDepthTargets();
    void destroyShadowResources();
    void destroyFrameResources();
    void destroyChunkBuffers();
    void destroyPreviewBuffers();
    void destroyEnvironmentResources();
    void destroyTransferResources();
    void destroyPipeline();
    bool waitForTimelineValue(uint64_t value) const;
    void scheduleBufferRelease(BufferHandle handle, uint64_t timelineValue);
    void collectCompletedBufferReleases();

    struct DeferredBufferRelease {
        BufferHandle handle = kInvalidBufferHandle;
        uint64_t timelineValue = 0;
    };

    struct ChunkDrawRange {
        uint32_t indexCount = 0;
        uint32_t firstIndex = 0;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        float offsetZ = 0.0f;
    };

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
    // Logical device with a graphics queue (draw+present) and a transfer queue.
    VkDevice m_device = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueFamilyIndex = 0;
    uint32_t m_graphicsQueueIndex = 0;
    uint32_t m_transferQueueFamilyIndex = 0;
    uint32_t m_transferQueueIndex = 0;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_transferQueue = VK_NULL_HANDLE;

    // Presentable image chain for the window.
    // Future render-graph integration can manage this as a backend target.
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D m_swapchainExtent{};
    VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_shadowDepthFormat = VK_FORMAT_UNDEFINED;
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<bool> m_swapchainImageInitialized;
    std::vector<VkImage> m_msaaColorImages;
    std::vector<VkDeviceMemory> m_msaaColorImageMemories;
    std::vector<VkImageView> m_msaaColorImageViews;
    std::vector<bool> m_msaaColorImageInitialized;
    std::vector<VkImage> m_hdrResolveImages;
    std::vector<VkDeviceMemory> m_hdrResolveImageMemories;
    std::vector<VkImageView> m_hdrResolveImageViews;
    std::vector<bool> m_hdrResolveImageInitialized;
    VkSampler m_hdrResolveSampler = VK_NULL_HANDLE;
    std::vector<VkImage> m_depthImages;
    std::vector<VkDeviceMemory> m_depthImageMemories;
    std::vector<VkImageView> m_depthImageViews;
    VkImage m_shadowDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory m_shadowDepthMemory = VK_NULL_HANDLE;
    VkImageView m_shadowDepthImageView = VK_NULL_HANDLE;
    std::array<VkImageView, kShadowCascadeCount> m_shadowDepthLayerViews{};
    VkSampler m_shadowDepthSampler = VK_NULL_HANDLE;
    bool m_shadowDepthInitialized = false;
    std::vector<uint64_t> m_swapchainImageTimelineValues;
    // One render-finished semaphore per swapchain image avoids reusing a semaphore
    // while presentation may still be waiting on it.
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    VkSampleCountFlagBits m_colorSampleCount = VK_SAMPLE_COUNT_4_BIT;
    VkFormat m_hdrColorFormat = VK_FORMAT_UNDEFINED;

    // Minimal one-pass pipeline using dynamic rendering.
    // Future material systems will replace this single hardcoded pipeline.
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipeline m_shadowPipeline = VK_NULL_HANDLE;
    VkPipeline m_skyboxPipeline = VK_NULL_HANDLE;
    VkPipeline m_tonemapPipeline = VK_NULL_HANDLE;
    VkPipeline m_previewAddPipeline = VK_NULL_HANDLE;
    VkPipeline m_previewRemovePipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kMaxFramesInFlight> m_descriptorSets{};
    bool m_supportsWireframePreview = false;
    VkDeviceSize m_uniformBufferAlignment = 256;

    // Static mesh buffers for one chunk.
    // Future chunk streaming can replace these with per-chunk GPU allocations.
    BufferAllocator m_bufferAllocator;
    FrameRingBuffer m_uploadRing;
    BufferHandle m_vertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_indexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_previewVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_previewIndexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_shadowVoxelBufferHandle = kInvalidBufferHandle;
    VkBufferView m_shadowVoxelBufferView = VK_NULL_HANDLE;
    uint32_t m_shadowVoxelGridSizeX = 0;
    uint32_t m_shadowVoxelGridSizeY = 0;
    uint32_t m_shadowVoxelGridSizeZ = 0;
    int32_t m_shadowVoxelGridOriginX = 0;
    int32_t m_shadowVoxelGridOriginY = 0;
    int32_t m_shadowVoxelGridOriginZ = 0;
    std::vector<DeferredBufferRelease> m_deferredBufferReleases;
    std::vector<ChunkDrawRange> m_chunkDrawRanges;
    uint32_t m_indexCount = 0;
    uint32_t m_previewIndexCount = 0;

    std::array<FrameResources, kMaxFramesInFlight> m_frames{};
    VkCommandPool m_transferCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_transferCommandBuffer = VK_NULL_HANDLE;
    std::array<uint64_t, kMaxFramesInFlight> m_frameTimelineValues{};
    VkSemaphore m_renderTimelineSemaphore = VK_NULL_HANDLE;
    uint64_t m_pendingTransferTimelineValue = 0;
    uint64_t m_currentChunkReadyTimelineValue = 0;
    uint64_t m_transferCommandBufferInFlightValue = 0;
    uint64_t m_lastGraphicsTimelineValue = 0;
    uint64_t m_nextTimelineValue = 1;
    uint32_t m_currentFrame = 0;
    bool m_debugUiVisible = false;
    ShadowDebugSettings m_shadowDebugSettings{};
#if defined(VOXEL_HAS_IMGUI)
    bool m_imguiInitialized = false;
    VkDescriptorPool m_imguiDescriptorPool = VK_NULL_HANDLE;
#endif
    // Dynamic cascade split distances in view-space units.
    // Updated per frame and consumed by shadow rendering + shading.
    std::array<float, kShadowCascadeCount> m_shadowCascadeSplits = {20.0f, 45.0f, 90.0f, 180.0f};
};

} // namespace render
