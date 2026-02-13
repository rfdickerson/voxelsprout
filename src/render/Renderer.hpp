#pragma once

#include "render/BufferHelpers.hpp"
#include "sim/Simulation.hpp"
#include "world/ClipmapIndex.hpp"
#include "world/ChunkGrid.hpp"
#include "world/ChunkMesher.hpp"
#include "world/SpatialIndex.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <vulkan/vulkan.h>
#if defined(VOXEL_HAS_VMA)
#if defined(__has_include)
#if __has_include(<vk_mem_alloc.h>)
#include <vk_mem_alloc.h>
#elif __has_include(<vma/vk_mem_alloc.h>)
#include <vma/vk_mem_alloc.h>
#else
#error "VOXEL_HAS_VMA is set but vk_mem_alloc.h was not found"
#endif
#else
#include <vk_mem_alloc.h>
#endif
#endif

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
    int brushSize = 1;
    Mode mode = Mode::Add;
    bool faceVisible = false;
    int faceX = 0;
    int faceY = 0;
    int faceZ = 0;
    uint32_t faceId = 0;
    bool pipeStyle = false;
    float pipeAxisX = 0.0f;
    float pipeAxisY = 1.0f;
    float pipeAxisZ = 0.0f;
    float pipeRadius = 0.45f;
    float pipeStyleId = 0.0f;
};

class Renderer {
public:
    struct ShadowDebugSettings {
        float casterConstantBiasBase = 1.1f;
        float casterConstantBiasCascadeScale = 0.9f;
        float casterSlopeBiasBase = 1.7f;
        float casterSlopeBiasCascadeScale = 0.85f;

        float receiverNormalOffsetNear = 0.03f;
        float receiverNormalOffsetFar = 0.12f;
        float receiverBaseBiasNearTexel = 0.05f;
        float receiverBaseBiasFarTexel = 4.6f;
        float receiverSlopeBiasNearTexel = 3.8f;
        float receiverSlopeBiasFarTexel = 7.2f;

        float cascadeBlendMin = 6.0f;
        float cascadeBlendFactor = 0.30f;

        float pcfRadius = 1.0f;
        int grassShadowCascadeCount = 1;

        float ssaoRadius = 0.55f;
        float ssaoBias = 0.03f;
        float ssaoIntensity = 0.60f;
    };

    struct SkyDebugSettings {
        float sunYawDegrees = -157.5f;
        float sunPitchDegrees = -13.0f;
        float rayleighStrength = 1.0f;
        float mieStrength = 1.0f;
        float mieAnisotropy = 0.55f;
        float skyExposure = 1.0f;
        float sunDiskIntensity = 1150.0f;
        float sunHaloIntensity = 22.0f;
        float sunDiskSize = 2.0f;
        float sunHazeFalloff = 0.35f;
        float bloomThreshold = 0.75f;
        float bloomSoftKnee = 0.5f;
        float bloomBaseIntensity = 0.08f;
        float bloomSunFacingBoost = 0.28f;
        bool autoSunriseTuning = true;
        float autoSunriseBlend = 1.0f;
        float autoSunriseAdaptSpeed = 4.0f;
        float plantQuadDirectionality = 0.34f;
    };

    struct VoxelGiDebugSettings {
        float strength = 0.45f;
        float injectSunScale = 0.50f;
        float injectShScale = 0.45f;
        float injectBounceScale = 0.85f;
        float propagateBlend = 0.62f;
        float propagateDecay = 0.93f;
        float ambientRebalanceStrength = 1.35f;
        float ambientFloor = 0.45f;
        int visualizationMode = 0; // 0 = off, 1 = radiance, 2 = false-color luminance
    };

    bool init(GLFWwindow* window, const world::ChunkGrid& chunkGrid);
    void clearMagicaVoxelMeshes();
    bool uploadMagicaVoxelMesh(const world::ChunkMeshData& mesh, float worldOffsetX, float worldOffsetY, float worldOffsetZ);
    void setVoxelBaseColorPalette(const std::array<std::uint32_t, 16>& paletteRgba);
    bool updateChunkMesh(const world::ChunkGrid& chunkGrid);
    bool updateChunkMesh(const world::ChunkGrid& chunkGrid, std::size_t chunkIndex);
    bool updateChunkMesh(const world::ChunkGrid& chunkGrid, std::span<const std::size_t> chunkIndices);
    bool useSpatialPartitioningQueries() const;
    world::ClipmapConfig clipmapQueryConfig() const;
    void setSpatialQueryStats(bool used, const world::SpatialQueryStats& stats, std::uint32_t visibleChunkCount);
    void renderFrame(
        const world::ChunkGrid& chunkGrid,
        const sim::Simulation& simulation,
        const CameraPose& camera,
        const VoxelPreview& preview,
        float simulationAlpha,
        std::span<const std::size_t> visibleChunkIndices
    );
    void setDebugUiVisible(bool visible);
    bool isDebugUiVisible() const;
    void setFrameStatsVisible(bool visible);
    bool isFrameStatsVisible() const;
    void setSunAngles(float yawDegrees, float pitchDegrees);
    float cameraFovDegrees() const;
    void shutdown();

private:
    static constexpr uint32_t kMaxFramesInFlight = 3;
    static constexpr uint32_t kShadowCascadeCount = 4;
    static constexpr uint32_t kShadowAtlasSize = 8192;
    static constexpr uint32_t kGpuTimestampQueryFrameStart = 0;
    static constexpr uint32_t kGpuTimestampQueryShadowStart = 1;
    static constexpr uint32_t kGpuTimestampQueryShadowEnd = 2;
    static constexpr uint32_t kGpuTimestampQueryGiInjectStart = 3;
    static constexpr uint32_t kGpuTimestampQueryGiInjectEnd = 4;
    static constexpr uint32_t kGpuTimestampQueryGiPropagateStart = 5;
    static constexpr uint32_t kGpuTimestampQueryGiPropagateEnd = 6;
    static constexpr uint32_t kGpuTimestampQueryPrepassStart = 7;
    static constexpr uint32_t kGpuTimestampQueryPrepassEnd = 8;
    static constexpr uint32_t kGpuTimestampQuerySsaoStart = 9;
    static constexpr uint32_t kGpuTimestampQuerySsaoEnd = 10;
    static constexpr uint32_t kGpuTimestampQuerySsaoBlurStart = 11;
    static constexpr uint32_t kGpuTimestampQuerySsaoBlurEnd = 12;
    static constexpr uint32_t kGpuTimestampQueryMainStart = 13;
    static constexpr uint32_t kGpuTimestampQueryMainEnd = 14;
    static constexpr uint32_t kGpuTimestampQueryPostStart = 15;
    static constexpr uint32_t kGpuTimestampQueryPostEnd = 16;
    static constexpr uint32_t kGpuTimestampQueryFrameEnd = 17;
    static constexpr uint32_t kGpuTimestampQueryCount = 18;
    static constexpr std::uint32_t kTimingHistorySampleCount = 240;

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
    bool createAoTargets();
    bool createShadowResources();
    bool createVoxelGiResources();
    bool createTimelineSemaphore();
    bool createGraphicsPipeline();
    bool createMagicaPipeline();
    bool createUploadRingBuffer();
    bool createTransferResources();
    bool createPipeBuffers();
    bool createPipePipeline();
    bool createAoPipelines();
    bool createPreviewBuffers();
    bool createEnvironmentResources();
    bool createDiffuseTextureResources();
    bool createDescriptorResources();
    bool createChunkBuffers(const world::ChunkGrid& chunkGrid, std::span<const std::size_t> remeshChunkIndices);
    bool createFrameResources();
    bool createGpuTimestampResources();
#if defined(VOXEL_HAS_IMGUI)
    bool createImGuiResources();
    void destroyImGuiResources();
    void buildFrameStatsUi();
    void buildMeshingDebugUi();
    void buildShadowDebugUi();
    void buildSunDebugUi();
    void buildAimReticleUi();
#endif
    bool recreateSwapchain();
    void destroySwapchain();
    void destroyHdrResolveTargets();
    void destroyMsaaColorTargets();
    void destroyDepthTargets();
    void destroyAoTargets();
    void destroyGpuTimestampResources();
    void destroyShadowResources();
    void destroyVoxelGiResources();
    void destroyFrameResources();
    void destroyChunkBuffers();
    void destroyMagicaBuffers();
    void destroyPipeBuffers();
    void destroyPreviewBuffers();
    void destroyEnvironmentResources();
    void destroyDiffuseTextureResources();
    void destroyTransferResources();
    void destroyPipeline();
    void loadDebugUtilsFunctions();
    void setObjectName(VkObjectType objectType, uint64_t objectHandle, const char* name) const;
    void beginDebugLabel(VkCommandBuffer commandBuffer, const char* name, float r, float g, float b, float a = 1.0f) const;
    void endDebugLabel(VkCommandBuffer commandBuffer) const;
    void insertDebugLabel(VkCommandBuffer commandBuffer, const char* name, float r, float g, float b, float a = 1.0f) const;
    void readGpuTimestampResults(uint32_t frameIndex);
    void scheduleBufferRelease(BufferHandle handle, uint64_t timelineValue);
    void collectCompletedBufferReleases();
    void updateDisplayTimingStats();
    bool isTimelineValueReached(uint64_t value) const;

    struct DeferredBufferRelease {
        BufferHandle handle = kInvalidBufferHandle;
        uint64_t timelineValue = 0;
    };

    struct ChunkDrawRange {
        uint32_t firstIndex = 0;
        int32_t vertexOffset = 0;
        uint32_t indexCount = 0;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        float offsetZ = 0.0f;
    };

    struct PipeVertex {
        float position[3];
        float normal[3];
    };

    struct PipeInstance {
        float originLength[4];
        float axisRadius[4];
        float tint[4];
        float extensions[4];
    };

    struct GrassBillboardVertex {
        float corner[2];
        float uv[2];
        float plane;
    };

    struct GrassBillboardInstance {
        float worldPosYaw[4];
        float colorTint[4];
    };

    struct MagicaMeshDraw {
        BufferHandle vertexBufferHandle = kInvalidBufferHandle;
        BufferHandle indexBufferHandle = kInvalidBufferHandle;
        std::uint32_t indexCount = 0;
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
    bool m_debugUtilsEnabled = false;
    PFN_vkSetDebugUtilsObjectNameEXT m_setDebugUtilsObjectName = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT m_cmdBeginDebugUtilsLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT m_cmdEndDebugUtilsLabel = nullptr;
    PFN_vkCmdInsertDebugUtilsLabelEXT m_cmdInsertDebugUtilsLabel = nullptr;
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
    VkExtent2D m_aoExtent{};
    VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_shadowDepthFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_normalDepthFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_ssaoFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_voxelGiFormat = VK_FORMAT_UNDEFINED;
    VkFormat m_voxelGiOccupancyFormat = VK_FORMAT_UNDEFINED;
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<bool> m_swapchainImageInitialized;
    std::vector<VkImage> m_msaaColorImages;
    std::vector<VkDeviceMemory> m_msaaColorImageMemories;
    std::vector<VkImageView> m_msaaColorImageViews;
    std::vector<bool> m_msaaColorImageInitialized;
#if defined(VOXEL_HAS_VMA)
    std::vector<VmaAllocation> m_msaaColorImageAllocations;
#endif
    std::vector<VkImage> m_hdrResolveImages;
    std::vector<VkDeviceMemory> m_hdrResolveImageMemories;
    std::vector<VkImageView> m_hdrResolveImageViews;
    std::vector<VkImageView> m_hdrResolveSampleImageViews;
    std::vector<TransientImageHandle> m_hdrResolveTransientHandles;
    std::vector<bool> m_hdrResolveImageInitialized;
    uint32_t m_hdrResolveMipLevels = 1;
    VkSampler m_hdrResolveSampler = VK_NULL_HANDLE;
    std::vector<VkImage> m_depthImages;
    std::vector<VkDeviceMemory> m_depthImageMemories;
    std::vector<VkImageView> m_depthImageViews;
#if defined(VOXEL_HAS_VMA)
    std::vector<VmaAllocation> m_depthImageAllocations;
#endif
    std::vector<VkImage> m_normalDepthImages;
    std::vector<VkDeviceMemory> m_normalDepthImageMemories;
    std::vector<VkImageView> m_normalDepthImageViews;
    std::vector<TransientImageHandle> m_normalDepthTransientHandles;
    std::vector<bool> m_normalDepthImageInitialized;
    std::vector<VkImage> m_aoDepthImages;
    std::vector<VkDeviceMemory> m_aoDepthImageMemories;
    std::vector<VkImageView> m_aoDepthImageViews;
    std::vector<TransientImageHandle> m_aoDepthTransientHandles;
    std::vector<bool> m_aoDepthImageInitialized;
    std::vector<VkImage> m_ssaoRawImages;
    std::vector<VkDeviceMemory> m_ssaoRawImageMemories;
    std::vector<VkImageView> m_ssaoRawImageViews;
    std::vector<TransientImageHandle> m_ssaoRawTransientHandles;
    std::vector<bool> m_ssaoRawImageInitialized;
    std::vector<VkImage> m_ssaoBlurImages;
    std::vector<VkDeviceMemory> m_ssaoBlurImageMemories;
    std::vector<VkImageView> m_ssaoBlurImageViews;
    std::vector<TransientImageHandle> m_ssaoBlurTransientHandles;
    std::vector<bool> m_ssaoBlurImageInitialized;
    VkSampler m_normalDepthSampler = VK_NULL_HANDLE;
    VkSampler m_ssaoSampler = VK_NULL_HANDLE;
    VkImage m_shadowDepthImage = VK_NULL_HANDLE;
    VkImageView m_shadowDepthImageView = VK_NULL_HANDLE;
    VkSampler m_shadowDepthSampler = VK_NULL_HANDLE;
    bool m_shadowDepthInitialized = false;
    std::array<VkImage, 2> m_voxelGiImages{};
    std::array<VkImageView, 2> m_voxelGiImageViews{};
    std::array<VkDeviceMemory, 2> m_voxelGiImageMemories{};
    VkSampler m_voxelGiSampler = VK_NULL_HANDLE;
    bool m_voxelGiInitialized = false;
    bool m_voxelGiComputeAvailable = false;
    VkImage m_voxelGiOccupancyImage = VK_NULL_HANDLE;
    VkImageView m_voxelGiOccupancyImageView = VK_NULL_HANDLE;
    VkDeviceMemory m_voxelGiOccupancyMemory = VK_NULL_HANDLE;
    VkSampler m_voxelGiOccupancySampler = VK_NULL_HANDLE;
    bool m_voxelGiOccupancyInitialized = false;
    bool m_voxelGiWorldDirty = true;
    bool m_voxelGiHasPreviousFrameState = false;
    std::array<float, 3> m_voxelGiPreviousGridOrigin{0.0f, 0.0f, 0.0f};
    std::array<float, 3> m_voxelGiPreviousSunDirection{0.0f, 0.0f, 0.0f};
    std::array<float, 3> m_voxelGiPreviousSunColor{0.0f, 0.0f, 0.0f};
    std::array<std::array<float, 3>, 9> m_voxelGiPreviousShIrradiance{};
    float m_voxelGiPreviousInjectSunScale = 0.0f;
    float m_voxelGiPreviousInjectShScale = 0.0f;
    float m_voxelGiPreviousInjectBounceScale = 0.0f;
    float m_voxelGiPreviousPropagateBlend = 0.0f;
    float m_voxelGiPreviousPropagateDecay = 0.0f;
#if defined(VOXEL_HAS_VMA)
    VmaAllocator m_vmaAllocator = VK_NULL_HANDLE;
    VmaAllocation m_shadowDepthAllocation = VK_NULL_HANDLE;
    VmaAllocation m_diffuseTextureAllocation = VK_NULL_HANDLE;
    std::array<VmaAllocation, 2> m_voxelGiImageAllocations{};
    VmaAllocation m_voxelGiOccupancyAllocation = VK_NULL_HANDLE;
#endif
    VkDeviceMemory m_shadowDepthMemory = VK_NULL_HANDLE;
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
    VkPipeline m_pipeShadowPipeline = VK_NULL_HANDLE;
    VkPipeline m_grassBillboardShadowPipeline = VK_NULL_HANDLE;
    VkPipeline m_skyboxPipeline = VK_NULL_HANDLE;
    VkPipeline m_tonemapPipeline = VK_NULL_HANDLE;
    VkPipeline m_pipePipeline = VK_NULL_HANDLE;
    VkPipeline m_grassBillboardPipeline = VK_NULL_HANDLE;
    VkPipeline m_voxelNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline m_pipeNormalDepthPipeline = VK_NULL_HANDLE;
    VkPipeline m_magicaPipeline = VK_NULL_HANDLE;
    VkPipeline m_ssaoPipeline = VK_NULL_HANDLE;
    VkPipeline m_ssaoBlurPipeline = VK_NULL_HANDLE;
    VkPipeline m_previewAddPipeline = VK_NULL_HANDLE;
    VkPipeline m_previewRemovePipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_voxelGiPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_voxelGiInjectPipeline = VK_NULL_HANDLE;
    VkPipeline m_voxelGiPropagatePipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_bindlessDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_voxelGiDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorPool m_bindlessDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorPool m_voxelGiDescriptorPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kMaxFramesInFlight> m_descriptorSets{};
    std::array<VkDescriptorSet, kMaxFramesInFlight> m_voxelGiDescriptorSets{};
    VkDescriptorSet m_bindlessDescriptorSet = VK_NULL_HANDLE;
    bool m_supportsWireframePreview = false;
    bool m_supportsSamplerAnisotropy = false;
    bool m_supportsMultiDrawIndirect = false;
    bool m_supportsBindlessDescriptors = false;
    bool m_supportsDisplayTiming = false;
    bool m_hasDisplayTimingExtension = false;
    bool m_enableDisplayTiming = false;
    uint32_t m_bindlessTextureCapacity = 0;
    bool m_gpuTimestampsSupported = false;
    float m_gpuTimestampPeriodNs = 0.0f;
    std::array<VkQueryPool, kMaxFramesInFlight> m_gpuTimestampQueryPools{};
    float m_maxSamplerAnisotropy = 1.0f;
    VkDeviceSize m_uniformBufferAlignment = 256;

    // Static mesh buffers per chunk draw range.
    // Future chunk streaming can replace this with sparse streaming allocations.
    BufferAllocator m_bufferAllocator;
    FrameArena m_frameArena;
    BufferHandle m_previewVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_previewIndexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_chunkVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_chunkIndexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_pipeVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_pipeIndexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_transportVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_transportIndexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_grassBillboardVertexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_grassBillboardIndexBufferHandle = kInvalidBufferHandle;
    BufferHandle m_grassBillboardInstanceBufferHandle = kInvalidBufferHandle;
    std::vector<DeferredBufferRelease> m_deferredBufferReleases;
    std::vector<ChunkDrawRange> m_chunkDrawRanges;
    std::vector<world::ChunkLodMeshes> m_chunkLodMeshCache;
    std::vector<std::vector<GrassBillboardInstance>> m_chunkGrassInstanceCache;
    std::vector<MagicaMeshDraw> m_magicaMeshDraws;
    bool m_chunkLodMeshCacheValid = false;
    world::MeshingOptions m_chunkMeshingOptions{};
    bool m_chunkMeshRebuildRequested = false;
    std::vector<std::size_t> m_pendingChunkRemeshIndices;
    uint32_t m_previewIndexCount = 0;
    uint32_t m_pipeIndexCount = 0;
    uint32_t m_transportIndexCount = 0;
    uint32_t m_grassBillboardIndexCount = 0;
    uint32_t m_grassBillboardInstanceCount = 0;
    std::array<std::uint32_t, 16> m_voxelBaseColorPaletteRgba{};
    VkImage m_diffuseTextureImage = VK_NULL_HANDLE;
    VkDeviceMemory m_diffuseTextureMemory = VK_NULL_HANDLE;
    VkImageView m_diffuseTextureImageView = VK_NULL_HANDLE;
    VkSampler m_diffuseTextureSampler = VK_NULL_HANDLE;
    VkSampler m_diffuseTexturePlantSampler = VK_NULL_HANDLE;

    std::array<FrameResources, kMaxFramesInFlight> m_frames{};
    VkCommandPool m_transferCommandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_transferCommandBuffer = VK_NULL_HANDLE;
    std::array<uint64_t, kMaxFramesInFlight> m_frameTimelineValues{};
    VkSemaphore m_renderTimelineSemaphore = VK_NULL_HANDLE;
    PFN_vkGetRefreshCycleDurationGOOGLE m_getRefreshCycleDurationGoogle = nullptr;
    PFN_vkGetPastPresentationTimingGOOGLE m_getPastPresentationTimingGoogle = nullptr;
    uint64_t m_pendingTransferTimelineValue = 0;
    uint64_t m_currentChunkReadyTimelineValue = 0;
    uint64_t m_transferCommandBufferInFlightValue = 0;
    uint64_t m_lastGraphicsTimelineValue = 0;
    uint64_t m_nextTimelineValue = 1;
    uint32_t m_nextDisplayTimingPresentId = 1;
    uint32_t m_lastSubmittedDisplayTimingPresentId = 0;
    uint32_t m_lastPresentedDisplayTimingPresentId = 0;
    uint32_t m_currentFrame = 0;
    bool m_debugUiVisible = false;
    bool m_showFrameStatsPanel = false;
    bool m_showMeshingPanel = false;
    bool m_showShadowPanel = false;
    bool m_showSunPanel = false;
    float m_debugCameraFovDegrees = 90.0f;
    bool m_debugCameraFovInitialized = false;
    bool m_debugEnableVertexAo = true;
    bool m_debugEnableSsao = true;
    bool m_debugVisualizeSsao = false;
    bool m_debugVisualizeAoNormals = false;
    ShadowDebugSettings m_shadowDebugSettings{};
    SkyDebugSettings m_skyDebugSettings{};
    VoxelGiDebugSettings m_voxelGiDebugSettings{};
    struct SkyTuningRuntimeState {
        bool initialized = false;
        float rayleighStrength = 1.0f;
        float mieStrength = 1.0f;
        float mieAnisotropy = 0.55f;
        float skyExposure = 1.0f;
        float sunDiskIntensity = 1150.0f;
        float sunHaloIntensity = 22.0f;
        float sunDiskSize = 2.0f;
        float sunHazeFalloff = 0.35f;
    } m_skyTuningRuntime{};
#if defined(VOXEL_HAS_IMGUI)
    bool m_imguiInitialized = false;
    VkDescriptorPool m_imguiDescriptorPool = VK_NULL_HANDLE;
#endif
    double m_lastFrameTimestampSeconds = 0.0;
    float m_debugFrameTimeMs = 0.0f;
    float m_debugGpuFrameTimeMs = 0.0f;
    float m_debugGpuShadowTimeMs = 0.0f;
    float m_debugGpuGiInjectTimeMs = 0.0f;
    float m_debugGpuGiPropagateTimeMs = 0.0f;
    float m_debugGpuPrepassTimeMs = 0.0f;
    float m_debugGpuSsaoTimeMs = 0.0f;
    float m_debugGpuSsaoBlurTimeMs = 0.0f;
    float m_debugGpuMainTimeMs = 0.0f;
    float m_debugGpuPostTimeMs = 0.0f;
    float m_debugDisplayRefreshMs = 0.0f;
    float m_debugDisplayPresentMarginMs = 0.0f;
    float m_debugDisplayActualEarliestDeltaMs = 0.0f;
    std::uint32_t m_debugDisplayTimingSampleCount = 0;
    std::array<float, kTimingHistorySampleCount> m_debugCpuFrameTotalMsHistory{};
    std::array<float, kTimingHistorySampleCount> m_debugCpuFrameWorkMsHistory{};
    std::array<float, kTimingHistorySampleCount> m_debugCpuFrameEwmaMsHistory{};
    std::uint32_t m_debugCpuFrameTimingMsHistoryWrite = 0;
    std::uint32_t m_debugCpuFrameTimingMsHistoryCount = 0;
    float m_debugCpuFrameWorkMs = 0.0f;
    float m_debugCpuFrameEwmaMs = 0.0f;
    bool m_debugCpuFrameEwmaInitialized = false;
    std::array<float, kTimingHistorySampleCount> m_debugGpuFrameTimingMsHistory{};
    std::uint32_t m_debugGpuFrameTimingMsHistoryWrite = 0;
    std::uint32_t m_debugGpuFrameTimingMsHistoryCount = 0;
    float m_debugFps = 0.0f;
    std::uint32_t m_debugChunkCount = 0;
    std::uint32_t m_debugMacroCellUniformCount = 0;
    std::uint32_t m_debugMacroCellRefined4Count = 0;
    std::uint32_t m_debugMacroCellRefined1Count = 0;
    std::uint32_t m_debugDrawnLod0Ranges = 0;
    std::uint32_t m_debugDrawnLod1Ranges = 0;
    std::uint32_t m_debugDrawnLod2Ranges = 0;
    bool m_debugEnableSpatialQueries = true;
    world::ClipmapConfig m_debugClipmapConfig{};
    bool m_debugSpatialQueriesUsed = false;
    world::SpatialQueryStats m_debugSpatialQueryStats{};
    std::uint32_t m_debugSpatialVisibleChunkCount = 0;
    std::uint32_t m_debugChunkIndirectCommandCount = 0;
    std::uint32_t m_debugDrawCallsTotal = 0;
    std::uint32_t m_debugDrawCallsShadow = 0;
    std::uint32_t m_debugDrawCallsPrepass = 0;
    std::uint32_t m_debugDrawCallsMain = 0;
    std::uint32_t m_debugDrawCallsPost = 0;
    std::uint32_t m_debugChunkMeshVertexCount = 0;
    std::uint32_t m_debugChunkMeshIndexCount = 0;
    std::uint32_t m_debugChunkLastRemeshedChunkCount = 0;
    std::uint32_t m_debugChunkLastRemeshActiveVertexCount = 0;
    std::uint32_t m_debugChunkLastRemeshActiveIndexCount = 0;
    std::uint32_t m_debugChunkLastRemeshNaiveVertexCount = 0;
    std::uint32_t m_debugChunkLastRemeshNaiveIndexCount = 0;
    float m_debugChunkLastRemeshReductionPercent = 0.0f;
    float m_debugChunkLastRemeshMs = 0.0f;
    float m_debugChunkLastFullRemeshMs = 0.0f;
    std::uint64_t m_debugFrameArenaUploadBytes = 0;
    std::uint32_t m_debugFrameArenaUploadAllocs = 0;
    std::uint64_t m_debugFrameArenaTransientBufferBytes = 0;
    std::uint32_t m_debugFrameArenaTransientBufferCount = 0;
    std::uint64_t m_debugFrameArenaTransientImageBytes = 0;
    std::uint32_t m_debugFrameArenaTransientImageCount = 0;
    std::uint32_t m_debugFrameArenaAliasReuses = 0;
    std::uint64_t m_debugFrameArenaResidentBufferBytes = 0;
    std::uint32_t m_debugFrameArenaResidentBufferCount = 0;
    std::uint64_t m_debugFrameArenaResidentImageBytes = 0;
    std::uint32_t m_debugFrameArenaResidentImageCount = 0;
    std::uint32_t m_debugFrameArenaResidentAliasReuses = 0;
    std::vector<FrameArenaAliasedImageInfo> m_debugAliasedImages;
    // Dynamic cascade split distances in view-space units.
    // Updated per frame and consumed by shadow rendering + shading.
    std::array<float, kShadowCascadeCount> m_shadowCascadeSplits = {20.0f, 45.0f, 90.0f, 180.0f};
    std::array<float, kShadowCascadeCount> m_shadowStableCascadeRadii = {0.0f, 0.0f, 0.0f, 0.0f};
    float m_shadowStableAspectRatio = -1.0f;
    float m_shadowStableFovDegrees = -1.0f;
};

} // namespace render
